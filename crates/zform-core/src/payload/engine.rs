//! Vector-at-a-time executor for [`Program`]: rows run in chunks, each SSA value lives in one
//! chunk-sized slot, and each instruction is one plain loop over the chunk that the compiler is
//! left to vectorize. Slots are reused once their value's last reader has run.
//!
//! A [`Plan`] (which slot holds which value) depends only on a program's operands, so one plan
//! serves a library entry under any constant overrides: constants are read from the program as
//! it runs.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use super::graph::FastMap;
use super::ir::{Ins, Len, Op, Program, TableBuf, Width};
use super::library::Code;

const NONE: u32 = u32::MAX;
/// Use count of a sink's value: it must outlive every instruction.
const PIN: u32 = u32::MAX / 2;

/// Rows a chunk holds: 128 and 64 measured slower, 1024 no faster.
pub(crate) const CHUNK: usize = 256;

/// Instruction `i` reading slots `a`, `b`, `c` into slot `d`.
#[derive(Clone, Copy)]
struct Step {
    i: u32,
    d: u32,
    a: u32,
    b: u32,
    c: u32,
}

struct Plan {
    /// Slot of each input, `NONE` when nothing reads it.
    in_slots: Vec<u32>,
    steps: Vec<Step>,
    /// Value slot and length slot, `NONE` for a fixed width.
    sinks: Vec<(u32, u32)>,
    n_slots: usize,
}

thread_local! {
    /// Plans of library entries by address: the blocks of a model run the same few.
    static PLANS: RefCell<FastMap<usize, (Arc<Program>, Rc<Plan>)>> = Default::default();
}

impl Plan {
    fn new(p: &Program) -> Self {
        let n_in = p.n_in as usize;
        let args = |ins: &Ins| -> ([u32; 3], usize) { ([ins.a, ins.b, ins.c], ins.op.arity()) };
        let mut uses = vec![0u32; n_in + p.ops.len()];
        for &(v, len) in &p.sinks {
            uses[v as usize] = PIN;
            if let Len::Lane(l) = len {
                uses[l as usize] = PIN;
            }
        }
        // Only instructions something reads run.
        for (i, ins) in p.ops.iter().enumerate().rev() {
            if uses[n_in + i] > 0 {
                let (xs, n) = args(ins);
                xs[..n]
                    .iter()
                    .for_each(|&x| uses[x as usize] = uses[x as usize].saturating_add(1));
            }
        }
        let (mut slot, mut free, mut n_slots) = (vec![NONE; uses.len()], Vec::new(), 0u32);
        let mut alloc = |free: &mut Vec<u32>| {
            free.pop().unwrap_or_else(|| {
                n_slots += 1;
                n_slots - 1
            })
        };
        for k in 0..n_in {
            if uses[k] > 0 {
                slot[k] = alloc(&mut free);
            }
        }
        let mut steps = Vec::with_capacity(p.ops.len());
        for (i, ins) in p.ops.iter().enumerate() {
            if uses[n_in + i] == 0 {
                continue;
            }
            let (xs, n) = args(ins);
            let xs = &xs[..n];
            xs.iter().for_each(|&x| uses[x as usize] -= 1);
            let dead = |x: u32| uses[x as usize] == 0;
            let (a, b) = (ins.a, ins.b);
            // A unary or binary result may overwrite an operand that dies here; a select's not.
            let d = match n {
                1 if dead(a) => slot[a as usize],
                2 if dead(a) && a != b => slot[a as usize],
                2 if dead(b) && a != b => slot[b as usize],
                _ => alloc(&mut free),
            };
            for (j, &x) in xs.iter().enumerate() {
                if dead(x) && slot[x as usize] != d && !xs[..j].contains(&x) {
                    free.push(slot[x as usize]);
                }
            }
            slot[n_in + i] = d;
            let at = |k: usize| xs.get(k).map_or(0, |&x| slot[x as usize]);
            let (a, b, c) = (at(0), at(1), at(2));
            steps.push(Step {
                i: i as u32,
                d,
                a,
                b,
                c,
            });
        }
        let sinks = p.sinks.iter().map(|&(v, len)| match len {
            Len::Fixed(_) => (slot[v as usize], NONE),
            Len::Lane(l) => (slot[v as usize], slot[l as usize]),
        });
        Self {
            in_slots: slot[..n_in].to_vec(),
            steps,
            sinks: sinks.collect(),
            n_slots: n_slots as usize,
        }
    }
}

pub struct CompiledProgram<'p> {
    plan: Rc<Plan>,
    p: &'p Program,
}

/// A 32- or 64-bit lane. Operations go through `u64` and truncate back, which keeps low bits
/// exact; shift counts keep their low five or six bits, as the IR specifies.
pub(crate) trait Lane: Copy + Default + PartialEq {
    const BITS: u32;
    const BYTES: usize = Self::BITS as usize / 8;
    fn from_u64(x: u64) -> Self;
    fn u64(self) -> u64;
    fn add(self, o: Self) -> Self {
        Self::from_u64(self.u64().wrapping_add(o.u64()))
    }
    fn sub(self, o: Self) -> Self {
        Self::from_u64(self.u64().wrapping_sub(o.u64()))
    }
    fn mul(self, o: Self) -> Self {
        Self::from_u64(self.u64().wrapping_mul(o.u64()))
    }
    fn shl(self, n: u32) -> Self {
        Self::from_u64(self.u64() << (n & (Self::BITS - 1)))
    }
    fn shr(self, n: u32) -> Self {
        Self::from_u64(self.u64() >> (n & (Self::BITS - 1)))
    }
    fn sar(self, n: u32) -> Self {
        let up = 64 - Self::BITS;
        Self::from_u64(((self.u64() << up) as i64 >> ((n & (Self::BITS - 1)) + up)) as u64)
    }
    fn ctz(self) -> Self {
        Self::from_u64(u64::from(self.u64().trailing_zeros().min(Self::BITS)))
    }
    fn swap(self) -> Self {
        Self::from_u64(self.u64().swap_bytes() >> (64 - Self::BITS))
    }
    fn map_bytes(self, f: impl Fn(u8) -> u8) -> Self {
        let mut b = self.u64().to_le_bytes();
        b[..Self::BYTES].iter_mut().for_each(|x| *x = f(*x));
        Self::from_u64(u64::from_le_bytes(b))
    }
}

impl Lane for u32 {
    const BITS: u32 = 32;
    fn from_u64(x: u64) -> Self {
        x as u32
    }
    fn u64(self) -> u64 {
        self.into()
    }
}

impl Lane for u64 {
    const BITS: u32 = 64;
    fn from_u64(x: u64) -> Self {
        x
    }
    fn u64(self) -> u64 {
        self
    }
}

fn f<T: Lane>(x: T) -> f64 {
    f64::from_bits(x.u64())
}

fn tf<T: Lane>(x: f64) -> T {
    T::from_u64(x.to_bits())
}

/// Slots, `n` rows used of each `nc`.
struct Slots<'a, T> {
    s: &'a mut [T],
    nc: usize,
    n: usize,
}

impl<T> Slots<'_, T> {
    /// Slot `d` to write, and a reader of the others.
    #[inline(always)]
    fn split<'s>(&'s mut self, d: u32) -> (&'s mut [T], impl Fn(u32) -> &'s [T]) {
        let (nc, n, d) = (self.nc, self.n, d as usize);
        let (lo, rest) = self.s.split_at_mut(d * nc);
        let (dst, hi) = rest.split_at_mut(nc);
        let (lo, hi): (&'s [T], &'s [T]) = (lo, hi);
        let read = move |x: u32| match x as usize {
            x if x < d => &lo[x * nc..][..n],
            x => &hi[(x - d - 1) * nc..][..n],
        };
        (&mut dst[..n], read)
    }
}

/// `d = f(a)` over the chunk; `d` may be `a`'s slot.
#[inline(always)]
fn un<T: Lane>(s: &mut Slots<T>, st: &Step, mut f: impl FnMut(T) -> T) {
    let (d, read) = s.split(st.d);
    if st.a == st.d {
        d.iter_mut().for_each(|x| *x = f(*x));
    } else {
        for (x, &a) in d.iter_mut().zip(read(st.a)) {
            *x = f(a);
        }
    }
}

/// `d = f(a, b)` over the chunk; `d` may be `a`'s or `b`'s slot.
#[inline(always)]
fn bin<T: Lane>(s: &mut Slots<T>, st: &Step, f: impl Fn(T, T) -> T) {
    let (d, read) = s.split(st.d);
    if st.a == st.d {
        for (x, &b) in d.iter_mut().zip(read(st.b)) {
            *x = f(*x, b);
        }
    } else if st.b == st.d {
        for (x, &a) in d.iter_mut().zip(read(st.a)) {
            *x = f(a, *x);
        }
    } else {
        for ((x, &a), &b) in d.iter_mut().zip(read(st.a)).zip(read(st.b)) {
            *x = f(a, b);
        }
    }
}

#[inline(always)]
fn exec<T: Lane>(st: &Step, p: &Program, s: &mut Slots<T>, bufs: &[TableBuf<'_>], scan: &mut T) {
    let Ins { op, aux, imm, .. } = p.ops[st.i as usize];
    let i = T::from_u64(imm as u64);
    let (fi, sh) = (f64::from_bits(imm as u64), imm as u32);
    match op {
        Op::Add => bin(s, st, T::add),
        Op::Sub => bin(s, st, T::sub),
        Op::Mul => bin(s, st, T::mul),
        Op::And => bin(s, st, |a, b| T::from_u64(a.u64() & b.u64())),
        Op::Or => bin(s, st, |a, b| T::from_u64(a.u64() | b.u64())),
        Op::Xor => bin(s, st, |a, b| T::from_u64(a.u64() ^ b.u64())),
        Op::Shl => bin(s, st, |a, b| a.shl(b.u64() as u32)),
        Op::Shr => bin(s, st, |a, b| a.shr(b.u64() as u32)),
        Op::F64Add => bin(s, st, |a, b| tf(f(a) + f(b))),
        Op::F64Sub => bin(s, st, |a, b| tf(f(a) - f(b))),
        Op::F64Mul => bin(s, st, |a, b| tf(f(a) * f(b))),
        Op::Select => {
            let (d, read) = s.split(st.d);
            for (((x, &a), &b), &c) in d.iter_mut().zip(read(st.a)).zip(read(st.b)).zip(read(st.c))
            {
                *x = if c != T::default() { a } else { b };
            }
        }
        Op::AddI => un(s, st, |a| a.add(i)),
        Op::MulI => un(s, st, |a| a.mul(i)),
        Op::AndI => un(s, st, |a| T::from_u64(a.u64() & i.u64())),
        Op::OrI => un(s, st, |a| T::from_u64(a.u64() | i.u64())),
        Op::XorI => un(s, st, |a| T::from_u64(a.u64() ^ i.u64())),
        Op::ShlI => un(s, st, |a| a.shl(sh)),
        Op::ShrI => un(s, st, |a| a.shr(sh)),
        Op::SarI => un(s, st, |a| a.sar(sh)),
        Op::MulHiI => {
            // In 32-bit halves, which vectorize where a 128-bit product does not.
            let (m0, m1) = (imm as u64 & 0xffff_ffff, imm as u64 >> 32);
            un(s, st, |a| {
                let (a0, a1) = (a.u64() & 0xffff_ffff, a.u64() >> 32);
                let (lo, mid) = (a0 * m0, a1 * m0);
                let cross = (lo >> 32) + (mid & 0xffff_ffff) + a0 * m1;
                T::from_u64(a1 * m1 + (mid >> 32) + (cross >> 32))
            })
        }
        Op::F64AddI => un(s, st, |a| tf(f(a) + fi)),
        Op::F64MulI => un(s, st, |a| tf(f(a) * fi)),
        Op::F64DivI => un(s, st, |a| tf(f(a) / fi)),
        Op::Ctz => un(s, st, T::ctz),
        Op::Bswap => un(s, st, T::swap),
        Op::U64ToF64 => un(s, st, |a| tf(a.u64() as f64)),
        Op::I64ToF64 => un(s, st, |a| tf(a.u64() as i64 as f64)),
        Op::NibbleLut => {
            let l = &p.luts[aux as usize];
            un(s, st, |a| a.map_bytes(|b| l[(b & 15) as usize]))
        }
        Op::ByteLut32 => {
            let mut t = [0u8; 32];
            t[..16].copy_from_slice(&p.luts[aux as usize]);
            t[16..].copy_from_slice(&p.luts[aux as usize + 1]);
            un(s, st, |a| a.map_bytes(|b| t[(b & 31) as usize]))
        }
        Op::Gather => {
            let buf = &bufs[(aux & 63) as usize];
            un(s, st, |a| {
                T::from_u64(buf.gather(a.u64(), imm as u64, aux >> 6) as u64)
            })
        }
        Op::ScanAdd => un(s, st, |a| {
            *scan = scan.add(a);
            *scan
        }),
        Op::ScanXor => un(s, st, |a| {
            *scan = T::from_u64(scan.u64() ^ a.u64());
            *scan
        }),
        Op::ScanF64Add => un(s, st, |a| {
            let prev = f(*scan);
            *scan = tf(if prev.is_nan() { 0.0 } else { prev } + f(a));
            *scan
        }),
    }
}

#[inline(always)]
fn steps<T: Lane>(steps: &[Step], p: &Program, s: &mut Slots<T>, b: &[TableBuf<'_>], sc: &mut [T]) {
    for (st, scan) in steps.iter().zip(sc) {
        exec(st, p, s, b, scan);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,fma,lzcnt,popcnt")]
fn steps_v3<T: Lane>(
    steps: &[Step],
    p: &Program,
    s: &mut Slots<T>,
    b: &[TableBuf<'_>],
    sc: &mut [T],
) {
    self::steps(steps, p, s, b, sc)
}

/// Runs one chunk, through the AVX2 build of the loops where the CPU has it.
fn chunk_steps<T: Lane>(
    st: &[Step],
    p: &Program,
    s: &mut Slots<T>,
    b: &[TableBuf<'_>],
    sc: &mut [T],
) {
    #[cfg(target_arch = "x86_64")]
    {
        static V3: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let v3 = V3.get_or_init(|| {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("bmi1")
                && is_x86_feature_detected!("bmi2")
                && is_x86_feature_detected!("popcnt")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("lzcnt")
        });
        if *v3 {
            // SAFETY: the CPU has the features `steps_v3` is compiled for.
            return unsafe { steps_v3(st, p, s, b, sc) };
        }
    }
    steps(st, p, s, b, sc)
}

/// A program's slots and scan states across the chunks of one run.
pub(crate) struct Runner<'c, 'p, T> {
    cp: &'c CompiledProgram<'p>,
    s: Vec<T>,
    scans: Vec<T>,
    nc: usize,
}

impl<'c, 'p, T: Lane> Runner<'c, 'p, T> {
    pub(crate) fn new(cp: &'c CompiledProgram<'p>, nc: usize) -> Self {
        let seed = |st: &Step| T::from_u64(cp.p.ops[st.i as usize].imm as u64);
        Self {
            cp,
            s: vec![T::default(); nc * cp.plan.n_slots],
            scans: cp.plan.steps.iter().map(seed).collect(),
            nc,
        }
    }

    /// Runs the next `n <= nc` rows, `load(k, slot)` filling input `k`.
    pub(crate) fn chunk(
        &mut self,
        n: usize,
        b: &[TableBuf<'_>],
        mut load: impl FnMut(usize, &mut [T]),
    ) {
        let (plan, nc) = (&*self.cp.plan, self.nc);
        for (k, &sl) in plan.in_slots.iter().enumerate() {
            if sl != NONE {
                load(k, &mut self.s[sl as usize * nc..][..n]);
            }
        }
        let mut slots = Slots {
            s: &mut self.s,
            nc,
            n,
        };
        chunk_steps(&plan.steps, self.cp.p, &mut slots, b, &mut self.scans);
    }

    /// The first sink's values in the last chunk.
    pub(crate) fn sink(&self) -> &[T] {
        &self.s[self.cp.plan.sinks[0].0 as usize * self.nc..][..self.nc]
    }
}

impl<'p> CompiledProgram<'p> {
    pub fn compile(p: &'p Program) -> Self {
        Self {
            plan: Rc::new(Plan::new(p)),
            p,
        }
    }

    /// Runs `f` on `code` compiled; a library entry's plan is made once a thread.
    pub fn with<R>(code: &Code, f: impl FnOnce(&CompiledProgram<'_>) -> R) -> R {
        let plan = match code.entry() {
            Some(entry) => PLANS.with(|m| {
                let mut m = m.borrow_mut();
                let key = Arc::as_ptr(entry) as usize;
                if !m.contains_key(&key) && m.len() >= 1024 {
                    m.clear();
                }
                let e = m.entry(key);
                e.or_insert_with(|| (entry.clone(), Rc::new(Plan::new(entry))))
                    .1
                    .clone()
            }),
            _ => Rc::new(Plan::new(code)),
        };
        f(&CompiledProgram { plan, p: code })
    }

    pub fn width(&self) -> Width {
        self.p.width
    }

    pub fn row_bytes(&self) -> usize {
        self.p.row_bytes()
    }

    /// Runs the program over `rows` rows, handing each finished chunk's slots to `emit`.
    fn run<T: Lane>(
        &self,
        rows: usize,
        bufs: &[TableBuf<'_>],
        load: impl Fn(usize, usize, &mut [T]),
        mut emit: impl FnMut(usize, usize, &[T]),
    ) {
        let nc = CHUNK.min(rows.max(1));
        let mut r = Runner::new(self, nc);
        let mut row = 0;
        while row < rows {
            let n = nc.min(rows - row);
            r.chunk(n, bufs, |k, d| load(k, row, d));
            emit(row, nc, &r.s);
            row += n;
        }
    }

    fn direct<T: Lane, I: Copy, O>(
        &self,
        inputs: &[&[I]],
        bufs: &[TableBuf<'_>],
        out: &mut [O],
        load: impl Fn(I) -> T,
        store: impl Fn(T) -> O,
    ) {
        let fixed = self.p.sinks.get(0).map(|s| s.1) == Some(Len::Fixed(T::BYTES as u8));
        assert!(self.p.sinks.len() == 1 && fixed);
        let rows = out.len();
        inputs.iter().for_each(|c| assert!(c.len() >= rows));
        let v = self.plan.sinks[0].0 as usize;
        self.run::<T>(
            rows,
            bufs,
            |k, row, d| {
                for (d, &x) in d.iter_mut().zip(&inputs[k][row..]) {
                    *d = load(x);
                }
            },
            |row, nc, s| {
                for (o, &x) in out[row..].iter_mut().zip(&s[v * nc..][..nc]) {
                    *o = store(x);
                }
            },
        );
    }

    /// Evaluate a single-8-byte-sink `W64` program directly into `out: &mut [i64]`.
    pub fn eval_i64(&self, inputs: &[&[i64]], bufs: &[TableBuf<'_>], out: &mut [i64]) {
        assert_eq!(self.p.width, Width::W64);
        self.direct::<u64, _, _>(inputs, bufs, out, |x| x as u64, |x| x as i64);
    }

    /// Evaluate a single-4-byte-sink `W32` program directly into `out: &mut [u32]`.
    pub fn eval_u32(&self, inputs: &[&[u32]], bufs: &[TableBuf<'_>], out: &mut [u32]) {
        assert_eq!(self.p.width, Width::W32);
        self.direct::<u32, _, _>(inputs, bufs, out, |x| x, |x| x);
    }

    /// Evaluate any fixed-width program into `out` (`rows * row_bytes` bytes).
    pub fn eval_fixed(
        &self,
        inputs: &[&[i64]],
        bufs: &[TableBuf<'_>],
        rows: usize,
        out: &mut [u8],
    ) {
        assert!(out.len() >= rows * self.row_bytes());
        self.cells(inputs, bufs, rows, out, None);
    }

    /// Evaluate a variable-length cell program: writes concatenated bytes into `text` and each
    /// row's ending byte offset into `ends`, returning total bytes written.
    pub fn eval_cells(
        &self,
        inputs: &[&[i64]],
        bufs: &[TableBuf<'_>],
        rows: usize,
        ends: &mut [u32],
        text: &mut [u8],
    ) -> usize {
        assert!(ends.len() >= rows && text.len() >= rows * self.row_bytes());
        self.cells(inputs, bufs, rows, text, Some(ends))
    }

    fn cells(
        &self,
        inputs: &[&[i64]],
        bufs: &[TableBuf<'_>],
        rows: usize,
        out: &mut [u8],
        ends: Option<&mut [u32]>,
    ) -> usize {
        assert_eq!(inputs.len(), self.plan.in_slots.len());
        inputs.iter().for_each(|c| assert!(c.len() >= rows));
        match self.p.width {
            Width::W32 => self.sink_rows::<u32>(inputs, bufs, rows, out, ends),
            Width::W64 => self.sink_rows::<u64>(inputs, bufs, rows, out, ends),
        }
    }

    fn sink_rows<T: Lane>(
        &self,
        inputs: &[&[i64]],
        bufs: &[TableBuf<'_>],
        rows: usize,
        out: &mut [u8],
        mut ends: Option<&mut [u32]>,
    ) -> usize {
        let mut pos = 0;
        let load = |k: usize, row: usize, d: &mut [T]| {
            for (d, &x) in d.iter_mut().zip(&inputs[k][row..]) {
                *d = T::from_u64(x as u64);
            }
        };
        let sinks: Vec<(usize, u32, usize)> = (self.plan.sinks.iter().zip(&self.p.sinks))
            .map(|(&(v, l), &(_, len))| match len {
                Len::Fixed(w) => (v as usize, NONE, w as usize),
                Len::Lane(_) => (v as usize, l, 0),
            })
            .collect();
        self.run::<T>(rows, bufs, load, |row, nc, s| {
            for r in 0..nc.min(rows - row) {
                for &(v, len, w) in &sinks {
                    let x = s[v * nc + r];
                    let w = match len {
                        NONE => w,
                        l => (s[l as usize * nc + r].u64() as usize).min(T::BYTES),
                    };
                    // Store a whole lane where it fits and advance by the sink's width.
                    if pos + T::BYTES <= out.len() {
                        out[pos..pos + T::BYTES]
                            .copy_from_slice(&x.u64().to_le_bytes()[..T::BYTES]);
                    } else {
                        out[pos..pos + w].copy_from_slice(&x.u64().to_le_bytes()[..w]);
                    }
                    pos += w;
                }
                if let Some(e) = ends.as_deref_mut() {
                    e[row + r] = pos as u32;
                }
            }
        });
        pos
    }
}
