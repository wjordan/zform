//! Portable, verifiable SSA lane-program IR and wire format.
//!
//! A [`Program`] is a straight-line sequence of operations over 32-bit or 64-bit lanes:
//! no branches, no loops, no arbitrary pointers. Validation is a single linear pass checking
//! that every operand names an earlier value, shifts stay within lane width, and buffer/LUT
//! indices are in bounds. Termination and memory safety hold by construction.

use std::cell::{Cell, RefCell};
use std::fmt;

/// Lane width of a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Width {
    W32 = 0,
    W64 = 1,
}

impl Width {
    #[inline]
    pub fn bits(self) -> u32 {
        match self {
            Self::W32 => 32,
            Self::W64 => 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Op {
    // Binary integer/bitwise: (a, b)
    Add = 0,
    Sub = 1,
    Mul = 2,
    And = 3,
    Or = 4,
    Xor = 5,
    Shl = 6,
    Shr = 7,
    // Binary deterministic IEEE-754 f64 (W64): (a, b)
    F64Add = 8,
    F64Sub = 9,
    F64Mul = 10,
    // Ternary: c != 0 ? a : b
    Select = 11,
    // Unary + immediate: (a, imm)
    AddI = 12,
    MulI = 13,
    AndI = 14,
    OrI = 15,
    XorI = 16,
    ShlI = 17,
    ShrI = 18,
    SarI = 19,
    MulHiI = 20,
    F64AddI = 21,
    F64MulI = 22,
    // Unary: (a)
    Ctz = 23,
    Bswap = 24,
    U64ToF64 = 25,
    // Register LUTs over each byte of `a`: `aux` indexes `Program::luts`
    NibbleLut = 26, // each byte `b` -> `luts[aux][b & 15]`
    ByteLut32 = 27, // each byte `b` -> `(luts[aux] ++ luts[aux + 1])[b & 31]`
    // Bounded read-only table gather: `buffers[buf][a + imm]` (0 if out of bounds)
    Gather = 28,
    // Stateful 1st-order scans across rows (`imm` = initial state before row 0)
    ScanAdd = 29,
    ScanXor = 30,
    ScanF64Add = 31,
    I64ToF64 = 32,
    F64DivI = 33,
}

pub const ALL_OPS: [Op; 34] = [
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::And,
    Op::Or,
    Op::Xor,
    Op::Shl,
    Op::Shr,
    Op::F64Add,
    Op::F64Sub,
    Op::F64Mul,
    Op::Select,
    Op::AddI,
    Op::MulI,
    Op::AndI,
    Op::OrI,
    Op::XorI,
    Op::ShlI,
    Op::ShrI,
    Op::SarI,
    Op::MulHiI,
    Op::F64AddI,
    Op::F64MulI,
    Op::Ctz,
    Op::Bswap,
    Op::U64ToF64,
    Op::NibbleLut,
    Op::ByteLut32,
    Op::Gather,
    Op::ScanAdd,
    Op::ScanXor,
    Op::ScanF64Add,
    Op::I64ToF64,
    Op::F64DivI,
];

impl Op {
    #[inline]
    pub fn arity(self) -> usize {
        match self {
            Self::Select => 3,
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::And
            | Self::Or
            | Self::Xor
            | Self::Shl
            | Self::Shr
            | Self::F64Add
            | Self::F64Sub
            | Self::F64Mul => 2,
            _ => 1,
        }
    }

    #[inline]
    pub fn is_shift_imm(self) -> bool {
        matches!(self, Self::ShlI | Self::ShrI | Self::SarI)
    }

    #[inline]
    /// Whether an instruction of a `width` program may carry `imm`: what decoding one enforces.
    pub fn imm_ok(self, width: Width, imm: i64) -> bool {
        if self.is_shift_imm() {
            (0..i64::from(width.bits())).contains(&imm)
        } else {
            self != Self::Gather || imm >= 0
        }
    }

    pub fn has_imm(self) -> bool {
        matches!(
            self,
            Self::AddI
                | Self::MulI
                | Self::AndI
                | Self::OrI
                | Self::XorI
                | Self::ShlI
                | Self::ShrI
                | Self::SarI
                | Self::MulHiI
                | Self::F64AddI
                | Self::F64MulI
                | Self::F64DivI
                | Self::Gather
                | Self::ScanAdd
                | Self::ScanXor
                | Self::ScanF64Add
        )
    }

    #[inline]
    pub fn is_scan(self) -> bool {
        matches!(self, Self::ScanAdd | Self::ScanXor | Self::ScanF64Add)
    }
}

/// One SSA instruction; defines value `n_in + i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ins {
    pub op: Op,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    /// For `NibbleLut`/`ByteLut32`: index in `luts`.
    /// For `Gather`: `(elem_log2 << 6) | buf_id`: reads `1 << elem_log2` bytes, little-endian, at
    /// byte offset `a + imm` of the buffer (unaligned; out of range reads as 0).
    pub aux: u8,
    pub imm: i64,
}

// Three words an instruction, not six fields: the caches keyed by `Program` hash one per column.
impl std::hash::Hash for Ins {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.op as u64 | (u64::from(self.aux) << 8) | (u64::from(self.a) << 32));
        state.write_u64(u64::from(self.b) | (u64::from(self.c) << 32));
        state.write_i64(self.imm);
    }
}

/// How many low bytes of a sink value each row emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Len {
    Fixed(u8),
    Lane(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Program {
    pub width: Width,
    pub n_in: u32,
    pub n_bufs: u8,
    pub luts: Vec<[u8; 16]>,
    pub ops: Vec<Ins>,
    pub sinks: Vec<(u32, Len)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invalid(pub &'static str);

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Invalid {}

pub fn put_uv(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub fn get_uv(b: &mut &[u8]) -> Result<u64, Invalid> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let (&x, rest) = b.split_first().ok_or(Invalid("truncated varint"))?;
        *b = rest;
        v |= ((x & 0x7f) as u64) << shift;
        if x < 0x80 {
            return Ok(v);
        }
    }
    Err(Invalid("varint overflow"))
}

pub fn put_iv(out: &mut Vec<u8>, v: i64) {
    put_uv(out, ((v << 1) ^ (v >> 63)) as u64);
}

pub fn get_iv(b: &mut &[u8]) -> Result<i64, Invalid> {
    let z = get_uv(b)?;
    Ok(((z >> 1) as i64) ^ -((z & 1) as i64))
}

impl Program {
    /// Maximum bytes emitted by one row.
    pub fn row_bytes(&self) -> usize {
        let max_w = match self.width {
            Width::W32 => 4usize,
            Width::W64 => 8usize,
        };
        self.sinks
            .iter()
            .map(|&(_, len)| match len {
                Len::Fixed(w) => w as usize,
                Len::Lane(_) => max_w,
            })
            .sum()
    }

    pub fn varlen(&self) -> bool {
        self.sinks
            .iter()
            .any(|&(_, len)| matches!(len, Len::Lane(_)))
    }

    pub fn has_scans(&self) -> bool {
        self.ops.iter().any(|i| i.op.is_scan())
    }

    pub fn has_gathers(&self) -> bool {
        self.ops.iter().any(|i| i.op == Op::Gather)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let hdr = (self.width as u8) | ((self.n_bufs.min(63)) << 1);
        out.push(hdr);
        put_uv(&mut out, self.n_in as u64);
        put_uv(&mut out, self.luts.len() as u64);
        for lut in &self.luts {
            out.extend_from_slice(lut);
        }
        put_uv(&mut out, self.ops.len() as u64);
        for (i, ins) in self.ops.iter().enumerate() {
            let me = self.n_in + i as u32;
            out.push(ins.op as u8);
            for &v in &[ins.a, ins.b, ins.c][..ins.op.arity()] {
                put_uv(&mut out, (me - v) as u64);
            }
            if matches!(ins.op, Op::NibbleLut | Op::ByteLut32 | Op::Gather) {
                out.push(ins.aux);
            }
            if ins.op.is_shift_imm() {
                out.push(ins.imm as u8);
            } else if matches!(
                ins.op,
                Op::F64AddI | Op::F64MulI | Op::F64DivI | Op::ScanF64Add
            ) {
                out.extend_from_slice(&ins.imm.to_le_bytes());
            } else if ins.op.has_imm() {
                put_iv(&mut out, ins.imm);
            }
        }
        put_uv(&mut out, self.sinks.len() as u64);
        let end = self.n_in + self.ops.len() as u32;
        for &(v, len) in &self.sinks {
            put_uv(&mut out, (end - v) as u64);
            match len {
                Len::Fixed(w) => out.push(w),
                Len::Lane(l) => {
                    out.push(0);
                    put_uv(&mut out, (end - l) as u64);
                }
            }
        }
        out
    }

    pub fn decode(mut b: &[u8]) -> Result<Self, Invalid> {
        let b = &mut b;
        let (&hdr, rest) = b.split_first().ok_or(Invalid("empty program"))?;
        *b = rest;
        if hdr & 0x80 != 0 {
            return Err(Invalid("reserved header bit"));
        }
        let width = if hdr & 1 == 0 { Width::W32 } else { Width::W64 };
        let n_bufs = (hdr >> 1) & 0x3f;
        let n_in = get_uv(b)?;
        let n_luts = get_uv(b)?;
        if n_in > 64 || n_luts > 64 {
            return Err(Invalid("too many inputs or LUTs"));
        }
        let n_in = n_in as u32;
        let mut luts = Vec::with_capacity(n_luts as usize);
        for _ in 0..n_luts {
            if b.len() < 16 {
                return Err(Invalid("truncated LUT"));
            }
            let (head, rest) = b.split_at(16);
            *b = rest;
            luts.push(head.try_into().unwrap());
        }
        let n_ops = get_uv(b)?;
        if n_ops > (1 << 14) {
            return Err(Invalid("program too large"));
        }
        let back = |b: &mut &[u8], me: u32| -> Result<u32, Invalid> {
            match get_uv(b)? {
                d if d >= 1 && d <= me as u64 => Ok(me - d as u32),
                _ => Err(Invalid("operand does not name an earlier value")),
            }
        };
        let mut ops = Vec::with_capacity(n_ops as usize);
        for i in 0..n_ops as u32 {
            let me = n_in + i;
            let (&code, rest) = b.split_first().ok_or(Invalid("truncated op"))?;
            *b = rest;
            let op = *ALL_OPS
                .get(code as usize)
                .ok_or(Invalid("unknown opcode"))?;
            if width == Width::W32
                && matches!(
                    op,
                    Op::MulHiI
                        | Op::F64Add
                        | Op::F64Sub
                        | Op::F64Mul
                        | Op::F64AddI
                        | Op::F64MulI
                        | Op::I64ToF64
                        | Op::F64DivI
                        | Op::U64ToF64
                        | Op::ScanF64Add
                )
            {
                return Err(Invalid("64-bit opcode in 32-bit program"));
            }
            let mut v = [0u32; 3];
            for slot in &mut v[..op.arity()] {
                *slot = back(b, me)?;
            }
            let aux = if matches!(op, Op::NibbleLut | Op::ByteLut32 | Op::Gather) {
                let (&a, rest) = b.split_first().ok_or(Invalid("truncated aux"))?;
                *b = rest;
                match op {
                    Op::NibbleLut if (a as usize) >= luts.len() => {
                        return Err(Invalid("LUT index out of bounds"));
                    }
                    Op::ByteLut32 if (a as usize) + 1 >= luts.len() => {
                        return Err(Invalid("ByteLut32 pair out of bounds"));
                    }
                    Op::Gather if (a & 0x3f) >= n_bufs => {
                        return Err(Invalid("gather buffer out of bounds"));
                    }
                    _ => a,
                }
            } else {
                0
            };
            let imm = if op.is_shift_imm() {
                let (&s, rest) = b.split_first().ok_or(Invalid("truncated shift"))?;
                *b = rest;
                if s as u32 >= width.bits() {
                    return Err(Invalid("shift count out of range"));
                }
                s as i64
            } else if matches!(op, Op::F64AddI | Op::F64MulI | Op::F64DivI | Op::ScanF64Add) {
                if b.len() < 8 {
                    return Err(Invalid("truncated f64 immediate"));
                }
                let (head, rest) = b.split_at(8);
                *b = rest;
                i64::from_le_bytes(head.try_into().unwrap())
            } else if op.has_imm() {
                let x = get_iv(b)?;
                if op == Op::Gather && x < 0 {
                    return Err(Invalid("negative gather offset"));
                }
                x
            } else {
                0
            };
            ops.push(Ins {
                op,
                a: v[0],
                b: v[1],
                c: v[2],
                aux,
                imm,
            });
        }
        let end = n_in + n_ops as u32;
        let n_sinks = get_uv(b)?;
        if n_sinks == 0 || n_sinks > 64 {
            return Err(Invalid("invalid sink count"));
        }
        let max_sink = if width == Width::W32 { 4 } else { 8 };
        let mut sinks = Vec::with_capacity(n_sinks as usize);
        for _ in 0..n_sinks {
            let v = back(b, end)?;
            let (&w, rest) = b.split_first().ok_or(Invalid("truncated sink"))?;
            *b = rest;
            sinks.push((
                v,
                match w {
                    0 => Len::Lane(back(b, end)?),
                    w if w <= max_sink => Len::Fixed(w),
                    _ => return Err(Invalid("sink width exceeds lane width")),
                },
            ));
        }
        if !b.is_empty() {
            return Err(Invalid("trailing bytes in program"));
        }
        Ok(Program {
            width,
            n_in,
            n_bufs,
            luts,
            ops,
            sinks,
        })
    }
}

/// Read-only buffer view passed to `Gather`.
#[derive(Clone, Copy)]
pub struct TableBuf<'a> {
    pub bytes: &'a [u8],
}

impl<'a> TableBuf<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    pub fn from_u32s(words: &'a [u32]) -> Self {
        Self {
            // SAFETY: every `u32` is four initialized bytes.
            bytes: unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 4) },
        }
    }
    #[inline]
    pub fn gather(&self, idx: u64, offset: u64, elem_log2: u8) -> i64 {
        let step = 1usize << (elem_log2 & 3);
        let Some(byte_at) = idx
            .checked_add(offset)
            .and_then(|i| usize::try_from(i).ok())
        else {
            return 0;
        };
        let Some(end) = byte_at.checked_add(step) else {
            return 0;
        };
        if end > self.bytes.len() {
            return 0;
        }
        let s = &self.bytes[byte_at..end];
        match step {
            1 => s[0] as i64,
            2 => u16::from_le_bytes(s.try_into().unwrap()) as i64,
            4 => u32::from_le_bytes(s.try_into().unwrap()) as i64,
            _ => i64::from_le_bytes(s.try_into().unwrap()),
        }
    }
}

/// Ergonomic SSA builder for encoders and codec packs.
///
/// Uses interior mutability so pack authors can write nested algebraic expressions directly,
/// e.g. `b.rr(Op::Add, x, b.ri(Op::MulI, q, -100))`.
pub struct Builder {
    width: Width,
    n_in: u32,
    n_bufs: Cell<u8>,
    luts: RefCell<Vec<[u8; 16]>>,
    ops: RefCell<Vec<Ins>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V(pub u32);

impl Builder {
    pub fn new(n_in: u32) -> Self {
        Self::with_width(Width::W64, n_in)
    }

    pub fn new_u32(n_in: u32) -> Self {
        Self::with_width(Width::W32, n_in)
    }

    pub fn with_width(width: Width, n_in: u32) -> Self {
        Self {
            width,
            n_in,
            n_bufs: Cell::new(0),
            luts: RefCell::new(Vec::new()),
            ops: RefCell::new(Vec::new()),
        }
    }

    pub fn input(&self, k: u32) -> V {
        assert!(k < self.n_in);
        V(k)
    }

    pub fn add_buf(&self) -> u8 {
        let id = self.n_bufs.get();
        self.n_bufs.set(id + 1);
        id
    }

    pub fn add_lut16(&self, table: [u8; 16]) -> u8 {
        let mut luts = self.luts.borrow_mut();
        if let Some(pos) = luts.iter().position(|t| *t == table) {
            return pos as u8;
        }
        let id = luts.len() as u8;
        luts.push(table);
        id
    }

    pub fn add_lut32(&self, table: &[u8; 32]) -> u8 {
        let mut luts = self.luts.borrow_mut();
        let lo: [u8; 16] = table[..16].try_into().unwrap();
        let hi: [u8; 16] = table[16..].try_into().unwrap();
        for i in 0..luts.len().saturating_sub(1) {
            if luts[i] == lo && luts[i + 1] == hi {
                return i as u8;
            }
        }
        let id = luts.len() as u8;
        luts.push(lo);
        luts.push(hi);
        id
    }

    fn push(&self, op: Op, a: u32, b: u32, c: u32, aux: u8, imm: i64) -> V {
        let mut ops = self.ops.borrow_mut();
        ops.push(Ins {
            op,
            a,
            b,
            c,
            aux,
            imm,
        });
        V(self.n_in + ops.len() as u32 - 1)
    }

    pub fn r(&self, op: Op, a: V) -> V {
        self.push(op, a.0, 0, 0, 0, 0)
    }
    pub fn rr(&self, op: Op, a: V, b: V) -> V {
        self.push(op, a.0, b.0, 0, 0, 0)
    }
    pub fn ri(&self, op: Op, a: V, imm: i64) -> V {
        self.push(op, a.0, 0, 0, 0, imm)
    }
    pub fn rf(&self, op: Op, a: V, imm: f64) -> V {
        self.push(op, a.0, 0, 0, 0, imm.to_bits() as i64)
    }
    pub fn select(&self, cond: V, a: V, b: V) -> V {
        self.push(Op::Select, a.0, b.0, cond.0, 0, 0)
    }
    pub fn nibble_lut(&self, a: V, lut_id: u8) -> V {
        self.push(Op::NibbleLut, a.0, 0, 0, lut_id, 0)
    }
    pub fn byte_lut32(&self, a: V, lut_id: u8) -> V {
        self.push(Op::ByteLut32, a.0, 0, 0, lut_id, 0)
    }
    pub fn gather(&self, buf_id: u8, elem_log2: u8, idx: V, offset: u64) -> V {
        assert!(elem_log2 <= 3 && buf_id < self.n_bufs.get());
        self.push(
            Op::Gather,
            idx.0,
            0,
            0,
            (elem_log2 << 6) | buf_id,
            offset as i64,
        )
    }
    pub fn scan_add(&self, a: V, seed: i64) -> V {
        self.push(Op::ScanAdd, a.0, 0, 0, 0, seed)
    }
    pub fn scan_xor(&self, a: V, seed: i64) -> V {
        self.push(Op::ScanXor, a.0, 0, 0, 0, seed)
    }
    pub fn scan_f64_add(&self, a: V, seed: f64) -> V {
        self.push(Op::ScanF64Add, a.0, 0, 0, 0, seed.to_bits() as i64)
    }

    /// Constant lane from any anchor input `any`.
    pub fn konst(&self, any: V, val: i64) -> V {
        let z = self.ri(Op::AndI, any, 0);
        if val == 0 {
            z
        } else {
            self.ri(Op::OrI, z, val)
        }
    }

    /// Unsigned `x / d` for `0 <= x < 2^bits`, synthesized as multiply + shift
    /// (falls back to `div_wide` via `MulHiI` when `m * (2^bits - 1)` overflows 64 bits).
    pub fn div(&self, x: V, d: u64, bits: u32) -> V {
        assert!(bits <= 64);
        let max_x = if bits == 64 {
            u64::MAX as u128
        } else {
            (1u128 << bits) - 1
        };
        for k in bits..64 {
            let m = (1u128 << k).div_ceil(d as u128);
            let err = m * d as u128 - (1u128 << k);
            if err << bits <= 1u128 << k && m * max_x < (1u128 << 64) {
                let t = self.ri(Op::MulI, x, m as i64);
                return self.ri(Op::ShrI, t, k as i64);
            }
        }
        self.div_wide(x, d)
    }

    /// Unsigned `x / d` across full 64-bit range via Granlund-Montgomery `MulHiI` + shifts.
    pub fn div_wide(&self, x: V, d: u64) -> V {
        assert!(d > 0);
        if d == 1 {
            return x;
        }
        let tz = d.trailing_zeros();
        let (y, d_odd, rem_bits) = if tz > 0 {
            (self.ri(Op::ShrI, x, tz as i64), d >> tz, 64 - tz)
        } else {
            (x, d, 64)
        };
        if d_odd == 1 {
            return y;
        }
        for k in 0..64 {
            let m = (1u128 << (64 + k)).div_ceil(d_odd as u128);
            let err = m * d_odd as u128 - (1u128 << (64 + k));
            if (m >> 64) == 0 && (err << rem_bits) <= (1u128 << (64 + k)) {
                let t = self.ri(Op::MulHiI, y, m as i64);
                return if k == 0 {
                    t
                } else {
                    self.ri(Op::ShrI, t, k as i64)
                };
            }
        }
        // Full 64-bit odd divisor requiring a 65-bit magic multiplier (`tz == 0`).
        let l = 64 - (d_odd - 1).leading_zeros();
        let m_full = ((1u128 << (64 + l)) / d_odd as u128) + 1;
        let m_lo = (m_full - (1u128 << 64)) as u64;
        let t1 = self.ri(Op::MulHiI, y, m_lo as i64);
        let diff = self.rr(Op::Sub, y, t1);
        let half = self.ri(Op::ShrI, diff, 1);
        let sum = self.rr(Op::Add, t1, half);
        if l <= 1 {
            sum
        } else {
            self.ri(Op::ShrI, sum, (l - 1) as i64)
        }
    }

    pub fn finish(self, sinks: &[(V, u8)]) -> Program {
        Program {
            width: self.width,
            n_in: self.n_in,
            n_bufs: self.n_bufs.get(),
            luts: self.luts.into_inner(),
            ops: self.ops.into_inner(),
            sinks: sinks.iter().map(|&(v, w)| (v.0, Len::Fixed(w))).collect(),
        }
    }

    pub fn finish_sinks(self, sinks: &[(V, Len)]) -> Program {
        Program {
            width: self.width,
            n_in: self.n_in,
            n_bufs: self.n_bufs.get(),
            luts: self.luts.into_inner(),
            ops: self.ops.into_inner(),
            sinks: sinks.iter().map(|&(v, l)| (v.0, l)).collect(),
        }
    }

    pub fn finish_var(self, sinks: &[(V, V)]) -> Program {
        Program {
            width: self.width,
            n_in: self.n_in,
            n_bufs: self.n_bufs.get(),
            luts: self.luts.into_inner(),
            ops: self.ops.into_inner(),
            sinks: sinks.iter().map(|&(v, l)| (v.0, Len::Lane(l.0))).collect(),
        }
    }
}

/// Reference row-by-row semantic specification of [`Program`] (including scans and gathers).
pub fn eval_reference(p: &Program, inputs: &[&[i64]], bufs: &[TableBuf<'_>]) -> Vec<Vec<u8>> {
    let rows = inputs.first().map_or(0, |c| c.len());
    let mut scan_state: Vec<i64> = p.ops.iter().map(|ins| ins.imm).collect();
    let mut out = Vec::with_capacity(rows);
    let mut v = Vec::with_capacity(p.n_in as usize + p.ops.len());
    for row in 0..rows {
        v.clear();
        for col in inputs {
            let x = col[row];
            v.push(match p.width {
                Width::W32 => (x as u32) as i64,
                Width::W64 => x,
            });
        }
        for (idx, ins) in p.ops.iter().enumerate() {
            let a = v[ins.a as usize];
            let b = v.get(ins.b as usize).copied().unwrap_or(0);
            let c = v.get(ins.c as usize).copied().unwrap_or(0);
            let res =
                match p.width {
                    Width::W32 => {
                        let (au, bu, cu, iu) = (a as u32, b as u32, c as u32, ins.imm as u32);
                        let r32: u32 = match ins.op {
                            Op::Add => au.wrapping_add(bu),
                            Op::Sub => au.wrapping_sub(bu),
                            Op::Mul => au.wrapping_mul(bu),
                            Op::And => au & bu,
                            Op::Or => au | bu,
                            Op::Xor => au ^ bu,
                            Op::Shl => au << (bu & 31),
                            Op::Shr => au >> (bu & 31),
                            Op::Select => {
                                if cu != 0 {
                                    au
                                } else {
                                    bu
                                }
                            }
                            Op::AddI => au.wrapping_add(iu),
                            Op::MulI => au.wrapping_mul(iu),
                            Op::AndI => au & iu,
                            Op::OrI => au | iu,
                            Op::XorI => au ^ iu,
                            Op::ShlI => au << (iu & 31),
                            Op::ShrI => au >> (iu & 31),
                            Op::SarI => ((au as i32) >> (iu & 31)) as u32,
                            Op::Ctz => au.trailing_zeros(),
                            Op::Bswap => au.swap_bytes(),
                            Op::NibbleLut => {
                                let lut = &p.luts[ins.aux as usize];
                                let mut bytes = au.to_le_bytes();
                                for byte in &mut bytes {
                                    *byte = lut[(*byte & 15) as usize];
                                }
                                u32::from_le_bytes(bytes)
                            }
                            Op::ByteLut32 => {
                                let lo = &p.luts[ins.aux as usize];
                                let hi = &p.luts[ins.aux as usize + 1];
                                let mut bytes = au.to_le_bytes();
                                for byte in &mut bytes {
                                    let k = (*byte & 31) as usize;
                                    *byte = if k < 16 { lo[k] } else { hi[k - 16] };
                                }
                                u32::from_le_bytes(bytes)
                            }
                            Op::Gather => {
                                let buf = &bufs[(ins.aux & 0x3f) as usize];
                                buf.gather(au as u64, ins.imm as u64, ins.aux >> 6) as u32
                            }
                            Op::ScanAdd => {
                                let acc = (scan_state[idx] as u32).wrapping_add(au);
                                scan_state[idx] = acc as i64;
                                acc
                            }
                            Op::ScanXor => {
                                let acc = (scan_state[idx] as u32) ^ au;
                                scan_state[idx] = acc as i64;
                                acc
                            }
                            _ => unreachable!(),
                        };
                        r32 as i64
                    }
                    Width::W64 => match ins.op {
                        Op::Add => a.wrapping_add(b),
                        Op::Sub => a.wrapping_sub(b),
                        Op::Mul => a.wrapping_mul(b),
                        Op::And => a & b,
                        Op::Or => a | b,
                        Op::Xor => a ^ b,
                        Op::Shl => ((a as u64) << (b & 63)) as i64,
                        Op::Shr => ((a as u64) >> (b & 63)) as i64,
                        Op::F64Add => {
                            (f64::from_bits(a as u64) + f64::from_bits(b as u64)).to_bits() as i64
                        }
                        Op::F64Sub => {
                            (f64::from_bits(a as u64) - f64::from_bits(b as u64)).to_bits() as i64
                        }
                        Op::F64Mul => {
                            (f64::from_bits(a as u64) * f64::from_bits(b as u64)).to_bits() as i64
                        }
                        Op::Select => {
                            if c != 0 {
                                a
                            } else {
                                b
                            }
                        }
                        Op::AddI => a.wrapping_add(ins.imm),
                        Op::MulI => a.wrapping_mul(ins.imm),
                        Op::AndI => a & ins.imm,
                        Op::OrI => a | ins.imm,
                        Op::XorI => a ^ ins.imm,
                        Op::ShlI => ((a as u64) << (ins.imm & 63)) as i64,
                        Op::ShrI => ((a as u64) >> (ins.imm & 63)) as i64,
                        Op::SarI => a >> (ins.imm & 63),
                        Op::MulHiI => ((a as u64 as u128 * ins.imm as u64 as u128) >> 64) as i64,
                        Op::F64AddI => (f64::from_bits(a as u64) + f64::from_bits(ins.imm as u64))
                            .to_bits() as i64,
                        Op::F64MulI => (f64::from_bits(a as u64) * f64::from_bits(ins.imm as u64))
                            .to_bits() as i64,
                        Op::Ctz => a.trailing_zeros() as i64,
                        Op::Bswap => a.swap_bytes(),
                        Op::U64ToF64 => ((a as u64) as f64).to_bits() as i64,
                        Op::I64ToF64 => (a as f64).to_bits() as i64,
                        Op::F64DivI => (f64::from_bits(a as u64) / f64::from_bits(ins.imm as u64))
                            .to_bits() as i64,
                        Op::NibbleLut => {
                            let lut = &p.luts[ins.aux as usize];
                            let mut bytes = a.to_le_bytes();
                            for byte in &mut bytes {
                                *byte = lut[(*byte & 15) as usize];
                            }
                            i64::from_le_bytes(bytes)
                        }
                        Op::ByteLut32 => {
                            let lo = &p.luts[ins.aux as usize];
                            let hi = &p.luts[ins.aux as usize + 1];
                            let mut bytes = a.to_le_bytes();
                            for byte in &mut bytes {
                                let k = (*byte & 31) as usize;
                                *byte = if k < 16 { lo[k] } else { hi[k - 16] };
                            }
                            i64::from_le_bytes(bytes)
                        }
                        Op::Gather => {
                            let buf = &bufs[(ins.aux & 0x3f) as usize];
                            buf.gather(a as u64, ins.imm as u64, ins.aux >> 6)
                        }
                        Op::ScanAdd => {
                            let acc = scan_state[idx].wrapping_add(a);
                            scan_state[idx] = acc;
                            acc
                        }
                        Op::ScanXor => {
                            let acc = scan_state[idx] ^ a;
                            scan_state[idx] = acc;
                            acc
                        }
                        Op::ScanF64Add => {
                            let prev = f64::from_bits(scan_state[idx] as u64);
                            let prev = if prev.is_nan() { 0.0 } else { prev };
                            let next = (prev + f64::from_bits(a as u64)).to_bits() as i64;
                            scan_state[idx] = next;
                            next
                        }
                    },
                };
            v.push(res);
        }
        let mut cell = Vec::with_capacity(p.row_bytes());
        for &(s, len) in &p.sinks {
            let n = match len {
                Len::Fixed(w) => w as usize,
                Len::Lane(l) => match p.width {
                    Width::W32 => (v[l as usize] as u32 as usize).min(4),
                    Width::W64 => (v[l as usize] as u64 as usize).min(8),
                },
            };
            cell.extend_from_slice(&v[s as usize].to_le_bytes()[..n]);
        }
        out.push(cell);
    }
    out
}
