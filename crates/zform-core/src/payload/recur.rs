//! Recurrences: `n_state` sequences of state words advancing together, each next word computed
//! from taps `state[lane][i - lag]` and optional integer columns (PROGRAMS.md "recurrences").
//!
//! [`CompiledRecurrence`] runs a spec through the program executor: with every lag at least `n`,
//! the next `n` steps depend only on steps already made, so one chunk of the program makes them
//! all from windows of the history. Lags under 3 go step by step through the scalar evaluator,
//! which measured faster there.

use super::engine::{CHUNK, CompiledProgram, Lane, Runner};
use super::ir::{Invalid, Len, Op, Program, Width, get_uv, put_uv};

const MAX_STATE: u32 = 8;
const MAX_TAPS: usize = 16;
const MAX_LAG: u32 = 1 << 20;

/// Wire-serializable specification of a recurrence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecurrenceSpec {
    pub n_state: u32,
    /// `(lane, lag)`: input `k` of `prog` is `state[lane][i - lag]`.
    pub taps: Vec<(u32, u32)>,
    /// Sinks: the next word of each state lane, then optionally the output.
    pub prog: Program,
}

impl RecurrenceSpec {
    pub fn max_lag(&self) -> usize {
        self.taps.iter().map(|t| t.1).max().unwrap_or(0) as usize
    }

    pub fn min_lag(&self) -> usize {
        self.taps.iter().map(|t| t.1).min().unwrap_or(0) as usize
    }

    pub fn n_cols(&self) -> usize {
        self.prog.n_in as usize - self.taps.len()
    }

    pub fn width(&self) -> Width {
        self.prog.width
    }

    /// The sink each step's output word comes from.
    fn out_sink(&self) -> usize {
        if self.prog.sinks.len() > self.n_state as usize {
            self.n_state as usize
        } else {
            0
        }
    }

    pub fn validate(&self) -> Result<(), Invalid> {
        if !(1..=MAX_STATE).contains(&self.n_state) {
            return Err(Invalid("recurrence must have 1..=8 state lanes"));
        }
        if self.taps.is_empty() || self.taps.len() > MAX_TAPS {
            return Err(Invalid("recurrence must have 1..=16 taps"));
        }
        for &(lane, lag) in &self.taps {
            if lane >= self.n_state || lag == 0 || lag > MAX_LAG {
                return Err(Invalid("recurrence tap out of range"));
            }
        }
        let p = &self.prog;
        if (p.n_in as usize) < self.taps.len() {
            return Err(Invalid("recurrence program has fewer inputs than taps"));
        }
        if p.has_scans() || p.has_gathers() {
            return Err(Invalid(
                "recurrence program cannot contain scans or gathers",
            ));
        }
        let full = Len::Fixed(p.width.bits() as u8 / 8);
        let n = self.n_state as usize;
        if !(n..=n + 1).contains(&p.sinks.len()) || p.sinks.iter().any(|s| s.1 != full) {
            return Err(Invalid(
                "recurrence program must emit full words per state lane",
            ));
        }
        Ok(())
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_uv(&mut out, u64::from(self.n_state));
        put_uv(&mut out, self.taps.len() as u64);
        for &(lane, lag) in &self.taps {
            put_uv(&mut out, u64::from(lane));
            put_uv(&mut out, u64::from(lag));
        }
        let pb = self.prog.encode();
        put_uv(&mut out, pb.len() as u64);
        out.extend_from_slice(&pb);
        out
    }

    pub fn decode(mut b: &[u8]) -> Result<Self, Invalid> {
        let b = &mut b;
        let n_state = get_uv(b)?;
        let n_taps = get_uv(b)?;
        if n_state == 0 || n_state > u64::from(MAX_STATE) || n_taps == 0 || n_taps > MAX_TAPS as u64
        {
            return Err(Invalid("invalid recurrence lane or tap count"));
        }
        let mut taps = Vec::with_capacity(n_taps as usize);
        for _ in 0..n_taps {
            let (lane, lag) = (get_uv(b)?, get_uv(b)?);
            if lane >= n_state || lag == 0 || lag > u64::from(MAX_LAG) {
                return Err(Invalid("recurrence tap out of range"));
            }
            taps.push((lane as u32, lag as u32));
        }
        let len = get_uv(b)? as usize;
        if b.len() != len {
            return Err(Invalid("recurrence program length mismatch"));
        }
        let spec = Self {
            n_state: n_state as u32,
            taps,
            prog: Program::decode(b)?,
        };
        spec.validate()?;
        Ok(spec)
    }
}

pub struct CompiledRecurrence {
    spec: RecurrenceSpec,
}

impl CompiledRecurrence {
    pub fn compile(spec: &RecurrenceSpec) -> Result<Self, Invalid> {
        spec.validate()?;
        Ok(Self { spec: spec.clone() })
    }

    /// Generate `out.len()` 32-bit outputs following `seed`: `max_lag` steps of `n_state` words.
    pub fn eval_u32(&self, seed: &[u32], cols: &[&[u32]], out: &mut [u32]) {
        if self.spec.min_lag() < 3 {
            return eval_scalar_u32(&self.spec, seed, cols, out);
        }
        self.generate::<u32, _>(seed, cols, out, |x| x, |x| x);
    }

    /// Generate `out.len()` 64-bit outputs following `seed`: `max_lag` steps of `n_state` words.
    pub fn eval_i64(&self, seed: &[i64], cols: &[&[i64]], out: &mut [i64]) {
        if self.spec.min_lag() < 3 {
            return eval_scalar_i64(&self.spec, seed, cols, out);
        }
        self.generate::<u64, _>(seed, cols, out, |x| x as u64, |x| x as i64);
    }

    fn generate<T: Lane, I: Copy>(
        &self,
        seed: &[I],
        cols: &[&[I]],
        out: &mut [I],
        to: impl Fn(I) -> T,
        from: impl Fn(T) -> I,
    ) {
        let (spec, len) = (&self.spec, out.len());
        let (ns, max) = (spec.n_state as usize, spec.max_lag());
        assert!(seed.len() == max * ns && cols.len() == spec.n_cols());
        cols.iter().for_each(|c| assert!(c.len() >= len));
        let step = spec.min_lag().min(CHUNK);
        let prog = CompiledProgram::compile(&spec.prog);
        let mut run = Runner::new(&prog, step);
        // Each lane's words: the seed's `max`, then those made.
        let mut st: Vec<Vec<T>> = (0..ns)
            .map(|s| {
                let mut v = Vec::with_capacity(max + len);
                v.extend(seed[s..].iter().step_by(ns).map(|&x| to(x)));
                v
            })
            .collect();
        let taps = &spec.taps;
        let mut j = 0;
        while j < len {
            let (n, i) = (step.min(len - j), max + j);
            run.chunk(n, &[], |k, d| match taps.get(k) {
                Some(&(lane, lag)) => {
                    d.copy_from_slice(&st[lane as usize][i - lag as usize..][..n])
                }
                None => d
                    .iter_mut()
                    .zip(&cols[k - taps.len()][j..])
                    .for_each(|(d, &x)| *d = to(x)),
            });
            for (s, lane) in st.iter_mut().enumerate() {
                lane.extend_from_slice(&run.sink(s)[..n]);
            }
            out[j..j + n]
                .iter_mut()
                .zip(&run.sink(spec.out_sink())[..n])
                .for_each(|(o, &x)| *o = from(x));
            j += n;
        }
    }
}

// ============================================================================
// Scalar Reference / Fallback Evaluators
// ============================================================================

fn eval_one_u32(p: &Program, v: &mut Vec<u32>) {
    for ins in &p.ops {
        let au = v.get(ins.a as usize).copied().unwrap_or(0);
        let bu = v.get(ins.b as usize).copied().unwrap_or(0);
        let cu = v.get(ins.c as usize).copied().unwrap_or(0);
        let iu = ins.imm as u32;
        let r = match ins.op {
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
            Op::Const => iu,
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
                for b in &mut bytes {
                    *b = lut[(*b & 15) as usize];
                }
                u32::from_le_bytes(bytes)
            }
            Op::ByteLut32 => {
                let lo = &p.luts[ins.aux as usize];
                let hi = &p.luts[ins.aux as usize + 1];
                let mut bytes = au.to_le_bytes();
                for b in &mut bytes {
                    let k = (*b & 31) as usize;
                    *b = if k < 16 { lo[k] } else { hi[k - 16] };
                }
                u32::from_le_bytes(bytes)
            }
            _ => unreachable!(),
        };
        v.push(r);
    }
}

fn eval_one_i64(p: &Program, v: &mut Vec<i64>) {
    for ins in &p.ops {
        let a = v.get(ins.a as usize).copied().unwrap_or(0);
        let b = v.get(ins.b as usize).copied().unwrap_or(0);
        let c = v.get(ins.c as usize).copied().unwrap_or(0);
        let r = match ins.op {
            Op::Add => a.wrapping_add(b),
            Op::Sub => a.wrapping_sub(b),
            Op::Mul => a.wrapping_mul(b),
            Op::And => a & b,
            Op::Or => a | b,
            Op::Xor => a ^ b,
            Op::Shl => ((a as u64) << (b & 63)) as i64,
            Op::Shr => ((a as u64) >> (b & 63)) as i64,
            Op::F64Add => (f64::from_bits(a as u64) + f64::from_bits(b as u64)).to_bits() as i64,
            Op::F64Sub => (f64::from_bits(a as u64) - f64::from_bits(b as u64)).to_bits() as i64,
            Op::F64Mul => (f64::from_bits(a as u64) * f64::from_bits(b as u64)).to_bits() as i64,
            Op::Select => {
                if c != 0 {
                    a
                } else {
                    b
                }
            }
            Op::Const => ins.imm,
            Op::AddI => a.wrapping_add(ins.imm),
            Op::MulI => a.wrapping_mul(ins.imm),
            Op::AndI => a & ins.imm,
            Op::OrI => a | ins.imm,
            Op::XorI => a ^ ins.imm,
            Op::ShlI => ((a as u64) << (ins.imm & 63)) as i64,
            Op::ShrI => ((a as u64) >> (ins.imm & 63)) as i64,
            Op::SarI => a >> (ins.imm & 63),
            Op::MulHiI => ((a as u64 as u128 * ins.imm as u64 as u128) >> 64) as i64,
            Op::F64AddI => {
                (f64::from_bits(a as u64) + f64::from_bits(ins.imm as u64)).to_bits() as i64
            }
            Op::F64MulI => {
                (f64::from_bits(a as u64) * f64::from_bits(ins.imm as u64)).to_bits() as i64
            }
            Op::Ctz => (a as u64).trailing_zeros() as i64,
            Op::Bswap => a.swap_bytes(),
            Op::U64ToF64 => ((a as u64) as f64).to_bits() as i64,
            Op::I64ToF64 => (a as f64).to_bits() as i64,
            Op::F64DivI => {
                (f64::from_bits(a as u64) / f64::from_bits(ins.imm as u64)).to_bits() as i64
            }
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
            _ => unreachable!(),
        };
        v.push(r);
    }
}

/// Runs `spec` one step at a time; `eval` appends the program's values to its inputs.
fn scalar<T: Copy + Default>(
    spec: &RecurrenceSpec,
    seed: &[T],
    cols: &[&[T]],
    out: &mut [T],
    eval: impl Fn(&Program, &mut Vec<T>),
) {
    let (ns, max) = (spec.n_state as usize, spec.max_lag());
    assert!(seed.len() == max * ns);
    let mask = (max + 1).next_power_of_two() - 1;
    // Steps by position modulo the ring, `n_state` words each.
    let mut ring = vec![T::default(); (mask + 1) * ns];
    ring[..seed.len()].copy_from_slice(seed);
    let sinks: Vec<usize> = spec.prog.sinks.iter().map(|s| s.0 as usize).collect();
    let out_at = sinks[spec.out_sink()];
    let mut v = Vec::with_capacity(spec.prog.n_in as usize + spec.prog.ops.len());
    for (j, dst) in out.iter_mut().enumerate() {
        let pos = max + j;
        v.clear();
        for &(lane, lag) in &spec.taps {
            v.push(ring[((pos - lag as usize) & mask) * ns + lane as usize]);
        }
        v.extend(cols.iter().map(|c| c[j]));
        eval(&spec.prog, &mut v);
        let at = (pos & mask) * ns;
        for (s, &k) in sinks[..ns].iter().enumerate() {
            ring[at + s] = v[k];
        }
        *dst = v[out_at];
    }
}

/// Generate `out.len()` 32-bit outputs one step at a time.
pub fn eval_scalar_u32(spec: &RecurrenceSpec, seed: &[u32], cols: &[&[u32]], out: &mut [u32]) {
    scalar(spec, seed, cols, out, eval_one_u32);
}

/// Generate `out.len()` 64-bit outputs one step at a time.
pub fn eval_scalar_i64(spec: &RecurrenceSpec, seed: &[i64], cols: &[&[i64]], out: &mut [i64]) {
    scalar(spec, seed, cols, out, eval_one_i64);
}
