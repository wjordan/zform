//! Lagged self-referential recurrences (`s[i] = f(s[i - lag_0], .., col[i])`).
//!
//! [`RecurrenceSpec`] is the portable wire representation carried inside a column payload: a list
//! of constant positive lags, a [`Program`] computing the next state word, and an optional
//! [`Program`] transforming each state word into an emitted output word (such as MT19937's temper).
//!
//! [`CompiledRecurrence`] runs a spec through the program executor: with every lag at least `n`,
//! the next `n` state words depend only on words already made, so one chunk of `state_prog` makes
//! them all from windows of the history. Lags under 3 go word by word through the scalar
//! evaluator, which measured faster there.

use super::engine::{CHUNK, CompiledProgram, Lane, Runner};
use super::ir::{Invalid, Len, Op, Program, Width, get_uv, put_uv};

/// Wire-serializable specification of a lagged recurrence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecurrenceSpec {
    pub lags: Vec<u32>,
    pub state_prog: Program,
    pub out_prog: Option<Program>,
}

impl RecurrenceSpec {
    pub fn max_lag(&self) -> usize {
        self.lags.iter().copied().max().unwrap_or(0) as usize
    }

    pub fn min_lag(&self) -> usize {
        self.lags.iter().copied().min().unwrap_or(0) as usize
    }

    pub fn n_cols(&self) -> usize {
        self.state_prog.n_in as usize - self.lags.len()
    }

    pub fn width(&self) -> Width {
        self.state_prog.width
    }

    pub fn validate(&self) -> Result<(), Invalid> {
        if self.lags.is_empty() || self.lags.len() > 16 {
            return Err(Invalid("recurrence must have 1..=16 lags"));
        }
        for &l in &self.lags {
            if l == 0 || l > (1 << 20) {
                return Err(Invalid("recurrence lag out of range"));
            }
        }
        if (self.state_prog.n_in as usize) < self.lags.len() {
            return Err(Invalid("state_prog has fewer inputs than lags"));
        }
        if self.state_prog.has_scans() || self.state_prog.has_gathers() {
            return Err(Invalid(
                "recurrence state_prog cannot contain scans or gathers",
            ));
        }
        let expected_sink = match self.state_prog.width {
            Width::W32 => Len::Fixed(4),
            Width::W64 => Len::Fixed(8),
        };
        if self.state_prog.sinks.len() != 1 || self.state_prog.sinks[0].1 != expected_sink {
            return Err(Invalid("state_prog must emit one full-lane word"));
        }
        if let Some(out_p) = &self.out_prog {
            if out_p.width != self.state_prog.width || out_p.n_in != 1 {
                return Err(Invalid("out_prog must have matching width and 1 input"));
            }
            if out_p.has_scans() || out_p.has_gathers() {
                return Err(Invalid(
                    "recurrence out_prog cannot contain scans or gathers",
                ));
            }
            if out_p.sinks.len() != 1 || out_p.sinks[0].1 != expected_sink {
                return Err(Invalid("out_prog must emit one full-lane word"));
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_uv(&mut out, self.lags.len() as u64);
        for &l in &self.lags {
            put_uv(&mut out, l as u64);
        }
        let sb = self.state_prog.encode();
        put_uv(&mut out, sb.len() as u64);
        out.extend_from_slice(&sb);
        match &self.out_prog {
            None => out.push(0),
            Some(op) => {
                out.push(1);
                let ob = op.encode();
                put_uv(&mut out, ob.len() as u64);
                out.extend_from_slice(&ob);
            }
        }
        out
    }

    pub fn decode(mut b: &[u8]) -> Result<Self, Invalid> {
        let b = &mut b;
        let n_lags = get_uv(b)?;
        if n_lags == 0 || n_lags > 16 {
            return Err(Invalid("invalid lag count"));
        }
        let mut lags = Vec::with_capacity(n_lags as usize);
        for _ in 0..n_lags {
            let l = get_uv(b)?;
            if l == 0 || l > (1 << 20) {
                return Err(Invalid("lag out of range"));
            }
            lags.push(l as u32);
        }
        let slen = get_uv(b)? as usize;
        if b.len() < slen {
            return Err(Invalid("truncated state_prog"));
        }
        let (sbytes, rest) = b.split_at(slen);
        *b = rest;
        let state_prog = Program::decode(sbytes)?;
        let (&has_out, rest) = b.split_first().ok_or(Invalid("truncated out_prog flag"))?;
        *b = rest;
        let out_prog = match has_out {
            0 => None,
            1 => {
                let olen = get_uv(b)? as usize;
                if b.len() < olen {
                    return Err(Invalid("truncated out_prog"));
                }
                let (obytes, rest) = b.split_at(olen);
                *b = rest;
                Some(Program::decode(obytes)?)
            }
            _ => return Err(Invalid("invalid out_prog flag")),
        };
        if !b.is_empty() {
            return Err(Invalid("trailing bytes in recurrence"));
        }
        let spec = Self {
            lags,
            state_prog,
            out_prog,
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

    /// Generate `out.len()` 32-bit words following `seed` (`seed.len() >= spec.max_lag()`).
    pub fn eval_u32(&self, seed: &[u32], cols: &[&[u32]], out: &mut [u32]) {
        if self.spec.min_lag() < 3 {
            return eval_scalar_u32(&self.spec, seed, cols, out);
        }
        self.generate::<u32, _>(seed, cols, out, |x| x, |x| x);
    }

    /// Generate `out.len()` 64-bit words following `seed` (`seed.len() >= spec.max_lag()`).
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
        let (lags, max) = (&spec.lags, spec.max_lag());
        assert!(seed.len() >= max && cols.len() == spec.n_cols());
        cols.iter().for_each(|c| assert!(c.len() >= len));
        let step = spec.min_lag().min(CHUNK);
        let state = CompiledProgram::compile(&spec.state_prog);
        let map = spec.out_prog.as_ref().map(CompiledProgram::compile);
        let mut rs = Runner::new(&state, step);
        let mut ro = map.as_ref().map(|p| Runner::new(p, step));
        // Every state word: the seed's last `max`, then those made.
        let mut st: Vec<T> = seed[seed.len() - max..].iter().map(|&x| to(x)).collect();
        st.reserve(len);
        let mut j = 0;
        while j < len {
            let (n, i) = (step.min(len - j), max + j);
            rs.chunk(n, &[], |k, d| match lags.get(k) {
                Some(&lag) => d.copy_from_slice(&st[i - lag as usize..][..n]),
                None => d
                    .iter_mut()
                    .zip(&cols[k - lags.len()][j..])
                    .for_each(|(d, &x)| *d = to(x)),
            });
            st.extend_from_slice(&rs.sink()[..n]);
            let made = match &mut ro {
                Some(ro) => {
                    ro.chunk(n, &[], |_, d| d.copy_from_slice(&st[i..][..n]));
                    &ro.sink()[..n]
                }
                None => &st[i..][..n],
            };
            out[j..j + n]
                .iter_mut()
                .zip(made)
                .for_each(|(o, &x)| *o = from(x));
            j += n;
        }
    }
}

// ============================================================================
// Scalar Reference / Fallback Evaluators
// ============================================================================

fn eval_one_u32(p: &Program, v: &mut Vec<u32>) -> u32 {
    for ins in &p.ops {
        let au = v[ins.a as usize];
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
    v[p.sinks[0].0 as usize]
}

fn eval_one_i64(p: &Program, v: &mut Vec<i64>) -> i64 {
    for ins in &p.ops {
        let a = v[ins.a as usize];
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
    v[p.sinks[0].0 as usize]
}

pub fn eval_scalar_u32(spec: &RecurrenceSpec, seed: &[u32], cols: &[&[u32]], out: &mut [u32]) {
    let max_lag = spec.max_lag();
    let ring_len = (max_lag + 1).next_power_of_two();
    let mask = ring_len - 1;
    let mut ring = vec![0u32; ring_len];
    for (i, &w) in seed[seed.len() - max_lag..].iter().enumerate() {
        ring[i] = w;
    }
    let mut v = Vec::with_capacity(spec.state_prog.n_in as usize + spec.state_prog.ops.len());
    let mut ov = Vec::new();
    for (row, dst) in out.iter_mut().enumerate() {
        let pos = max_lag + row;
        v.clear();
        for &l in &spec.lags {
            v.push(ring[pos.wrapping_sub(l as usize) & mask]);
        }
        for c in cols {
            v.push(c[row]);
        }
        let st = eval_one_u32(&spec.state_prog, &mut v);
        ring[pos & mask] = st;
        *dst = if let Some(op) = &spec.out_prog {
            ov.clear();
            ov.push(st);
            eval_one_u32(op, &mut ov)
        } else {
            st
        };
    }
}

pub fn eval_scalar_i64(spec: &RecurrenceSpec, seed: &[i64], cols: &[&[i64]], out: &mut [i64]) {
    let max_lag = spec.max_lag();
    let ring_len = (max_lag + 1).next_power_of_two();
    let mask = ring_len - 1;
    let mut ring = vec![0i64; ring_len];
    for (i, &w) in seed[seed.len() - max_lag..].iter().enumerate() {
        ring[i] = w;
    }
    let mut v = Vec::with_capacity(spec.state_prog.n_in as usize + spec.state_prog.ops.len());
    let mut ov = Vec::new();
    for (row, dst) in out.iter_mut().enumerate() {
        let pos = max_lag + row;
        v.clear();
        for &l in &spec.lags {
            v.push(ring[pos.wrapping_sub(l as usize) & mask]);
        }
        for c in cols {
            v.push(c[row]);
        }
        let st = eval_one_i64(&spec.state_prog, &mut v);
        ring[pos & mask] = st;
        *dst = if let Some(op) = &spec.out_prog {
            ov.clear();
            ov.push(st);
            eval_one_i64(op, &mut ov)
        } else {
            st
        };
    }
}
