//! Lagged self-referential recurrence engine (`s[i] = f(s[i - lag_0], .., col[i])`).
//!
//! [`RecurrenceSpec`] is the portable wire representation carried inside a column payload: a list
//! of constant positive lags, a [`Program`] computing the next state word, and an optional
//! [`Program`] transforming each state word into an emitted output word (such as MT19937's temper).
//!
//! [`CompiledRecurrence::compile`] lowers any [`RecurrenceSpec`] with zero hardcoded generator IDs:
//! - **Wide lags (`min_lag >= T4`)**: fuses `state_prog` + `out_prog` into one 4-vector
//!   Direct-Threaded SIMD loop over a power-of-two mirrored L1d ring buffer (`64 u32` or `32 i64`
//!   lanes/step on AVX-512; `32 u32` or `16 i64` on AVX2).
//! - **Short lags (`2 <= min_lag < T4`)**: algebraically unfolds linear (`Add`/`Xor`) or general
//!   SSA recurrences until `min_lag >= LANES` (1 hardware vector), automatically warming the ring
//!   buffer from the caller's original `max_lag` seed before running a 1-vector Direct-Threaded
//!   SIMD loop.
//! - **Unit lag (`min_lag == 1`)**: dispatches associative `Add`/`Xor` scans to a tight register
//!   scan and arbitrary nonlinear 1-lag recurrences to a register-pinned scalar loop.

#![allow(
    unsafe_op_in_unsafe_fn,
    unused_assignments,
    unused_mut,
    improper_ctypes_definitions
)]

use std::collections::{BTreeMap, HashMap};

use super::ir::{Ins, Invalid, Len, Op, Program, Width, get_uv, put_uv};
use super::simd_arch::{self as arch, LANES32, LANES64, T32, T64, Tile, V};

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

// ============================================================================
// 4-Vector Direct-Threaded Ring-Buffer Engine (`T32` / `T64`)
// ============================================================================

#[cfg(target_arch = "x86_64")]
type RHandler4 = unsafe extern "sysv64" fn(
    ip: *const RNode4,
    ctx: *mut RCtx4,
    ring: *mut u8,
    pos: usize,
    out_ptr: *mut u8,
    a0: V,
    a1: V,
    a2: V,
    a3: V,
    b0: V,
    b1: V,
    b2: V,
    b3: V,
);

#[cfg(not(target_arch = "x86_64"))]
type RHandler4 = unsafe extern "C" fn(
    ip: *const RNode4,
    ctx: *mut RCtx4,
    ring: *mut u8,
    pos: usize,
    out_ptr: *mut u8,
    a0: V,
    a1: V,
    a2: V,
    a3: V,
    b0: V,
    b1: V,
    b2: V,
    b3: V,
);

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct RNode4 {
    vec_imm0: V,
    vec_imm1: V,
    next: RHandler4,
    aux: usize,
    imm_u64: u64,
}

#[repr(C, align(64))]
struct RCtx4 {
    spills: *mut Tile,
    col_ptrs: *const *const u8,
    mask: usize,
    ring_len: usize,
}

macro_rules! def_r4 {
    ($name:ident, |$ip:ident, $ctx:ident, $ring:ident, $pos:ident, $out:ident, $r0:ident, $r1:ident| $body:block) => {
        #[cfg(target_arch = "x86_64")]
        unsafe extern "sysv64" fn $name(
            $ip: *const RNode4,
            $ctx: *mut RCtx4,
            $ring: *mut u8,
            $pos: usize,
            $out: *mut u8,
            a0: V,
            a1: V,
            a2: V,
            a3: V,
            b0: V,
            b1: V,
            b2: V,
            b3: V,
        ) {
            #[allow(unused_mut)]
            let mut $r0: Tile = [a0, a1, a2, a3];
            #[allow(unused_mut)]
            let mut $r1: Tile = [b0, b1, b2, b3];
            $body
            let next_ip = $ip.add(1);
            ((*next_ip).next)(
                next_ip, $ctx, $ring, $pos, $out, $r0[0], $r0[1], $r0[2], $r0[3], $r1[0], $r1[1],
                $r1[2], $r1[3],
            )
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe extern "C" fn $name(
            $ip: *const RNode4,
            $ctx: *mut RCtx4,
            $ring: *mut u8,
            $pos: usize,
            $out: *mut u8,
            a0: V,
            a1: V,
            a2: V,
            a3: V,
            b0: V,
            b1: V,
            b2: V,
            b3: V,
        ) {
            #[allow(unused_mut)]
            let mut $r0: Tile = [a0, a1, a2, a3];
            #[allow(unused_mut)]
            let mut $r1: Tile = [b0, b1, b2, b3];
            $body
            let next_ip = $ip.add(1);
            ((*next_ip).next)(
                next_ip, $ctx, $ring, $pos, $out, $r0[0], $r0[1], $r0[2], $r0[3], $r1[0], $r1[1],
                $r1[2], $r1[3],
            )
        }
    };
}

#[cfg(target_arch = "x86_64")]
unsafe extern "sysv64" fn r4_end_tile(
    _ip: *const RNode4,
    _ctx: *mut RCtx4,
    _ring: *mut u8,
    _pos: usize,
    _out: *mut u8,
    _a0: V,
    _a1: V,
    _a2: V,
    _a3: V,
    _b0: V,
    _b1: V,
    _b2: V,
    _b3: V,
) {
}

#[cfg(not(target_arch = "x86_64"))]
unsafe extern "C" fn r4_end_tile(
    _ip: *const RNode4,
    _ctx: *mut RCtx4,
    _ring: *mut u8,
    _pos: usize,
    _out: *mut u8,
    _a0: V,
    _a1: V,
    _a2: V,
    _a3: V,
    _b0: V,
    _b1: V,
    _b2: V,
    _b3: V,
) {
}

// Ring loads/stores & external column loads (32-bit and 64-bit)
def_r4!(r4_load_lag32_0, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).aux) & (*ctx).mask;
    r0 = arch::load4_u32(ring.cast::<u32>().add(idx));
});
def_r4!(r4_load_lag32_1, |ip, ctx, ring, pos, _out, _r0, r1| {
    let idx = pos.wrapping_sub((*ip).aux) & (*ctx).mask;
    r1 = arch::load4_u32(ring.cast::<u32>().add(idx));
});
def_r4!(r4_xor_lag32_0, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).aux) & (*ctx).mask;
    r0 = arch::map2(r0, arch::load4_u32(ring.cast::<u32>().add(idx)), |x, y| {
        arch::xor_v(x, y)
    });
});
def_r4!(r4_add_lag32_0, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).aux) & (*ctx).mask;
    r0 = arch::map2(r0, arch::load4_u32(ring.cast::<u32>().add(idx)), |x, y| {
        arch::add32(x, y)
    });
});
def_r4!(r4_load_lag64_0, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).aux) & (*ctx).mask;
    r0 = arch::load4_i64(ring.cast::<i64>().add(idx));
});
def_r4!(r4_load_lag64_1, |ip, ctx, ring, pos, _out, _r0, r1| {
    let idx = pos.wrapping_sub((*ip).aux) & (*ctx).mask;
    r1 = arch::load4_i64(ring.cast::<i64>().add(idx));
});
def_r4!(r4_xor_lag64_0, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).aux) & (*ctx).mask;
    r0 = arch::map2(r0, arch::load4_i64(ring.cast::<i64>().add(idx)), |x, y| {
        arch::xor_v(x, y)
    });
});
def_r4!(r4_add_lag64_0, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).aux) & (*ctx).mask;
    r0 = arch::map2(r0, arch::load4_i64(ring.cast::<i64>().add(idx)), |x, y| {
        arch::add64(x, y)
    });
});

def_r4!(r4_load_col32_0, |ip, ctx, _ring, _pos, _out, r0, _r1| {
    let p = *(*ctx).col_ptrs.add((*ip).aux);
    r0 = arch::load4_u32(p.cast());
});
def_r4!(r4_load_col32_1, |ip, ctx, _ring, _pos, _out, _r0, r1| {
    let p = *(*ctx).col_ptrs.add((*ip).aux);
    r1 = arch::load4_u32(p.cast());
});
def_r4!(r4_load_col64_0, |ip, ctx, _ring, _pos, _out, r0, _r1| {
    let p = *(*ctx).col_ptrs.add((*ip).aux);
    r0 = arch::load4_i64(p.cast());
});
def_r4!(r4_load_col64_1, |ip, ctx, _ring, _pos, _out, _r0, r1| {
    let p = *(*ctx).col_ptrs.add((*ip).aux);
    r1 = arch::load4_i64(p.cast());
});

def_r4!(r4_store_ring32_0, |_ip, ctx, ring, pos, _out, r0, _r1| {
    let mask = (*ctx).mask;
    let ring_len = (*ctx).ring_len;
    let p = pos & mask;
    let rp = ring.cast::<u32>();
    arch::store4_u32(rp.add(p), r0);
    if p < T32 {
        arch::store4_u32(rp.add(ring_len + p), r0);
    } else if p + T32 > ring_len {
        let over = p + T32 - ring_len;
        core::ptr::copy_nonoverlapping(rp.add(ring_len), rp, over);
    }
});
def_r4!(r4_store_ring64_0, |_ip, ctx, ring, pos, _out, r0, _r1| {
    let mask = (*ctx).mask;
    let ring_len = (*ctx).ring_len;
    let p = pos & mask;
    let rp = ring.cast::<i64>();
    arch::store4_i64(rp.add(p), r0);
    if p < T64 {
        arch::store4_i64(rp.add(ring_len + p), r0);
    } else if p + T64 > ring_len {
        let over = p + T64 - ring_len;
        core::ptr::copy_nonoverlapping(rp.add(ring_len), rp, over);
    }
});
def_r4!(r4_store_out32_0, |_ip, _ctx, _ring, _pos, out, r0, _r1| {
    arch::store4_u32(out.cast(), r0);
});
def_r4!(r4_store_out64_0, |_ip, _ctx, _ring, _pos, out, r0, _r1| {
    arch::store4_i64(out.cast(), r0);
});

def_r4!(r4_load_spill0, |ip, ctx, _ring, _pos, _out, r0, _r1| {
    r0 = *(*ctx).spills.add((*ip).aux);
});
def_r4!(r4_load_spill1, |ip, ctx, _ring, _pos, _out, _r0, r1| {
    r1 = *(*ctx).spills.add((*ip).aux);
});
def_r4!(r4_spill0, |ip, ctx, _ring, _pos, _out, r0, _r1| {
    *(*ctx).spills.add((*ip).aux) = r0;
});
def_r4!(r4_spill1, |ip, ctx, _ring, _pos, _out, _r0, r1| {
    *(*ctx).spills.add((*ip).aux) = r1;
});
def_r4!(r4_mov10, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r1 = r0;
});
def_r4!(r4_mov01, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = r1;
});
def_r4!(r4_swap01, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    core::mem::swap(&mut r0, &mut r1);
});
def_r4!(r4_const0, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = [c, c, c, c];
});

def_r4!(r4_and, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::and_v(x, y));
});
def_r4!(r4_or, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::or_v(x, y));
});
def_r4!(r4_xor, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::xor_v(x, y));
});
def_r4!(r4_andi, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::and_v(x, c));
});
def_r4!(r4_ori, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::or_v(x, c));
});
def_r4!(r4_xori, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::xor_v(x, c));
});
def_r4!(r4_bitselect, |ip, _ctx, _ring, _pos, _out, r0, r1| {
    let m = (*ip).vec_imm0;
    r0 = arch::map2(r0, r1, |x, y| arch::bitselect(x, y, m));
});
def_r4!(r4_nibble_lut, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let t = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::nibble_lut(x, t));
});
def_r4!(r4_byte_lut32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let (lo, hi) = ((*ip).vec_imm0, (*ip).vec_imm1);
    r0 = arch::map1(r0, |x| arch::byte_lut32(x, lo, hi));
});
def_r4!(r4_f64_add, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::f64_add(x, y));
});
def_r4!(r4_f64_sub, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::f64_sub(x, y));
});
def_r4!(r4_f64_mul, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::f64_mul(x, y));
});
def_r4!(r4_f64_addi, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::f64_add(x, c));
});
def_r4!(r4_f64_muli, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::f64_mul(x, c));
});
def_r4!(r4_u64_to_f64, |_ip, _ctx, _ring, _pos, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::u64_to_f64(x));
});

def_r4!(r4_add32, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::add32(x, y));
});
def_r4!(r4_sub32, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::sub32(x, y));
});
def_r4!(r4_mul32, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::mul32(x, y));
});
def_r4!(r4_shl32, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::shl32_v(x, y));
});
def_r4!(r4_shr32, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::shr32_v(x, y));
});
def_r4!(r4_addi32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::add32(x, c));
});
def_r4!(r4_muli32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::mul32(x, c));
});
def_r4!(r4_muladdi32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let (m, c) = ((*ip).vec_imm0, (*ip).vec_imm1);
    r0 = arch::map1(r0, |x| arch::add32(arch::mul32(x, m), c));
});
def_r4!(r4_shli32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::shl32_i(x, cnt));
});
def_r4!(r4_shri32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::shr32_i(x, cnt));
});
def_r4!(r4_sari32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::sar32_i(x, cnt));
});
def_r4!(r4_ctz32, |_ip, _ctx, _ring, _pos, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::ctz32(x));
});
def_r4!(r4_bswap32, |_ip, _ctx, _ring, _pos, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::bswap32(x));
});
def_r4!(r4_select32_spill, |ip, ctx, _ring, _pos, _out, r0, r1| {
    let c = *(*ctx).spills.add((*ip).aux);
    r0 = arch::map3(c, r0, r1, |k, x, y| arch::select32(k, x, y));
});
def_r4!(r4_xor_shri32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::xor_v(x, arch::shr32_i(x, cnt)));
});
def_r4!(r4_xor_shl_andi32, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::xor_shl_and32(x, cnt, c));
});
def_r4!(r4_mask_lowbit_xor32, |ip,
                               _ctx,
                               _ring,
                               _pos,
                               _out,
                               r0,
                               r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map2(r0, r1, |x, y| arch::mask_lowbit_xor32(x, y, c));
});

def_r4!(r4_add64, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::add64(x, y));
});
def_r4!(r4_sub64, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::sub64(x, y));
});
def_r4!(r4_mul64, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::mul64(x, y));
});
def_r4!(r4_shl64, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::shl64_v(x, y));
});
def_r4!(r4_shr64, |_ip, _ctx, _ring, _pos, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::shr64_v(x, y));
});
def_r4!(r4_addi64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::add64(x, c));
});
def_r4!(r4_muli64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::mul64(x, c));
});
def_r4!(r4_muladdi64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let (m, c) = ((*ip).vec_imm0, (*ip).vec_imm1);
    r0 = arch::map1(r0, |x| arch::add64(arch::mul64(x, m), c));
});
def_r4!(r4_mulhii64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::mulhi64(x, c));
});
def_r4!(r4_shli64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::shl64_i(x, cnt));
});
def_r4!(r4_shri64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::shr64_i(x, cnt));
});
def_r4!(r4_sari64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::sar64_i(x, cnt));
});
def_r4!(r4_ctz64, |_ip, _ctx, _ring, _pos, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::ctz64(x));
});
def_r4!(r4_bswap64, |_ip, _ctx, _ring, _pos, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::bswap64(x));
});
def_r4!(r4_select64_spill, |ip, ctx, _ring, _pos, _out, r0, r1| {
    let c = *(*ctx).spills.add((*ip).aux);
    r0 = arch::map3(c, r0, r1, |k, x, y| arch::select64(k, x, y));
});
def_r4!(r4_xor_shri64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::xor_v(x, arch::shr64_i(x, cnt)));
});
def_r4!(r4_xor_shl_andi64, |ip, _ctx, _ring, _pos, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::xor_shl_and64(x, cnt, c));
});
def_r4!(r4_mask_lowbit_xor64, |ip,
                               _ctx,
                               _ring,
                               _pos,
                               _out,
                               r0,
                               r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map2(r0, r1, |x, y| arch::mask_lowbit_xor64(x, y, c));
});

// ============================================================================
// 1-Vector Direct-Threaded Linear Ring Engine (`LANES32` / `LANES64`)
// ============================================================================
// Used when short lags (e.g. LFG(3, 31)) unfold cleanly into `min_lag >= LANES` (16 on AVX-512,
// 8 on AVX2) without needing 4x wider combinational blowup.

#[cfg(target_arch = "x86_64")]
type RHandler1 = unsafe extern "sysv64" fn(
    ip: *const RNode1,
    ctx: *const RCtx1,
    ring: *mut u8,
    pos: usize,
    out_ptr: *mut u8,
    r0: V,
    r1: V,
);

#[cfg(not(target_arch = "x86_64"))]
type RHandler1 = unsafe extern "C" fn(
    ip: *const RNode1,
    ctx: *const RCtx1,
    ring: *mut u8,
    pos: usize,
    out_ptr: *mut u8,
    r0: V,
    r1: V,
);

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct RNode1 {
    vec_imm: V,
    next: RHandler1,
    lag: usize,
}

#[repr(C)]
struct RCtx1 {
    mask: usize,
    ring_len: usize,
}

macro_rules! def_r1 {
    ($name:ident, |$ip:ident, $ctx:ident, $ring:ident, $pos:ident, $out:ident, $r0:ident, $r1:ident| $body:block) => {
        #[cfg(target_arch = "x86_64")]
        unsafe extern "sysv64" fn $name(
            $ip: *const RNode1,
            $ctx: *const RCtx1,
            $ring: *mut u8,
            $pos: usize,
            $out: *mut u8,
            mut $r0: V,
            mut $r1: V,
        ) {
            let _ = &mut $r1;
            $body
            let next_ip = $ip.add(1);
            ((*next_ip).next)(next_ip, $ctx, $ring, $pos, $out, $r0, $r1)
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe extern "C" fn $name(
            $ip: *const RNode1,
            $ctx: *const RCtx1,
            $ring: *mut u8,
            $pos: usize,
            $out: *mut u8,
            mut $r0: V,
            mut $r1: V,
        ) {
            let _ = &mut $r1;
            $body
            let next_ip = $ip.add(1);
            ((*next_ip).next)(next_ip, $ctx, $ring, $pos, $out, $r0, $r1)
        }
    };
}

#[cfg(target_arch = "x86_64")]
unsafe extern "sysv64" fn r1_end_tile(
    _ip: *const RNode1,
    _ctx: *const RCtx1,
    _ring: *mut u8,
    _pos: usize,
    _out: *mut u8,
    _r0: V,
    _r1: V,
) {
}

#[cfg(not(target_arch = "x86_64"))]
unsafe extern "C" fn r1_end_tile(
    _ip: *const RNode1,
    _ctx: *const RCtx1,
    _ring: *mut u8,
    _pos: usize,
    _out: *mut u8,
    _r0: V,
    _r1: V,
) {
}

def_r1!(r1_load32, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    r0 = arch::load1_u32(ring.cast::<u32>().add(idx));
});
def_r1!(r1_mul_load32, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    r0 = arch::mul32(arch::load1_u32(ring.cast::<u32>().add(idx)), (*ip).vec_imm);
});
def_r1!(r1_add_lag32, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    r0 = arch::add32(r0, arch::load1_u32(ring.cast::<u32>().add(idx)));
});
def_r1!(r1_add_mul_lag32, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    let v = arch::mul32(arch::load1_u32(ring.cast::<u32>().add(idx)), (*ip).vec_imm);
    r0 = arch::add32(r0, v);
});
def_r1!(r1_xor_lag32, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    r0 = arch::xor_v(r0, arch::load1_u32(ring.cast::<u32>().add(idx)));
});
def_r1!(r1_store_both32, |_ip, ctx, ring, pos, out, r0, _r1| {
    let mask = (*ctx).mask;
    let ring_len = (*ctx).ring_len;
    let p = pos & mask;
    let rp = ring.cast::<u32>();
    arch::store1_u32(rp.add(p), r0);
    if p < LANES32 {
        arch::store1_u32(rp.add(ring_len + p), r0);
    } else if p + LANES32 > ring_len {
        let over = p + LANES32 - ring_len;
        core::ptr::copy_nonoverlapping(rp.add(ring_len), rp, over);
    }
    arch::store1_u32(out.cast(), r0);
});

def_r1!(r1_load64, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    r0 = arch::load1_i64(ring.cast::<i64>().add(idx));
});
def_r1!(r1_mul_load64, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    r0 = arch::mul64(arch::load1_i64(ring.cast::<i64>().add(idx)), (*ip).vec_imm);
});
def_r1!(r1_add_lag64, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    r0 = arch::add64(r0, arch::load1_i64(ring.cast::<i64>().add(idx)));
});
def_r1!(r1_add_mul_lag64, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    let v = arch::mul64(arch::load1_i64(ring.cast::<i64>().add(idx)), (*ip).vec_imm);
    r0 = arch::add64(r0, v);
});
def_r1!(r1_xor_lag64, |ip, ctx, ring, pos, _out, r0, _r1| {
    let idx = pos.wrapping_sub((*ip).lag) & (*ctx).mask;
    r0 = arch::xor_v(r0, arch::load1_i64(ring.cast::<i64>().add(idx)));
});
def_r1!(r1_store_both64, |_ip, ctx, ring, pos, out, r0, _r1| {
    let mask = (*ctx).mask;
    let ring_len = (*ctx).ring_len;
    let p = pos & mask;
    let rp = ring.cast::<i64>();
    arch::store1_i64(rp.add(p), r0);
    if p < LANES64 {
        arch::store1_i64(rp.add(ring_len + p), r0);
    } else if p + LANES64 > ring_len {
        let over = p + LANES64 - ring_len;
        core::ptr::copy_nonoverlapping(rp.add(ring_len), rp, over);
    }
    arch::store1_i64(out.cast(), r0);
});

// ============================================================================
// CompiledRecurrence & Automatic Lowering
// ============================================================================

enum ExecPlan {
    /// Associative lag-1 scan (`r[i] = r[i - 1] + col[i]` or `^ col[i]`).
    Scan1 { is_xor: bool },
    /// 4-vector wide Direct-Threaded ring-buffer execution (`min_lag >= T4`).
    Wide4 {
        nodes: Vec<RNode4>,
        n_spills: usize,
        max_lag: usize,
    },
    /// 1-vector unfolded linear (`Add`/`Xor`) recurrence (`min_lag >= LANES1` after unfolding).
    Linear1 {
        nodes: Vec<RNode1>,
        unfolded_max_lag: usize,
    },
    /// General scalar fallback (covers nonlinear `min_lag == 1` or exotic short-lag recurrences
    /// with external columns / out_prog).
    Scalar,
}

pub struct CompiledRecurrence {
    spec: RecurrenceSpec,
    plan: ExecPlan,
}

impl CompiledRecurrence {
    pub fn compile(spec: &RecurrenceSpec) -> Result<Self, Invalid> {
        spec.validate()?;
        let is32 = spec.width() == Width::W32;
        let t4 = if is32 { T32 } else { T64 };
        let lanes1 = if is32 { LANES32 } else { LANES64 };
        let min_lag = spec.min_lag();

        // 1. Check unit-lag associative scan (`r[i] = r[i - 1] + col[i]` or `r[i - 1] ^ col[i]`).
        if spec.lags == [1]
            && spec.n_cols() == 1
            && spec.out_prog.is_none()
            && spec.state_prog.ops.len() == 1
        {
            let ins = spec.state_prog.ops[0];
            if ((ins.a == 0 && ins.b == 1) || (ins.a == 1 && ins.b == 0))
                && matches!(ins.op, Op::Add | Op::Xor)
            {
                return Ok(Self {
                    spec: spec.clone(),
                    plan: ExecPlan::Scan1 {
                        is_xor: ins.op == Op::Xor,
                    },
                });
            }
        }

        // 2. Wide 4-vector ring mode (`min_lag >= T4`).
        if min_lag >= t4 {
            let (nodes, n_spills) = compile_wide4(spec);
            return Ok(Self {
                spec: spec.clone(),
                plan: ExecPlan::Wide4 {
                    nodes,
                    n_spills,
                    max_lag: spec.max_lag(),
                },
            });
        }

        // 3. Short-lag algebraic unfolding (`2 <= min_lag < T4`, pure autonomous recurrence).
        if min_lag >= 2 && spec.n_cols() == 0 && spec.out_prog.is_none() {
            if let Some((nodes, unfolded_max_lag)) = try_compile_linear1(spec, lanes1) {
                return Ok(Self {
                    spec: spec.clone(),
                    plan: ExecPlan::Linear1 {
                        nodes,
                        unfolded_max_lag,
                    },
                });
            }
            // Nonlinear autonomous short-lag recurrence: try general SSA DAG unfolding to `t4`.
            if let Some(unfolded_spec) = try_unfold_general_dag(spec, t4, 64) {
                let max_lag = unfolded_spec.max_lag();
                let (nodes, n_spills) = compile_wide4(&unfolded_spec);
                return Ok(Self {
                    spec: spec.clone(),
                    plan: ExecPlan::Wide4 {
                        nodes,
                        n_spills,
                        max_lag,
                    },
                });
            }
        }

        // 4. Register-fast scalar evaluator for general nonlinear lag-1 or remaining short lags.
        Ok(Self {
            spec: spec.clone(),
            plan: ExecPlan::Scalar,
        })
    }

    pub fn spec(&self) -> &RecurrenceSpec {
        &self.spec
    }

    /// Generate `out.len()` 32-bit words following `seed` (`seed.len() >= spec.max_lag()`).
    pub fn eval_u32(&self, seed: &[u32], cols: &[&[u32]], out: &mut [u32]) {
        assert_eq!(self.spec.width(), Width::W32);
        assert!(seed.len() >= self.spec.max_lag());
        assert_eq!(cols.len(), self.spec.n_cols());
        for c in cols {
            assert!(c.len() >= out.len());
        }
        if out.is_empty() {
            return;
        }

        match &self.plan {
            ExecPlan::Scan1 { is_xor } => {
                let mut acc = *seed.last().unwrap();
                if *is_xor {
                    for (o, &x) in out.iter_mut().zip(cols[0]) {
                        acc ^= x;
                        *o = acc;
                    }
                } else {
                    for (o, &x) in out.iter_mut().zip(cols[0]) {
                        acc = acc.wrapping_add(x);
                        *o = acc;
                    }
                }
            }
            ExecPlan::Wide4 {
                nodes,
                n_spills,
                max_lag,
            } => {
                run_wide4_u32(&self.spec, nodes, *n_spills, *max_lag, seed, cols, out);
            }
            ExecPlan::Linear1 {
                nodes,
                unfolded_max_lag,
            } => {
                run_linear1_u32(&self.spec, nodes, *unfolded_max_lag, seed, out);
            }
            ExecPlan::Scalar => {
                eval_scalar_u32(&self.spec, seed, cols, out);
            }
        }
    }

    /// Generate `out.len()` 64-bit words following `seed` (`seed.len() >= spec.max_lag()`).
    pub fn eval_i64(&self, seed: &[i64], cols: &[&[i64]], out: &mut [i64]) {
        assert_eq!(self.spec.width(), Width::W64);
        assert!(seed.len() >= self.spec.max_lag());
        assert_eq!(cols.len(), self.spec.n_cols());
        for c in cols {
            assert!(c.len() >= out.len());
        }
        if out.is_empty() {
            return;
        }

        match &self.plan {
            ExecPlan::Scan1 { is_xor } => {
                let mut acc = *seed.last().unwrap();
                if *is_xor {
                    for (o, &x) in out.iter_mut().zip(cols[0]) {
                        acc ^= x;
                        *o = acc;
                    }
                } else {
                    for (o, &x) in out.iter_mut().zip(cols[0]) {
                        acc = acc.wrapping_add(x);
                        *o = acc;
                    }
                }
            }
            ExecPlan::Wide4 {
                nodes,
                n_spills,
                max_lag,
            } => {
                run_wide4_i64(&self.spec, nodes, *n_spills, *max_lag, seed, cols, out);
            }
            ExecPlan::Linear1 {
                nodes,
                unfolded_max_lag,
            } => {
                run_linear1_i64(&self.spec, nodes, *unfolded_max_lag, seed, out);
            }
            ExecPlan::Scalar => {
                eval_scalar_i64(&self.spec, seed, cols, out);
            }
        }
    }
}

// ============================================================================
// Wide4 Compiler & Runners
// ============================================================================

#[derive(Clone, Copy)]
enum RVOp {
    Raw(Ins),
    Konst(i64),
    MulAddI { a: u32, mul: i64, add: i64 },
    BitSelect { a: u32, b: u32, mask: i64 },
    MaskLowBitXor { a: u32, b: u32, mask: i64 },
    XorShrI { a: u32, sh: u8 },
    XorShlAndI { a: u32, sh: u8, mask: i64 },
    XorLag { a: u32, lag: usize },
    AddLag { a: u32, lag: usize },
    StoreState { a: u32 },
}

fn compile_wide4(spec: &RecurrenceSpec) -> (Vec<RNode4>, usize) {
    let is32 = spec.width() == Width::W32;
    let bits = spec.width().bits();
    let wmask = if is32 { 0xffff_ffffu64 as i64 } else { -1i64 };
    let n_lags = spec.lags.len();
    let n_in = spec.state_prog.n_in as usize;

    // Combine `state_prog` + `StoreState` + optional `out_prog` into one unified SSA list!
    let mut all_ops: Vec<Ins> = spec.state_prog.ops.clone();
    let mut all_luts = spec.state_prog.luts.clone();
    let state_sink = spec.state_prog.sinks[0].0;
    let state_op_len = all_ops.len();

    let final_sink = if let Some(op) = &spec.out_prog {
        let lut_base = all_luts.len() as u8;
        all_luts.extend_from_slice(&op.luts);
        let base_val = (n_in + all_ops.len()) as u32;
        let map_v = |v: u32| -> u32 {
            if v == 0 {
                state_sink
            } else {
                base_val + (v - 1)
            }
        };
        for ins in &op.ops {
            let mut m = *ins;
            let ar = m.op.arity();
            if ar >= 1 {
                m.a = map_v(m.a);
            }
            if ar >= 2 {
                m.b = map_v(m.b);
            }
            if ar >= 3 {
                m.c = map_v(m.c);
            }
            if matches!(m.op, Op::NibbleLut | Op::ByteLut32) {
                m.aux += lut_base;
            }
            all_ops.push(m);
        }
        map_v(op.sinks[0].0)
    } else {
        state_sink
    };

    let mut base_uses = vec![0u32; n_in + all_ops.len()];
    for ins in &all_ops {
        for &v in &[ins.a, ins.b, ins.c][..ins.op.arity()] {
            base_uses[v as usize] += 1;
        }
    }
    base_uses[state_sink as usize] += 1;
    base_uses[final_sink as usize] += 1;

    let def_of = |v: u32| -> Option<&Ins> {
        (v as usize)
            .checked_sub(n_in)
            .and_then(|idx| all_ops.get(idx))
    };

    let mut vops: Vec<(usize, RVOp)> = Vec::with_capacity(all_ops.len() + 1);
    let mut const_val: Vec<Option<i64>> = vec![None; n_in + all_ops.len()];
    let mut store_emitted = false;
    if (state_sink as usize) < n_in {
        vops.push((usize::MAX, RVOp::StoreState { a: state_sink }));
        store_emitted = true;
    }

    for (i, &ins) in all_ops.iter().enumerate() {
        let vid = n_in + i;
        let rv = match ins.op {
            Op::AndI if (ins.imm & wmask) == 0 => {
                const_val[vid] = Some(0);
                RVOp::Konst(0)
            }
            Op::OrI | Op::AddI | Op::XorI if const_val[ins.a as usize].is_some() => {
                let p = const_val[ins.a as usize].unwrap();
                let k = match ins.op {
                    Op::OrI => p | ins.imm,
                    Op::AddI => p.wrapping_add(ins.imm),
                    _ => p ^ ins.imm,
                };
                const_val[vid] = Some(k);
                RVOp::Konst(k)
            }
            Op::AddI => {
                if let Some(d) = def_of(ins.a)
                    && d.op == Op::MulI
                    && base_uses[ins.a as usize] == 1
                {
                    RVOp::MulAddI {
                        a: d.a,
                        mul: d.imm,
                        add: ins.imm,
                    }
                } else {
                    RVOp::Raw(ins)
                }
            }
            Op::Or => {
                if let (Some(da), Some(db)) = (def_of(ins.a), def_of(ins.b))
                    && da.op == Op::AndI
                    && db.op == Op::AndI
                    && ((da.imm ^ db.imm) & wmask) == wmask
                    && base_uses[ins.a as usize] == 1
                    && base_uses[ins.b as usize] == 1
                {
                    RVOp::BitSelect {
                        a: da.a,
                        b: db.a,
                        mask: da.imm,
                    }
                } else {
                    RVOp::Raw(ins)
                }
            }
            Op::Add => {
                if (ins.b as usize) < n_lags
                    && base_uses[ins.b as usize] == 1
                    && (ins.a as usize) >= n_in
                {
                    RVOp::AddLag {
                        a: ins.a,
                        lag: spec.lags[ins.b as usize] as usize,
                    }
                } else if (ins.a as usize) < n_lags
                    && base_uses[ins.a as usize] == 1
                    && (ins.b as usize) >= n_in
                {
                    RVOp::AddLag {
                        a: ins.b,
                        lag: spec.lags[ins.a as usize] as usize,
                    }
                } else {
                    RVOp::Raw(ins)
                }
            }
            Op::Xor => {
                let try_fuse = |x: u32, y: u32| -> Option<RVOp> {
                    if (y as usize) < n_lags && base_uses[y as usize] == 1 && (x as usize) >= n_in {
                        return Some(RVOp::XorLag {
                            a: x,
                            lag: spec.lags[y as usize] as usize,
                        });
                    }
                    let dy = def_of(y)?;
                    if base_uses[y as usize] != 1 {
                        return None;
                    }
                    if dy.op == Op::ShrI && dy.a == x {
                        return Some(RVOp::XorShrI {
                            a: x,
                            sh: dy.imm as u8,
                        });
                    }
                    if dy.op == Op::AndI
                        && base_uses[dy.a as usize] == 1
                        && let Some(dsh) = def_of(dy.a)
                    {
                        if dsh.op == Op::ShlI && dsh.a == x {
                            return Some(RVOp::XorShlAndI {
                                a: x,
                                sh: dsh.imm as u8,
                                mask: dy.imm,
                            });
                        }
                        if dsh.op == Op::SarI
                            && dsh.imm as u32 == bits - 1
                            && base_uses[dsh.a as usize] == 1
                            && let Some(dshl) = def_of(dsh.a)
                            && dshl.op == Op::ShlI
                            && dshl.imm as u32 == bits - 1
                        {
                            return Some(RVOp::MaskLowBitXor {
                                a: x,
                                b: dshl.a,
                                mask: dy.imm,
                            });
                        }
                    }
                    None
                };
                try_fuse(ins.a, ins.b)
                    .or_else(|| try_fuse(ins.b, ins.a))
                    .unwrap_or(RVOp::Raw(ins))
            }
            _ => RVOp::Raw(ins),
        };
        vops.push((vid, rv));
        if i + 1 == state_op_len && !store_emitted {
            vops.push((usize::MAX, RVOp::StoreState { a: state_sink }));
            store_emitted = true;
        }
    }

    // Recompute exact use counts over lowered `vops`.
    let total_vals = n_in + all_ops.len();
    let mut uses = vec![0u32; total_vals];
    uses[final_sink as usize] += 1;
    for &(vid, rv) in vops.iter().rev() {
        let needed = if vid == usize::MAX {
            true
        } else {
            uses[vid] > 0
        };
        if !needed {
            continue;
        }
        match rv {
            RVOp::Konst(_) => {}
            RVOp::StoreState { a }
            | RVOp::MulAddI { a, .. }
            | RVOp::XorShrI { a, .. }
            | RVOp::XorShlAndI { a, .. }
            | RVOp::XorLag { a, .. }
            | RVOp::AddLag { a, .. } => uses[a as usize] += 1,
            RVOp::BitSelect { a, b, .. } | RVOp::MaskLowBitXor { a, b, .. } => {
                uses[a as usize] += 1;
                uses[b as usize] += 1;
            }
            RVOp::Raw(ins) => {
                for &v in &[ins.a, ins.b, ins.c][..ins.op.arity()] {
                    uses[v as usize] += 1;
                }
            }
        }
    }

    let z = unsafe { arch::zero() };
    let splat = |imm: i64| -> V {
        unsafe {
            if is32 {
                arch::splat32(imm as u32)
            } else {
                arch::splat64(imm)
            }
        }
    };
    let mk = |next: RHandler4, aux: usize, imm_u64: u64, v0: V, v1: V| RNode4 {
        vec_imm0: v0,
        vec_imm1: v1,
        next,
        aux,
        imm_u64,
    };

    let mut nodes: Vec<RNode4> = Vec::with_capacity(vops.len() * 2 + 4);
    let mut rem_uses = uses;
    let mut reg: [Option<usize>; 2] = [None, None];
    let mut spill_of: Vec<Option<usize>> = vec![None; total_vals];
    let mut free_spills: Vec<usize> = Vec::new();
    let mut n_spills = 0usize;

    let alloc_spill = |free_spills: &mut Vec<usize>, n_spills: &mut usize| -> usize {
        free_spills.pop().unwrap_or_else(|| {
            let s = *n_spills;
            *n_spills += 1;
            s
        })
    };

    let retire = |v: usize,
                  rem_uses: &[u32],
                  reg: &mut [Option<usize>; 2],
                  spill_of: &mut [Option<usize>],
                  free_spills: &mut Vec<usize>| {
        if rem_uses[v] == 0 {
            if reg[0] == Some(v) {
                reg[0] = None;
            }
            if reg[1] == Some(v) {
                reg[1] = None;
            }
            if let Some(s) = spill_of[v].take() {
                free_spills.push(s);
            }
        }
    };

    // Note: lag inputs (`v < n_lags`) MUST be loaded BEFORE `StoreState` overwrites the ring at `pos`,
    // or if `min_lag >= T4` note that `pos - lag` never overlaps `[pos, pos + T4)` unless `ring_len` wraps!
    // Since `ring_len >= (max_lag + 2*T4).next_power_of_two()`, `pos - lag` modulo `ring_len` NEVER
    // aliases `[pos, pos + T4)`! Even better, `out_prog` never references raw lag inputs (`v < n_lags`).
    let save_reg = |r: usize,
                    nodes: &mut Vec<RNode4>,
                    rem_uses: &[u32],
                    reg: &mut [Option<usize>; 2],
                    spill_of: &mut [Option<usize>],
                    free_spills: &mut Vec<usize>,
                    n_spills: &mut usize| {
        if let Some(v) = reg[r] {
            if rem_uses[v] > 0 && v >= n_in && spill_of[v].is_none() && reg[1 - r] != Some(v) {
                let s = alloc_spill(free_spills, n_spills);
                nodes.push(mk(if r == 0 { r4_spill0 } else { r4_spill1 }, s, 0, z, z));
                spill_of[v] = Some(s);
            }
            reg[r] = None;
        }
    };

    let load_into = |r: usize,
                     v: usize,
                     nodes: &mut Vec<RNode4>,
                     rem_uses: &[u32],
                     reg: &mut [Option<usize>; 2],
                     spill_of: &mut [Option<usize>],
                     free_spills: &mut Vec<usize>,
                     n_spills: &mut usize| {
        if reg[r] == Some(v) {
            return;
        }
        save_reg(r, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
        if reg[1 - r] == Some(v) {
            nodes.push(mk(if r == 0 { r4_mov01 } else { r4_mov10 }, 0, 0, z, z));
        } else if let Some(s) = spill_of[v] {
            nodes.push(mk(
                if r == 0 {
                    r4_load_spill0
                } else {
                    r4_load_spill1
                },
                s,
                0,
                z,
                z,
            ));
        } else if v < n_lags {
            let lag = spec.lags[v] as usize;
            let h = match (is32, r) {
                (true, 0) => r4_load_lag32_0,
                (true, _) => r4_load_lag32_1,
                (false, 0) => r4_load_lag64_0,
                (false, _) => r4_load_lag64_1,
            };
            nodes.push(mk(h, lag, 0, z, z));
        } else {
            debug_assert!(v < n_in);
            let col_idx = v - n_lags;
            let h = match (is32, r) {
                (true, 0) => r4_load_col32_0,
                (true, _) => r4_load_col32_1,
                (false, 0) => r4_load_col64_0,
                (false, _) => r4_load_col64_1,
            };
            nodes.push(mk(h, col_idx, 0, z, z));
        }
        reg[r] = Some(v);
    };

    let prep_unary = |a: usize,
                      nodes: &mut Vec<RNode4>,
                      rem_uses: &mut [u32],
                      reg: &mut [Option<usize>; 2],
                      spill_of: &mut [Option<usize>],
                      free_spills: &mut Vec<usize>,
                      n_spills: &mut usize| {
        load_into(0, a, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
        rem_uses[a] -= 1;
        if rem_uses[a] > 0 && a >= n_in && spill_of[a].is_none() && reg[1] != Some(a) {
            if reg[1].is_none_or(|x| rem_uses[x] == 0) {
                nodes.push(mk(r4_mov10, 0, 0, z, z));
                reg[1] = Some(a);
            } else {
                save_reg(0, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
            }
        }
        retire(a, rem_uses, reg, spill_of, free_spills);
    };

    let prep_bin = |mut a: usize,
                    mut b: usize,
                    comm: bool,
                    nodes: &mut Vec<RNode4>,
                    rem_uses: &mut [u32],
                    reg: &mut [Option<usize>; 2],
                    spill_of: &mut [Option<usize>],
                    free_spills: &mut Vec<usize>,
                    n_spills: &mut usize| {
        if a == b {
            load_into(0, a, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
            save_reg(1, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
            nodes.push(mk(r4_mov10, 0, 0, z, z));
            reg[1] = Some(a);
            rem_uses[a] -= 2;
            retire(a, rem_uses, reg, spill_of, free_spills);
            return;
        }
        if comm && (reg[0] == Some(b) || reg[1] == Some(a)) {
            core::mem::swap(&mut a, &mut b);
        }
        if reg[0] == Some(b) && reg[1] == Some(a) {
            nodes.push(mk(r4_swap01, 0, 0, z, z));
            reg.swap(0, 1);
        } else {
            if reg[0] == Some(b) {
                save_reg(1, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
                nodes.push(mk(r4_mov10, 0, 0, z, z));
                reg[1] = Some(b);
                reg[0] = None;
            }
            load_into(0, a, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
            load_into(1, b, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
        }
        rem_uses[a] -= 1;
        rem_uses[b] -= 1;
        if rem_uses[a] > 0 && a >= n_in && spill_of[a].is_none() {
            save_reg(0, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
        }
        retire(a, rem_uses, reg, spill_of, free_spills);
        retire(b, rem_uses, reg, spill_of, free_spills);
    };

    for &(vid, rv) in &vops {
        if vid != usize::MAX && rem_uses[vid] == 0 {
            continue;
        }
        match rv {
            RVOp::StoreState { a } => {
                let a = a as usize;
                load_into(
                    0,
                    a,
                    &mut nodes,
                    &rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                rem_uses[a] -= 1;
                nodes.push(mk(
                    if is32 {
                        r4_store_ring32_0
                    } else {
                        r4_store_ring64_0
                    },
                    0,
                    0,
                    z,
                    z,
                ));
                // Note: `r4_store_ring*_0` does NOT clobber `r0`! `a` stays live in `r0` for `out_prog`!
                retire(a, &rem_uses, &mut reg, &mut spill_of, &mut free_spills);
            }
            RVOp::Konst(k) => {
                save_reg(
                    0,
                    &mut nodes,
                    &rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(mk(r4_const0, 0, 0, splat(k), z));
                reg[0] = Some(vid);
            }
            RVOp::MulAddI { a, mul, add } => {
                prep_unary(
                    a as usize,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(mk(
                    if is32 { r4_muladdi32 } else { r4_muladdi64 },
                    0,
                    0,
                    splat(mul),
                    splat(add),
                ));
                reg[0] = Some(vid);
            }
            RVOp::XorShrI { a, sh } => {
                prep_unary(
                    a as usize,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(mk(
                    if is32 { r4_xor_shri32 } else { r4_xor_shri64 },
                    0,
                    sh as u64,
                    z,
                    z,
                ));
                reg[0] = Some(vid);
            }
            RVOp::XorShlAndI { a, sh, mask } => {
                prep_unary(
                    a as usize,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(mk(
                    if is32 {
                        r4_xor_shl_andi32
                    } else {
                        r4_xor_shl_andi64
                    },
                    0,
                    sh as u64,
                    splat(mask),
                    z,
                ));
                reg[0] = Some(vid);
            }
            RVOp::XorLag { a, lag } => {
                prep_unary(
                    a as usize,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(mk(
                    if is32 { r4_xor_lag32_0 } else { r4_xor_lag64_0 },
                    lag,
                    0,
                    z,
                    z,
                ));
                reg[0] = Some(vid);
            }
            RVOp::AddLag { a, lag } => {
                prep_unary(
                    a as usize,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(mk(
                    if is32 { r4_add_lag32_0 } else { r4_add_lag64_0 },
                    lag,
                    0,
                    z,
                    z,
                ));
                reg[0] = Some(vid);
            }
            RVOp::BitSelect { a, b, mask } => {
                prep_bin(
                    a as usize,
                    b as usize,
                    false,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(mk(r4_bitselect, 0, 0, splat(mask), z));
                reg[0] = Some(vid);
            }
            RVOp::MaskLowBitXor { a, b, mask } => {
                prep_bin(
                    a as usize,
                    b as usize,
                    false,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(mk(
                    if is32 {
                        r4_mask_lowbit_xor32
                    } else {
                        r4_mask_lowbit_xor64
                    },
                    0,
                    0,
                    splat(mask),
                    z,
                ));
                reg[0] = Some(vid);
            }
            RVOp::Raw(ins) => match ins.op.arity() {
                3 => {
                    let c = ins.c as usize;
                    let cs = if let Some(s) = spill_of[c] {
                        s
                    } else {
                        let s = alloc_spill(&mut free_spills, &mut n_spills);
                        load_into(
                            0,
                            c,
                            &mut nodes,
                            &rem_uses,
                            &mut reg,
                            &mut spill_of,
                            &mut free_spills,
                            &mut n_spills,
                        );
                        nodes.push(mk(r4_spill0, s, 0, z, z));
                        spill_of[c] = Some(s);
                        s
                    };
                    prep_bin(
                        ins.a as usize,
                        ins.b as usize,
                        false,
                        &mut nodes,
                        &mut rem_uses,
                        &mut reg,
                        &mut spill_of,
                        &mut free_spills,
                        &mut n_spills,
                    );
                    nodes.push(mk(
                        if is32 {
                            r4_select32_spill
                        } else {
                            r4_select64_spill
                        },
                        cs,
                        0,
                        z,
                        z,
                    ));
                    rem_uses[c] -= 1;
                    retire(c, &rem_uses, &mut reg, &mut spill_of, &mut free_spills);
                    reg[0] = Some(vid);
                }
                2 => {
                    let comm = matches!(
                        ins.op,
                        Op::Add | Op::Mul | Op::And | Op::Or | Op::Xor | Op::F64Add | Op::F64Mul
                    );
                    prep_bin(
                        ins.a as usize,
                        ins.b as usize,
                        comm,
                        &mut nodes,
                        &mut rem_uses,
                        &mut reg,
                        &mut spill_of,
                        &mut free_spills,
                        &mut n_spills,
                    );
                    let h: RHandler4 = match (is32, ins.op) {
                        (true, Op::Add) => r4_add32,
                        (true, Op::Sub) => r4_sub32,
                        (true, Op::Mul) => r4_mul32,
                        (true, Op::Shl) => r4_shl32,
                        (true, Op::Shr) => r4_shr32,
                        (false, Op::Add) => r4_add64,
                        (false, Op::Sub) => r4_sub64,
                        (false, Op::Mul) => r4_mul64,
                        (false, Op::Shl) => r4_shl64,
                        (false, Op::Shr) => r4_shr64,
                        (_, Op::And) => r4_and,
                        (_, Op::Or) => r4_or,
                        (_, Op::Xor) => r4_xor,
                        (_, Op::F64Add) => r4_f64_add,
                        (_, Op::F64Sub) => r4_f64_sub,
                        (_, Op::F64Mul) => r4_f64_mul,
                        _ => unreachable!(),
                    };
                    nodes.push(mk(h, 0, 0, z, z));
                    reg[0] = Some(vid);
                }
                _ => {
                    prep_unary(
                        ins.a as usize,
                        &mut nodes,
                        &mut rem_uses,
                        &mut reg,
                        &mut spill_of,
                        &mut free_spills,
                        &mut n_spills,
                    );
                    let n = match (is32, ins.op) {
                        (true, Op::AddI) => mk(r4_addi32, 0, 0, splat(ins.imm), z),
                        (false, Op::AddI) => mk(r4_addi64, 0, 0, splat(ins.imm), z),
                        (true, Op::MulI) => mk(r4_muli32, 0, 0, splat(ins.imm), z),
                        (false, Op::MulI) => mk(r4_muli64, 0, 0, splat(ins.imm), z),
                        (_, Op::AndI) => mk(r4_andi, 0, 0, splat(ins.imm), z),
                        (_, Op::OrI) => mk(r4_ori, 0, 0, splat(ins.imm), z),
                        (_, Op::XorI) => mk(r4_xori, 0, 0, splat(ins.imm), z),
                        (true, Op::ShlI) => mk(r4_shli32, 0, ins.imm as u64, z, z),
                        (false, Op::ShlI) => mk(r4_shli64, 0, ins.imm as u64, z, z),
                        (true, Op::ShrI) => mk(r4_shri32, 0, ins.imm as u64, z, z),
                        (false, Op::ShrI) => mk(r4_shri64, 0, ins.imm as u64, z, z),
                        (true, Op::SarI) => mk(r4_sari32, 0, ins.imm as u64, z, z),
                        (false, Op::SarI) => mk(r4_sari64, 0, ins.imm as u64, z, z),
                        (false, Op::MulHiI) => mk(r4_mulhii64, 0, 0, splat(ins.imm), z),
                        (_, Op::F64AddI) => mk(r4_f64_addi, 0, 0, splat(ins.imm), z),
                        (_, Op::F64MulI) => mk(r4_f64_muli, 0, 0, splat(ins.imm), z),
                        (true, Op::Ctz) => mk(r4_ctz32, 0, 0, z, z),
                        (false, Op::Ctz) => mk(r4_ctz64, 0, 0, z, z),
                        (true, Op::Bswap) => mk(r4_bswap32, 0, 0, z, z),
                        (false, Op::Bswap) => mk(r4_bswap64, 0, 0, z, z),
                        (_, Op::U64ToF64) => mk(r4_u64_to_f64, 0, 0, z, z),
                        (_, Op::NibbleLut) => {
                            let tbl = unsafe { arch::broadcast_lut16(&all_luts[ins.aux as usize]) };
                            mk(r4_nibble_lut, 0, 0, tbl, z)
                        }
                        (_, Op::ByteLut32) => {
                            let (lo, hi) = unsafe {
                                arch::broadcast_lut32(
                                    &all_luts[ins.aux as usize],
                                    &all_luts[ins.aux as usize + 1],
                                )
                            };
                            mk(r4_byte_lut32, 0, 0, lo, hi)
                        }
                        _ => unreachable!(),
                    };
                    nodes.push(n);
                    reg[0] = Some(vid);
                }
            },
        }
    }

    let fs = final_sink as usize;
    load_into(
        0,
        fs,
        &mut nodes,
        &rem_uses,
        &mut reg,
        &mut spill_of,
        &mut free_spills,
        &mut n_spills,
    );
    nodes.push(mk(
        if is32 {
            r4_store_out32_0
        } else {
            r4_store_out64_0
        },
        0,
        0,
        z,
        z,
    ));
    nodes.push(mk(r4_end_tile, 0, 0, z, z));
    (nodes, n_spills)
}

fn run_wide4_u32(
    spec: &RecurrenceSpec,
    nodes: &[RNode4],
    n_spills: usize,
    max_lag: usize,
    seed: &[u32],
    cols: &[&[u32]],
    out: &mut [u32],
) {
    // If `max_lag > seed.len()` (from nonlinear DAG unfolding), warm up the extra seed words first.
    let mut ext_seed: Vec<u32>;
    let mut seed_ref = seed;
    let mut out_offset = 0usize;
    if seed.len() < max_lag {
        let need = (max_lag - seed.len()).min(out.len());
        eval_scalar_u32(spec, seed, cols, &mut out[..need]);
        if need == out.len() {
            return;
        }
        ext_seed = Vec::with_capacity(seed.len() + need);
        ext_seed.extend_from_slice(seed);
        ext_seed.extend_from_slice(&out[..need]);
        seed_ref = &ext_seed;
        out_offset = need;
    }

    let out_rem = &mut out[out_offset..];
    let ring_len = (max_lag + 2 * T32).next_power_of_two();
    let mask = ring_len - 1;
    let mut ring = vec![0u32; ring_len + 2 * T32];
    ring[..max_lag].copy_from_slice(&seed_ref[seed_ref.len() - max_lag..]);
    ring.copy_within(0..T32, ring_len);

    let z = unsafe { arch::zero() };
    let mut spills = vec![[z; 4]; n_spills.max(1)];
    let mut col_ptrs = vec![core::ptr::null::<u8>(); cols.len()];
    let mut ctx = RCtx4 {
        spills: spills.as_mut_ptr(),
        col_ptrs: col_ptrs.as_ptr(),
        mask,
        ring_len,
    };

    let full = (out_rem.len() / T32) * T32;
    let ring_ptr = ring.as_mut_ptr().cast::<u8>();
    let start = nodes.as_ptr();
    let mut idx = 0usize;
    while idx < full {
        for (k, c) in cols.iter().enumerate() {
            col_ptrs[k] = unsafe { c.as_ptr().add(out_offset + idx).cast() };
        }
        unsafe {
            ((*start).next)(
                start,
                &raw mut ctx,
                ring_ptr,
                max_lag + idx,
                out_rem.as_mut_ptr().add(idx).cast(),
                z,
                z,
                z,
                z,
                z,
                z,
                z,
                z,
            );
        }
        idx += T32;
    }
    if idx < out_rem.len() {
        let rem = out_rem.len() - idx;
        let mut tail_cols = vec![[0u32; T32]; cols.len()];
        for (k, c) in cols.iter().enumerate() {
            tail_cols[k][..rem].copy_from_slice(&c[out_offset + idx..out_offset + out_rem.len()]);
            col_ptrs[k] = tail_cols[k].as_ptr().cast();
        }
        let mut tail_out = [0u32; T32];
        unsafe {
            ((*start).next)(
                start,
                &raw mut ctx,
                ring_ptr,
                max_lag + idx,
                tail_out.as_mut_ptr().cast(),
                z,
                z,
                z,
                z,
                z,
                z,
                z,
                z,
            );
        }
        out_rem[idx..].copy_from_slice(&tail_out[..rem]);
    }
}

fn run_wide4_i64(
    spec: &RecurrenceSpec,
    nodes: &[RNode4],
    n_spills: usize,
    max_lag: usize,
    seed: &[i64],
    cols: &[&[i64]],
    out: &mut [i64],
) {
    let mut ext_seed: Vec<i64>;
    let mut seed_ref = seed;
    let mut out_offset = 0usize;
    if seed.len() < max_lag {
        let need = (max_lag - seed.len()).min(out.len());
        eval_scalar_i64(spec, seed, cols, &mut out[..need]);
        if need == out.len() {
            return;
        }
        ext_seed = Vec::with_capacity(seed.len() + need);
        ext_seed.extend_from_slice(seed);
        ext_seed.extend_from_slice(&out[..need]);
        seed_ref = &ext_seed;
        out_offset = need;
    }

    let out_rem = &mut out[out_offset..];
    let ring_len = (max_lag + 2 * T64).next_power_of_two();
    let mask = ring_len - 1;
    let mut ring = vec![0i64; ring_len + 2 * T64];
    ring[..max_lag].copy_from_slice(&seed_ref[seed_ref.len() - max_lag..]);
    ring.copy_within(0..T64, ring_len);

    let z = unsafe { arch::zero() };
    let mut spills = vec![[z; 4]; n_spills.max(1)];
    let mut col_ptrs = vec![core::ptr::null::<u8>(); cols.len()];
    let mut ctx = RCtx4 {
        spills: spills.as_mut_ptr(),
        col_ptrs: col_ptrs.as_ptr(),
        mask,
        ring_len,
    };

    let full = (out_rem.len() / T64) * T64;
    let ring_ptr = ring.as_mut_ptr().cast::<u8>();
    let start = nodes.as_ptr();
    let mut idx = 0usize;
    while idx < full {
        for (k, c) in cols.iter().enumerate() {
            col_ptrs[k] = unsafe { c.as_ptr().add(out_offset + idx).cast() };
        }
        unsafe {
            ((*start).next)(
                start,
                &raw mut ctx,
                ring_ptr,
                max_lag + idx,
                out_rem.as_mut_ptr().add(idx).cast(),
                z,
                z,
                z,
                z,
                z,
                z,
                z,
                z,
            );
        }
        idx += T64;
    }
    if idx < out_rem.len() {
        let rem = out_rem.len() - idx;
        let mut tail_cols = vec![[0i64; T64]; cols.len()];
        for (k, c) in cols.iter().enumerate() {
            tail_cols[k][..rem].copy_from_slice(&c[out_offset + idx..out_offset + out_rem.len()]);
            col_ptrs[k] = tail_cols[k].as_ptr().cast();
        }
        let mut tail_out = [0i64; T64];
        unsafe {
            ((*start).next)(
                start,
                &raw mut ctx,
                ring_ptr,
                max_lag + idx,
                tail_out.as_mut_ptr().cast(),
                z,
                z,
                z,
                z,
                z,
                z,
                z,
                z,
            );
        }
        out_rem[idx..].copy_from_slice(&tail_out[..rem]);
    }
}

// ============================================================================
// Short-Lag Unfolding (Linear Ring & General SSA CSE)
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum RingKind {
    Add,
    Xor,
}

/// Detect whether `spec.state_prog` is a linear combination over `Add`/`MulI` or `Xor` of its lags,
/// and if so unfold short lags `< target_lanes` algebraically into a minimal 1-vector program.
fn try_compile_linear1(spec: &RecurrenceSpec, target_lanes: usize) -> Option<(Vec<RNode1>, usize)> {
    let is32 = spec.width() == Width::W32;
    let p = &spec.state_prog;
    let n_lags = spec.lags.len();

    // Determine whether the program is purely additive (`Add`/`Sub`/`MulI`) or purely XOR (`Xor`).
    let has_xor = p.ops.iter().any(|i| i.op == Op::Xor);
    let has_add = p
        .ops
        .iter()
        .any(|i| matches!(i.op, Op::Add | Op::Sub | Op::MulI));
    if has_xor && has_add {
        return None;
    }
    let kind = if has_xor {
        RingKind::Xor
    } else {
        RingKind::Add
    };

    // Symbolic linear form per SSA value: map `lag -> coeff`.
    let mut forms: Vec<BTreeMap<usize, u64>> = Vec::with_capacity(n_lags + p.ops.len());
    for &l in &spec.lags {
        let mut m = BTreeMap::new();
        *m.entry(l as usize).or_insert(0) = 1;
        forms.push(m);
    }
    let wrap = |x: u64| -> u64 { if is32 { x as u32 as u64 } else { x } };

    for ins in &p.ops {
        let mut cur = BTreeMap::new();
        match (kind, ins.op) {
            (RingKind::Add, Op::Add) => {
                for (&l, &c) in &forms[ins.a as usize] {
                    *cur.entry(l).or_insert(0) =
                        wrap(cur.get(&l).copied().unwrap_or(0u64).wrapping_add(c));
                }
                for (&l, &c) in &forms[ins.b as usize] {
                    let v = wrap(cur.get(&l).copied().unwrap_or(0u64).wrapping_add(c));
                    if v == 0 {
                        cur.remove(&l);
                    } else {
                        cur.insert(l, v);
                    }
                }
            }
            (RingKind::Add, Op::Sub) => {
                for (&l, &c) in &forms[ins.a as usize] {
                    *cur.entry(l).or_insert(0) = wrap(c);
                }
                for (&l, &c) in &forms[ins.b as usize] {
                    let v = wrap(cur.get(&l).copied().unwrap_or(0u64).wrapping_sub(c));
                    if v == 0 {
                        cur.remove(&l);
                    } else {
                        cur.insert(l, v);
                    }
                }
            }
            (RingKind::Add, Op::MulI) => {
                let k = wrap(ins.imm as u64);
                if k != 0 {
                    for (&l, &c) in &forms[ins.a as usize] {
                        let v = wrap(c.wrapping_mul(k));
                        if v != 0 {
                            cur.insert(l, v);
                        }
                    }
                }
            }
            (RingKind::Xor, Op::Xor) => {
                for (&l, &c) in &forms[ins.a as usize] {
                    if c & 1 != 0 {
                        cur.insert(l, 1);
                    }
                }
                for (&l, &c) in &forms[ins.b as usize] {
                    if c & 1 != 0 && cur.remove(&l).is_none() {
                        cur.insert(l, 1);
                    }
                }
            }
            _ => return None,
        }
        forms.push(cur);
    }

    let base_terms = forms[p.sinks[0].0 as usize].clone();
    if base_terms.is_empty() {
        return None;
    }

    // Substitute `r[i - smallest]` using `base_terms` shifted by `smallest` until `smallest >= target_lanes`.
    let mut terms = base_terms.clone();
    let mut steps = 0usize;
    while let Some((&smallest, &coeff)) = terms.iter().next() {
        if smallest >= target_lanes {
            break;
        }
        steps += 1;
        if steps > 128 {
            return None;
        }
        terms.remove(&smallest);
        if coeff == 0 {
            continue;
        }
        for (&bl, &bc) in &base_terms {
            let nl = smallest + bl;
            match kind {
                RingKind::Add => {
                    let add_c = wrap(coeff.wrapping_mul(bc));
                    let nv = wrap(terms.get(&nl).copied().unwrap_or(0).wrapping_add(add_c));
                    if nv == 0 {
                        terms.remove(&nl);
                    } else {
                        terms.insert(nl, nv);
                    }
                }
                RingKind::Xor => {
                    if (coeff & bc & 1) != 0 && terms.remove(&nl).is_none() {
                        terms.insert(nl, 1);
                    }
                }
            }
        }
    }

    if terms.is_empty() || terms.len() > 32 {
        return None;
    }

    let unfolded_max_lag = *terms.keys().max().unwrap();
    let z = unsafe { arch::zero() };
    let splat = |c: u64| -> V {
        unsafe {
            if is32 {
                arch::splat32(c as u32)
            } else {
                arch::splat64(c as i64)
            }
        }
    };

    let mut nodes = Vec::with_capacity(terms.len() + 2);
    for (idx, (&lag, &coeff)) in terms.iter().enumerate() {
        let n = match (is32, kind, idx == 0, coeff) {
            (true, _, true, 1) => RNode1 {
                vec_imm: z,
                next: r1_load32,
                lag,
            },
            (true, RingKind::Add, true, c) => RNode1 {
                vec_imm: splat(c),
                next: r1_mul_load32,
                lag,
            },
            (true, RingKind::Add, false, 1) => RNode1 {
                vec_imm: z,
                next: r1_add_lag32,
                lag,
            },
            (true, RingKind::Add, false, c) => RNode1 {
                vec_imm: splat(c),
                next: r1_add_mul_lag32,
                lag,
            },
            (true, RingKind::Xor, false, _) => RNode1 {
                vec_imm: z,
                next: r1_xor_lag32,
                lag,
            },
            (false, _, true, 1) => RNode1 {
                vec_imm: z,
                next: r1_load64,
                lag,
            },
            (false, RingKind::Add, true, c) => RNode1 {
                vec_imm: splat(c),
                next: r1_mul_load64,
                lag,
            },
            (false, RingKind::Add, false, 1) => RNode1 {
                vec_imm: z,
                next: r1_add_lag64,
                lag,
            },
            (false, RingKind::Add, false, c) => RNode1 {
                vec_imm: splat(c),
                next: r1_add_mul_lag64,
                lag,
            },
            (false, RingKind::Xor, false, _) => RNode1 {
                vec_imm: z,
                next: r1_xor_lag64,
                lag,
            },
            _ => unreachable!(),
        };
        nodes.push(n);
    }
    nodes.push(RNode1 {
        vec_imm: z,
        next: if is32 {
            r1_store_both32
        } else {
            r1_store_both64
        },
        lag: 0,
    });
    nodes.push(RNode1 {
        vec_imm: z,
        next: r1_end_tile,
        lag: 0,
    });
    Some((nodes, unfolded_max_lag))
}

fn run_linear1_u32(
    spec: &RecurrenceSpec,
    nodes: &[RNode1],
    unfolded_max_lag: usize,
    seed: &[u32],
    out: &mut [u32],
) {
    let mut ext_seed: Vec<u32>;
    let mut seed_ref = seed;
    let mut out_offset = 0usize;
    if seed.len() < unfolded_max_lag {
        let need = (unfolded_max_lag - seed.len()).min(out.len());
        eval_scalar_u32(spec, seed, &[], &mut out[..need]);
        if need == out.len() {
            return;
        }
        ext_seed = Vec::with_capacity(seed.len() + need);
        ext_seed.extend_from_slice(seed);
        ext_seed.extend_from_slice(&out[..need]);
        seed_ref = &ext_seed;
        out_offset = need;
    }

    let out_rem = &mut out[out_offset..];
    let ring_len = (unfolded_max_lag + 2 * LANES32).next_power_of_two();
    let mask = ring_len - 1;
    let mut ring = vec![0u32; ring_len + 2 * LANES32];
    ring[..unfolded_max_lag].copy_from_slice(&seed_ref[seed_ref.len() - unfolded_max_lag..]);
    ring.copy_within(0..LANES32, ring_len);

    let ctx = RCtx1 { mask, ring_len };
    let z = unsafe { arch::zero() };
    let ring_ptr = ring.as_mut_ptr().cast::<u8>();
    let start = nodes.as_ptr();
    let full = (out_rem.len() / LANES32) * LANES32;
    let mut idx = 0usize;
    while idx < full {
        unsafe {
            ((*start).next)(
                start,
                &raw const ctx,
                ring_ptr,
                unfolded_max_lag + idx,
                out_rem.as_mut_ptr().add(idx).cast(),
                z,
                z,
            );
        }
        idx += LANES32;
    }
    if idx < out_rem.len() {
        let rem = out_rem.len() - idx;
        let mut tail = [0u32; LANES32];
        unsafe {
            ((*start).next)(
                start,
                &raw const ctx,
                ring_ptr,
                unfolded_max_lag + idx,
                tail.as_mut_ptr().cast(),
                z,
                z,
            );
        }
        out_rem[idx..].copy_from_slice(&tail[..rem]);
    }
}

fn run_linear1_i64(
    spec: &RecurrenceSpec,
    nodes: &[RNode1],
    unfolded_max_lag: usize,
    seed: &[i64],
    out: &mut [i64],
) {
    let mut ext_seed: Vec<i64>;
    let mut seed_ref = seed;
    let mut out_offset = 0usize;
    if seed.len() < unfolded_max_lag {
        let need = (unfolded_max_lag - seed.len()).min(out.len());
        eval_scalar_i64(spec, seed, &[], &mut out[..need]);
        if need == out.len() {
            return;
        }
        ext_seed = Vec::with_capacity(seed.len() + need);
        ext_seed.extend_from_slice(seed);
        ext_seed.extend_from_slice(&out[..need]);
        seed_ref = &ext_seed;
        out_offset = need;
    }

    let out_rem = &mut out[out_offset..];
    let ring_len = (unfolded_max_lag + 2 * LANES64).next_power_of_two();
    let mask = ring_len - 1;
    let mut ring = vec![0i64; ring_len + 2 * LANES64];
    ring[..unfolded_max_lag].copy_from_slice(&seed_ref[seed_ref.len() - unfolded_max_lag..]);
    ring.copy_within(0..LANES64, ring_len);

    let ctx = RCtx1 { mask, ring_len };
    let z = unsafe { arch::zero() };
    let ring_ptr = ring.as_mut_ptr().cast::<u8>();
    let start = nodes.as_ptr();
    let full = (out_rem.len() / LANES64) * LANES64;
    let mut idx = 0usize;
    while idx < full {
        unsafe {
            ((*start).next)(
                start,
                &raw const ctx,
                ring_ptr,
                unfolded_max_lag + idx,
                out_rem.as_mut_ptr().add(idx).cast(),
                z,
                z,
            );
        }
        idx += LANES64;
    }
    if idx < out_rem.len() {
        let rem = out_rem.len() - idx;
        let mut tail = [0i64; LANES64];
        unsafe {
            ((*start).next)(
                start,
                &raw const ctx,
                ring_ptr,
                unfolded_max_lag + idx,
                tail.as_mut_ptr().cast(),
                z,
                z,
            );
        }
        out_rem[idx..].copy_from_slice(&tail[..rem]);
    }
}

/// General SSA DAG unfolder with hash-consing CSE for nonlinear autonomous recurrences.
fn try_unfold_general_dag(
    spec: &RecurrenceSpec,
    target_min_lag: usize,
    max_ops: usize,
) -> Option<RecurrenceSpec> {
    #[derive(Clone, Hash, PartialEq, Eq)]
    enum SymNode {
        Lag(u32),
        Ins {
            op: Op,
            a: usize,
            b: usize,
            c: usize,
            aux: u8,
            imm: i64,
        },
    }
    let mut nodes: Vec<SymNode> = Vec::new();
    let mut memo: HashMap<SymNode, usize> = HashMap::new();
    let mut intern =
        |n: SymNode, nodes: &mut Vec<SymNode>, memo: &mut HashMap<SymNode, usize>| -> usize {
            if let Some(&id) = memo.get(&n) {
                id
            } else {
                let id = nodes.len();
                memo.insert(n.clone(), id);
                nodes.push(n);
                id
            }
        };

    fn expand(
        spec: &RecurrenceSpec,
        shift: u32,
        target: u32,
        nodes: &mut Vec<SymNode>,
        memo: &mut HashMap<SymNode, usize>,
        cache: &mut HashMap<u32, usize>,
        intern: &mut impl FnMut(SymNode, &mut Vec<SymNode>, &mut HashMap<SymNode, usize>) -> usize,
        max_ops: usize,
    ) -> Option<usize> {
        if shift >= target {
            return Some(intern(SymNode::Lag(shift), nodes, memo));
        }
        if let Some(&id) = cache.get(&shift) {
            return Some(id);
        }
        if nodes.len() > max_ops * 2 {
            return None;
        }
        let p = &spec.state_prog;
        let mut vals = Vec::with_capacity(spec.lags.len() + p.ops.len());
        for &l in &spec.lags {
            vals.push(expand(
                spec,
                shift + l,
                target,
                nodes,
                memo,
                cache,
                intern,
                max_ops,
            )?);
        }
        for ins in &p.ops {
            let ar = ins.op.arity();
            let a = if ar >= 1 { vals[ins.a as usize] } else { 0 };
            let b = if ar >= 2 { vals[ins.b as usize] } else { 0 };
            let c = if ar >= 3 { vals[ins.c as usize] } else { 0 };
            vals.push(intern(
                SymNode::Ins {
                    op: ins.op,
                    a,
                    b,
                    c,
                    aux: ins.aux,
                    imm: ins.imm,
                },
                nodes,
                memo,
            ));
        }
        let root = vals[p.sinks[0].0 as usize];
        cache.insert(shift, root);
        Some(root)
    }

    let mut cache = HashMap::new();
    let root = expand(
        spec,
        0,
        target_min_lag as u32,
        &mut nodes,
        &mut memo,
        &mut cache,
        &mut intern,
        max_ops,
    )?;

    // Collect reachable lags (<= 16) and instructions (<= max_ops).
    let mut lags: Vec<u32> = Vec::new();
    let mut remap: HashMap<usize, u32> = HashMap::new();
    for (id, n) in nodes.iter().enumerate() {
        if let SymNode::Lag(l) = *n {
            let lid = lags.len() as u32;
            lags.push(l);
            remap.insert(id, lid);
        }
    }
    if lags.is_empty() || lags.len() > 16 {
        return None;
    }
    let mut ops: Vec<Ins> = Vec::new();
    for (id, n) in nodes.iter().enumerate() {
        if let SymNode::Ins {
            op,
            a,
            b,
            c,
            aux,
            imm,
        } = *n
        {
            let ar = op.arity();
            let ma = if ar >= 1 { remap[&a] } else { 0 };
            let mb = if ar >= 2 { remap[&b] } else { 0 };
            let mc = if ar >= 3 { remap[&c] } else { 0 };
            let vid = lags.len() as u32 + ops.len() as u32;
            ops.push(Ins {
                op,
                a: ma,
                b: mb,
                c: mc,
                aux,
                imm,
            });
            remap.insert(id, vid);
        }
    }
    if ops.len() > max_ops {
        return None;
    }
    let sink_w = if spec.width() == Width::W32 { 4 } else { 8 };
    Some(RecurrenceSpec {
        lags,
        state_prog: Program {
            width: spec.width(),
            n_in: remap.len() as u32 - ops.len() as u32,
            n_bufs: 0,
            luts: spec.state_prog.luts.clone(),
            ops,
            sinks: vec![(remap[&root], Len::Fixed(sink_w))],
        },
        out_prog: None,
    })
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
