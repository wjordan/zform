//! Zero-JIT Direct-Threaded SIMD execution engine for [`Program`].
//!
//! [`CompiledProgram::compile`] lowers a validated [`Program`] in a few microseconds:
//! 1. Applies target-independent algebraic peepholes (`Konst`, `BitSelect`, `MaskLowBitXor`,
//!    `XorShrI`, `XorShlAndI`, `MulAddI`, `SelectMaskI`) over the SSA DAG.
//! 2. Schedules the DAG onto two 4-vector hardware accumulators (`r0`, `r1`, holding 64 `u32`
//!    or 32 `i64` lanes at once on AVX-512; 32 `u32` or 16 `i64` on AVX2) backed by reusable L1d
//!    spill slots when live width exceeds two tiles.
//! 3. Emits a flat array of cache-line-aligned `Node`s whose handlers chain via register-pinned
//!    tail calls (`extern "sysv64"` on x86-64, `extern "C"` elsewhere) within each tile and return
//!    cleanly at the end of every tile so stack usage is O(1) even in unoptimized debug builds.

#![allow(
    unsafe_op_in_unsafe_fn,
    unused_assignments,
    unused_mut,
    improper_ctypes_definitions
)]

use std::sync::Arc;

use super::ir::{Ins, Len, Op, Program, TableBuf, Width};
use super::library::Code;
use super::simd_arch::{self as arch, T32, T64, Tile, V};

#[cfg(target_arch = "x86_64")]
type Handler = unsafe extern "sysv64" fn(
    ip: *const Node,
    ctx: *mut ExecCtx,
    tile_rows: usize,
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
type Handler = unsafe extern "C" fn(
    ip: *const Node,
    ctx: *mut ExecCtx,
    tile_rows: usize,
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
struct Node {
    vec_imm0: V,
    vec_imm1: V,
    next: Handler,
    aux0: usize,
    aux1: usize,
    imm_u64: u64,
}

#[repr(C, align(64))]
struct ExecCtx {
    in_ptrs: *const *const u8,
    spills: *mut Tile,
    sinks: *mut Tile,
    bufs: *const TableBuf<'static>,
    n_bufs: usize,
    scan_states: *mut i64,
}

macro_rules! def_handler {
    ($name:ident, |$ip:ident, $ctx:ident, $rows:ident, $out:ident, $r0:ident, $r1:ident| $body:block) => {
        #[cfg(target_arch = "x86_64")]
        unsafe extern "sysv64" fn $name(
            $ip: *const Node,
            $ctx: *mut ExecCtx,
            $rows: usize,
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
                next_ip, $ctx, $rows, $out, $r0[0], $r0[1], $r0[2], $r0[3], $r1[0], $r1[1], $r1[2],
                $r1[3],
            )
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe extern "C" fn $name(
            $ip: *const Node,
            $ctx: *mut ExecCtx,
            $rows: usize,
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
                next_ip, $ctx, $rows, $out, $r0[0], $r0[1], $r0[2], $r0[3], $r1[0], $r1[1], $r1[2],
                $r1[3],
            )
        }
    };
}

#[cfg(target_arch = "x86_64")]
unsafe extern "sysv64" fn op_end_tile(
    _ip: *const Node,
    _ctx: *mut ExecCtx,
    _rows: usize,
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
unsafe extern "C" fn op_end_tile(
    _ip: *const Node,
    _ctx: *mut ExecCtx,
    _rows: usize,
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

// --- Data movement & direct-output handlers ---
def_handler!(op_load_in32_0, |ip, ctx, _rows, _out, r0, _r1| {
    let p = *(*ctx).in_ptrs.add((*ip).aux0);
    r0 = arch::load4_u32(p.cast());
});
def_handler!(op_load_in32_1, |ip, ctx, _rows, _out, _r0, r1| {
    let p = *(*ctx).in_ptrs.add((*ip).aux0);
    r1 = arch::load4_u32(p.cast());
});
def_handler!(op_load_in64_0, |ip, ctx, _rows, _out, r0, _r1| {
    let p = *(*ctx).in_ptrs.add((*ip).aux0);
    r0 = arch::load4_i64(p.cast());
});
def_handler!(op_load_in64_1, |ip, ctx, _rows, _out, _r0, r1| {
    let p = *(*ctx).in_ptrs.add((*ip).aux0);
    r1 = arch::load4_i64(p.cast());
});
// Load an `&[i64]` input directly into a W32 tile by truncating each 64-bit word to 32 bits.
def_handler!(op_load_in64_as_u32_0, |ip, ctx, _rows, _out, r0, _r1| {
    let p = (*(*ctx).in_ptrs.add((*ip).aux0)).cast::<i64>();
    let mut tmp = [0u32; T32];
    for i in 0..T32 {
        tmp[i] = *p.add(i) as u32;
    }
    r0 = arch::load4_u32(tmp.as_ptr());
});
def_handler!(op_load_in64_as_u32_1, |ip, ctx, _rows, _out, _r0, r1| {
    let p = (*(*ctx).in_ptrs.add((*ip).aux0)).cast::<i64>();
    let mut tmp = [0u32; T32];
    for i in 0..T32 {
        tmp[i] = *p.add(i) as u32;
    }
    r1 = arch::load4_u32(tmp.as_ptr());
});
def_handler!(op_load_spill0, |ip, ctx, _rows, _out, r0, _r1| {
    r0 = *(*ctx).spills.add((*ip).aux0);
});
def_handler!(op_load_spill1, |ip, ctx, _rows, _out, _r0, r1| {
    r1 = *(*ctx).spills.add((*ip).aux0);
});
def_handler!(op_spill0, |ip, ctx, _rows, _out, r0, _r1| {
    *(*ctx).spills.add((*ip).aux0) = r0;
});
def_handler!(op_spill1, |ip, ctx, _rows, _out, _r0, r1| {
    *(*ctx).spills.add((*ip).aux0) = r1;
});
def_handler!(op_mov10, |_ip, _ctx, _rows, _out, r0, r1| {
    r1 = r0;
});
def_handler!(op_mov01, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = r1;
});
def_handler!(op_swap01, |_ip, _ctx, _rows, _out, r0, r1| {
    core::mem::swap(&mut r0, &mut r1);
});
def_handler!(op_const0, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = [c, c, c, c];
});
def_handler!(op_store_sink0, |ip, ctx, _rows, _out, r0, _r1| {
    *(*ctx).sinks.add((*ip).aux0) = r0;
});
def_handler!(op_store_sink1, |ip, ctx, _rows, _out, _r0, r1| {
    *(*ctx).sinks.add((*ip).aux0) = r1;
});
def_handler!(op_store_direct32_0, |_ip, _ctx, _rows, out, r0, _r1| {
    arch::store4_u32(out.cast(), r0);
});
def_handler!(op_store_direct64_0, |_ip, _ctx, _rows, out, r0, _r1| {
    arch::store4_i64(out.cast(), r0);
});

// --- Width-independent bitwise / LUT / IEEE-754 f64 handlers ---
def_handler!(op_and, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::and_v(x, y));
});
def_handler!(op_or, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::or_v(x, y));
});
def_handler!(op_xor, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::xor_v(x, y));
});
def_handler!(op_andi, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::and_v(x, c));
});
def_handler!(op_ori, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::or_v(x, c));
});
def_handler!(op_xori, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::xor_v(x, c));
});
def_handler!(op_nibble_lut, |ip, _ctx, _rows, _out, r0, _r1| {
    let tbl = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::nibble_lut(x, tbl));
});
def_handler!(op_byte_lut32, |ip, _ctx, _rows, _out, r0, _r1| {
    let (lo, hi) = ((*ip).vec_imm0, (*ip).vec_imm1);
    r0 = arch::map1(r0, |x| arch::byte_lut32(x, lo, hi));
});
def_handler!(op_bitselect, |ip, _ctx, _rows, _out, r0, r1| {
    let m = (*ip).vec_imm0;
    r0 = arch::map2(r0, r1, |x, y| arch::bitselect(x, y, m));
});
def_handler!(op_f64_add, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::f64_add(x, y));
});
def_handler!(op_f64_sub, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::f64_sub(x, y));
});
def_handler!(op_f64_mul, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::f64_mul(x, y));
});
def_handler!(op_f64_addi, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::f64_add(x, c));
});
def_handler!(op_f64_muli, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::f64_mul(x, c));
});
def_handler!(op_f64_divi, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::f64_div(x, c));
});
def_handler!(op_u64_to_f64, |_ip, _ctx, _rows, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::u64_to_f64(x));
});
def_handler!(op_i64_to_f64, |_ip, _ctx, _rows, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::i64_to_f64(x));
});

// --- 32-bit lane handlers ---
def_handler!(op_add32, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::add32(x, y));
});
def_handler!(op_sub32, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::sub32(x, y));
});
def_handler!(op_mul32, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::mul32(x, y));
});
def_handler!(op_shl32, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::shl32_v(x, y));
});
def_handler!(op_shr32, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::shr32_v(x, y));
});
def_handler!(op_addi32, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::add32(x, c));
});
def_handler!(op_muli32, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::mul32(x, c));
});
def_handler!(op_muladdi32, |ip, _ctx, _rows, _out, r0, _r1| {
    let (m, c) = ((*ip).vec_imm0, (*ip).vec_imm1);
    r0 = arch::map1(r0, |x| arch::add32(arch::mul32(x, m), c));
});
def_handler!(op_shli32, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::shl32_i(x, cnt));
});
def_handler!(op_shri32, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::shr32_i(x, cnt));
});
def_handler!(op_sari32, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::sar32_i(x, cnt));
});
def_handler!(op_ctz32, |_ip, _ctx, _rows, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::ctz32(x));
});
def_handler!(op_bswap32, |_ip, _ctx, _rows, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::bswap32(x));
});
def_handler!(op_select32_spill, |ip, ctx, _rows, _out, r0, r1| {
    let cond = *(*ctx).spills.add((*ip).aux0);
    r0 = arch::map3(cond, r0, r1, |c, x, y| arch::select32(c, x, y));
});
def_handler!(op_select_mask32_spill, |ip, ctx, _rows, _out, r0, r1| {
    let src = *(*ctx).spills.add((*ip).aux0);
    let m = (*ip).vec_imm0;
    r0 = arch::map3(src, r0, r1, |s, x, y| {
        arch::select32(arch::and_v(s, m), x, y)
    });
});
def_handler!(op_xor_shri32, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::xor_v(x, arch::shr32_i(x, cnt)));
});
def_handler!(op_xor_shl_andi32, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::xor_shl_and32(x, cnt, c));
});
def_handler!(op_mask_lowbit_xor32, |ip, _ctx, _rows, _out, r0, r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map2(r0, r1, |x, y| arch::mask_lowbit_xor32(x, y, c));
});
def_handler!(op_gather32, |ip, ctx, rows, _out, r0, _r1| {
    let buf_id = (*ip).aux0;
    let elem_log2 = (*ip).aux1 as u8;
    let offset = (*ip).imm_u64;
    let mut tmp = [0u32; T32];
    arch::store4_u32(tmp.as_mut_ptr(), r0);
    if buf_id < (*ctx).n_bufs {
        let buf = &*(*ctx).bufs.add(buf_id);
        let n = rows.min(T32);
        for x in &mut tmp[..n] {
            *x = buf.gather(*x as u64, offset, elem_log2) as u32;
        }
        tmp[n..].fill(0);
    } else {
        tmp.fill(0);
    }
    r0 = arch::load4_u32(tmp.as_ptr());
});
def_handler!(op_scan_add32, |ip, ctx, rows, _out, r0, _r1| {
    let st = (*ctx).scan_states.add((*ip).aux0);
    let mut acc = *st as u32;
    let mut tmp = [0u32; T32];
    arch::store4_u32(tmp.as_mut_ptr(), r0);
    for x in &mut tmp[..rows.min(T32)] {
        acc = acc.wrapping_add(*x);
        *x = acc;
    }
    *st = acc as i64;
    r0 = arch::load4_u32(tmp.as_ptr());
});
def_handler!(op_scan_xor32, |ip, ctx, rows, _out, r0, _r1| {
    let st = (*ctx).scan_states.add((*ip).aux0);
    let mut acc = *st as u32;
    let mut tmp = [0u32; T32];
    arch::store4_u32(tmp.as_mut_ptr(), r0);
    for x in &mut tmp[..rows.min(T32)] {
        acc ^= *x;
        *x = acc;
    }
    *st = acc as i64;
    r0 = arch::load4_u32(tmp.as_ptr());
});

// --- 64-bit lane handlers ---
def_handler!(op_add64, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::add64(x, y));
});
def_handler!(op_sub64, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::sub64(x, y));
});
def_handler!(op_mul64, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::mul64(x, y));
});
def_handler!(op_shl64, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::shl64_v(x, y));
});
def_handler!(op_shr64, |_ip, _ctx, _rows, _out, r0, r1| {
    r0 = arch::map2(r0, r1, |x, y| arch::shr64_v(x, y));
});
def_handler!(op_addi64, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::add64(x, c));
});
def_handler!(op_muli64, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::mul64(x, c));
});
def_handler!(op_muladdi64, |ip, _ctx, _rows, _out, r0, _r1| {
    let (m, c) = ((*ip).vec_imm0, (*ip).vec_imm1);
    r0 = arch::map1(r0, |x| arch::add64(arch::mul64(x, m), c));
});
def_handler!(op_mulhii64, |ip, _ctx, _rows, _out, r0, _r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::mulhi64(x, c));
});
def_handler!(op_shli64, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::shl64_i(x, cnt));
});
def_handler!(op_shri64, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::shr64_i(x, cnt));
});
def_handler!(op_sari64, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::sar64_i(x, cnt));
});
def_handler!(op_ctz64, |_ip, _ctx, _rows, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::ctz64(x));
});
def_handler!(op_bswap64, |_ip, _ctx, _rows, _out, r0, _r1| {
    r0 = arch::map1(r0, |x| arch::bswap64(x));
});
def_handler!(op_select64_spill, |ip, ctx, _rows, _out, r0, r1| {
    let cond = *(*ctx).spills.add((*ip).aux0);
    r0 = arch::map3(cond, r0, r1, |c, x, y| arch::select64(c, x, y));
});
def_handler!(op_select_mask64_spill, |ip, ctx, _rows, _out, r0, r1| {
    let src = *(*ctx).spills.add((*ip).aux0);
    let m = (*ip).vec_imm0;
    r0 = arch::map3(src, r0, r1, |s, x, y| {
        arch::select64(arch::and_v(s, m), x, y)
    });
});
def_handler!(op_xor_shri64, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    r0 = arch::map1(r0, |x| arch::xor_v(x, arch::shr64_i(x, cnt)));
});
def_handler!(op_xor_shl_andi64, |ip, _ctx, _rows, _out, r0, _r1| {
    let cnt = &raw const (*ip).imm_u64;
    let c = (*ip).vec_imm0;
    r0 = arch::map1(r0, |x| arch::xor_shl_and64(x, cnt, c));
});
def_handler!(op_mask_lowbit_xor64, |ip, _ctx, _rows, _out, r0, r1| {
    let c = (*ip).vec_imm0;
    r0 = arch::map2(r0, r1, |x, y| arch::mask_lowbit_xor64(x, y, c));
});
def_handler!(op_gather64, |ip, ctx, rows, _out, r0, _r1| {
    let buf_id = (*ip).aux0;
    let elem_log2 = (*ip).aux1 as u8;
    let offset = (*ip).imm_u64;
    let mut tmp = [0i64; T64];
    arch::store4_i64(tmp.as_mut_ptr(), r0);
    if buf_id < (*ctx).n_bufs {
        let buf = &*(*ctx).bufs.add(buf_id);
        let n = rows.min(T64);
        for x in &mut tmp[..n] {
            *x = buf.gather(*x as u64, offset, elem_log2);
        }
        tmp[n..].fill(0);
    } else {
        tmp.fill(0);
    }
    r0 = arch::load4_i64(tmp.as_ptr());
});
def_handler!(op_scan_add64, |ip, ctx, rows, _out, r0, _r1| {
    let st = (*ctx).scan_states.add((*ip).aux0);
    let mut acc = *st;
    let mut tmp = [0i64; T64];
    arch::store4_i64(tmp.as_mut_ptr(), r0);
    for x in &mut tmp[..rows.min(T64)] {
        acc = acc.wrapping_add(*x);
        *x = acc;
    }
    *st = acc;
    r0 = arch::load4_i64(tmp.as_ptr());
});
def_handler!(op_scan_xor64, |ip, ctx, rows, _out, r0, _r1| {
    let st = (*ctx).scan_states.add((*ip).aux0);
    let mut acc = *st;
    let mut tmp = [0i64; T64];
    arch::store4_i64(tmp.as_mut_ptr(), r0);
    for x in &mut tmp[..rows.min(T64)] {
        acc ^= *x;
        *x = acc;
    }
    *st = acc;
    r0 = arch::load4_i64(tmp.as_ptr());
});
def_handler!(op_scan_f64_add, |ip, ctx, rows, _out, r0, _r1| {
    let st = (*ctx).scan_states.add((*ip).aux0);
    let mut acc = f64::from_bits(*st as u64);
    if acc.is_nan() {
        acc = 0.0;
    }
    let mut tmp = [0i64; T64];
    arch::store4_i64(tmp.as_mut_ptr(), r0);
    for x in &mut tmp[..rows.min(T64)] {
        acc += f64::from_bits(*x as u64);
        *x = acc.to_bits() as i64;
    }
    *st = acc.to_bits() as i64;
    r0 = arch::load4_i64(tmp.as_ptr());
});

// ============================================================================
// Peephole IR & Register-Pinned Tile Compiler
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum VOp {
    Raw(Ins),
    Konst(i64),
    MulAddI { a: u32, mul: i64, add: i64 },
    BitSelect { a: u32, b: u32, mask: i64 },
    MaskLowBitXor { a: u32, b: u32, mask: i64 },
    XorShrI { a: u32, sh: u8 },
    XorShlAndI { a: u32, sh: u8, mask: i64 },
    SelectMaskI { src: u32, mask: i64, a: u32, b: u32 },
}

#[derive(Clone, Copy)]
enum SinkPlan {
    Direct32,
    Direct64,
    StagedFixed { widths: [u8; 8], count: usize },
    StagedGeneral,
}

#[derive(Clone, Copy)]
enum SinkSlot {
    Fixed { val_slot: usize, width: u8 },
    Lane { val_slot: usize, len_slot: usize },
}

/// Ready-to-run compiled form of a [`Program`].
pub struct CompiledProgram {
    width: Width,
    n_in: usize,
    nodes_native: Vec<Node>,
    /// Variant used when calling `eval_*` with `&[i64]` inputs on a `W32` program.
    nodes_i64_in: Option<Vec<Node>>,
    n_spills: usize,
    sink_slots: Vec<SinkSlot>,
    n_sink_tiles: usize,
    sink_plan: SinkPlan,
    row_bytes: usize,
    scan_seeds: Vec<i64>,
    /// Where each instruction constant sits, in the order of the program's instructions that
    /// carry one, and those instructions' operations.
    slots: Vec<Slot>,
    slot_ops: Vec<Op>,
}

/// Where the compiled program holds one instruction's constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    /// Folded into the program's shape: another value needs another compile.
    Fixed,
    /// The instruction is dead; its constant is nowhere.
    Unused,
    Seed(u32),
    /// In a node, by its index in the native and in the `i64`-input schedule.
    Node(u32, u32, Field),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Vector,
    /// The addend of a fused multiply-add.
    Addend,
    Scalar,
    /// The vector holds the instruction applied to this constant.
    Folded(i64),
}

/// `op` of a constant operand and its immediate: the instructions that fold.
fn fold(op: Op, prev: i64, imm: i64) -> i64 {
    match op {
        Op::OrI => prev | imm,
        Op::AddI => prev.wrapping_add(imm),
        _ => prev ^ imm,
    }
}

/// A library entry compiled for this thread, and which of its constants are not its own.
struct Bound {
    /// Keeps the entry's address, the cache key, from being reused.
    _entry: Arc<Program>,
    /// The entry's own constants, by slot.
    own: Vec<i64>,
    replaced: Vec<u32>,
    cp: CompiledProgram,
}

thread_local! {
    static BOUND: std::cell::RefCell<super::graph::FastMap<usize, Bound>> =
        std::cell::RefCell::new(Default::default());
    static COMPILED: std::cell::RefCell<super::graph::FastMap<Program, std::rc::Rc<CompiledProgram>>> =
        std::cell::RefCell::new(Default::default());
}

impl Bound {
    /// Sets the program's constants to the entry's own but for `replaced`; false, changing
    /// nothing, if one of those is part of the compiled shape.
    fn rebind(&mut self, replaced: &[(u32, i64)]) -> bool {
        let fixed = |&(slot, _): &(u32, i64)| !matches!(self.cp.slots.get(slot as usize), Some(s) if *s != Slot::Fixed);
        if replaced.iter().any(fixed) {
            return false;
        }
        for slot in self.replaced.drain(..) {
            self.cp.put(slot, self.own[slot as usize]);
        }
        for &(slot, imm) in replaced {
            self.cp.put(slot, imm);
            self.replaced.push(slot);
        }
        true
    }
}

impl CompiledProgram {
    /// `p` compiled, kept for the next block: blocks of one model run the same few programs.
    pub fn cached(p: &Program) -> std::rc::Rc<Self> {
        COMPILED.with(|c| {
            let mut c = c.borrow_mut();
            if let Some(cp) = c.get(p) {
                return cp.clone();
            }
            if c.len() >= 256 {
                c.clear();
            }
            let cp = std::rc::Rc::new(Self::compile(p));
            c.insert(p.clone(), cp.clone());
            cp
        })
    }

    /// Runs `f` on `code` compiled. A library entry is compiled once a thread, and a block that
    /// replaces some of its constants has them set in place: compiling is most of what a small
    /// block's program costs.
    pub fn with<R>(code: &Code, f: impl FnOnce(&Self) -> R) -> R {
        let mut f = Some(f);
        let ran = code.entry().and_then(|(entry, replaced)| {
            BOUND.with(|b| {
                let mut b = b.borrow_mut();
                let key = Arc::as_ptr(entry) as usize;
                if !b.contains_key(&key) {
                    if b.len() >= 1024 {
                        b.clear();
                    }
                    let own = entry.ops.iter().filter(|i| i.op.has_imm());
                    b.insert(
                        key,
                        Bound {
                            _entry: entry.clone(),
                            own: own.map(|i| i.imm).collect(),
                            replaced: Vec::new(),
                            cp: Self::compile(entry),
                        },
                    );
                }
                let bound = b.get_mut(&key).unwrap();
                bound
                    .rebind(replaced)
                    .then(|| (f.take().unwrap())(&bound.cp))
            })
        });
        match ran {
            Some(r) => r,
            None => (f.take().unwrap())(&Self::cached(code)),
        }
    }

    pub fn compile(p: &Program) -> Self {
        let n_in = p.n_in as usize;
        let bits = p.width.bits();
        let wmask = if p.width == Width::W32 {
            0xffff_ffffu64 as i64
        } else {
            -1i64
        };

        // Step 1: Algebraic peephole recognition over `p.ops` (preserving SSA ids `n_in + i`).
        let mut base_uses = vec![0u32; n_in + p.ops.len()];
        for ins in &p.ops {
            for &v in &[ins.a, ins.b, ins.c][..ins.op.arity()] {
                base_uses[v as usize] += 1;
            }
        }
        for &(v, len) in &p.sinks {
            base_uses[v as usize] += 1;
            if let Len::Lane(l) = len {
                base_uses[l as usize] += 1;
            }
        }

        let def_of = |v: u32| -> Option<&Ins> {
            (v as usize)
                .checked_sub(n_in)
                .and_then(|idx| p.ops.get(idx))
        };
        let mut pinned = vec![false; p.ops.len()];
        // The constant each folded instruction was applied to.
        let mut folded: Vec<Option<i64>> = vec![None; p.ops.len()];
        let mut vops: Vec<VOp> = Vec::with_capacity(p.ops.len());
        let mut const_val: Vec<Option<i64>> = vec![None; n_in + p.ops.len()];

        for (i, &ins) in p.ops.iter().enumerate() {
            let vid = n_in + i;
            let vop = match ins.op {
                Op::AndI if (ins.imm & wmask) == 0 => {
                    const_val[vid] = Some(0);
                    VOp::Konst(0)
                }
                Op::OrI | Op::AddI | Op::XorI if const_val[ins.a as usize].is_some() => {
                    let prev = const_val[ins.a as usize].unwrap();
                    let k = fold(ins.op, prev, ins.imm);
                    const_val[vid] = Some(k);
                    folded[i] = Some(prev);
                    VOp::Konst(k)
                }
                Op::AddI => {
                    if let Some(d) = def_of(ins.a)
                        && d.op == Op::MulI
                        && base_uses[ins.a as usize] == 1
                    {
                        VOp::MulAddI {
                            a: d.a,
                            mul: d.imm,
                            add: ins.imm,
                        }
                    } else {
                        VOp::Raw(ins)
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
                        VOp::BitSelect {
                            a: da.a,
                            b: db.a,
                            mask: da.imm,
                        }
                    } else {
                        VOp::Raw(ins)
                    }
                }
                Op::Xor => {
                    // Check `Xor(x, ShrI(x, s))`, `Xor(x, AndI(ShlI(x, s), m))`,
                    // and `Xor(a, AndI(SarI(ShlI(b, W-1), W-1), m))` in either operand order.
                    let try_fuse = |x: u32, y: u32| -> Option<VOp> {
                        let dy = def_of(y)?;
                        if base_uses[y as usize] != 1 {
                            return None;
                        }
                        if dy.op == Op::ShrI && dy.a == x {
                            return Some(VOp::XorShrI {
                                a: x,
                                sh: dy.imm as u8,
                            });
                        }
                        if dy.op == Op::AndI
                            && base_uses[dy.a as usize] == 1
                            && let Some(dsh) = def_of(dy.a)
                        {
                            if dsh.op == Op::ShlI && dsh.a == x {
                                return Some(VOp::XorShlAndI {
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
                                return Some(VOp::MaskLowBitXor {
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
                        .unwrap_or(VOp::Raw(ins))
                }
                Op::Select => {
                    if let Some(dc) = def_of(ins.c)
                        && dc.op == Op::AndI
                        && base_uses[ins.c as usize] == 1
                    {
                        VOp::SelectMaskI {
                            src: dc.a,
                            mask: dc.imm,
                            a: ins.a,
                            b: ins.b,
                        }
                    } else {
                        VOp::Raw(ins)
                    }
                }
                _ => VOp::Raw(ins),
            };
            // A fused operation is built from the constants of the instructions it replaces:
            // those constants are part of the program's shape now.
            let mut pin = |v: u32| {
                if let Some(d) = (v as usize).checked_sub(n_in) {
                    pinned[d] = true;
                }
            };
            let other = |x: u32| if ins.a == x { ins.b } else { ins.a };
            match vop {
                VOp::Raw(_) => {}
                VOp::Konst(_) => {
                    if folded[i].is_some() {
                        pin(ins.a)
                    }
                }
                VOp::MulAddI { .. } => pin(ins.a),
                VOp::BitSelect { .. } => {
                    pin(ins.a);
                    pin(ins.b);
                }
                VOp::SelectMaskI { .. } => pin(ins.c),
                VOp::XorShrI { a, .. } => pin(other(a)),
                // The mask and the shift under it; for the low-bit form, the shift under that too.
                VOp::XorShlAndI { a, .. } | VOp::MaskLowBitXor { a, .. } => {
                    let depth = if matches!(vop, VOp::XorShlAndI { .. }) {
                        2
                    } else {
                        3
                    };
                    let mut v = other(a);
                    for _ in 0..depth {
                        pin(v);
                        let Some(d) = def_of(v) else { break };
                        v = d.a;
                    }
                }
            }
            vops.push(vop);
        }

        // Step 2: Recompute exact liveness over reachable `vops` (dead fused intermediates get 0 uses!).
        let total_vals = n_in + vops.len();
        let mut uses = vec![0u32; total_vals];
        for &(v, len) in &p.sinks {
            uses[v as usize] += 1;
            if let Len::Lane(l) = len {
                uses[l as usize] += 1;
            }
        }
        for (i, vop) in vops.iter().enumerate().rev() {
            let vid = n_in + i;
            if uses[vid] == 0 && !matches!(vop, VOp::Raw(ins) if ins.op.is_scan()) {
                continue;
            }
            for dep in vop_deps(*vop) {
                uses[dep as usize] += 1;
            }
        }

        // Determine sink materialization plan.
        let direct_single = p.sinks.len() == 1
            && matches!(
                (p.width, p.sinks[0].1),
                (Width::W32, Len::Fixed(4)) | (Width::W64, Len::Fixed(8))
            );

        let mut sink_slots = Vec::with_capacity(p.sinks.len());
        let mut n_sink_tiles = 0usize;
        for &(_, len) in &p.sinks {
            let val_slot = n_sink_tiles;
            n_sink_tiles += 1;
            match len {
                Len::Fixed(width) => sink_slots.push(SinkSlot::Fixed { val_slot, width }),
                Len::Lane(_) => {
                    let len_slot = n_sink_tiles;
                    n_sink_tiles += 1;
                    sink_slots.push(SinkSlot::Lane { val_slot, len_slot });
                }
            }
        }

        let sink_plan = if direct_single {
            match p.width {
                Width::W32 => SinkPlan::Direct32,
                Width::W64 => SinkPlan::Direct64,
            }
        } else if !p.varlen() && p.sinks.len() <= 8 {
            let mut widths = [0u8; 8];
            for (i, &(_, len)) in p.sinks.iter().enumerate() {
                let Len::Fixed(w) = len else { unreachable!() };
                widths[i] = w;
            }
            SinkPlan::StagedFixed {
                widths,
                count: p.sinks.len(),
            }
        } else {
            SinkPlan::StagedGeneral
        };

        // Step 3: Schedule onto `(r0, r1)` + reusable `spills`.
        let (nodes_native, n_spills, scan_seeds, at) =
            schedule_program(p, &vops, &uses, &sink_slots, direct_single, false);
        let (nodes_i64_in, at_i64) = if p.width == Width::W32 && n_in > 0 {
            let s = schedule_program(p, &vops, &uses, &sink_slots, direct_single, true);
            (Some(s.0), s.3)
        } else {
            (None, at.clone())
        };

        // Which of the program's own instructions reach a sink, fusing aside.
        let mut live = vec![false; n_in + p.ops.len()];
        for &(v, len) in &p.sinks {
            live[v as usize] = true;
            if let Len::Lane(l) = len {
                live[l as usize] = true;
            }
        }
        for (i, ins) in p.ops.iter().enumerate().rev() {
            if live[n_in + i] || ins.op.is_scan() {
                for &v in &[ins.a, ins.b, ins.c][..ins.op.arity()] {
                    live[v as usize] = true;
                }
            }
        }
        let slots = (0..p.ops.len())
            .filter(|&i| p.ops[i].op.has_imm())
            .map(|i| {
                let op = p.ops[i].op;
                let fused_add = matches!(vops[i], VOp::MulAddI { .. });
                let own = fused_add || folded[i].is_some() || matches!(vops[i], VOp::Raw(_));
                if pinned[i] || !own {
                    Slot::Fixed
                } else if at[i] == u32::MAX {
                    // Live in the program yet never scheduled: something swallowed it.
                    if live[n_in + i] || op.is_scan() {
                        Slot::Fixed
                    } else {
                        Slot::Unused
                    }
                } else if op.is_scan() {
                    Slot::Seed(nodes_native[at[i] as usize].aux0 as u32)
                } else {
                    let field = match folded[i] {
                        Some(prev) => Field::Folded(prev),
                        None if fused_add => Field::Addend,
                        None if op.is_shift_imm() || op == Op::Gather => Field::Scalar,
                        None => Field::Vector,
                    };
                    Slot::Node(at[i], at_i64[i], field)
                }
            })
            .collect();

        Self {
            width: p.width,
            n_in,
            nodes_native,
            nodes_i64_in,
            n_spills,
            sink_slots,
            n_sink_tiles,
            sink_plan,
            row_bytes: p.row_bytes(),
            scan_seeds,
            slots,
            slot_ops: p.ops.iter().map(|i| i.op).filter(|o| o.has_imm()).collect(),
        }
    }

    /// Sets constant `slot` to `imm`, as compiling the program with it there would have.
    fn put(&mut self, slot: u32, imm: i64) {
        let (width, op) = (self.width, self.slot_ops[slot as usize]);
        let splat = |imm: i64| unsafe {
            match width {
                Width::W32 => arch::splat32(imm as u32),
                Width::W64 => arch::splat64(imm),
            }
        };
        match self.slots[slot as usize] {
            Slot::Fixed | Slot::Unused => {}
            Slot::Seed(s) => self.scan_seeds[s as usize] = imm,
            Slot::Node(at, at_i64, field) => {
                let set = |n: &mut Node| match field {
                    Field::Vector => n.vec_imm0 = splat(imm),
                    Field::Addend => n.vec_imm1 = splat(imm),
                    Field::Scalar => n.imm_u64 = imm as u64,
                    Field::Folded(prev) => n.vec_imm0 = splat(fold(op, prev, imm)),
                };
                set(&mut self.nodes_native[at as usize]);
                if let Some(nodes) = &mut self.nodes_i64_in {
                    set(&mut nodes[at_i64 as usize]);
                }
            }
        }
    }

    /// Which instruction constants can be set in place, for tests.
    #[cfg(test)]
    pub(crate) fn settable(&self) -> Vec<bool> {
        self.slots.iter().map(|s| *s != Slot::Fixed).collect()
    }

    #[inline]
    pub fn width(&self) -> Width {
        self.width
    }

    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Evaluate a single-8-byte-sink `W64` program directly into `out: &mut [i64]`.
    pub fn eval_i64(&self, inputs: &[&[i64]], bufs: &[TableBuf<'_>], out: &mut [i64]) {
        assert!(matches!(self.sink_plan, SinkPlan::Direct64));
        let out_bytes = unsafe {
            core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 8)
        };
        self.run_i64_inputs(inputs, bufs, out.len(), out_bytes, None);
    }

    /// Evaluate a single-4-byte-sink `W32` program directly into `out: &mut [u32]`.
    pub fn eval_u32(&self, inputs: &[&[u32]], bufs: &[TableBuf<'_>], out: &mut [u32]) {
        assert!(matches!(self.sink_plan, SinkPlan::Direct32));
        let rows = out.len();
        for col in inputs {
            assert!(col.len() >= rows);
        }
        let z = unsafe { arch::zero() };
        let mut spills = vec![[z; 4]; self.n_spills.max(1)];
        let mut sinks = vec![[z; 4]; 1];
        let mut scan_states = self.scan_seeds.clone();
        let mut in_ptrs = vec![core::ptr::null::<u8>(); self.n_in];
        let bufs_static: &[TableBuf<'static>] = unsafe { core::mem::transmute(bufs) };

        let mut ctx = ExecCtx {
            in_ptrs: in_ptrs.as_ptr(),
            spills: spills.as_mut_ptr(),
            sinks: sinks.as_mut_ptr(),
            bufs: bufs_static.as_ptr(),
            n_bufs: bufs_static.len(),
            scan_states: scan_states.as_mut_ptr(),
        };

        let full = (rows / T32) * T32;
        let mut row = 0usize;
        while row < full {
            for (k, col) in inputs.iter().enumerate() {
                in_ptrs[k] = unsafe { col.as_ptr().add(row).cast() };
            }
            unsafe {
                let start = self.nodes_native.as_ptr();
                ((*start).next)(
                    start,
                    &raw mut ctx,
                    T32,
                    out.as_mut_ptr().add(row).cast(),
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
            row += T32;
        }
        if row < rows {
            let rem = rows - row;
            let mut tail_in = vec![[0u32; T32]; self.n_in];
            for (k, col) in inputs.iter().enumerate() {
                tail_in[k][..rem].copy_from_slice(&col[row..rows]);
                in_ptrs[k] = tail_in[k].as_ptr().cast();
            }
            let mut tail_out = [0u32; T32];
            unsafe {
                let start = self.nodes_native.as_ptr();
                ((*start).next)(
                    start,
                    &raw mut ctx,
                    rem,
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
            out[row..rows].copy_from_slice(&tail_out[..rem]);
        }
    }

    /// Evaluate any fixed-width program into `out` (`rows * row_bytes` bytes).
    pub fn eval_fixed(
        &self,
        inputs: &[&[i64]],
        bufs: &[TableBuf<'_>],
        rows: usize,
        out: &mut [u8],
    ) {
        assert!(out.len() >= rows * self.row_bytes);
        self.run_i64_inputs(inputs, bufs, rows, out, None);
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
        assert!(ends.len() >= rows && text.len() >= rows * self.row_bytes);
        self.run_i64_inputs(inputs, bufs, rows, text, Some(ends))
    }

    /// Same as `eval_cells` where `out` begins with `rows * 4` bytes of `u32` end offsets
    /// followed by text bytes (matching `kernels::fmt_int` layout).
    pub fn eval_var_packed(
        &self,
        inputs: &[&[i64]],
        bufs: &[TableBuf<'_>],
        rows: usize,
        out: &mut [u8],
    ) -> usize {
        let (ends_bytes, text) = out.split_at_mut(rows * 4);
        let mut ends_aligned = vec![0u32; rows];
        let written = self.eval_cells(inputs, bufs, rows, &mut ends_aligned, text);
        let src_bytes =
            unsafe { core::slice::from_raw_parts(ends_aligned.as_ptr().cast::<u8>(), rows * 4) };
        ends_bytes.copy_from_slice(src_bytes);
        written
    }

    fn run_i64_inputs(
        &self,
        inputs: &[&[i64]],
        bufs: &[TableBuf<'_>],
        rows: usize,
        out: &mut [u8],
        mut ends: Option<&mut [u32]>,
    ) -> usize {
        assert_eq!(inputs.len(), self.n_in);
        for col in inputs {
            assert!(col.len() >= rows);
        }
        let tile_lanes = match self.width {
            Width::W32 => T32,
            Width::W64 => T64,
        };
        let nodes = if self.width == Width::W32 && self.n_in > 0 {
            self.nodes_i64_in.as_ref().unwrap()
        } else {
            &self.nodes_native
        };

        let z = unsafe { arch::zero() };
        let mut spills = vec![[z; 4]; self.n_spills.max(1)];
        let mut sinks = vec![[z; 4]; self.n_sink_tiles.max(1)];
        let mut scan_states = self.scan_seeds.clone();
        let mut in_ptrs = vec![core::ptr::null::<u8>(); self.n_in];
        let bufs_static: &[TableBuf<'static>] = unsafe { core::mem::transmute(bufs) };

        let mut ctx = ExecCtx {
            in_ptrs: in_ptrs.as_ptr(),
            spills: spills.as_mut_ptr(),
            sinks: sinks.as_mut_ptr(),
            bufs: bufs_static.as_ptr(),
            n_bufs: bufs_static.len(),
            scan_states: scan_states.as_mut_ptr(),
        };

        let full = (rows / tile_lanes) * tile_lanes;
        let mut row = 0usize;
        let mut byte_pos = 0usize;

        while row < full {
            for (k, col) in inputs.iter().enumerate() {
                in_ptrs[k] = unsafe { col.as_ptr().add(row).cast() };
            }
            let direct_ptr = unsafe { out.as_mut_ptr().add(row * self.row_bytes) };
            unsafe {
                let start = nodes.as_ptr();
                ((*start).next)(
                    start,
                    &raw mut ctx,
                    tile_lanes,
                    direct_ptr,
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
            byte_pos =
                self.emit_staged_tile(&sinks, tile_lanes, row, byte_pos, out, ends.as_deref_mut());
            row += tile_lanes;
        }

        if row < rows {
            let rem = rows - row;
            let mut tail_in = vec![vec![0i64; tile_lanes]; self.n_in];
            for (k, col) in inputs.iter().enumerate() {
                tail_in[k][..rem].copy_from_slice(&col[row..rows]);
                in_ptrs[k] = tail_in[k].as_ptr().cast();
            }
            let mut direct_tail = [0u8; 64 * 8];
            unsafe {
                let start = nodes.as_ptr();
                ((*start).next)(
                    start,
                    &raw mut ctx,
                    rem,
                    direct_tail.as_mut_ptr(),
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
            if matches!(self.sink_plan, SinkPlan::Direct32 | SinkPlan::Direct64) {
                let w = self.row_bytes;
                out[row * w..rows * w].copy_from_slice(&direct_tail[..rem * w]);
            }
            byte_pos = self.emit_staged_tile(&sinks, rem, row, byte_pos, out, ends);
        }
        byte_pos
    }

    #[inline(always)]
    fn emit_staged_tile(
        &self,
        sinks: &[Tile],
        n_rows: usize,
        base_row: usize,
        mut pos: usize,
        out: &mut [u8],
        ends: Option<&mut [u32]>,
    ) -> usize {
        match (self.sink_plan, ends) {
            (SinkPlan::Direct32 | SinkPlan::Direct64, None) => pos + n_rows * self.row_bytes,
            (SinkPlan::Direct32 | SinkPlan::Direct64, Some(ends)) => {
                let w = self.row_bytes;
                for r in 0..n_rows {
                    pos += w;
                    ends[base_row + r] = pos as u32;
                }
                pos
            }
            (SinkPlan::StagedFixed { widths, count }, None) => {
                let out_len = out.len();
                let out_ptr = out.as_mut_ptr();
                match self.width {
                    Width::W64 => {
                        if count == 3 && widths[0] == 8 && widths[1] == 8 && widths[2] == 4 {
                            // Ultra-fast 20-byte ISO-8601 path: 2x u64 + 1x u32 unaligned writes.
                            let s0: &[u64; T64] = unsafe { &*(sinks[0].as_ptr().cast()) };
                            let s1: &[u64; T64] = unsafe { &*(sinks[1].as_ptr().cast()) };
                            let s2: &[u64; T64] = unsafe { &*(sinks[2].as_ptr().cast()) };
                            for r in 0..n_rows {
                                unsafe {
                                    let p = out_ptr.add(pos);
                                    core::ptr::write_unaligned(p.cast::<u64>(), s0[r]);
                                    core::ptr::write_unaligned(p.add(8).cast::<u64>(), s1[r]);
                                    core::ptr::write_unaligned(
                                        p.add(16).cast::<u32>(),
                                        s2[r] as u32,
                                    );
                                }
                                pos += 20;
                            }
                            return pos;
                        }
                        for r in 0..n_rows {
                            if pos + self.row_bytes + 8 <= out_len {
                                for s in 0..count {
                                    let col: &[u64; T64] = unsafe { &*(sinks[s].as_ptr().cast()) };
                                    unsafe {
                                        core::ptr::write_unaligned(
                                            out_ptr.add(pos).cast::<u64>(),
                                            col[r],
                                        );
                                    }
                                    pos += widths[s] as usize;
                                }
                            } else {
                                for s in 0..count {
                                    let col: &[u64; T64] = unsafe { &*(sinks[s].as_ptr().cast()) };
                                    let w = widths[s] as usize;
                                    out[pos..pos + w].copy_from_slice(&col[r].to_le_bytes()[..w]);
                                    pos += w;
                                }
                            }
                        }
                    }
                    Width::W32 => {
                        for r in 0..n_rows {
                            for s in 0..count {
                                let col: &[u32; T32] = unsafe { &*(sinks[s].as_ptr().cast()) };
                                let w = widths[s] as usize;
                                if pos + 4 <= out_len {
                                    unsafe {
                                        core::ptr::write_unaligned(
                                            out_ptr.add(pos).cast::<u32>(),
                                            col[r],
                                        );
                                    }
                                } else {
                                    out[pos..pos + w].copy_from_slice(&col[r].to_le_bytes()[..w]);
                                }
                                pos += w;
                            }
                        }
                    }
                }
                pos
            }
            (_, mut ends_opt) => {
                let out_len = out.len();
                let out_ptr = out.as_mut_ptr();
                match self.width {
                    Width::W64 => {
                        if self.sink_slots.len() == 1
                            && let SinkSlot::Lane {
                                val_slot: 0,
                                len_slot: 1,
                            } = self.sink_slots[0]
                        {
                            // Single variable-width sink fast path (`fmt_small` etc.).
                            let vals: &[u64; T64] = unsafe { &*(sinks[0].as_ptr().cast()) };
                            let lens: &[u64; T64] = unsafe { &*(sinks[1].as_ptr().cast()) };
                            for r in 0..n_rows {
                                let w = (lens[r] as usize).min(8);
                                if pos + 8 <= out_len {
                                    unsafe {
                                        core::ptr::write_unaligned(
                                            out_ptr.add(pos).cast::<u64>(),
                                            vals[r],
                                        );
                                    }
                                } else {
                                    out[pos..pos + w].copy_from_slice(&vals[r].to_le_bytes()[..w]);
                                }
                                pos += w;
                                if let Some(e) = ends_opt.as_deref_mut() {
                                    e[base_row + r] = pos as u32;
                                }
                            }
                            return pos;
                        }
                        for r in 0..n_rows {
                            if pos + self.row_bytes + 8 <= out_len {
                                for slot in &self.sink_slots {
                                    let (vs, w) = match *slot {
                                        SinkSlot::Fixed { val_slot, width } => {
                                            (val_slot, width as usize)
                                        }
                                        SinkSlot::Lane { val_slot, len_slot } => {
                                            let ls: &[u64; T64] =
                                                unsafe { &*(sinks[len_slot].as_ptr().cast()) };
                                            (val_slot, (ls[r] as usize).min(8))
                                        }
                                    };
                                    let vs_arr: &[u64; T64] =
                                        unsafe { &*(sinks[vs].as_ptr().cast()) };
                                    unsafe {
                                        core::ptr::write_unaligned(
                                            out_ptr.add(pos).cast::<u64>(),
                                            vs_arr[r],
                                        );
                                    }
                                    pos += w;
                                }
                            } else {
                                for slot in &self.sink_slots {
                                    let (vs, w) = match *slot {
                                        SinkSlot::Fixed { val_slot, width } => {
                                            (val_slot, width as usize)
                                        }
                                        SinkSlot::Lane { val_slot, len_slot } => {
                                            let ls: &[u64; T64] =
                                                unsafe { &*(sinks[len_slot].as_ptr().cast()) };
                                            (val_slot, (ls[r] as usize).min(8))
                                        }
                                    };
                                    let vs_arr: &[u64; T64] =
                                        unsafe { &*(sinks[vs].as_ptr().cast()) };
                                    out[pos..pos + w]
                                        .copy_from_slice(&vs_arr[r].to_le_bytes()[..w]);
                                    pos += w;
                                }
                            }
                            if let Some(e) = ends_opt.as_deref_mut() {
                                e[base_row + r] = pos as u32;
                            }
                        }
                    }
                    Width::W32 => {
                        for r in 0..n_rows {
                            for slot in &self.sink_slots {
                                let (vs, w) = match *slot {
                                    SinkSlot::Fixed { val_slot, width } => {
                                        (val_slot, width as usize)
                                    }
                                    SinkSlot::Lane { val_slot, len_slot } => {
                                        let ls: &[u32; T32] =
                                            unsafe { &*(sinks[len_slot].as_ptr().cast()) };
                                        (val_slot, (ls[r] as usize).min(4))
                                    }
                                };
                                let vs_arr: &[u32; T32] = unsafe { &*(sinks[vs].as_ptr().cast()) };
                                if pos + 4 <= out_len {
                                    unsafe {
                                        core::ptr::write_unaligned(
                                            out_ptr.add(pos).cast::<u32>(),
                                            vs_arr[r],
                                        );
                                    }
                                } else {
                                    out[pos..pos + w]
                                        .copy_from_slice(&vs_arr[r].to_le_bytes()[..w]);
                                }
                                pos += w;
                            }
                            if let Some(e) = ends_opt.as_deref_mut() {
                                e[base_row + r] = pos as u32;
                            }
                        }
                    }
                }
                pos
            }
        }
    }
}

fn vop_deps(vop: VOp) -> impl Iterator<Item = u32> {
    let (arr, n) = match vop {
        VOp::Konst(_) => ([0, 0, 0], 0),
        VOp::MulAddI { a, .. } | VOp::XorShrI { a, .. } | VOp::XorShlAndI { a, .. } => {
            ([a, 0, 0], 1)
        }
        VOp::BitSelect { a, b, .. } | VOp::MaskLowBitXor { a, b, .. } => ([a, b, 0], 2),
        VOp::SelectMaskI { src, a, b, .. } => ([a, b, src], 3),
        VOp::Raw(ins) => ([ins.a, ins.b, ins.c], ins.op.arity()),
    };
    arr.into_iter().take(n)
}

fn schedule_program(
    p: &Program,
    vops: &[VOp],
    initial_uses: &[u32],
    sink_slots: &[SinkSlot],
    direct_single: bool,
    i64_in_for_w32: bool,
) -> (Vec<Node>, usize, Vec<i64>, Vec<u32>) {
    let n_in = p.n_in as usize;
    let is32 = p.width == Width::W32;
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
    let node = |next: Handler, aux0: usize, aux1: usize, imm_u64: u64, v0: V, v1: V| Node {
        vec_imm0: v0,
        vec_imm1: v1,
        next,
        aux0,
        aux1,
        imm_u64,
    };

    let mut nodes: Vec<Node> = Vec::with_capacity(vops.len() * 2 + p.sinks.len() + 2);
    let mut rem_uses = initial_uses.to_vec();
    let mut reg: [Option<usize>; 2] = [None, None];
    let mut spill_of: Vec<Option<usize>> = vec![None; n_in + vops.len()];
    let mut free_spills: Vec<usize> = Vec::new();
    let mut n_spills = 0usize;
    let mut scan_seeds: Vec<i64> = Vec::new();
    // The node each instruction's constant went into, where one did.
    let mut imm_at = vec![u32::MAX; vops.len()];

    let load_in_h = |r: usize| -> Handler {
        match (is32, i64_in_for_w32, r) {
            (true, false, 0) => op_load_in32_0,
            (true, false, _) => op_load_in32_1,
            (true, true, 0) => op_load_in64_as_u32_0,
            (true, true, _) => op_load_in64_as_u32_1,
            (false, _, 0) => op_load_in64_0,
            (false, _, _) => op_load_in64_1,
        }
    };

    let alloc_spill = |free_spills: &mut Vec<usize>, n_spills: &mut usize| -> usize {
        free_spills.pop().unwrap_or_else(|| {
            let s = *n_spills;
            *n_spills += 1;
            s
        })
    };

    let maybe_retire = |v: usize,
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

    // Spill `reg[r]` to L1d if it holds a live computed value not yet backed by a spill slot.
    let save_reg = |r: usize,
                    nodes: &mut Vec<Node>,
                    rem_uses: &[u32],
                    reg: &mut [Option<usize>; 2],
                    spill_of: &mut [Option<usize>],
                    free_spills: &mut Vec<usize>,
                    n_spills: &mut usize| {
        if let Some(v) = reg[r] {
            if rem_uses[v] > 0 && v >= n_in && spill_of[v].is_none() {
                // If the other accumulator already holds `v`, no memory spill is needed.
                if reg[1 - r] != Some(v) {
                    let s = alloc_spill(free_spills, n_spills);
                    let h = if r == 0 { op_spill0 } else { op_spill1 };
                    nodes.push(node(h, s, 0, 0, z, z));
                    spill_of[v] = Some(s);
                }
            }
            reg[r] = None;
        }
    };

    // Load `v` into `reg[r]` (clobbering `reg[r]` after saving if live). Does NOT decrement `rem_uses[v]`.
    let load_into = |r: usize,
                     v: usize,
                     nodes: &mut Vec<Node>,
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
            let h = if r == 0 { op_mov01 } else { op_mov10 };
            nodes.push(node(h, 0, 0, 0, z, z));
        } else if let Some(s) = spill_of[v] {
            let h = if r == 0 {
                op_load_spill0
            } else {
                op_load_spill1
            };
            nodes.push(node(h, s, 0, 0, z, z));
        } else {
            debug_assert!(v < n_in, "value {v} neither in reg nor spilled");
            nodes.push(node(load_in_h(r), v, 0, 0, z, z));
        }
        reg[r] = Some(v);
    };

    // Ensure `v` has a spill slot (loading from input if `v < n_in`) and return that spill slot.
    let ensure_in_spill = |v: usize,
                           nodes: &mut Vec<Node>,
                           rem_uses: &[u32],
                           reg: &mut [Option<usize>; 2],
                           spill_of: &mut [Option<usize>],
                           free_spills: &mut Vec<usize>,
                           n_spills: &mut usize|
     -> usize {
        if let Some(s) = spill_of[v] {
            return s;
        }
        let s = alloc_spill(free_spills, n_spills);
        if reg[0] == Some(v) {
            nodes.push(node(op_spill0, s, 0, 0, z, z));
        } else if reg[1] == Some(v) {
            nodes.push(node(op_spill1, s, 0, 0, z, z));
        } else {
            debug_assert!(v < n_in);
            save_reg(0, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
            nodes.push(node(load_in_h(0), v, 0, 0, z, z));
            nodes.push(node(op_spill0, s, 0, 0, z, z));
            reg[0] = Some(v);
        }
        spill_of[v] = Some(s);
        s
    };

    for (i, &vop) in vops.iter().enumerate() {
        let vid = n_in + i;
        let is_scan = matches!(vop, VOp::Raw(ins) if ins.op.is_scan());
        if rem_uses[vid] == 0 && !is_scan {
            continue;
        }

        match vop {
            VOp::Konst(k) => {
                save_reg(
                    0,
                    &mut nodes,
                    &rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                imm_at[i] = nodes.len() as u32;
                nodes.push(node(op_const0, 0, 0, 0, splat(k), z));
                reg[0] = Some(vid);
            }
            VOp::MulAddI { a, mul, add } => {
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
                // `r0` is about to be overwritten; if `a` still has future uses, preserve it in `r1` (if dead) or spill.
                if rem_uses[a] > 0 && a >= n_in && spill_of[a].is_none() && reg[1] != Some(a) {
                    if reg[1].is_none_or(|x| rem_uses[x] == 0) {
                        nodes.push(node(op_mov10, 0, 0, 0, z, z));
                        reg[1] = Some(a);
                    } else {
                        save_reg(
                            0,
                            &mut nodes,
                            &rem_uses,
                            &mut reg,
                            &mut spill_of,
                            &mut free_spills,
                            &mut n_spills,
                        );
                    }
                }
                maybe_retire(a, &rem_uses, &mut reg, &mut spill_of, &mut free_spills);
                let h = if is32 { op_muladdi32 } else { op_muladdi64 };
                imm_at[i] = nodes.len() as u32;
                nodes.push(node(h, 0, 0, 0, splat(mul), splat(add)));
                reg[0] = Some(vid);
            }
            VOp::XorShrI { a, sh } => {
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
                if rem_uses[a] > 0 && a >= n_in && spill_of[a].is_none() && reg[1] != Some(a) {
                    if reg[1].is_none_or(|x| rem_uses[x] == 0) {
                        nodes.push(node(op_mov10, 0, 0, 0, z, z));
                        reg[1] = Some(a);
                    } else {
                        save_reg(
                            0,
                            &mut nodes,
                            &rem_uses,
                            &mut reg,
                            &mut spill_of,
                            &mut free_spills,
                            &mut n_spills,
                        );
                    }
                }
                maybe_retire(a, &rem_uses, &mut reg, &mut spill_of, &mut free_spills);
                let h = if is32 { op_xor_shri32 } else { op_xor_shri64 };
                nodes.push(node(h, 0, 0, sh as u64, z, z));
                reg[0] = Some(vid);
            }
            VOp::XorShlAndI { a, sh, mask } => {
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
                if rem_uses[a] > 0 && a >= n_in && spill_of[a].is_none() && reg[1] != Some(a) {
                    if reg[1].is_none_or(|x| rem_uses[x] == 0) {
                        nodes.push(node(op_mov10, 0, 0, 0, z, z));
                        reg[1] = Some(a);
                    } else {
                        save_reg(
                            0,
                            &mut nodes,
                            &rem_uses,
                            &mut reg,
                            &mut spill_of,
                            &mut free_spills,
                            &mut n_spills,
                        );
                    }
                }
                maybe_retire(a, &rem_uses, &mut reg, &mut spill_of, &mut free_spills);
                let h = if is32 {
                    op_xor_shl_andi32
                } else {
                    op_xor_shl_andi64
                };
                nodes.push(node(h, 0, 0, sh as u64, splat(mask), z));
                reg[0] = Some(vid);
            }
            VOp::BitSelect { a, b, mask } => {
                prep_binary(
                    a as usize,
                    b as usize,
                    false,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                    &load_into,
                    &save_reg,
                    &maybe_retire,
                    n_in,
                    z,
                );
                nodes.push(node(op_bitselect, 0, 0, 0, splat(mask), z));
                reg[0] = Some(vid);
            }
            VOp::MaskLowBitXor { a, b, mask } => {
                prep_binary(
                    a as usize,
                    b as usize,
                    false,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                    &load_into,
                    &save_reg,
                    &maybe_retire,
                    n_in,
                    z,
                );
                let h = if is32 {
                    op_mask_lowbit_xor32
                } else {
                    op_mask_lowbit_xor64
                };
                nodes.push(node(h, 0, 0, 0, splat(mask), z));
                reg[0] = Some(vid);
            }
            VOp::SelectMaskI { src, mask, a, b } => {
                let cond_slot = ensure_in_spill(
                    src as usize,
                    &mut nodes,
                    &rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                prep_binary(
                    a as usize,
                    b as usize,
                    false,
                    &mut nodes,
                    &mut rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                    &load_into,
                    &save_reg,
                    &maybe_retire,
                    n_in,
                    z,
                );
                let h = if is32 {
                    op_select_mask32_spill
                } else {
                    op_select_mask64_spill
                };
                nodes.push(node(h, cond_slot, 0, 0, splat(mask), z));
                rem_uses[src as usize] -= 1;
                maybe_retire(
                    src as usize,
                    &rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                );
                reg[0] = Some(vid);
            }
            VOp::Raw(ins) => match ins.op.arity() {
                3 => {
                    // `Select(cond = ins.c, a = ins.a, b = ins.b)`
                    let cond_slot = ensure_in_spill(
                        ins.c as usize,
                        &mut nodes,
                        &rem_uses,
                        &mut reg,
                        &mut spill_of,
                        &mut free_spills,
                        &mut n_spills,
                    );
                    prep_binary(
                        ins.a as usize,
                        ins.b as usize,
                        false,
                        &mut nodes,
                        &mut rem_uses,
                        &mut reg,
                        &mut spill_of,
                        &mut free_spills,
                        &mut n_spills,
                        &load_into,
                        &save_reg,
                        &maybe_retire,
                        n_in,
                        z,
                    );
                    let h = if is32 {
                        op_select32_spill
                    } else {
                        op_select64_spill
                    };
                    nodes.push(node(h, cond_slot, 0, 0, z, z));
                    rem_uses[ins.c as usize] -= 1;
                    maybe_retire(
                        ins.c as usize,
                        &rem_uses,
                        &mut reg,
                        &mut spill_of,
                        &mut free_spills,
                    );
                    reg[0] = Some(vid);
                }
                2 => {
                    let commutative = matches!(
                        ins.op,
                        Op::Add | Op::Mul | Op::And | Op::Or | Op::Xor | Op::F64Add | Op::F64Mul
                    );
                    prep_binary(
                        ins.a as usize,
                        ins.b as usize,
                        commutative,
                        &mut nodes,
                        &mut rem_uses,
                        &mut reg,
                        &mut spill_of,
                        &mut free_spills,
                        &mut n_spills,
                        &load_into,
                        &save_reg,
                        &maybe_retire,
                        n_in,
                        z,
                    );
                    let h: Handler = match (is32, ins.op) {
                        (true, Op::Add) => op_add32,
                        (true, Op::Sub) => op_sub32,
                        (true, Op::Mul) => op_mul32,
                        (true, Op::Shl) => op_shl32,
                        (true, Op::Shr) => op_shr32,
                        (false, Op::Add) => op_add64,
                        (false, Op::Sub) => op_sub64,
                        (false, Op::Mul) => op_mul64,
                        (false, Op::Shl) => op_shl64,
                        (false, Op::Shr) => op_shr64,
                        (_, Op::And) => op_and,
                        (_, Op::Or) => op_or,
                        (_, Op::Xor) => op_xor,
                        (_, Op::F64Add) => op_f64_add,
                        (_, Op::F64Sub) => op_f64_sub,
                        (_, Op::F64Mul) => op_f64_mul,
                        _ => unreachable!(),
                    };
                    nodes.push(node(h, 0, 0, 0, z, z));
                    reg[0] = Some(vid);
                }
                _ => {
                    let a = ins.a as usize;
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
                    if rem_uses[a] > 0 && a >= n_in && spill_of[a].is_none() && reg[1] != Some(a) {
                        if reg[1].is_none_or(|x| rem_uses[x] == 0) {
                            nodes.push(node(op_mov10, 0, 0, 0, z, z));
                            reg[1] = Some(a);
                        } else {
                            save_reg(
                                0,
                                &mut nodes,
                                &rem_uses,
                                &mut reg,
                                &mut spill_of,
                                &mut free_spills,
                                &mut n_spills,
                            );
                        }
                    }
                    maybe_retire(a, &rem_uses, &mut reg, &mut spill_of, &mut free_spills);

                    let n = match (is32, ins.op) {
                        (true, Op::AddI) => node(op_addi32, 0, 0, 0, splat(ins.imm), z),
                        (false, Op::AddI) => node(op_addi64, 0, 0, 0, splat(ins.imm), z),
                        (true, Op::MulI) => node(op_muli32, 0, 0, 0, splat(ins.imm), z),
                        (false, Op::MulI) => node(op_muli64, 0, 0, 0, splat(ins.imm), z),
                        (_, Op::AndI) => node(op_andi, 0, 0, 0, splat(ins.imm), z),
                        (_, Op::OrI) => node(op_ori, 0, 0, 0, splat(ins.imm), z),
                        (_, Op::XorI) => node(op_xori, 0, 0, 0, splat(ins.imm), z),
                        (true, Op::ShlI) => node(op_shli32, 0, 0, ins.imm as u64, z, z),
                        (false, Op::ShlI) => node(op_shli64, 0, 0, ins.imm as u64, z, z),
                        (true, Op::ShrI) => node(op_shri32, 0, 0, ins.imm as u64, z, z),
                        (false, Op::ShrI) => node(op_shri64, 0, 0, ins.imm as u64, z, z),
                        (true, Op::SarI) => node(op_sari32, 0, 0, ins.imm as u64, z, z),
                        (false, Op::SarI) => node(op_sari64, 0, 0, ins.imm as u64, z, z),
                        (false, Op::MulHiI) => node(op_mulhii64, 0, 0, 0, splat(ins.imm), z),
                        (_, Op::F64AddI) => node(op_f64_addi, 0, 0, 0, splat(ins.imm), z),
                        (_, Op::F64MulI) => node(op_f64_muli, 0, 0, 0, splat(ins.imm), z),
                        (_, Op::F64DivI) => node(op_f64_divi, 0, 0, 0, splat(ins.imm), z),
                        (true, Op::Ctz) => node(op_ctz32, 0, 0, 0, z, z),
                        (false, Op::Ctz) => node(op_ctz64, 0, 0, 0, z, z),
                        (true, Op::Bswap) => node(op_bswap32, 0, 0, 0, z, z),
                        (false, Op::Bswap) => node(op_bswap64, 0, 0, 0, z, z),
                        (_, Op::U64ToF64) => node(op_u64_to_f64, 0, 0, 0, z, z),
                        (_, Op::I64ToF64) => node(op_i64_to_f64, 0, 0, 0, z, z),
                        (_, Op::NibbleLut) => {
                            let tbl = unsafe { arch::broadcast_lut16(&p.luts[ins.aux as usize]) };
                            node(op_nibble_lut, 0, 0, 0, tbl, z)
                        }
                        (_, Op::ByteLut32) => {
                            let (lo, hi) = unsafe {
                                arch::broadcast_lut32(
                                    &p.luts[ins.aux as usize],
                                    &p.luts[ins.aux as usize + 1],
                                )
                            };
                            node(op_byte_lut32, 0, 0, 0, lo, hi)
                        }
                        (true, Op::Gather) => node(
                            op_gather32,
                            (ins.aux & 0x3f) as usize,
                            (ins.aux >> 6) as usize,
                            ins.imm as u64,
                            z,
                            z,
                        ),
                        (false, Op::Gather) => node(
                            op_gather64,
                            (ins.aux & 0x3f) as usize,
                            (ins.aux >> 6) as usize,
                            ins.imm as u64,
                            z,
                            z,
                        ),
                        (true, Op::ScanAdd) => {
                            let sid = scan_seeds.len();
                            scan_seeds.push(ins.imm);
                            node(op_scan_add32, sid, 0, 0, z, z)
                        }
                        (false, Op::ScanAdd) => {
                            let sid = scan_seeds.len();
                            scan_seeds.push(ins.imm);
                            node(op_scan_add64, sid, 0, 0, z, z)
                        }
                        (true, Op::ScanXor) => {
                            let sid = scan_seeds.len();
                            scan_seeds.push(ins.imm);
                            node(op_scan_xor32, sid, 0, 0, z, z)
                        }
                        (false, Op::ScanXor) => {
                            let sid = scan_seeds.len();
                            scan_seeds.push(ins.imm);
                            node(op_scan_xor64, sid, 0, 0, z, z)
                        }
                        (false, Op::ScanF64Add) => {
                            let sid = scan_seeds.len();
                            scan_seeds.push(ins.imm);
                            node(op_scan_f64_add, sid, 0, 0, z, z)
                        }
                        _ => unreachable!(),
                    };
                    if ins.op.has_imm() {
                        imm_at[i] = nodes.len() as u32;
                    }
                    nodes.push(n);
                    reg[0] = Some(vid);
                }
            },
        }
    }

    // Step 4: Emit sink stores + `op_end_tile`.
    if direct_single {
        let sv = p.sinks[0].0 as usize;
        load_into(
            0,
            sv,
            &mut nodes,
            &rem_uses,
            &mut reg,
            &mut spill_of,
            &mut free_spills,
            &mut n_spills,
        );
        rem_uses[sv] -= 1;
        let h = if is32 {
            op_store_direct32_0
        } else {
            op_store_direct64_0
        };
        nodes.push(node(h, 0, 0, 0, z, z));
    } else {
        let mut store_val = |v: usize, sink_tile_slot: usize| {
            if reg[0] == Some(v) {
                nodes.push(node(op_store_sink0, sink_tile_slot, 0, 0, z, z));
            } else if reg[1] == Some(v) {
                nodes.push(node(op_store_sink1, sink_tile_slot, 0, 0, z, z));
            } else {
                load_into(
                    0,
                    v,
                    &mut nodes,
                    &rem_uses,
                    &mut reg,
                    &mut spill_of,
                    &mut free_spills,
                    &mut n_spills,
                );
                nodes.push(node(op_store_sink0, sink_tile_slot, 0, 0, z, z));
            }
            rem_uses[v] -= 1;
            maybe_retire(v, &rem_uses, &mut reg, &mut spill_of, &mut free_spills);
        };
        for (&(v, len), slot) in p.sinks.iter().zip(sink_slots) {
            match *slot {
                SinkSlot::Fixed { val_slot, .. } => store_val(v as usize, val_slot),
                SinkSlot::Lane { val_slot, len_slot } => {
                    let Len::Lane(l) = len else { unreachable!() };
                    store_val(v as usize, val_slot);
                    store_val(l as usize, len_slot);
                }
            }
        }
    }
    nodes.push(node(op_end_tile, 0, 0, 0, z, z));

    (nodes, n_spills, scan_seeds, imm_at)
}

#[allow(clippy::too_many_arguments)]
fn prep_binary<FLoad, FSave, FRetire>(
    mut a: usize,
    mut b: usize,
    commutative: bool,
    nodes: &mut Vec<Node>,
    rem_uses: &mut [u32],
    reg: &mut [Option<usize>; 2],
    spill_of: &mut [Option<usize>],
    free_spills: &mut Vec<usize>,
    n_spills: &mut usize,
    load_into: &FLoad,
    save_reg: &FSave,
    maybe_retire: &FRetire,
    n_in: usize,
    z: V,
) where
    FLoad: Fn(
        usize,
        usize,
        &mut Vec<Node>,
        &[u32],
        &mut [Option<usize>; 2],
        &mut [Option<usize>],
        &mut Vec<usize>,
        &mut usize,
    ),
    FSave: Fn(
        usize,
        &mut Vec<Node>,
        &[u32],
        &mut [Option<usize>; 2],
        &mut [Option<usize>],
        &mut Vec<usize>,
        &mut usize,
    ),
    FRetire: Fn(usize, &[u32], &mut [Option<usize>; 2], &mut [Option<usize>], &mut Vec<usize>),
{
    let mk = |next: Handler| Node {
        vec_imm0: z,
        vec_imm1: z,
        next,
        aux0: 0,
        aux1: 0,
        imm_u64: 0,
    };

    if a == b {
        load_into(0, a, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
        save_reg(1, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
        nodes.push(mk(op_mov10));
        reg[1] = Some(a);
        rem_uses[a] -= 2;
        // `r1` STILL holds `a` after `r0 = op(r0, r1)`!
        maybe_retire(a, rem_uses, reg, spill_of, free_spills);
        return;
    }

    if commutative && (reg[0] == Some(b) || reg[1] == Some(a)) {
        core::mem::swap(&mut a, &mut b);
    }

    if reg[0] == Some(b) && reg[1] == Some(a) {
        nodes.push(mk(op_swap01));
        reg.swap(0, 1);
    } else {
        if reg[0] == Some(b) {
            // Move `b` out of `r0` into `r1` before loading `a` into `r0`.
            save_reg(1, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
            nodes.push(mk(op_mov10));
            reg[1] = Some(b);
            reg[0] = None;
        }
        load_into(0, a, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
        load_into(1, b, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
    }

    rem_uses[a] -= 1;
    rem_uses[b] -= 1;
    // `r0` (`a`) will be overwritten by the result; `r1` (`b`) survives in `reg[1]`!
    if rem_uses[a] > 0 && a >= n_in && spill_of[a].is_none() {
        save_reg(0, nodes, rem_uses, reg, spill_of, free_spills, n_spills);
    }
    maybe_retire(a, rem_uses, reg, spill_of, free_spills);
    maybe_retire(b, rem_uses, reg, spill_of, free_spills);
}

/// Lower a linear numeric SSA chain to one loop. The operations and constants still come
/// entirely from the payload; short columns avoid a handler transition for each arithmetic step.
pub(crate) fn eval_numeric_chain(p: &Program, input: &[i64], out: &mut [i64]) -> bool {
    if p.width != Width::W64
        || p.n_in != 1
        || p.n_bufs != 0
        || !p.luts.is_empty()
        || p.sinks != [(p.ops.len() as u32, Len::Fixed(8))]
        || input.len() != out.len()
    {
        return false;
    }
    if p.ops.iter().enumerate().any(|(i, op)| op.a != i as u32) {
        return false;
    }
    let mut ops = p.ops.iter().peekable();
    let scan = if ops.peek().is_some_and(|i| i.op == Op::ScanAdd) {
        Some(ops.next().unwrap().imm)
    } else {
        None
    };
    let shift = if ops.peek().is_some_and(|i| i.op == Op::ShlI) {
        ops.next().unwrap().imm as u32
    } else {
        0
    };
    let factor = if ops.peek().is_some_and(|i| i.op == Op::MulI) {
        ops.next().unwrap().imm
    } else {
        1
    };
    let add = if ops.peek().is_some_and(|i| i.op == Op::AddI) {
        ops.next().unwrap().imm
    } else {
        0
    };
    let float = ops.peek().is_some_and(|i| i.op == Op::I64ToF64);
    if float {
        ops.next();
    }
    let scale = if float
        && ops
            .peek()
            .is_some_and(|i| matches!(i.op, Op::F64DivI | Op::F64MulI))
    {
        ops.next().copied()
    } else {
        None
    };
    if ops.next().is_some() {
        return false;
    }
    fn run(input: &[i64], out: &mut [i64], scan: Option<i64>, f: impl Fn(i64) -> i64) {
        if let Some(mut accumulator) = scan {
            for (&v, output) in input.iter().zip(out) {
                accumulator = accumulator.wrapping_add(v);
                *output = f(accumulator);
            }
        } else {
            for (&v, output) in input.iter().zip(out) {
                *output = f(v);
            }
        }
    }
    let integer = |v: i64| v.wrapping_shl(shift).wrapping_mul(factor).wrapping_add(add);
    match (float, scale) {
        (false, _) => run(input, out, scan, integer),
        (true, None) => run(input, out, scan, |v| (integer(v) as f64).to_bits() as i64),
        (true, Some(i)) if i.op == Op::F64DivI => {
            let divisor = f64::from_bits(i.imm as u64);
            run(input, out, scan, |v| {
                (integer(v) as f64 / divisor).to_bits() as i64
            });
        }
        (true, Some(i)) => {
            let multiplier = f64::from_bits(i.imm as u64);
            run(input, out, scan, |v| {
                (integer(v) as f64 * multiplier).to_bits() as i64
            });
        }
    }
    true
}
