//! Hardware-vector tile primitives for the Direct-Threaded SIMD engine.
//!
//! Three compile-time tiers keep every vector value inside physical SIMD registers across
//! tail-called handlers:
//! - x86-64 + `avx512f`: `V = __m512i` (4-vector tile = 64 `u32` lanes or 32 `i64` lanes)
//! - x86-64 + `avx2`:    `V = __m256i` (4-vector tile = 32 `u32` lanes or 16 `i64` lanes)
//! - x86-64 baseline (`sse2`), `aarch64` (`neon`), or other targets: 128-bit `V`
//!   (4-vector tile = 16 `u32` lanes or 8 `i64` lanes)

//!
//! Every function here is `unsafe` for one of two reasons: it runs vector instructions of its
//! tier, which the `cfg` that selects the tier guarantees the target has, or it reads or
//! writes a whole vector through a raw pointer, which the caller must keep in bounds.

#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

// ============================================================================
// Tier 1: x86-64 AVX-512 (`__m512i`)
// ============================================================================
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod imp {
    use core::arch::x86_64::*;

    pub type V = __m512i;
    pub const VEC_BYTES: usize = 64;
    pub const LANES32: usize = 16;
    pub const LANES64: usize = 8;

    #[inline(always)]
    pub unsafe fn zero() -> V {
        _mm512_setzero_si512()
    }
    #[inline(always)]
    pub unsafe fn splat32(x: u32) -> V {
        _mm512_set1_epi32(x as i32)
    }
    #[inline(always)]
    pub unsafe fn splat64(x: i64) -> V {
        _mm512_set1_epi64(x)
    }
    #[inline(always)]
    pub unsafe fn broadcast_lut16(lut: &[u8; 16]) -> V {
        let x = _mm_loadu_si128(lut.as_ptr().cast());
        _mm512_broadcast_i32x4(x)
    }
    #[inline(always)]
    pub unsafe fn broadcast_lut32(lo: &[u8; 16], hi: &[u8; 16]) -> (V, V) {
        #[cfg(target_feature = "avx512vbmi")]
        {
            let mut b = [0u8; 64];
            b[..16].copy_from_slice(lo);
            b[16..32].copy_from_slice(hi);
            b[32..48].copy_from_slice(lo);
            b[48..64].copy_from_slice(hi);
            let v = _mm512_loadu_si512(b.as_ptr().cast());
            (v, v)
        }
        #[cfg(not(target_feature = "avx512vbmi"))]
        {
            (broadcast_lut16(lo), broadcast_lut16(hi))
        }
    }

    #[inline(always)]
    pub unsafe fn load1_u32(p: *const u32) -> V {
        _mm512_loadu_si512(p.cast())
    }
    #[inline(always)]
    pub unsafe fn store1_u32(p: *mut u32, v: V) {
        _mm512_storeu_si512(p.cast(), v);
    }
    #[inline(always)]
    pub unsafe fn load1_i64(p: *const i64) -> V {
        _mm512_loadu_si512(p.cast())
    }
    #[inline(always)]
    pub unsafe fn store1_i64(p: *mut i64, v: V) {
        _mm512_storeu_si512(p.cast(), v);
    }

    // --- 32-bit lane operations ---
    #[inline(always)]
    pub unsafe fn add32(a: V, b: V) -> V {
        _mm512_add_epi32(a, b)
    }
    #[inline(always)]
    pub unsafe fn sub32(a: V, b: V) -> V {
        _mm512_sub_epi32(a, b)
    }
    #[inline(always)]
    pub unsafe fn mul32(a: V, b: V) -> V {
        _mm512_mullo_epi32(a, b)
    }
    #[inline(always)]
    pub unsafe fn and_v(a: V, b: V) -> V {
        _mm512_and_si512(a, b)
    }
    #[inline(always)]
    pub unsafe fn or_v(a: V, b: V) -> V {
        _mm512_or_si512(a, b)
    }
    #[inline(always)]
    pub unsafe fn xor_v(a: V, b: V) -> V {
        _mm512_xor_si512(a, b)
    }
    #[inline(always)]
    pub unsafe fn shl32_v(a: V, b: V) -> V {
        _mm512_sllv_epi32(a, _mm512_and_si512(b, _mm512_set1_epi32(31)))
    }
    #[inline(always)]
    pub unsafe fn shr32_v(a: V, b: V) -> V {
        _mm512_srlv_epi32(a, _mm512_and_si512(b, _mm512_set1_epi32(31)))
    }
    #[inline(always)]
    pub unsafe fn shl32_i(a: V, cnt: *const u64) -> V {
        _mm512_sll_epi32(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn shr32_i(a: V, cnt: *const u64) -> V {
        _mm512_srl_epi32(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn sar32_i(a: V, cnt: *const u64) -> V {
        _mm512_sra_epi32(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn select32(cond: V, a: V, b: V) -> V {
        let m = _mm512_test_epi32_mask(cond, cond);
        _mm512_mask_blend_epi32(m, b, a)
    }
    #[inline(always)]
    pub unsafe fn ctz32(a: V) -> V {
        #[cfg(target_feature = "avx512cd")]
        {
            let t = _mm512_andnot_si512(a, _mm512_sub_epi32(a, _mm512_set1_epi32(1)));
            _mm512_sub_epi32(_mm512_set1_epi32(32), _mm512_lzcnt_epi32(t))
        }
        #[cfg(not(target_feature = "avx512cd"))]
        {
            let arr: [u32; 16] = core::mem::transmute(a);
            core::mem::transmute(arr.map(|x| x.trailing_zeros()))
        }
    }
    #[inline(always)]
    pub unsafe fn bswap32(a: V) -> V {
        #[cfg(target_feature = "avx512bw")]
        {
            let shuf = broadcast_lut16(&[3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]);
            _mm512_shuffle_epi8(a, shuf)
        }
        #[cfg(not(target_feature = "avx512bw"))]
        {
            let arr: [u32; 16] = core::mem::transmute(a);
            core::mem::transmute(arr.map(u32::swap_bytes))
        }
    }

    // --- 64-bit lane operations ---
    #[inline(always)]
    pub unsafe fn add64(a: V, b: V) -> V {
        _mm512_add_epi64(a, b)
    }
    #[inline(always)]
    pub unsafe fn sub64(a: V, b: V) -> V {
        _mm512_sub_epi64(a, b)
    }
    #[inline(always)]
    pub unsafe fn mul64(a: V, b: V) -> V {
        #[cfg(target_feature = "avx512dq")]
        {
            _mm512_mullo_epi64(a, b)
        }
        #[cfg(not(target_feature = "avx512dq"))]
        {
            let a_hi = _mm512_srli_epi64::<32>(a);
            let b_hi = _mm512_srli_epi64::<32>(b);
            let lo_lo = _mm512_mul_epu32(a, b);
            let cross = _mm512_add_epi64(_mm512_mul_epu32(a, b_hi), _mm512_mul_epu32(a_hi, b));
            _mm512_add_epi64(lo_lo, _mm512_slli_epi64::<32>(cross))
        }
    }
    #[inline(always)]
    pub unsafe fn mulhi64(a: V, b: V) -> V {
        let mask32 = _mm512_set1_epi64(0xffff_ffff);
        let a_hi = _mm512_srli_epi64::<32>(a);
        let b_hi = _mm512_srli_epi64::<32>(b);
        let p00 = _mm512_mul_epu32(a, b);
        let p01 = _mm512_mul_epu32(a, b_hi);
        let p10 = _mm512_mul_epu32(a_hi, b);
        let p11 = _mm512_mul_epu32(a_hi, b_hi);
        let mid = _mm512_add_epi64(
            _mm512_srli_epi64::<32>(p00),
            _mm512_add_epi64(_mm512_and_si512(p01, mask32), _mm512_and_si512(p10, mask32)),
        );
        _mm512_add_epi64(
            _mm512_add_epi64(p11, _mm512_srli_epi64::<32>(mid)),
            _mm512_add_epi64(_mm512_srli_epi64::<32>(p01), _mm512_srli_epi64::<32>(p10)),
        )
    }
    #[inline(always)]
    pub unsafe fn shl64_v(a: V, b: V) -> V {
        _mm512_sllv_epi64(a, _mm512_and_si512(b, _mm512_set1_epi64(63)))
    }
    #[inline(always)]
    pub unsafe fn shr64_v(a: V, b: V) -> V {
        _mm512_srlv_epi64(a, _mm512_and_si512(b, _mm512_set1_epi64(63)))
    }
    #[inline(always)]
    pub unsafe fn shl64_i(a: V, cnt: *const u64) -> V {
        _mm512_sll_epi64(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn shr64_i(a: V, cnt: *const u64) -> V {
        _mm512_srl_epi64(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn sar64_i(a: V, cnt: *const u64) -> V {
        _mm512_sra_epi64(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn select64(cond: V, a: V, b: V) -> V {
        let m = _mm512_test_epi64_mask(cond, cond);
        _mm512_mask_blend_epi64(m, b, a)
    }
    #[inline(always)]
    pub unsafe fn ctz64(a: V) -> V {
        #[cfg(target_feature = "avx512cd")]
        {
            let t = _mm512_andnot_si512(a, _mm512_sub_epi64(a, _mm512_set1_epi64(1)));
            _mm512_sub_epi64(_mm512_set1_epi64(64), _mm512_lzcnt_epi64(t))
        }
        #[cfg(not(target_feature = "avx512cd"))]
        {
            let arr: [u64; 8] = core::mem::transmute(a);
            core::mem::transmute(arr.map(|x| x.trailing_zeros() as u64))
        }
    }
    #[inline(always)]
    pub unsafe fn bswap64(a: V) -> V {
        #[cfg(target_feature = "avx512bw")]
        {
            let shuf = broadcast_lut16(&[7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]);
            _mm512_shuffle_epi8(a, shuf)
        }
        #[cfg(not(target_feature = "avx512bw"))]
        {
            let arr: [u64; 8] = core::mem::transmute(a);
            core::mem::transmute(arr.map(u64::swap_bytes))
        }
    }

    // --- Shared byte-LUT & IEEE-754 f64 operations ---
    #[inline(always)]
    pub unsafe fn nibble_lut(a: V, tbl: V) -> V {
        #[cfg(target_feature = "avx512bw")]
        {
            _mm512_shuffle_epi8(tbl, _mm512_and_si512(a, _mm512_set1_epi8(0x0f)))
        }
        #[cfg(not(target_feature = "avx512bw"))]
        {
            let bytes: [u8; 64] = core::mem::transmute(a);
            let lut: [u8; 64] = core::mem::transmute(tbl);
            core::mem::transmute(bytes.map(|b| lut[(b & 15) as usize]))
        }
    }
    #[inline(always)]
    pub unsafe fn byte_lut32(a: V, tbl_lo: V, tbl_hi: V) -> V {
        #[cfg(target_feature = "avx512vbmi")]
        {
            let _ = tbl_hi;
            _mm512_permutexvar_epi8(_mm512_and_si512(a, _mm512_set1_epi8(0x1f)), tbl_lo)
        }
        #[cfg(all(not(target_feature = "avx512vbmi"), target_feature = "avx512bw"))]
        {
            let idx = _mm512_and_si512(a, _mm512_set1_epi8(0x0f));
            let lo = _mm512_shuffle_epi8(tbl_lo, idx);
            let hi = _mm512_shuffle_epi8(tbl_hi, idx);
            let m = _mm512_movepi8_mask(_mm512_slli_epi16::<3>(a));
            _mm512_mask_blend_epi8(m, lo, hi)
        }
        #[cfg(not(target_feature = "avx512bw"))]
        {
            let bytes: [u8; 64] = core::mem::transmute(a);
            let lo: [u8; 64] = core::mem::transmute(tbl_lo);
            let hi: [u8; 64] = core::mem::transmute(tbl_hi);
            core::mem::transmute(bytes.map(|b| {
                let k = (b & 31) as usize;
                if k < 16 { lo[k] } else { hi[k - 16] }
            }))
        }
    }
    #[inline(always)]
    pub unsafe fn f64_add(a: V, b: V) -> V {
        _mm512_castpd_si512(_mm512_add_pd(
            _mm512_castsi512_pd(a),
            _mm512_castsi512_pd(b),
        ))
    }
    #[inline(always)]
    pub unsafe fn f64_sub(a: V, b: V) -> V {
        _mm512_castpd_si512(_mm512_sub_pd(
            _mm512_castsi512_pd(a),
            _mm512_castsi512_pd(b),
        ))
    }
    #[inline(always)]
    pub unsafe fn f64_mul(a: V, b: V) -> V {
        _mm512_castpd_si512(_mm512_mul_pd(
            _mm512_castsi512_pd(a),
            _mm512_castsi512_pd(b),
        ))
    }
    #[inline(always)]
    pub unsafe fn u64_to_f64(a: V) -> V {
        #[cfg(target_feature = "avx512dq")]
        {
            _mm512_castpd_si512(_mm512_cvtepu64_pd(a))
        }
        #[cfg(not(target_feature = "avx512dq"))]
        {
            let arr: [u64; 8] = core::mem::transmute(a);
            core::mem::transmute(arr.map(|u| (u as f64).to_bits()))
        }
    }

    // --- Fused peephole primitives (`vpternlogd` / `vpternlogq`) ---
    #[inline(always)]
    pub unsafe fn bitselect(a: V, b: V, mask: V) -> V {
        _mm512_ternarylogic_epi32::<0xE4>(a, b, mask)
    }
    #[inline(always)]
    pub unsafe fn mask_lowbit_xor32(a: V, b: V, c: V) -> V {
        let m = _mm512_srai_epi32::<31>(_mm512_slli_epi32::<31>(b));
        _mm512_ternarylogic_epi32::<0x78>(a, m, c)
    }
    #[inline(always)]
    pub unsafe fn mask_lowbit_xor64(a: V, b: V, c: V) -> V {
        let m = _mm512_srai_epi64::<63>(_mm512_slli_epi64::<63>(b));
        _mm512_ternarylogic_epi64::<0x78>(a, m, c)
    }
    #[inline(always)]
    pub unsafe fn xor_shl_and32(a: V, cnt: *const u64, c: V) -> V {
        let sh = _mm512_sll_epi32(a, _mm_loadl_epi64(cnt.cast()));
        _mm512_ternarylogic_epi32::<0x78>(a, sh, c)
    }
    #[inline(always)]
    pub unsafe fn xor_shl_and64(a: V, cnt: *const u64, c: V) -> V {
        let sh = _mm512_sll_epi64(a, _mm_loadl_epi64(cnt.cast()));
        _mm512_ternarylogic_epi64::<0x78>(a, sh, c)
    }
}

// ============================================================================
// Tier 2: x86-64 AVX2 (`__m256i`)
// ============================================================================
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
mod imp {
    use core::arch::x86_64::*;

    pub type V = __m256i;
    pub const VEC_BYTES: usize = 32;
    pub const LANES32: usize = 8;
    pub const LANES64: usize = 4;

    #[inline(always)]
    pub unsafe fn zero() -> V {
        _mm256_setzero_si256()
    }
    #[inline(always)]
    pub unsafe fn splat32(x: u32) -> V {
        _mm256_set1_epi32(x as i32)
    }
    #[inline(always)]
    pub unsafe fn splat64(x: i64) -> V {
        _mm256_set1_epi64x(x)
    }
    #[inline(always)]
    pub unsafe fn broadcast_lut16(lut: &[u8; 16]) -> V {
        let x = _mm_loadu_si128(lut.as_ptr().cast());
        _mm256_broadcastsi128_si256(x)
    }
    #[inline(always)]
    pub unsafe fn broadcast_lut32(lo: &[u8; 16], hi: &[u8; 16]) -> (V, V) {
        (broadcast_lut16(lo), broadcast_lut16(hi))
    }

    #[inline(always)]
    pub unsafe fn load1_u32(p: *const u32) -> V {
        _mm256_loadu_si256(p.cast())
    }
    #[inline(always)]
    pub unsafe fn store1_u32(p: *mut u32, v: V) {
        _mm256_storeu_si256(p.cast(), v);
    }
    #[inline(always)]
    pub unsafe fn load1_i64(p: *const i64) -> V {
        _mm256_loadu_si256(p.cast())
    }
    #[inline(always)]
    pub unsafe fn store1_i64(p: *mut i64, v: V) {
        _mm256_storeu_si256(p.cast(), v);
    }

    #[inline(always)]
    pub unsafe fn add32(a: V, b: V) -> V {
        _mm256_add_epi32(a, b)
    }
    #[inline(always)]
    pub unsafe fn sub32(a: V, b: V) -> V {
        _mm256_sub_epi32(a, b)
    }
    #[inline(always)]
    pub unsafe fn mul32(a: V, b: V) -> V {
        _mm256_mullo_epi32(a, b)
    }
    #[inline(always)]
    pub unsafe fn and_v(a: V, b: V) -> V {
        _mm256_and_si256(a, b)
    }
    #[inline(always)]
    pub unsafe fn or_v(a: V, b: V) -> V {
        _mm256_or_si256(a, b)
    }
    #[inline(always)]
    pub unsafe fn xor_v(a: V, b: V) -> V {
        _mm256_xor_si256(a, b)
    }
    #[inline(always)]
    pub unsafe fn shl32_v(a: V, b: V) -> V {
        _mm256_sllv_epi32(a, _mm256_and_si256(b, _mm256_set1_epi32(31)))
    }
    #[inline(always)]
    pub unsafe fn shr32_v(a: V, b: V) -> V {
        _mm256_srlv_epi32(a, _mm256_and_si256(b, _mm256_set1_epi32(31)))
    }
    #[inline(always)]
    pub unsafe fn shl32_i(a: V, cnt: *const u64) -> V {
        _mm256_sll_epi32(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn shr32_i(a: V, cnt: *const u64) -> V {
        _mm256_srl_epi32(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn sar32_i(a: V, cnt: *const u64) -> V {
        _mm256_sra_epi32(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn select32(cond: V, a: V, b: V) -> V {
        let eqz = _mm256_cmpeq_epi32(cond, _mm256_setzero_si256());
        _mm256_blendv_epi8(a, b, eqz)
    }
    #[inline(always)]
    pub unsafe fn ctz32(a: V) -> V {
        let t = _mm256_andnot_si256(a, _mm256_sub_epi32(a, _mm256_set1_epi32(1)));
        let lut = broadcast_lut16(&[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]);
        let lo = _mm256_and_si256(t, _mm256_set1_epi8(0x0f));
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(t), _mm256_set1_epi8(0x0f));
        let bytes = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
        _mm256_srli_epi32::<24>(_mm256_mullo_epi32(bytes, _mm256_set1_epi32(0x0101_0101)))
    }
    #[inline(always)]
    pub unsafe fn bswap32(a: V) -> V {
        let shuf = broadcast_lut16(&[3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]);
        _mm256_shuffle_epi8(a, shuf)
    }

    #[inline(always)]
    pub unsafe fn add64(a: V, b: V) -> V {
        _mm256_add_epi64(a, b)
    }
    #[inline(always)]
    pub unsafe fn sub64(a: V, b: V) -> V {
        _mm256_sub_epi64(a, b)
    }
    #[inline(always)]
    pub unsafe fn mul64(a: V, b: V) -> V {
        let a_hi = _mm256_srli_epi64::<32>(a);
        let b_hi = _mm256_srli_epi64::<32>(b);
        let lo_lo = _mm256_mul_epu32(a, b);
        let cross = _mm256_add_epi64(_mm256_mul_epu32(a, b_hi), _mm256_mul_epu32(a_hi, b));
        _mm256_add_epi64(lo_lo, _mm256_slli_epi64::<32>(cross))
    }
    #[inline(always)]
    pub unsafe fn mulhi64(a: V, b: V) -> V {
        let mask32 = _mm256_set1_epi64x(0xffff_ffff);
        let a_hi = _mm256_srli_epi64::<32>(a);
        let b_hi = _mm256_srli_epi64::<32>(b);
        let p00 = _mm256_mul_epu32(a, b);
        let p01 = _mm256_mul_epu32(a, b_hi);
        let p10 = _mm256_mul_epu32(a_hi, b);
        let p11 = _mm256_mul_epu32(a_hi, b_hi);
        let mid = _mm256_add_epi64(
            _mm256_srli_epi64::<32>(p00),
            _mm256_add_epi64(_mm256_and_si256(p01, mask32), _mm256_and_si256(p10, mask32)),
        );
        _mm256_add_epi64(
            _mm256_add_epi64(p11, _mm256_srli_epi64::<32>(mid)),
            _mm256_add_epi64(_mm256_srli_epi64::<32>(p01), _mm256_srli_epi64::<32>(p10)),
        )
    }
    #[inline(always)]
    pub unsafe fn shl64_v(a: V, b: V) -> V {
        _mm256_sllv_epi64(a, _mm256_and_si256(b, _mm256_set1_epi64x(63)))
    }
    #[inline(always)]
    pub unsafe fn shr64_v(a: V, b: V) -> V {
        _mm256_srlv_epi64(a, _mm256_and_si256(b, _mm256_set1_epi64x(63)))
    }
    #[inline(always)]
    pub unsafe fn shl64_i(a: V, cnt: *const u64) -> V {
        _mm256_sll_epi64(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn shr64_i(a: V, cnt: *const u64) -> V {
        _mm256_srl_epi64(a, _mm_loadl_epi64(cnt.cast()))
    }
    #[inline(always)]
    pub unsafe fn sar64_i(a: V, cnt: *const u64) -> V {
        let arr: [i64; 4] = core::mem::transmute(a);
        let sh = (*cnt) as u32 & 63;
        core::mem::transmute(arr.map(|x| x >> sh))
    }
    #[inline(always)]
    pub unsafe fn select64(cond: V, a: V, b: V) -> V {
        let eqz = _mm256_cmpeq_epi64(cond, _mm256_setzero_si256());
        _mm256_blendv_epi8(a, b, eqz)
    }
    #[inline(always)]
    pub unsafe fn ctz64(a: V) -> V {
        let t = _mm256_andnot_si256(a, _mm256_sub_epi64(a, _mm256_set1_epi64x(1)));
        let lut = broadcast_lut16(&[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]);
        let lo = _mm256_and_si256(t, _mm256_set1_epi8(0x0f));
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(t), _mm256_set1_epi8(0x0f));
        let bytes = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
        _mm256_sad_epu8(bytes, _mm256_setzero_si256())
    }
    #[inline(always)]
    pub unsafe fn bswap64(a: V) -> V {
        let shuf = broadcast_lut16(&[7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]);
        _mm256_shuffle_epi8(a, shuf)
    }

    #[inline(always)]
    pub unsafe fn nibble_lut(a: V, tbl: V) -> V {
        _mm256_shuffle_epi8(tbl, _mm256_and_si256(a, _mm256_set1_epi8(0x0f)))
    }
    #[inline(always)]
    pub unsafe fn byte_lut32(a: V, tbl_lo: V, tbl_hi: V) -> V {
        let idx = _mm256_and_si256(a, _mm256_set1_epi8(0x0f));
        let lo = _mm256_shuffle_epi8(tbl_lo, idx);
        let hi = _mm256_shuffle_epi8(tbl_hi, idx);
        _mm256_blendv_epi8(lo, hi, _mm256_slli_epi16::<3>(a))
    }
    #[inline(always)]
    pub unsafe fn f64_add(a: V, b: V) -> V {
        _mm256_castpd_si256(_mm256_add_pd(
            _mm256_castsi256_pd(a),
            _mm256_castsi256_pd(b),
        ))
    }
    #[inline(always)]
    pub unsafe fn f64_sub(a: V, b: V) -> V {
        _mm256_castpd_si256(_mm256_sub_pd(
            _mm256_castsi256_pd(a),
            _mm256_castsi256_pd(b),
        ))
    }
    #[inline(always)]
    pub unsafe fn f64_mul(a: V, b: V) -> V {
        _mm256_castpd_si256(_mm256_mul_pd(
            _mm256_castsi256_pd(a),
            _mm256_castsi256_pd(b),
        ))
    }
    #[inline(always)]
    pub unsafe fn u64_to_f64(a: V) -> V {
        let arr: [u64; 4] = core::mem::transmute(a);
        core::mem::transmute(arr.map(|u| (u as f64).to_bits()))
    }

    #[inline(always)]
    pub unsafe fn bitselect(a: V, b: V, mask: V) -> V {
        _mm256_or_si256(_mm256_and_si256(a, mask), _mm256_andnot_si256(mask, b))
    }
    #[inline(always)]
    pub unsafe fn mask_lowbit_xor32(a: V, b: V, c: V) -> V {
        let m = _mm256_srai_epi32::<31>(_mm256_slli_epi32::<31>(b));
        _mm256_xor_si256(a, _mm256_and_si256(m, c))
    }
    #[inline(always)]
    pub unsafe fn mask_lowbit_xor64(a: V, b: V, c: V) -> V {
        let m = _mm256_sub_epi64(
            _mm256_setzero_si256(),
            _mm256_and_si256(b, _mm256_set1_epi64x(1)),
        );
        _mm256_xor_si256(a, _mm256_and_si256(m, c))
    }
    #[inline(always)]
    pub unsafe fn xor_shl_and32(a: V, cnt: *const u64, c: V) -> V {
        let sh = _mm256_sll_epi32(a, _mm_loadl_epi64(cnt.cast()));
        _mm256_xor_si256(a, _mm256_and_si256(sh, c))
    }
    #[inline(always)]
    pub unsafe fn xor_shl_and64(a: V, cnt: *const u64, c: V) -> V {
        let sh = _mm256_sll_epi64(a, _mm_loadl_epi64(cnt.cast()));
        _mm256_xor_si256(a, _mm256_and_si256(sh, c))
    }
}

// ============================================================================
// Tier 3: 128-bit baseline (`__m128i` on x86-64 SSE2, `uint64x2_t` on aarch64,
//         `[u64; 2]` elsewhere)
// ============================================================================
#[cfg(not(all(
    target_arch = "x86_64",
    any(target_feature = "avx512f", target_feature = "avx2")
)))]
mod imp {
    #[cfg(target_arch = "x86_64")]
    pub type V = core::arch::x86_64::__m128i;
    #[cfg(target_arch = "aarch64")]
    pub type V = core::arch::aarch64::uint64x2_t;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub struct V(pub [u64; 2]);

    pub const VEC_BYTES: usize = 16;
    pub const LANES32: usize = 4;
    pub const LANES64: usize = 2;

    #[inline(always)]
    fn as_u32(v: V) -> [u32; 4] {
        unsafe { core::mem::transmute(v) }
    }
    #[inline(always)]
    fn from_u32(a: [u32; 4]) -> V {
        unsafe { core::mem::transmute(a) }
    }
    #[inline(always)]
    fn as_i64(v: V) -> [i64; 2] {
        unsafe { core::mem::transmute(v) }
    }
    #[inline(always)]
    fn from_i64(a: [i64; 2]) -> V {
        unsafe { core::mem::transmute(a) }
    }
    #[inline(always)]
    fn as_u8(v: V) -> [u8; 16] {
        unsafe { core::mem::transmute(v) }
    }
    #[inline(always)]
    fn from_u8(a: [u8; 16]) -> V {
        unsafe { core::mem::transmute(a) }
    }

    #[inline(always)]
    pub unsafe fn zero() -> V {
        from_i64([0, 0])
    }
    #[inline(always)]
    pub unsafe fn splat32(x: u32) -> V {
        from_u32([x; 4])
    }
    #[inline(always)]
    pub unsafe fn splat64(x: i64) -> V {
        from_i64([x; 2])
    }
    #[inline(always)]
    pub unsafe fn broadcast_lut16(lut: &[u8; 16]) -> V {
        from_u8(*lut)
    }
    #[inline(always)]
    pub unsafe fn broadcast_lut32(lo: &[u8; 16], hi: &[u8; 16]) -> (V, V) {
        (from_u8(*lo), from_u8(*hi))
    }

    #[inline(always)]
    pub unsafe fn load1_u32(p: *const u32) -> V {
        core::ptr::read_unaligned(p.cast())
    }
    #[inline(always)]
    pub unsafe fn store1_u32(p: *mut u32, v: V) {
        core::ptr::write_unaligned(p.cast(), v);
    }
    #[inline(always)]
    pub unsafe fn load1_i64(p: *const i64) -> V {
        core::ptr::read_unaligned(p.cast())
    }
    #[inline(always)]
    pub unsafe fn store1_i64(p: *mut i64, v: V) {
        core::ptr::write_unaligned(p.cast(), v);
    }

    #[inline(always)]
    pub unsafe fn add32(a: V, b: V) -> V {
        let (x, y) = (as_u32(a), as_u32(b));
        from_u32([
            x[0].wrapping_add(y[0]),
            x[1].wrapping_add(y[1]),
            x[2].wrapping_add(y[2]),
            x[3].wrapping_add(y[3]),
        ])
    }
    #[inline(always)]
    pub unsafe fn sub32(a: V, b: V) -> V {
        let (x, y) = (as_u32(a), as_u32(b));
        from_u32([
            x[0].wrapping_sub(y[0]),
            x[1].wrapping_sub(y[1]),
            x[2].wrapping_sub(y[2]),
            x[3].wrapping_sub(y[3]),
        ])
    }
    #[inline(always)]
    pub unsafe fn mul32(a: V, b: V) -> V {
        let (x, y) = (as_u32(a), as_u32(b));
        from_u32([
            x[0].wrapping_mul(y[0]),
            x[1].wrapping_mul(y[1]),
            x[2].wrapping_mul(y[2]),
            x[3].wrapping_mul(y[3]),
        ])
    }
    #[inline(always)]
    pub unsafe fn and_v(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([x[0] & y[0], x[1] & y[1]])
    }
    #[inline(always)]
    pub unsafe fn or_v(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([x[0] | y[0], x[1] | y[1]])
    }
    #[inline(always)]
    pub unsafe fn xor_v(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([x[0] ^ y[0], x[1] ^ y[1]])
    }
    #[inline(always)]
    pub unsafe fn shl32_v(a: V, b: V) -> V {
        let (x, y) = (as_u32(a), as_u32(b));
        from_u32([
            x[0] << (y[0] & 31),
            x[1] << (y[1] & 31),
            x[2] << (y[2] & 31),
            x[3] << (y[3] & 31),
        ])
    }
    #[inline(always)]
    pub unsafe fn shr32_v(a: V, b: V) -> V {
        let (x, y) = (as_u32(a), as_u32(b));
        from_u32([
            x[0] >> (y[0] & 31),
            x[1] >> (y[1] & 31),
            x[2] >> (y[2] & 31),
            x[3] >> (y[3] & 31),
        ])
    }
    #[inline(always)]
    pub unsafe fn shl32_i(a: V, cnt: *const u64) -> V {
        let sh = (*cnt) as u32 & 31;
        from_u32(as_u32(a).map(|x| x << sh))
    }
    #[inline(always)]
    pub unsafe fn shr32_i(a: V, cnt: *const u64) -> V {
        let sh = (*cnt) as u32 & 31;
        from_u32(as_u32(a).map(|x| x >> sh))
    }
    #[inline(always)]
    pub unsafe fn sar32_i(a: V, cnt: *const u64) -> V {
        let sh = (*cnt) as u32 & 31;
        from_u32(as_u32(a).map(|x| ((x as i32) >> sh) as u32))
    }
    #[inline(always)]
    pub unsafe fn select32(cond: V, a: V, b: V) -> V {
        let (c, x, y) = (as_u32(cond), as_u32(a), as_u32(b));
        from_u32([
            if c[0] != 0 { x[0] } else { y[0] },
            if c[1] != 0 { x[1] } else { y[1] },
            if c[2] != 0 { x[2] } else { y[2] },
            if c[3] != 0 { x[3] } else { y[3] },
        ])
    }
    #[inline(always)]
    pub unsafe fn ctz32(a: V) -> V {
        from_u32(as_u32(a).map(u32::trailing_zeros))
    }
    #[inline(always)]
    pub unsafe fn bswap32(a: V) -> V {
        from_u32(as_u32(a).map(u32::swap_bytes))
    }

    #[inline(always)]
    pub unsafe fn add64(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([x[0].wrapping_add(y[0]), x[1].wrapping_add(y[1])])
    }
    #[inline(always)]
    pub unsafe fn sub64(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([x[0].wrapping_sub(y[0]), x[1].wrapping_sub(y[1])])
    }
    #[inline(always)]
    pub unsafe fn mul64(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([x[0].wrapping_mul(y[0]), x[1].wrapping_mul(y[1])])
    }
    #[inline(always)]
    pub unsafe fn mulhi64(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([
            (((x[0] as u64 as u128) * (y[0] as u64 as u128)) >> 64) as i64,
            (((x[1] as u64 as u128) * (y[1] as u64 as u128)) >> 64) as i64,
        ])
    }
    #[inline(always)]
    pub unsafe fn shl64_v(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([
            ((x[0] as u64) << (y[0] & 63)) as i64,
            ((x[1] as u64) << (y[1] & 63)) as i64,
        ])
    }
    #[inline(always)]
    pub unsafe fn shr64_v(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([
            ((x[0] as u64) >> (y[0] & 63)) as i64,
            ((x[1] as u64) >> (y[1] & 63)) as i64,
        ])
    }
    #[inline(always)]
    pub unsafe fn shl64_i(a: V, cnt: *const u64) -> V {
        let sh = (*cnt) as u32 & 63;
        from_i64(as_i64(a).map(|x| ((x as u64) << sh) as i64))
    }
    #[inline(always)]
    pub unsafe fn shr64_i(a: V, cnt: *const u64) -> V {
        let sh = (*cnt) as u32 & 63;
        from_i64(as_i64(a).map(|x| ((x as u64) >> sh) as i64))
    }
    #[inline(always)]
    pub unsafe fn sar64_i(a: V, cnt: *const u64) -> V {
        let sh = (*cnt) as u32 & 63;
        from_i64(as_i64(a).map(|x| x >> sh))
    }
    #[inline(always)]
    pub unsafe fn select64(cond: V, a: V, b: V) -> V {
        let (c, x, y) = (as_i64(cond), as_i64(a), as_i64(b));
        from_i64([
            if c[0] != 0 { x[0] } else { y[0] },
            if c[1] != 0 { x[1] } else { y[1] },
        ])
    }
    #[inline(always)]
    pub unsafe fn ctz64(a: V) -> V {
        from_i64(as_i64(a).map(|x| (x as u64).trailing_zeros() as i64))
    }
    #[inline(always)]
    pub unsafe fn bswap64(a: V) -> V {
        from_i64(as_i64(a).map(i64::swap_bytes))
    }

    #[inline(always)]
    pub unsafe fn nibble_lut(a: V, tbl: V) -> V {
        let lut = as_u8(tbl);
        from_u8(as_u8(a).map(|b| lut[(b & 15) as usize]))
    }
    #[inline(always)]
    pub unsafe fn byte_lut32(a: V, tbl_lo: V, tbl_hi: V) -> V {
        let (lo, hi) = (as_u8(tbl_lo), as_u8(tbl_hi));
        from_u8(as_u8(a).map(|b| {
            let k = (b & 31) as usize;
            if k < 16 { lo[k] } else { hi[k - 16] }
        }))
    }
    #[inline(always)]
    pub unsafe fn f64_add(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([
            (f64::from_bits(x[0] as u64) + f64::from_bits(y[0] as u64)).to_bits() as i64,
            (f64::from_bits(x[1] as u64) + f64::from_bits(y[1] as u64)).to_bits() as i64,
        ])
    }
    #[inline(always)]
    pub unsafe fn f64_sub(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([
            (f64::from_bits(x[0] as u64) - f64::from_bits(y[0] as u64)).to_bits() as i64,
            (f64::from_bits(x[1] as u64) - f64::from_bits(y[1] as u64)).to_bits() as i64,
        ])
    }
    #[inline(always)]
    pub unsafe fn f64_mul(a: V, b: V) -> V {
        let (x, y) = (as_i64(a), as_i64(b));
        from_i64([
            (f64::from_bits(x[0] as u64) * f64::from_bits(y[0] as u64)).to_bits() as i64,
            (f64::from_bits(x[1] as u64) * f64::from_bits(y[1] as u64)).to_bits() as i64,
        ])
    }
    #[inline(always)]
    pub unsafe fn u64_to_f64(a: V) -> V {
        from_i64(as_i64(a).map(|x| ((x as u64) as f64).to_bits() as i64))
    }

    #[inline(always)]
    pub unsafe fn bitselect(a: V, b: V, mask: V) -> V {
        let (x, y, m) = (as_i64(a), as_i64(b), as_i64(mask));
        from_i64([
            (x[0] & m[0]) | (y[0] & !m[0]),
            (x[1] & m[1]) | (y[1] & !m[1]),
        ])
    }
    #[inline(always)]
    pub unsafe fn mask_lowbit_xor32(a: V, b: V, c: V) -> V {
        let (x, y, k) = (as_u32(a), as_u32(b), as_u32(c));
        from_u32([
            x[0] ^ ((y[0] & 1).wrapping_neg() & k[0]),
            x[1] ^ ((y[1] & 1).wrapping_neg() & k[1]),
            x[2] ^ ((y[2] & 1).wrapping_neg() & k[2]),
            x[3] ^ ((y[3] & 1).wrapping_neg() & k[3]),
        ])
    }
    #[inline(always)]
    pub unsafe fn mask_lowbit_xor64(a: V, b: V, c: V) -> V {
        let (x, y, k) = (as_i64(a), as_i64(b), as_i64(c));
        from_i64([x[0] ^ (-(y[0] & 1) & k[0]), x[1] ^ (-(y[1] & 1) & k[1])])
    }
    #[inline(always)]
    pub unsafe fn xor_shl_and32(a: V, cnt: *const u64, c: V) -> V {
        let sh = (*cnt) as u32 & 31;
        let (x, k) = (as_u32(a), as_u32(c));
        from_u32([
            x[0] ^ ((x[0] << sh) & k[0]),
            x[1] ^ ((x[1] << sh) & k[1]),
            x[2] ^ ((x[2] << sh) & k[2]),
            x[3] ^ ((x[3] << sh) & k[3]),
        ])
    }
    #[inline(always)]
    pub unsafe fn xor_shl_and64(a: V, cnt: *const u64, c: V) -> V {
        let sh = (*cnt) as u32 & 63;
        let (x, k) = (as_i64(a), as_i64(c));
        from_i64([
            x[0] ^ ((((x[0] as u64) << sh) as i64) & k[0]),
            x[1] ^ ((((x[1] as u64) << sh) as i64) & k[1]),
        ])
    }
}

pub use imp::*;

pub const T32: usize = LANES32 * 4;
pub const T64: usize = LANES64 * 4;
pub type Tile = [V; 4];

#[inline(always)]
pub unsafe fn load4_u32(p: *const u32) -> Tile {
    [
        load1_u32(p),
        load1_u32(p.add(LANES32)),
        load1_u32(p.add(LANES32 * 2)),
        load1_u32(p.add(LANES32 * 3)),
    ]
}

#[inline(always)]
pub unsafe fn store4_u32(p: *mut u32, t: Tile) {
    store1_u32(p, t[0]);
    store1_u32(p.add(LANES32), t[1]);
    store1_u32(p.add(LANES32 * 2), t[2]);
    store1_u32(p.add(LANES32 * 3), t[3]);
}

#[inline(always)]
pub unsafe fn load4_i64(p: *const i64) -> Tile {
    [
        load1_i64(p),
        load1_i64(p.add(LANES64)),
        load1_i64(p.add(LANES64 * 2)),
        load1_i64(p.add(LANES64 * 3)),
    ]
}

#[inline(always)]
pub unsafe fn store4_i64(p: *mut i64, t: Tile) {
    store1_i64(p, t[0]);
    store1_i64(p.add(LANES64), t[1]);
    store1_i64(p.add(LANES64 * 2), t[2]);
    store1_i64(p.add(LANES64 * 3), t[3]);
}

#[inline(always)]
pub fn map1(a: Tile, mut f: impl FnMut(V) -> V) -> Tile {
    [f(a[0]), f(a[1]), f(a[2]), f(a[3])]
}

#[inline(always)]
pub fn map2(a: Tile, b: Tile, mut f: impl FnMut(V, V) -> V) -> Tile {
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
}

#[inline(always)]
pub fn map3(a: Tile, b: Tile, c: Tile, mut f: impl FnMut(V, V, V) -> V) -> Tile {
    [
        f(a[0], b[0], c[0]),
        f(a[1], b[1], c[1]),
        f(a[2], b[2], c[2]),
        f(a[3], b[3], c[3]),
    ]
}

/// Name of the active compile-time SIMD backend.
pub const BACKEND_NAME: &str = if cfg!(all(target_arch = "x86_64", target_feature = "avx512f")) {
    "x86_64-avx512"
} else if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
    "x86_64-avx2"
} else if cfg!(target_arch = "x86_64") {
    "x86_64-sse2"
} else if cfg!(target_arch = "aarch64") {
    "aarch64-neon"
} else {
    "portable-128"
};
