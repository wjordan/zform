//! High-density typed column codecs, single- & multi-block FSST/front-coded dictionaries,
//! MTF-4096 recency coding (on both binary IDs and `i64` IDs), IEEE-754 byte-shuffle/bit-delta,
//! horizontal cross-field derivations, and per-column zone-map / Bloom summaries.

use crate::cells::{Cells, Col, NULL_CELL, ends_of};
use crate::lex::{
    format_base64, format_exact_decimal, format_exact_int, format_hex_bytes, format_ipv4_or_port,
    format_traceparent, format_ulid, format_uuid, parse_base64, parse_exact_decimal,
    parse_exact_int, parse_hex_exact, parse_ipv4_or_port, parse_timestamp, parse_traceparent,
    parse_ulid, parse_uuid, split_email, split_hierarchical,
};
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, compress_leaf, decompress_leaf,
    put_ivarint, put_uvarint,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

mod lcg;
mod mt;
pub(crate) use mt::{Seeds, with_seeds};

// ============================================================================
// 1. Per-Column Block Summary
// ============================================================================

/// What the index keeps about one column of one block.
#[derive(Debug, Clone, Default)]
pub struct ColSummary {
    /// `0` bytes, `1` integers, `2` timestamps in nanoseconds.
    pub numeric_kind: u8,
    pub min_val: i64,
    pub max_val: i64,
    /// No row of the block has a value in this column.
    pub all_null: bool,
    /// Cell `i` is row `i`'s: every row of the block has exactly one value in this column.
    pub per_row: bool,
    /// The ASCII letters in the column's cells as records show them, one bit each from `a`, case
    /// folded, and `ESCAPES` if any cell has a backslash.
    pub letters: u32,
}

/// The `ColSummary::letters` bit for a backslash: a cell may spell a value other than its bytes.
pub const ESCAPES: u32 = 1 << 26;
/// What `letters` is for a column whose cells are shown in another form than they are stored.
pub const ANY_LETTERS: u32 = (1 << 27) - 1;

/// The letters of `text`, as `ColSummary::letters` holds them.
pub fn letters_of(text: &[u8]) -> u32 {
    let mut mask = 0u32;
    for &b in text {
        let l = (b | 0x20).wrapping_sub(b'a');
        if l < 26 {
            mask |= 1 << l;
        } else if b == b'\\' {
            mask |= ESCAPES;
        }
    }
    mask
}

/// Bounds of a column whose every value is a timestamp, bare or as a JSON string.
fn time_bounds(vals: &[&[u8]]) -> Option<(i64, i64)> {
    let mut times = vals.iter().map(|v| {
        let v = v
            .strip_prefix(b"\"")
            .and_then(|v| v.strip_suffix(b"\""))
            .unwrap_or(v);
        parse_timestamp(v).map(|(ns, _)| ns)
    });
    let first = times.next()??;
    times.try_fold((first, first), |(lo, hi), v| {
        v.map(|v| (lo.min(v), hi.max(v)))
    })
}

/// Bounds of a column whose every value is an exact integer, whichever codec stores it: a
/// dictionary of status codes prunes range queries as well as a packed integer column does.
fn int_bounds(vals: &[&[u8]]) -> Option<(i64, i64)> {
    let mut ints = vals.iter().map(|v| parse_exact_int(v).map(|(iv, _)| iv));
    let first = ints.next()??;
    ints.try_fold((first, first), |(lo, hi), v| {
        v.map(|v| (lo.min(v), hi.max(v)))
    })
}

impl ColSummary {
    pub fn of(vals: &[&[u8]]) -> Self {
        let letters = vals.iter().fold(0, |m, v| m | letters_of(v));
        Self {
            all_null: vals.is_empty(),
            letters,
            ..Self::default()
        }
    }
}
// ============================================================================
// 2. Integer & Ordinal Sequence Codecs (Const, FOR Bitpack, GCD Delta, DoD, MTF-4096)
// ============================================================================

#[inline]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn bitpack_u64(vals: &[u64], width: u8, out: &mut Vec<u8>) {
    if width == 0 || vals.is_empty() {
        return;
    }
    if width == 64 {
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        return;
    }
    let total_bits = vals.len() * (width as usize);
    let byte_len = total_bits.div_ceil(8);
    let base = out.len();
    out.resize(base + byte_len, 0);
    let dst = &mut out[base..];
    let mask = (1u64 << width) - 1;
    let mut bit_pos = 0usize;
    for &v in vals {
        let u = v & mask;
        let byte_idx = bit_pos >> 3;
        let bit_off = bit_pos & 7;
        let cur = (u as u128) << bit_off;
        let bytes = cur.to_le_bytes();
        let n_write = ((bit_off + width as usize).div_ceil(8)).min(dst.len() - byte_idx);
        for k in 0..n_write {
            dst[byte_idx + k] |= bytes[k];
        }
        bit_pos += width as usize;
    }
}

fn bitunpack_u64(src: &[u8], count: usize, width: u8, budget: &DecodeBudget) -> Result<Vec<u64>> {
    budget.check_alloc(count.saturating_mul(8), "bitunpack_u64")?;
    if width == 0 {
        return Ok(vec![0u64; count]);
    }
    if width > 64 {
        return Err(Error::Format(format!("invalid bitpack width {width}")));
    }
    let needed = (count * (width as usize)).div_ceil(8);
    if src.len() < needed {
        return Err(Error::Format("truncated bitpack buffer".into()));
    }
    if width == 64 {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            out.push(u64::from_le_bytes(
                src[i * 8..i * 8 + 8].try_into().unwrap(),
            ));
        }
        return Ok(out);
    }
    let mask = (1u64 << width) - 1;
    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let byte_idx = bit_pos >> 3;
        let bit_off = bit_pos & 7;
        // Eight bytes hold any value of up to 57 bits at any bit offset.
        if width <= 57 && byte_idx + 8 <= src.len() {
            let word = u64::from_le_bytes(src[byte_idx..byte_idx + 8].try_into().unwrap());
            out.push((word >> bit_off) & mask);
            bit_pos += width as usize;
            continue;
        }
        let mut raw = [0u8; 16];
        let avail = (src.len() - byte_idx).min(9);
        raw[..avail].copy_from_slice(&src[byte_idx..byte_idx + avail]);
        let word = (u128::from_le_bytes(raw) >> bit_off) as u64;
        out.push(word & mask);
        bit_pos += width as usize;
    }
    Ok(out)
}

/// Encodes a slice of `u32` ordinals/tags using the smallest of `Const`, `RLE`, `FOR bitpack`,
/// byte planes, delta or bytes with a continued tail.
pub fn encode_u32_seq(vals: &[u32], backend: EntropyBackend, level: i32, out: &mut Vec<u8>) {
    encode_u32_seq_with(vals, backend, level, true, out);
}

/// `tail` allows the mode that nests a second sequence.
fn encode_u32_seq_with(
    vals: &[u32],
    backend: EntropyBackend,
    level: i32,
    tail: bool,
    out: &mut Vec<u8>,
) {
    put_uvarint(out, vals.len() as u64);
    if vals.is_empty() {
        return;
    }
    let first = vals[0];
    let mut min_v = first;
    let mut max_v = first;
    let mut runs = 1usize;
    for i in 1..vals.len() {
        let v = vals[i];
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        if v != vals[i - 1] {
            runs += 1;
        }
    }
    if min_v == max_v {
        out.push(0); // Const
        put_uvarint(out, first as u64);
        return;
    }
    // Candidate 1: RLE (great when runs << vals.len())
    let mut best_mode = 2u8;
    let mut best_buf = Vec::new();
    if runs * 3 < vals.len() {
        let mut rle = Vec::with_capacity(runs * 4);
        put_uvarint(&mut rle, runs as u64);
        let mut i = 0usize;
        while i < vals.len() {
            let v = vals[i];
            let mut j = i + 1;
            while j < vals.len() && vals[j] == v {
                j += 1;
            }
            put_uvarint(&mut rle, v as u64);
            put_uvarint(&mut rle, (j - i) as u64);
            i = j;
        }
        let mut c = Vec::new();
        compress_leaf(&mut c, &rle, backend, level);
        best_mode = 1;
        best_buf = c;
    }
    // Candidate 2: FOR bitpack + entropy leaf
    let span = (max_v - min_v) as u64;
    let width = (64 - span.leading_zeros()) as u8;
    let mut packed = Vec::new();
    let shifted: Vec<u64> = vals.iter().map(|&v| (v - min_v) as u64).collect();
    bitpack_u64(&shifted, width, &mut packed);
    let mut for_cand = Vec::new();
    put_uvarint(&mut for_cand, min_v as u64);
    for_cand.push(width);
    compress_leaf(&mut for_cand, &packed, backend, level);
    if best_buf.is_empty() || for_cand.len() < best_buf.len() {
        best_mode = 2;
        best_buf = for_cand;
    }

    // Candidate 3: Byte-aligned FOR planes + entropy leaf (unlocks Zstd FSE fractional bits + LZ77!)
    if vals.len() >= 24 && span <= 0xffff_ffff {
        let bw: u8 = if span <= 0xff {
            1
        } else if span <= 0xffff {
            2
        } else {
            4
        };
        let n = vals.len();
        let mut planes = vec![0u8; n * (bw as usize)];
        for (i, &s) in shifted.iter().enumerate() {
            let le = (s as u32).to_le_bytes();
            for p in 0..(bw as usize) {
                planes[p * n + i] = le[p];
            }
        }
        let mut pcand = Vec::new();
        put_uvarint(&mut pcand, min_v as u64);
        pcand.push(bw);
        compress_leaf(&mut pcand, &planes, backend, level);
        if pcand.len() < best_buf.len() {
            best_mode = 3;
            best_buf = pcand;
        }
    }

    // Candidate 4: First-Order ZigZag Delta (`COL_DICT_REF` style — crushes first-seen dict ordinals!)
    if vals.len() >= 32 {
        let mut draw = Vec::with_capacity(vals.len() * 2);
        for i in 1..vals.len() {
            put_ivarint(&mut draw, (vals[i] as i64) - (vals[i - 1] as i64));
        }
        let mut dcand = Vec::new();
        put_uvarint(&mut dcand, first as u64);
        compress_leaf(&mut dcand, &draw, backend, level);
        if dcand.len() < best_buf.len() {
            best_mode = 4;
            best_buf = dcand;
        }
    }

    // Candidate 5: one byte per value, those of 255 and over continued in a second sequence.
    // Recency ranks and frequency-ordered ordinals have a long tail, whose low bytes are noise
    // in a byte plane; here the common values keep a plane to themselves.
    if tail && vals.len() >= 64 && span > 0xff {
        let mut bytes = Vec::with_capacity(vals.len());
        let mut over = Vec::new();
        for &s in &shifted {
            bytes.push(s.min(0xff) as u8);
            if s >= 0xff {
                over.push((s - 0xff) as u32);
            }
        }
        if over.len() * 2 <= vals.len() {
            let mut cand = Vec::new();
            put_uvarint(&mut cand, min_v as u64);
            compress_leaf(&mut cand, &bytes, backend, level);
            encode_u32_seq_with(&over, backend, level, false, &mut cand);
            if cand.len() < best_buf.len() {
                best_mode = 5;
                best_buf = cand;
            }
        }
    }

    out.push(best_mode);
    out.extend_from_slice(&best_buf);
}

pub fn decode_u32_seq(cur: &mut Cursor<'_>, budget: &DecodeBudget) -> Result<Vec<u32>> {
    decode_u32_seq_with(cur, budget, true)
}

fn decode_u32_seq_with(
    cur: &mut Cursor<'_>,
    budget: &DecodeBudget,
    tail: bool,
) -> Result<Vec<u32>> {
    let n = cur.uvar()? as usize;
    if n > budget.max_records_per_block * 4 {
        return Err(Error::BudgetExceeded("u32 sequence count"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    match cur.u8()? {
        0 => {
            let v = cur.uvar()? as u32;
            budget.check_alloc(n * 4, "const u32 seq")?;
            Ok(vec![v; n])
        }
        1 => {
            let raw = decompress_leaf(cur, budget)?;
            let mut rc = Cursor::new(&raw);
            let runs = rc.uvar()? as usize;
            budget.check_alloc(n * 4, "rle u32 seq")?;
            let mut out = Vec::with_capacity(n);
            for _ in 0..runs {
                let v = rc.uvar()? as u32;
                let len = rc.uvar()? as usize;
                if out.len().saturating_add(len) > n {
                    return Err(Error::Format("RLE u32 overflow".into()));
                }
                out.resize(out.len() + len, v);
            }
            if out.len() != n {
                return Err(Error::Format("RLE u32 count mismatch".into()));
            }
            Ok(out)
        }
        2 => {
            let min_v = cur.uvar()? as u32;
            let width = cur.u8()?;
            if width > 32 {
                return Err(Error::Format("u32 FOR width > 32".into()));
            }
            let packed = decompress_leaf(cur, budget)?;
            let shifted = bitunpack_u64(&packed, n, width, budget)?;
            Ok(shifted
                .into_iter()
                .map(|s| min_v.wrapping_add(s as u32))
                .collect())
        }
        3 => {
            let min_v = cur.uvar()? as u32;
            let bw = cur.u8()? as usize;
            if !matches!(bw, 1 | 2 | 4) {
                return Err(Error::Format("invalid u32 plane width".into()));
            }
            let planes = decompress_leaf(cur, budget)?;
            if planes.len() != n.saturating_mul(bw) {
                return Err(Error::Format("u32 plane buffer mismatch".into()));
            }
            budget.check_alloc(n * 4, "u32 plane seq")?;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut le = [0u8; 4];
                for p in 0..bw {
                    le[p] = planes[p * n + i];
                }
                out.push(min_v.wrapping_add(u32::from_le_bytes(le)));
            }
            Ok(out)
        }
        4 => {
            let first = cur.uvar()? as u32;
            let draw = decompress_leaf(cur, budget)?;
            let mut rc = Cursor::new(&draw);
            budget.check_alloc(n * 4, "u32 delta seq")?;
            let mut out = Vec::with_capacity(n);
            out.push(first);
            let mut acc = first as i64;
            for _ in 1..n {
                acc = acc.wrapping_add(rc.ivar()?);
                out.push(acc as u32);
            }
            Ok(out)
        }
        5 if tail => {
            let min_v = cur.uvar()? as u32;
            let bytes = decompress_leaf(cur, budget)?;
            let mut over = decode_u32_seq_with(cur, budget, false)?.into_iter();
            if bytes.len() != n {
                return Err(Error::Format("u32 byte buffer mismatch".into()));
            }
            budget.check_alloc(n * 4, "u32 byte seq")?;
            let mut out = Vec::with_capacity(n);
            for b in bytes {
                let v = match b {
                    0xff => {
                        0xff + over
                            .next()
                            .ok_or_else(|| Error::Format("u32 tail underflow".into()))?
                            as u64
                    }
                    b => b as u64,
                };
                out.push(min_v.wrapping_add(v as u32));
            }
            if over.next().is_some() {
                return Err(Error::Format("u32 tail overflow".into()));
            }
            Ok(out)
        }
        m => Err(Error::Format(format!("unknown u32 seq mode {m}"))),
    }
}

/// The transforms a sequence may apply. Each hands what it leaves to a nested sequence, which
/// is allowed the rest.
const ALLOW_RECENCY: u8 = 1;
const ALLOW_GENERATOR: u8 = 2;

/// Encodes `i64` sequences choosing the smallest of:
/// `0 = Const`, `1 = Plain FOR`, `2 = GCD-scaled Delta-ZigZag`, `3 = GCD-scaled Delta-of-Delta`,
/// `5 = FOR byte planes`, `6 = delta byte planes`, `7 = generator positions` and
/// `8 = MTF-4096 recency`, whose new values are a sequence of their own (interleaved 64-bit
/// block and request ids). Mode `4`, recency with raw new values, is only read.
pub fn encode_i64_seq(vals: &[i64], backend: EntropyBackend, level: i32, out: &mut Vec<u8>) {
    encode_i64_seq_with(vals, backend, level, ALLOW_RECENCY | ALLOW_GENERATOR, out);
}

/// Returns the mode it chose.
fn encode_i64_seq_with(
    vals: &[i64],
    backend: EntropyBackend,
    level: i32,
    allow: u8,
    out: &mut Vec<u8>,
) -> u8 {
    put_uvarint(out, vals.len() as u64);
    if vals.is_empty() {
        return 0;
    }
    let first = vals[0];
    let mut min_v = first;
    let mut max_v = first;
    for &v in &vals[1..] {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if min_v == max_v {
        out.push(0);
        put_ivarint(out, first);
        return 0;
    }
    let span = (max_v as i128) - (min_v as i128);

    let mut best_mode = 2u8;
    let mut best_payload = Vec::new();

    // Candidates 8 and 7: MTF-4096 recency coding (repeated high-entropy 64-bit ids like HDFS
    // `blk_<i64>`), else generator positions; behind recency those replace its new values and
    // not every repeat. Generator outputs are noise to the candidates below, so a match is
    // final and skips them.
    if allow & ALLOW_RECENCY != 0
        && vals.len() >= 32
        && let Some((cand, new_values)) =
            try_encode_i64_mtf(vals, backend, level, allow & !ALLOW_RECENCY)
    {
        best_mode = 8;
        best_payload = cand;
        if new_values == 7 {
            out.push(best_mode);
            out.extend_from_slice(&best_payload);
            return best_mode;
        }
    } else if allow & ALLOW_GENERATOR != 0
        && vals.len() >= 4
        && span > (1i128 << 48)
        && let Some(cand) = lcg::encode(vals, backend, level)
    {
        out.push(7);
        out.extend_from_slice(&cand);
        return 7;
    }

    // Candidate 1: Plain FOR bitpack (if unsigned span fits u64 without wrapping ambiguity)
    if span >= 0 && span <= u64::MAX as i128 {
        let span_u = span as u64;
        let width = if span_u == u64::MAX {
            64
        } else {
            (64 - span_u.leading_zeros()) as u8
        };
        let shifted: Vec<u64> = vals
            .iter()
            .map(|&v| ((v as i128) - (min_v as i128)) as u64)
            .collect();
        let mut pk = Vec::new();
        bitpack_u64(&shifted, width, &mut pk);
        let mut cand = Vec::new();
        put_ivarint(&mut cand, min_v);
        cand.push(width);
        compress_leaf(&mut cand, &pk, backend, level);
        if best_payload.is_empty() || cand.len() < best_payload.len() {
            best_mode = 1;
            best_payload = cand;
        }
    }

    // Candidate 2: GCD-normalized First-Order Delta
    {
        let mut deltas = Vec::with_capacity(vals.len() - 1);
        let mut g = 0u64;
        for i in 1..vals.len() {
            let d = vals[i].wrapping_sub(vals[i - 1]);
            deltas.push(d);
            if d != 0 {
                g = gcd_u64(g, d.unsigned_abs());
            }
        }
        let scale = if g > 1 { g } else { 1 };
        let mut raw = Vec::with_capacity(vals.len() * 2);
        for &d in &deltas {
            let q = if scale > 1 && d != i64::MIN {
                d / (scale as i64)
            } else {
                d
            };
            // Verify exactness on i64::MIN edge case
            if q.wrapping_mul(scale as i64) != d {
                raw.clear();
                break;
            }
            put_ivarint(&mut raw, q);
        }
        if !raw.is_empty() || vals.len() == 1 {
            let mut cand = Vec::new();
            put_ivarint(&mut cand, first);
            put_uvarint(&mut cand, scale);
            compress_leaf(&mut cand, &raw, backend, level);
            if best_payload.is_empty() || cand.len() < best_payload.len() {
                best_mode = 2;
                best_payload = cand;
            }
        } else {
            let mut raw2 = Vec::with_capacity(vals.len() * 3);
            for &d in &deltas {
                put_ivarint(&mut raw2, d);
            }
            let mut cand = Vec::new();
            put_ivarint(&mut cand, first);
            put_uvarint(&mut cand, 1);
            compress_leaf(&mut cand, &raw2, backend, level);
            if best_payload.is_empty() || cand.len() < best_payload.len() {
                best_mode = 2;
                best_payload = cand;
            }
        }
    }

    // Candidate 3: GCD-normalized Delta-of-Delta (ideal for timestamps & counters)
    if vals.len() >= 4 {
        let d0 = vals[1].wrapping_sub(vals[0]);
        let mut dods = Vec::with_capacity(vals.len() - 2);
        let mut prev_d = d0;
        let mut g = 0u64;
        for i in 2..vals.len() {
            let d = vals[i].wrapping_sub(vals[i - 1]);
            let dd = d.wrapping_sub(prev_d);
            dods.push(dd);
            prev_d = d;
            if dd != 0 {
                g = gcd_u64(g, dd.unsigned_abs());
            }
        }
        let scale = if g > 1 && g <= (i64::MAX as u64) {
            g
        } else {
            1
        };
        let mut raw = Vec::with_capacity(vals.len() * 2);
        let mut exact = true;
        for &dd in &dods {
            let q = if scale > 1 && dd != i64::MIN {
                dd / (scale as i64)
            } else {
                dd
            };
            if q.wrapping_mul(scale as i64) != dd {
                exact = false;
                break;
            }
            put_ivarint(&mut raw, q);
        }
        if exact {
            let mut cand = Vec::new();
            put_ivarint(&mut cand, first);
            put_ivarint(&mut cand, d0);
            put_uvarint(&mut cand, scale);
            compress_leaf(&mut cand, &raw, backend, level);
            if best_payload.is_empty() || cand.len() < best_payload.len() {
                best_mode = 3;
                best_payload = cand;
            }
        }
    }

    // Candidate 5 (`PACK_PLANES`): Byte-Plane Transposed FOR
    if vals.len() >= 24 && span >= 0 && span <= u64::MAX as i128 {
        let span_u = span as u64;
        let bw = ((64 - span_u.leading_zeros()).div_ceil(8)).max(1) as usize;
        let n = vals.len();
        let mut planes = vec![0u8; n * bw];
        for (i, &v) in vals.iter().enumerate() {
            let u = ((v as i128) - (min_v as i128)) as u64;
            let le = u.to_le_bytes();
            for p in 0..bw {
                planes[p * n + i] = le[p];
            }
        }
        let mut cand = Vec::new();
        put_ivarint(&mut cand, min_v);
        cand.push(bw as u8);
        compress_leaf(&mut cand, &planes, backend, level);
        if cand.len() < best_payload.len() {
            best_mode = 5;
            best_payload = cand;
        }
    }

    // Candidate 6 (`PACK_DELTA | PACK_PLANES`): GCD-Scaled Delta + FOR + Byte-Plane Transpose
    if vals.len() >= 24 {
        let mut g = 0u64;
        for i in 1..vals.len() {
            let d = vals[i].wrapping_sub(vals[i - 1]);
            if d != 0 {
                g = gcd_u64(g, d.unsigned_abs());
            }
        }
        let scale = if g > 1 && g <= (i64::MAX as u64) {
            g
        } else {
            1
        };
        let mut qs = Vec::with_capacity(vals.len() - 1);
        let mut min_q = i64::MAX;
        let mut max_q = i64::MIN;
        let mut exact = true;
        for i in 1..vals.len() {
            let d = vals[i].wrapping_sub(vals[i - 1]);
            let q = if scale > 1 && d != i64::MIN {
                d / (scale as i64)
            } else {
                d
            };
            if q.wrapping_mul(scale as i64) != d {
                exact = false;
                break;
            }
            min_q = min_q.min(q);
            max_q = max_q.max(q);
            qs.push(q);
        }
        let dspan = (max_q as i128) - (min_q as i128);
        if exact && dspan >= 0 && dspan <= u64::MAX as i128 {
            let bw = ((64 - (dspan as u64).leading_zeros()).div_ceil(8)).max(1) as usize;
            let m = qs.len();
            let mut planes = vec![0u8; m * bw];
            for (i, &q) in qs.iter().enumerate() {
                let u = ((q as i128) - (min_q as i128)) as u64;
                let le = u.to_le_bytes();
                for p in 0..bw {
                    planes[p * m + i] = le[p];
                }
            }
            let mut cand = Vec::new();
            put_ivarint(&mut cand, first);
            put_uvarint(&mut cand, scale);
            put_ivarint(&mut cand, min_q);
            cand.push(bw as u8);
            compress_leaf(&mut cand, &planes, backend, level);
            if cand.len() < best_payload.len() {
                best_mode = 6;
                best_payload = cand;
            }
        }
    }

    out.push(best_mode);
    out.extend_from_slice(&best_payload);
    best_mode
}

pub fn decode_i64_seq(cur: &mut Cursor<'_>, budget: &DecodeBudget) -> Result<Vec<i64>> {
    decode_i64_seq_at(cur, budget, 0)
}

/// `depth` counts the sequences this one is nested in. Two deep is the limit, which bounds
/// the recursion on untrusted input.
fn decode_i64_seq_at(cur: &mut Cursor<'_>, budget: &DecodeBudget, depth: u8) -> Result<Vec<i64>> {
    let n = cur.uvar()? as usize;
    if n > budget.max_records_per_block * 4 {
        return Err(Error::BudgetExceeded("i64 sequence count"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    match cur.u8()? {
        0 => {
            let v = cur.ivar()?;
            budget.check_alloc(n * 8, "const i64 seq")?;
            Ok(vec![v; n])
        }
        1 => {
            let min_v = cur.ivar()?;
            let width = cur.u8()?;
            let pk = decompress_leaf(cur, budget)?;
            let shifted = bitunpack_u64(&pk, n, width, budget)?;
            Ok(shifted
                .into_iter()
                .map(|s| ((min_v as i128).wrapping_add(s as i128)) as i64)
                .collect())
        }
        2 => {
            let first = cur.ivar()?;
            let scale = cur.uvar()?;
            let raw = decompress_leaf(cur, budget)?;
            let mut rc = Cursor::new(&raw);
            budget.check_alloc(n * 8, "delta i64 seq")?;
            let mut out = Vec::with_capacity(n);
            out.push(first);
            let mut acc = first;
            for _ in 1..n {
                let d = rc.ivar()?.wrapping_mul(scale as i64);
                acc = acc.wrapping_add(d);
                out.push(acc);
            }
            Ok(out)
        }
        3 => {
            let first = cur.ivar()?;
            let mut d = cur.ivar()?;
            let scale = cur.uvar()?;
            let raw = decompress_leaf(cur, budget)?;
            let mut rc = Cursor::new(&raw);
            budget.check_alloc(n * 8, "dod i64 seq")?;
            let mut out = Vec::with_capacity(n);
            out.push(first);
            if n >= 2 {
                let second = first.wrapping_add(d);
                out.push(second);
                let mut acc = second;
                for _ in 2..n {
                    let dd = rc.ivar()?.wrapping_mul(scale as i64);
                    d = d.wrapping_add(dd);
                    acc = acc.wrapping_add(d);
                    out.push(acc);
                }
            }
            Ok(out)
        }
        4 => {
            let codes = decode_u32_seq(cur, budget)?;
            let m = cur.uvar()? as usize;
            if m > n {
                return Err(Error::Format("i64 MTF miss count > n".into()));
            }
            let planes = decompress_leaf(cur, budget)?;
            if planes.len() != m * 8 {
                return Err(Error::Format("i64 MTF miss byte length mismatch".into()));
            }
            let misses = (0..m).map(|i| {
                let mut b = [0u8; 8];
                for p in 0..8 {
                    b[p] = planes[p * m + i];
                }
                i64::from_le_bytes(b)
            });
            decode_i64_mtf(codes, misses, n, budget)
        }
        5 => {
            let min_v = cur.ivar()?;
            let bw = cur.u8()? as usize;
            if !(1..=8).contains(&bw) {
                return Err(Error::Format("invalid i64 plane width".into()));
            }
            let planes = decompress_leaf(cur, budget)?;
            if planes.len() != n.saturating_mul(bw) {
                return Err(Error::Format("i64 plane buffer mismatch".into()));
            }
            budget.check_alloc(n * 8, "i64 FOR planes")?;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut le = [0u8; 8];
                for p in 0..bw {
                    le[p] = planes[p * n + i];
                }
                let u = u64::from_le_bytes(le);
                out.push(((min_v as i128).wrapping_add(u as i128)) as i64);
            }
            Ok(out)
        }
        6 => {
            let first = cur.ivar()?;
            let scale = cur.uvar()? as i64;
            let min_q = cur.ivar()?;
            let bw = cur.u8()? as usize;
            if !(1..=8).contains(&bw) {
                return Err(Error::Format("invalid i64 delta plane width".into()));
            }
            let m = n.saturating_sub(1);
            let planes = decompress_leaf(cur, budget)?;
            if planes.len() != m.saturating_mul(bw) {
                return Err(Error::Format("i64 delta plane buffer mismatch".into()));
            }
            budget.check_alloc(n * 8, "i64 delta planes")?;
            let mut out = Vec::with_capacity(n);
            out.push(first);
            let mut acc = first;
            for i in 0..m {
                let mut le = [0u8; 8];
                for p in 0..bw {
                    le[p] = planes[p * m + i];
                }
                let u = u64::from_le_bytes(le);
                let q = ((min_q as i128).wrapping_add(u as i128)) as i64;
                acc = acc.wrapping_add(q.wrapping_mul(scale));
                out.push(acc);
            }
            Ok(out)
        }
        7 if depth < 2 => lcg::decode(cur, n, budget, depth),
        8 if depth < 2 => {
            let codes = decode_u32_seq(cur, budget)?;
            let misses = decode_i64_seq_at(cur, budget, depth + 1)?;
            decode_i64_mtf(codes, misses.into_iter(), n, budget)
        }
        m => Err(Error::Format(format!("unknown i64 seq mode {m}"))),
    }
}

/// The last `window` distinct values, most recent first. A new value costs O(1) and a repeat
/// its rank: columns of ids are mostly new values, and shifting the whole list for each was
/// an eighth of the time to read records back.
struct Recency<T> {
    buf: Vec<T>,
    head: usize,
    len: usize,
    window: usize,
}

impl<T: Copy + Default + PartialEq> Recency<T> {
    /// For a sequence of `n` values.
    fn new(window: usize, n: usize) -> Self {
        let size = 2 * window.min(n);
        Self {
            buf: vec![T::default(); size],
            head: size,
            len: 0,
            window,
        }
    }

    fn rank(&self, v: T) -> Option<usize> {
        self.buf[self.head..self.head + self.len]
            .iter()
            .position(|&x| x == v)
    }

    fn push(&mut self, v: T) {
        if self.len == self.window {
            self.len -= 1;
        }
        if self.head == 0 {
            let at = self.buf.len() - self.len;
            self.buf.copy_within(0..self.len, at);
            self.head = at;
        }
        self.head -= 1;
        self.buf[self.head] = v;
        self.len += 1;
    }

    /// Moves the value at `rank` to the front and returns it.
    fn hit(&mut self, rank: usize) -> Option<T> {
        if rank >= self.len {
            return None;
        }
        let run = &mut self.buf[self.head..=self.head + rank];
        run.rotate_right(1);
        Some(run[0])
    }
}

/// Bounded Move-to-Front (`W = 4096`) recency coder for `i64` identifier streams:
/// code `0` = miss (the next value of the nested `misses` sequence, inserted at the front);
/// code `1..=W` = hit at 1-based recency rank `r` (move entry `r-1` to front).
/// Returns the payload and the mode of the miss sequence.
fn try_encode_i64_mtf(
    vals: &[i64],
    backend: EntropyBackend,
    level: i32,
    allow: u8,
) -> Option<(Vec<u8>, u8)> {
    const W: usize = 4096;
    let mut ring = Recency::new(W, vals.len());
    let mut codes = Vec::<u32>::with_capacity(vals.len());
    let mut misses = Vec::<i64>::new();
    let mut hits = 0usize;
    // A value seen for the first time is a miss without a search of the window, which is
    // every value of a column of counters or timestamps.
    let mut seen = HashSet::new();
    for &v in vals {
        if !seen.insert(v)
            && let Some(pos) = ring.rank(v)
        {
            hits += 1;
            codes.push((pos + 1) as u32);
            ring.hit(pos);
        } else {
            codes.push(0);
            misses.push(v);
            ring.push(v);
        }
    }
    // Only pay MTF overhead if recency hits >= 20% of rows
    if hits * 5 < vals.len() {
        return None;
    }
    let mut out = Vec::new();
    encode_u32_seq(&codes, backend, level, &mut out);
    let mode = encode_i64_seq_with(&misses, backend, level, allow, &mut out);
    Some((out, mode))
}

fn decode_i64_mtf(
    codes: Vec<u32>,
    mut misses: impl Iterator<Item = i64>,
    n: usize,
    budget: &DecodeBudget,
) -> Result<Vec<i64>> {
    const W: usize = 4096;
    if codes.len() != n {
        return Err(Error::Format("i64 MTF code count mismatch".into()));
    }
    let mut ring = Recency::new(W, n);
    budget.check_alloc(n * 8, "i64 MTF output")?;
    let mut out = Vec::with_capacity(n);
    for c in codes {
        if c == 0 {
            let v = misses
                .next()
                .ok_or_else(|| Error::Format("i64 MTF miss underflow".into()))?;
            ring.push(v);
            out.push(v);
        } else {
            let v = ring
                .hit((c - 1) as usize)
                .ok_or_else(|| Error::Format("i64 MTF rank out of bounds".into()))?;
            out.push(v);
        }
    }
    if misses.next().is_some() {
        return Err(Error::Format("i64 MTF miss overflow".into()));
    }
    Ok(out)
}

// ============================================================================
// 3. Dictionary Table (Works on Single-Block <8 MiB AND Multi-Block, Front-Coded + FSST!)
// ============================================================================

const FSST_RESTART: usize = 32;

#[derive(Debug, Clone)]
pub struct DictTable {
    /// Unique byte strings in frequency-descending order (`0` = most frequent).
    pub entries: Vec<Vec<u8>>,
}

impl DictTable {
    pub fn encode(&self, backend: EntropyBackend, level: i32, out: &mut Vec<u8>) {
        put_uvarint(out, self.entries.len() as u64);
        if self.entries.is_empty() {
            return;
        }
        // Mode 0: Plain length-prefixed in frequency order + entropy leaf
        let mut plain_raw = Vec::new();
        let mut total_bytes = 0usize;
        for e in &self.entries {
            put_uvarint(&mut plain_raw, e.len() as u64);
            plain_raw.extend_from_slice(e);
            total_bytes += e.len();
        }
        let mut best_mode = 0u8;
        let mut best_buf = Vec::new();
        compress_leaf(&mut best_buf, &plain_raw, backend, level);

        // Mode 1 & Mode 2: Lexicographically sorted Front-Coding (with optional FSST!)
        if self.entries.len() >= 16 && total_bytes >= 256 {
            let mut lex_order: Vec<u32> = (0..self.entries.len() as u32).collect();
            lex_order
                .sort_unstable_by(|&a, &b| self.entries[a as usize].cmp(&self.entries[b as usize]));
            // `lex_to_freq[lex_idx] = freq_idx`
            let mut fc_raw = Vec::with_capacity(total_bytes);
            for (li, &fi) in lex_order.iter().enumerate() {
                let cur_s = self.entries[fi as usize].as_slice();
                let shared = if li % FSST_RESTART == 0 {
                    0
                } else {
                    let prev_s = self.entries[lex_order[li - 1] as usize].as_slice();
                    cur_s.iter().zip(prev_s).take_while(|(a, b)| a == b).count()
                };
                put_uvarint(&mut fc_raw, shared as u64);
                put_uvarint(&mut fc_raw, (cur_s.len() - shared) as u64);
                fc_raw.extend_from_slice(&cur_s[shared..]);
            }
            let mut mode1 = Vec::new();
            encode_u32_seq(&lex_order, backend, level, &mut mode1);
            compress_leaf(&mut mode1, &fc_raw, backend, level);
            if mode1.len() < best_buf.len() {
                best_mode = 1;
                best_buf = mode1;
            }

            // Mode 2: FSST symbol-compressed Front-Coded suffixes (great when >= 2 KB of URLs/IDs/paths)
            let avg_len = total_bytes / self.entries.len().max(1);
            if total_bytes >= 2048 && avg_len <= 512 {
                let mut suffixes = Vec::with_capacity(self.entries.len());
                let mut shared_lens = Vec::with_capacity(self.entries.len());
                for (li, &fi) in lex_order.iter().enumerate() {
                    let cur_s = self.entries[fi as usize].as_slice();
                    let shared = if li % FSST_RESTART == 0 {
                        0
                    } else {
                        let prev_s = self.entries[lex_order[li - 1] as usize].as_slice();
                        cur_s.iter().zip(prev_s).take_while(|(a, b)| a == b).count()
                    };
                    shared_lens.push(shared);
                    suffixes.push(&cur_s[shared..]);
                }
                let comp = fsst::Compressor::train(&suffixes);
                let mut mode2 = Vec::new();
                encode_u32_seq(&lex_order, backend, level, &mut mode2);
                let lens = comp.symbol_lengths();
                mode2.push(lens.len() as u8);
                mode2.extend_from_slice(lens);
                for (sym, &l) in comp.symbol_table().iter().zip(lens) {
                    mode2.extend_from_slice(&sym.to_u64().to_le_bytes()[..l as usize]);
                }
                let mut fsst_body = Vec::with_capacity(total_bytes / 2);
                for (suf, &sh) in suffixes.iter().zip(&shared_lens) {
                    put_uvarint(&mut fsst_body, sh as u64);
                    let codes = comp.compress(suf);
                    put_uvarint(&mut fsst_body, codes.len() as u64);
                    fsst_body.extend_from_slice(&codes);
                }
                // Light entropy pass on FSST stream (falls back to Raw automatically if already dense)
                compress_leaf(&mut mode2, &fsst_body, backend, level);
                if mode2.len() < best_buf.len() {
                    best_mode = 2;
                    best_buf = mode2;
                }
            }
        }

        // Mode 3 (`DICT_INTS`): When all dictionary entries are canonical integers, pack via `i64_seq`!
        if self.entries.len() >= 8 {
            let mut ints = Vec::with_capacity(self.entries.len());
            let mut all_canon_int = true;
            for e in &self.entries {
                if let Some((iv, 0)) = parse_exact_int(e) {
                    ints.push(iv);
                } else {
                    all_canon_int = false;
                    break;
                }
            }
            if all_canon_int {
                let mut mode3 = Vec::new();
                encode_i64_seq(&ints, backend, level, &mut mode3);
                if mode3.len() < best_buf.len() {
                    best_mode = 3;
                    best_buf = mode3;
                }
            }
        }

        out.push(best_mode);
        out.extend_from_slice(&best_buf);
    }

    pub fn decode(cur: &mut Cursor<'_>, budget: &DecodeBudget) -> Result<Self> {
        let count = cur.uvar()? as usize;
        if count > budget.max_records_per_block * 2 {
            return Err(Error::BudgetExceeded("dictionary entry count"));
        }
        if count == 0 {
            return Ok(Self {
                entries: Vec::new(),
            });
        }
        match cur.u8()? {
            0 => {
                let raw = decompress_leaf(cur, budget)?;
                let mut rc = Cursor::new(&raw);
                let mut entries = Vec::with_capacity(count);
                for _ in 0..count {
                    entries.push(rc.bytes()?.to_vec());
                }
                Ok(Self { entries })
            }
            1 => {
                let lex_order = decode_u32_seq(cur, budget)?;
                if lex_order.len() != count {
                    return Err(Error::Format("dict lex_order length mismatch".into()));
                }
                let fc_raw = decompress_leaf(cur, budget)?;
                let mut rc = Cursor::new(&fc_raw);
                let mut entries = vec![Vec::new(); count];
                let mut prev = Vec::<u8>::new();
                for (li, &fi) in lex_order.iter().enumerate() {
                    if (fi as usize) >= count {
                        return Err(Error::Format("dict freq index out of range".into()));
                    }
                    let shared = rc.uvar()? as usize;
                    let suf = rc.bytes()?;
                    if li % FSST_RESTART == 0 && shared != 0 {
                        return Err(Error::Format("non-zero shared prefix at restart".into()));
                    }
                    if shared > prev.len() {
                        return Err(Error::Format("front-coded shared > prev".into()));
                    }
                    prev.truncate(shared);
                    prev.extend_from_slice(suf);
                    entries[fi as usize] = prev.clone();
                }
                Ok(Self { entries })
            }
            2 => {
                let lex_order = decode_u32_seq(cur, budget)?;
                if lex_order.len() != count {
                    return Err(Error::Format("fsst dict lex_order mismatch".into()));
                }
                let n_syms = cur.u8()? as usize;
                let sym_lens = cur.slice(n_syms)?;
                let mut sym_tab = [[0u8; 8]; 256];
                let mut len_tab = [0u8; 256];
                for (i, &l) in sym_lens.iter().enumerate() {
                    if !(1..=8).contains(&l) {
                        return Err(Error::Format("invalid fsst symbol length".into()));
                    }
                    let sb = cur.slice(l as usize)?;
                    sym_tab[i][..l as usize].copy_from_slice(sb);
                    len_tab[i] = l;
                }
                let fsst_body = decompress_leaf(cur, budget)?;
                let mut rc = Cursor::new(&fsst_body);
                let mut entries = vec![Vec::new(); count];
                let mut prev = Vec::<u8>::new();
                for &fi in &lex_order {
                    if (fi as usize) >= count {
                        return Err(Error::Format("fsst dict freq index out of range".into()));
                    }
                    let shared = rc.uvar()? as usize;
                    if shared > prev.len() {
                        return Err(Error::Format("fsst front-coded shared > prev".into()));
                    }
                    prev.truncate(shared);
                    let codes = rc.bytes()?;
                    fsst_decode_into(&sym_tab, &len_tab, codes, &mut prev, budget)?;
                    entries[fi as usize] = prev.clone();
                }
                Ok(Self { entries })
            }
            3 => {
                let ints = decode_i64_seq(cur, budget)?;
                if ints.len() != count {
                    return Err(Error::Format("dict ints count mismatch".into()));
                }
                let mut entries = Vec::with_capacity(count);
                for iv in ints {
                    let mut v = Vec::with_capacity(20);
                    format_exact_int(iv, 0, &mut v);
                    entries.push(v);
                }
                Ok(Self { entries })
            }
            m => Err(Error::Format(format!("unknown dict table mode {m}"))),
        }
    }
}

fn fsst_decode_into(
    sym: &[[u8; 8]; 256],
    len: &[u8; 256],
    codes: &[u8],
    out: &mut Vec<u8>,
    budget: &DecodeBudget,
) -> Result<()> {
    budget.check_alloc(
        out.len().saturating_add(codes.len().saturating_mul(8)),
        "fsst entry",
    )?;
    let mut i = 0usize;
    while i < codes.len() {
        let c = codes[i];
        if c == 255 {
            let &lit = codes
                .get(i + 1)
                .ok_or_else(|| Error::Format("dangling fsst escape".into()))?;
            out.push(lit);
            i += 2;
        } else {
            let l = len[c as usize] as usize;
            if l == 0 {
                return Err(Error::Format(format!("unknown fsst code {c}")));
            }
            out.extend_from_slice(&sym[c as usize][..l]);
            i += 1;
        }
    }
    Ok(())
}

// ============================================================================
// 4. Typed & Observability Leaf Column Encoder / Decoder
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LeafCodecTag {
    Const = 0,
    Raw = 1,
    InlineDict = 2,
    SharedDictRef = 3,
    IntExact = 4,
    DecimalExact = 5,
    Timestamp = 6,
    Uuid = 7,
    Ulid = 8,
    Traceparent = 9,
    HexMtf = 10,
    Ipv4Port = 11,
    EmailSplit = 12,
    HierarchicalSplit = 13,
    FixedBinary = 14,
    QuotedOrNull = 15,
    HexId = 16,
    ProseSlice = 17,
    Base64 = 18,
}

pub(crate) fn encode_present_slice(
    vals: &[&[u8]],
    backend: EntropyBackend,
    level: i32,
    depth: u8,
) -> (u8, i64, i64, Vec<u8>) {
    debug_assert!(!vals.is_empty());
    let first = vals[0];
    if vals.iter().all(|&v| v == first) {
        let mut b = Vec::with_capacity(first.len() + 8);
        b.push(LeafCodecTag::Const as u8);
        put_uvarint(&mut b, first.len() as u64);
        b.extend_from_slice(first);
        let (nk, mn, mx) = if let Some((iv, _)) = parse_exact_int(first) {
            (1, iv, iv)
        } else if let Some((ts, _)) = parse_timestamp(first) {
            (2, ts, ts)
        } else {
            (0, 0, 0)
        };
        return (nk, mn, mx, b);
    }

    // Baseline Candidate: Raw (`u32_seq` lengths + `compress_leaf` payload)
    let mut best_body = encode_raw_present(vals, backend, level);
    let mut best_num = (0u8, 0i64, 0i64);

    // Quick sample check on first 8 rows before running full-column typed parsers
    let p0 = vals[0];

    // 1. Exact Integer (Canonical + Padded/Signed + MTF-4096 / FOR / GCD Delta / DoD)
    if parse_exact_int(p0).is_some() {
        let mut ints = Vec::with_capacity(vals.len());
        let mut fmts = Vec::with_capacity(vals.len());
        let mut mn = i64::MAX;
        let mut mx = i64::MIN;
        let mut ok = true;
        for &v in vals {
            if let Some((iv, f)) = parse_exact_int(v) {
                mn = mn.min(iv);
                mx = mx.max(iv);
                ints.push(iv);
                fmts.push(f as u32);
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            let mut cand = vec![LeafCodecTag::IntExact as u8];
            encode_i64_seq(&ints, backend, level, &mut cand);
            encode_u32_seq(&fmts, backend, level, &mut cand);
            if cand.len() < best_body.len() {
                best_body = cand;
                best_num = (1, mn, mx);
            }
        }
    }

    // 2. Exact Decimal Float
    if parse_exact_decimal(p0).is_some() {
        let mut mants = Vec::with_capacity(vals.len());
        let mut specs = Vec::with_capacity(vals.len());
        let mut ok = true;
        for &v in vals {
            if let Some((m, sp)) = parse_exact_decimal(v) {
                mants.push(m);
                specs.push(sp as u32);
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            let mut cand = vec![LeafCodecTag::DecimalExact as u8];
            encode_i64_seq(&mants, backend, level, &mut cand);
            encode_u32_seq(&specs, backend, level, &mut cand);
            if cand.len() < best_body.len() {
                best_body = cand;
                best_num = (0, 0, 0);
            }
        }
    }

    // 3. Canonical Timestamps (ISO-8601 / RFC3339 / Apache CLF / Syslog)
    if parse_timestamp(p0).is_some() {
        let mut ns_vec = Vec::with_capacity(vals.len());
        let mut layouts = Vec::with_capacity(vals.len());
        let mut mn = i64::MAX;
        let mut mx = i64::MIN;
        let mut ok = true;
        for &v in vals {
            if let Some((ns, lay)) = parse_timestamp(v) {
                mn = mn.min(ns);
                mx = mx.max(ns);
                ns_vec.push(ns);
                layouts.push(lay);
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            let mut cand = vec![LeafCodecTag::Timestamp as u8];
            encode_i64_seq(&ns_vec, backend, level, &mut cand);
            encode_u32_seq(&layouts, backend, level, &mut cand);
            if cand.len() < best_body.len() {
                best_body = cand;
                best_num = (2, mn, mx);
            }
        }
    }

    // 4. UUID (v4 & UUIDv7 timestamp-delta split + MTF-4096 recency!)
    if p0.len() == 36
        && parse_uuid(p0).is_some()
        && let Some(cand) = try_encode_uuids(vals, backend, level)
        && cand.len() < best_body.len()
    {
        best_body = cand;
        best_num = (0, 0, 0);
    }

    // 5. ULID (26-char Crockford Base32 -> 48b ms timestamp delta + 80b random)
    if p0.len() == 26
        && parse_ulid(p0).is_some()
        && let Some(cand) = try_encode_ulids(vals, backend, level)
        && cand.len() < best_body.len()
    {
        best_body = cand;
        best_num = (0, 0, 0);
    }

    // 6. W3C `traceparent` (55B -> MTF-4096 16B trace_id + 8B span_id + flags)
    if p0.len() == 55
        && parse_traceparent(p0).is_some()
        && let Some(cand) = try_encode_traceparents(vals, backend, level)
        && cand.len() < best_body.len()
    {
        best_body = cand;
        best_num = (0, 0, 0);
    }

    // 7. Fixed-Width Hex Identifiers: O(1) Unbounded Distance Backref (`HexId`) + MTF-4096 (`HexMtf`)
    if p0.len() >= 8
        && p0.len() <= 64
        && p0.len().is_multiple_of(2)
        && parse_hex_exact(p0).is_some()
    {
        if let Some(cand) = try_encode_hex_id(vals, backend, level)
            && cand.len() < best_body.len()
        {
            best_body = cand;
            best_num = (0, 0, 0);
        }
        if vals.len() <= 1024
            && let Some(cand) = try_encode_hex_mtf(vals, backend, level)
            && cand.len() < best_body.len()
        {
            best_body = cand;
            best_num = (0, 0, 0);
        }
    }

    // 8. IPv4 & `IPv4:port` (Split 4B IP MTF + u32 port stream!)
    if p0.len() >= 7
        && p0.len() <= 21
        && parse_ipv4_or_port(p0).is_some()
        && let Some(cand) = try_encode_ipv4_port(vals, backend, level)
        && cand.len() < best_body.len()
    {
        best_body = cand;
        best_num = (0, 0, 0);
    }

    // 9. Single- & Multi-Block Dictionary (Frequency-Sorted + First-Seen Order + Front-Coding + FSST + DICT_INTS!)
    if let Some(cand) = try_encode_inline_dict(vals, backend, level)
        && cand.len() < best_body.len()
    {
        best_body = cand;
        // Preserve numeric summary if all values were valid ints/timestamps even if dict won on tiny blocks!
    }

    // 10. Structured String Splitters (`local@domain` Email & `/` or `-` Hierarchical Stem/Suffix)
    if depth <= 1 && vals.len() >= 16 {
        if split_email(p0).is_some()
            && let Some(cand) = try_encode_email_split(vals, backend, level)
            && cand.len() < best_body.len()
        {
            best_body = cand;
            best_num = (0, 0, 0);
        }
        if split_hierarchical(p0).is_some()
            && let Some(cand) = try_encode_hierarchical_split(vals, backend, level)
            && cand.len() < best_body.len()
        {
            best_body = cand;
            best_num = (0, 0, 0);
        }
    }

    // 11. Fixed-Width Binary Field (Byte-Plane Transpose + IEEE-754 f64/f32 Order-Preserving Bit Delta!)
    let w = p0.len();
    if (2..=32).contains(&w)
        && vals.len() >= 16
        && vals.iter().all(|v| v.len() == w)
        && let Some(cand) = try_encode_fixed_binary(vals, w, backend, level)
        && cand.len() < best_body.len()
    {
        best_body = cand;
        best_num = (0, 0, 0);
    }

    // 12. Base64 (decoded bytes, which may in turn be a generator's)
    if p0.len() >= 16
        && let Some(cand) = try_encode_base64(vals, backend, level)
        && cand.len() < best_body.len()
    {
        best_body = cand;
        best_num = (0, 0, 0);
    }

    // 13. JSON Quoted-or-Null Wrapper (`"..."` lexemes + optional `null` literals -> unquoted typed codecs!)
    if depth == 0 && (p0.starts_with(b"\"") || p0 == b"null") {
        let mut null_count = 0usize;
        let mut str_count = 0usize;
        let mut all_q_or_null = true;
        for &v in vals {
            if v.len() >= 2 && v[0] == b'"' && v[v.len() - 1] == b'"' {
                str_count += 1;
            } else if v == b"null" {
                null_count += 1;
            } else {
                all_q_or_null = false;
                break;
            }
        }
        if all_q_or_null && str_count > 0 {
            if null_count == 0 {
                let stripped: Vec<&[u8]> = vals.iter().map(|v| &v[1..v.len() - 1]).collect();
                let (nk, mn, mx, inner) = encode_present_slice(&stripped, backend, level, 1);
                let mut cand = Vec::with_capacity(2 + inner.len());
                cand.push(LeafCodecTag::QuotedOrNull as u8);
                cand.push(0u8); // submode 0 = all quoted strings
                cand.extend_from_slice(&inner);
                if cand.len() < best_body.len() {
                    best_body = cand;
                    best_num = (nk, mn, mx);
                }
            } else {
                let mut is_str = Vec::with_capacity(vals.len());
                let mut stripped = Vec::with_capacity(str_count);
                for &v in vals {
                    if v.len() >= 2 && v[0] == b'"' {
                        is_str.push(1u32);
                        stripped.push(&v[1..v.len() - 1]);
                    } else {
                        is_str.push(0u32);
                    }
                }
                let (_, _, _, inner) = encode_present_slice(&stripped, backend, level, 1);
                let mut cand = Vec::with_capacity(8 + inner.len());
                cand.push(LeafCodecTag::QuotedOrNull as u8);
                cand.push(1u8); // submode 1 = mix of `"..."` and `null`
                encode_u32_seq(&is_str, backend, level, &mut cand);
                cand.extend_from_slice(&inner);
                if cand.len() < best_body.len() {
                    best_body = cand;
                    best_num = (0, 0, 0);
                }
            }
        }
    }

    (best_num.0, best_num.1, best_num.2, best_body)
}

fn encode_raw_present(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Vec<u8> {
    let mut out = vec![LeafCodecTag::Raw as u8];
    let lens: Vec<u32> = vals.iter().map(|v| v.len() as u32).collect();
    encode_u32_seq(&lens, backend, level, &mut out);
    let total: usize = vals.iter().map(|v| v.len()).sum();
    let mut concat = Vec::with_capacity(total);
    for &v in vals {
        concat.extend_from_slice(v);
    }
    compress_leaf(&mut out, &concat, backend, level);
    out
}

fn try_encode_inline_dict(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    // A column with more than a quarter of its cells distinct almost never ends up a dictionary,
    // and trying it is most of the cost of encoding text: on GitHub events, stopping here halves
    // encode time for 0.1% of size.
    let max_unique = (vals.len() / 4).clamp(4, 8192);
    let mut counts: HashMap<&[u8], (u32, u32)> = HashMap::with_capacity(vals.len().min(1024));
    let mut first_entries: Vec<Vec<u8>> = Vec::new();
    let mut first_ords: Vec<u32> = Vec::with_capacity(vals.len());
    for &v in vals {
        if let Some(entry) = counts.get_mut(v) {
            entry.0 += 1;
            first_ords.push(entry.1);
        } else {
            if counts.len() >= max_unique {
                return None;
            }
            let fid = first_entries.len() as u32;
            counts.insert(v, (1, fid));
            first_entries.push(v.to_vec());
            first_ords.push(fid);
        }
    }

    // 1. Frequency-descending dictionary + static or MTF-4096 ordinals
    let mut uniq: Vec<(&[u8], u32)> = counts.into_iter().map(|(k, (c, _))| (k, c)).collect();
    uniq.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let mut id_map = HashMap::with_capacity(uniq.len());
    let mut entries = Vec::with_capacity(uniq.len());
    for (i, (bytes, _)) in uniq.into_iter().enumerate() {
        id_map.insert(bytes, i as u32);
        entries.push(bytes.to_vec());
    }
    let ords: Vec<u32> = vals.iter().map(|v| id_map[v]).collect();

    let table = DictTable { entries };
    let mut dict_bytes = Vec::new();
    table.encode(backend, level, &mut dict_bytes);

    let mut ord_static = Vec::new();
    ord_static.push(0u8); // submode 0 = direct ordinals
    encode_u32_seq(&ords, backend, level, &mut ord_static);

    let mut best_ord = ord_static;
    if table.entries.len() >= 16
        && vals.len() >= 64
        && let Some(mtf_ranks) = mtf_transform_u32(&ords, 4096)
    {
        let mut ord_mtf = Vec::new();
        ord_mtf.push(1u8); // submode 1 = MTF-4096 transformed ordinals
        encode_u32_seq(&mtf_ranks, backend, level, &mut ord_mtf);
        if ord_mtf.len() < best_ord.len() {
            best_ord = ord_mtf;
        }
    }

    let mut best_total = Vec::with_capacity(1 + dict_bytes.len() + best_ord.len());
    best_total.push(LeafCodecTag::InlineDict as u8);
    best_total.extend_from_slice(&dict_bytes);
    best_total.extend_from_slice(&best_ord);

    // 2. First-seen entry order trial (monotonically increasing new IDs -> crushed by Mode 4 ZigZag delta!)
    if first_entries.len() >= 8 && vals.len() >= 32 {
        let fs_table = DictTable {
            entries: first_entries,
        };
        let mut fs_dict = Vec::new();
        fs_table.encode(backend, level, &mut fs_dict);
        if 1 + fs_dict.len() < best_total.len() {
            let mut fs_ord = vec![0u8];
            encode_u32_seq(&first_ords, backend, level, &mut fs_ord);
            if 1 + fs_dict.len() + fs_ord.len() < best_total.len() {
                best_total.clear();
                best_total.push(LeafCodecTag::InlineDict as u8);
                best_total.extend_from_slice(&fs_dict);
                best_total.extend_from_slice(&fs_ord);
            }
        }
    }

    Some(best_total)
}

/// Reversible Move-to-Front transform over `0..vocab_size` ordinals with bounded window `W`:
/// `rank == 0..W` if found in `ring` at `rank` (encoded as `rank * 2 + 1`),
/// or literal `ord` when not in `ring` (encoded as `ord * 2`, inserting `ord` at front).
///
/// `None` when recency is no cheap predictor: few of the first ordinals repeat one of the 64
/// before them, or ranks take over 64 steps to find on average.
fn mtf_transform_u32(ords: &[u32], window: usize) -> Option<Vec<u32>> {
    let head = &ords[..ords.len().min(4096)];
    let near = |i: &usize| head[i.saturating_sub(64)..*i].contains(&head[*i]);
    if (1..head.len()).filter(near).count() * 8 < head.len() {
        return None;
    }
    let mut ring = Recency::new(window, ords.len());
    let mut out = Vec::with_capacity(ords.len());
    let mut steps = 0;
    for (i, &o) in ords.iter().enumerate() {
        if i % 4096 == 4095 && steps > 64 * i {
            return None;
        }
        if let Some(pos) = ring.rank(o) {
            out.push((pos as u32) * 2 + 1);
            ring.hit(pos);
            steps += pos;
        } else {
            out.push(o.saturating_mul(2));
            ring.push(o);
            steps += ring.len;
        }
    }
    Some(out)
}

fn mtf_untransform_u32(codes: &[u32], window: usize, vocab: usize) -> Result<Vec<u32>> {
    let mut ring = Recency::new(window, codes.len());
    let mut out = Vec::with_capacity(codes.len());
    for &c in codes {
        if (c & 1) == 1 {
            let o = ring
                .hit((c >> 1) as usize)
                .ok_or_else(|| Error::Format("dict MTF rank out of range".into()))?;
            out.push(o);
        } else {
            let o = c >> 1;
            if (o as usize) >= vocab {
                return Err(Error::Format(
                    "dict MTF literal ordinal out of range".into(),
                ));
            }
            ring.push(o);
            out.push(o);
        }
    }
    Ok(out)
}

/// Fixed-width byte-slice MTF-4096 helper (`W` bytes per item, e.g. 16B trace_id or hex IDs).
fn encode_fixed_bytes_mtf(
    items: &[&[u8]],
    width: usize,
    backend: EntropyBackend,
    level: i32,
    out: &mut Vec<u8>,
) {
    const WIN: usize = 4096;
    let mut ring = Vec::<Vec<u8>>::with_capacity(256);
    let mut codes = Vec::<u32>::with_capacity(items.len());
    let mut misses = Vec::<u8>::new();
    for &it in items {
        if let Some(pos) = ring.iter().position(|x| x.as_slice() == it) {
            codes.push((pos + 1) as u32);
            if pos > 0 {
                let v = ring.remove(pos);
                ring.insert(0, v);
            }
        } else {
            codes.push(0);
            misses.extend_from_slice(it);
            if ring.len() == WIN {
                ring.pop();
            }
            ring.insert(0, it.to_vec());
        }
    }
    put_uvarint(out, width as u64);
    encode_u32_seq(&codes, backend, level, out);
    put_uvarint(out, (misses.len() / width.max(1)) as u64);
    mt::encode_leaf(
        out,
        &misses,
        width,
        &[mt::Layout::Le, mt::Layout::Be],
        backend,
        level,
    );
}

/// Decodes `n` rows of one byte width, returned back to back.
fn decode_fixed_bytes_mtf(
    cur: &mut Cursor<'_>,
    n: usize,
    budget: &DecodeBudget,
) -> Result<(usize, Vec<u8>)> {
    const WIN: usize = 4096;
    let width = cur.uvar()? as usize;
    if width == 0 || width > 64 {
        return Err(Error::Format("invalid fixed MTF byte width".into()));
    }
    let codes = decode_u32_seq(cur, budget)?;
    if codes.len() != n {
        return Err(Error::Format("fixed MTF code count mismatch".into()));
    }
    let n_miss = cur.uvar()? as usize;
    let misses = mt::decode_leaf(cur, budget)?;
    if misses.len() != n_miss.saturating_mul(width) {
        return Err(Error::Format("fixed MTF misses length mismatch".into()));
    }
    // The recency ring holds, most recent first, a row index that carries each value.
    let mut ring = Recency::new(WIN, n);
    let mut mi = 0usize;
    budget.check_alloc(n.saturating_mul(width), "fixed MTF output")?;
    let mut out = Vec::with_capacity(n * width);
    for (i, c) in codes.into_iter().enumerate() {
        if c == 0 {
            if mi >= n_miss {
                return Err(Error::Format("fixed MTF miss underflow".into()));
            }
            out.extend_from_slice(&misses[mi * width..(mi + 1) * width]);
            mi += 1;
            ring.push(i as u32);
        } else {
            let src = ring
                .hit((c - 1) as usize)
                .ok_or_else(|| Error::Format("fixed MTF rank out of range".into()))?;
            out.extend_from_within(src as usize * width..(src as usize + 1) * width);
        }
    }
    Ok((width, out))
}

fn try_encode_uuids(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let mut uppers = Vec::with_capacity(vals.len());
    let mut raw16 = Vec::with_capacity(vals.len());
    let mut all_v7 = true;
    for &v in vals {
        let (up, b) = parse_uuid(v)?;
        if (b[6] >> 4) != 7 {
            all_v7 = false;
        }
        uppers.push(u32::from(up));
        raw16.push(b);
    }
    let mut out = vec![LeafCodecTag::Uuid as u8];
    encode_u32_seq(&uppers, backend, level, &mut out);
    if all_v7 {
        out.push(1); // UUIDv7: split 48-bit ms timestamp (delta/DoD) + 10B suffix!
        let mut ts_ms = Vec::with_capacity(vals.len());
        let mut tails = Vec::with_capacity(vals.len() * 10);
        for b in &raw16 {
            let mut t8 = [0u8; 8];
            t8[2..8].copy_from_slice(&b[0..6]);
            ts_ms.push(u64::from_be_bytes(t8) as i64);
            tails.extend_from_slice(&b[6..16]);
        }
        encode_i64_seq(&ts_ms, backend, level, &mut out);
        mt::encode_leaf(&mut out, &tails, 10, &[mt::Layout::Uuid7], backend, level);
    } else {
        out.push(0); // General UUID + MTF-4096 recency!
        let refs: Vec<&[u8]> = raw16.iter().map(|b| b.as_slice()).collect();
        encode_fixed_bytes_mtf(&refs, 16, backend, level, &mut out);
    }
    Some(out)
}

fn try_encode_ulids(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let mut ts_vec = Vec::with_capacity(vals.len());
    let mut rand_bytes = Vec::with_capacity(vals.len() * 10);
    for &v in vals {
        let (ts, r10) = parse_ulid(v)?;
        ts_vec.push(ts as i64);
        rand_bytes.extend_from_slice(&r10);
    }
    let mut out = vec![LeafCodecTag::Ulid as u8];
    encode_i64_seq(&ts_vec, backend, level, &mut out);
    compress_leaf(&mut out, &rand_bytes, backend, level);
    Some(out)
}

fn try_encode_traceparents(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let mut tids = Vec::with_capacity(vals.len());
    let mut sids = Vec::with_capacity(vals.len() * 8);
    let mut flags = Vec::with_capacity(vals.len());
    for &v in vals {
        let (t, s, f) = parse_traceparent(v)?;
        tids.push(t);
        sids.extend_from_slice(&s);
        flags.push(f as u32);
    }
    let mut out = vec![LeafCodecTag::Traceparent as u8];
    let t_refs: Vec<&[u8]> = tids.iter().map(|t| t.as_slice()).collect();
    encode_fixed_bytes_mtf(&t_refs, 16, backend, level, &mut out);
    mt::encode_leaf(
        &mut out,
        &sids,
        8,
        &[mt::Layout::Le, mt::Layout::Be],
        backend,
        level,
    );
    encode_u32_seq(&flags, backend, level, &mut out);
    Some(out)
}

fn try_encode_hex_id(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let hex_len = vals[0].len();
    let byte_w = hex_len / 2;
    let mut uppers = Vec::with_capacity(vals.len());
    let mut codes = Vec::with_capacity(vals.len());
    let mut misses = Vec::new();
    let mut last_seen: HashMap<[u8; 32], usize> = HashMap::with_capacity(vals.len().min(2048));
    for (i, &v) in vals.iter().enumerate() {
        if v.len() != hex_len {
            return None;
        }
        let (up, b, bw) = parse_hex_exact(v)?;
        debug_assert_eq!(bw as usize, byte_w);
        uppers.push(u32::from(up));
        if let Some(&prev) = last_seen.get(&b) {
            codes.push((i - prev) as u32);
        } else {
            codes.push(0);
            misses.extend_from_slice(&b[..byte_w]);
        }
        last_seen.insert(b, i);
    }
    let mut out = vec![LeafCodecTag::HexId as u8];
    put_uvarint(&mut out, byte_w as u64);
    encode_u32_seq(&uppers, backend, level, &mut out);
    encode_u32_seq(&codes, backend, level, &mut out);
    put_uvarint(&mut out, (misses.len() / byte_w.max(1)) as u64);
    mt::encode_leaf(
        &mut out,
        &misses,
        byte_w,
        &[mt::Layout::Le, mt::Layout::Be],
        backend,
        level,
    );
    Some(out)
}

fn try_encode_hex_mtf(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let hex_len = vals[0].len();
    let byte_w = hex_len / 2;
    let mut uppers = Vec::with_capacity(vals.len());
    let mut bins = Vec::with_capacity(vals.len());
    for &v in vals {
        if v.len() != hex_len {
            return None;
        }
        let (up, b, bw) = parse_hex_exact(v)?;
        debug_assert_eq!(bw as usize, byte_w);
        uppers.push(u32::from(up));
        bins.push(b[..byte_w].to_vec());
    }
    let mut out = vec![LeafCodecTag::HexMtf as u8];
    encode_u32_seq(&uppers, backend, level, &mut out);
    let refs: Vec<&[u8]> = bins.iter().map(|b| b.as_slice()).collect();
    encode_fixed_bytes_mtf(&refs, byte_w, backend, level, &mut out);
    Some(out)
}

fn try_encode_base64(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let mut lens = Vec::with_capacity(vals.len());
    let mut bytes = Vec::new();
    for &v in vals {
        let at = bytes.len();
        parse_base64(v, &mut bytes)?;
        lens.push((bytes.len() - at) as u32);
    }
    let mut out = vec![LeafCodecTag::Base64 as u8];
    encode_u32_seq(&lens, backend, level, &mut out);
    mt::encode_leaf(&mut out, &bytes, 4, &[mt::Layout::Le], backend, level);
    Some(out)
}

fn try_encode_ipv4_port(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let mut ips = Vec::with_capacity(vals.len());
    let mut ports = Vec::with_capacity(vals.len());
    for &v in vals {
        let (ip, port) = parse_ipv4_or_port(v)?;
        ips.push(ip);
        ports.push(port as u32);
    }
    let mut out = vec![LeafCodecTag::Ipv4Port as u8];
    let refs: Vec<&[u8]> = ips.iter().map(|ip| ip.as_slice()).collect();
    encode_fixed_bytes_mtf(&refs, 4, backend, level, &mut out);
    encode_u32_seq(&ports, backend, level, &mut out);
    Some(out)
}

fn try_encode_email_split(vals: &[&[u8]], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let mut locals = Vec::with_capacity(vals.len());
    let mut domains = Vec::with_capacity(vals.len());
    for &v in vals {
        let (l, d) = split_email(v)?;
        locals.push(l);
        domains.push(d);
    }
    let mut out = vec![LeafCodecTag::EmailSplit as u8];
    let (_, _, _, dom_enc) = encode_present_slice(&domains, backend, level, 2);
    let (_, _, _, loc_enc) = encode_present_slice(&locals, backend, level, 2);
    put_uvarint(&mut out, dom_enc.len() as u64);
    out.extend_from_slice(&dom_enc);
    put_uvarint(&mut out, loc_enc.len() as u64);
    out.extend_from_slice(&loc_enc);
    Some(out)
}

fn try_encode_hierarchical_split(
    vals: &[&[u8]],
    backend: EntropyBackend,
    level: i32,
) -> Option<Vec<u8>> {
    let mut prefixes = Vec::with_capacity(vals.len());
    let mut suffixes = Vec::with_capacity(vals.len());
    for &v in vals {
        let (p, s) = split_hierarchical(v)?;
        prefixes.push(p);
        suffixes.push(s);
    }
    // Require meaningful prefix deduplication before splitting
    let mut pset = HashMap::<&[u8], usize>::new();
    for &p in &prefixes {
        *pset.entry(p).or_insert(0) += 1;
        if pset.len() * 2 > vals.len() {
            return None;
        }
    }
    let mut out = vec![LeafCodecTag::HierarchicalSplit as u8];
    let (_, _, _, pre_enc) = encode_present_slice(&prefixes, backend, level, 2);
    let (_, _, _, suf_enc) = encode_present_slice(&suffixes, backend, level, 2);
    put_uvarint(&mut out, pre_enc.len() as u64);
    out.extend_from_slice(&pre_enc);
    put_uvarint(&mut out, suf_enc.len() as u64);
    out.extend_from_slice(&suf_enc);
    Some(out)
}

/// Order-preserving IEEE-754 bit mapping + byte-plane shuffle for fixed-width binary columns (`sao`!).
#[inline]
fn f64_bits_to_ordered(u: u64) -> u64 {
    if (u >> 63) != 0 { !u } else { u ^ (1u64 << 63) }
}

#[inline]
fn ordered_to_f64_bits(o: u64) -> u64 {
    if (o >> 63) != 0 { o ^ (1u64 << 63) } else { !o }
}

fn try_encode_fixed_binary(
    vals: &[&[u8]],
    width: usize,
    backend: EntropyBackend,
    level: i32,
) -> Option<Vec<u8>> {
    let n = vals.len();
    // Submode 0: Byte-plane transpose (`ByteShuffle`)
    let mut shuffled = vec![0u8; n * width];
    for (r, &v) in vals.iter().enumerate() {
        for b in 0..width {
            shuffled[b * n + r] = v[b];
        }
    }
    let mut best_sub = 0u8;
    let mut best_payload = Vec::new();
    compress_leaf(&mut best_payload, &shuffled, backend, level);

    // Submode 1: If 8-byte IEEE-754 f64 (like `sao` RA/Dec!), test ordered integer-bit delta + byte-shuffled delta!
    if width == 8 {
        let mut delta_shuf = vec![0u8; n * 8];
        let mut prev = 0u64;
        for (r, &v) in vals.iter().enumerate() {
            let cur = f64_bits_to_ordered(u64::from_le_bytes(v.try_into().unwrap()));
            let d = zigzag_u64(cur.wrapping_sub(prev));
            prev = cur;
            let db = d.to_le_bytes();
            for b in 0..8 {
                delta_shuf[b * n + r] = db[b];
            }
        }
        let mut cand = Vec::new();
        compress_leaf(&mut cand, &delta_shuf, backend, level);
        if cand.len() < best_payload.len() {
            best_sub = 1;
            best_payload = cand;
        }
        // Submode 3: doubles that step by draws of a generator.
        let rows = vals.concat();
        let mut cand = Vec::new();
        if let Some(walk) = mt::Layout::walk(&rows)
            && mt::try_leaf(&mut cand, &rows, 8, &[walk], backend, level)
            && cand.len() < best_payload.len()
        {
            best_sub = 3;
            best_payload = cand;
        }
    } else if width == 4 {
        let mut delta_shuf = vec![0u8; n * 4];
        let mut prev = 0u32;
        for (r, &v) in vals.iter().enumerate() {
            let raw_u = u32::from_le_bytes(v.try_into().unwrap());
            let cur = if (raw_u >> 31) != 0 {
                !raw_u
            } else {
                raw_u ^ (1u32 << 31)
            };
            let d_s = (cur.wrapping_sub(prev)) as i32;
            let dz = ((d_s << 1) ^ (d_s >> 31)) as u32;
            prev = cur;
            let db = dz.to_le_bytes();
            for b in 0..4 {
                delta_shuf[b * n + r] = db[b];
            }
        }
        let mut cand = Vec::new();
        compress_leaf(&mut cand, &delta_shuf, backend, level);
        if cand.len() < best_payload.len() {
            best_sub = 2;
            best_payload = cand;
        }
    }

    let mut out = vec![LeafCodecTag::FixedBinary as u8];
    put_uvarint(&mut out, width as u64);
    out.push(best_sub);
    out.extend_from_slice(&best_payload);
    Some(out)
}

#[inline]
fn zigzag_u64(u: u64) -> u64 {
    let s = u as i64;
    ((s << 1) ^ (s >> 63)) as u64
}

#[inline]
fn unzigzag_u64(z: u64) -> u64 {
    (((z >> 1) as i64) ^ (-((z & 1) as i64))) as u64
}

fn not_loaded() -> Error {
    Error::NotLoaded
}

/// The tables several columns of a block share, decoded on first use. A part is `None` when the
/// reader did not fetch it.
pub(crate) struct SharedPools<'a> {
    pub dict_blobs: Vec<Option<&'a [u8]>>,
    pub dicts: Vec<Option<Arc<DictTable>>>,
    pub pool_leaf: Option<&'a [u8]>,
    pub pool: Option<Arc<Vec<u8>>>,
}

impl<'a> SharedPools<'a> {
    pub fn new(dict_blobs: Vec<Option<&'a [u8]>>, pool_leaf: Option<&'a [u8]>) -> Self {
        let dicts = vec![None; dict_blobs.len()];
        Self {
            dict_blobs,
            dicts,
            pool_leaf,
            pool: None,
        }
    }

    pub fn get_dict(&mut self, id: usize, budget: &DecodeBudget) -> Result<Arc<DictTable>> {
        let blob = *self
            .dict_blobs
            .get(id)
            .ok_or_else(|| Error::Format("shared dict_id out of bounds".into()))?;
        if self.dicts[id].is_none() {
            let dt = DictTable::decode(&mut Cursor::new(blob.ok_or_else(not_loaded)?), budget)?;
            self.dicts[id] = Some(Arc::new(dt));
        }
        Ok(self.dicts[id].clone().unwrap())
    }

    pub fn get_prose(&mut self, budget: &DecodeBudget) -> Result<Arc<Vec<u8>>> {
        if self.pool.is_none() {
            let leaf = self.pool_leaf.ok_or_else(not_loaded)?;
            self.pool = Some(Arc::new(decompress_leaf(&mut Cursor::new(leaf), budget)?));
        }
        Ok(self.pool.clone().unwrap())
    }
}

pub(crate) fn decode_present_slice(
    cur: &mut Cursor<'_>,
    n: usize,
    budget: &DecodeBudget,
) -> Result<Col> {
    decode_present_slice_ctx(cur, n, None, budget)
}

fn dict_col(dict: Arc<DictTable>, sub: u8, raw_ords: Vec<u32>, n: usize) -> Result<Col> {
    if raw_ords.len() != n {
        return Err(Error::Format("dict ordinal count mismatch".into()));
    }
    let ords = if sub == 1 {
        mtf_untransform_u32(&raw_ords, 4096, dict.entries.len())?
    } else {
        raw_ords
    };
    if ords.iter().any(|&o| o as usize >= dict.entries.len()) {
        return Err(Error::Format("dict ordinal out of bounds".into()));
    }
    Ok(Col::Dict { dict, ords })
}

/// Cells of one byte width, formatted by `fmt` from `n` fixed-size inputs.
fn formatted(
    n: usize,
    width: usize,
    mut fmt: impl FnMut(usize, &mut Vec<u8>) -> Result<()>,
) -> Result<Col> {
    let mut out = Cells::with_capacity(n, n * width);
    for i in 0..n {
        fmt(i, out.tail())?;
        out.seal();
    }
    Ok(Col::Flat(out))
}

pub(crate) fn decode_present_slice_ctx(
    cur: &mut Cursor<'_>,
    n: usize,
    mut pools: Option<&mut SharedPools<'_>>,
    budget: &DecodeBudget,
) -> Result<Col> {
    if n == 0 {
        return Ok(Col::Flat(Cells::default()));
    }
    match cur.u8()? {
        0 => Ok(Col::Const {
            val: cur.bytes()?.to_vec(),
            n,
        }),
        1 => {
            let lens = decode_u32_seq(cur, budget)?;
            if lens.len() != n {
                return Err(Error::Format("raw col length count mismatch".into()));
            }
            Ok(Col::Flat(Cells::from_lens(
                decompress_leaf(cur, budget)?,
                &lens,
            )?))
        }
        2 => {
            let dict = DictTable::decode(cur, budget)?;
            let sub = cur.u8()?;
            dict_col(Arc::new(dict), sub, decode_u32_seq(cur, budget)?, n)
        }
        3 => {
            let dict_id = cur.uvar()? as usize;
            let sub = cur.u8()?;
            let raw_ords = decode_u32_seq(cur, budget)?;
            let p = pools
                .as_deref_mut()
                .ok_or_else(|| Error::Format("SharedDictRef without block pools".into()))?;
            dict_col(p.get_dict(dict_id, budget)?, sub, raw_ords, n)
        }
        4 => {
            let vals = decode_i64_seq(cur, budget)?;
            let fmts = decode_u32_seq(cur, budget)?;
            if vals.len() != n || fmts.len() != n {
                return Err(Error::Format("IntExact length mismatch".into()));
            }
            Ok(Col::Int { vals, fmts })
        }
        5 => {
            let mants = decode_i64_seq(cur, budget)?;
            let specs = decode_u32_seq(cur, budget)?;
            if mants.len() != n || specs.len() != n {
                return Err(Error::Format("DecimalExact length mismatch".into()));
            }
            formatted(n, 12, |i, out| {
                format_exact_decimal(mants[i], specs[i] as u16, out);
                Ok(())
            })
        }
        6 => {
            let ns = decode_i64_seq(cur, budget)?;
            let layouts = decode_u32_seq(cur, budget)?;
            if ns.len() != n || layouts.len() != n {
                return Err(Error::Format("Timestamp length mismatch".into()));
            }
            Ok(Col::Ts { ns, layouts })
        }
        7 => {
            let uppers = decode_u32_seq(cur, budget)?;
            if uppers.len() != n {
                return Err(Error::Format("UUID upper flag mismatch".into()));
            }
            if cur.u8()? == 1 {
                let ts_ms = decode_i64_seq(cur, budget)?;
                let tails = mt::decode_leaf(cur, budget)?;
                if ts_ms.len() != n || tails.len() != n * 10 {
                    return Err(Error::Format("UUIDv7 length mismatch".into()));
                }
                formatted(n, 36, |i, out| {
                    let mut b = [0u8; 16];
                    b[0..6].copy_from_slice(&(ts_ms[i] as u64).to_be_bytes()[2..8]);
                    b[6..16].copy_from_slice(&tails[i * 10..(i + 1) * 10]);
                    format_uuid(&b, uppers[i] != 0, out);
                    Ok(())
                })
            } else {
                let (w, raw) = decode_fixed_bytes_mtf(cur, n, budget)?;
                if w != 16 {
                    return Err(Error::Format("UUID MTF width != 16".into()));
                }
                formatted(n, 36, |i, out| {
                    let b: [u8; 16] = raw[i * 16..(i + 1) * 16].try_into().unwrap();
                    format_uuid(&b, uppers[i] != 0, out);
                    Ok(())
                })
            }
        }
        8 => {
            let ts_vec = decode_i64_seq(cur, budget)?;
            let rands = decompress_leaf(cur, budget)?;
            if ts_vec.len() != n || rands.len() != n * 10 {
                return Err(Error::Format("ULID length mismatch".into()));
            }
            formatted(n, 26, |i, out| {
                let r10: [u8; 10] = rands[i * 10..(i + 1) * 10].try_into().unwrap();
                format_ulid(ts_vec[i] as u64, &r10, out);
                Ok(())
            })
        }
        9 => {
            let (w, tids) = decode_fixed_bytes_mtf(cur, n, budget)?;
            let sids = mt::decode_leaf(cur, budget)?;
            let flags = decode_u32_seq(cur, budget)?;
            if w != 16 || sids.len() != n * 8 || flags.len() != n {
                return Err(Error::Format("traceparent length mismatch".into()));
            }
            formatted(n, 55, |i, out| {
                let tid: [u8; 16] = tids[i * 16..(i + 1) * 16].try_into().unwrap();
                let sid: [u8; 8] = sids[i * 8..(i + 1) * 8].try_into().unwrap();
                format_traceparent(&tid, &sid, flags[i] as u8, out);
                Ok(())
            })
        }
        10 => {
            let uppers = decode_u32_seq(cur, budget)?;
            let (w, bins) = decode_fixed_bytes_mtf(cur, n, budget)?;
            if uppers.len() != n {
                return Err(Error::Format("HexMtf length mismatch".into()));
            }
            formatted(n, w * 2, |i, out| {
                format_hex_bytes(&bins[i * w..(i + 1) * w], uppers[i] != 0, out);
                Ok(())
            })
        }
        11 => {
            let (w, ips) = decode_fixed_bytes_mtf(cur, n, budget)?;
            let ports = decode_u32_seq(cur, budget)?;
            if w != 4 || ports.len() != n {
                return Err(Error::Format("Ipv4Port width/length mismatch".into()));
            }
            formatted(n, 21, |i, out| {
                if ports[i] > 65535 {
                    return Err(Error::Format("Ipv4Port port > 65535".into()));
                }
                let ip: [u8; 4] = ips[i * 4..(i + 1) * 4].try_into().unwrap();
                format_ipv4_or_port(ip, ports[i] as u16, out);
                Ok(())
            })
        }
        12 => {
            let dom_bytes = cur.bytes()?;
            let loc_bytes = cur.bytes()?;
            let doms = decode_present_slice_ctx(&mut Cursor::new(dom_bytes), n, None, budget)?;
            let locs = decode_present_slice_ctx(&mut Cursor::new(loc_bytes), n, None, budget)?;
            formatted(n, 24, |i, out| {
                locs.write(i, out)?;
                out.push(b'@');
                doms.write(i, out)
            })
        }
        13 => {
            let pre_bytes = cur.bytes()?;
            let suf_bytes = cur.bytes()?;
            let pres = decode_present_slice_ctx(&mut Cursor::new(pre_bytes), n, None, budget)?;
            let sufs = decode_present_slice_ctx(&mut Cursor::new(suf_bytes), n, None, budget)?;
            formatted(n, 24, |i, out| {
                pres.write(i, out)?;
                sufs.write(i, out)
            })
        }
        14 => {
            let width = cur.uvar()? as usize;
            let sub = cur.u8()?;
            let shuf = match sub {
                3 => mt::decode_leaf(cur, budget)?,
                _ => decompress_leaf(cur, budget)?,
            };
            if width == 0 || width > 64 || shuf.len() != n.saturating_mul(width) {
                return Err(Error::Format("FixedBinary length mismatch".into()));
            }
            match sub {
                3 => formatted(n, width, |r, out| {
                    out.extend_from_slice(&shuf[r * width..][..width]);
                    Ok(())
                }),
                0 => formatted(n, width, |r, out| {
                    out.extend((0..width).map(|b| shuf[b * n + r]));
                    Ok(())
                }),
                1 if width == 8 => {
                    let mut prev = 0u64;
                    formatted(n, 8, |r, out| {
                        let mut db = [0u8; 8];
                        for b in 0..8 {
                            db[b] = shuf[b * n + r];
                        }
                        prev = prev.wrapping_add(unzigzag_u64(u64::from_le_bytes(db)));
                        out.extend_from_slice(&ordered_to_f64_bits(prev).to_le_bytes());
                        Ok(())
                    })
                }
                2 if width == 4 => {
                    let mut prev = 0u32;
                    formatted(n, 4, |r, out| {
                        let mut db = [0u8; 4];
                        for b in 0..4 {
                            db[b] = shuf[b * n + r];
                        }
                        let dz = u32::from_le_bytes(db);
                        let d_s = ((dz >> 1) as i32) ^ (-((dz & 1) as i32));
                        prev = prev.wrapping_add(d_s as u32);
                        let raw_u = if (prev >> 31) != 0 {
                            prev ^ (1u32 << 31)
                        } else {
                            !prev
                        };
                        out.extend_from_slice(&raw_u.to_le_bytes());
                        Ok(())
                    })
                }
                _ => Err(Error::Format(format!("invalid FixedBinary submode {sub}"))),
            }
        }
        15 => match cur.u8()? {
            0 => {
                let inner = decode_present_slice_ctx(cur, n, pools, budget)?;
                Ok(Col::Quoted {
                    inner: Box::new(inner),
                    map: Vec::new(),
                })
            }
            1 => {
                let is_str = decode_u32_seq(cur, budget)?;
                if is_str.len() != n {
                    return Err(Error::Format("QuotedOrNull flag count mismatch".into()));
                }
                let mut k = 0u32;
                let map: Vec<u32> = is_str
                    .iter()
                    .map(|&f| {
                        if f == 0 {
                            return NULL_CELL;
                        }
                        k += 1;
                        k - 1
                    })
                    .collect();
                let inner = if k > 0 {
                    decode_present_slice_ctx(cur, k as usize, pools, budget)?
                } else {
                    Col::Flat(Cells::default())
                };
                Ok(Col::Quoted {
                    inner: Box::new(inner),
                    map,
                })
            }
            sub => Err(Error::Format(format!("unknown QuotedOrNull submode {sub}"))),
        },
        16 => {
            let byte_w = cur.uvar()? as usize;
            if !(4..=32).contains(&byte_w) {
                return Err(Error::Format("invalid HexId byte_w".into()));
            }
            let uppers = decode_u32_seq(cur, budget)?;
            let codes = decode_u32_seq(cur, budget)?;
            let n_miss = cur.uvar()? as usize;
            let misses = mt::decode_leaf(cur, budget)?;
            if uppers.len() != n
                || codes.len() != n
                || misses.len() != n_miss.saturating_mul(byte_w)
            {
                return Err(Error::Format("HexId stream length mismatch".into()));
            }
            budget.check_alloc(n.saturating_mul(byte_w * 2), "HexId output")?;
            let mut bins: Vec<[u8; 32]> = Vec::with_capacity(n);
            let mut mi = 0usize;
            formatted(n, byte_w * 2, |i, out| {
                let c = codes[i];
                let b = if c == 0 {
                    if mi >= n_miss {
                        return Err(Error::Format("HexId miss underflow".into()));
                    }
                    let mut arr = [0u8; 32];
                    arr[..byte_w].copy_from_slice(&misses[mi * byte_w..(mi + 1) * byte_w]);
                    mi += 1;
                    arr
                } else {
                    let dist = c as usize;
                    if dist > i {
                        return Err(Error::Format("HexId backref out of bounds".into()));
                    }
                    bins[i - dist]
                };
                bins.push(b);
                format_hex_bytes(&b[..byte_w], uppers[i] != 0, out);
                Ok(())
            })
        }
        17 => {
            let base = cur.uvar()? as usize;
            let byte_len = cur.uvar()? as usize;
            let lens = decode_u32_seq(cur, budget)?;
            if lens.len() != n {
                return Err(Error::Format("ProseSlice length count mismatch".into()));
            }
            let p = pools.ok_or_else(|| Error::Format("ProseSlice without block pools".into()))?;
            let pool = p.get_prose(budget)?;
            if base
                .checked_add(byte_len)
                .is_none_or(|end| end > pool.len())
            {
                return Err(Error::Format("ProseSlice range out of bounds".into()));
            }
            Ok(Col::Pool {
                pool,
                base,
                ends: ends_of(&lens, byte_len)?,
            })
        }
        18 => {
            let lens = decode_u32_seq(cur, budget)?;
            let bytes = mt::decode_leaf(cur, budget)?;
            let total: u64 = lens.iter().map(|&l| u64::from(l)).sum();
            if lens.len() != n || total != bytes.len() as u64 {
                return Err(Error::Format("Base64 length mismatch".into()));
            }
            budget.check_alloc(bytes.len() / 3 * 4 + 4 * n, "Base64 output")?;
            let mut rest = bytes.as_slice();
            formatted(n, bytes.len() / 3 * 4 / n.max(1) + 4, |i, out| {
                let (cell, tail) = rest.split_at(lens[i] as usize);
                format_base64(cell, out);
                rest = tail;
                Ok(())
            })
        }
        tag => Err(Error::Format(format!("unknown leaf codec tag {tag}"))),
    }
}

// ============================================================================
// 6. Dense Occurrence Block Encoder / Decoder + SharedDictPool + ProseSlice + CrossHexId
// ============================================================================

pub struct DenseBlockEncoded {
    pub summaries: Vec<ColSummary>,
    pub shared_dict_blobs: Vec<Vec<u8>>,
    pub prose_pool_leaf: Option<Vec<u8>>,
    pub col_blobs: Vec<Vec<u8>>,
}

#[inline]
fn unquote_json_opt(s: &[u8]) -> (&[u8], bool) {
    if s.len() >= 2 && s[0] == b'"' && s[s.len() - 1] == b'"' {
        (&s[1..s.len() - 1], true)
    } else {
        (s, false)
    }
}

#[inline]
fn uvarint_len(mut x: u64) -> usize {
    let mut n = 1usize;
    while x >= 0x80 {
        x >>= 7;
        n += 1;
    }
    n
}

fn is_raw_backed_leaf(body: &[u8]) -> bool {
    match body.first().copied() {
        Some(t) if t == LeafCodecTag::Raw as u8 => true,
        Some(t) if t == LeafCodecTag::QuotedOrNull as u8 => match body.get(1).copied() {
            Some(0) => body.get(2).copied() == Some(LeafCodecTag::Raw as u8),
            Some(1) => {
                let mut cur = Cursor::new(&body[2..]);
                let Ok(n) = cur.uvar() else { return false };
                let Ok(mode) = cur.u8() else { return false };
                let ok = match mode {
                    0 => cur.uvar().is_ok(),
                    1 => {
                        if let (Ok(_min), Ok(w)) = (cur.uvar(), cur.u8()) {
                            cur.slice((n as usize * w as usize).div_ceil(8)).is_ok()
                        } else {
                            false
                        }
                    }
                    2 | 3 => {
                        cur.uvar().is_ok()
                            && cur.u8().is_ok()
                            && crate::wire::skip_leaf(&mut cur).is_ok()
                    }
                    4 => crate::wire::skip_leaf(&mut cur).is_ok(),
                    _ => false,
                };
                ok && cur.u8().ok() == Some(LeafCodecTag::Raw as u8)
            }
            _ => false,
        },
        _ => false,
    }
}

fn detect_hex_col_width(vals: &[&[u8]]) -> Option<usize> {
    let mut found_w = None;
    let mut checked = 0usize;
    for &v in vals {
        if v == b"null" {
            continue;
        }
        let (inner, _) = unquote_json_opt(v);
        if inner.len() < 8 || inner.len() > 64 || inner.len() % 2 != 0 {
            return None;
        }
        let (_, _, bw) = parse_hex_exact(inner)?;
        let w = bw as usize;
        if let Some(prev_w) = found_w {
            if prev_w != w {
                return None;
            }
        } else {
            found_w = Some(w);
        }
        checked += 1;
        if checked >= 8 {
            break;
        }
    }
    found_w
}

fn try_encode_cross_hex_id(
    vals: &[&[u8]],
    src_vals: &[&[u8]],
    src_i: usize,
    byte_w: usize,
    backend: EntropyBackend,
    level: i32,
) -> Option<Vec<u8>> {
    let (_, has_quotes) = unquote_json_opt(vals[0]);
    let mut peer_map: HashMap<[u8; 32], usize> = HashMap::with_capacity(src_vals.len().min(4096));
    for (p, &sv) in src_vals.iter().enumerate() {
        let (si, _) = unquote_json_opt(sv);
        if si.len() == byte_w * 2
            && let Some((_, sb, _)) = parse_hex_exact(si)
        {
            peer_map.entry(sb).or_insert(p);
        }
    }
    if peer_map.is_empty() {
        return None;
    }

    let mut uppers = Vec::with_capacity(vals.len());
    let mut codes = Vec::with_capacity(vals.len());
    let mut misses = Vec::new();
    let mut intra_last: HashMap<[u8; 32], usize> = HashMap::with_capacity(vals.len().min(2048));
    let mut prev_peer = 0i64;
    let mut peer_hits = 0usize;

    for (i, &v) in vals.iter().enumerate() {
        let (inner, q) = unquote_json_opt(v);
        if q != has_quotes || inner.len() != byte_w * 2 {
            return None;
        }
        let (up, b, bw) = parse_hex_exact(inner)?;
        debug_assert_eq!(bw as usize, byte_w);
        uppers.push(u32::from(up));
        if let Some(&prev) = intra_last.get(&b) {
            let dist = (i - 1 - prev) as u64;
            let c = 1u64.saturating_add(dist.saturating_mul(2));
            if c <= u32::MAX as u64 {
                codes.push(c as u32);
            } else {
                codes.push(0);
                misses.extend_from_slice(&b[..byte_w]);
            }
        } else if let Some(&p) = peer_map.get(&b) {
            let delta = (p as i64) - prev_peer;
            let dz = ((delta << 1) ^ (delta >> 63)) as u64;
            let c = 2u64.saturating_add(dz.saturating_mul(2));
            if c <= u32::MAX as u64 {
                prev_peer = p as i64;
                codes.push(c as u32);
                peer_hits += 1;
            } else {
                codes.push(0);
                misses.extend_from_slice(&b[..byte_w]);
            }
        } else {
            codes.push(0);
            misses.extend_from_slice(&b[..byte_w]);
        }
        intra_last.insert(b, i);
    }

    if peer_hits < 2 {
        return None;
    }

    let mut out = Vec::new();
    put_uvarint(&mut out, vals.len() as u64);
    out.push(5u8); // h_mode = 5 (CrossHexId)
    put_uvarint(&mut out, src_i as u64);
    put_uvarint(&mut out, byte_w as u64);
    out.push(u8::from(has_quotes));
    encode_u32_seq(&uppers, backend, level, &mut out);
    encode_u32_seq(&codes, backend, level, &mut out);
    put_uvarint(&mut out, (misses.len() / byte_w.max(1)) as u64);
    compress_leaf(&mut out, &misses, backend, level);
    Some(out)
}

fn try_encode_dense_affine_int(
    vals_c: &[&[u8]],
    vals_a: &[&[u8]],
    vals_b: &[&[u8]],
    col_a: usize,
    col_b: usize,
    sub: bool,
    backend: EntropyBackend,
    level: i32,
) -> Option<Vec<u8>> {
    let n = vals_c.len();
    let probe = n.min(32);
    let mut zero_ish = 0usize;
    for r in 0..probe {
        let (sc, qc) = unquote_json_opt(vals_c[r]);
        let (sa, _) = unquote_json_opt(vals_a[r]);
        let (sb, _) = unquote_json_opt(vals_b[r]);
        if qc {
            // keep quote flag consistent below
        }
        let (vc, _) = parse_exact_int(sc)?;
        let (va, _) = parse_exact_int(sa)?;
        let (vb, _) = parse_exact_int(sb)?;
        let pred = if sub {
            va.wrapping_sub(vb)
        } else {
            va.wrapping_add(vb)
        };
        if vc.wrapping_sub(pred).abs() <= 1 {
            zero_ish += 1;
        }
    }
    if zero_ish * 2 < probe {
        return None;
    }

    let (_, has_quotes) = unquote_json_opt(vals_c[0]);
    let mut flags = Vec::with_capacity(n);
    let mut residuals = Vec::new();
    let mut fmts = Vec::new();
    let mut fallbacks = Vec::new();
    for r in 0..n {
        let (sc, qc) = unquote_json_opt(vals_c[r]);
        let (sa, _) = unquote_json_opt(vals_a[r]);
        let (sb, _) = unquote_json_opt(vals_b[r]);
        if qc == has_quotes
            && let (Some((vc, fc)), Some((va, _)), Some((vb, _))) = (
                parse_exact_int(sc),
                parse_exact_int(sa),
                parse_exact_int(sb),
            )
        {
            let pred = if sub {
                va.wrapping_sub(vb)
            } else {
                va.wrapping_add(vb)
            };
            flags.push(1u32);
            residuals.push(vc.wrapping_add(-pred));
            fmts.push(fc as u32);
        } else {
            flags.push(0u32);
            fallbacks.push(vals_c[r]);
        }
    }
    if residuals.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    put_uvarint(&mut out, n as u64);
    out.push(3u8); // h_mode = 3 (AffineInt)
    put_uvarint(&mut out, col_a as u64);
    put_uvarint(&mut out, col_b as u64);
    out.push(u8::from(sub) | (u8::from(has_quotes) << 1));
    encode_u32_seq(&flags, backend, level, &mut out);
    encode_i64_seq(&residuals, backend, level, &mut out);
    encode_u32_seq(&fmts, backend, level, &mut out);
    if !fallbacks.is_empty() {
        let (_, _, _, fb_enc) = encode_present_slice(&fallbacks, backend, level, 1);
        out.extend_from_slice(&fb_enc);
    }
    Some(out)
}

/// Encodes all dense occurrence columns for a block, running standalone typed codecs,
/// horizontal derivations (`ExactCopy`, `SubstringInterp`, `AffineInt`, `CrossHexId`),
/// block-level `SharedDictPool` clustering, and `ProseSlice` shared-window pooling.
#[allow(clippy::too_many_arguments)]
pub fn encode_dense_block_columns(
    occ_streams: &[Vec<&[u8]>],
    dict_groups: &[&str],
    horiz_cands: &[Vec<usize>],
    preencoded_nested: &mut [Option<(ColSummary, Vec<u8>)>],
    standalone: &[bool],
    backend: EntropyBackend,
    level: i32,
) -> DenseBlockEncoded {
    let n_cols = occ_streams.len();
    let mut summaries = Vec::with_capacity(n_cols);
    let mut col_blobs = Vec::with_capacity(n_cols);
    let mut used_horiz = vec![false; n_cols];
    let mut raw_backed = vec![false; n_cols];
    let hex_widths: Vec<Option<usize>> = occ_streams
        .iter()
        .map(|s| detect_hex_col_width(s))
        .collect();

    // Phase 1: Per-column standalone + horizontal competition
    for cid in 0..n_cols {
        if let Some((sum, blob)) = preencoded_nested[cid].take() {
            summaries.push(sum);
            col_blobs.push(blob);
            used_horiz[cid] = true;
            continue;
        }
        let vals = &occ_streams[cid];
        if vals.is_empty() {
            summaries.push(ColSummary::of(&[]));
            col_blobs.push(vec![0u8]);
            continue;
        }
        let occ_len = vals.len();
        let mut sum = ColSummary::of(vals);
        let (nk, mn, mx, standalone_body) = encode_present_slice(vals, backend, level, 0);
        if nk != 0 {
            sum.numeric_kind = nk;
            sum.min_val = mn;
            sum.max_val = mx;
        } else if let Some((lo, hi)) = int_bounds(vals) {
            (sum.numeric_kind, sum.min_val, sum.max_val) = (1, lo, hi);
        } else if let Some((lo, hi)) = time_bounds(vals) {
            (sum.numeric_kind, sum.min_val, sum.max_val) = (2, lo, hi);
        }
        let is_raw_leaf = is_raw_backed_leaf(&standalone_body);
        let mut best_blob = Vec::with_capacity(10 + standalone_body.len());
        put_uvarint(&mut best_blob, occ_len as u64);
        best_blob.push(0u8); // h_mode = 0 (Standalone)
        best_blob.extend_from_slice(&standalone_body);

        if occ_len >= 8 {
            let cands = &horiz_cands[cid];
            // 1a. Equal-length ExactCopy & SubstringInterp
            let mut interp_srcs = Vec::<usize>::new();
            for &src_i in cands {
                if src_i >= cid || occ_streams[src_i].len() != occ_len {
                    continue;
                }
                let src_vals = &occ_streams[src_i];
                let probe = occ_len.min(64);
                let mut eq_probe = 0usize;
                let mut sub_probe = 0usize;
                for r in 0..probe {
                    let tv = vals[r];
                    let sv = src_vals[r];
                    if tv == sv {
                        eq_probe += 1;
                    } else {
                        let (sv_u, _) = unquote_json_opt(sv);
                        if sv_u.len() >= 3
                            && tv.len() > sv_u.len()
                            && !sv_u.contains(&0x1f)
                            && memchr::memmem::find(tv, sv_u).is_some()
                        {
                            sub_probe += 1;
                        }
                    }
                }
                if eq_probe * 4 >= probe {
                    let mut flags = Vec::with_capacity(occ_len);
                    let mut fallbacks = Vec::new();
                    for r in 0..occ_len {
                        if vals[r] == src_vals[r] {
                            flags.push(1u32);
                        } else {
                            flags.push(0u32);
                            fallbacks.push(vals[r]);
                        }
                    }
                    let mut cand = Vec::new();
                    put_uvarint(&mut cand, occ_len as u64);
                    cand.push(1u8); // h_mode = 1 (ExactCopy)
                    put_uvarint(&mut cand, src_i as u64);
                    encode_u32_seq(&flags, backend, level, &mut cand);
                    if !fallbacks.is_empty() {
                        let (_, _, _, fb_enc) = encode_present_slice(&fallbacks, backend, level, 1);
                        cand.extend_from_slice(&fb_enc);
                    }
                    if cand.len() + 6 < best_blob.len() {
                        best_blob = cand;
                        used_horiz[cid] = true;
                    }
                }
                if interp_srcs.len() < 3 && sub_probe * 4 >= probe {
                    interp_srcs.push(src_i);
                }
            }

            if !interp_srcs.is_empty() {
                let mut templated: Vec<Vec<u8>> = Vec::with_capacity(occ_len);
                let mut flags = Vec::with_capacity(occ_len);
                let mut hits = 0usize;
                for r in 0..occ_len {
                    let target = vals[r];
                    if target.contains(&0x1f) {
                        flags.push(0u32);
                        templated.push(target.to_vec());
                        continue;
                    }
                    let mut cur = target.to_vec();
                    let mut replaced = false;
                    for (slot_k, &src_i) in interp_srcs.iter().enumerate() {
                        let sv = occ_streams[src_i][r];
                        let (sv_u, was_q) = unquote_json_opt(sv);
                        if sv.len() >= 3
                            && cur.len() > sv.len()
                            && !sv.contains(&0x1f)
                            && let Some(next) =
                                replace_subslice(&cur, sv, &[0x1f, (slot_k * 2) as u8])
                        {
                            cur = next;
                            replaced = true;
                        } else if was_q
                            && sv_u.len() >= 3
                            && cur.len() > sv_u.len()
                            && !sv_u.contains(&0x1f)
                            && let Some(next) =
                                replace_subslice(&cur, sv_u, &[0x1f, (slot_k * 2 + 1) as u8])
                        {
                            cur = next;
                            replaced = true;
                        }
                    }
                    if replaced {
                        hits += 1;
                        flags.push(1u32);
                    } else {
                        flags.push(0u32);
                    }
                    templated.push(cur);
                }
                if hits * 4 >= occ_len {
                    let t_refs: Vec<&[u8]> = templated.iter().map(|x| x.as_slice()).collect();
                    let (_, _, _, t_enc) = encode_present_slice(&t_refs, backend, level, 0);
                    let mut cand = Vec::new();
                    put_uvarint(&mut cand, occ_len as u64);
                    cand.push(2u8); // h_mode = 2 (SubstringInterp)
                    put_uvarint(&mut cand, interp_srcs.len() as u64);
                    for &src_i in &interp_srcs {
                        put_uvarint(&mut cand, src_i as u64);
                    }
                    encode_u32_seq(&flags, backend, level, &mut cand);
                    cand.extend_from_slice(&t_enc);
                    if cand.len() + 6 < best_blob.len() {
                        best_blob = cand;
                        used_horiz[cid] = true;
                    }
                }
            }

            // 1b. Equal-length AffineInt
            if sum.numeric_kind == 1 {
                let mut int_srcs = Vec::new();
                for &src_i in cands {
                    if src_i < cid
                        && occ_streams[src_i].len() == occ_len
                        && occ_streams[src_i]
                            .iter()
                            .take(16)
                            .all(|b| parse_exact_int(unquote_json_opt(b).0).is_some())
                    {
                        int_srcs.push(src_i);
                        if int_srcs.len() >= 4 {
                            break;
                        }
                    }
                }
                for ia in 0..int_srcs.len() {
                    for ib in 0..int_srcs.len() {
                        if ia == ib {
                            continue;
                        }
                        for sub in [false, true] {
                            if let Some(cand) = try_encode_dense_affine_int(
                                vals,
                                &occ_streams[int_srcs[ia]],
                                &occ_streams[int_srcs[ib]],
                                int_srcs[ia],
                                int_srcs[ib],
                                sub,
                                backend,
                                level,
                            ) && cand.len() + 6 < best_blob.len()
                            {
                                best_blob = cand;
                                used_horiz[cid] = true;
                            }
                        }
                    }
                }
            }

            // 1c. Any-length CrossHexId (`$.payload.before` -> `$.payload.head` -> `$.payload.commits[].sha`!)
            if let Some(bw) = hex_widths[cid] {
                let mut tried = 0usize;
                for src_i in (0..cid).rev() {
                    if hex_widths[src_i] == Some(bw) && !occ_streams[src_i].is_empty() {
                        if let Some(cand) = try_encode_cross_hex_id(
                            vals,
                            &occ_streams[src_i],
                            src_i,
                            bw,
                            backend,
                            level,
                        ) && cand.len() + 6 < best_blob.len()
                        {
                            best_blob = cand;
                            used_horiz[cid] = true;
                        }
                        tried += 1;
                        if tried >= 3 {
                            break;
                        }
                    }
                }
            }
        }

        if !used_horiz[cid]
            && (is_raw_leaf || dict_groups.get(cid).copied().unwrap_or("") == "text")
        {
            raw_backed[cid] = true;
        }
        summaries.push(sum);
        col_blobs.push(best_blob);
    }

    // Phase 2: Block-level `SharedDictPool` clustering across categorical/string columns
    struct DictCand<'a> {
        cid: usize,
        vocab: HashMap<&'a [u8], u32>,
    }
    let mut dict_cands = Vec::<DictCand<'_>>::new();
    for cid in 0..n_cols {
        let vals = &occ_streams[cid];
        if vals.is_empty()
            || used_horiz[cid]
            || summaries[cid].numeric_kind != 0
            || hex_widths[cid].is_some()
            || dict_groups.get(cid).copied().unwrap_or("") == "text"
        {
            continue;
        }
        let mut vocab: HashMap<&[u8], u32> = HashMap::with_capacity(vals.len().min(512));
        let mut uniq_bytes = 0usize;
        let mut ok = true;
        for &v in vals {
            if let Some(c) = vocab.get_mut(v) {
                *c += 1;
            } else {
                if vocab.len() >= 4096 || uniq_bytes + v.len() > (256 << 10) {
                    ok = false;
                    break;
                }
                uniq_bytes += v.len();
                vocab.insert(v, 1);
            }
        }
        if ok && vocab.len() >= 2 {
            dict_cands.push(DictCand { cid, vocab });
        }
    }

    let mut parent: Vec<usize> = (0..dict_cands.len()).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for i in 0..dict_cands.len() {
        let ga = dict_groups.get(dict_cands[i].cid).copied().unwrap_or("");
        for j in (i + 1)..dict_cands.len() {
            let ri = find(&mut parent, i);
            let rj = find(&mut parent, j);
            if ri == rj {
                continue;
            }
            let gb = dict_groups.get(dict_cands[j].cid).copied().unwrap_or("");
            let same_group = ga == "ip" && gb == "ip";
            let should_merge = if same_group {
                true
            } else {
                let (sm, lg) = if dict_cands[i].vocab.len() <= dict_cands[j].vocab.len() {
                    (&dict_cands[i].vocab, &dict_cands[j].vocab)
                } else {
                    (&dict_cands[j].vocab, &dict_cands[i].vocab)
                };
                let inter = sm.keys().filter(|k| lg.contains_key(*k)).count();
                let min_sz = sm.len();
                (inter >= 2 && inter * 10 >= min_sz)
                    || (min_sz <= 4 && inter >= 1 && inter * 2 >= min_sz)
            };
            if should_merge {
                parent[rj] = ri;
            }
        }
    }

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..dict_cands.len() {
        let r = find(&mut parent, i);
        clusters.entry(r).or_default().push(i);
    }
    let mut cluster_list: Vec<Vec<usize>> =
        clusters.into_values().filter(|m| m.len() >= 2).collect();
    cluster_list.sort_by_key(|m| dict_cands[m[0]].cid);

    let mut shared_dict_blobs = Vec::<Vec<u8>>::new();
    for members in cluster_list {
        let mut combined: HashMap<&[u8], u32> = HashMap::new();
        for &mi in &members {
            for (&k, &cnt) in &dict_cands[mi].vocab {
                *combined.entry(k).or_insert(0) += cnt;
            }
        }
        if combined.len() < 2 || combined.len() > 8192 {
            continue;
        }
        // Try frequency-sorted, first-seen, and lexicographic entry orders!
        let mut orderings: Vec<Vec<&[u8]>> = Vec::with_capacity(3);
        let mut freq_v: Vec<(&[u8], u32)> = combined.iter().map(|(&k, &c)| (k, c)).collect();
        freq_v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        orderings.push(freq_v.into_iter().map(|x| x.0).collect());

        let mut first_seen = Vec::with_capacity(combined.len());
        let mut fs_set = std::collections::HashSet::with_capacity(combined.len());
        for &mi in &members {
            for &v in &occ_streams[dict_cands[mi].cid] {
                if fs_set.insert(v) {
                    first_seen.push(v);
                }
            }
        }
        if first_seen.len() == combined.len() {
            orderings.push(first_seen);
        }
        let mut lex_v: Vec<&[u8]> = combined.keys().copied().collect();
        lex_v.sort_unstable();
        orderings.push(lex_v);

        let dict_id = shared_dict_blobs.len();
        let old_sum: usize = members
            .iter()
            .map(|&mi| col_blobs[dict_cands[mi].cid].len())
            .sum();
        let mut best_cluster: Option<(usize, Vec<u8>, Vec<(usize, Vec<u8>)>)> = None;

        for ord_entries in orderings {
            let mut id_map = HashMap::with_capacity(ord_entries.len());
            let mut entries = Vec::with_capacity(ord_entries.len());
            for (idx, &bytes) in ord_entries.iter().enumerate() {
                id_map.insert(bytes, idx as u32);
                entries.push(bytes.to_vec());
            }
            let table = DictTable { entries };
            let mut dict_blob = Vec::new();
            table.encode(backend, level, &mut dict_blob);

            let mut ref_blobs = Vec::with_capacity(members.len());
            let mut new_sum = uvarint_len(dict_blob.len() as u64) + dict_blob.len();
            for &mi in &members {
                let cid = dict_cands[mi].cid;
                let vals = &occ_streams[cid];
                let ords: Vec<u32> = vals.iter().map(|v| id_map[v]).collect();
                let mut ord_static = vec![0u8];
                encode_u32_seq(&ords, backend, level, &mut ord_static);
                let mut best_ord = ord_static;
                if table.entries.len() >= 16
                    && vals.len() >= 64
                    && let Some(mtf_ranks) = mtf_transform_u32(&ords, 4096)
                {
                    let mut ord_mtf = vec![1u8];
                    encode_u32_seq(&mtf_ranks, backend, level, &mut ord_mtf);
                    if ord_mtf.len() < best_ord.len() {
                        best_ord = ord_mtf;
                    }
                }
                let mut rb = Vec::with_capacity(12 + best_ord.len());
                put_uvarint(&mut rb, vals.len() as u64);
                rb.push(0u8); // h_mode = 0
                rb.push(LeafCodecTag::SharedDictRef as u8);
                put_uvarint(&mut rb, dict_id as u64);
                rb.extend_from_slice(&best_ord);
                new_sum += rb.len();
                ref_blobs.push((cid, rb));
            }
            if best_cluster.as_ref().is_none_or(|b| new_sum < b.0) {
                best_cluster = Some((new_sum, dict_blob, ref_blobs));
            }
        }

        if let Some((new_sum, dict_blob, ref_blobs)) = best_cluster
            && new_sum + 8 < old_sum
        {
            shared_dict_blobs.push(dict_blob);
            for (cid, rb) in ref_blobs {
                col_blobs[cid] = rb;
                raw_backed[cid] = false;
            }
        }
    }

    // Phase 3: Block-level `ProseSlice` shared LZ77/FSE window across unstructured text/JSON-subtree columns
    let mut prose_cands = Vec::<usize>::new();
    let mut total_prose_raw = 0usize;
    for cid in 0..n_cols {
        if raw_backed[cid] && !standalone[cid] && !occ_streams[cid].is_empty() {
            let raw_b: usize = occ_streams[cid].iter().map(|v| v.len()).sum();
            if raw_b >= PROSE_MIN_CELL * occ_streams[cid].len() {
                prose_cands.push(cid);
                total_prose_raw += raw_b;
            }
        }
    }

    let mut prose_pool_leaf = None;
    if prose_cands.len() >= 2 && total_prose_raw >= 128 {
        let mut pool_raw = Vec::with_capacity(total_prose_raw);
        let mut slice_blobs = Vec::with_capacity(prose_cands.len());
        let mut old_cost = 0usize;
        let mut slices_cost = 0usize;

        for &cid in &prose_cands {
            let vals = &occ_streams[cid];
            let byte_off = pool_raw.len();
            let mut lens = Vec::with_capacity(vals.len());
            for &v in vals {
                pool_raw.extend_from_slice(v);
                lens.push(v.len() as u32);
            }
            let byte_len = pool_raw.len() - byte_off;
            let mut sb = Vec::new();
            put_uvarint(&mut sb, vals.len() as u64);
            sb.push(0u8); // h_mode = 0
            sb.push(LeafCodecTag::ProseSlice as u8);
            put_uvarint(&mut sb, byte_off as u64);
            put_uvarint(&mut sb, byte_len as u64);
            encode_u32_seq(&lens, backend, level, &mut sb);
            old_cost += col_blobs[cid].len();
            slices_cost += sb.len();
            slice_blobs.push((cid, sb));
        }

        let mut leaf = Vec::new();
        compress_leaf(&mut leaf, &pool_raw, backend, level);
        if 1 + leaf.len() + slices_cost + 8 < old_cost {
            prose_pool_leaf = Some(leaf);
            for (cid, sb) in slice_blobs {
                col_blobs[cid] = sb;
            }
        }
    }

    DenseBlockEncoded {
        summaries,
        shared_dict_blobs,
        prose_pool_leaf,
        col_blobs,
    }
}

/// The mean cell length from which a column of raw text joins the block's pool. Shorter cells are
/// names and identifiers, which queries read on their own: pooled, each would cost the whole pool.
/// Measured on GitHub events at 8 MiB blocks, keeping them out costs 1.0% of the file. A column
/// the caller marks `standalone` stays out whatever its cells.
const PROSE_MIN_CELL: usize = 40;

fn replace_subslice(haystack: &[u8], needle: &[u8], token: &[u8; 2]) -> Option<Vec<u8>> {
    let first = memchr::memmem::find(haystack, needle)?;
    let mut out = Vec::with_capacity(haystack.len() - needle.len() + 2);
    let mut pos = 0usize;
    let mut cur_find = Some(first);
    while let Some(rel) = cur_find {
        let idx = pos + rel;
        out.extend_from_slice(&haystack[pos..idx]);
        out.extend_from_slice(token);
        pos = idx + needle.len();
        cur_find = memchr::memmem::find(&haystack[pos..], needle);
    }
    out.extend_from_slice(&haystack[pos..]);
    Some(out)
}

/// What a column's bytes need from the rest of its block before they decode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColDeps {
    pub pool: bool,
    pub dict: Option<u32>,
    pub cols: Vec<u32>,
}

/// Reads the dependencies out of a column blob's header.
pub fn column_deps(blob: &[u8]) -> ColDeps {
    let mut d = ColDeps::default();
    let mut cur = Cursor::new(blob);
    if cur.uvar().unwrap_or(0) == 0 {
        return d;
    }
    let src = |cur: &mut Cursor<'_>, d: &mut ColDeps| {
        if let Ok(s) = cur.uvar() {
            d.cols.push(s as u32);
        }
    };
    match cur.u8() {
        Ok(0) => match cur.u8() {
            Ok(t) if t == LeafCodecTag::SharedDictRef as u8 => {
                d.dict = cur.uvar().ok().map(|i| i as u32)
            }
            Ok(t) if t == LeafCodecTag::ProseSlice as u8 => d.pool = true,
            _ => {}
        },
        Ok(1) | Ok(5) => src(&mut cur, &mut d),
        Ok(2) => {
            for _ in 0..cur.uvar().unwrap_or(0).min(8) {
                src(&mut cur, &mut d);
            }
        }
        Ok(3) => {
            src(&mut cur, &mut d);
            src(&mut cur, &mut d);
        }
        _ => {}
    }
    d
}

/// The parts of one block and the columns decoded from them so far.
pub struct BlockDecodeCtx<'a> {
    pub col_blobs: Vec<Option<&'a [u8]>>,
    pub(crate) pools: SharedPools<'a>,
    pub cache: Vec<Option<Col>>,
}

impl<'a> BlockDecodeCtx<'a> {
    pub fn new(
        col_blobs: Vec<Option<&'a [u8]>>,
        dict_blobs: Vec<Option<&'a [u8]>>,
        pool_leaf: Option<&'a [u8]>,
    ) -> Self {
        let n_cols = col_blobs.len();
        Self {
            col_blobs,
            pools: SharedPools::new(dict_blobs, pool_leaf),
            cache: (0..n_cols).map(|_| None).collect(),
        }
    }
}

/// Decodes column `col_idx` into `ctx.cache`, with the earlier columns, dictionary or text pool it
/// refers to.
pub fn ensure_dense_block_column<'c>(
    col_idx: usize,
    ctx: &'c mut BlockDecodeCtx<'_>,
    budget: &DecodeBudget,
) -> Result<&'c Col> {
    if ctx.cache[col_idx].is_some() {
        return Ok(ctx.cache[col_idx].as_ref().unwrap());
    }
    let mut cur = Cursor::new(ctx.col_blobs[col_idx].ok_or_else(not_loaded)?);
    let occ_len = cur.uvar()? as usize;
    if occ_len == 0 {
        ctx.cache[col_idx] = Some(Col::Flat(Cells::default()));
        return Ok(ctx.cache[col_idx].as_ref().unwrap());
    }
    if occ_len > budget.max_records_per_block.saturating_mul(16) {
        return Err(Error::BudgetExceeded("dense column occurrence count"));
    }
    let h_mode = cur.u8()?;
    let source = |ctx: &mut BlockDecodeCtx<'_>, src_i: usize, same_len: bool| -> Result<()> {
        if src_i >= col_idx {
            return Err(Error::Format("horizontal forward/self reference".into()));
        }
        let src = ensure_dense_block_column(src_i, ctx, budget)?;
        if same_len && src.len() != occ_len {
            return Err(Error::Format("horizontal source occ_len mismatch".into()));
        }
        Ok(())
    };
    let res = match h_mode {
        0 => decode_present_slice_ctx(&mut cur, occ_len, Some(&mut ctx.pools), budget)?,
        1 => {
            let src_i = cur.uvar()? as usize;
            let flags = decode_u32_seq(&mut cur, budget)?;
            if flags.len() != occ_len {
                return Err(Error::Format("ExactCopy flag count mismatch".into()));
            }
            let n_miss = flags.iter().filter(|&&f| f == 0).count();
            let fb = decode_present_slice_ctx(&mut cur, n_miss, Some(&mut ctx.pools), budget)?;
            source(ctx, src_i, true)?;
            let src_col = ctx.cache[src_i].as_ref().unwrap();
            if n_miss == 0 {
                src_col.clone()
            } else {
                let mut k = 0usize;
                formatted(occ_len, 16, |r, out| {
                    if flags[r] != 0 {
                        return src_col.write(r, out);
                    }
                    k += 1;
                    fb.write(k - 1, out)
                })?
            }
        }
        2 => {
            let k = cur.uvar()? as usize;
            if k == 0 || k > 8 {
                return Err(Error::Format("invalid SubstringInterp arity".into()));
            }
            let mut srcs = Vec::with_capacity(k);
            for _ in 0..k {
                srcs.push(cur.uvar()? as usize);
            }
            let flags = decode_u32_seq(&mut cur, budget)?;
            let t_col = decode_present_slice_ctx(&mut cur, occ_len, Some(&mut ctx.pools), budget)?;
            if flags.len() != occ_len {
                return Err(Error::Format("SubstringInterp occ_len mismatch".into()));
            }
            for &s in &srcs {
                source(ctx, s, true)?;
            }
            let (mut tb_buf, mut stem_buf) = (Vec::new(), Vec::new());
            formatted(occ_len, 32, |r, out| {
                if flags[r] == 0 {
                    return t_col.write(r, out);
                }
                let b = t_col.get(r, &mut tb_buf)?;
                let start = out.len();
                let mut i = 0usize;
                while i < b.len() {
                    if b[i] != 0x1f {
                        out.push(b[i]);
                        i += 1;
                        continue;
                    }
                    let &code = b
                        .get(i + 1)
                        .ok_or_else(|| Error::Format("dangling interp escape".into()))?;
                    let src_i = *srcs
                        .get((code >> 1) as usize)
                        .ok_or_else(|| Error::Format("interp slot out of bounds".into()))?;
                    let stem = ctx.cache[src_i].as_ref().unwrap().get(r, &mut stem_buf)?;
                    out.extend_from_slice(if code & 1 != 0 {
                        unquote_json_opt(stem).0
                    } else {
                        stem
                    });
                    i += 2;
                }
                budget.check_alloc(out.len() - start, "SubstringInterp row")
            })?
        }
        3 => {
            let col_a = cur.uvar()? as usize;
            let col_b = cur.uvar()? as usize;
            let mode_bits = cur.u8()?;
            let sub = (mode_bits & 1) != 0;
            let has_quotes = (mode_bits & 2) != 0;
            let flags = decode_u32_seq(&mut cur, budget)?;
            let residuals = decode_i64_seq(&mut cur, budget)?;
            let fmts = decode_u32_seq(&mut cur, budget)?;
            if flags.len() != occ_len || residuals.len() != fmts.len() {
                return Err(Error::Format("AffineInt stream length mismatch".into()));
            }
            let n_miss = flags.iter().filter(|&&f| f == 0).count();
            let fb = decode_present_slice_ctx(&mut cur, n_miss, Some(&mut ctx.pools), budget)?;
            source(ctx, col_a, true)?;
            source(ctx, col_b, true)?;
            let a_vals = ctx.cache[col_a].as_ref().unwrap();
            let b_vals = ctx.cache[col_b].as_ref().unwrap();
            let (mut ri, mut fi) = (0usize, 0usize);
            let (mut sa, mut sb) = (Vec::new(), Vec::new());
            formatted(occ_len, 22, |r, out| {
                if flags[r] == 0 {
                    fi += 1;
                    return fb.write(fi - 1, out);
                }
                if ri >= residuals.len() {
                    return Err(Error::Format("AffineInt residual underflow".into()));
                }
                let (va, _) = parse_exact_int(unquote_json_opt(a_vals.get(r, &mut sa)?).0)
                    .ok_or_else(|| Error::Format("AffineInt non-int parent a".into()))?;
                let (vb, _) = parse_exact_int(unquote_json_opt(b_vals.get(r, &mut sb)?).0)
                    .ok_or_else(|| Error::Format("AffineInt non-int parent b".into()))?;
                let pred = if sub {
                    va.wrapping_sub(vb)
                } else {
                    va.wrapping_add(vb)
                };
                if has_quotes {
                    out.push(b'"');
                }
                format_exact_int(pred.wrapping_add(residuals[ri]), fmts[ri] as u8, out);
                if has_quotes {
                    out.push(b'"');
                }
                ri += 1;
                Ok(())
            })?
        }
        5 => {
            let src_i = cur.uvar()? as usize;
            let byte_w = cur.uvar()? as usize;
            let has_quotes = cur.u8()? != 0;
            if !(4..=32).contains(&byte_w) {
                return Err(Error::Format("invalid CrossHexId header".into()));
            }
            let uppers = decode_u32_seq(&mut cur, budget)?;
            let codes = decode_u32_seq(&mut cur, budget)?;
            let n_miss = cur.uvar()? as usize;
            let misses = decompress_leaf(&mut cur, budget)?;
            if uppers.len() != occ_len
                || codes.len() != occ_len
                || misses.len() != n_miss.saturating_mul(byte_w)
            {
                return Err(Error::Format("CrossHexId length mismatch".into()));
            }
            source(ctx, src_i, false)?;
            let src_col = ctx.cache[src_i].as_ref().unwrap();
            let mut scratch = Vec::new();
            let mut peer_bins: Vec<Option<[u8; 32]>> = Vec::with_capacity(src_col.len());
            for i in 0..src_col.len() {
                let (si, _) = unquote_json_opt(src_col.get(i, &mut scratch)?);
                peer_bins.push(if si.len() == byte_w * 2 {
                    parse_hex_exact(si).map(|(_, sb, _)| sb)
                } else {
                    None
                });
            }
            budget.check_alloc(occ_len.saturating_mul(byte_w * 2 + 2), "CrossHexId output")?;
            let mut bins: Vec<[u8; 32]> = Vec::with_capacity(occ_len);
            let mut mi = 0usize;
            let mut prev_peer = 0i64;
            formatted(occ_len, byte_w * 2 + 2, |i, out| {
                let c = codes[i];
                let b = if c == 0 {
                    if mi >= n_miss {
                        return Err(Error::Format("CrossHexId miss underflow".into()));
                    }
                    let mut arr = [0u8; 32];
                    arr[..byte_w].copy_from_slice(&misses[mi * byte_w..(mi + 1) * byte_w]);
                    mi += 1;
                    arr
                } else if (c & 1) == 1 {
                    let dist = ((c - 1) >> 1) as usize;
                    if dist >= i {
                        return Err(Error::Format(
                            "CrossHexId intra backref out of bounds".into(),
                        ));
                    }
                    bins[i - 1 - dist]
                } else {
                    let dz = ((c - 2) >> 1) as u64;
                    let delta = ((dz >> 1) as i64) ^ (-((dz & 1) as i64));
                    let p_i64 = prev_peer.wrapping_add(delta);
                    if p_i64 < 0 || (p_i64 as usize) >= peer_bins.len() {
                        return Err(Error::Format(
                            "CrossHexId peer backref out of bounds".into(),
                        ));
                    }
                    prev_peer = p_i64;
                    peer_bins[p_i64 as usize]
                        .ok_or_else(|| Error::Format("CrossHexId peer non-hex entry".into()))?
                };
                bins.push(b);
                if has_quotes {
                    out.push(b'"');
                }
                format_hex_bytes(&b[..byte_w], uppers[i] != 0, out);
                if has_quotes {
                    out.push(b'"');
                }
                Ok(())
            })?
        }
        m => return Err(Error::Format(format!("unknown dense horizontal mode {m}"))),
    };
    ctx.cache[col_idx] = Some(res);
    Ok(ctx.cache[col_idx].as_ref().unwrap())
}

#[cfg(test)]
mod tests {
    #[test]
    fn mtf_ordinals_need_cheap_recency() {
        let runs: Vec<u32> = (0..9000).map(|i| i / 40 % 300).collect();
        let ranks = mtf_transform_u32(&runs, 4096).unwrap();
        assert_eq!(mtf_untransform_u32(&ranks, 4096, 300).unwrap(), runs);
        // No value near its last use.
        let spread: Vec<u32> = (0..9000u32)
            .map(|i| i.wrapping_mul(2_654_435_761) >> 22)
            .collect();
        assert!(mtf_transform_u32(&spread, 4096).is_none());
        // Pairs, each 2000 values after the pair before it: half the ranks are deep.
        let deep: Vec<u32> = (0..9000).map(|i| i / 2 % 2000).collect();
        assert!(mtf_transform_u32(&deep, 4096).is_none());
    }

    #[test]
    fn recency_matches_a_shifted_list() {
        let (window, n) = (5usize, 400usize);
        let mut ring = super::Recency::<u32>::new(window, n);
        let mut list: Vec<u32> = Vec::new();
        let mut x = 7u32;
        for _ in 0..n {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let v = (x >> 16) % 9;
            let rank = list.iter().position(|&l| l == v);
            assert_eq!(ring.rank(v), rank);
            match rank {
                Some(r) => {
                    list.remove(r);
                    assert_eq!(ring.hit(r), Some(v));
                }
                None => {
                    list.truncate(window - 1);
                    ring.push(v);
                }
            }
            list.insert(0, v);
            assert_eq!(ring.hit(list.len()), None);
        }
    }

    use super::*;

    const Z: EntropyBackend = EntropyBackend::Zstd;

    fn noise(x: &mut u32) -> u32 {
        *x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
        *x >> 16
    }

    #[test]
    fn long_tailed_ordinals_keep_a_byte_plane() {
        let budget = DecodeBudget::default();
        // Small values with one in eight from a tail up to 4,000, as recency ranks are.
        let mut x = 1u32;
        let vals: Vec<u32> = (0..4000)
            .map(|_| match noise(&mut x) % 8 {
                0 => 255 + noise(&mut x) % 3745,
                _ => noise(&mut x) % 40,
            })
            .collect();
        let (mut enc, mut flat) = (Vec::new(), Vec::new());
        encode_u32_seq(&vals, Z, 3, &mut enc);
        encode_u32_seq_with(&vals, Z, 3, false, &mut flat);
        assert_eq!(enc[2], 5);
        assert!(
            enc.len() * 10 < flat.len() * 9,
            "{} {}",
            enc.len(),
            flat.len()
        );
        assert_eq!(
            decode_u32_seq(&mut Cursor::new(&enc), &budget).unwrap(),
            vals
        );
        for end in 0..enc.len() {
            assert!(decode_u32_seq(&mut Cursor::new(&enc[..end]), &budget).is_err());
        }

        let build = |bytes: &[u8], over: &[u8]| {
            let mut b = vec![bytes.len() as u8, 5, 0];
            compress_leaf(&mut b, bytes, EntropyBackend::Raw, 3);
            b.extend_from_slice(over);
            b
        };
        // Two values past 254, continued as the constant 7.
        let ok = build(&[1, 255, 255], &[2, 0, 7]);
        assert_eq!(
            decode_u32_seq(&mut Cursor::new(&ok), &budget).unwrap(),
            [1, 262, 262]
        );
        for bad in [
            build(&[1, 255, 255], &[1, 0, 7]),
            build(&[1, 255, 255], &[3, 0, 7]),
            build(&[1, 255], &[2, 0, 7]),
            // The tail is not continued again.
            [
                build(&[1, 255, 255], &[2, 5, 0]),
                build(&[9, 9], &[0])[3..].to_vec(),
            ]
            .concat(),
        ] {
            assert!(decode_u32_seq(&mut Cursor::new(&bad), &budget).is_err());
        }
    }

    #[test]
    fn recency_stores_new_values_as_a_sequence() {
        let budget = DecodeBudget::default();
        // 500 ids a thousand apart, each referenced eight times among its neighbours.
        let mut x = 3u32;
        let vals: Vec<i64> = (0..4000usize)
            .map(|i| 1_000_000_000_000_000 + ((i / 8 + noise(&mut x) as usize % 6) as i64) * 1000)
            .collect();
        let mut enc = Vec::new();
        assert_eq!(encode_i64_seq_with(&vals, Z, 3, ALLOW_RECENCY, &mut enc), 8);
        assert!(enc.len() < 4000, "{}", enc.len());
        assert_eq!(
            decode_i64_seq(&mut Cursor::new(&enc), &budget).unwrap(),
            vals
        );
        for end in 0..enc.len() {
            assert!(decode_i64_seq(&mut Cursor::new(&enc[..end]), &budget).is_err());
        }

        // Codes: new, new, the older of the two, new.
        let build = |mode: u8, new: &[i64]| {
            let mut b = vec![4, mode];
            encode_u32_seq(&[0, 0, 2, 0], Z, 3, &mut b);
            if mode == 4 {
                b.push(new.len() as u8);
                let planes: Vec<u8> = (0..8)
                    .flat_map(|p| new.iter().map(move |v| v.to_le_bytes()[p]))
                    .collect();
                compress_leaf(&mut b, &planes, Z, 3);
            } else {
                encode_i64_seq_with(new, Z, 3, 0, &mut b);
            }
            b
        };
        for mode in [4, 8] {
            let ok = build(mode, &[5, -6, 7]);
            assert_eq!(
                decode_i64_seq(&mut Cursor::new(&ok), &budget).unwrap(),
                [5, -6, 5, 7]
            );
            for new in [&[5, -6][..], &[5, -6, 7, 8]] {
                assert!(decode_i64_seq(&mut Cursor::new(&build(mode, new)), &budget).is_err());
            }
        }
    }

    #[test]
    fn dense_columns_roundtrip_through_horizontal_modes() {
        let budget = DecodeBudget::default();
        let n = 120usize;
        let cols: Vec<Vec<Vec<u8>>> = vec![
            (0..n)
                .map(|i| format!("org{}/service-{}", i % 5, i % 11).into_bytes())
                .collect(),
            (0..n)
                .map(|i| {
                    format!(
                        "https://api.github.com/repos/org{}/service-{}/events",
                        i % 5,
                        i % 11
                    )
                    .into_bytes()
                })
                .collect(),
            (0..n)
                .map(|i| {
                    (1_700_000_000_000i64 + i as i64 * 50)
                        .to_string()
                        .into_bytes()
                })
                .collect(),
            (0..n)
                .map(|i| (12 + i as i64 % 7).to_string().into_bytes())
                .collect(),
            (0..n)
                .map(|i| {
                    (1_700_000_000_000i64 + i as i64 * 50 + 12 + i as i64 % 7)
                        .to_string()
                        .into_bytes()
                })
                .collect(),
            (0..n)
                .map(|i| {
                    format!(
                        "2026-11-10T11:24:{:02}.{:03}Z",
                        (i / 10) % 60,
                        (i * 100) % 1000
                    )
                    .into_bytes()
                })
                .collect(),
            (0..n)
                .map(|i| format!("\"note {} about a thing that happened\"", i * 7919).into_bytes())
                .collect(),
            (0..n)
                .map(|i| {
                    if i % 3 == 0 {
                        b"null".to_vec()
                    } else {
                        format!("\"u{}\"", i % 4).into_bytes()
                    }
                })
                .collect(),
        ];
        let streams: Vec<Vec<&[u8]>> = cols
            .iter()
            .map(|c| c.iter().map(|v| v.as_slice()).collect())
            .collect();
        let groups = vec![""; cols.len()];
        let cands: Vec<Vec<usize>> = (0..cols.len()).map(|ci| (0..ci).collect()).collect();
        let mut nested = vec![None; cols.len()];
        let alone = vec![false; cols.len()];
        let enc = encode_dense_block_columns(
            &streams,
            &groups,
            &cands,
            &mut nested,
            &alone,
            EntropyBackend::Zstd,
            3,
        );
        // The byte after the one-byte occurrence count is the horizontal mode.
        assert_eq!(
            enc.col_blobs[1][1], 2,
            "the URL column should interpolate the repo"
        );
        assert_eq!(
            enc.col_blobs[4][1], 3,
            "the end column should be start + duration"
        );
        assert_eq!(column_deps(&enc.col_blobs[1]).cols, vec![0]);
        assert_eq!(column_deps(&enc.col_blobs[4]).cols, vec![2, 3]);

        let blobs: Vec<Option<&[u8]>> = enc.col_blobs.iter().map(|b| Some(b.as_slice())).collect();
        let dicts: Vec<Option<&[u8]>> = enc
            .shared_dict_blobs
            .iter()
            .map(|b| Some(b.as_slice()))
            .collect();
        let mut ctx = BlockDecodeCtx::new(blobs, dicts, enc.prose_pool_leaf.as_deref());
        let mut scratch = Vec::new();
        for (ci, want) in cols.iter().enumerate() {
            let got = ensure_dense_block_column(ci, &mut ctx, &budget).unwrap();
            assert_eq!(got.len(), n);
            for (r, w) in want.iter().enumerate() {
                assert_eq!(
                    got.get(r, &mut scratch).unwrap(),
                    w.as_slice(),
                    "col {ci} row {r}"
                );
            }
            assert!(got.get(n, &mut scratch).is_err());
        }
    }
}
