//! Self-describing column payload graph (`PayloadGraph`) and the zero-knowledge [`CoreDecoder`].
//!
//! [`CoreDecoder`] knows nothing about timestamps, UUIDs, ULIDs, IPs, emails, decimals, floats,
//! Java LCGs, or MT19937: every codec pack compiles down to a [`PayloadGraph`] of domain-agnostic
//! byte, integer, recurrence, and SSA lane-program primitives, while returning lazy [`Col`] views
//! (`Const`, `Dict`, `Pool`, `Quoted`, `Flat`) so selective queries and skipped rows stay fast.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use crate::cells::{Cells, Col, NOT_HELD, NULL_CELL, Pool};
use crate::dictionary::with_current;
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, compress_leaf, compress_leaf_with,
    decompress_leaf_with, put_ivarint, put_uvarint,
};

use super::engine::CompiledProgram;
use super::ir::{Builder, Invalid, Len, Op, Program, TableBuf, Width};
use super::library::{Code, Library, code_discount, table_bytes};
use super::recur::{CompiledRecurrence, RecurrenceSpec};
use super::tape::{self, Site};

thread_local! {
    static ENC_OPTS: Cell<(EntropyBackend, i32)> =
        const { Cell::new((EntropyBackend::Zstd, 3)) };
    static CONTEXT_MIXING: Cell<bool> = const { Cell::new(false) };
    static ENCODE_MEMO: RefCell<Option<EncodeMemo>> = const { RefCell::new(None) };
}

/// Candidate encoders of one block often build the same integer stream several times;
/// these results are pure functions of the values and the encode options in scope.
#[derive(Default)]
struct EncodeMemo {
    ints: FastMap<u64, Vec<(IntKey, Box<[i64]>, IntStream)>>,
    held: usize,
}

type IntKey = (bool, EntropyBackend, i32);

/// Values a memo may hold before it stops recording; it bounds memory, not output.
const MEMO_MAX_HELD: usize = 1 << 22;
const MEMO_MIN_LEN: usize = 32;

/// Shares identical encoder work for the lifetime of the guard. Nested scopes reuse the outer memo.
pub(crate) struct EncodeMemoScope(bool);

impl EncodeMemoScope {
    pub(crate) fn new() -> Self {
        Self(ENCODE_MEMO.with(|m| {
            let mut m = m.borrow_mut();
            let fresh = m.is_none();
            if fresh {
                *m = Some(EncodeMemo::default());
            }
            fresh
        }))
    }
}

impl Drop for EncodeMemoScope {
    fn drop(&mut self) {
        if self.0 {
            ENCODE_MEMO.with(|m| m.borrow_mut().take());
        }
    }
}

fn int_key() -> IntKey {
    let (backend, level) = ENC_OPTS.with(Cell::get);
    (CONTEXT_MIXING.with(Cell::get), backend, level)
}

fn hash_ints(vals: &[i64]) -> u64 {
    use core::hash::Hasher;
    let mut h = FastHasher::default();
    for &v in vals {
        h.write_i64(v);
    }
    h.write_usize(vals.len());
    h.finish()
}

/// Enables the bounded integer context-mixing candidate within `f`. The fast numeric path
/// remains unchanged. State belongs to each stream; nothing carries across payloads or blocks.
pub fn with_context_mixing<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let _guard = ContextMixScope::new(enabled);
    f()
}

pub(crate) struct ContextMixScope(bool);

impl ContextMixScope {
    pub(crate) fn new(enabled: bool) -> Self {
        Self(CONTEXT_MIXING.with(|c| c.replace(enabled)))
    }
}

impl Drop for ContextMixScope {
    fn drop(&mut self) {
        CONTEXT_MIXING.with(|c| c.set(self.0));
    }
}

/// Runs `f` with `(backend, level)` active as the default leaf entropy settings for [`GraphBuilder`].
pub fn with_encode_opts<R>(backend: EntropyBackend, level: i32, f: impl FnOnce() -> R) -> R {
    struct Guard((EntropyBackend, i32));
    impl Drop for Guard {
        fn drop(&mut self) {
            ENC_OPTS.with(|c| c.set(self.0));
        }
    }
    let prev = ENC_OPTS.with(|c| {
        let old = c.get();
        c.set((backend, level));
        old
    });
    let _guard = Guard(prev);
    f()
}

/// A leaf of an integer stream. It is not compressed against the dictionary, whose values it
/// would rarely match for more than the search costs.
#[inline]
pub(crate) fn pack_leaf(raw: &[u8]) -> Vec<u8> {
    let (backend, level) = ENC_OPTS.with(|c| c.get());
    let mut out = Vec::with_capacity(raw.len() + 12);
    compress_leaf(&mut out, raw, backend, level);
    out
}

/// The level a column search prices leaves at; the winner is recompressed at `level`. Level 3
/// ranks candidates as the middle levels do, at about a quarter less work on logs; the highest
/// levels search for ratio and price at their own.
pub(crate) fn search_level(level: i32) -> i32 {
    match level {
        4..=9 => 3,
        _ => level,
    }
}

/// `leaf` compressed afresh at `level`; `text` leaves are compressed against the dictionary.
pub(crate) fn relevel_leaf(
    leaf: &[u8],
    backend: EntropyBackend,
    level: i32,
    text: bool,
) -> Result<Vec<u8>> {
    // A raw leaf may be a table a library shares by its bytes; other backends keep theirs.
    let zstd = [EntropyBackend::Zstd as u8, EntropyBackend::ZstdDict as u8];
    if !leaf.first().is_some_and(|t| zstd.contains(t)) {
        return Ok(leaf.to_vec());
    }
    let raw = unpack_leaf(leaf, &DecodeBudget::default())?;
    let mut out = Vec::with_capacity(leaf.len() + 12);
    match text {
        true => with_current(|dict| compress_leaf_with(&mut out, &raw, backend, level, dict)),
        false => compress_leaf(&mut out, &raw, backend, level),
    }
    Ok(out)
}

/// A leaf of bytes, compressed against the dictionary in scope.
#[inline]
pub(crate) fn pack_text_leaf(raw: &[u8]) -> Vec<u8> {
    let (backend, level) = ENC_OPTS.with(|c| c.get());
    let mut out = Vec::with_capacity(raw.len() + 12);
    with_current(|dict| compress_leaf_with(&mut out, raw, backend, level, dict));
    out
}

#[inline]
pub(crate) fn unpack_leaf(leaf: &[u8], budget: &DecodeBudget) -> Result<Vec<u8>> {
    let mut cur = Cursor::new(leaf);
    let raw = with_current(|dict| decompress_leaf_with(&mut cur, budget, dict))?;
    if cur.remaining() != 0 {
        return Err(Error::Format("trailing bytes in payload leaf".into()));
    }
    Ok(raw)
}

impl From<Invalid> for Error {
    fn from(e: Invalid) -> Self {
        Error::Format(e.0.into())
    }
}

/// Fast non-cryptographic 64-bit multiply-xor hasher (`FxHash`-style, 8B chunks) for encoder tables.
#[derive(Default, Clone)]
pub(crate) struct FastHasher(u64);

impl core::hash::Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // Hash-table buckets use low bits. Big-endian text suffixes must influence them,
        // otherwise zero-padded identifiers collide despite differing upper hash bits.
        let mut h = self.0;
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^ (h >> 31)
    }
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        let mut h = self.0;
        while bytes.len() >= 8 {
            let w = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            h = (h.rotate_left(5) ^ w).wrapping_mul(K);
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let w = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64;
            h = (h.rotate_left(5) ^ w).wrapping_mul(K);
            bytes = &bytes[4..];
        }
        for &b in bytes {
            h = (h.rotate_left(5) ^ (b as u64)).wrapping_mul(K);
        }
        self.0 = h;
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

pub(crate) type FastMap<K, V> =
    std::collections::HashMap<K, V, core::hash::BuildHasherDefault<FastHasher>>;
pub(crate) type FastSet<K> =
    std::collections::HashSet<K, core::hash::BuildHasherDefault<FastHasher>>;

pub(crate) fn fast_map_with_capacity<K, V>(cap: usize) -> FastMap<K, V> {
    FastMap::with_capacity_and_hasher(cap, core::hash::BuildHasherDefault::default())
}

/// Each value less the one before it, the first less zero taken as zero.
pub(crate) fn deltas(vals: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(vals.len());
    out.push(0);
    out.extend(vals.windows(2).map(|w| w[1].wrapping_sub(w[0])));
    out
}

/// Domain-agnostic integer sequence transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntStream {
    Const {
        val: i64,
        len: u32,
    },
    Rle {
        len: u32,
        leaf: Vec<u8>,
    },
    ForBitpack {
        len: u32,
        min: i64,
        bits: u8,
        leaf: Vec<u8>,
    },
    ZigZagVarint {
        len: u32,
        leaf: Vec<u8>,
    },
    Radix {
        len: u32,
        min: i64,
        radix: u32,
        bytes: Vec<u8>,
    },
    Rice {
        len: u32,
        bytes: Vec<u8>,
    },
    NibbleMix {
        len: u32,
        min: i64,
        bits: u8,
        table_bits: u8,
        escape: bool,
        bytes: Vec<u8>,
    },
    ForPlanes {
        len: u32,
        min: i64,
        width: u8,
        leaf: Vec<u8>,
    },
    ZigZagBytes {
        len: u32,
        head_leaf: Vec<u8>,
        tail: Box<IntStream>,
    },
    ForTail255 {
        len: u32,
        min: i64,
        head_leaf: Vec<u8>,
        tail: Box<IntStream>,
    },
}

impl IntStream {
    fn relevel(&mut self, backend: EntropyBackend, level: i32) -> Result<()> {
        match self {
            Self::Rle { leaf, .. }
            | Self::ForBitpack { leaf, .. }
            | Self::ZigZagVarint { leaf, .. }
            | Self::ForPlanes { leaf, .. } => *leaf = relevel_leaf(leaf, backend, level, false)?,
            Self::ZigZagBytes {
                head_leaf, tail, ..
            }
            | Self::ForTail255 {
                head_leaf, tail, ..
            } => {
                *head_leaf = relevel_leaf(head_leaf, backend, level, false)?;
                tail.relevel(backend, level)?;
            }
            Self::Const { .. }
            | Self::Radix { .. }
            | Self::Rice { .. }
            | Self::NibbleMix { .. } => {}
        }
        Ok(())
    }

    /// Automatically chooses the smallest domain-agnostic encoding for `vals`.
    pub fn auto_encode(vals: &[i64]) -> Self {
        if vals.len() < MEMO_MIN_LEN || ENCODE_MEMO.with(|m| m.borrow().is_none()) {
            return Self::auto_encode_uncached(vals);
        }
        let key = int_key();
        let h = hash_ints(vals);
        let hit = ENCODE_MEMO.with(|m| {
            let m = m.borrow();
            let entries = m.as_ref()?.ints.get(&h)?;
            entries
                .iter()
                .find(|(k, v, _)| *k == key && **v == *vals)
                .map(|(_, _, s)| s.clone())
        });
        if let Some(s) = hit {
            return s;
        }
        let s = Self::auto_encode_uncached(vals);
        ENCODE_MEMO.with(|m| {
            if let Some(m) = m.borrow_mut().as_mut()
                && m.held + vals.len() <= MEMO_MAX_HELD
            {
                m.held += vals.len();
                m.ints
                    .entry(h)
                    .or_default()
                    .push((key, vals.into(), s.clone()));
            }
        });
        s
    }

    fn auto_encode_uncached(vals: &[i64]) -> Self {
        let base = Self::auto_encode_inner(vals, true);
        if CONTEXT_MIXING.with(Cell::get) {
            Self::consider_context_mix(vals, base)
        } else {
            base
        }
    }

    fn consider_context_mix(vals: &[i64], base: Self) -> Self {
        if vals.len() < 512 || matches!(base, Self::Const { .. }) {
            return base;
        }
        let min = *vals.iter().min().unwrap();
        let max = *vals.iter().max().unwrap();
        let span = (max as u64).wrapping_sub(min as u64);
        let bits = (64 - span.leading_zeros()) as u8;
        if !(1..=16).contains(&bits) {
            return base;
        }
        let mut wire = Vec::new();
        base.encode(&mut wire);
        if wire.len() < 128 {
            return base;
        }
        let target = wire.len() - 32.max(wire.len() / 50);
        let table_bits = (usize::BITS - vals.len().leading_zeros()).clamp(8, 14) as u8;
        let escape = super::nibble_mix::escapes(vals, min, bits);
        let mut enc = super::nibble_mix::Encoder::new(min, bits, table_bits, escape);
        let mut rest = vals;
        if vals.len() > 2048 {
            let sample;
            (sample, rest) = vals.split_at(2048);
            let sample_base = Self::auto_encode_inner(sample, true);
            wire.clear();
            sample_base.encode(&mut wire);
            // A small warm-up loss can recover on a longer lane; clear losses stop here.
            let limit = wire.len() + wire.len() / 8;
            // The full lane goes on from the sample's state, which must fit `target` too.
            if !enc.push(sample, limit) || !enc.fits(limit) || !enc.fits(target) {
                return base;
            }
        }
        if !enc.push(rest, target) {
            return base;
        }
        let Some(bytes) = enc.finish(target) else {
            return base;
        };
        let candidate = Self::NibbleMix {
            len: vals.len() as u32,
            min,
            bits,
            table_bits,
            escape,
            bytes,
        };
        wire.clear();
        candidate.encode(&mut wire);
        if wire.len() <= target {
            candidate
        } else {
            base
        }
    }

    fn auto_encode_inner(vals: &[i64], allow_tail: bool) -> Self {
        let len = vals.len() as u32;
        if vals.is_empty() {
            return Self::Const { val: 0, len: 0 };
        }
        // One pass for the run count and range; a single run is a constant.
        let (mut runs, mut min, mut max) = (1usize, vals[0], vals[0]);
        for (&a, &b) in vals.iter().zip(&vals[1..]) {
            runs += usize::from(a != b);
            min = min.min(b);
            max = max.max(b);
        }
        if runs == 1 {
            return Self::Const { val: vals[0], len };
        }
        let mut best: Option<(usize, Self)> = None;
        if runs * 3 < vals.len() {
            let mut raw = Vec::with_capacity(runs * 5);
            put_uvarint(&mut raw, runs as u64);
            let mut i = 0;
            while i < vals.len() {
                let v = vals[i];
                let mut j = i + 1;
                while j < vals.len() && vals[j] == v {
                    j += 1;
                }
                put_ivarint(&mut raw, v);
                put_uvarint(&mut raw, (j - i) as u64);
                i = j;
            }
            let leaf = pack_leaf(&raw);
            best = Some((leaf.len() + 4, Self::Rle { len, leaf }));
        }

        let span = (max as u128).wrapping_sub(min as u128);
        if span <= u64::MAX as u128 {
            let max_u = span as u64;
            let bits = (64 - max_u.leading_zeros()) as u8;
            let packed = bitpack_u64(
                vals.iter().map(|&v| (v as u64).wrapping_sub(min as u64)),
                bits,
                vals.len(),
            );
            let leaf = pack_leaf(&packed);
            let cost = leaf.len() + 10;
            if best.as_ref().is_none_or(|(b, _)| cost < *b) {
                best = Some((
                    cost,
                    Self::ForBitpack {
                        len,
                        min,
                        bits,
                        leaf,
                    },
                ));
            }

            if vals.len() >= 24 && bits != 8 {
                let bw: u8 = if max_u <= 0xff {
                    1
                } else if max_u <= 0xffff {
                    2
                } else if max_u <= 0xffff_ffff {
                    4
                } else {
                    8
                };
                let n = vals.len();
                let mut planes = vec![0u8; n * (bw as usize)];
                // A plane at a time keeps the stores sequential.
                for (p, plane) in planes.chunks_exact_mut(n).enumerate() {
                    for (dst, &v) in plane.iter_mut().zip(vals) {
                        *dst = ((v as u64).wrapping_sub(min as u64) >> (8 * p)) as u8;
                    }
                }
                let pay_p = pack_leaf(&planes);
                let p_cost = pay_p.len() + 10;
                if best.as_ref().is_none_or(|(b, _)| p_cost < *b) {
                    best = Some((
                        p_cost,
                        Self::ForPlanes {
                            len,
                            min,
                            width: bw,
                            leaf: pay_p,
                        },
                    ));
                }
            }

            if allow_tail && vals.len() >= 64 && max_u > 0xff && max_u <= (i64::MAX as u64) {
                let probe = &vals[..vals.len().min(128)];
                let probe_over = probe
                    .iter()
                    .filter(|&&v| (v as u64).wrapping_sub(min as u64) >= 0xff)
                    .count();
                if probe_over * 5 <= probe.len() * 3 {
                    let mut head_raw = Vec::with_capacity(vals.len());
                    let mut over = Vec::new();
                    for &v in vals {
                        let s = (v as u64).wrapping_sub(min as u64);
                        head_raw.push(s.min(0xff) as u8);
                        if s >= 0xff {
                            over.push((s - 0xff) as i64);
                        }
                    }
                    if over.len() * 2 <= vals.len() {
                        let head_leaf = pack_leaf(&head_raw);
                        let tail_s = Self::auto_encode_inner(&over, false);
                        let mut tb = Vec::new();
                        tail_s.encode(&mut tb);
                        let t_cost = head_leaf.len() + tb.len() + 10;
                        if best.as_ref().is_none_or(|(b, _)| t_cost < *b) {
                            best = Some((
                                t_cost,
                                Self::ForTail255 {
                                    len,
                                    min,
                                    head_leaf,
                                    tail: Box::new(tail_s),
                                },
                            ));
                        }
                    }
                }
            }
        }

        if span > u64::MAX as u128 || min < 0 || vals.len() < 24 {
            let mut zz = Vec::with_capacity(vals.len() * 2);
            for &v in vals {
                put_ivarint(&mut zz, v);
            }
            let leaf = pack_leaf(&zz);
            let cost = leaf.len() + 3;
            if best.as_ref().is_none_or(|(b, _)| cost < *b) {
                best = Some((cost, Self::ZigZagVarint { len, leaf }));
            }
        }

        best.unwrap().1
    }

    /// Linear-time selection for latency-sensitive native numeric columns. Uses the same
    /// generic streams as exhaustive selection, with bounded trials for short streams.
    pub fn encode_fast(vals: &[i64]) -> Self {
        let len = vals.len() as u32;
        if vals.is_empty() {
            return Self::Const { val: 0, len };
        }
        let (mut min, mut max, mut runs) = (vals[0], vals[0], 1usize);
        for w in vals.windows(2) {
            min = min.min(w[1]);
            max = max.max(w[1]);
            runs += usize::from(w[0] != w[1]);
        }
        if min == max {
            return Self::Const { val: min, len };
        }
        if runs * 16 < vals.len() {
            let mut raw = Vec::with_capacity(runs * 12);
            put_uvarint(&mut raw, runs as u64);
            let mut i = 0;
            while i < vals.len() {
                let value = vals[i];
                let start = i;
                while i < vals.len() && vals[i] == value {
                    i += 1;
                }
                put_ivarint(&mut raw, value);
                put_uvarint(&mut raw, (i - start) as u64);
            }
            return Self::Rle {
                len,
                leaf: pack_leaf(&raw),
            };
        }
        let bits = (64 - (max as u64).wrapping_sub(min as u64).leading_zeros()) as u8;
        let varbytes = vals
            .iter()
            .map(|&v| {
                ((64 - (((v << 1) ^ (v >> 63)) as u64).leading_zeros()).max(1) as usize).div_ceil(7)
            })
            .sum::<usize>();
        let candidate = if varbytes + 8 < vals.len() * bits as usize / 8 {
            let mut head = Vec::with_capacity(vals.len());
            let mut tail = Vec::new();
            for &v in vals {
                let z = ((v << 1) ^ (v >> 63)) as u64;
                head.push(z.min(255) as u8);
                if z >= 255 {
                    tail.push((z - 255) as i64);
                }
            }
            Self::ZigZagBytes {
                len,
                head_leaf: pack_leaf(&head),
                tail: Box::new(Self::auto_encode_inner(&tail, false)),
            }
        } else {
            let raw = bitpack_u64(
                vals.iter().map(|&v| (v as u64).wrapping_sub(min as u64)),
                bits,
                vals.len(),
            );
            Self::ForBitpack {
                len,
                min,
                bits,
                leaf: pack_leaf(&raw),
            }
        };
        if vals.len() <= 1024 {
            let mut candidate = candidate;
            let mut framed = Vec::new();
            candidate.encode(&mut framed);
            // Varints preserve byte-aligned repeated differences, including rare jumps.
            // Their compressed form can win even when the uncompressed integers need many bytes.
            if varbytes < vals.len() * 5
                && (varbytes + 5 < framed.len()
                    || vals.len() <= 128 && varbytes <= 2 * framed.len())
            {
                let mut varints = Vec::with_capacity(varbytes);
                for &v in vals {
                    put_ivarint(&mut varints, v);
                }
                let leaf = pack_leaf(&varints);
                if leaf.len() + 4 < framed.len() {
                    candidate = Self::ZigZagVarint { len, leaf };
                    framed.clear();
                    candidate.encode(&mut framed);
                }
            }
            let radix = u32::try_from(i128::from(max) - i128::from(min) + 1)
                .ok()
                .and_then(super::radix::Layout::new);
            let cost = radix.as_ref().map_or(usize::MAX, |layout| {
                let signed_min = ((min << 1) ^ (min >> 63)) as u64;
                let min_bytes = (64 - signed_min.leading_zeros()).max(1).div_ceil(7);
                let radix_bytes = (32 - layout.radix.leading_zeros()).div_ceil(7);
                layout.size(vals.len()) + (min_bytes + radix_bytes) as usize + 4
            });
            if let Some(bytes) = super::rice::encode(vals, framed.len().min(cost).saturating_sub(6))
            {
                return Self::Rice { len, bytes };
            }
            if let Some(layout) = radix.filter(|_| cost < framed.len()) {
                return Self::Radix {
                    len,
                    min,
                    radix: layout.radix,
                    bytes: layout.pack(vals, min),
                };
            }
            return candidate;
        }
        candidate
    }

    pub fn len(&self) -> usize {
        match *self {
            Self::Const { len, .. }
            | Self::Rle { len, .. }
            | Self::Radix { len, .. }
            | Self::Rice { len, .. }
            | Self::NibbleMix { len, .. }
            | Self::ForBitpack { len, .. }
            | Self::ZigZagVarint { len, .. }
            | Self::ForPlanes { len, .. }
            | Self::ZigZagBytes { len, .. }
            | Self::ForTail255 { len, .. } => len as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::NibbleMix {
                len,
                min,
                bits,
                table_bits,
                escape,
                bytes,
            } => {
                out.push(10);
                put_uvarint(out, *len as u64);
                put_ivarint(out, *min);
                out.extend_from_slice(&[*bits, *table_bits, *escape as u8]);
                put_uvarint(out, bytes.len() as u64);
                out.extend_from_slice(bytes);
            }
            Self::Const { val, len } => {
                let shift = val.trailing_zeros().min(63);
                let coefficient = *val >> shift;
                let size = |v: i64| {
                    (64 - (((v << 1) ^ (v >> 63)) as u64).leading_zeros())
                        .max(1)
                        .div_ceil(7)
                };
                if shift > 0 && 1 + size(coefficient) < size(*val) {
                    out.push(7);
                    put_uvarint(out, *len as u64);
                    out.push(shift as u8);
                    put_ivarint(out, coefficient);
                } else {
                    out.push(0);
                    put_uvarint(out, *len as u64);
                    put_ivarint(out, *val);
                }
            }
            Self::Radix {
                len,
                min,
                radix,
                bytes,
            } => {
                out.push(11);
                put_uvarint(out, *len as u64);
                put_ivarint(out, *min);
                put_uvarint(out, u64::from(*radix));
                out.extend_from_slice(bytes);
            }
            Self::Rice { len, bytes } => {
                out.push(8);
                put_uvarint(out, *len as u64);
                put_uvarint(out, bytes.len() as u64);
                out.extend_from_slice(bytes);
            }
            Self::Rle { len, leaf } => {
                out.push(1);
                put_uvarint(out, *len as u64);
                put_uvarint(out, leaf.len() as u64);
                out.extend_from_slice(leaf);
            }
            Self::ForBitpack {
                len,
                min,
                bits,
                leaf,
            } => {
                out.push(2);
                put_uvarint(out, *len as u64);
                put_ivarint(out, *min);
                out.push(*bits);
                put_uvarint(out, leaf.len() as u64);
                out.extend_from_slice(leaf);
            }
            Self::ZigZagVarint { len, leaf } => {
                out.push(3);
                put_uvarint(out, *len as u64);
                put_uvarint(out, leaf.len() as u64);
                out.extend_from_slice(leaf);
            }
            Self::ForPlanes {
                len,
                min,
                width,
                leaf,
            } => {
                out.push(4);
                put_uvarint(out, *len as u64);
                put_ivarint(out, *min);
                out.push(*width);
                put_uvarint(out, leaf.len() as u64);
                out.extend_from_slice(leaf);
            }
            Self::ZigZagBytes {
                len,
                head_leaf,
                tail,
            } => {
                out.push(6);
                put_uvarint(out, *len as u64);
                put_uvarint(out, head_leaf.len() as u64);
                out.extend_from_slice(head_leaf);
                tail.encode(out);
            }
            Self::ForTail255 {
                len,
                min,
                head_leaf,
                tail,
            } => {
                out.push(5);
                put_uvarint(out, *len as u64);
                put_ivarint(out, *min);
                put_uvarint(out, head_leaf.len() as u64);
                out.extend_from_slice(head_leaf);
                tail.encode(out);
            }
        }
    }

    pub fn decode_wire(cur: &mut Cursor<'_>, budget: &DecodeBudget, depth: usize) -> Result<Self> {
        if depth > 1 {
            return Err(Error::Format("nested IntStream tail".into()));
        }
        let kind = cur.u8()?;
        let len_u64 = cur.uvar()?;
        if len_u64 > u32::MAX as u64 || len_u64 > (budget.max_single_alloc_bytes / 8) as u64 {
            return Err(Error::BudgetExceeded("IntStream length"));
        }
        let len = len_u64 as u32;
        match kind {
            10 => {
                let min = cur.ivar()?;
                let bits = cur.u8()?;
                let table_bits = cur.u8()?;
                let escape = match cur.u8()? {
                    0 => false,
                    1 => true,
                    _ => return Err(Error::Format("invalid nibble escape flag".into())),
                };
                let bytes = cur.bytes()?;
                super::nibble_mix::validate(
                    len as usize,
                    bits,
                    table_bits,
                    escape,
                    bytes.len(),
                    budget,
                )?;
                budget.check_alloc(bytes.len(), "nibble mixing bytes")?;
                Ok(Self::NibbleMix {
                    len,
                    min,
                    bits,
                    table_bits,
                    escape,
                    bytes: bytes.to_vec(),
                })
            }
            11 => {
                let min = cur.ivar()?;
                let radix = u32::try_from(cur.uvar()?)
                    .map_err(|_| Error::Format("radix too large".into()))?;
                let layout = super::radix::Layout::new(radix)
                    .ok_or_else(|| Error::Format("invalid radix".into()))?;
                let size = layout.size(len as usize);
                budget.check_alloc(size, "radix bytes")?;
                Ok(Self::Radix {
                    len,
                    min,
                    radix,
                    bytes: cur.slice(size)?.to_vec(),
                })
            }
            8 => Ok(Self::Rice {
                len,
                bytes: cur.bytes()?.to_vec(),
            }),
            7 => {
                let shift = cur.u8()?;
                if shift == 0 || shift >= 64 {
                    return Err(Error::Format("invalid constant shift".into()));
                }
                let coefficient = cur.ivar()?;
                let val = coefficient.wrapping_shl(shift as u32);
                if val >> shift != coefficient {
                    return Err(Error::Format("constant shift overflow".into()));
                }
                Ok(Self::Const { val, len })
            }
            0 => {
                let val = cur.ivar()?;
                Ok(Self::Const { val, len })
            }
            1 => {
                let leaf = cur.bytes()?.to_vec();
                Ok(Self::Rle { len, leaf })
            }
            2 => {
                let min = cur.ivar()?;
                let bits = cur.u8()?;
                if bits > 64 {
                    return Err(Error::Format("IntStream bitwidth > 64".into()));
                }
                let leaf = cur.bytes()?.to_vec();
                Ok(Self::ForBitpack {
                    len,
                    min,
                    bits,
                    leaf,
                })
            }
            3 => {
                let leaf = cur.bytes()?.to_vec();
                Ok(Self::ZigZagVarint { len, leaf })
            }
            4 => {
                let min = cur.ivar()?;
                let width = cur.u8()?;
                if !matches!(width, 1 | 2 | 4 | 8) {
                    return Err(Error::Format("invalid ForPlanes width".into()));
                }
                let leaf = cur.bytes()?.to_vec();
                Ok(Self::ForPlanes {
                    len,
                    min,
                    width,
                    leaf,
                })
            }
            6 => {
                let head_leaf = cur.bytes()?.to_vec();
                let tail = Box::new(Self::decode_wire(cur, budget, depth + 1)?);
                Ok(Self::ZigZagBytes {
                    len,
                    head_leaf,
                    tail,
                })
            }
            5 => {
                let min = cur.ivar()?;
                let head_leaf = cur.bytes()?.to_vec();
                let tail = Box::new(Self::decode_wire(cur, budget, depth + 1)?);
                Ok(Self::ForTail255 {
                    len,
                    min,
                    head_leaf,
                    tail,
                })
            }
            _ => Err(Error::Format("unknown IntStream kind".into())),
        }
    }

    pub fn unpack(&self, budget: &DecodeBudget) -> Result<Vec<i64>> {
        let n = self.len();
        budget.check_alloc(n.saturating_mul(8), "IntStream")?;
        match self {
            Self::Const { val, len } => Ok(vec![*val; *len as usize]),
            Self::Radix {
                len,
                min,
                radix,
                bytes,
            } => super::radix::Layout::new(*radix)
                .ok_or_else(|| Error::Format("invalid radix".into()))?
                .unpack(bytes, *len as usize, *min),
            Self::Rice { len, bytes } => super::rice::decode(bytes, *len as usize, budget),
            Self::NibbleMix {
                len,
                min,
                bits,
                table_bits,
                escape,
                bytes,
            } => super::nibble_mix::decode(
                bytes,
                *len as usize,
                *min,
                *bits,
                *table_bits,
                *escape,
                budget,
            ),
            Self::Rle { len, leaf } => {
                let n = *len as usize;
                let raw = unpack_leaf(leaf, budget)?;
                let mut s = Cursor::new(&raw);
                let runs = s.uvar()? as usize;
                let mut out = Vec::with_capacity(n);
                for _ in 0..runs {
                    let v = s.ivar()?;
                    let c = s.uvar()? as usize;
                    if out.len().saturating_add(c) > n {
                        return Err(Error::Format("RLE run exceeds stream length".into()));
                    }
                    out.resize(out.len() + c, v);
                }
                if out.len() != n || s.remaining() != 0 {
                    return Err(Error::Format("RLE length mismatch".into()));
                }
                Ok(out)
            }
            Self::ForBitpack {
                len,
                min,
                bits,
                leaf,
            } => {
                let n = *len as usize;
                let need_bytes = (n * (*bits as usize)).div_ceil(8);
                let raw = unpack_leaf(leaf, budget)?;
                if raw.len() < need_bytes {
                    return Err(Error::Format("truncated bitpack payload".into()));
                }
                Ok(bitunpack_u64(raw, *bits, n, *min))
            }
            Self::ZigZagVarint { len, leaf } => {
                let n = *len as usize;
                let raw = unpack_leaf(leaf, budget)?;
                let mut s = Cursor::new(&raw);
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    out.push(s.ivar()?);
                }
                if s.remaining() != 0 {
                    return Err(Error::Format("trailing bytes in ZigZagVarint".into()));
                }
                Ok(out)
            }
            Self::ForPlanes {
                len,
                min,
                width,
                leaf,
            } => {
                let n = *len as usize;
                let w = *width as usize;
                let need = n
                    .checked_mul(w)
                    .ok_or_else(|| Error::Format("ForPlanes overflow".into()))?;
                let raw = unpack_leaf(leaf, budget)?;
                if raw.len() != need {
                    return Err(Error::Format("ForPlanes payload length mismatch".into()));
                }
                let mut out = vec![*min; n];
                // One pass per plane: each is a simple loop that vectorizes, where gathering
                // a lane's bytes from all planes at once does not.
                for (k, plane) in raw.chunks_exact(n.max(1)).enumerate() {
                    for (o, &b) in out.iter_mut().zip(plane) {
                        *o = o.wrapping_add(((b as u64) << (8 * k)) as i64);
                    }
                }
                Ok(out)
            }
            Self::ZigZagBytes {
                len,
                head_leaf,
                tail,
            } => {
                let head = unpack_leaf(head_leaf, budget)?;
                if head.len() != *len as usize {
                    return Err(Error::Format("zigzag byte count mismatch".into()));
                }
                let tail = tail.unpack(budget)?;
                let mut patches = tail.iter();
                let mut out = Vec::with_capacity(head.len());
                if tail.is_empty() {
                    if head.contains(&255) {
                        return Err(Error::Format("missing zigzag byte exception".into()));
                    }
                    out.extend(head.iter().map(|&v| ((v >> 1) as i64) ^ -((v & 1) as i64)));
                } else {
                    for byte in head {
                        let z = if byte == 255 {
                            (*patches.next().ok_or_else(|| {
                                Error::Format("missing zigzag byte exception".into())
                            })? as u64)
                                .checked_add(255)
                                .ok_or_else(|| {
                                    Error::Format("zigzag byte exception overflow".into())
                                })?
                        } else {
                            byte as u64
                        };
                        out.push(((z >> 1) as i64) ^ -((z & 1) as i64));
                    }
                    if patches.next().is_some() {
                        return Err(Error::Format("unused zigzag byte exceptions".into()));
                    }
                }
                Ok(out)
            }
            Self::ForTail255 {
                len,
                min,
                head_leaf,
                tail,
            } => {
                let n = *len as usize;
                let h_raw = unpack_leaf(head_leaf, budget)?;
                if h_raw.len() != n {
                    return Err(Error::Format("ForTail255 head length mismatch".into()));
                }
                let over = tail.unpack(budget)?;
                let mut out: Vec<i64> = h_raw.iter().map(|&b| min.wrapping_add(b as i64)).collect();
                // Exceptions are rare: find them a word at a time and add their tail values.
                let mut tail = over.iter();
                let chunks = h_raw.chunks_exact(8);
                let mut rest = [0u8; 8];
                rest[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
                let words = chunks.map(|c| u64::from_le_bytes(c.try_into().unwrap()));
                for (k, x) in words.chain([u64::from_le_bytes(rest)]).enumerate() {
                    // The high bit of each 0xff byte: only 0x7f carries into bit 7.
                    let mut m = x & ((x & 0x7f7f_7f7f_7f7f_7f7f) + 0x0101_0101_0101_0101);
                    m &= 0x8080_8080_8080_8080;
                    while m != 0 {
                        let i = 8 * k + m.trailing_zeros() as usize / 8;
                        match tail.next() {
                            Some(&t) if t >= 0 => out[i] = out[i].wrapping_add(t),
                            _ => {
                                return Err(Error::Format(
                                    "ForTail255 tail underflow or negative".into(),
                                ));
                            }
                        }
                        m &= m - 1;
                    }
                }
                if tail.next().is_some() {
                    return Err(Error::Format("ForTail255 trailing tail items".into()));
                }
                Ok(out)
            }
        }
    }
}

/// Initial state of a key before its first delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumSeed {
    Zero,
    /// Retains the cross-key initialization used by original kind-25 payloads.
    Previous,
}

/// One node in a [`PayloadGraph`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphNode {
    Bytes {
        leaf: Vec<u8>,
    },
    IntStream(IntStream),
    ByteChunks {
        buf: u16,
        chunk_bytes: u8,
    },
    MtfFixed {
        codes: u16,
        fallback_buf: u16,
        width: u8,
    },
    RecurBuf {
        spec: RecurrenceSpec,
        seed: u16,
        cols: Vec<u16>,
        out_words: u32,
        include_seed: bool,
    },
    ProgramI64 {
        prog: Code,
        inputs: Vec<u16>,
        bufs: Vec<u16>,
    },
    ProgramCells {
        prog: Code,
        inputs: Vec<u16>,
        bufs: Vec<u16>,
    },
    PatchCells {
        base: u16,
        indices: u16,
        values: u16,
    },
    MtfInts {
        codes: u16,
        misses: u16,
    },
    SliceBuf {
        buf: u16,
        base: u32,
        lens: u16,
    },
    /// Cells of `lens` bytes, each in page `pages[i]` of `buf`: a page's cells lie back to back
    /// in cell order from `bases[page]`.
    PagedSlice {
        buf: u16,
        lens: u16,
        pages: u16,
        bases: u16,
    },
    IndexCells {
        source: u16,
        indices: u16,
    },
    TemplateCells {
        template: u16,
        slots: Vec<u16>,
    },
    ConstCells {
        val: Vec<u8>,
        count: u32,
    },
    QuoteOrNull {
        inner: u16,
        map: Option<u16>,
    },
    ParseAsciiInt {
        cells: u16,
    },
    ExtPool,
    ExtDict {
        dict_id: u16,
    },
    ExtPeer {
        slot: u16,
    },
    /// For each cell of `at` (`0`: this column, `s + 1`: peer slot `s`), the id of the value of
    /// the first cell of peer slot `key` in the same record: the rank of that value's first
    /// occurrence among the key column's distinct values.
    ExtAlign {
        key: u16,
        at: u16,
    },
    /// For each needle, the position in `hay` of the `codes`-th entry equal to it; code `0`
    /// gives `0`.
    MatchIndex {
        needles: u16,
        hay: u16,
        codes: u16,
    },
    /// A wrapping running sum per key. Optional order values determine processing order,
    /// with ties in input order; outputs retain their input positions. Key 0 passes through.
    KeyedSum {
        keys: u16,
        deltas: u16,
        order: Option<u16>,
        seed: SumSeed,
    },
    /// For each cell of this column, the variant id of its record.
    ExtVariant,
    /// Each cell's value id: the rank of its value's first occurrence.
    CellIds {
        cells: u16,
    },
    /// Row `i` is the next value of `streams[sel[i]]`, so each stream codes its own rows.
    Interleave {
        sel: u16,
        streams: Vec<u16>,
    },
}

impl GraphNode {
    #[inline]
    pub fn for_each_dep(&self, mut f: impl FnMut(u16)) {
        match self {
            Self::Bytes { .. }
            | Self::IntStream(_)
            | Self::ConstCells { .. }
            | Self::ExtPool
            | Self::ExtDict { .. }
            | Self::ExtPeer { .. }
            | Self::ExtAlign { .. }
            | Self::ExtVariant => {}
            Self::ByteChunks { buf, .. } => f(*buf),
            Self::KeyedSum { keys, deltas, order, .. } => {
                f(*keys);
                f(*deltas);
                if let Some(order) = order { f(*order); }
            }
            Self::CellIds { cells } => f(*cells),
            Self::Interleave { sel, streams } => {
                f(*sel);
                for &x in streams {
                    f(x);
                }
            }
            Self::MatchIndex {
                needles,
                hay,
                codes,
            } => {
                f(*needles);
                f(*hay);
                f(*codes);
            }
            Self::MtfFixed {
                codes,
                fallback_buf,
                ..
            } => {
                f(*codes);
                f(*fallback_buf);
            }
            Self::RecurBuf { seed, cols, .. } => {
                f(*seed);
                for &c in cols {
                    f(c);
                }
            }
            Self::ProgramI64 { inputs, bufs, .. } | Self::ProgramCells { inputs, bufs, .. } => {
                for &x in inputs {
                    f(x);
                }
                for &x in bufs {
                    f(x);
                }
            }
            Self::PatchCells {
                base,
                indices,
                values,
            } => {
                f(*base);
                f(*indices);
                f(*values);
            }
            Self::MtfInts { codes, misses, .. } => {
                f(*codes);
                f(*misses);
            }
            Self::SliceBuf { buf, lens, .. } => {
                f(*buf);
                f(*lens);
            }
            Self::PagedSlice {
                buf,
                lens,
                pages,
                bases,
            } => {
                f(*buf);
                f(*lens);
                f(*pages);
                f(*bases);
            }
            Self::IndexCells { source, indices } => {
                f(*source);
                f(*indices);
            }
            Self::TemplateCells { template, slots } => {
                f(*template);
                for &s in slots {
                    f(s);
                }
            }
            Self::QuoteOrNull { inner, map } => {
                f(*inner);
                if let Some(m) = map {
                    f(*m);
                }
            }
            Self::ParseAsciiInt { cells } => f(*cells),
        }
    }

    pub fn remap(&self, mut f: impl FnMut(u16) -> u16) -> Self {
        match self {
            Self::Bytes { leaf } => Self::Bytes { leaf: leaf.clone() },
            Self::IntStream(s) => Self::IntStream(s.clone()),
            Self::ByteChunks { buf, chunk_bytes } => Self::ByteChunks {
                buf: f(*buf),
                chunk_bytes: *chunk_bytes,
            },
            Self::MtfFixed {
                codes,
                fallback_buf,
                width,
            } => Self::MtfFixed {
                codes: f(*codes),
                fallback_buf: f(*fallback_buf),
                width: *width,
            },
            Self::RecurBuf {
                spec,
                seed,
                cols,
                out_words,
                include_seed,
            } => Self::RecurBuf {
                spec: spec.clone(),
                seed: f(*seed),
                cols: cols.iter().map(|&c| f(c)).collect(),
                out_words: *out_words,
                include_seed: *include_seed,
            },
            Self::ProgramI64 { prog, inputs, bufs } => Self::ProgramI64 {
                prog: prog.clone(),
                inputs: inputs.iter().map(|&x| f(x)).collect(),
                bufs: bufs.iter().map(|&x| f(x)).collect(),
            },
            Self::ProgramCells { prog, inputs, bufs } => Self::ProgramCells {
                prog: prog.clone(),
                inputs: inputs.iter().map(|&x| f(x)).collect(),
                bufs: bufs.iter().map(|&x| f(x)).collect(),
            },
            Self::PatchCells {
                base,
                indices,
                values,
            } => Self::PatchCells {
                base: f(*base),
                indices: f(*indices),
                values: f(*values),
            },
            Self::MtfInts { codes, misses } => Self::MtfInts {
                codes: f(*codes),
                misses: f(*misses),
            },
            Self::SliceBuf { buf, base, lens } => Self::SliceBuf {
                buf: f(*buf),
                base: *base,
                lens: f(*lens),
            },
            Self::PagedSlice {
                buf,
                lens,
                pages,
                bases,
            } => Self::PagedSlice {
                buf: f(*buf),
                lens: f(*lens),
                pages: f(*pages),
                bases: f(*bases),
            },
            Self::IndexCells { source, indices } => Self::IndexCells {
                source: f(*source),
                indices: f(*indices),
            },
            Self::TemplateCells { template, slots } => Self::TemplateCells {
                template: f(*template),
                slots: slots.iter().map(|&s| f(s)).collect(),
            },
            Self::ConstCells { val, count } => Self::ConstCells {
                val: val.clone(),
                count: *count,
            },
            Self::QuoteOrNull { inner, map } => Self::QuoteOrNull {
                inner: f(*inner),
                map: map.map(&mut f),
            },
            Self::ParseAsciiInt { cells } => Self::ParseAsciiInt { cells: f(*cells) },
            Self::KeyedSum { keys, deltas, order, seed } => Self::KeyedSum {
                keys: f(*keys),
                deltas: f(*deltas),
                order: order.map(&mut f),
                seed: *seed,
            },
            Self::CellIds { cells } => Self::CellIds { cells: f(*cells) },
            Self::Interleave { sel, streams } => Self::Interleave {
                sel: f(*sel),
                streams: streams.iter().map(|&x| f(x)).collect(),
            },
            Self::ExtVariant => Self::ExtVariant,
            Self::ExtPool => Self::ExtPool,
            Self::ExtDict { dict_id } => Self::ExtDict { dict_id: *dict_id },
            Self::ExtPeer { slot } => Self::ExtPeer { slot: *slot },
            Self::ExtAlign { key, at } => Self::ExtAlign { key: *key, at: *at },
            Self::MatchIndex {
                needles,
                hay,
                codes,
            } => Self::MatchIndex {
                needles: f(*needles),
                hay: f(*hay),
                codes: f(*codes),
            },
        }
    }
}

pub const EXT_FLAG_POOL: u8 = 0x01;
pub const EXT_FLAG_DICT: u8 = 0x02;
pub const EXT_FLAG_PEERS: u8 = 0x04;
/// Marker byte used by `container.rs::encode_nested_column_tagged` for recursive nested plans.
pub const NESTED_COLUMN_TAG: u8 = 0x80;

/// Complete self-describing compressed payload for a single column or standalone slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PayloadGraph {
    pub rows: u32,
    pub ext_pool: bool,
    pub ext_dict: Option<u16>,
    pub ext_peers: Vec<u32>,
    pub nodes: Vec<GraphNode>,
    pub root: u16,
}

impl PayloadGraph {
    /// Tries context mixing only on this graph's selected integer transports. Programs,
    /// dependencies and lazy cell views stay intact; losing pack candidates pay no mixer cost.
    pub fn with_context_mixing(mut self) -> Self {
        let budget = DecodeBudget::default();
        for node in &mut self.nodes {
            let GraphNode::IntStream(stream) = node else {
                continue;
            };
            if stream.len() < 512
                || matches!(
                    stream,
                    IntStream::Const { .. }
                        | IntStream::NibbleMix { .. }
                )
            {
                continue;
            }
            let mut wire = Vec::new();
            stream.encode(&mut wire);
            if wire.len() < 128 {
                continue;
            }
            if let Ok(values) = stream.unpack(&budget) {
                let base = std::mem::replace(stream, IntStream::Const { val: 0, len: 0 });
                *stream = IntStream::consider_context_mix(&values, base);
            }
        }
        self
    }

    pub fn empty() -> Self {
        Self {
            rows: 0,
            ext_pool: false,
            ext_dict: None,
            ext_peers: Vec::new(),
            nodes: Vec::new(),
            root: 0,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        self.encode_linked(None)
    }

    /// A single integer input followed by a short unary program. The enclosing
    /// frame already delimits the integer bytes and supplies their lane count.
    pub fn encode_numeric(&self) -> Vec<u8> {
        let (stream, prog) = match self.nodes.as_slice() {
            [GraphNode::IntStream(s)] if self.root == 0 => (s, None),
            [
                GraphNode::IntStream(s),
                GraphNode::ProgramI64 { prog, inputs, bufs },
            ] if self.root == 1
                && inputs == &[0]
                && bufs.is_empty()
                && prog.is_linear_lane()
                && (1..=7).contains(&prog.ops.len()) =>
            {
                (s, Some(prog))
            }
            _ => return self.encode(),
        };
        if self.rows == 0
            || stream.len() != self.rows as usize
            || self.ext_pool
            || self.ext_dict.is_some()
            || !self.ext_peers.is_empty()
        {
            return self.encode();
        }
        let mut out = Vec::new();
        put_uvarint(&mut out, self.rows as u64);
        let tag = out.len();
        out.push(0x80 | (prog.map_or(0, |p| p.ops.len() as u8) << 4));
        if let Some(prog) = prog {
            prog.encode_linear_ops(&mut out, true);
        }
        fn raw_bytes(leaf: &[u8]) -> Option<&[u8]> {
            let mut input = Cursor::new(leaf);
            if input.u8().ok()? != EntropyBackend::Raw as u8 {
                return None;
            }
            let len = input.uvar().ok()?;
            let bytes = input.bytes().ok()?;
            (len == bytes.len() as u64 && input.remaining() == 0).then_some(bytes)
        }
        let kind = match stream {
            IntStream::Rice { bytes, .. } => {
                out.extend_from_slice(bytes);
                8
            }
            IntStream::Radix {
                min, radix, bytes, ..
            } => {
                put_ivarint(&mut out, *min);
                put_uvarint(&mut out, u64::from(*radix));
                out.extend_from_slice(bytes);
                11
            }
            IntStream::Const { val, .. } => {
                let shift = val.trailing_zeros().min(63);
                if *val != 0 && shift >= 14 {
                    out.push(shift as u8);
                    put_ivarint(&mut out, *val >> shift);
                    7
                } else {
                    put_ivarint(&mut out, *val);
                    0
                }
            }
            IntStream::ZigZagVarint { leaf, .. } => {
                if let Some(bytes) = raw_bytes(leaf) {
                    out.extend_from_slice(bytes);
                    4
                } else {
                    out.extend_from_slice(leaf);
                    5
                }
            }
            IntStream::Rle { leaf, .. } => {
                out.extend_from_slice(leaf);
                1
            }
            IntStream::ForBitpack {
                min, bits, leaf, ..
            } => {
                put_ivarint(&mut out, *min);
                out.push(*bits);
                if let Some(bytes) = raw_bytes(leaf) {
                    out.extend_from_slice(bytes);
                    3
                } else {
                    out.extend_from_slice(leaf);
                    2
                }
            }
            _ => return self.encode(),
        };
        out[tag] |= kind;
        out
    }

    /// Fuses consecutive numeric programs and removes unreachable intermediates. This
    /// keeps a predictor and its output conversion in one compiled row loop.
    pub fn fuse_numeric_programs(mut self) -> Self {
        if self.rows == 0 {
            return self;
        }
        for i in 0..self.nodes.len() {
            let GraphNode::ProgramI64 { prog, inputs, bufs } = &self.nodes[i] else {
                continue;
            };
            if inputs.len() != 1 || !bufs.is_empty() || !prog.luts.is_empty() {
                continue;
            }
            let GraphNode::ProgramI64 {
                prog: prior,
                inputs: prior_inputs,
                bufs: prior_bufs,
            } = &self.nodes[inputs[0] as usize]
            else {
                continue;
            };
            if prior.width != prog.width
                || !prior_bufs.is_empty()
                || !prior.luts.is_empty()
                || prior.sinks.len() != 1
                || prior.sinks[0].1 != Len::Fixed(8)
            {
                continue;
            }
            let mut fused = (**prior).clone();
            let map = |r: u32| {
                if r == 0 {
                    prior.sinks[0].0
                } else {
                    prior.n_in + prior.ops.len() as u32 + r - 1
                }
            };
            for ins in &prog.ops {
                let mut ins = *ins;
                if ins.op.arity() >= 1 {
                    ins.a = map(ins.a);
                }
                if ins.op.arity() >= 2 {
                    ins.b = map(ins.b);
                }
                if ins.op.arity() >= 3 {
                    ins.c = map(ins.c);
                }
                fused.ops.push(ins);
            }
            fused.sinks = prog
                .sinks
                .iter()
                .map(|&(r, len)| {
                    (
                        map(r),
                        match len {
                            Len::Lane(r) => Len::Lane(map(r)),
                            other => other,
                        },
                    )
                })
                .collect();
            self.nodes[i] = GraphNode::ProgramI64 {
                prog: fused.into(),
                inputs: prior_inputs.clone(),
                bufs: vec![],
            };
        }
        let mut live = vec![false; self.nodes.len()];
        live[self.root as usize] = true;
        for i in (0..self.nodes.len()).rev() {
            if live[i] {
                self.nodes[i].for_each_dep(|d| live[d as usize] = true);
            }
        }
        let mut remap = vec![0; live.len()];
        let mut nodes = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if live[i] {
                remap[i] = nodes.len() as u16;
                nodes.push(node.remap(|d| remap[d as usize]));
            }
        }
        self.root = remap[self.root as usize];
        self.nodes = nodes;
        self
    }

    /// The size selection compares: the encoded size less what shared code will not cost.
    pub fn cost(&self) -> usize {
        self.encode().len() - code_discount(self)
    }

    /// The encoded graph and its [`cost`](Self::cost).
    pub fn priced(&self) -> (Vec<u8>, usize) {
        let wire = self.encode();
        let cost = wire.len() - code_discount(self);
        (wire, cost)
    }

    /// Encodes the graph, naming code in `lib` instead of carrying it.
    pub fn encode_linked(&self, lib: Option<&Library>) -> Vec<u8> {
        if self.rows == 0 {
            return vec![0u8];
        }
        let mut out = Vec::with_capacity(64);
        put_uvarint(&mut out, self.rows as u64);
        if lib.is_none() && !self.ext_pool && self.ext_dict.is_none() && self.ext_peers.is_empty() {
            match self.nodes.as_slice() {
                [GraphNode::IntStream(stream)] if self.root == 0 => {
                    out.push(8);
                    stream.encode(&mut out);
                    return out;
                }
                [
                    GraphNode::IntStream(stream),
                    GraphNode::ProgramI64 { prog, inputs, bufs },
                ] if self.root == 1 && inputs == &[0] && bufs.is_empty() => {
                    out.push(16);
                    stream.encode(&mut out);
                    out.extend(prog.encode());
                    return out;
                }
                _ => {}
            }
        }

        let flags = (u8::from(self.ext_pool))
            | (u8::from(self.ext_dict.is_some()) << 1)
            | (u8::from(!self.ext_peers.is_empty()) << 2);
        out.push(flags);
        if let Some(d) = self.ext_dict {
            put_uvarint(&mut out, d as u64);
        }
        if !self.ext_peers.is_empty() {
            put_uvarint(&mut out, self.ext_peers.len() as u64);
            for &p in &self.ext_peers {
                put_uvarint(&mut out, p as u64);
            }
        }
        put_uvarint(&mut out, self.nodes.len() as u64);
        for n in &self.nodes {
            encode_node_linked(n, lib, &mut out);
        }
        put_uvarint(&mut out, self.root as u64);
        out
    }

    pub fn decode(b: &[u8], budget: &DecodeBudget) -> Result<Self> {
        Self::decode_linked(b, &Library::default(), budget)
    }

    /// Recompresses every leaf at `level`, which a search may have priced at a cheaper one.
    pub(crate) fn relevel(&mut self, backend: EntropyBackend, level: i32) -> Result<()> {
        for node in &mut self.nodes {
            match node {
                GraphNode::Bytes { leaf } => *leaf = relevel_leaf(leaf, backend, level, true)?,
                GraphNode::IntStream(s) => s.relevel(backend, level)?,
                _ => {}
            }
        }
        Ok(())
    }

    /// Whether the graph holds code a library could carry.
    pub fn has_code(&self) -> bool {
        self.nodes.iter().any(|n| {
            matches!(
                n,
                GraphNode::ProgramI64 { .. }
                    | GraphNode::ProgramCells { .. }
                    | GraphNode::RecurBuf { .. }
            )
        })
    }

    /// Decodes a graph whose code references name entries of `lib`.
    pub fn decode_linked(b: &[u8], lib: &Library, budget: &DecodeBudget) -> Result<Self> {
        let mut cur = Cursor::new(b);
        let rows = cur.uvar()?;
        Self::decode_body(rows, cur, lib, budget)
    }

    fn decode_body(
        rows_u64: u64,
        mut cur: Cursor<'_>,
        lib: &Library,
        budget: &DecodeBudget,
    ) -> Result<Self> {
        if rows_u64 == 0 {
            if cur.remaining() != 0 {
                return Err(Error::Format(
                    "trailing bytes after empty PayloadGraph".into(),
                ));
            }
            return Ok(Self::empty());
        }
        if rows_u64 > budget.max_records_per_block.saturating_mul(16).max(65_536) as u64
            || rows_u64 > (budget.max_single_alloc_bytes / 4).max(1) as u64
        {
            return Err(Error::BudgetExceeded("PayloadGraph row count"));
        }
        let rows = u32::try_from(rows_u64).map_err(|_| Error::BudgetExceeded("graph rows"))?;
        let flags = cur.u8()?;
        if flags & 0x80 != 0 || matches!(flags, 8 | 16) {
            let (stream, prog, _) = read_numeric(&mut cur, rows, flags, budget)?;
            let mut nodes = vec![GraphNode::IntStream(stream)];
            if let Some(prog) = prog {
                nodes.push(GraphNode::ProgramI64 {
                    prog: prog.into(),
                    inputs: vec![0],
                    bufs: vec![],
                });
            }
            if cur.remaining() != 0 {
                return Err(Error::Format("trailing numeric graph bytes".into()));
            }
            return Ok(Self {
                rows,
                ext_pool: false,
                ext_dict: None,
                ext_peers: vec![],
                root: (nodes.len() - 1) as u16,
                nodes,
            });
        }
        if flags & !0x07 != 0 {
            return Err(Error::Format("unknown PayloadGraph ext_flags".into()));
        }
        let ext_pool = (flags & EXT_FLAG_POOL) != 0;
        let ext_dict = if (flags & EXT_FLAG_DICT) != 0 {
            let d = cur.uvar()?;
            if d > u16::MAX as u64 {
                return Err(Error::Format("ext_dict id overflow".into()));
            }
            Some(d as u16)
        } else {
            None
        };
        let mut ext_peers = Vec::new();
        if (flags & EXT_FLAG_PEERS) != 0 {
            let n_peers = cur.uvar()? as usize;
            if n_peers == 0 || n_peers > 64 {
                return Err(Error::Format("invalid ext_peers count".into()));
            }
            ext_peers.reserve(n_peers);
            for _ in 0..n_peers {
                let p = cur.uvar()?;
                if p > u32::MAX as u64 {
                    return Err(Error::Format("ext_peer id overflow".into()));
                }
                ext_peers.push(p as u32);
            }
        }
        let n_nodes = cur.uvar()? as usize;
        if n_nodes == 0 || n_nodes > 4096 {
            return Err(Error::Format("invalid PayloadGraph node count".into()));
        }
        let mut nodes = Vec::with_capacity(n_nodes);
        for idx in 0..n_nodes {
            let node = decode_node(&mut cur, idx, lib, budget)?;
            match &node {
                GraphNode::ExtPool if !ext_pool => {
                    return Err(Error::Format("undeclared ExtPool dependency".into()));
                }
                GraphNode::ExtDict { dict_id } if ext_dict != Some(*dict_id) => {
                    return Err(Error::Format("undeclared ExtDict dependency".into()));
                }
                GraphNode::ExtPeer { slot } if (*slot as usize) >= ext_peers.len() => {
                    return Err(Error::Format("undeclared ExtPeer dependency".into()));
                }
                GraphNode::ExtAlign { key, at }
                    if (*key as usize) >= ext_peers.len() || (*at as usize) > ext_peers.len() =>
                {
                    return Err(Error::Format("undeclared ExtAlign dependency".into()));
                }
                _ => {}
            }
            nodes.push(node);
        }
        let root = cur.uvar()?;
        if root as usize >= n_nodes || cur.remaining() != 0 {
            return Err(Error::Format(
                "invalid root or trailing bytes in PayloadGraph".into(),
            ));
        }
        Ok(Self {
            rows,
            ext_pool,
            ext_dict,
            ext_peers,
            nodes,
            root: root as u16,
        })
    }
}

/// [`PayloadGraph::decode_linked`], with the wire bytes of each node in node order. The rest of
/// `b` is the graph's header and root.
pub(crate) fn decode_linked_sizes(
    b: &[u8],
    lib: &Library,
    budget: &DecodeBudget,
) -> Result<(PayloadGraph, Vec<usize>)> {
    let graph = PayloadGraph::decode_linked(b, lib, budget)?;
    let mut sizes = Vec::with_capacity(graph.nodes.len());
    if graph.rows == 0 {
        return Ok((graph, sizes));
    }
    let mut cur = Cursor::new(b);
    cur.uvar()?;
    let flags = cur.u8()?;
    if flags & 0x80 != 0 || matches!(flags, 8 | 16) {
        let size = cur.remaining();
        let (_, prog, code) = read_numeric(&mut cur, graph.rows, flags, budget)?;
        sizes.push(size - code);
        if prog.is_some() {
            sizes.push(code);
        }
        return Ok((graph, sizes));
    }
    if flags & EXT_FLAG_DICT != 0 {
        cur.uvar()?;
    }
    if flags & EXT_FLAG_PEERS != 0 {
        for _ in 0..cur.uvar()? {
            cur.uvar()?;
        }
    }
    let n_nodes = cur.uvar()? as usize;
    for idx in 0..n_nodes {
        let at = cur.pos;
        decode_node(&mut cur, idx, lib, budget)?;
        sizes.push(cur.pos - at);
    }
    Ok((graph, sizes))
}

/// Builder for a single-column [`PayloadGraph`].
pub struct GraphBuilder {
    pub rows: u32,
    pub ext_pool: bool,
    pub ext_dict: Option<u16>,
    pub ext_peers: Vec<u32>,
    pub nodes: Vec<GraphNode>,
}

/// The last `window` distinct values, most recent first. A new value costs O(1) and a repeat
/// its rank: columns of ids are mostly new values, and shifting the whole list for each one
/// dominated reading them back.
struct Recency {
    buf: Vec<i64>,
    head: usize,
    len: usize,
    window: usize,
}

impl Recency {
    /// For a sequence of `n` values.
    fn new(window: usize, n: usize) -> Self {
        let size = 2 * window.min(n).max(1);
        Self {
            buf: vec![0; size],
            head: size,
            len: 0,
            window,
        }
    }

    #[cfg(test)]
    fn rank(&self, v: i64) -> Option<usize> {
        self.buf[self.head..self.head + self.len]
            .iter()
            .position(|&x| x == v)
    }

    fn push(&mut self, v: i64) {
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
    fn hit(&mut self, rank: usize) -> Option<i64> {
        if rank >= self.len {
            return None;
        }
        let run = &mut self.buf[self.head..=self.head + rank];
        run.rotate_right(1);
        Some(run[0])
    }
}

/// Distinct values an `MtfInts` node remembers.
pub(crate) const MTF_WINDOW: usize = 4096;

/// Move-to-front codes of `vals` over the last [`MTF_WINDOW`] distinct values: `0` takes the
/// next of the returned misses, `k` the value `k - 1` places back.
pub(crate) fn mtf_split(vals: &[i64]) -> (Vec<i64>, Vec<i64>) {
    // A value's place in the list is how many distinct values were used since its own last
    // use, counted over a bitmap of last-use times instead of searched for in the list.
    let mut uses = LastUses::new(vals.len());
    // Last uses by offset from the minimum when the values span little more than their count.
    let (lo, hi) = vals.iter().fold((i64::MAX, i64::MIN), |(l, h), &v| (l.min(v), h.max(v)));
    let span = (hi as i128 - lo as i128) as u128;
    let mut dense = if span < (vals.len() as u128) * 2 + 1024 {
        vec![u32::MAX; span as usize + 1]
    } else {
        Vec::new()
    };
    let mut sparse: FastMap<i64, u32> = FastMap::default();
    let mut codes = Vec::with_capacity(vals.len());
    let mut misses = Vec::new();
    for (t, &v) in vals.iter().enumerate() {
        let prev = if dense.is_empty() {
            sparse.insert(v, t as u32)
        } else {
            let slot = &mut dense[(v as i128 - lo as i128) as usize];
            Some(std::mem::replace(slot, t as u32)).filter(|&p| p != u32::MAX)
        };
        let rank = prev.map(|prev| {
            uses.clear(prev as usize);
            uses.between(prev as usize, t)
        });
        match rank {
            Some(r) if r < MTF_WINDOW => codes.push(r as i64 + 1),
            _ => {
                codes.push(0);
                misses.push(v);
            }
        }
        uses.set(t);
    }
    (codes, misses)
}

/// Times that are some value's last use, with a count per span of [`SPAN_BITS`] times.
struct LastUses {
    words: Vec<u64>,
    spans: Vec<u32>,
}

const SPAN_BITS: usize = 4096;

impl LastUses {
    fn new(n: usize) -> Self {
        Self {
            words: vec![0; n.div_ceil(64)],
            spans: vec![0; n.div_ceil(SPAN_BITS)],
        }
    }

    fn set(&mut self, t: usize) {
        self.words[t / 64] |= 1 << (t % 64);
        self.spans[t / SPAN_BITS] += 1;
    }

    fn clear(&mut self, t: usize) {
        self.words[t / 64] &= !(1 << (t % 64));
        self.spans[t / SPAN_BITS] -= 1;
    }

    /// Set times strictly between `a` and `b`.
    fn between(&self, a: usize, b: usize) -> usize {
        let (lo, hi) = (a + 1, b);
        if lo >= hi {
            return 0;
        }
        let (wl, wh) = (lo / 64, (hi - 1) / 64);
        let lo_mask = !0u64 << (lo % 64);
        let hi_mask = !0u64 >> (63 - (hi - 1) % 64);
        if wl == wh {
            return (self.words[wl] & lo_mask & hi_mask).count_ones() as usize;
        }
        let mut n =
            (self.words[wl] & lo_mask).count_ones() + (self.words[wh] & hi_mask).count_ones();
        const SPAN_WORDS: usize = SPAN_BITS / 64;
        let mut w = wl + 1;
        while w < wh {
            if w % SPAN_WORDS == 0 && w + SPAN_WORDS <= wh {
                n += self.spans[w / SPAN_WORDS];
                w += SPAN_WORDS;
            } else {
                n += self.words[w].count_ones();
                w += 1;
            }
        }
        n as usize
    }
}

/// [`mtf_split`] by searching the list itself, as a reader keeps it.
#[cfg(test)]
pub(crate) fn mtf_split_searched(vals: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut ring = Recency::new(MTF_WINDOW, vals.len());
    let mut codes = Vec::with_capacity(vals.len());
    let mut misses = Vec::new();
    for &v in vals {
        if let Some(pos) = ring.rank(v) {
            codes.push((pos + 1) as i64);
            ring.hit(pos);
        } else {
            codes.push(0);
            misses.push(v);
            ring.push(v);
        }
    }
    (codes, misses)
}

impl GraphBuilder {
    pub fn new(rows: usize) -> Self {
        Self {
            rows: rows as u32,
            ext_pool: false,
            ext_dict: None,
            ext_peers: Vec::new(),
            nodes: Vec::new(),
        }
    }

    pub fn push(&mut self, n: GraphNode) -> u16 {
        let id = self.nodes.len() as u16;
        self.nodes.push(n);
        id
    }

    pub fn add_ext_pool(&mut self) -> u16 {
        self.ext_pool = true;
        self.push(GraphNode::ExtPool)
    }

    pub fn add_ext_dict(&mut self, dict_id: u16) -> u16 {
        self.ext_dict = Some(dict_id);
        self.push(GraphNode::ExtDict { dict_id })
    }

    fn peer_slot(&mut self, peer_col_id: u32) -> u16 {
        if let Some(pos) = self.ext_peers.iter().position(|&p| p == peer_col_id) {
            pos as u16
        } else {
            let s = self.ext_peers.len() as u16;
            self.ext_peers.push(peer_col_id);
            s
        }
    }

    pub fn add_ext_peer(&mut self, peer_col_id: u32) -> u16 {
        let slot = self.peer_slot(peer_col_id);
        self.push(GraphNode::ExtPeer { slot })
    }

    /// Indices into column `key` of its first cell in the record of each cell of column `at`
    /// (`None`: this column).
    pub fn add_ext_align(&mut self, key: u32, at: Option<u32>) -> u16 {
        let key = self.peer_slot(key);
        let at = at.map_or(0, |c| self.peer_slot(c) + 1);
        self.push(GraphNode::ExtAlign { key, at })
    }

    pub fn add_match_index(&mut self, needles: u16, hay: u16, codes: u16) -> u16 {
        self.push(GraphNode::MatchIndex {
            needles,
            hay,
            codes,
        })
    }

    pub fn add_const_cells(&mut self, val: &[u8], count: usize) -> u16 {
        self.push(GraphNode::ConstCells {
            val: val.to_vec(),
            count: count as u32,
        })
    }

    pub fn add_quote_or_null(&mut self, inner: u16, map: Option<u16>) -> u16 {
        self.push(GraphNode::QuoteOrNull { inner, map })
    }

    pub fn add_parse_ascii_int(&mut self, cells: u16) -> u16 {
        self.push(GraphNode::ParseAsciiInt { cells })
    }

    pub fn add_bytes(&mut self, bytes: &[u8]) -> u16 {
        self.push(GraphNode::Bytes {
            leaf: pack_text_leaf(bytes),
        })
    }

    /// Unsigned little-endian lanes from `width` byte planes: all low bytes, then all second bytes.
    pub fn add_planes(&mut self, planes: &[u8], width: u8) -> u16 {
        assert!(matches!(width, 1 | 2 | 4 | 8) && planes.len().is_multiple_of(width as usize));
        self.push(GraphNode::IntStream(IntStream::ForPlanes {
            len: (planes.len() / width as usize) as u32,
            min: 0,
            width,
            leaf: pack_leaf(planes),
        }))
    }

    /// Cells of the low `width` bytes of each lane of `ints`, little-endian.
    pub fn add_fixed_cells(&mut self, ints: u16, width: u8) -> u16 {
        let b = Builder::new(1);
        let x = b.input(0);
        self.add_prog_cells(b.finish(&[(x, width)]), vec![ints], vec![])
    }

    pub fn add_raw_bytes(&mut self, bytes: &[u8]) -> u16 {
        let mut leaf = Vec::with_capacity(bytes.len() + 10);
        compress_leaf(&mut leaf, bytes, EntropyBackend::Raw, 0);
        self.push(GraphNode::Bytes { leaf })
    }

    pub fn merge_subgraph(&mut self, sub: GraphBuilder, root: u16) -> u16 {
        if sub.ext_pool {
            self.ext_pool = true;
        }
        if let Some(d) = sub.ext_dict {
            self.ext_dict = Some(d);
        }
        let mut peer_slot_map = Vec::with_capacity(sub.ext_peers.len());
        for &p in &sub.ext_peers {
            let slot = if let Some(pos) = self.ext_peers.iter().position(|&x| x == p) {
                pos as u16
            } else {
                let s = self.ext_peers.len() as u16;
                self.ext_peers.push(p);
                s
            };
            peer_slot_map.push(slot);
        }
        let base = self.nodes.len() as u16;
        for n in sub.nodes {
            let mapped = match n {
                GraphNode::ExtPeer { slot } => GraphNode::ExtPeer {
                    slot: peer_slot_map[slot as usize],
                },
                GraphNode::ExtAlign { key, at } => GraphNode::ExtAlign {
                    key: peer_slot_map[key as usize],
                    at: if at == 0 {
                        0
                    } else {
                        peer_slot_map[at as usize - 1] + 1
                    },
                },
                other => other.remap(|old| old + base),
            };
            self.nodes.push(mapped);
        }
        root + base
    }

    pub fn merge_graph(&mut self, g: &PayloadGraph) -> u16 {
        if g.rows == 0 && g.nodes.is_empty() {
            return self.add_const_cells(b"", 0);
        }
        let sub = GraphBuilder {
            rows: g.rows,
            ext_pool: g.ext_pool,
            ext_dict: g.ext_dict,
            ext_peers: g.ext_peers.clone(),
            nodes: g.nodes.clone(),
        };
        self.merge_subgraph(sub, g.root)
    }

    pub fn add_ints(&mut self, vals: &[i64]) -> u16 {
        match tape::recorded(Site::Ints) {
            Some(1) if !vals.is_empty() => {
                return self.push_deltas(vals[0], IntStream::auto_encode(&deltas(vals)));
            }
            Some(0) => return self.push(GraphNode::IntStream(IntStream::auto_encode(vals))),
            _ => {}
        }
        let direct = IntStream::auto_encode(vals);
        // Short streams and those whose head rules differences out say little of longer ones.
        let mut chose = tape::OPEN;
        if vals.len() >= 16 {
            let mut b_dir = Vec::new();
            direct.encode(&mut b_dir);
            if b_dir.len() > 36 {
                let try_delta = if vals.len() < 128 {
                    true
                } else {
                    let k = 128usize;
                    let head = &vals[..k];
                    let v_min = *head.iter().min().unwrap();
                    let v_max = *head.iter().max().unwrap();
                    let v_span = (v_max as u128).wrapping_sub(v_min as u128);
                    let mut d_min = i64::MAX;
                    let mut d_max = i64::MIN;
                    let mut v_runs = 1usize;
                    let mut d_runs = 1usize;
                    let mut prev_d = 0i64;
                    let mut small_d = 0usize;
                    let half_span = ((v_span.min(u64::MAX as u128) as u64) >> 1).max(16);
                    for i in 1..k {
                        if head[i] != head[i - 1] {
                            v_runs += 1;
                        }
                        let d = head[i].wrapping_sub(head[i - 1]);
                        d_min = d_min.min(d);
                        d_max = d_max.max(d);
                        if i > 1 && d != prev_d {
                            d_runs += 1;
                        }
                        prev_d = d;
                        if d.unsigned_abs() < half_span {
                            small_d += 1;
                        }
                    }
                    let d_span = (d_max as u128).wrapping_sub(d_min as u128);
                    d_span < v_span || d_runs < v_runs || small_d * 4 >= k
                };
                if try_delta {
                    let d_stream = IntStream::auto_encode(&deltas(vals));
                    let mut b_del = Vec::new();
                    d_stream.encode(&mut b_del);
                    if b_del.len() + 24 < b_dir.len() {
                        tape::note(Site::Ints, 1, &[]);
                        return self.push_deltas(vals[0], d_stream);
                    }
                    chose = 0;
                }
            }
        }
        tape::note(Site::Ints, chose, &[]);
        self.push(GraphNode::IntStream(direct))
    }

    /// Integers as a running sum of `d_stream` from `first`.
    fn push_deltas(&mut self, first: i64, d_stream: IntStream) -> u16 {
        let dn = self.push(GraphNode::IntStream(d_stream));
        let sb = Builder::new(1);
        let x = sb.input(0);
        let s = sb.scan_add(x, first);
        self.add_prog_i64(sb.finish(&[(s, 8)]), vec![dn], vec![])
    }

    /// Predicts signed lanes when their deltas need materially fewer bits. Prediction is
    /// an ordinary scan program, so readers need no knowledge of the encoder's policy.
    pub fn add_ints_fast(&mut self, vals: &[i64]) -> u16 {
        if vals.len() < 2 || vals.iter().all(|v| *v == vals[0]) {
            return self.push(GraphNode::IntStream(IntStream::encode_fast(vals)));
        }
        let runs = 1 + vals.windows(2).filter(|w| w[0] != w[1]).count();
        if runs * 16 < vals.len() {
            return self.push(GraphNode::IntStream(IntStream::encode_fast(vals)));
        }
        let step = vals[1].wrapping_sub(vals[0]);
        if vals.windows(2).all(|w| w[1].wrapping_sub(w[0]) == step) {
            let input = self.push(GraphNode::IntStream(IntStream::Const {
                val: step,
                len: vals.len() as u32,
            }));
            let b = Builder::new(1);
            let output = b.scan_add(b.input(0), vals[0].wrapping_sub(step));
            return self.add_prog_i64(b.finish(&[(output, 8)]), vec![input], vec![]);
        }
        let min = *vals.iter().min().unwrap();
        let mut factor = 0u64;
        for &v in vals {
            let mut delta = (v as u64).wrapping_sub(min as u64);
            while delta != 0 {
                (factor, delta) = (delta, factor % delta);
            }
            if factor == 1 {
                break;
            }
        }
        let saved_bits = 63 - factor.max(1).leading_zeros();
        if factor > 1 && vals.len() * saved_bits as usize > 256 {
            let normalized: Vec<_> = vals
                .iter()
                .map(|&v| ((v as u64).wrapping_sub(min as u64) / factor) as i64)
                .collect();
            let input = self.add_ints_fast(&normalized);
            let b = Builder::new(1);
            let value = if factor.is_power_of_two() {
                b.ri(
                    super::ir::Op::ShlI,
                    b.input(0),
                    factor.trailing_zeros() as i64,
                )
            } else {
                b.ri(super::ir::Op::MulI, b.input(0), factor as i64)
            };
            let output = b.ri(super::ir::Op::AddI, value, min);
            return self.add_prog_i64(b.finish(&[(output, 8)]), vec![input], vec![]);
        }
        let span = |lo: i64, hi: i64| 64 - (hi as u64).wrapping_sub(lo as u64).leading_zeros();
        let bits = span(min, *vals.iter().max().unwrap()) as usize;
        let mut deltas = Vec::with_capacity(vals.len());
        // Use a real step in the first lane so centering sees no artificial outlier.
        deltas.push(step);
        deltas.extend(vals.windows(2).map(|w| w[1].wrapping_sub(w[0])));
        let delta_bits =
            span(*deltas.iter().min().unwrap(), *deltas.iter().max().unwrap()) as usize;
        let varint_bytes = || {
            deltas
                .iter()
                .map(|&v| {
                    let zigzag = ((v << 1) ^ (v >> 63)) as u64;
                    (64 - zigzag.leading_zeros()).max(1).div_ceil(7) as usize
                })
                .sum::<usize>()
        };
        if bits.saturating_sub(delta_bits) * vals.len() > 40
            || varint_bytes() * 8 + 40 < bits * vals.len()
        {
            let input = self.push(GraphNode::IntStream(IntStream::encode_fast(&deltas)));
            let b = Builder::new(1);
            let output = b.scan_add(b.input(0), vals[0].wrapping_sub(deltas[0]));
            self.add_prog_i64(b.finish(&[(output, 8)]), vec![input], vec![])
        } else {
            self.push(GraphNode::IntStream(IntStream::encode_fast(vals)))
        }
    }

    pub fn add_byte_chunks(&mut self, buf: u16, chunk_bytes: u8) -> u16 {
        assert!((1..=8).contains(&chunk_bytes));
        self.push(GraphNode::ByteChunks { buf, chunk_bytes })
    }

    pub fn add_mtf_fixed(&mut self, flat_items: &[u8], width: u8) -> u16 {
        let w = width as usize;
        assert!(w >= 1 && flat_items.len().is_multiple_of(w));
        let count = flat_items.len() / w;
        // High-cardinality wide cells almost never hit the recent-item table. Sample
        // across the column before paying for a 64-entry search for every item.
        if w >= 8 && count >= 256 {
            let hits = (0..128)
                .filter(|&sample| {
                    let i = sample * (count - 1) / 127;
                    let item = &flat_items[i * w..(i + 1) * w];
                    flat_items[i.saturating_sub(64) * w..i * w]
                        .chunks_exact(w)
                        .any(|prior| prior == item)
                })
                .count();
            if hits < 8 {
                return self.add_bytes(flat_items);
            }
        }
        let mut codes = Vec::with_capacity(count);
        let mut fallback = Vec::new();
        if w <= 16 {
            // Items as integers: comparing a few bytes through `memcmp` dominated the search.
            let key = |c: &[u8]| c.iter().rev().fold(0u128, |k, &b| k << 8 | u128::from(b));
            let mut table: Vec<u128> = Vec::with_capacity(64);
            for chunk in flat_items.chunks_exact(w) {
                let k = key(chunk);
                if let Some(pos) = table.iter().position(|&x| x == k) {
                    codes.push((pos + 1) as i64);
                    table[..=pos].rotate_right(1);
                } else {
                    codes.push(0);
                    fallback.extend_from_slice(chunk);
                    if table.len() == 64 {
                        table.pop();
                    }
                    table.insert(0, k);
                }
            }
        } else {
            let mut table: Vec<&[u8]> = Vec::with_capacity(64);
            for chunk in flat_items.chunks_exact(w) {
                if let Some(pos) = table.iter().position(|&x| x == chunk) {
                    codes.push((pos + 1) as i64);
                    table[..=pos].rotate_right(1);
                } else {
                    codes.push(0);
                    fallback.extend_from_slice(chunk);
                    if table.len() == 64 {
                        table.pop();
                    }
                    table.insert(0, chunk);
                }
            }
        }
        if fallback.len() == flat_items.len()
            || (w >= 8 && fallback.len() * 20 >= flat_items.len() * 19)
        {
            return self.add_bytes(flat_items);
        }
        let codes_id = self.add_ints(&codes);
        let fb_id = self.add_bytes(&fallback);
        self.push(GraphNode::MtfFixed {
            codes: codes_id,
            fallback_buf: fb_id,
            width,
        })
    }

    pub fn add_slice_buf(&mut self, buf: u16, base: u32, lens: &[i64]) -> u16 {
        let lens_id = self.add_ints(lens);
        self.push(GraphNode::SliceBuf {
            buf,
            base,
            lens: lens_id,
        })
    }

    pub fn add_paged_slice(&mut self, buf: u16, lens: &[i64], pages: &[i64], bases: &[i64]) -> u16 {
        let lens = self.add_ints(lens);
        let pages = self.add_ints(pages);
        let bases = self.add_ints(bases);
        self.push(GraphNode::PagedSlice {
            buf,
            lens,
            pages,
            bases,
        })
    }

    pub fn add_index_cells(&mut self, source: u16, indices: u16) -> u16 {
        self.push(GraphNode::IndexCells { source, indices })
    }

    pub fn add_template_cells(&mut self, template: u16, slots: Vec<u16>) -> u16 {
        self.push(GraphNode::TemplateCells { template, slots })
    }

    pub fn add_mtf_ints(&mut self, vals: &[i64]) -> u16 {
        // The graph, its root and the node of its codes.
        let split = |codes: &[i64], misses: &[i64]| {
            let mut gb = Self::new(vals.len());
            let codes_id = gb.add_ints(codes);
            // How the misses of a few values code says little of the misses of many.
            let apart = tape::Scope::apart();
            let misses_id = gb.add_ints(misses);
            drop(apart);
            let root = gb.push(GraphNode::MtfInts {
                codes: codes_id,
                misses: misses_id,
            });
            (gb, root, codes_id)
        };
        match tape::recorded(Site::Mtf) {
            Some(1) => {
                let (codes, misses) = mtf_split(vals);
                let (gb, root, _) = split(&codes, &misses);
                return self.merge_subgraph(gb, root);
            }
            Some(0) => return self.add_ints(vals),
            _ => {}
        }
        // Whatever is not worth splitting is coded as it is. More values may be worth it.
        let plain = |gb: &mut Self| {
            tape::note(Site::Mtf, tape::OPEN, &[]);
            let _apart = tape::Scope::apart();
            gb.add_ints(vals)
        };
        if vals.len() < 64 {
            return plain(self);
        }
        let lookback = 512;
        let head = &vals[..vals.len().min(2048)];
        // A value recurs within the lookback when its previous occurrence is that recent.
        let mut prior: FastMap<i64, usize> = fast_map_with_capacity(head.len());
        let near_hits = head
            .iter()
            .enumerate()
            .filter(|&(i, &v)| prior.insert(v, i).is_some_and(|j| j + lookback >= i))
            .count();
        if near_hits * 10 < head.len() {
            if head.len() < 2048 {
                return plain(self);
            }
            // Values that do not come back in a head this long do not later.
            let scope = tape::Scope::record();
            let mut tmp = Self::new(vals.len());
            let root = tmp.add_ints(vals);
            tape::note(Site::Mtf, 0, &scope.end());
            return self.merge_subgraph(tmp, root);
        }

        let (codes, misses) = mtf_split(vals);
        if misses.len() * 8 >= vals.len() * 7 {
            return plain(self);
        }

        let wire_len = |nodes: &[GraphNode]| {
            nodes.iter().fold(0usize, |acc, n| {
                let mut b = Vec::new();
                encode_node(n, &mut b);
                acc + b.len()
            })
        };
        let scope = tape::Scope::record();
        let mut tmp_dir = Self::new(vals.len());
        let r_dir = tmp_dir.add_ints(vals);
        let t_dir = scope.end();
        let scope = tape::Scope::record();
        let (tmp_mtf, r_mtf, codes_id) = split(&codes, &misses);
        let t_mtf = scope.end();
        let dir_wire_len = wire_len(&tmp_dir.nodes);
        let mtf_wire_len = wire_len(&tmp_mtf.nodes);
        if mtf_wire_len + 32 < dir_wire_len {
            tape::note(Site::Mtf, 1, &t_mtf);
            self.merge_subgraph(tmp_mtf, r_mtf)
        } else {
            // Among more values fewer are misses, and the values span more: the split loses
            // for good only where its codes alone cost well over what the values do.
            let codes_wire_len = wire_len(&tmp_mtf.nodes[..=codes_id as usize]);
            match dir_wire_len * 10 <= codes_wire_len * 9 {
                true => tape::note(Site::Mtf, 0, &t_dir),
                false => tape::note(Site::Mtf, tape::OPEN, &[]),
            }
            self.merge_subgraph(tmp_dir, r_dir)
        }
    }

    pub fn add_row_index(&mut self, len: usize) -> u16 {
        let ones = self.push(GraphNode::IntStream(IntStream::Const {
            val: 1,
            len: len as u32,
        }));
        let b = Builder::new(1);
        let x = b.input(0);
        let idx = b.scan_add(x, -1);
        self.add_prog_i64(b.finish(&[(idx, 8)]), vec![ones], vec![])
    }

    pub fn add_dict_cells(&mut self, rows: &[&[u8]]) -> u16 {
        if rows.is_empty() {
            return self.add_const_cells(b"", 0);
        }
        let first = rows[0];
        if rows.iter().all(|&r| r == first) {
            return self.add_const_cells(first, rows.len());
        }

        let emit_raw_slice = |gb: &mut Self| -> u16 {
            let mut raw = Vec::with_capacity(rows.iter().map(|r| r.len()).sum());
            let mut lens = Vec::with_capacity(rows.len());
            for &r in rows {
                lens.push(r.len() as i64);
                raw.extend_from_slice(r);
            }
            let buf_id = gb.add_bytes(&raw);
            gb.add_slice_buf(buf_id, 0, &lens)
        };

        let max_dict_uniq = if rows.len() >= 32 {
            (rows.len() / 3).clamp(8, 8192)
        } else {
            rows.len()
        };
        let mut counts: FastMap<&[u8], (u32, usize)> = fast_map_with_capacity(rows.len().min(1024));
        let mut n_uniq = 0usize;
        // Each row's value by order of first appearance, so codes need no second lookup.
        let mut row_uniq = Vec::with_capacity(rows.len());
        for &r in rows {
            if let Some(e) = counts.get_mut(r) {
                e.0 += 1;
                row_uniq.push(e.1);
            } else {
                if n_uniq >= max_dict_uniq {
                    return emit_raw_slice(self);
                }
                counts.insert(r, (1, n_uniq));
                row_uniq.push(n_uniq);
                n_uniq += 1;
            }
        }

        if n_uniq == rows.len() && rows.len() > 1 {
            return emit_raw_slice(self);
        }

        let mut sorted: Vec<(&[u8], u32, usize)> = counts
            .into_iter()
            .map(|(s, (cnt, first_idx))| (s, cnt, first_idx))
            .collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));

        let mut remap = vec![0i64; sorted.len()];
        let mut dict_raw = Vec::new();
        let mut lens = Vec::with_capacity(sorted.len());
        for (new_id, (s, _, uniq)) in sorted.into_iter().enumerate() {
            remap[uniq] = new_id as i64;
            lens.push(s.len() as i64);
            dict_raw.extend_from_slice(s);
        }
        let codes: Vec<i64> = row_uniq.iter().map(|&u| remap[u]).collect();

        let codes_id = if lens.len() >= 16 && rows.len() >= 64 {
            self.add_mtf_ints(&codes)
        } else {
            self.add_ints(&codes)
        };
        let bytes_id = self.add_bytes(&dict_raw);
        let dict = self.add_slice_buf(bytes_id, 0, &lens);
        self.add_index_cells(dict, codes_id)
    }

    pub fn add_recur_node(
        &mut self,
        spec: RecurrenceSpec,
        seed: u16,
        cols: Vec<u16>,
        out_words: u32,
        include_seed: bool,
    ) -> u16 {
        self.push(GraphNode::RecurBuf {
            spec,
            seed,
            cols,
            out_words,
            include_seed,
        })
    }

    pub fn add_recur_u32(
        &mut self,
        spec: RecurrenceSpec,
        seed: &[u32],
        cols: Vec<u16>,
        out_words: u32,
        include_seed: bool,
    ) -> u16 {
        let mut seed_bytes = Vec::with_capacity(seed.len() * 4);
        for &w in seed {
            seed_bytes.extend_from_slice(&w.to_le_bytes());
        }
        let seed_node = self.add_bytes(&seed_bytes);
        self.add_recur_node(spec, seed_node, cols, out_words, include_seed)
    }

    pub fn add_prog_i64(&mut self, prog: Program, inputs: Vec<u16>, bufs: Vec<u16>) -> u16 {
        let prog = prog.into();
        self.push(GraphNode::ProgramI64 { prog, inputs, bufs })
    }

    pub fn add_prog_cells(&mut self, prog: Program, inputs: Vec<u16>, bufs: Vec<u16>) -> u16 {
        let prog = prog.into();
        self.push(GraphNode::ProgramCells { prog, inputs, bufs })
    }

    pub fn add_patch(&mut self, base: u16, indices: &[usize], values: &[&[u8]]) -> u16 {
        assert_eq!(indices.len(), values.len());
        if indices.is_empty() {
            return base;
        }
        let mut idx_deltas = Vec::with_capacity(indices.len());
        let mut prev = 0usize;
        for (k, &idx) in indices.iter().enumerate() {
            idx_deltas.push(if k == 0 {
                idx as i64
            } else {
                (idx - prev) as i64
            });
            prev = idx;
        }
        let idx_node = self.add_ints(&idx_deltas);
        let val_node = self.add_dict_cells(values);
        self.push(GraphNode::PatchCells {
            base,
            indices: idx_node,
            values: val_node,
        })
    }

    /// `count` cells of `literals[0]`, slot 0's cell, `literals[1]`, ..., `literals[n]`.
    pub fn add_template(&mut self, literals: &[&[u8]], slots: Vec<u16>, count: usize) -> u16 {
        assert!(literals.len() == slots.len() + 1 && slots.len() <= 64);
        if slots.is_empty() {
            return self.add_const_cells(literals[0], count);
        }
        if slots.len() == 1 && literals.iter().all(|l| l.is_empty()) {
            return slots[0];
        }
        let mut template = Vec::new();
        for (i, lit) in literals.iter().enumerate() {
            for &b in *lit {
                template.push(b);
                if b == 0x1f {
                    template.push(0xff);
                }
            }
            if i < slots.len() {
                template.extend([0x1f, (i as u8) << 1]);
            }
        }
        let template = self.add_const_cells(&template, count);
        self.add_template_cells(template, slots)
    }

    pub fn finish(self, root: u16) -> PayloadGraph {
        if self.rows == 0 {
            return PayloadGraph::empty();
        }
        PayloadGraph {
            rows: self.rows,
            ext_pool: self.ext_pool,
            ext_dict: self.ext_dict,
            ext_peers: self.ext_peers,
            nodes: self.nodes,
            root,
        }
    }
}

/// Writes code inline: an even length tag, then the bytes.
fn put_inline_code(out: &mut Vec<u8>, code: &[u8]) {
    put_uvarint(out, (code.len() as u64) << 1);
    out.extend_from_slice(code);
}

fn put_program(out: &mut Vec<u8>, prog: &Program, lib: Option<&Library>) {
    if !lib.is_some_and(|l| l.write_program(prog, out)) {
        put_inline_code(out, &prog.encode());
    }
}

pub(crate) fn encode_node(n: &GraphNode, out: &mut Vec<u8>) {
    encode_node_linked(n, None, out);
}

fn encode_node_linked(n: &GraphNode, lib: Option<&Library>, out: &mut Vec<u8>) {
    match n {
        GraphNode::Bytes { leaf } => {
            if let Some(lib) = lib.filter(|l| l.has_tables())
                && let Some(id) = table_bytes(n).and_then(|raw| lib.find_bytes(&raw))
            {
                out.push(21);
                put_uvarint(out, u64::from(id));
                return;
            }
            out.push(0);
            put_uvarint(out, leaf.len() as u64);
            out.extend_from_slice(leaf);
        }
        GraphNode::IntStream(s) => {
            out.push(1);
            s.encode(out);
        }
        GraphNode::ByteChunks { buf, chunk_bytes } => {
            out.push(2);
            put_uvarint(out, *buf as u64);
            out.push(*chunk_bytes);
        }
        GraphNode::MtfFixed {
            codes,
            fallback_buf,
            width,
        } => {
            out.push(3);
            put_uvarint(out, *codes as u64);
            put_uvarint(out, *fallback_buf as u64);
            out.push(*width);
        }
        GraphNode::RecurBuf {
            spec,
            seed,
            cols,
            out_words,
            include_seed,
        } => {
            out.push(5 | (u8::from(*include_seed) << 7));
            let sb = spec.encode();
            match lib.and_then(|l| l.find_recur(&sb)) {
                Some(id) => put_uvarint(out, (u64::from(id) << 1) | 1),
                None => put_inline_code(out, &sb),
            }
            put_uvarint(out, *seed as u64);
            put_u16_list(out, cols);
            put_uvarint(out, *out_words as u64);
        }
        GraphNode::ProgramI64 { prog, inputs, bufs } => {
            out.push(6);
            put_program(out, prog, lib);
            put_u16_list(out, inputs);
            put_u16_list(out, bufs);
        }
        GraphNode::ProgramCells { prog, inputs, bufs } => {
            out.push(7);
            put_program(out, prog, lib);
            put_u16_list(out, inputs);
            put_u16_list(out, bufs);
        }
        GraphNode::PatchCells {
            base,
            indices,
            values,
        } => {
            out.push(8);
            put_uvarint(out, *base as u64);
            put_uvarint(out, *indices as u64);
            put_uvarint(out, *values as u64);
        }
        GraphNode::MtfInts { codes, misses } => {
            out.push(11);
            put_uvarint(out, *codes as u64);
            put_uvarint(out, *misses as u64);
        }
        GraphNode::SliceBuf { buf, base, lens } => {
            out.push(12);
            put_uvarint(out, *buf as u64);
            put_uvarint(out, *base as u64);
            put_uvarint(out, *lens as u64);
        }
        GraphNode::PagedSlice {
            buf,
            lens,
            pages,
            bases,
        } => {
            out.push(24);
            for x in [buf, lens, pages, bases] {
                put_uvarint(out, *x as u64);
            }
        }
        GraphNode::IndexCells { source, indices } => {
            out.push(13);
            put_uvarint(out, *source as u64);
            put_uvarint(out, *indices as u64);
        }
        GraphNode::TemplateCells { template, slots } => {
            out.push(14);
            put_uvarint(out, *template as u64);
            put_u16_list(out, slots);
        }
        GraphNode::ConstCells { val, count } => {
            out.push(15);
            put_uvarint(out, *count as u64);
            put_uvarint(out, val.len() as u64);
            out.extend_from_slice(val);
        }
        GraphNode::QuoteOrNull { inner, map } => {
            out.push(16 | (u8::from(map.is_some()) << 7));
            put_uvarint(out, *inner as u64);
            if let Some(m) = map {
                put_uvarint(out, *m as u64);
            }
        }
        GraphNode::ParseAsciiInt { cells } => {
            out.push(17);
            put_uvarint(out, *cells as u64);
        }
        GraphNode::ExtPool => {
            out.push(18);
        }
        GraphNode::ExtDict { dict_id } => {
            out.push(19);
            put_uvarint(out, *dict_id as u64);
        }
        GraphNode::ExtPeer { slot } => {
            out.push(20);
            put_uvarint(out, *slot as u64);
        }
        GraphNode::ExtAlign { key, at } => {
            out.push(22);
            put_uvarint(out, *key as u64);
            put_uvarint(out, *at as u64);
        }
        GraphNode::MatchIndex {
            needles,
            hay,
            codes,
        } => {
            out.push(23);
            put_uvarint(out, *needles as u64);
            put_uvarint(out, *hay as u64);
            put_uvarint(out, *codes as u64);
        }
        GraphNode::KeyedSum { keys, deltas, order, seed } => {
            let extended = order.is_some() || *seed == SumSeed::Zero;
            out.push(25 | if extended { 0x80 } else { 0 });
            put_uvarint(out, *keys as u64);
            put_uvarint(out, *deltas as u64);
            if extended {
                out.push(u8::from(*seed == SumSeed::Zero) | (u8::from(order.is_some()) << 1));
                if let Some(order) = order { put_uvarint(out, *order as u64); }
            }
        }
        GraphNode::ExtVariant => out.push(26),
        GraphNode::CellIds { cells } => {
            out.push(27);
            put_uvarint(out, *cells as u64);
        }
        GraphNode::Interleave { sel, streams } => {
            out.push(28);
            put_uvarint(out, *sel as u64);
            put_u16_list(out, streams);
        }
    }
}

/// The inline bytes of a code reference, or the library entry it names.
fn code_ref<'a>(cur: &mut Cursor<'a>) -> Result<std::result::Result<&'a [u8], u64>> {
    let t = cur.uvar()?;
    if t & 1 == 1 {
        return Ok(Err(t >> 1));
    }
    let len = usize::try_from(t >> 1).map_err(|_| Error::Format("code length overflow".into()))?;
    Ok(Ok(cur.slice(len)?))
}

fn decode_node(
    cur: &mut Cursor<'_>,
    idx: usize,
    lib: &Library,
    budget: &DecodeBudget,
) -> Result<GraphNode> {
    let tag = cur.u8()?;
    let flag = (tag & 0x80) != 0;
    let kind = tag & 0x7f;
    let check_ref = |r: u64| -> Result<u16> {
        if (r as usize) < idx {
            Ok(r as u16)
        } else {
            Err(Error::Format(
                "GraphNode references future or invalid node".into(),
            ))
        }
    };
    match kind {
        0 => {
            if flag {
                return Err(Error::Format("invalid flag on Bytes node".into()));
            }
            let leaf = cur.bytes()?.to_vec();
            Ok(GraphNode::Bytes { leaf })
        }
        1 => Ok(GraphNode::IntStream(IntStream::decode_wire(
            cur, budget, 0,
        )?)),
        2 => {
            let buf = check_ref(cur.uvar()?)?;
            let cb = cur.u8()?;
            if !(1..=8).contains(&cb) {
                return Err(Error::Format("chunk_bytes must be 1..=8".into()));
            }
            Ok(GraphNode::ByteChunks {
                buf,
                chunk_bytes: cb,
            })
        }
        3 => {
            let codes = check_ref(cur.uvar()?)?;
            let fallback_buf = check_ref(cur.uvar()?)?;
            let width = cur.u8()?;
            if width == 0 || width > 64 {
                return Err(Error::Format("MTF width out of range".into()));
            }
            Ok(GraphNode::MtfFixed {
                codes,
                fallback_buf,
                width,
            })
        }
        5 => {
            let spec = match code_ref(cur)? {
                Ok(sbytes) => RecurrenceSpec::decode(sbytes)?,
                Err(id) => lib.read_recur(id)?,
            };
            let seed = check_ref(cur.uvar()?)?;
            let cols = get_u16_list(cur, idx)?;
            if cols.len() != spec.n_cols() {
                return Err(Error::Format("RecurBuf column count mismatch".into()));
            }
            let out_words = cur.uvar()?;
            let elem_bytes = match spec.width() {
                Width::W32 => 4usize,
                Width::W64 => 8usize,
            };
            if out_words > (budget.max_single_alloc_bytes / elem_bytes) as u64 {
                return Err(Error::BudgetExceeded("RecurBuf out_words"));
            }
            Ok(GraphNode::RecurBuf {
                spec,
                seed,
                cols,
                out_words: out_words as u32,
                include_seed: flag,
            })
        }
        6 | 7 => {
            let prog = match code_ref(cur)? {
                Ok(pbytes) => Program::decode(pbytes)?.into(),
                Err(id) => lib.read_program(id, cur)?,
            };
            let inputs = get_u16_list(cur, idx)?;
            let bufs = get_u16_list(cur, idx)?;
            if inputs.len() != prog.n_in as usize || bufs.len() != prog.n_bufs as usize {
                return Err(Error::Format(
                    "Program node input/buf arity mismatch".into(),
                ));
            }
            if kind == 6 {
                if prog.width != Width::W64
                    || prog.sinks.len() != 1
                    || prog.sinks[0].1 != Len::Fixed(8)
                {
                    return Err(Error::Format("ProgramI64 requires 1x8 W64 sink".into()));
                }
                Ok(GraphNode::ProgramI64 { prog, inputs, bufs })
            } else {
                Ok(GraphNode::ProgramCells { prog, inputs, bufs })
            }
        }
        8 => {
            let base = check_ref(cur.uvar()?)?;
            let indices = check_ref(cur.uvar()?)?;
            let values = check_ref(cur.uvar()?)?;
            Ok(GraphNode::PatchCells {
                base,
                indices,
                values,
            })
        }
        11 => {
            let codes = check_ref(cur.uvar()?)?;
            let misses = check_ref(cur.uvar()?)?;
            Ok(GraphNode::MtfInts { codes, misses })
        }
        12 => {
            let buf = check_ref(cur.uvar()?)?;
            let base = cur.uvar()?;
            if base > u32::MAX as u64 {
                return Err(Error::Format("SliceBuf base overflow".into()));
            }
            let lens = check_ref(cur.uvar()?)?;
            Ok(GraphNode::SliceBuf {
                buf,
                base: base as u32,
                lens,
            })
        }
        13 => {
            let source = check_ref(cur.uvar()?)?;
            let indices = check_ref(cur.uvar()?)?;
            Ok(GraphNode::IndexCells { source, indices })
        }
        24 => Ok(GraphNode::PagedSlice {
            buf: check_ref(cur.uvar()?)?,
            lens: check_ref(cur.uvar()?)?,
            pages: check_ref(cur.uvar()?)?,
            bases: check_ref(cur.uvar()?)?,
        }),
        14 => {
            let template = check_ref(cur.uvar()?)?;
            let slots = get_u16_list(cur, idx)?;
            if slots.is_empty() || slots.len() > 64 {
                return Err(Error::Format("invalid TemplateCells slot count".into()));
            }
            Ok(GraphNode::TemplateCells { template, slots })
        }
        15 => {
            let count = cur.uvar()?;
            if count > budget.max_records_per_block.saturating_mul(16).max(65_536) as u64 {
                return Err(Error::BudgetExceeded("ConstCells count"));
            }
            let val = cur.bytes()?;
            budget.check_alloc(val.len(), "ConstCells")?;
            Ok(GraphNode::ConstCells {
                val: val.to_vec(),
                count: count as u32,
            })
        }
        16 => {
            let inner = check_ref(cur.uvar()?)?;
            let map = if flag {
                Some(check_ref(cur.uvar()?)?)
            } else {
                None
            };
            Ok(GraphNode::QuoteOrNull { inner, map })
        }
        17 => {
            let cells = check_ref(cur.uvar()?)?;
            Ok(GraphNode::ParseAsciiInt { cells })
        }
        18 => Ok(GraphNode::ExtPool),
        19 => {
            let d = cur.uvar()?;
            if d > u16::MAX as u64 {
                return Err(Error::Format("ExtDict id overflow".into()));
            }
            Ok(GraphNode::ExtDict { dict_id: d as u16 })
        }
        20 => {
            let s = cur.uvar()?;
            if s >= 64 {
                return Err(Error::Format("ExtPeer slot out of bounds".into()));
            }
            Ok(GraphNode::ExtPeer { slot: s as u16 })
        }
        22 => {
            let (key, at) = (cur.uvar()?, cur.uvar()?);
            if key >= 64 || at > 64 {
                return Err(Error::Format("ExtAlign slot out of bounds".into()));
            }
            Ok(GraphNode::ExtAlign {
                key: key as u16,
                at: at as u16,
            })
        }
        23 => Ok(GraphNode::MatchIndex {
            needles: check_ref(cur.uvar()?)?,
            hay: check_ref(cur.uvar()?)?,
            codes: check_ref(cur.uvar()?)?,
        }),
        25 => {
            let (keys, deltas) = (check_ref(cur.uvar()?)?, check_ref(cur.uvar()?)?);
            let options = if flag { cur.u8()? } else { 0 };
            if options & !3 != 0 || (flag && options == 0) {
                return Err(Error::Format("invalid KeyedSum options".into()));
            }
            let seed = if options & 1 != 0 { SumSeed::Zero } else { SumSeed::Previous };
            let order = if options & 2 != 0 { Some(check_ref(cur.uvar()?)?) } else { None };
            Ok(GraphNode::KeyedSum { keys, deltas, order, seed })
        }
        26 => Ok(GraphNode::ExtVariant),
        27 => Ok(GraphNode::CellIds {
            cells: check_ref(cur.uvar()?)?,
        }),
        28 => {
            let sel = check_ref(cur.uvar()?)?;
            let streams = get_u16_list(cur, idx)?;
            if streams.is_empty() {
                return Err(Error::Format("Interleave without streams".into()));
            }
            Ok(GraphNode::Interleave { sel, streams })
        }
        21 => {
            if flag {
                return Err(Error::Format("invalid flag on LibraryBytes node".into()));
            }
            let raw = lib.read_bytes(cur.uvar()?)?;
            let mut leaf = Vec::with_capacity(raw.len() + 4);
            compress_leaf(&mut leaf, raw, EntropyBackend::Raw, 0);
            Ok(GraphNode::Bytes { leaf })
        }
        _ => Err(Error::Format(format!("unknown GraphNode kind {kind}"))),
    }
}

/// External block-level inputs supplied when decoding a column [`PayloadGraph`].
#[derive(Default)]
pub struct ExtInputs<'a> {
    pub pool: Option<&'a Arc<Pool>>,
    pub dict: Option<(u16, &'a Arc<Cells>)>,
    pub peers: &'a [Arc<Cells>],
    /// The value of each `ExtAlign` and `ExtVariant` node, in node order, which the block
    /// layout determines.
    pub aligned: &'a [Vec<i64>],
}

impl<'a> ExtInputs<'a> {
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Zero-knowledge core decoder: executes any [`PayloadGraph`] without codec-pack code.
pub struct CoreDecoder;

enum Val {
    Empty,
    Buf(Vec<u8>),
    SharedBuf(Arc<Pool>),
    Ints(Vec<i64>),
    Cells(Cells),
    SharedCells(Arc<Cells>),
}

impl CoreDecoder {
    /// Decodes `wire` with no external block dependencies into materialized [`Cells`].
    pub fn decode_bytes(wire: &[u8], budget: &DecodeBudget) -> Result<Cells> {
        let graph = PayloadGraph::decode(wire, budget)?;
        Self::decode_cells(&graph, &ExtInputs::empty(), budget)
    }

    /// Numeric decoding can read a standalone stream without allocating a graph directory.
    pub fn decode_ints_bytes(bytes: &[u8], budget: &DecodeBudget) -> Result<Vec<i64>> {
        let mut cursor = Cursor::new(bytes);
        let rows = cursor.uvar()?;
        Self::decode_ints_body(cursor.slice(cursor.remaining())?, rows, budget)
    }

    /// Decodes a numeric payload whose row count was supplied by its enclosing frame.
    pub fn decode_ints_body(bytes: &[u8], rows: u64, budget: &DecodeBudget) -> Result<Vec<i64>> {
        let mut compact = Cursor::new(bytes);
        if rows > 0 && rows <= budget.max_records_per_block as u64 {
            let flags = compact.u8()?;
            if flags & 0x80 != 0 || matches!(flags, 8 | 16) {
                let rows =
                    u32::try_from(rows).map_err(|_| Error::BudgetExceeded("numeric rows"))?;
                let (stream, prog, _) = read_numeric(&mut compact, rows, flags, budget)?;
                let values = stream.unpack(budget)?;
                if let Some(prog) = prog {
                    budget.check_alloc(values.len().saturating_mul(8), "numeric lane output")?;
                    let mut out = vec![0; values.len()];
                    if !super::chain::eval_numeric_chain(&prog, &values, &mut out) {
                        CompiledProgram::with(&prog.into(), |cp| {
                            cp.eval_i64(&[&values], &[], &mut out)
                        });
                    }
                    return Ok(out);
                }
                return Ok(values);
            }
        }
        let mut cursor = Cursor::new(bytes);
        if rows > 0
            && rows <= budget.max_records_per_block as u64
            && cursor.u8()? == 0
            && cursor.uvar()? == 1
            && cursor.u8()? == 1
        {
            let stream = IntStream::decode_wire(&mut cursor, budget, 0)?;
            if stream.len() as u64 != rows || cursor.uvar()? != 0 || cursor.remaining() != 0 {
                return Err(Error::Format("numeric graph shape mismatch".into()));
            }
            return stream.unpack(budget);
        }
        let graph =
            PayloadGraph::decode_body(rows, Cursor::new(bytes), &Library::default(), budget)?;
        Self::decode_ints(&graph, &ExtInputs::empty(), budget)
    }

    /// Executes a numeric graph without allocating per-cell offsets or formatting bytes.
    pub fn decode_ints(
        graph: &PayloadGraph,
        ext: &ExtInputs<'_>,
        budget: &DecodeBudget,
    ) -> Result<Vec<i64>> {
        let rows = graph.rows as usize;
        if rows == 0 {
            return Ok(Vec::new());
        }
        if let GraphNode::IntStream(stream) = &graph.nodes[graph.root as usize] {
            if stream.len() != rows {
                return Err(Error::Format("numeric row count mismatch".into()));
            }
            return stream.unpack(budget);
        }
        let mut uses = vec![0u32; graph.nodes.len()];
        for node in &graph.nodes {
            node.for_each_dep(|dep| uses[dep as usize] += 1);
        }
        let mut values = Self::execute_nodes(
            &graph.nodes,
            rows,
            ext,
            budget,
            &vec![false; graph.nodes.len()],
            &mut uses,
            &[],
        )?;
        match std::mem::replace(&mut values[graph.root as usize], Val::Empty) {
            Val::Ints(values) if values.len() == rows => Ok(values),
            _ => Err(Error::Format(
                "numeric graph root type or row count mismatch".into(),
            )),
        }
    }

    /// Executes `graph` into materialized [`Cells`].
    pub fn decode_cells(
        graph: &PayloadGraph,
        ext: &ExtInputs<'_>,
        budget: &DecodeBudget,
    ) -> Result<Cells> {
        let col = Self::decode_col(graph, ext, budget)?;
        match col {
            Col::Flat(c) => Ok(c),
            other => other.to_cells(),
        }
    }

    /// Executes `graph` into a [`Col`], preserving lazy `Const`, `Dict`, `Pool`, and `Quoted`
    /// representations whenever the root node allows it.
    pub fn decode_col(
        graph: &PayloadGraph,
        ext: &ExtInputs<'_>,
        budget: &DecodeBudget,
    ) -> Result<Col> {
        Self::decode_col_reading(graph, ext, budget, None)
    }

    /// [`Self::decode_col`] for a reader of only the cells in `reads`, ascending `(start, end)`
    /// ranges. A column built row by row stops after the last; one built by a program without
    /// scans builds just those, and its other cells are `NotLoaded`. Any other column is whole.
    pub fn decode_col_reading(
        graph: &PayloadGraph,
        ext: &ExtInputs<'_>,
        budget: &DecodeBudget,
        reads: Option<&[(usize, usize)]>,
    ) -> Result<Col> {
        let rows = graph.rows as usize;
        if rows == 0 {
            return Ok(Col::Flat(Cells::default()));
        }
        let n_nodes = graph.nodes.len();
        let root = graph.root as usize;

        let mut rem_uses = vec![0u32; n_nodes];
        for n in &graph.nodes {
            n.for_each_dep(|dep| rem_uses[dep as usize] += 1);
        }

        // Determine if `root` (and optionally `QuoteOrNull.inner`) can construct a lazy `Col`
        // directly from its input nodes without eagerly materializing duplicate cell bytes.
        let mut skip = vec![false; n_nodes];
        let is_lazy_leaf = |n: &GraphNode| {
            matches!(
                n,
                GraphNode::ConstCells { .. }
                    | GraphNode::IndexCells { .. }
                    | GraphNode::SliceBuf { .. }
                    | GraphNode::PagedSlice { .. }
            )
        };
        match &graph.nodes[root] {
            n if is_lazy_leaf(n) => {
                skip[root] = true;
                // `rem_uses` already counted `root`'s references to its inputs, and skipping `root`
                // prevents decrementing them during `execute_nodes`, keeping them alive!
            }
            GraphNode::QuoteOrNull { inner, map } => {
                skip[root] = true;
                let inner_idx = *inner as usize;
                // Only `root` references `inner` (`rem_uses[inner_idx] == 1`)? Then `inner` can
                // also skip eager materialization if it is a lazy leaf!
                let map_uses_inner = map == &Some(*inner);
                if !map_uses_inner
                    && rem_uses[inner_idx] == 1
                    && is_lazy_leaf(&graph.nodes[inner_idx])
                {
                    skip[inner_idx] = true;
                }
            }
            _ => {
                rem_uses[root] += 1;
            }
        }

        let upto = reads.and_then(|r| r.last()).map_or(usize::MAX, |r| r.1);
        // The program that builds the cells, when it builds each from its own row alone.
        let sparse = reads.and_then(|_| {
            let t = match &graph.nodes[root] {
                GraphNode::QuoteOrNull { inner, map: None } if rem_uses[*inner as usize] == 1 => {
                    *inner as usize
                }
                _ => root,
            };
            let GraphNode::ProgramCells { prog, .. } = &graph.nodes[t] else {
                return None;
            };
            let scans = prog
                .ops
                .iter()
                .any(|i| matches!(i.op, Op::ScanAdd | Op::ScanXor | Op::ScanF64Add));
            (!scans).then_some(t)
        });
        if let Some(t) = sparse {
            skip[t] = true;
        }
        let lens = prefix_lens(&graph.nodes, root, upto);
        let mut vals =
            Self::execute_nodes(&graph.nodes, rows, ext, budget, &skip, &mut rem_uses, &lens)?;
        let col = match (sparse, reads) {
            (Some(t), Some(reads)) => {
                let cells = sparse_cells(&graph.nodes[t], &vals, rows.min(upto), reads, budget)?;
                match t == root {
                    true => cells,
                    false => Col::Quoted {
                        inner: Box::new(cells),
                        map: Vec::new(),
                    },
                }
            }
            _ => Self::build_col(root, &graph.nodes, &skip, &mut vals, budget)?,
        };
        if col.len() != rows && col.len() != rows.min(upto) {
            return Err(Error::Format("PayloadGraph root row count mismatch".into()));
        }
        Ok(col)
    }

    fn build_col(
        nid: usize,
        nodes: &[GraphNode],
        skip: &[bool],
        vals: &mut [Val],
        budget: &DecodeBudget,
    ) -> Result<Col> {
        if !skip[nid] {
            return match std::mem::replace(&mut vals[nid], Val::Empty) {
                Val::Cells(c) => Ok(Col::Flat(c)),
                Val::SharedCells(sc) => Ok(Col::Flat((*sc).clone())),
                _ => Err(Error::Format(
                    "PayloadGraph node is not a Cells value".into(),
                )),
            };
        }
        match &nodes[nid] {
            GraphNode::ConstCells { val, count } => Ok(Col::Const {
                val: val.clone(),
                n: *count as usize,
            }),
            GraphNode::IndexCells { source, indices } => {
                if *source == *indices {
                    return Err(Error::Format("IndexCells source == indices".into()));
                }
                let dict = take_shared_cells(&mut vals[*source as usize])?;
                let idx_col = as_ints(&vals[*indices as usize])?;
                let dict_len = dict.len();
                let mut ords = Vec::with_capacity(idx_col.len());
                for &i in idx_col {
                    if i < 0 || (i as usize) >= dict_len {
                        return Err(Error::Format("IndexCells index out of bounds".into()));
                    }
                    ords.push(i as u32);
                }
                Ok(Col::Dict { dict, ords })
            }
            GraphNode::SliceBuf { buf, base, lens } => {
                let lens_col = as_ints(&vals[*lens as usize])?;
                let raw_len = buf_len(&vals[*buf as usize])?;
                let start = *base as usize;
                if start > raw_len {
                    return Err(Error::Format("SliceBuf base out of bounds".into()));
                }
                let mut ends = Vec::with_capacity(lens_col.len());
                let mut acc = 0usize;
                for &l in lens_col {
                    if l < 0 {
                        return Err(Error::Format("negative SliceBuf cell length".into()));
                    }
                    acc = acc
                        .checked_add(l as usize)
                        .ok_or_else(|| Error::Format("SliceBuf overflow".into()))?;
                    if acc > u32::MAX as usize {
                        return Err(Error::BudgetExceeded("SliceBuf column bytes"));
                    }
                    ends.push(acc as u32);
                }
                let end = start
                    .checked_add(acc)
                    .ok_or_else(|| Error::Format("SliceBuf overflow".into()))?;
                if end > raw_len {
                    return Err(Error::Format("SliceBuf slice exceeds buffer".into()));
                }
                if let Val::SharedBuf(pool) = &vals[*buf as usize] {
                    // Only a paged slice reads a pool held in part.
                    if !pool.is_whole() {
                        return Err(Error::NotLoaded);
                    }
                    return Ok(Col::Pool {
                        pool: Arc::clone(pool),
                        base: start,
                        ends,
                    });
                }
                let data = match std::mem::replace(&mut vals[*buf as usize], Val::Empty) {
                    Val::Buf(b) if start == 0 && end == b.len() => b,
                    Val::Buf(b) => b[start..end].to_vec(),
                    other => as_table_bytes(&other)?[start..end].to_vec(),
                };
                Ok(Col::Flat(Cells::from_parts(data, ends)?))
            }
            GraphNode::PagedSlice {
                buf,
                lens,
                pages,
                bases,
            } => paged_col(vals, [*buf, *lens, *pages, *bases], budget),
            GraphNode::QuoteOrNull { inner, map } => {
                let map_u32 =
                    if let Some(m_id) = map {
                        let m_ints = as_ints(&vals[*m_id as usize])?;
                        let mut out = Vec::with_capacity(m_ints.len());
                        for &v in m_ints {
                            if v < 0 {
                                out.push(NULL_CELL);
                            } else {
                                out.push(u32::try_from(v).map_err(|_| {
                                    Error::Format("QuoteOrNull map overflow".into())
                                })?);
                            }
                        }
                        out
                    } else {
                        Vec::new()
                    };
                let inner_col = Self::build_col(*inner as usize, nodes, skip, vals, budget)?;
                let inner_len = inner_col.len();
                for &u in &map_u32 {
                    if u != NULL_CELL && (u as usize) >= inner_len {
                        return Err(Error::Format("QuoteOrNull map index out of bounds".into()));
                    }
                }
                Ok(Col::Quoted {
                    inner: Box::new(inner_col),
                    map: map_u32,
                })
            }
            _ => unreachable!(),
        }
    }

    fn execute_nodes(
        nodes: &[GraphNode],
        rows: usize,
        ext: &ExtInputs<'_>,
        budget: &DecodeBudget,
        skip: &[bool],
        rem_uses: &mut [u32],
        upto: &[usize],
    ) -> Result<Vec<Val>> {
        let mut vals: Vec<Val> = Vec::with_capacity(nodes.len());
        // Constant cells used only as `TemplateCells` templates are read from the node itself.
        let mut template_uses = vec![0u32; nodes.len()];
        for node in nodes {
            if let GraphNode::TemplateCells { template, .. } = node {
                template_uses[*template as usize] += 1;
            }
        }

        let mut n_aligned = 0;
        for (idx, node) in nodes.iter().enumerate() {
            let template_only = matches!(node, GraphNode::ConstCells { .. })
                && template_uses[idx] > 0
                && template_uses[idx] == rem_uses[idx];
            if skip[idx] || template_only {
                vals.push(Val::Empty);
                continue;
            }
            // A node cut short reads the same leading rows of its row inputs.
            let upto = upto.get(idx).copied().unwrap_or(usize::MAX);
            let val = match node {
                GraphNode::Bytes { leaf } => Val::Buf(unpack_leaf(leaf, budget)?),
                GraphNode::IntStream(s) => Val::Ints(s.unpack(budget)?),
                GraphNode::ByteChunks { buf, chunk_bytes } => {
                    let bytes = as_table_bytes(&vals[*buf as usize])?;
                    let cb = *chunk_bytes as usize;
                    if !bytes.len().is_multiple_of(cb) {
                        return Err(Error::Format(
                            "ByteChunks buffer not aligned to chunk_bytes".into(),
                        ));
                    }
                    let n = bytes.len() / cb;
                    budget.check_alloc(n.saturating_mul(8), "ByteChunks")?;
                    let mut col = Vec::with_capacity(n);
                    if cb == 8 {
                        for ch in bytes.chunks_exact(8) {
                            col.push(i64::from_le_bytes(ch.try_into().unwrap()));
                        }
                    } else if cb == 4 {
                        for ch in bytes.chunks_exact(4) {
                            col.push(u32::from_le_bytes(ch.try_into().unwrap()) as i64);
                        }
                    } else {
                        for ch in bytes.chunks_exact(cb) {
                            let mut b8 = [0u8; 8];
                            b8[..cb].copy_from_slice(ch);
                            col.push(i64::from_le_bytes(b8));
                        }
                    }
                    Val::Ints(col)
                }
                GraphNode::MtfFixed {
                    codes,
                    fallback_buf,
                    width,
                } => {
                    let codes_col = as_ints(&vals[*codes as usize])?;
                    let fb = as_buf(&vals[*fallback_buf as usize])?;
                    let w = *width as usize;
                    let total = codes_col
                        .len()
                        .checked_mul(w)
                        .ok_or(Error::BudgetExceeded("MtfFixed"))?;
                    budget.check_alloc(total, "MtfFixed")?;
                    let mut table: Vec<[u8; 64]> = Vec::with_capacity(64);
                    let mut fb_pos = 0usize;
                    let mut out = Vec::with_capacity(total);
                    for &c in codes_col {
                        if c == 0 {
                            if fb_pos + w > fb.len() {
                                return Err(Error::Format("MTF fallback buffer underrun".into()));
                            }
                            let mut slot = [0u8; 64];
                            slot[..w].copy_from_slice(&fb[fb_pos..fb_pos + w]);
                            fb_pos += w;
                            out.extend_from_slice(&slot[..w]);
                            if table.len() == 64 {
                                table.pop();
                            }
                            table.insert(0, slot);
                        } else if c >= 1 && (c as usize) <= table.len() {
                            let k = (c as usize) - 1;
                            let slot = table[k];
                            out.extend_from_slice(&slot[..w]);
                            if k > 0 {
                                table.copy_within(0..k, 1);
                                table[0] = slot;
                            }
                        } else {
                            return Err(Error::Format("MTF code out of bounds".into()));
                        }
                    }
                    if fb_pos != fb.len() {
                        return Err(Error::Format("MTF trailing fallback bytes".into()));
                    }
                    Val::Buf(out)
                }
                GraphNode::RecurBuf {
                    spec,
                    seed,
                    cols,
                    out_words,
                    include_seed,
                } => {
                    let raw_seed = as_table_bytes(&vals[*seed as usize])?;
                    let compiled = CompiledRecurrence::compile(spec)?;
                    match spec.width() {
                        Width::W32 => {
                            if !raw_seed.len().is_multiple_of(4) {
                                return Err(Error::Format("unaligned W32 seed".into()));
                            }
                            let seed: Vec<u32> = raw_seed
                                .chunks_exact(4)
                                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                            if seed.len() != spec.max_lag() * spec.n_state as usize {
                                return Err(Error::Format("recurrence seed size mismatch".into()));
                            }
                            let mut col_u32: Vec<Vec<u32>> = Vec::with_capacity(cols.len());
                            for &cid in cols {
                                let ci = as_ints(&vals[cid as usize])?;
                                if ci.len() < *out_words as usize {
                                    return Err(Error::Format(
                                        "RecurBuf input column too short".into(),
                                    ));
                                }
                                col_u32.push(ci.iter().map(|&x| x as u32).collect());
                            }
                            let col_refs: Vec<&[u32]> =
                                col_u32.iter().map(|v| v.as_slice()).collect();
                            let total =
                                *out_words as usize + if *include_seed { seed.len() } else { 0 };
                            budget.check_alloc(total.saturating_mul(4), "RecurBuf W32")?;
                            let mut words = vec![0u32; total];
                            if *include_seed {
                                words[..seed.len()].copy_from_slice(&seed);
                                compiled.eval_u32(&seed, &col_refs, &mut words[seed.len()..]);
                            } else {
                                compiled.eval_u32(&seed, &col_refs, &mut words);
                            }
                            let mut bytes = Vec::with_capacity(words.len() * 4);
                            for w in words {
                                bytes.extend_from_slice(&w.to_le_bytes());
                            }
                            Val::Buf(bytes)
                        }
                        Width::W64 => {
                            if !raw_seed.len().is_multiple_of(8) {
                                return Err(Error::Format("unaligned W64 seed".into()));
                            }
                            let seed: Vec<i64> = raw_seed
                                .chunks_exact(8)
                                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                            if seed.len() != spec.max_lag() * spec.n_state as usize {
                                return Err(Error::Format("recurrence seed size mismatch".into()));
                            }
                            let mut col_refs = Vec::with_capacity(cols.len());
                            for &cid in cols {
                                let ci = as_ints(&vals[cid as usize])?;
                                if ci.len() < *out_words as usize {
                                    return Err(Error::Format(
                                        "RecurBuf input column too short".into(),
                                    ));
                                }
                                col_refs.push(ci);
                            }
                            let total =
                                *out_words as usize + if *include_seed { seed.len() } else { 0 };
                            budget.check_alloc(total.saturating_mul(8), "RecurBuf W64")?;
                            let mut words = vec![0i64; total];
                            if *include_seed {
                                words[..seed.len()].copy_from_slice(&seed);
                                compiled.eval_i64(&seed, &col_refs, &mut words[seed.len()..]);
                            } else {
                                compiled.eval_i64(&seed, &col_refs, &mut words);
                            }
                            let mut bytes = Vec::with_capacity(words.len() * 8);
                            for w in words {
                                bytes.extend_from_slice(&w.to_le_bytes());
                            }
                            Val::Buf(bytes)
                        }
                    }
                }
                GraphNode::ProgramI64 { prog, inputs, bufs } => {
                    let mut in_refs = Vec::with_capacity(inputs.len());
                    let mut n_rows = rows.min(upto);
                    for &iid in inputs {
                        let col = head(as_ints(&vals[iid as usize])?, upto);
                        if in_refs.is_empty() {
                            n_rows = col.len();
                        } else if col.len() != n_rows {
                            return Err(Error::Format("ProgramI64 input length mismatch".into()));
                        }
                        in_refs.push(col);
                    }
                    let mut tbufs = Vec::with_capacity(bufs.len());
                    for &bid in bufs {
                        tbufs.push(TableBuf::from_bytes(as_table_bytes(&vals[bid as usize])?));
                    }
                    budget.check_alloc(n_rows.saturating_mul(8), "ProgramI64")?;
                    let mut out = vec![0i64; n_rows];
                    if in_refs.len() != 1
                        || !tbufs.is_empty()
                        || !super::chain::eval_numeric_chain(prog, in_refs[0], &mut out)
                    {
                        CompiledProgram::with(prog, |cp| cp.eval_i64(&in_refs, &tbufs, &mut out));
                    }
                    Val::Ints(out)
                }
                GraphNode::ProgramCells { prog, inputs, bufs } => {
                    let mut in_refs = Vec::with_capacity(inputs.len());
                    let mut n_rows = rows.min(upto);
                    for &iid in inputs {
                        let col = head(as_ints(&vals[iid as usize])?, upto);
                        if in_refs.is_empty() {
                            n_rows = col.len();
                        } else if col.len() != n_rows {
                            return Err(Error::Format("ProgramCells input length mismatch".into()));
                        }
                        in_refs.push(col);
                    }
                    let mut tbufs = Vec::with_capacity(bufs.len());
                    for &bid in bufs {
                        tbufs.push(TableBuf::from_bytes(as_table_bytes(&vals[bid as usize])?));
                    }
                    Val::Cells(program_cells(prog, &in_refs, &tbufs, n_rows, budget)?)
                }
                GraphNode::PatchCells {
                    base,
                    indices,
                    values,
                } => {
                    let base_c = as_cells(&vals[*base as usize])?;
                    let deltas = as_ints(&vals[*indices as usize])?;
                    let patch_c = as_cells(&vals[*values as usize])?;
                    if deltas.len() != patch_c.len() {
                        return Err(Error::Format(
                            "PatchCells index/value count mismatch".into(),
                        ));
                    }
                    let mut patched_idx = Vec::with_capacity(deltas.len());
                    let mut cur_i = 0usize;
                    for (k, &d) in deltas.iter().enumerate() {
                        if d < 0 {
                            return Err(Error::Format("negative patch index delta".into()));
                        }
                        cur_i = if k == 0 {
                            d as usize
                        } else {
                            cur_i
                                .checked_add(d as usize)
                                .ok_or_else(|| Error::Format("patch index overflow".into()))?
                        };
                        if cur_i >= base_c.len() {
                            return Err(Error::Format("patch index out of bounds".into()));
                        }
                        patched_idx.push(cur_i);
                    }
                    let est = base_c.bytes().len().saturating_add(patch_c.bytes().len());
                    budget.check_alloc(est, "PatchCells")?;
                    let mut data = Vec::with_capacity(est);
                    let mut ends = Vec::with_capacity(base_c.len());
                    let mut p = 0usize;
                    for r in 0..base_c.len() {
                        if p < patched_idx.len() && patched_idx[p] == r {
                            data.extend_from_slice(patch_c.get(p).unwrap());
                            p += 1;
                        } else {
                            data.extend_from_slice(base_c.get(r).unwrap());
                        }
                        ends.push(data.len() as u32);
                    }
                    Val::Cells(Cells::from_parts(data, ends)?)
                }
                GraphNode::MtfInts { codes, misses } => {
                    let codes_col = head(as_ints(&vals[*codes as usize])?, upto);
                    let misses_col = as_ints(&vals[*misses as usize])?;
                    budget.check_alloc(codes_col.len().saturating_mul(8), "MtfInts")?;
                    let mut ring = Recency::new(MTF_WINDOW, codes_col.len());
                    let mut misses_it = misses_col.iter();
                    let mut out = Vec::with_capacity(codes_col.len());
                    for &c in codes_col {
                        let v = if c == 0 {
                            let &v = misses_it.next().ok_or_else(|| {
                                Error::Format("MtfInts miss stream underrun".into())
                            })?;
                            ring.push(v);
                            v
                        } else {
                            usize::try_from(c)
                                .ok()
                                .and_then(|c| ring.hit(c - 1))
                                .ok_or_else(|| Error::Format("MtfInts code out of bounds".into()))?
                        };
                        out.push(v);
                    }
                    if upto == usize::MAX && misses_it.next().is_some() {
                        return Err(Error::Format("MtfInts trailing misses".into()));
                    }
                    Val::Ints(out)
                }
                GraphNode::SliceBuf { buf, base, lens } => {
                    let raw_len = as_table_bytes(&vals[*buf as usize])?.len();
                    let lens_col = as_ints(&vals[*lens as usize])?;
                    let start = *base as usize;
                    if start > raw_len {
                        return Err(Error::Format("SliceBuf base out of bounds".into()));
                    }
                    let mut ends = Vec::with_capacity(lens_col.len());
                    let mut acc = 0usize;
                    for &l in lens_col {
                        if l < 0 {
                            return Err(Error::Format("negative SliceBuf cell length".into()));
                        }
                        acc = acc
                            .checked_add(l as usize)
                            .ok_or_else(|| Error::Format("SliceBuf overflow".into()))?;
                        if acc > u32::MAX as usize {
                            return Err(Error::BudgetExceeded("SliceBuf column bytes"));
                        }
                        ends.push(acc as u32);
                    }
                    let end = start
                        .checked_add(acc)
                        .ok_or_else(|| Error::Format("SliceBuf overflow".into()))?;
                    if end > raw_len {
                        return Err(Error::Format("SliceBuf slice exceeds buffer".into()));
                    }
                    budget.check_alloc(acc, "SliceBuf")?;
                    let can_steal_buf = start == 0
                        && end == raw_len
                        && *buf != *lens
                        && rem_uses[*buf as usize] == 1
                        && matches!(vals[*buf as usize], Val::Buf(_));
                    let data = if can_steal_buf {
                        match std::mem::replace(&mut vals[*buf as usize], Val::Empty) {
                            Val::Buf(b) => b,
                            _ => unreachable!(),
                        }
                    } else {
                        as_table_bytes(&vals[*buf as usize])?[start..end].to_vec()
                    };
                    Val::Cells(Cells::from_parts(data, ends)?)
                }
                GraphNode::PagedSlice {
                    buf,
                    lens,
                    pages,
                    bases,
                } => {
                    let bytes = as_table_bytes(&vals[*buf as usize])?;
                    let spans = paged_spans(
                        as_ints(&vals[*lens as usize])?,
                        as_ints(&vals[*pages as usize])?,
                        as_ints(&vals[*bases as usize])?,
                        bytes.len(),
                        budget,
                    )?;
                    Val::Cells(gather(bytes, &spans)?)
                }
                GraphNode::IndexCells { source, indices } => {
                    let src = as_cells(&vals[*source as usize])?;
                    let idx_col = as_ints(&vals[*indices as usize])?;
                    let mut data = Vec::with_capacity(idx_col.len() * 8);
                    let mut ends = Vec::with_capacity(idx_col.len());
                    for &i in idx_col {
                        if i < 0 || (i as usize) >= src.len() {
                            return Err(Error::Format("IndexCells index out of bounds".into()));
                        }
                        let sl = src.get(i as usize).unwrap();
                        if data.len().saturating_add(sl.len()) > budget.max_single_alloc_bytes {
                            return Err(Error::BudgetExceeded("IndexCells expanded bytes"));
                        }
                        data.extend_from_slice(sl);
                        ends.push(data.len() as u32);
                    }
                    Val::Cells(Cells::from_parts(data, ends)?)
                }
                GraphNode::TemplateCells { template, slots } => {
                    let slot_cols: Vec<&Cells> = slots
                        .iter()
                        .map(|&sid| as_cells(&vals[sid as usize]))
                        .collect::<Result<_>>()?;
                    let slot_bytes: usize = slot_cols.iter().map(|c| c.bytes().len()).sum();
                    // A constant template is split once rather than per row.
                    let (n, fixed, per_row, cap) = match &nodes[*template as usize] {
                        GraphNode::ConstCells { val, count } => {
                            let mut pieces = Vec::new();
                            walk_template(val, slot_cols.len(), |pc| pieces.push(pc))?;
                            let n = *count as usize;
                            (n, pieces, None, slot_bytes + val.len().saturating_mul(n))
                        }
                        _ => {
                            let tmpl = as_cells(&vals[*template as usize])?;
                            let cap = slot_bytes + tmpl.bytes().len();
                            (tmpl.len(), Vec::new(), Some(tmpl), cap)
                        }
                    };
                    if slot_cols.iter().any(|c| c.len() != n) {
                        return Err(Error::Format("TemplateCells row count mismatch".into()));
                    }
                    let mut data = Vec::with_capacity(cap.min(budget.max_single_alloc_bytes));
                    let mut ends = Vec::with_capacity(n);
                    let emit = |data: &mut Vec<u8>, piece: &Piece, r: usize| match *piece {
                        Piece::Lit(b) => data.extend_from_slice(b),
                        Piece::Slot(i, strip) => {
                            let sv = slot_cols[i].get(r).unwrap();
                            data.extend_from_slice(if strip && sv.len() >= 2 {
                                &sv[1..sv.len() - 1]
                            } else {
                                sv
                            });
                        }
                    };
                    for r in 0..n {
                        match per_row {
                            None => fixed.iter().for_each(|pc| emit(&mut data, pc, r)),
                            Some(t) => walk_template(t.get(r).unwrap(), slot_cols.len(), |pc| {
                                emit(&mut data, &pc, r)
                            })?,
                        }
                        if data.len() > budget.max_single_alloc_bytes {
                            return Err(Error::BudgetExceeded("TemplateCells expanded bytes"));
                        }
                        ends.push(data.len() as u32);
                    }
                    Val::Cells(Cells::from_parts(data, ends)?)
                }
                GraphNode::ConstCells { val, count } => {
                    let n = *count as usize;
                    let total = val
                        .len()
                        .checked_mul(n)
                        .ok_or(Error::BudgetExceeded("ConstCells"))?;
                    budget.check_alloc(total, "ConstCells")?;
                    let mut data = Vec::with_capacity(total);
                    let mut ends = Vec::with_capacity(n);
                    for _ in 0..n {
                        data.extend_from_slice(val);
                        ends.push(data.len() as u32);
                    }
                    Val::Cells(Cells::from_parts(data, ends)?)
                }
                GraphNode::QuoteOrNull { inner, map } => {
                    let inner_c = as_cells(&vals[*inner as usize])?;
                    if let Some(m_id) = map {
                        let m_ints = as_ints(&vals[*m_id as usize])?;
                        let mut data = Vec::with_capacity(inner_c.bytes().len() + m_ints.len() * 2);
                        let mut ends = Vec::with_capacity(m_ints.len());
                        for &j in m_ints {
                            if j < 0 {
                                data.extend_from_slice(b"null");
                            } else {
                                let sl = inner_c
                                    .get(j as usize)
                                    .ok_or_else(|| Error::Format("QuoteOrNull index".into()))?;
                                data.push(b'"');
                                data.extend_from_slice(sl);
                                data.push(b'"');
                            }
                            if data.len() > budget.max_single_alloc_bytes {
                                return Err(Error::BudgetExceeded("QuoteOrNull"));
                            }
                            ends.push(data.len() as u32);
                        }
                        Val::Cells(Cells::from_parts(data, ends)?)
                    } else {
                        let n = inner_c.len();
                        let total = inner_c.bytes().len().saturating_add(n.saturating_mul(2));
                        budget.check_alloc(total, "QuoteOrNull")?;
                        let mut data = Vec::with_capacity(total);
                        let mut ends = Vec::with_capacity(n);
                        for r in 0..n {
                            data.push(b'"');
                            data.extend_from_slice(inner_c.get(r).unwrap());
                            data.push(b'"');
                            ends.push(data.len() as u32);
                        }
                        Val::Cells(Cells::from_parts(data, ends)?)
                    }
                }
                GraphNode::ParseAsciiInt { cells } => {
                    let c = as_cells(&vals[*cells as usize])?;
                    budget.check_alloc(c.len().saturating_mul(8), "ParseAsciiInt")?;
                    let mut out = Vec::with_capacity(c.len());
                    for r in 0..c.len() {
                        out.push(parse_cell_ascii_i64(c.get(r).unwrap()));
                    }
                    Val::Ints(out)
                }
                GraphNode::ExtPool => {
                    let pool = ext.pool.ok_or(Error::NotLoaded)?;
                    Val::SharedBuf(Arc::clone(pool))
                }
                GraphNode::ExtDict { dict_id } => match ext.dict {
                    Some((id, cells)) if id == *dict_id => Val::SharedCells(Arc::clone(cells)),
                    _ => return Err(Error::NotLoaded),
                },
                GraphNode::ExtPeer { slot } => {
                    let peer = ext.peers.get(*slot as usize).ok_or(Error::NotLoaded)?;
                    Val::SharedCells(Arc::clone(peer))
                }
                GraphNode::ExtAlign { .. } | GraphNode::ExtVariant => {
                    let v = ext.aligned.get(n_aligned).ok_or(Error::NotLoaded)?;
                    n_aligned += 1;
                    Val::Ints(v.clone())
                }
                GraphNode::KeyedSum { keys, deltas, order, seed } => {
                    // A later input row can precede an early requested row in scan order.
                    let limit = if order.is_some() { usize::MAX } else { upto };
                    let keys = as_ints(&vals[*keys as usize])?;
                    let order = order.map(|o| as_ints(&vals[o as usize])).transpose()?;
                    Val::Ints(keyed_sum(
                        head(keys, limit), head(as_ints(&vals[*deltas as usize])?, limit),
                        order, *seed, keys.len(), budget,
                    )?)
                }
                GraphNode::Interleave { sel, streams } => {
                    let sel = head(as_ints(&vals[*sel as usize])?, upto);
                    let streams: Vec<&[i64]> = (streams.iter())
                        .map(|&s| as_ints(&vals[s as usize]))
                        .collect::<Result<_>>()?;
                    budget.check_alloc(sel.len().saturating_mul(8), "Interleave")?;
                    let mut next = vec![0usize; streams.len()];
                    let mut out = Vec::with_capacity(sel.len());
                    for &s in sel {
                        let v = usize::try_from(s).ok().and_then(|s| {
                            let at = next.get_mut(s)?;
                            *at += 1;
                            streams[s].get(*at - 1)
                        });
                        out.push(*v.ok_or_else(|| Error::Format("Interleave stream underrun".into()))?);
                    }
                    if upto == usize::MAX && next.iter().zip(&streams).any(|(&n, s)| n != s.len()) {
                        return Err(Error::Format("Interleave trailing values".into()));
                    }
                    Val::Ints(out)
                }
                GraphNode::CellIds { cells } => {
                    let c = as_cells(&vals[*cells as usize])?;
                    budget.check_alloc(c.len().saturating_mul(12), "CellIds")?;
                    Val::Ints(value_ids(c).into_iter().map(i64::from).collect())
                }
                GraphNode::MatchIndex {
                    needles,
                    hay,
                    codes,
                } => Val::Ints(match_index(
                    as_ints(&vals[*needles as usize])?,
                    as_ints(&vals[*hay as usize])?,
                    as_ints(&vals[*codes as usize])?,
                    budget,
                )?),
            };
            node.for_each_dep(|dep| {
                let d = dep as usize;
                rem_uses[d] -= 1;
                if rem_uses[d] == 0 {
                    vals[d] = Val::Empty;
                }
            });
            vals.push(val);
        }

        Ok(vals)
    }
}

enum Piece<'a> {
    Lit(&'a [u8]),
    Slot(usize, bool),
}

/// `KeyedSum` over a dense state per key; keys are bounded by the row count.
fn keyed_sum(
    keys: &[i64],
    deltas: &[i64],
    order: Option<&[i64]>,
    seed: SumSeed,
    key_limit: usize,
    budget: &DecodeBudget,
) -> Result<Vec<i64>> {
    if keys.len() != deltas.len() || order.is_some_and(|o| o.len() != keys.len()) {
        return Err(Error::Format("KeyedSum length mismatch".into()));
    }
    let max = keys.iter().copied().max().unwrap_or(0);
    if keys.iter().any(|&k| k < 0) || max as u64 > key_limit as u64 {
        return Err(Error::Format("KeyedSum key out of range".into()));
    }
    budget.check_alloc((max as usize + 1 + keys.len()).saturating_mul(16), "KeyedSum")?;
    let indices = order.map(|o| {
        let mut indices: Vec<usize> = (0..keys.len()).filter(|&i| keys[i] != 0).collect();
        indices.sort_unstable_by_key(|&i| (o[i], i));
        indices
    });
    let mut state = vec![0i64; max as usize + 1];
    let mut seen = if seed == SumSeed::Previous { vec![false; state.len()] } else { Vec::new() };
    let mut last = 0i64;
    let mut out = deltas.to_vec();
    let mut step = |i: usize| {
        let k = keys[i] as usize;
        if k == 0 { return; }
        if seed == SumSeed::Previous && !seen[k] {
            state[k] = last;
            seen[k] = true;
        }
        state[k] = state[k].wrapping_add(deltas[i]);
        last = state[k];
        out[i] = state[k];
    };
    if let Some(indices) = indices {
        indices.into_iter().for_each(step);
    } else {
        (0..keys.len()).for_each(&mut step);
    }
    Ok(out)
}

/// Each cell's value id: the rank of its value's first occurrence. Open addressing on a
/// multiply-fold hash; a hash match is confirmed on the bytes.
pub(crate) fn value_ids(cells: &Cells) -> Vec<u32> {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let word = |c: &[u8], at: usize| u64::from_le_bytes(c[at..at + 8].try_into().unwrap());
    // Whole words, then the last eight bytes again when they overlap; short cells byte by byte.
    let hash = |c: &[u8]| {
        let mut h = c.len() as u64;
        if c.len() >= 8 {
            for at in (0..c.len() - 7).step_by(8) {
                h = (h ^ word(c, at)).wrapping_mul(K).rotate_left(29);
            }
            h ^= word(c, c.len() - 8);
        } else {
            for &b in c {
                h = (h << 8) | u64::from(b);
            }
        }
        (h ^ (h >> 29)).wrapping_mul(K) >> 32
    };
    let (bytes, ends) = (cells.bytes(), cells.ends());
    let mask = (ends.len() * 2).next_power_of_two().max(16) - 1;
    // (hash, first cell's start and end, id); an empty slot has `end == 0` and `start == 1`.
    let mut slots = vec![(0u64, 1u32, 0u32, 0u32); mask + 1];
    let mut out = Vec::with_capacity(ends.len());
    let (mut next, mut start) = (0u32, 0u32);
    for &end in ends {
        let c = &bytes[start as usize..end as usize];
        let h = hash(c);
        let mut p = h as usize & mask;
        loop {
            let (sh, s0, s1, id) = slots[p];
            if s0 > s1 {
                slots[p] = (h, start, end, next);
                out.push(next);
                next += 1;
                break;
            }
            if sh == h && &bytes[s0 as usize..s1 as usize] == c {
                out.push(id);
                break;
            }
            p = (p + 1) & mask;
        }
        start = end;
    }
    out
}

/// `MatchIndex`: hay positions grouped by key in a counting sort, so each lookup is O(1). Keys
/// index dense tables, so hay keys must be small nonnegative integers.
fn match_index(
    needles: &[i64],
    hay: &[i64],
    codes: &[i64],
    budget: &DecodeBudget,
) -> Result<Vec<i64>> {
    if codes.len() != needles.len() {
        return Err(Error::Format("MatchIndex code count mismatch".into()));
    }
    let max = hay.iter().copied().max().unwrap_or(-1);
    if hay.iter().any(|&h| h < 0) || max >= u32::MAX as i64 {
        return Err(Error::Format("MatchIndex key out of range".into()));
    }
    let groups = (max + 1) as usize;
    budget.check_alloc(
        groups.saturating_add(hay.len()).saturating_mul(8) + codes.len().saturating_mul(8),
        "MatchIndex",
    )?;
    // Counts per key become exclusive starts, then each key's hay positions fill in order.
    let mut starts = vec![0u32; groups + 1];
    for &h in hay {
        starts[h as usize + 1] += 1;
    }
    for g in 0..groups {
        starts[g + 1] += starts[g];
    }
    let mut fill = starts.clone();
    let mut by_key = vec![0u32; hay.len()];
    for (j, &h) in hay.iter().enumerate() {
        by_key[fill[h as usize] as usize] = j as u32;
        fill[h as usize] += 1;
    }
    let mut out = Vec::with_capacity(codes.len());
    for (&n, &code) in needles.iter().zip(codes) {
        if code == 0 {
            out.push(0);
            continue;
        }
        let pos = usize::try_from(n)
            .ok()
            .filter(|&n| n < groups)
            .and_then(|n| {
                let (lo, hi) = (starts[n] as i64, starts[n + 1] as i64);
                (code > 0 && code <= hi - lo).then(|| by_key[(lo + code - 1) as usize])
            });
        out.push(pos.ok_or_else(|| Error::Format("MatchIndex code out of bounds".into()))? as i64);
    }
    Ok(out)
}

fn fill_low_bytes<const W: usize>(lanes: &[i64], out: &mut [u8]) {
    for (d, v) in out.chunks_exact_mut(W).zip(lanes) {
        d.copy_from_slice(&v.to_le_bytes()[..W]);
    }
}

/// Splits one `TemplateCells` template into literal runs and slot references.
fn walk_template<'a>(tb: &'a [u8], n_slots: usize, mut f: impl FnMut(Piece<'a>)) -> Result<()> {
    let mut p = 0;
    while let Some(hit) = tb[p..].iter().position(|&b| b == 0x1f) {
        if hit > 0 {
            f(Piece::Lit(&tb[p..p + hit]));
        }
        let code = *tb
            .get(p + hit + 1)
            .ok_or_else(|| Error::Format("truncated TemplateCells escape".into()))?;
        if code == 0xff {
            f(Piece::Lit(&[0x1f]));
        } else if (code >> 1) as usize >= n_slots {
            return Err(Error::Format("TemplateCells slot out of bounds".into()));
        } else {
            f(Piece::Slot((code >> 1) as usize, code & 1 != 0));
        }
        p += hit + 2;
    }
    if p < tb.len() {
        f(Piece::Lit(&tb[p..]));
    }
    Ok(())
}

#[inline]
pub(crate) fn parse_cell_ascii_i64(mut s: &[u8]) -> i64 {
    if s.len() >= 2 && s[0] == b'"' && s[s.len() - 1] == b'"' {
        s = &s[1..s.len() - 1];
    }
    if s.is_empty() {
        return 0;
    }
    let (neg, digits) = if s[0] == b'-' {
        (true, &s[1..])
    } else if s[0] == b'+' {
        (false, &s[1..])
    } else {
        (false, s)
    };
    if digits.is_empty() || digits.len() > 19 {
        return 0;
    }
    let mut acc = 0u64;
    for &b in digits {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return 0;
        }
        acc = match acc.checked_mul(10).and_then(|x| x.checked_add(d as u64)) {
            Some(v) => v,
            None => return 0,
        };
    }
    if neg {
        if acc == (i64::MAX as u64) + 1 {
            i64::MIN
        } else if acc <= i64::MAX as u64 {
            -(acc as i64)
        } else {
            0
        }
    } else if acc <= i64::MAX as u64 {
        acc as i64
    } else {
        0
    }
}

fn as_buf(v: &Val) -> Result<&[u8]> {
    match v {
        Val::Buf(b) => Ok(b),
        Val::SharedBuf(b) => b.all(),
        _ => Err(Error::Format("expected Buf node".into())),
    }
}

/// The length of a buffer, which a pool knows without holding all its bytes.
fn buf_len(v: &Val) -> Result<usize> {
    match v {
        Val::SharedBuf(p) => Ok(p.len()),
        v => Ok(as_table_bytes(v)?.len()),
    }
}

/// Each `PagedSlice` cell's `(start, end)` in a buffer of `buf_len` bytes.
/// A `PagedSlice` node's column from its `[buf, lens, pages, bases]` inputs. Cells in pages the
/// pool lacks get `NOT_HELD`.
fn paged_col(
    vals: &[Val],
    [buf, lens, pages, bases]: [u16; 4],
    budget: &DecodeBudget,
) -> Result<Col> {
    let mut spans = paged_spans(
        as_ints(&vals[lens as usize])?,
        as_ints(&vals[pages as usize])?,
        as_ints(&vals[bases as usize])?,
        buf_len(&vals[buf as usize])?,
        budget,
    )?;
    match &vals[buf as usize] {
        Val::SharedBuf(pool) => {
            if !pool.is_whole() {
                spans
                    .iter_mut()
                    .filter(|(s, e)| !pool.held(*s as usize, *e as usize))
                    .for_each(|s| *s = NOT_HELD);
            }
            Ok(Col::Paged {
                pool: Arc::clone(pool),
                spans,
            })
        }
        v => Ok(Col::Flat(gather(as_table_bytes(v)?, &spans)?)),
    }
}

fn paged_spans(
    lens: &[i64],
    pages: &[i64],
    bases: &[i64],
    buf_len: usize,
    budget: &DecodeBudget,
) -> Result<Vec<(u32, u32)>> {
    if lens.len() != pages.len() {
        return Err(Error::Format("PagedSlice page count != cell count".into()));
    }
    if buf_len > u32::MAX as usize {
        return Err(Error::BudgetExceeded("PagedSlice buffer"));
    }
    budget.check_alloc(lens.len().saturating_mul(8), "PagedSlice")?;
    let mut at: Vec<u64> = Vec::with_capacity(bases.len());
    for &b in bases {
        if b < 0 || b as u64 > buf_len as u64 {
            return Err(Error::Format("PagedSlice base out of bounds".into()));
        }
        at.push(b as u64);
    }
    let starts = at.clone();
    let mut spans = Vec::with_capacity(lens.len());
    for (&l, &p) in lens.iter().zip(pages) {
        let next = usize::try_from(p).ok().and_then(|p| at.get_mut(p));
        let (Some(next), true) = (next, l >= 0) else {
            return Err(Error::Format("invalid PagedSlice cell".into()));
        };
        let (s, e) = (*next, *next + l as u64);
        if e > buf_len as u64 {
            return Err(Error::Format("PagedSlice cell exceeds buffer".into()));
        }
        *next = e;
        spans.push((s as u32, e as u32));
    }
    // Runs may not overlap, so the cells' bytes stay within the buffer's.
    let mut runs: Vec<(u64, u64)> = starts.into_iter().zip(at).collect();
    runs.sort_unstable();
    if runs.windows(2).any(|w| w[0].1 > w[1].0) {
        return Err(Error::Format("PagedSlice runs overlap".into()));
    }
    Ok(spans)
}

/// The bytes of `spans` as cells.
fn gather(bytes: &[u8], spans: &[(u32, u32)]) -> Result<Cells> {
    let mut out = Cells::with_capacity(spans.len(), 0);
    for &(s, e) in spans {
        out.tail().extend_from_slice(&bytes[s as usize..e as usize]);
        out.seal();
    }
    Ok(out)
}

fn as_table_bytes(v: &Val) -> Result<&[u8]> {
    match v {
        Val::Buf(b) => Ok(b),
        Val::SharedBuf(b) => b.all(),
        Val::Ints(x) => {
            Ok(unsafe { core::slice::from_raw_parts(x.as_ptr().cast::<u8>(), x.len() * 8) })
        }
        Val::Cells(c) => Ok(c.bytes()),
        Val::SharedCells(c) => Ok(c.bytes()),
        Val::Empty => Err(Error::Format("referenced empty node".into())),
    }
}

fn as_ints(v: &Val) -> Result<&[i64]> {
    match v {
        Val::Ints(x) => Ok(x),
        _ => Err(Error::Format("expected IntStream/ProgramI64 node".into())),
    }
}

fn as_cells(v: &Val) -> Result<&Cells> {
    match v {
        Val::Cells(c) => Ok(c),
        Val::SharedCells(c) => Ok(c),
        _ => Err(Error::Format("expected Cells node".into())),
    }
}

fn head(col: &[i64], n: usize) -> &[i64] {
    &col[..col.len().min(n)]
}

/// By node: how many leading rows of its output are read when the root's first `upto` are.
/// Rows flow one to one only through programs, whose scans run forward, the codes of `MtfInts`
/// and a quote without a map; any other input is read whole (`usize::MAX`, as is every node
/// when the list is empty).
fn prefix_lens(nodes: &[GraphNode], root: usize, upto: usize) -> Vec<usize> {
    if upto == usize::MAX || root >= nodes.len() {
        return Vec::new();
    }
    let mut lens = vec![0usize; nodes.len()];
    lens[root] = upto;
    for idx in (0..nodes.len()).rev() {
        // Unreachable from the root: run whole, as before.
        if lens[idx] == 0 {
            lens[idx] = usize::MAX;
        }
        let n = lens[idx];
        let mut need = |d: u16, n: usize| {
            if let Some(l) = lens.get_mut(d as usize) {
                *l = (*l).max(n);
            }
        };
        match &nodes[idx] {
            GraphNode::ProgramI64 { inputs, bufs, .. }
            | GraphNode::ProgramCells { inputs, bufs, .. } => {
                inputs.iter().for_each(|&d| need(d, n));
                bufs.iter().for_each(|&d| need(d, usize::MAX));
            }
            GraphNode::MtfInts { codes, misses } => {
                need(*codes, n);
                need(*misses, usize::MAX);
            }
            GraphNode::QuoteOrNull { inner, map: None } => need(*inner, n),
            GraphNode::KeyedSum { keys, deltas, order: None, .. } => {
                // Key ids are bounded by the complete stream, even for a short prefix read.
                need(*keys, usize::MAX);
                need(*deltas, n);
            }
            GraphNode::Interleave { sel, streams } => {
                need(*sel, n);
                streams.iter().for_each(|&d| need(d, usize::MAX));
            }
            node => node.for_each_dep(|d| need(d, usize::MAX)),
        }
    }
    lens
}

/// The cells of `n_rows` rows a program builds from its inputs' rows.
fn program_cells(
    prog: &Code,
    in_refs: &[&[i64]],
    tbufs: &[TableBuf<'_>],
    n_rows: usize,
    budget: &DecodeBudget,
) -> Result<Cells> {
    let rb = prog.row_bytes();
    let cap = n_rows
        .checked_mul(rb)
        .ok_or(Error::BudgetExceeded("ProgramCells"))?;
    budget.check_alloc(cap + 8, "ProgramCells")?;
    // Fixed-width cells of a lane's low bytes skip the program engine.
    if let (Some(&x), [(0, Len::Fixed(w))], true) =
        (in_refs.first(), &prog.sinks[..], prog.ops.is_empty())
    {
        let mut data = vec![0u8; cap];
        match *w {
            8 => fill_low_bytes::<8>(x, &mut data),
            4 => fill_low_bytes::<4>(x, &mut data),
            2 => fill_low_bytes::<2>(x, &mut data),
            w => {
                for (d, v) in data.chunks_exact_mut(w as usize).zip(x) {
                    d.copy_from_slice(&v.to_le_bytes()[..w as usize]);
                }
            }
        }
        Cells::from_fixed(n_rows, rb, data)
    } else if !prog.varlen() {
        let mut data = vec![0u8; cap];
        CompiledProgram::with(prog, |cp| cp.eval_fixed(in_refs, tbufs, n_rows, &mut data));
        Cells::from_fixed(n_rows, rb, data)
    } else {
        let mut ends = vec![0u32; n_rows];
        let mut data = vec![0u8; cap + 8];
        let written = CompiledProgram::with(prog, |cp| {
            cp.eval_cells(in_refs, tbufs, n_rows, &mut ends, &mut data)
        });
        data.truncate(written);
        Cells::from_parts(data, ends)
    }
}

/// The first `n` cells of a row-local `ProgramCells` node, built only where `reads` reads.
fn sparse_cells(
    node: &GraphNode,
    vals: &[Val],
    n: usize,
    reads: &[(usize, usize)],
    budget: &DecodeBudget,
) -> Result<Col> {
    let GraphNode::ProgramCells { prog, inputs, bufs } = node else {
        return Err(Error::Format(
            "sparse cells of a node that is no program".into(),
        ));
    };
    let at: Vec<usize> = reads.iter().flat_map(|&(s, e)| s..e.min(n)).collect();
    let mut cols = Vec::with_capacity(inputs.len());
    for &iid in inputs {
        let col = as_ints(&vals[iid as usize])?;
        if col.len() < n {
            return Err(Error::Format("ProgramCells input length mismatch".into()));
        }
        cols.push(at.iter().map(|&r| col[r]).collect::<Vec<i64>>());
    }
    let in_refs: Vec<&[i64]> = cols.iter().map(Vec::as_slice).collect();
    let mut tbufs = Vec::with_capacity(bufs.len());
    for &bid in bufs {
        tbufs.push(TableBuf::from_bytes(as_table_bytes(&vals[bid as usize])?));
    }
    let cells = program_cells(prog, &in_refs, &tbufs, at.len(), budget)?;
    let mut spans = vec![NOT_HELD; n];
    let mut start = 0u32;
    for (&r, &end) in at.iter().zip(cells.ends()) {
        spans[r] = (start, end);
        start = end;
    }
    Ok(Col::Paged {
        pool: Arc::new(Pool::whole(cells.bytes().to_vec())),
        spans,
    })
}

fn take_shared_cells(v: &mut Val) -> Result<Arc<Cells>> {
    match std::mem::replace(v, Val::Empty) {
        Val::SharedCells(sc) => Ok(sc),
        Val::Cells(c) => Ok(Arc::new(c)),
        _ => Err(Error::Format("expected Cells node".into())),
    }
}

fn put_u16_list(out: &mut Vec<u8>, items: &[u16]) {
    put_uvarint(out, items.len() as u64);
    for &x in items {
        put_uvarint(out, x as u64);
    }
}

fn get_u16_list(cur: &mut Cursor<'_>, limit: usize) -> Result<Vec<u16>> {
    let n = cur.uvar()? as usize;
    if n > 64 {
        return Err(Error::Format("node list too long".into()));
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let r = cur.uvar()? as usize;
        if r >= limit {
            return Err(Error::Format("node reference out of bounds".into()));
        }
        out.push(r as u16);
    }
    Ok(out)
}

fn bitpack_u64(vals: impl Iterator<Item = u64>, bits: u8, len: usize) -> Vec<u8> {
    if bits == 0 {
        return Vec::new();
    }
    let b = bits as u32;
    let mask = if b == 64 { u64::MAX } else { (1u64 << b) - 1 };
    // A register accumulator keeps each lane off the bytes the previous lane just stored.
    let mut out = Vec::with_capacity((len * b as usize).div_ceil(8) + 8);
    let (mut acc, mut held) = (0u64, 0u32);
    for v in vals {
        let v = v & mask;
        acc |= v << held;
        held += b;
        if held >= 64 {
            out.extend_from_slice(&acc.to_le_bytes());
            held -= 64;
            // The lane's top `held` bits did not fit in the word just written.
            acc = v.checked_shr(b - held).unwrap_or(0);
        }
    }
    out.extend_from_slice(&acc.to_le_bytes()[..held.div_ceil(8) as usize]);
    out
}

fn bitunpack_u64(mut bytes: Vec<u8>, bits: u8, len: usize, min: i64) -> Vec<i64> {
    if bits == 0 {
        return vec![min; len];
    }
    let (b, mask) = (bits as usize, u64::MAX >> (64 - bits));
    // Zero padding gives every lane a whole 16-byte window, so no lane takes a tail path.
    bytes.resize(bytes.len() + 16, 0);
    (0..len)
        .map(|i| {
            let at = i * b / 8;
            let w = u128::from_le_bytes(bytes[at..at + 16].try_into().unwrap());
            (min as u64).wrapping_add((w >> (i * b % 8)) as u64 & mask) as i64
        })
        .collect()
}

fn read_numeric(
    cur: &mut Cursor<'_>,
    rows: u32,
    flags: u8,
    budget: &DecodeBudget,
) -> Result<(IntStream, Option<Program>, usize)> {
    budget.check_alloc((rows as usize).saturating_mul(8), "numeric lanes")?;
    if flags & 0x80 == 0 {
        let stream = IntStream::decode_wire(cur, budget, 0)?;
        if stream.len() != rows as usize {
            return Err(Error::Format("numeric graph row count mismatch".into()));
        }
        let code = cur.remaining();
        let prog = if flags == 16 {
            Some(decode_lane_program(cur.slice(code)?)?)
        } else {
            None
        };
        if cur.remaining() != 0 {
            return Err(Error::Format("trailing numeric graph bytes".into()));
        }
        return Ok((stream, prog, code));
    }
    let ops = (flags >> 4) & 7;
    let mut rest = &cur.buf[cur.pos..];
    let prog = if ops > 0 {
        Some(Program::decode_linear_ops(&mut rest, u32::from(ops), true)?)
    } else {
        None
    };
    let code = cur.remaining() - rest.len();
    cur.slice(code)?;
    let len = rows;
    budget.check_alloc(cur.remaining(), "numeric transport bytes")?;
    let stream = match flags & 15 {
        0 => IntStream::Const {
            len,
            val: cur.ivar()?,
        },
        7 => {
            let shift = cur.u8()?;
            if !(1..64).contains(&shift) {
                return Err(Error::Format("invalid constant shift".into()));
            }
            let c = cur.ivar()?;
            let val = c.wrapping_shl(u32::from(shift));
            if val >> shift != c {
                return Err(Error::Format("constant shift overflow".into()));
            }
            IntStream::Const { len, val }
        }
        4 | 5 => {
            let data = cur.slice(cur.remaining())?;
            let leaf = if flags & 15 == 4 {
                let mut leaf = Vec::with_capacity(data.len() + 12);
                crate::wire::put_leaf(&mut leaf, EntropyBackend::Raw, data.len(), data);
                leaf
            } else {
                data.to_vec()
            };
            IntStream::ZigZagVarint { len, leaf }
        }
        1 => IntStream::Rle {
            len,
            leaf: cur.slice(cur.remaining())?.to_vec(),
        },
        2 | 3 => {
            let min = cur.ivar()?;
            let bits = cur.u8()?;
            if bits > 64 {
                return Err(Error::Format("invalid integer bit width".into()));
            }
            let raw = cur.slice(cur.remaining())?;
            let leaf = if flags & 15 == 3 {
                if raw.len() != (u64::from(len) * u64::from(bits)).div_ceil(8) as usize {
                    return Err(Error::Format("invalid packed width".into()));
                }
                let mut v = Vec::with_capacity(raw.len() + 12);
                crate::wire::put_leaf(&mut v, EntropyBackend::Raw, raw.len(), raw);
                v
            } else {
                raw.to_vec()
            };
            IntStream::ForBitpack {
                len,
                min,
                bits,
                leaf,
            }
        }
        8 => IntStream::Rice {
            len,
            bytes: cur.slice(cur.remaining())?.to_vec(),
        },
        11 => {
            let min = cur.ivar()?;
            let radix =
                u32::try_from(cur.uvar()?).map_err(|_| Error::Format("invalid radix".into()))?;
            let layout = super::radix::Layout::new(radix)
                .ok_or_else(|| Error::Format("invalid radix".into()))?;
            IntStream::Radix {
                len,
                min,
                radix,
                bytes: cur.slice(layout.size(len as usize))?.to_vec(),
            }
        }
        _ => return Err(Error::Format("unknown numeric stream".into())),
    };
    if cur.remaining() != 0 {
        return Err(Error::Format("trailing numeric stream bytes".into()));
    }
    Ok((stream, prog, code))
}

fn decode_lane_program(bytes: &[u8]) -> Result<Program> {
    let prog = Program::decode(bytes)?;
    if prog.width != Width::W64
        || prog.n_in != 1
        || prog.n_bufs != 0
        || prog.sinks.len() != 1
        || prog.sinks[0].1 != Len::Fixed(8)
    {
        return Err(Error::Format("numeric lane program shape mismatch".into()));
    }
    Ok(prog)
}

#[cfg(test)]
#[test]
fn bitpack_roundtrips_every_width() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for bits in 1..=64u8 {
        let mask = u64::MAX >> (64 - bits);
        for len in [1, 7, 64, 65, 333] {
            let vals: Vec<u64> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state & mask
                })
                .collect();
            let packed = bitpack_u64(vals.iter().copied(), bits, len);
            assert_eq!(packed.len(), (len * bits as usize).div_ceil(8));
            let back = bitunpack_u64(packed, bits, len, 0);
            assert!(back.iter().zip(&vals).all(|(&a, &b)| a as u64 == b), "{bits}");
        }
    }
}
