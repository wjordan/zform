//! Self-describing column payload graph (`PayloadGraph`) and the zero-knowledge [`CoreDecoder`].
//!
//! [`CoreDecoder`] knows nothing about timestamps, UUIDs, ULIDs, IPs, emails, decimals, floats,
//! Java LCGs, or MT19937: every codec pack compiles down to a [`PayloadGraph`] of domain-agnostic
//! byte, integer, recurrence, and SSA lane-program primitives, while returning lazy [`Col`] views
//! (`Const`, `Dict`, `Pool`, `Quoted`, `Flat`) so selective queries and skipped rows stay fast.

use std::cell::Cell;
use std::sync::Arc;

use crate::cells::{Cells, Col, NULL_CELL};
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, compress_leaf, decompress_leaf,
    put_ivarint, put_uvarint,
};

use super::engine::CompiledProgram;
use super::ir::{Builder, Invalid, Len, Program, TableBuf, Width};
use super::library::{Code, Library, code_discount, table_bytes};
use super::recur::{CompiledRecurrence, RecurrenceSpec};

thread_local! {
    static ENC_OPTS: Cell<(EntropyBackend, i32)> =
        const { Cell::new((EntropyBackend::Zstd, 3)) };
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

#[inline]
pub(crate) fn pack_leaf(raw: &[u8]) -> Vec<u8> {
    let (backend, level) = ENC_OPTS.with(|c| c.get());
    let mut out = Vec::with_capacity(raw.len() + 12);
    compress_leaf(&mut out, raw, backend, level);
    out
}

#[inline]
pub(crate) fn unpack_leaf(leaf: &[u8], budget: &DecodeBudget) -> Result<Vec<u8>> {
    let mut cur = Cursor::new(leaf);
    let raw = decompress_leaf(&mut cur, budget)?;
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
        self.0
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
    /// Automatically chooses the smallest domain-agnostic encoding for `vals`.
    pub fn auto_encode(vals: &[i64]) -> Self {
        Self::auto_encode_inner(vals, true)
    }

    fn auto_encode_inner(vals: &[i64], allow_tail: bool) -> Self {
        let len = vals.len() as u32;
        if vals.is_empty() {
            return Self::Const { val: 0, len: 0 };
        }
        let first = vals[0];
        if vals.iter().all(|&x| x == first) {
            return Self::Const { val: first, len };
        }

        let mut runs = 1usize;
        for w in vals.windows(2) {
            if w[0] != w[1] {
                runs += 1;
            }
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

        let min = *vals.iter().min().unwrap();
        let max = *vals.iter().max().unwrap();
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
                for (i, &v) in vals.iter().enumerate() {
                    let le = (v as u64).wrapping_sub(min as u64).to_le_bytes();
                    for p in 0..(bw as usize) {
                        planes[p * n + i] = le[p];
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
    /// generic streams as exhaustive selection, with one entropy-compression attempt.
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
        if varbytes + 8 < vals.len() * bits as usize / 8 {
            let mut head = Vec::with_capacity(vals.len());
            let mut tail = Vec::new();
            for &v in vals {
                let z = ((v << 1) ^ (v >> 63)) as u64;
                head.push(z.min(255) as u8);
                if z >= 255 {
                    tail.push((z - 255) as i64);
                }
            }
            return Self::ZigZagBytes {
                len,
                head_leaf: pack_leaf(&head),
                tail: Box::new(Self::auto_encode(&tail)),
            };
        }
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
    }

    pub fn len(&self) -> usize {
        match *self {
            Self::Const { len, .. }
            | Self::Rle { len, .. }
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
            Self::Const { val, len } => {
                out.push(0);
                put_uvarint(out, *len as u64);
                put_ivarint(out, *val);
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
        if depth > 4 {
            return Err(Error::Format("IntStream nesting too deep".into()));
        }
        let kind = cur.u8()?;
        let len_u64 = cur.uvar()?;
        if len_u64 > (budget.max_single_alloc_bytes / 8) as u64 {
            return Err(Error::BudgetExceeded("IntStream length"));
        }
        let len = len_u64 as u32;
        match kind {
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
                Ok(bitunpack_u64(&raw, *bits, n, *min))
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
                let mut out = Vec::with_capacity(n);
                let base = *min as u64;
                match w {
                    1 => {
                        for &b0 in &raw[..n] {
                            out.push(base.wrapping_add(b0 as u64) as i64);
                        }
                    }
                    2 => {
                        let (p0, p1) = raw.split_at(n);
                        for i in 0..n {
                            let u = u16::from_le_bytes([p0[i], p1[i]]) as u64;
                            out.push(base.wrapping_add(u) as i64);
                        }
                    }
                    4 => {
                        let (p0, rest) = raw.split_at(n);
                        let (p1, rest) = rest.split_at(n);
                        let (p2, p3) = rest.split_at(n);
                        for i in 0..n {
                            let u = u32::from_le_bytes([p0[i], p1[i], p2[i], p3[i]]) as u64;
                            out.push(base.wrapping_add(u) as i64);
                        }
                    }
                    8 => {
                        let (p0, rest) = raw.split_at(n);
                        let (p1, rest) = rest.split_at(n);
                        let (p2, rest) = rest.split_at(n);
                        let (p3, rest) = rest.split_at(n);
                        let (p4, rest) = rest.split_at(n);
                        let (p5, rest) = rest.split_at(n);
                        let (p6, p7) = rest.split_at(n);
                        for i in 0..n {
                            let u = u64::from_le_bytes([
                                p0[i], p1[i], p2[i], p3[i], p4[i], p5[i], p6[i], p7[i],
                            ]);
                            out.push(base.wrapping_add(u) as i64);
                        }
                    }
                    _ => unreachable!(),
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
                let mut oi = 0usize;
                let base = *min as u64;
                let mut out = Vec::with_capacity(n);
                for &b0 in &h_raw {
                    if b0 < 0xff {
                        out.push(base.wrapping_add(b0 as u64) as i64);
                    } else {
                        if oi >= over.len() || over[oi] < 0 {
                            return Err(Error::Format(
                                "ForTail255 tail underflow or negative".into(),
                            ));
                        }
                        let u = 0xffu64.wrapping_add(over[oi] as u64);
                        oi += 1;
                        out.push(base.wrapping_add(u) as i64);
                    }
                }
                if oi != over.len() {
                    return Err(Error::Format("ForTail255 trailing tail items".into()));
                }
                Ok(out)
            }
        }
    }
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
    DictCells {
        codes: u16,
        dict_ends: IntStream,
        dict_bytes: u16,
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
    ConcatCells {
        parts: Vec<u16>,
    },
    ByteUnshuffle {
        buf: u16,
        width: u8,
    },
    MtfInts {
        codes: u16,
        misses: u16,
        window: u16,
    },
    SliceBuf {
        buf: u16,
        base: u32,
        lens: u16,
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
            | Self::ExtPeer { .. } => {}
            Self::ByteChunks { buf, .. } | Self::ByteUnshuffle { buf, .. } => f(*buf),
            Self::MtfFixed {
                codes,
                fallback_buf,
                ..
            } => {
                f(*codes);
                f(*fallback_buf);
            }
            Self::DictCells {
                codes, dict_bytes, ..
            } => {
                f(*codes);
                f(*dict_bytes);
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
            Self::ConcatCells { parts } => {
                for &p in parts {
                    f(p);
                }
            }
            Self::MtfInts { codes, misses, .. } => {
                f(*codes);
                f(*misses);
            }
            Self::SliceBuf { buf, lens, .. } => {
                f(*buf);
                f(*lens);
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
            Self::DictCells {
                codes,
                dict_ends,
                dict_bytes,
            } => Self::DictCells {
                codes: f(*codes),
                dict_ends: dict_ends.clone(),
                dict_bytes: f(*dict_bytes),
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
            Self::ConcatCells { parts } => Self::ConcatCells {
                parts: parts.iter().map(|&p| f(p)).collect(),
            },
            Self::ByteUnshuffle { buf, width } => Self::ByteUnshuffle {
                buf: f(*buf),
                width: *width,
            },
            Self::MtfInts {
                codes,
                misses,
                window,
            } => Self::MtfInts {
                codes: f(*codes),
                misses: f(*misses),
                window: *window,
            },
            Self::SliceBuf { buf, base, lens } => Self::SliceBuf {
                buf: f(*buf),
                base: *base,
                lens: f(*lens),
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
            Self::ExtPool => Self::ExtPool,
            Self::ExtDict { dict_id } => Self::ExtDict { dict_id: *dict_id },
            Self::ExtPeer { slot } => Self::ExtPeer { slot: *slot },
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
                ins.a = map(ins.a);
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
        let rows_u64 = cur.uvar()?;
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
        let rows = rows_u64 as u32;
        let flags = cur.u8()?;
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

/// Move-to-front codes of `vals` over the last `win` distinct values: `0` takes the next of the
/// returned misses, `k` the value `k - 1` places back.
pub(crate) fn mtf_split(vals: &[i64], win: usize) -> (Vec<i64>, Vec<i64>) {
    let mut ring = Recency::new(win, vals.len());
    let mut codes = Vec::with_capacity(vals.len());
    let mut misses = Vec::new();
    // A value seen for the first time is a miss without a search of the window.
    let mut seen: FastSet<i64> = FastSet::default();
    for &v in vals {
        if !seen.insert(v)
            && let Some(pos) = ring.rank(v)
        {
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

    pub fn add_ext_peer(&mut self, peer_col_id: u32) -> u16 {
        let slot = if let Some(pos) = self.ext_peers.iter().position(|&p| p == peer_col_id) {
            pos as u16
        } else {
            let s = self.ext_peers.len() as u16;
            self.ext_peers.push(peer_col_id);
            s
        };
        self.push(GraphNode::ExtPeer { slot })
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
            leaf: pack_leaf(bytes),
        })
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
        let direct = IntStream::auto_encode(vals);
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
                    let mut deltas = Vec::with_capacity(vals.len());
                    deltas.push(0);
                    for w in vals.windows(2) {
                        deltas.push(w[1].wrapping_sub(w[0]));
                    }
                    let d_stream = IntStream::auto_encode(&deltas);
                    let mut b_del = Vec::new();
                    d_stream.encode(&mut b_del);
                    if b_del.len() + 24 < b_dir.len() {
                        let dn = self.push(GraphNode::IntStream(d_stream));
                        let sb = Builder::new(1);
                        let x = sb.input(0);
                        let s = sb.scan_add(x, vals[0]);
                        return self.add_prog_i64(sb.finish(&[(s, 8)]), vec![dn], vec![]);
                    }
                }
            }
        }
        self.push(GraphNode::IntStream(direct))
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
        let min = *vals.iter().min().unwrap();
        let common = vals
            .iter()
            .fold(0u64, |mask, &v| mask | (v as u64).wrapping_sub(min as u64));
        let shift = common.trailing_zeros();
        if shift >= 2 && shift < 64 && vals.len() * shift as usize > 256 {
            let normalized: Vec<_> = vals
                .iter()
                .map(|&v| ((v as u64).wrapping_sub(min as u64) >> shift) as i64)
                .collect();
            let input = self.add_ints_fast(&normalized);
            let b = Builder::new(1);
            let output = b.ri(
                super::ir::Op::AddI,
                b.ri(super::ir::Op::ShlI, b.input(0), shift as i64),
                min,
            );
            return self.add_prog_i64(b.finish(&[(output, 8)]), vec![input], vec![]);
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
        let span = |v: &[i64]| {
            let lo = *v.iter().min().unwrap();
            let hi = *v.iter().max().unwrap();
            64 - (hi as u64).wrapping_sub(lo as u64).leading_zeros()
        };
        let mut deltas = Vec::with_capacity(vals.len());
        deltas.push(0);
        deltas.extend(vals.windows(2).map(|w| w[1].wrapping_sub(w[0])));
        let delta_bytes = deltas
            .iter()
            .map(|&v| {
                ((64 - (((v << 1) ^ (v >> 63)) as u64).leading_zeros()).max(1) as usize).div_ceil(7)
            })
            .sum::<usize>();
        if (span(vals).saturating_sub(span(&deltas)) as usize) * vals.len() > 256
            || delta_bytes * 8 + 256 < span(vals) as usize * vals.len()
        {
            let input = self.push(GraphNode::IntStream(IntStream::encode_fast(&deltas)));
            let b = Builder::new(1);
            let output = b.scan_add(b.input(0), vals[0]);
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
        let mut table: Vec<Vec<u8>> = Vec::with_capacity(64);
        let mut codes = Vec::with_capacity(count);
        let mut fallback = Vec::new();

        for chunk in flat_items.chunks_exact(w) {
            if let Some(pos) = table.iter().position(|x| x.as_slice() == chunk) {
                codes.push((pos + 1) as i64);
                if pos > 0 {
                    let item = table.remove(pos);
                    table.insert(0, item);
                }
            } else {
                codes.push(0);
                fallback.extend_from_slice(chunk);
                if table.len() == 64 {
                    table.pop();
                }
                table.insert(0, chunk.to_vec());
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

    pub fn add_byte_unshuffle(&mut self, buf: u16, width: u8) -> u16 {
        assert!((1..=64).contains(&width));
        self.push(GraphNode::ByteUnshuffle { buf, width })
    }

    pub fn add_slice_buf(&mut self, buf: u16, base: u32, lens: &[i64]) -> u16 {
        let lens_id = self.add_ints(lens);
        self.push(GraphNode::SliceBuf {
            buf,
            base,
            lens: lens_id,
        })
    }

    pub fn add_index_cells(&mut self, source: u16, indices: u16) -> u16 {
        self.push(GraphNode::IndexCells { source, indices })
    }

    pub fn add_template_cells(&mut self, template: u16, slots: Vec<u16>) -> u16 {
        self.push(GraphNode::TemplateCells { template, slots })
    }

    pub fn add_mtf_ints(&mut self, vals: &[i64], window: u16) -> u16 {
        let win = window as usize;
        if vals.len() < 64 || win == 0 || win > 4096 {
            return self.add_ints(vals);
        }
        let lookback = win.min(512);
        let head = &vals[..vals.len().min(2048)];
        let near_hits = (1..head.len())
            .filter(|&i| head[i.saturating_sub(lookback)..i].contains(&head[i]))
            .count();
        if near_hits * 10 < head.len() {
            return self.add_ints(vals);
        }

        let (codes, misses) = mtf_split(vals, win);
        if misses.len() * 8 >= vals.len() * 7 {
            return self.add_ints(vals);
        }

        let mut tmp_dir = Self::new(vals.len());
        let r_dir = tmp_dir.add_ints(vals);
        let mut tmp_mtf = Self::new(vals.len());
        let c_id = tmp_mtf.add_ints(&codes);
        let m_id = tmp_mtf.add_ints(&misses);
        let r_mtf = tmp_mtf.push(GraphNode::MtfInts {
            codes: c_id,
            misses: m_id,
            window,
        });
        let dir_wire_len = tmp_dir.nodes.iter().fold(0usize, |acc, n| {
            let mut b = Vec::new();
            encode_node(n, &mut b);
            acc + b.len()
        });
        let mtf_wire_len = tmp_mtf.nodes.iter().fold(0usize, |acc, n| {
            let mut b = Vec::new();
            encode_node(n, &mut b);
            acc + b.len()
        });
        if mtf_wire_len + 32 < dir_wire_len {
            self.merge_subgraph(tmp_mtf, r_mtf)
        } else {
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
        for &r in rows {
            if let Some(e) = counts.get_mut(r) {
                e.0 += 1;
            } else {
                if n_uniq >= max_dict_uniq {
                    return emit_raw_slice(self);
                }
                counts.insert(r, (1, n_uniq));
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

        let mut remap = fast_map_with_capacity(sorted.len());
        let mut dict_raw = Vec::new();
        let mut lens = Vec::with_capacity(sorted.len());
        for (new_id, (s, _, _)) in sorted.into_iter().enumerate() {
            remap.insert(s, new_id as i64);
            lens.push(s.len() as i64);
            dict_raw.extend_from_slice(s);
        }
        let codes: Vec<i64> = rows.iter().map(|r| remap[r]).collect();

        let codes_id = if lens.len() >= 16 && rows.len() >= 64 {
            self.add_mtf_ints(&codes, 4096)
        } else {
            self.add_ints(&codes)
        };
        let bytes_id = self.add_bytes(&dict_raw);
        self.push(GraphNode::DictCells {
            codes: codes_id,
            dict_ends: IntStream::auto_encode(&lens),
            dict_bytes: bytes_id,
        })
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

    pub fn add_concat(&mut self, parts: Vec<u16>) -> u16 {
        self.push(GraphNode::ConcatCells { parts })
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
        GraphNode::DictCells {
            codes,
            dict_ends,
            dict_bytes,
        } => {
            out.push(4);
            put_uvarint(out, *codes as u64);
            dict_ends.encode(out);
            put_uvarint(out, *dict_bytes as u64);
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
        GraphNode::ConcatCells { parts } => {
            out.push(9);
            put_u16_list(out, parts);
        }
        GraphNode::ByteUnshuffle { buf, width } => {
            out.push(10);
            put_uvarint(out, *buf as u64);
            out.push(*width);
        }
        GraphNode::MtfInts {
            codes,
            misses,
            window,
        } => {
            out.push(11);
            put_uvarint(out, *codes as u64);
            put_uvarint(out, *misses as u64);
            put_uvarint(out, *window as u64);
        }
        GraphNode::SliceBuf { buf, base, lens } => {
            out.push(12);
            put_uvarint(out, *buf as u64);
            put_uvarint(out, *base as u64);
            put_uvarint(out, *lens as u64);
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
        4 => {
            let codes = check_ref(cur.uvar()?)?;
            let dict_ends = IntStream::decode_wire(cur, budget, 0)?;
            let dict_bytes = check_ref(cur.uvar()?)?;
            Ok(GraphNode::DictCells {
                codes,
                dict_ends,
                dict_bytes,
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
        9 => {
            let parts = get_u16_list(cur, idx)?;
            if parts.is_empty() || parts.len() > 32 {
                return Err(Error::Format("invalid ConcatCells part count".into()));
            }
            Ok(GraphNode::ConcatCells { parts })
        }
        10 => {
            let buf = check_ref(cur.uvar()?)?;
            let width = cur.u8()?;
            if !(1..=64).contains(&width) {
                return Err(Error::Format("ByteUnshuffle width out of range".into()));
            }
            Ok(GraphNode::ByteUnshuffle { buf, width })
        }
        11 => {
            let codes = check_ref(cur.uvar()?)?;
            let misses = check_ref(cur.uvar()?)?;
            let win = cur.uvar()?;
            if !(1..=4096).contains(&win) {
                return Err(Error::Format("MtfInts window out of range".into()));
            }
            Ok(GraphNode::MtfInts {
                codes,
                misses,
                window: win as u16,
            })
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
    pub pool: Option<&'a Arc<Vec<u8>>>,
    pub dict: Option<(u16, &'a Arc<Cells>)>,
    pub peers: &'a [Arc<Cells>],
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
    SharedBuf(Arc<Vec<u8>>),
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
        let graph = PayloadGraph::decode(bytes, budget)?;
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
                    | GraphNode::DictCells { .. }
                    | GraphNode::IndexCells { .. }
                    | GraphNode::SliceBuf { .. }
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

        let mut vals = Self::execute_nodes(&graph.nodes, rows, ext, budget, &skip, &mut rem_uses)?;
        let col = Self::build_col(root, &graph.nodes, &skip, &mut vals, budget)?;
        if col.len() != rows {
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
            GraphNode::DictCells {
                codes,
                dict_ends,
                dict_bytes,
            } => {
                if *codes == *dict_bytes {
                    return Err(Error::Format("DictCells codes == dict_bytes".into()));
                }
                let lens_i64 = dict_ends.unpack(budget)?;
                let raw = take_or_clone_buf(&mut vals[*dict_bytes as usize])?;
                let codes_col = as_ints(&vals[*codes as usize])?;
                let mut ends = Vec::with_capacity(lens_i64.len());
                let mut acc = 0usize;
                for &l in &lens_i64 {
                    if l < 0 {
                        return Err(Error::Format("negative dict entry length".into()));
                    }
                    acc = acc
                        .checked_add(l as usize)
                        .ok_or_else(|| Error::Format("dict length overflow".into()))?;
                    ends.push(acc as u32);
                }
                if acc != raw.len() {
                    return Err(Error::Format(
                        "dict entry lengths do not cover dict_bytes".into(),
                    ));
                }
                let dict = Arc::new(Cells::from_parts(raw, ends)?);
                let dict_len = dict.len();
                let mut ords = Vec::with_capacity(codes_col.len());
                for &c in codes_col {
                    if c < 0 || (c as usize) >= dict_len {
                        return Err(Error::Format("dict code out of bounds".into()));
                    }
                    ords.push(c as u32);
                }
                Ok(Col::Dict { dict, ords })
            }
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
                let raw_len = as_table_bytes(&vals[*buf as usize])?.len();
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
    ) -> Result<Vec<Val>> {
        let mut vals: Vec<Val> = Vec::with_capacity(nodes.len());

        for (idx, node) in nodes.iter().enumerate() {
            if skip[idx] {
                vals.push(Val::Empty);
                continue;
            }
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
                GraphNode::DictCells {
                    codes,
                    dict_ends,
                    dict_bytes,
                } => {
                    let codes_col = as_ints(&vals[*codes as usize])?;
                    let lens = dict_ends.unpack(budget)?;
                    let raw = as_buf(&vals[*dict_bytes as usize])?;
                    let mut offsets = Vec::with_capacity(lens.len() + 1);
                    offsets.push(0usize);
                    let mut acc = 0usize;
                    for &l in &lens {
                        if l < 0 {
                            return Err(Error::Format("negative dict entry length".into()));
                        }
                        acc = acc
                            .checked_add(l as usize)
                            .ok_or_else(|| Error::Format("dict overflow".into()))?;
                        if acc > raw.len() {
                            return Err(Error::Format("dict entry exceeds buffer".into()));
                        }
                        offsets.push(acc);
                    }
                    let mut data = Vec::with_capacity(codes_col.len() * 8);
                    let mut ends = Vec::with_capacity(codes_col.len());
                    for &c in codes_col {
                        if c < 0 || (c as usize) >= lens.len() {
                            return Err(Error::Format("dict code out of bounds".into()));
                        }
                        let k = c as usize;
                        let sl = &raw[offsets[k]..offsets[k + 1]];
                        if data.len().saturating_add(sl.len()) > budget.max_single_alloc_bytes {
                            return Err(Error::BudgetExceeded("DictCells expanded bytes"));
                        }
                        data.extend_from_slice(sl);
                        ends.push(data.len() as u32);
                    }
                    Val::Cells(Cells::from_parts(data, ends)?)
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
                            if seed.len() < spec.max_lag() {
                                return Err(Error::Format("seed shorter than max_lag".into()));
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
                            if seed.len() < spec.max_lag() {
                                return Err(Error::Format("seed shorter than max_lag".into()));
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
                    let mut n_rows = rows;
                    for &iid in inputs {
                        let col = as_ints(&vals[iid as usize])?;
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
                        || !super::engine::eval_numeric_chain(prog, in_refs[0], &mut out)
                    {
                        CompiledProgram::with(prog, |cp| cp.eval_i64(&in_refs, &tbufs, &mut out));
                    }
                    Val::Ints(out)
                }
                GraphNode::ProgramCells { prog, inputs, bufs } => {
                    let mut in_refs = Vec::with_capacity(inputs.len());
                    let mut n_rows = rows;
                    for &iid in inputs {
                        let col = as_ints(&vals[iid as usize])?;
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
                    let rb = prog.row_bytes();
                    let cap = n_rows
                        .checked_mul(rb)
                        .ok_or(Error::BudgetExceeded("ProgramCells"))?;
                    budget.check_alloc(cap + 8, "ProgramCells")?;
                    if !prog.varlen() {
                        let mut data = vec![0u8; cap];
                        CompiledProgram::with(prog, |cp| {
                            cp.eval_fixed(&in_refs, &tbufs, n_rows, &mut data)
                        });
                        Val::Cells(Cells::from_fixed(n_rows, rb, data)?)
                    } else {
                        let mut ends = vec![0u32; n_rows];
                        let mut data = vec![0u8; cap + 8];
                        let written = CompiledProgram::with(prog, |cp| {
                            cp.eval_cells(&in_refs, &tbufs, n_rows, &mut ends, &mut data)
                        });
                        data.truncate(written);
                        Val::Cells(Cells::from_parts(data, ends)?)
                    }
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
                GraphNode::ConcatCells { parts } => {
                    let cols: Vec<&Cells> = parts
                        .iter()
                        .map(|&pid| as_cells(&vals[pid as usize]))
                        .collect::<Result<_>>()?;
                    let n = cols[0].len();
                    if cols.iter().any(|c| c.len() != n) {
                        return Err(Error::Format("ConcatCells row count mismatch".into()));
                    }
                    let total_bytes: usize = cols.iter().map(|c| c.bytes().len()).sum();
                    budget.check_alloc(total_bytes, "ConcatCells")?;
                    let mut data = Vec::with_capacity(total_bytes);
                    let mut ends = Vec::with_capacity(n);
                    for r in 0..n {
                        for c in &cols {
                            data.extend_from_slice(c.get(r).unwrap());
                        }
                        ends.push(data.len() as u32);
                    }
                    Val::Cells(Cells::from_parts(data, ends)?)
                }
                GraphNode::ByteUnshuffle { buf, width } => {
                    let bytes = as_buf(&vals[*buf as usize])?;
                    let w = *width as usize;
                    if !bytes.len().is_multiple_of(w) {
                        return Err(Error::Format(
                            "ByteUnshuffle buffer not aligned to width".into(),
                        ));
                    }
                    budget.check_alloc(bytes.len(), "ByteUnshuffle")?;
                    let n = bytes.len() / w;
                    let mut out = vec![0u8; bytes.len()];
                    match w {
                        1 => out.copy_from_slice(bytes),
                        2 => {
                            let (p0, p1) = bytes.split_at(n);
                            for (i, ch) in out.chunks_exact_mut(2).enumerate() {
                                ch[0] = p0[i];
                                ch[1] = p1[i];
                            }
                        }
                        4 => {
                            let p0 = &bytes[0..n];
                            let p1 = &bytes[n..2 * n];
                            let p2 = &bytes[2 * n..3 * n];
                            let p3 = &bytes[3 * n..4 * n];
                            for (i, ch) in out.chunks_exact_mut(4).enumerate() {
                                ch[0] = p0[i];
                                ch[1] = p1[i];
                                ch[2] = p2[i];
                                ch[3] = p3[i];
                            }
                        }
                        8 => {
                            let p0 = &bytes[0..n];
                            let p1 = &bytes[n..2 * n];
                            let p2 = &bytes[2 * n..3 * n];
                            let p3 = &bytes[3 * n..4 * n];
                            let p4 = &bytes[4 * n..5 * n];
                            let p5 = &bytes[5 * n..6 * n];
                            let p6 = &bytes[6 * n..7 * n];
                            let p7 = &bytes[7 * n..8 * n];
                            for (i, ch) in out.chunks_exact_mut(8).enumerate() {
                                ch[0] = p0[i];
                                ch[1] = p1[i];
                                ch[2] = p2[i];
                                ch[3] = p3[i];
                                ch[4] = p4[i];
                                ch[5] = p5[i];
                                ch[6] = p6[i];
                                ch[7] = p7[i];
                            }
                        }
                        _ => {
                            for (i, ch) in out.chunks_exact_mut(w).enumerate() {
                                for (k, slot) in ch.iter_mut().enumerate() {
                                    *slot = bytes[k * n + i];
                                }
                            }
                        }
                    }
                    Val::Buf(out)
                }
                GraphNode::MtfInts {
                    codes,
                    misses,
                    window,
                } => {
                    let codes_col = as_ints(&vals[*codes as usize])?;
                    let misses_col = as_ints(&vals[*misses as usize])?;
                    let win = *window as usize;
                    budget.check_alloc(codes_col.len().saturating_mul(8), "MtfInts")?;
                    let mut ring = Recency::new(win, codes_col.len());
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
                    if misses_it.next().is_some() {
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
                    let tmpl = as_cells(&vals[*template as usize])?;
                    let slot_cols: Vec<&Cells> = slots
                        .iter()
                        .map(|&sid| as_cells(&vals[sid as usize]))
                        .collect::<Result<_>>()?;
                    let n = tmpl.len();
                    if slot_cols.iter().any(|c| c.len() != n) {
                        return Err(Error::Format("TemplateCells row count mismatch".into()));
                    }
                    let mut data = Vec::with_capacity(tmpl.bytes().len() * 2);
                    let mut ends = Vec::with_capacity(n);
                    for r in 0..n {
                        let tb = tmpl.get(r).unwrap();
                        let mut p = 0usize;
                        while p < tb.len() {
                            if let Some(hit) = tb[p..].iter().position(|&b| b == 0x1f) {
                                data.extend_from_slice(&tb[p..p + hit]);
                                let esc = p + hit + 1;
                                if esc >= tb.len() {
                                    return Err(Error::Format(
                                        "truncated TemplateCells escape".into(),
                                    ));
                                }
                                let code = tb[esc];
                                if code == 0xff {
                                    data.push(0x1f);
                                } else {
                                    let slot_i = (code >> 1) as usize;
                                    let strip = (code & 1) != 0;
                                    if slot_i >= slot_cols.len() {
                                        return Err(Error::Format(
                                            "TemplateCells slot out of bounds".into(),
                                        ));
                                    }
                                    let sv = slot_cols[slot_i].get(r).unwrap();
                                    let ins = if strip && sv.len() >= 2 {
                                        &sv[1..sv.len() - 1]
                                    } else {
                                        sv
                                    };
                                    data.extend_from_slice(ins);
                                }
                                p = esc + 1;
                            } else {
                                data.extend_from_slice(&tb[p..]);
                                break;
                            }
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

#[inline]
fn parse_cell_ascii_i64(mut s: &[u8]) -> i64 {
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
        Val::SharedBuf(b) => Ok(b),
        _ => Err(Error::Format("expected Buf node".into())),
    }
}

fn take_or_clone_buf(v: &mut Val) -> Result<Vec<u8>> {
    match std::mem::replace(v, Val::Empty) {
        Val::Buf(b) => Ok(b),
        Val::SharedBuf(b) => Ok((*b).clone()),
        _ => Err(Error::Format("expected Buf node".into())),
    }
}

fn as_table_bytes(v: &Val) -> Result<&[u8]> {
    match v {
        Val::Buf(b) => Ok(b),
        Val::SharedBuf(b) => Ok(b),
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
    let b = bits as usize;
    let mut out = vec![0u8; (len * b).div_ceil(8)];
    let mut bit_pos = 0usize;
    let mask = if b == 64 { u64::MAX } else { (1u64 << b) - 1 };
    for v in vals {
        let x = v & mask;
        let byte_idx = bit_pos >> 3;
        let shift = bit_pos & 7;
        let lo = x << shift;
        let n_lo = (out.len() - byte_idx).min(8);
        let mut cur = [0u8; 8];
        cur[..n_lo].copy_from_slice(&out[byte_idx..byte_idx + n_lo]);
        let merged = u64::from_le_bytes(cur) | lo;
        out[byte_idx..byte_idx + n_lo].copy_from_slice(&merged.to_le_bytes()[..n_lo]);
        if shift + b > 64 {
            out[byte_idx + 8] |= (x >> (64 - shift)) as u8;
        }
        bit_pos += b;
    }
    out
}

fn bitunpack_u64(bytes: &[u8], bits: u8, len: usize, min: i64) -> Vec<i64> {
    if bits == 0 {
        return vec![min; len];
    }
    let b = bits as usize;
    let mask = if b == 64 { u64::MAX } else { (1u64 << b) - 1 };
    let mut out = Vec::with_capacity(len);
    let mut bit_pos = 0usize;
    for _ in 0..len {
        let byte_idx = bit_pos >> 3;
        let shift = bit_pos & 7;
        let word = if byte_idx + 8 <= bytes.len() {
            u64::from_le_bytes(bytes[byte_idx..byte_idx + 8].try_into().unwrap())
        } else {
            let mut tail = [0u8; 8];
            let rest = &bytes[byte_idx..];
            tail[..rest.len()].copy_from_slice(rest);
            u64::from_le_bytes(tail)
        };
        let mut x = word >> shift;
        if shift + b > 64 && byte_idx + 8 < bytes.len() {
            x |= (bytes[byte_idx + 8] as u64) << (64 - shift);
        }
        out.push((min as u64).wrapping_add(x & mask) as i64);
        bit_pos += b;
    }
    out
}
