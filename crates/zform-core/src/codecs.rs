//! Thin block-column summary, dependency inspection, and lazy `CoreDecoder` execution glue.
//!
//! Columns carry self-describing [`PayloadGraph`] programs emitted by the writer's
//! [`PackRegistry`] and executed by [`CoreDecoder`]. This module records their dependencies
//! and loads shared block parts on demand.

use std::sync::Arc;

use crate::cells::{Cells, Col};
use crate::lex::{parse_exact_int, parse_timestamp};
pub use crate::packs::EncodedDenseBlock;
use crate::packs::PackRegistry;
pub(crate) use crate::packs::{Seeds, with_seeds};
use crate::payload::graph::{
    CoreDecoder, EXT_FLAG_DICT, EXT_FLAG_PEERS, EXT_FLAG_POOL, ExtInputs, IntStream,
    NESTED_COLUMN_TAG, PayloadGraph, unpack_leaf, with_encode_opts,
};
use crate::payload::library::Library;
use crate::wire::{Cursor, DecodeBudget, EntropyBackend, Error, Result, put_bytes};

// ============================================================================
// 1. Per-Column Block Summary & Index Metadata
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
pub(crate) fn time_bounds(vals: &[&[u8]]) -> Option<(i64, i64)> {
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

/// Bounds of a column whose every value is an exact integer.
pub(crate) fn int_bounds(vals: &[&[u8]]) -> Option<(i64, i64)> {
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
// 2. Structural `u32` Sequence & Standalone Slice Helpers
// ============================================================================

/// Encodes a slice of `u32` ordinals/tags via [`IntStream`].
pub fn encode_u32_seq(vals: &[u32], backend: EntropyBackend, level: i32, out: &mut Vec<u8>) {
    let ints: Vec<i64> = vals.iter().map(|&v| v as i64).collect();
    with_encode_opts(backend, level, || {
        IntStream::auto_encode(&ints).encode(out);
    });
}

/// Decodes a self-delimited `u32` sequence written by [`encode_u32_seq`].
pub fn decode_u32_seq(cur: &mut Cursor<'_>, budget: &DecodeBudget) -> Result<Vec<u32>> {
    let stream = IntStream::decode_wire(cur, budget, 0)?;
    let ints = stream.unpack(budget)?;
    let mut out = Vec::with_capacity(ints.len());
    for v in ints {
        let u = u32::try_from(v).map_err(|_| Error::Format("u32_seq value out of range".into()))?;
        out.push(u);
    }
    Ok(out)
}

/// Standalone column-slice encoder used by model training (`plan.rs`) and nested-plan fallbacks.
pub(crate) fn encode_present_slice(
    vals: &[&[u8]],
    backend: EntropyBackend,
    level: i32,
    _depth: u8,
) -> (u8, i64, i64, Vec<u8>) {
    let (nk, mn, mx) = if let Some((lo, hi)) = int_bounds(vals) {
        (1, lo, hi)
    } else if let Some((lo, hi)) = time_bounds(vals) {
        (2, lo, hi)
    } else {
        (0, 0, 0)
    };
    let wire = with_encode_opts(backend, level, || PackRegistry::default().encode_best(vals));
    let mut framed = Vec::with_capacity(wire.len() + 5);
    put_bytes(&mut framed, &wire);
    (nk, mn, mx, framed)
}

/// Decodes a standalone slice written by [`encode_present_slice`] via [`CoreDecoder`].
pub(crate) fn decode_present_slice(
    cur: &mut Cursor<'_>,
    expected_rows: usize,
    budget: &DecodeBudget,
) -> Result<Col> {
    let wire = cur.bytes()?;
    let graph = PayloadGraph::decode(wire, budget)?;
    let col = CoreDecoder::decode_col(&graph, &ExtInputs::empty(), budget)?;
    if col.len() != expected_rows {
        return Err(Error::Format(
            "decode_present_slice row count mismatch".into(),
        ));
    }
    Ok(col)
}

// ============================================================================
// 3. Block-Level Column Dependencies & Lazy `CoreDecoder` Context
// ============================================================================

/// External block-part dependencies declared in a column's [`PayloadGraph`] header.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColDeps {
    pub pool: bool,
    pub dict: Option<u32>,
    pub cols: Vec<u32>,
}

/// Reads the external dependencies out of a column blob's [`PayloadGraph`] header without
/// decoding its nodes.
pub fn column_deps(blob: &[u8]) -> ColDeps {
    let mut d = ColDeps::default();
    let mut cur = Cursor::new(blob);
    if cur.uvar().unwrap_or(0) == 0 {
        return d;
    }
    let Ok(flags) = cur.u8() else {
        return d;
    };
    if flags == NESTED_COLUMN_TAG || (flags & !0x07) != 0 {
        return d;
    }
    if (flags & EXT_FLAG_POOL) != 0 {
        d.pool = true;
    }
    if (flags & EXT_FLAG_DICT) != 0 {
        d.dict = cur.uvar().ok().map(|i| i as u32);
    }
    if (flags & EXT_FLAG_PEERS) != 0
        && let Ok(n) = cur.uvar()
    {
        for _ in 0..n.min(64) {
            if let Ok(c) = cur.uvar() {
                d.cols.push(c as u32);
            }
        }
    }
    d
}

/// Encodes all dense occurrence columns of a block using `packs`.
#[allow(clippy::too_many_arguments)]
pub fn encode_dense_block_columns(
    occ_streams: &[Vec<&[u8]>],
    dict_groups: &[&str],
    horiz_cands: &[Vec<usize>],
    preencoded_nested: &mut [Option<(ColSummary, Vec<u8>)>],
    standalone: &[bool],
    packs: &PackRegistry,
    backend: EntropyBackend,
    level: i32,
) -> EncodedDenseBlock {
    packs.encode_dense_block(
        occ_streams,
        dict_groups,
        horiz_cands,
        preencoded_nested,
        standalone,
        backend,
        level,
    )
}

/// Lazily materialized block-level shared dictionaries, prose pool, and peer columns.
pub struct BlockDecodeCtx<'a> {
    pub col_blobs: Vec<Option<&'a [u8]>>,
    dict_blobs: Vec<Option<&'a [u8]>>,
    dict_cache: Vec<Option<Arc<Cells>>>,
    pool_leaf: Option<&'a [u8]>,
    pool_cache: Option<Arc<Vec<u8>>>,
    pub cache: Vec<Option<Col>>,
    cells_cache: Vec<Option<Arc<Cells>>>,
    /// The model's library, which the block's code references name.
    library: Arc<Library>,
}

impl<'a> BlockDecodeCtx<'a> {
    pub fn new(
        col_blobs: Vec<Option<&'a [u8]>>,
        dict_blobs: Vec<Option<&'a [u8]>>,
        pool_leaf: Option<&'a [u8]>,
        library: Arc<Library>,
    ) -> Self {
        let n_cols = col_blobs.len();
        let n_dicts = dict_blobs.len();
        Self {
            col_blobs,
            dict_blobs,
            dict_cache: (0..n_dicts).map(|_| None).collect(),
            pool_leaf,
            pool_cache: None,
            cache: (0..n_cols).map(|_| None).collect(),
            cells_cache: (0..n_cols).map(|_| None).collect(),
            library,
        }
    }

    fn ensure_pool(&mut self, budget: &DecodeBudget) -> Result<&Arc<Vec<u8>>> {
        if self.pool_cache.is_none() {
            let leaf = self.pool_leaf.ok_or(Error::NotLoaded)?;
            let raw = unpack_leaf(leaf, budget)?;
            self.pool_cache = Some(Arc::new(raw));
        }
        Ok(self.pool_cache.as_ref().unwrap())
    }

    fn ensure_dict(&mut self, dict_id: usize, budget: &DecodeBudget) -> Result<&Arc<Cells>> {
        let slot = self
            .dict_blobs
            .get(dict_id)
            .ok_or_else(|| Error::Format("shared dict id out of bounds".into()))?;
        if self.dict_cache[dict_id].is_none() {
            let wire = (*slot).ok_or(Error::NotLoaded)?;
            let graph = PayloadGraph::decode_linked(wire, &self.library, budget)?;
            let cells = CoreDecoder::decode_cells(&graph, &ExtInputs::empty(), budget)?;
            self.dict_cache[dict_id] = Some(Arc::new(cells));
        }
        Ok(self.dict_cache[dict_id].as_ref().unwrap())
    }
}

/// Decodes column `col_idx` into `ctx.cache` via [`CoreDecoder`], resolving any declared block
/// pool, shared dictionary, or earlier peer columns on demand.
pub fn ensure_dense_block_column<'c>(
    col_idx: usize,
    ctx: &'c mut BlockDecodeCtx<'_>,
    budget: &DecodeBudget,
) -> Result<&'c Col> {
    ensure_col_inner(col_idx, ctx, budget, 0)?;
    Ok(ctx.cache[col_idx].as_ref().unwrap())
}

fn ensure_col_inner(
    col_idx: usize,
    ctx: &mut BlockDecodeCtx<'_>,
    budget: &DecodeBudget,
    depth: usize,
) -> Result<()> {
    if col_idx >= ctx.col_blobs.len() {
        return Err(Error::Format("column index out of bounds".into()));
    }
    if ctx.cache[col_idx].is_some() {
        return Ok(());
    }
    if depth > 16 {
        return Err(Error::Format("excessive peer dependency depth".into()));
    }
    let blob = ctx.col_blobs[col_idx].ok_or(Error::NotLoaded)?;
    let graph = PayloadGraph::decode_linked(blob, &ctx.library, budget)?;
    if graph.rows as usize > budget.max_records_per_block.saturating_mul(16) {
        return Err(Error::BudgetExceeded("dense column occurrence count"));
    }

    let mut peers = Vec::with_capacity(graph.ext_peers.len());
    for &peer_id in &graph.ext_peers {
        let pid = peer_id as usize;
        if pid >= col_idx {
            return Err(Error::Format(
                "horizontal forward or self column reference".into(),
            ));
        }
        ensure_col_inner(pid, ctx, budget, depth + 1)?;
        if ctx.cells_cache[pid].is_none() {
            let cells = ctx.cache[pid].as_ref().unwrap().to_cells()?;
            ctx.cells_cache[pid] = Some(Arc::new(cells));
        }
        peers.push(Arc::clone(ctx.cells_cache[pid].as_ref().unwrap()));
    }

    let pool_arc = if graph.ext_pool {
        Some(Arc::clone(ctx.ensure_pool(budget)?))
    } else {
        None
    };
    let dict_arc = if let Some(did) = graph.ext_dict {
        Some((did, Arc::clone(ctx.ensure_dict(did as usize, budget)?)))
    } else {
        None
    };

    let ext = ExtInputs {
        pool: pool_arc.as_ref(),
        dict: dict_arc.as_ref().map(|(d, a)| (*d, a)),
        peers: &peers,
    };
    let col = CoreDecoder::decode_col(&graph, &ext, budget)?;
    ctx.cache[col_idx] = Some(col);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{PackOptions, pack_stream, unpack_stream};
    use crate::packs::{CodecPack, ColumnProfile};
    use crate::payload::graph::GraphBuilder;
    use crate::payload::ir::{Builder, Op};

    #[test]
    fn u32_seq_roundtrips_and_rejects_truncation() {
        let mut vals = Vec::new();
        for i in 0..400u32 {
            vals.push(if i % 37 == 0 { 10_000 + i } else { i % 7 });
        }
        let mut enc = Vec::new();
        encode_u32_seq(&vals, EntropyBackend::Zstd, 3, &mut enc);
        let budget = DecodeBudget::default();
        assert_eq!(
            decode_u32_seq(&mut Cursor::new(&enc), &budget).unwrap(),
            vals
        );
        for end in 0..enc.len() {
            assert!(decode_u32_seq(&mut Cursor::new(&enc[..end]), &budget).is_err());
        }
    }

    #[test]
    fn custom_runtime_pack_decodes_via_bare_core_decoder() {
        /// Custom writer-only pack for `"ORD-YYYY-NNNNNN"` order IDs: zero changes to `CoreDecoder`!
        struct OrderIdPack;
        impl CodecPack for OrderIdPack {
            fn name(&self) -> &'static str {
                "custom/order-id"
            }
            fn precheck(&self, p: &ColumnProfile) -> bool {
                p.min_len == 15 && p.max_len == 15
            }
            fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
                if rows
                    .iter()
                    .any(|r| r.len() != 15 || !r.starts_with(b"ORD-2026-"))
                {
                    return None;
                }
                let mut nums = Vec::with_capacity(rows.len());
                for &r in rows {
                    let s = core::str::from_utf8(&r[9..15]).ok()?;
                    nums.push(s.parse::<i64>().ok()?);
                }
                let mut deltas = Vec::with_capacity(nums.len());
                deltas.push(0);
                for w in nums.windows(2) {
                    deltas.push(w[1] - w[0]);
                }
                let mut gb = GraphBuilder::new(rows.len());
                let pfx = gb.add_const_cells(b"ORD-2026-", rows.len());
                let b = Builder::new(1);
                let d = b.input(0);
                let v = b.scan_add(d, nums[0]);
                let scaled = b.ri(Op::MulI, v, 100);
                let (_, asc) = crate::packs::eight_digits_raw_and_ascii(&b, scaled);
                let dn = gb.add_ints(&deltas);
                let seq_cells = gb.add_prog_cells(b.finish(&[(asc, 6)]), vec![dn], vec![]);
                let cat = gb.add_concat(vec![pfx, seq_cells]);
                Some(gb.finish(cat))
            }
        }

        let mut raw = String::new();
        for i in 0..1000 {
            raw.push_str(&format!("{{\"order\":\"ORD-2026-{:06}\"}}\n", 10_000 + i));
        }
        let mut reg = PackRegistry::new();
        reg.register(OrderIdPack);
        let opts = PackOptions {
            index: false,
            packs: reg,
            ..Default::default()
        };
        let mut packed = Vec::new();
        pack_stream(raw.as_bytes(), &mut packed, &opts).unwrap();
        assert!(packed.len() < 400, "custom pack size={}", packed.len());

        let mut unpacked = Vec::new();
        unpack_stream(packed.as_slice(), &mut unpacked, &DecodeBudget::default()).unwrap();
        assert_eq!(unpacked, raw.as_bytes());
    }

    #[test]
    fn packs_reach_mt19937_generators_through_columns() {
        let pack = |raw: &[u8]| {
            let opts = PackOptions {
                index: false,
                ..Default::default()
            };
            let mut out = Vec::new();
            pack_stream(raw, &mut out, &opts).unwrap();
            let mut back = Vec::new();
            unpack_stream(out.as_slice(), &mut back, &DecodeBudget::default()).unwrap();
            assert_eq!(back, raw);
            out.len()
        };

        // Helper generating Python `random.Random(seed)` 32-bit words.
        let py_words = |seed: u32, count: usize| -> Vec<u32> {
            let mut s = [0u32; 624];
            s[0] = 19_650_218;
            for i in 1..624 {
                s[i] = (s[i - 1] ^ (s[i - 1] >> 30))
                    .wrapping_mul(1_812_433_253)
                    .wrapping_add(i as u32);
            }
            let mut idx = 1usize;
            for _ in 0..624 {
                s[idx] = (s[idx] ^ (s[idx - 1] ^ (s[idx - 1] >> 30)).wrapping_mul(1_664_525))
                    .wrapping_add(seed);
                idx += 1;
                if idx == 624 {
                    s[0] = s[623];
                    idx = 1;
                }
            }
            for _ in 1..624 {
                s[idx] = (s[idx] ^ (s[idx - 1] ^ (s[idx - 1] >> 30)).wrapping_mul(1_566_083_941))
                    .wrapping_sub(idx as u32);
                idx += 1;
                if idx == 624 {
                    s[0] = s[623];
                    idx = 1;
                }
            }
            s[0] = 1 << 31;
            let twist = |hi: u32, lo: u32, far: u32| {
                let y = (hi & (1 << 31)) | (lo & !(1 << 31));
                far ^ (y >> 1) ^ (0x9908_b0dfu32.wrapping_mul(y & 1))
            };
            let mut out = Vec::with_capacity(count + 624);
            while out.len() < count {
                for i in 0..(624 - 397) {
                    s[i] = twist(s[i], s[i + 1], s[i + 397]);
                }
                for i in (624 - 397)..623 {
                    s[i] = twist(s[i], s[i + 1], s[i + 397 - 624]);
                }
                s[623] = twist(s[623], s[0], s[396]);
                for &x in &s {
                    let mut y = x;
                    y ^= y >> 11;
                    y ^= (y << 7) & 0x9d2c_5680;
                    y ^= (y << 15) & 0xefc6_0000;
                    y ^= y >> 18;
                    out.push(y);
                }
            }
            out.truncate(count);
            out
        };

        // 1. Base64 lines of 48 bytes drawn back-to-back from `Random(123_456)` (self-seeded head).
        let w = py_words(123_456, 24_000);
        let mut raw_b = Vec::with_capacity(2000 * 48);
        for &word in &w[..24_000] {
            raw_b.extend_from_slice(&word.to_le_bytes());
        }
        let mut text = Vec::new();
        for line in raw_b.chunks_exact(48) {
            crate::lex::format_base64(line, &mut text);
            text.push(b'\n');
        }
        let size = pack(&text);
        assert!(size < 4 * 624 + 500, "base64 mt size={size}");

        // 2. JSON whose `traceparent` traces and spans come from `Random(42)`.
        let w = py_words(42, 30_000);
        let hex_be16 = |p: usize| -> String {
            let mut out = String::with_capacity(32);
            for k in (0..4).rev() {
                for b in w[p + k].to_be_bytes() {
                    out.push_str(&format!("{b:02x}"));
                }
            }
            out
        };
        let hex_le8 = |p: usize| -> String {
            let mut out = String::with_capacity(16);
            for k in 0..2 {
                for b in w[p + k].to_le_bytes() {
                    out.push_str(&format!("{b:02x}"));
                }
            }
            out
        };
        let mut json = String::new();
        for i in 0..2000 {
            let p = 200 + 13 * i;
            let trace = hex_be16(40 + 4 * (w[p] as usize % 16));
            let span = hex_le8(p + 1);
            json += &format!("{{\"seq\":{i},\"traceparent\":\"00-{trace}-{span}-01\"}}\n");
        }
        let size = pack(json.as_bytes());
        assert!(size < 3000, "traceparent mt size={size}");

        // 3. 24-byte binary records after a header: two doubles walking by `Random(7)` draws.
        let w = py_words(7, 24_002);
        let walk = |a: f64, b: f64, words: &[u32], prev: [u8; 8]| -> [u8; 8] {
            let r = u64::from(words[0] >> 5) << 26 | u64::from(words[1] >> 6);
            let step = a + (b - a) * (r as f64 / (1u64 << 53) as f64);
            (f64::from_le_bytes(prev) + step).to_le_bytes()
        };
        let mut recs = b"A_HEADER_OF_24_BYTES____".to_vec();
        let (mut ra, mut dec) = (0.01f64.to_le_bytes(), (-1.2f64).to_le_bytes());
        for i in 0..3000 {
            ra = walk(0.0, 0.00003, &w[8 * i..], ra);
            dec = walk(-0.0001, 0.00012, &w[8 * i + 2..], dec);
            recs.extend(ra.iter().chain(&dec));
            recs.extend([b'G', b'0' + (i % 8) as u8]);
            recs.extend((400 + (i % 650) as u16).to_le_bytes());
            recs.extend((f32::from((w[8 * i + 4] >> 26) as u8) * 1e-8).to_le_bytes());
        }
        let size = pack(&recs);
        assert!(size < 4000, "walk mt size={size}");
    }
}
