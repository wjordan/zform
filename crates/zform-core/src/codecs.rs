//! Thin block-column summary, dependency inspection, and lazy `CoreDecoder` execution glue.
//!
//! Columns carry self-describing [`PayloadGraph`] programs emitted by the writer's
//! [`PackRegistry`] and executed by [`CoreDecoder`]. This module records their dependencies
//! and loads shared block parts on demand.

use std::sync::Arc;

use crate::cells::{Cells, Col, Pool};
use crate::grammar::{ModelPlan, PlanVariant};
use crate::lex::parse_timestamp;
pub use crate::packs::EncodedDenseBlock;
use crate::packs::PackRegistry;
pub(crate) use crate::packs::{Seeds, with_seeds};
use crate::payload::graph::{
    CoreDecoder, EXT_FLAG_DICT, EXT_FLAG_PEERS, EXT_FLAG_POOL, ExtInputs, GraphNode, IntStream,
    NESTED_COLUMN_TAG, PayloadGraph, unpack_leaf, with_encode_opts,
};
use crate::payload::library::Library;
use crate::dictionary::with_current;
use crate::tables::TABLE_PEER;
use crate::wire::{Cursor, DecodeBudget, EntropyBackend, Error, Result, expand_page, put_bytes};

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
    // `parse_exact_int`'s value without its format: at most 19 digits cannot overflow.
    let value = |v: &[u8]| {
        let (neg, digits) = match v.first()? {
            b'+' => (false, &v[1..]),
            b'-' => (true, &v[1..]),
            _ => (false, v),
        };
        if digits.is_empty() || digits.len() > 19 {
            return None;
        }
        let mut u = 0u64;
        for &b in digits {
            let d = b.wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            u = u * 10 + d as u64;
        }
        match neg {
            true => 0i64.checked_sub_unsigned(u),
            false => i64::try_from(u).ok(),
        }
    };
    let mut ints = vals.iter().map(|v| value(v));
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
    layout: Option<(&ModelPlan, &[u32])>,
    paging: Option<crate::packs::Paging<'_>>,
    packs: &PackRegistry,
    backend: EntropyBackend,
    level: i32,
    picks: Option<&[crate::packs::Pick]>,
) -> EncodedDenseBlock {
    packs.encode_dense_block_picked(
        occ_streams,
        dict_groups,
        horiz_cands,
        preencoded_nested,
        standalone,
        layout,
        paging,
        backend,
        level,
        picks,
    )
}

/// A block's text pool as a reader holds it.
pub enum PoolParts<'a> {
    Leaf(&'a [u8]),
    /// Each page's leaf, `None` if not held, and its raw length.
    Pages(Vec<(Option<&'a [u8]>, u32)>),
}

impl PoolParts<'_> {
    /// Compressed bytes of the leaves held.
    pub(crate) fn held_bytes(&self) -> usize {
        match self {
            PoolParts::Leaf(l) => l.len(),
            PoolParts::Pages(pages) => pages.iter().flat_map(|p| p.0).map(<[u8]>::len).sum(),
        }
    }
}

/// Lazily materialized block-level shared dictionaries, prose pool, and peer columns.
pub struct BlockDecodeCtx<'a> {
    pub col_blobs: Vec<Option<&'a [u8]>>,
    dict_blobs: Vec<Option<&'a [u8]>>,
    dict_cache: Vec<Option<Arc<Cells>>>,
    pool_parts: Option<PoolParts<'a>>,
    pool_cache: Option<Arc<Pool>>,
    pub cache: Vec<Option<Col>>,
    cells_cache: Vec<Option<Arc<Cells>>>,
    /// The model's library, which the block's code references name.
    library: Arc<Library>,
    /// The model and each record's variant, which place the cells `ExtAlign` pairs up.
    layout: Option<(&'a ModelPlan, Arc<[u32]>)>,
    /// By column: each cell's value id, the rank of its first occurrence among distinct values.
    key_ids: Vec<Option<Arc<Vec<u32>>>>,
    /// By row a reader said it reads: each column's cursor before the row and after it.
    rows_read: Vec<(Vec<usize>, Vec<usize>)>,
    /// By column: whether the cached column holds only the cells `rows_read` reads.
    cut: Vec<bool>,
}

/// For each cell of column `at`, the index of the first cell of column `key` in the same record.
/// Only shape variants place cells by the variant alone, so any other variant is an error, as is
/// a record holding an `at` cell and no `key` cell.
pub fn align_cells(plan: &ModelPlan, variant_ids: &[u32], key: usize, at: usize) -> Result<Vec<i64>> {
    // Per variant, key and `at` cells per record, `at` cells `u32::MAX` for a variant whose
    // record is an error.
    let counts: Vec<(i64, u32)> = plan
        .variants
        .iter()
        .map(|v| match v {
            PlanVariant::Shape(s) => {
                let n = |c: usize| s.slots.iter().filter(|&&x| x as usize == c).count() as u32;
                let (nk, na) = (n(key), n(at));
                (i64::from(nk), if na > 0 && nk == 0 { u32::MAX } else { na })
            }
            _ => (0, u32::MAX),
        })
        .collect();
    let mut out = Vec::with_capacity(variant_ids.len());
    let mut k = 0i64;
    for &v in variant_ids {
        let &(nk, na) = counts
            .get(v as usize)
            .ok_or_else(|| Error::Format("variant id out of range".into()))?;
        if na != 0 {
            if na == u32::MAX {
                return Err(Error::Format("ExtAlign record has no placed key cell".into()));
            }
            out.extend(std::iter::repeat_n(k, na as usize));
        }
        k += nk;
    }
    Ok(out)
}

/// Each cell's value id: the rank of its value's first occurrence. Open addressing on a
/// multiply-fold hash; a hash match is confirmed on the bytes.
fn value_ids(cells: &Cells) -> Vec<u32> {
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

/// The pages held laid out at their places in the pool, the rest left missing. Every page but
/// the first expands against the first, so without it none does.
fn assemble_pages(
    pages: &[(Option<&[u8]>, u32)],
    budget: &DecodeBudget,
) -> Result<Pool> {
    let total: u64 = pages.iter().map(|p| p.1 as u64).sum();
    let Some(&(first, first_len)) = pages.first() else {
        return Err(Error::Format("text pool without pages".into()));
    };
    if total > u32::MAX as u64 {
        return Err(Error::BudgetExceeded("text pool"));
    }
    budget.check_alloc(total as usize, "text pool")?;
    let mut bytes = vec![0u8; total as usize];
    let (head, mut rest) = bytes.split_at_mut(first_len as usize);
    let mut missing = Vec::new();
    match first {
        Some(leaf) => expand_page(leaf, None, head)?,
        None => missing.push((0, first_len)),
    }
    let mut at = first_len;
    for &(leaf, len) in &pages[1..] {
        let (dst, tail) = std::mem::take(&mut rest).split_at_mut(len as usize);
        rest = tail;
        match leaf.filter(|_| first.is_some()) {
            Some(leaf) => expand_page(leaf, Some(head), dst)?,
            None => missing.push((at, at + len)),
        }
        at += len;
    }
    Ok(Pool::partial(bytes, missing))
}

impl<'a> BlockDecodeCtx<'a> {
    pub fn new(
        col_blobs: Vec<Option<&'a [u8]>>,
        dict_blobs: Vec<Option<&'a [u8]>>,
        pool_parts: Option<PoolParts<'a>>,
        library: Arc<Library>,
    ) -> Self {
        let n_cols = col_blobs.len();
        let n_dicts = dict_blobs.len();
        Self {
            col_blobs,
            dict_blobs,
            dict_cache: (0..n_dicts).map(|_| None).collect(),
            pool_parts,
            pool_cache: None,
            cache: (0..n_cols).map(|_| None).collect(),
            cells_cache: (0..n_cols).map(|_| None).collect(),
            library,
            layout: None,
            key_ids: (0..n_cols).map(|_| None).collect(),
            rows_read: Vec::new(),
            cut: vec![false; n_cols],
        }
    }

    /// Promises that only the cells of `rows`, given as each column's cursor before a row and
    /// after it, will be read, so a column may be built only that far. A column another one
    /// reads is still built whole.
    pub fn read_only(&mut self, rows: Vec<(Vec<usize>, Vec<usize>)>) {
        self.rows_read = rows;
    }

    /// The cells of column `c` the rows read, as ascending ranges; `None` for all.
    fn reads(&self, c: usize) -> Option<Vec<(usize, usize)>> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        for (before, after) in &self.rows_read {
            let (s, e) = (*before.get(c)?, *after.get(c)?);
            match out.last_mut() {
                _ if s >= e => {}
                Some(last) if last.1 == s => last.1 = e,
                _ => out.push((s, e)),
            }
        }
        (!out.is_empty()).then_some(out)
    }

    pub fn with_layout(mut self, plan: &'a ModelPlan, variant_ids: Arc<[u32]>) -> Self {
        self.layout = Some((plan, variant_ids));
        self
    }

    fn ensure_pool(&mut self, budget: &DecodeBudget) -> Result<&Arc<Pool>> {
        if self.pool_cache.is_none() {
            let pool = match self.pool_parts.as_ref().ok_or(Error::NotLoaded)? {
                PoolParts::Leaf(leaf) => Pool::whole(unpack_leaf(leaf, budget)?),
                PoolParts::Pages(pages) => assemble_pages(pages, budget)?,
            };
            self.pool_cache = Some(Arc::new(pool));
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
    if ctx.cache.get(col_idx).is_none_or(Option::is_none) {
        let reads = ctx.reads(col_idx);
        ensure_col_inner(col_idx, ctx, budget, 0, reads.as_deref())?;
    }
    Ok(ctx.cache[col_idx].as_ref().unwrap())
}

fn ensure_col_inner(
    col_idx: usize,
    ctx: &mut BlockDecodeCtx<'_>,
    budget: &DecodeBudget,
    depth: usize,
    reads: Option<&[(usize, usize)]>,
) -> Result<()> {
    if col_idx >= ctx.col_blobs.len() {
        return Err(Error::Format("column index out of bounds".into()));
    }
    if ctx.cache[col_idx].is_some() && !(ctx.cut[col_idx] && reads.is_none()) {
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

    // Peers only `ExtAlign` reads stay unmaterialized; their slot holds no cells.
    let mut as_cells = vec![false; graph.ext_peers.len()];
    for n in &graph.nodes {
        if let GraphNode::ExtPeer { slot } = *n {
            as_cells[slot as usize] = true;
        }
    }
    let tables = if graph.ext_peers.iter().any(|&p| p & TABLE_PEER != 0) {
        Some(with_current(|d| d.and_then(|d| d.tables().cloned())).ok_or(Error::NotLoaded)?)
    } else {
        None
    };
    let table_col = |p: u32| -> Result<Option<usize>> {
        if p & TABLE_PEER == 0 {
            return Ok(None);
        }
        let t = (p & !TABLE_PEER) as usize;
        match &tables {
            Some(tables) if t < tables.len() => Ok(Some(t)),
            _ => Err(Error::Format("table column out of range".into())),
        }
    };
    let mut peers = Vec::with_capacity(graph.ext_peers.len());
    for (&peer_id, &as_cells) in graph.ext_peers.iter().zip(&as_cells) {
        if let Some(t) = table_col(peer_id)? {
            let tables = tables.as_ref().unwrap();
            peers.push(if as_cells { Arc::clone(tables.col(t)?) } else { Arc::default() });
            continue;
        }
        let pid = peer_id as usize;
        if pid >= col_idx {
            return Err(Error::Format(
                "horizontal forward or self column reference".into(),
            ));
        }
        ensure_col_inner(pid, ctx, budget, depth + 1, None)?;
        if !as_cells {
            peers.push(Arc::default());
            continue;
        }
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

    let mut aligned = Vec::new();
    for node in &graph.nodes {
        if let GraphNode::ExtAlign { key, at } = *node {
            let peer = |s: u16| graph.ext_peers[s as usize];
            if table_col(peer(key))?.is_some() {
                return Err(Error::Format("ExtAlign key is a table column".into()));
            }
            let key = peer(key) as usize;
            let at_peer = (at > 0).then(|| peer(at - 1));
            // `at` naming a table column aligns this column, with key ids joint with the table's.
            let joint = at_peer.map(table_col).transpose()?.flatten();
            let at = match (joint, at_peer) {
                (None, Some(p)) => p as usize,
                _ => col_idx,
            };
            let ids = if let Some(t) = joint {
                let cells = ctx.cache[key].as_ref().unwrap().to_cells()?;
                Arc::new(tables.as_ref().unwrap().joint_ids(t, cells.iter())?)
            } else {
                if ctx.key_ids[key].is_none() {
                    let ids = match ctx.cache[key].as_ref().unwrap() {
                        Col::Dict { dict, ords } => {
                            let d = value_ids(dict);
                            ords.iter().map(|&o| d[o as usize]).collect()
                        }
                        Col::Flat(c) => value_ids(c),
                        other => value_ids(&other.to_cells()?),
                    };
                    ctx.key_ids[key] = Some(Arc::new(ids));
                }
                Arc::clone(ctx.key_ids[key].as_ref().unwrap())
            };
            let (plan, vids) = ctx.layout.as_ref().ok_or(Error::NotLoaded)?;
            let mut idx = align_cells(plan, vids, key, at)?;
            for i in &mut idx {
                *i = i64::from(
                    *ids.get(*i as usize)
                        .ok_or_else(|| Error::Format("ExtAlign key cell underflow".into()))?,
                );
            }
            aligned.push(idx);
        }
    }
    let ext = ExtInputs {
        pool: pool_arc.as_ref(),
        dict: dict_arc.as_ref().map(|(d, a)| (*d, a)),
        peers: &peers,
        aligned: &aligned,
    };
    let col = CoreDecoder::decode_col_reading(&graph, &ext, budget, reads)?;
    ctx.cut[col_idx] = reads.is_some();
    ctx.cache[col_idx] = Some(col);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{PackOptions, pack_stream, unpack_stream};
    use crate::packs::{CodecPack, ColumnProfile};
    use crate::payload::graph::GraphBuilder;

    #[test]
    fn int_bounds_parse_as_exact_ints_do() {
        let spellings = [
            "", "+", "-", "0", "-0", "+0", "007", "-007", "42", "4a", " 1", "1 ",
            "9223372036854775807", "9223372036854775808", "-9223372036854775808", "-9223372036854775809",
            "9999999999999999999", "-9999999999999999999", "+9999999999999999999", "10000000000000000000",
        ];
        for s in spellings {
            let want = crate::lex::parse_exact_int(s.as_bytes()).map(|(v, _)| (v, v));
            assert_eq!(int_bounds(&[s.as_bytes()]), want, "{s:?}");
        }
    }
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
                let b = Builder::new(1);
                let d = b.input(0);
                let v = b.scan_add(d, nums[0]);
                let scaled = b.ri(Op::MulI, v, 100);
                let (_, asc) = crate::packs::eight_digits_raw_and_ascii(&b, scaled);
                let dn = gb.add_ints(&deltas);
                let seq_cells = gb.add_prog_cells(b.finish(&[(asc, 6)]), vec![dn], vec![]);
                let cat = gb.add_template(&[b"ORD-2026-", b""], vec![seq_cells], rows.len());
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

    /// Columns `src`, `key`, `dst` over `recv <key> from <src>` and `got <key> at <dst>`
    /// records: every key gets one `recv` row, then two `got` rows that mostly repeat its `src`.
    fn keyed_block() -> (ModelPlan, Vec<u32>, Vec<Vec<Vec<u8>>>) {
        use crate::grammar::{FrontEnd, Shape};
        let mut plan = ModelPlan::residual_only();
        for name in ["src", "key", "dst"] {
            plan.add_or_get_col(name);
        }
        let shape = |lits: &[&[u8]], slots: &[u16]| {
            PlanVariant::Shape(Shape {
                literals: lits.iter().map(|l| l.to_vec()).collect(),
                slots: slots.to_vec(),
            })
        };
        plan.push_variant(shape(&[b"recv ", b" from ", b""], &[2, 1]), FrontEnd::None);
        plan.push_variant(shape(&[b"got ", b" at ", b""], &[2, 3]), FrontEnd::None);
        let mut cols = vec![Vec::new(); 4];
        let mut vids = Vec::new();
        let mut x = 7u64;
        let mut ip = || {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            format!("10.0.{}.{}", (x >> 40) % 8, (x >> 50) % 200).into_bytes()
        };
        let mut pending: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for i in 0..1500u32 {
            let key = format!("blk_{}", i.wrapping_mul(2_654_435_761)).into_bytes();
            let src = ip();
            vids.push(1);
            cols[2].push(key.clone());
            cols[1].push(src.clone());
            pending.push((key, src));
            if pending.len() > 4 {
                let (key, src) = pending.remove(0);
                for j in 0..2 {
                    vids.push(2);
                    cols[2].push(key.clone());
                    cols[3].push(if (i + j) % 13 == 0 { ip() } else { src.clone() });
                }
            }
        }
        (plan, vids, cols)
    }

    #[test]
    fn keyed_lookup_round_trips_through_block_layer() {
        let (plan, vids, cols) = keyed_block();
        let occ: Vec<Vec<&[u8]>> = cols.iter().map(|c| c.iter().map(|v| &v[..]).collect()).collect();
        let n = occ.len();
        let enc = PackRegistry::default().encode_dense_block(
            &occ,
            &vec![""; n],
            &vec![Vec::new(); n],
            &mut vec![None; n],
            &vec![false; n],
            Some((&plan, &vids)),
            None,
            EntropyBackend::Zstd,
            3,
        );
        let budget = DecodeBudget::default();
        let g = PayloadGraph::decode(&enc.col_blobs[3], &budget).unwrap();
        assert!(g.nodes.iter().any(|n| matches!(n, GraphNode::MatchIndex { .. })));
        let blobs = enc.col_blobs.iter().map(|b| Some(&b[..])).collect();
        let dicts = enc.shared_dict_blobs.iter().map(|b| Some(&b[..])).collect();
        let mut ctx = BlockDecodeCtx::new(blobs, dicts, None, Arc::default())
            .with_layout(&plan, vids.clone().into());
        let got = ensure_dense_block_column(3, &mut ctx, &budget).unwrap().to_cells().unwrap();
        assert!(got.iter().eq(occ[3].iter().copied()));
        // Without the layout the column cannot be aligned.
        let blobs = enc.col_blobs.iter().map(|b| Some(&b[..])).collect();
        let mut bare = BlockDecodeCtx::new(blobs, vec![], None, Arc::default());
        assert!(ensure_dense_block_column(3, &mut bare, &budget).is_err());
    }

    #[test]
    fn align_cells_pairs_records_and_rejects_unplaced_keys() {
        let (mut plan, vids, cols) = keyed_block();
        // Each `got` record's own key cell, in record order.
        let mut want = Vec::new();
        let mut k = 0i64;
        for &v in &vids {
            if v == 2 {
                want.push(k);
            }
            k += 1;
        }
        assert_eq!(align_cells(&plan, &vids, 2, 3).unwrap(), want);
        assert_eq!(want.len(), cols[3].len());
        // Records of the second layout hold `dst` but no `src`.
        assert!(align_cells(&plan, &vids, 1, 3).is_err());
        // Only shapes place cells.
        plan.push_variant(PlanVariant::JsonTree, crate::grammar::FrontEnd::None);
        assert!(align_cells(&plan, &[1, 3], 2, 1).is_err());
        assert!(align_cells(&plan, &[1, 9], 2, 1).is_err());
    }
}
