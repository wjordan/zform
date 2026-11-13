//! Queryable files: a data region of blocks and an index region over them.
//!
//! The reader does no IO. Each operation takes a `Source`, and when the source lacks bytes the
//! operation returns the ranges it found missing and is run again once the host has them. Rows
//! are numbered from 0 across the file.

use crate::dictionary::{Dictionary, Scope};
use crate::container::{BlockParts, BlockRows, FrameFit, PackOptions, StreamEncoder, Structure};
use crate::framing::{FramingSpec, split_passthrough};
use crate::grammar::{ModelPlan, PlanVariant};
use crate::index::{
    BlockDir, BlockEntry, BlockKind, Filter, INDEX_TAIL_LEN, KIND_JSON, Root, locate_root,
};
use crate::query::{
    BlockMeta, FieldEv, Need, Plans, Predicate, UNKNOWN, YES, eval, eval_raw_record, field_events,
    filter_use, json_variant, may_match, needs, plain_may_contain, raw_field_values,
};
use crate::wire::{Cursor, DecodeBudget, Error, Result, decompress_leaf};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

// ============================================================================
// Bytes
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Region {
    Data,
    Index,
}

/// `len` bytes at `off` from the start of a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteRange {
    pub region: Region,
    pub off: u64,
    pub len: u64,
}

impl ByteRange {
    pub fn end(&self) -> u64 {
        self.off + self.len
    }
}

/// Where a file's bytes come from.
pub trait Source {
    /// The bytes of `r`, or `None` if the host does not have all of them at hand.
    fn get(&self, r: ByteRange) -> Option<&[u8]>;
}

/// The result of an operation: its value, or the byte ranges it needs before it can run.
#[derive(Debug)]
pub enum Step<T> {
    Ready(T),
    Missing(Vec<ByteRange>),
}

/// Both regions in memory.
pub struct MemSource<'a> {
    pub data: &'a [u8],
    pub index: &'a [u8],
}

impl Source for MemSource<'_> {
    fn get(&self, r: ByteRange) -> Option<&[u8]> {
        let buf = match r.region {
            Region::Data => self.data,
            Region::Index => self.index,
        };
        buf.get(r.off as usize..r.end() as usize)
    }
}

/// Ranges a host has fetched.
#[derive(Default)]
pub struct RangeCache {
    parts: BTreeMap<(Region, u64), Vec<u8>>,
}

impl RangeCache {
    /// Adds a range, joining it with the ranges it touches so that no two held ranges touch.
    pub fn insert(&mut self, r: ByteRange, bytes: Vec<u8>) {
        debug_assert_eq!(bytes.len() as u64, r.len);
        let from = self
            .start_at_or_before(r.region, r.off)
            .filter(|s| s.1 >= r.off)
            .map_or(r.off, |s| s.0);
        let joined: Vec<u64> = self
            .parts
            .range((r.region, from)..=(r.region, r.end()))
            .map(|(k, _)| k.1)
            .collect();
        if joined.is_empty() {
            self.parts.insert((r.region, r.off), bytes);
            return;
        }
        let mut buf = Vec::new();
        for off in joined {
            let part = self.parts.remove(&(r.region, off)).unwrap();
            let at = (off - from) as usize;
            buf.resize(buf.len().max(at + part.len()), 0);
            buf[at..at + part.len()].copy_from_slice(&part);
        }
        let at = (r.off - from) as usize;
        buf.resize(buf.len().max(at + bytes.len()), 0);
        buf[at..at + bytes.len()].copy_from_slice(&bytes);
        self.parts.insert((r.region, from), buf);
    }

    /// The held range starting at or before `off`, as (start, end).
    fn start_at_or_before(&self, region: Region, off: u64) -> Option<(u64, u64)> {
        let (k, v) = self.parts.range((region, 0)..=(region, off)).next_back()?;
        Some((k.1, k.1 + v.len() as u64))
    }

    /// The parts of `r` not held.
    pub fn gaps(&self, r: ByteRange) -> Vec<ByteRange> {
        let mut at = match self.start_at_or_before(r.region, r.off) {
            Some((_, end)) => end.max(r.off),
            None => r.off,
        };
        let mut out = Vec::new();
        for (k, v) in self.parts.range((r.region, r.off + 1)..(r.region, r.end())) {
            if k.1 > at {
                out.push(ByteRange {
                    region: r.region,
                    off: at,
                    len: k.1 - at,
                });
            }
            at = k.1 + v.len() as u64;
        }
        if at < r.end() {
            out.push(ByteRange {
                region: r.region,
                off: at,
                len: r.end() - at,
            });
        }
        out
    }

    /// Drops the ranges of one region.
    pub fn clear_region(&mut self, region: Region) {
        self.parts.retain(|k, _| k.0 != region);
    }

    /// Bytes held.
    pub fn len(&self) -> u64 {
        self.parts.values().map(|v| v.len() as u64).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}

impl Source for RangeCache {
    fn get(&self, r: ByteRange) -> Option<&[u8]> {
        let (k, v) = self
            .parts
            .range((r.region, 0)..=(r.region, r.off))
            .next_back()?;
        v.get((r.off - k.1) as usize..(r.end() - k.1) as usize)
    }
}

/// Sorts ranges and joins those that touch or overlap.
pub fn coalesce(mut ranges: Vec<ByteRange>) -> Vec<ByteRange> {
    ranges.sort_unstable();
    let mut out = Vec::<ByteRange>::with_capacity(ranges.len());
    for r in ranges.into_iter().filter(|r| r.len > 0) {
        match out.last_mut() {
            Some(last) if last.region == r.region && r.off <= last.end() => {
                last.len = last.len.max(r.end() - last.off);
            }
            _ => out.push(r),
        }
    }
    out
}

/// Runs `op` until it is ready. Each time it reports ranges missing, `fetch` is called once with
/// the parts of them the cache does not hold, joined where they touch, and returns the bytes of
/// each in order.
pub fn drive<T>(
    cache: &mut RangeCache,
    mut fetch: impl FnMut(&[ByteRange]) -> Result<Vec<Vec<u8>>>,
    mut op: impl FnMut(&RangeCache) -> Result<Step<T>>,
) -> Result<T> {
    loop {
        let missing = match op(cache)? {
            Step::Ready(v) => return Ok(v),
            Step::Missing(m) => coalesce(m.into_iter().flat_map(|r| cache.gaps(r)).collect()),
        };
        let before = cache.len();
        let parts = fetch(&missing)?;
        if parts.len() != missing.len()
            || parts
                .iter()
                .zip(&missing)
                .any(|(p, r)| p.len() as u64 != r.len)
        {
            return Err(Error::Io("short read of a file range".into()));
        }
        for (r, bytes) in missing.into_iter().zip(parts) {
            cache.insert(r, bytes);
        }
        if cache.len() == before {
            return Err(Error::Format(
                "reader asked again for ranges it was given".into(),
            ));
        }
    }
}

// ============================================================================
// Row sets
// ============================================================================

/// A set of row numbers, kept as sorted ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowSet {
    /// Half-open, ascending, neither touching nor overlapping.
    ranges: Vec<(u64, u64)>,
}

impl RowSet {
    pub fn range(start: u64, end: u64) -> Self {
        let mut s = Self::default();
        s.push_range(start, end);
        s
    }

    /// Rows in ascending order.
    pub fn from_sorted(rows: impl IntoIterator<Item = u64>) -> Self {
        let mut s = Self::default();
        for r in rows {
            s.push_range(r, r + 1);
        }
        s
    }

    /// Adds `start..end`, which must not begin before the last range added.
    pub fn push_range(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        match self.ranges.last_mut() {
            Some(last) if start <= last.1 => {
                debug_assert!(start >= last.0);
                last.1 = last.1.max(end);
            }
            _ => self.ranges.push((start, end)),
        }
    }

    pub fn len(&self) -> u64 {
        self.ranges.iter().map(|r| r.1 - r.0).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub fn ranges(&self) -> &[(u64, u64)] {
        &self.ranges
    }

    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.ranges.iter().flat_map(|r| r.0..r.1)
    }

    pub fn contains(&self, row: u64) -> bool {
        let i = self.ranges.partition_point(|r| r.1 <= row);
        self.ranges.get(i).is_some_and(|r| r.0 <= row)
    }

    pub fn intersect(&self, other: &RowSet) -> RowSet {
        let (a, b) = (&self.ranges, &other.ranges);
        let (mut i, mut j) = (0, 0);
        let mut out = RowSet::default();
        while i < a.len() && j < b.len() {
            out.push_range(a[i].0.max(b[j].0), a[i].1.min(b[j].1));
            if a[i].1 < b[j].1 {
                i += 1;
            } else {
                j += 1;
            }
        }
        out
    }

    pub fn union(&self, other: &RowSet) -> RowSet {
        let mut all: Vec<_> = self.ranges.iter().chain(&other.ranges).copied().collect();
        all.sort_unstable();
        let mut out = RowSet::default();
        for (s, e) in all {
            out.push_range(s, e);
        }
        out
    }

    /// The rows of `start..end`, as offsets from `start`.
    fn within(&self, start: u64, end: u64) -> impl Iterator<Item = (u32, u32)> + '_ {
        let i = self.ranges.partition_point(|r| r.1 <= start);
        self.ranges[i..]
            .iter()
            .take_while(move |r| r.0 < end)
            .map(move |r| {
                (
                    (r.0.max(start) - start) as u32,
                    (r.1.min(end) - start) as u32,
                )
            })
    }
}

// ============================================================================
// Files
// ============================================================================

/// An open file: its root and models.
pub struct File {
    root: Root,
    plans: Vec<ModelPlan>,
    /// First row of each block, then the row count.
    starts: Vec<u64>,
    budget: DecodeBudget,
    /// Query plans, shared by the file's sessions.
    query_plans: Arc<Mutex<Plans>>,
    /// The dictionary, once a read has needed it.
    dict: OnceLock<Arc<Dictionary>>,
}

/// How much of the index region's end to ask for before the root's size is known.
const TAIL_GUESS: u64 = 256 << 10;

/// Where the root starts in an index region. `File::open` reads only from there on, so a host
/// can keep that tail apart from the pages before it.
pub fn root_offset(index: &[u8]) -> Result<u64> {
    let tail = index
        .len()
        .checked_sub(INDEX_TAIL_LEN)
        .ok_or_else(|| Error::Format("index region too short".into()))?;
    Ok(locate_root(&index[tail..], index.len() as u64)?.0)
}

impl File {
    /// Opens the file whose index region is `index_len` bytes long.
    pub fn open(src: &dyn Source, index_len: u64, budget: DecodeBudget) -> Result<Step<File>> {
        let tail_len = INDEX_TAIL_LEN as u64;
        if index_len < tail_len {
            return Err(Error::Format("index region too short".into()));
        }
        let index = |off: u64, len: u64| ByteRange {
            region: Region::Index,
            off,
            len,
        };
        let Some(tail) = src.get(index(index_len - tail_len, tail_len)) else {
            let guess = index_len.min(TAIL_GUESS);
            return Ok(Step::Missing(vec![index(index_len - guess, guess)]));
        };
        let (off, len, crc) = locate_root(tail, index_len)?;
        let Some(root) = src.get(index(off, len)) else {
            return Ok(Step::Missing(vec![index(off, len)]));
        };
        let root = Root::decode(root, crc, &budget)?;
        let mut plans = Vec::with_capacity(root.models.len());
        for m in &root.models {
            plans.push(ModelPlan::decode(&mut Cursor::new(m), 0)?);
        }
        let mut starts = Vec::with_capacity(root.blocks.len() + 1);
        let mut row = 0u64;
        for b in &root.blocks {
            if b.rows > budget.max_records_per_block as u64 || b.dir_off + b.dir_len as u64 > off {
                return Err(Error::Format("index root block entry out of bounds".into()));
            }
            starts.push(row);
            row += b.rows;
        }
        starts.push(row);
        Ok(Step::Ready(File {
            root,
            plans,
            starts,
            budget,
            query_plans: Arc::default(),
            dict: OnceLock::new(),
        }))
    }

    /// The id of the dictionary the file's blocks were written with, if any. The first read of
    /// a block fetches and decodes it, and the file keeps it. A host that kept it from an
    /// earlier open spares that with `set_dictionary`.
    pub fn dictionary_id(&self) -> Option<u64> {
        self.root.dict.map(|d| d.id)
    }

    /// The dictionary, if the file holds it by now.
    pub fn dictionary(&self) -> Option<&Arc<Dictionary>> {
        self.dict.get()
    }

    pub fn set_dictionary(&self, dict: Arc<Dictionary>) -> Result<()> {
        if self.dictionary_id() != Some(dict.id()) {
            return Err(Error::InvalidArgument("not the file's dictionary".into()));
        }
        let _ = self.dict.set(dict);
        Ok(())
    }

    /// Reads the dictionary ahead of the first block read, which would read it otherwise.
    pub fn load_dictionary(&self, src: &dyn Source) -> Result<Step<()>> {
        Ok(match self.dict_scope(src)? {
            Ok(_) => Step::Ready(()),
            Err(range) => Step::Missing(vec![range]),
        })
    }

    /// Puts the dictionary in scope for the leaves decoded while the guard lives. `Err` is the
    /// range to fetch first.
    fn dict_scope(&self, src: &dyn Source) -> Result<std::result::Result<Scope, ByteRange>> {
        if let (Some(e), None) = (&self.root.dict, self.dict.get()) {
            let range = data(e.off, e.len);
            let Some(leaf) = src.get(range) else {
                return Ok(Err(range));
            };
            let _ = self.dict.set(Arc::new(Dictionary::read(leaf, e.id, &self.budget)?));
        }
        Ok(Ok(Scope::enter(self.dict.get().cloned())))
    }

    pub fn rows(&self) -> u64 {
        *self.starts.last().unwrap()
    }

    /// Length of the data region.
    pub fn data_len(&self) -> u64 {
        self.root.data_len
    }

    /// Each block's first row and row count.
    pub fn blocks(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.starts.windows(2).map(|w| (w[0], w[1] - w[0]))
    }

    /// The field paths the file's models define, as `fields` and predicates take them.
    pub fn paths(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .plans
            .iter()
            .filter_map(|p| p.json_tree.as_ref())
            .flat_map(|jt| {
                jt.paths
                    .iter()
                    .filter(|p| p.shapes.is_empty() && p.value_col != u16::MAX)
            })
            .map(|p| p.name.strip_prefix("$.").unwrap_or(&p.name).to_string())
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn session(&self) -> Session<'_> {
        Session {
            file: self,
            dirs: vec![None; self.root.blocks.len()],
            plans: Plans::sharing(&self.query_plans),
            filters: true,
            match_key: String::new(),
            matched: HashMap::new(),
            value_key: String::new(),
            values: HashMap::new(),
            stats: Stats::default(),
        }
    }

    /// Blocks holding rows of `rows`, or every block.
    fn blocks_in(&self, rows: Option<&RowSet>) -> Vec<usize> {
        let Some(rows) = rows else {
            return (0..self.root.blocks.len()).collect();
        };
        let mut out = Vec::new();
        for &(s, e) in rows.ranges() {
            let first = self.starts.partition_point(|&st| st <= s).saturating_sub(1);
            let from = out.last().map_or(first, |&l: &usize| first.max(l + 1));
            out.extend((from..self.root.blocks.len()).take_while(|&b| self.starts[b] < e));
        }
        out
    }
}

/// What a session did, for a host's accounting.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Blocks an operation considered, and those the index ruled out.
    pub blocks: usize,
    pub blocks_pruned: usize,
    /// Blocks read in part, and in full.
    pub blocks_partial: usize,
    pub blocks_whole: usize,
    /// Blocks read in full that hold their records verbatim, with no model.
    pub blocks_verbatim: usize,
    /// Records rebuilt to test or to read fields from.
    pub rebuilt: usize,
}

/// The decoded state of one query. An operation that returned `Missing` is run again on the same
/// session with the same arguments.
pub struct Session<'f> {
    file: &'f File,
    dirs: Vec<Option<Rc<BlockDir>>>,
    plans: Plans,
    /// Filter pages are read to rule blocks out.
    filters: bool,
    /// The predicate `matched` answers, and by block its rows as ranges within the block.
    match_key: String,
    matched: HashMap<usize, Rc<Vec<(u32, u32)>>>,
    /// The paths and rows `values` holds, by block.
    value_key: String,
    values: HashMap<usize, BlockVals>,
    pub stats: Stats,
}

/// Field values of one block's selected rows.
struct BlockVals {
    arena: Vec<u8>,
    /// By path: `(row, start, end)` into `arena`, by row.
    vals: Vec<Vec<(u32, u32, u32)>>,
}

fn data(off: u64, len: u64) -> ByteRange {
    ByteRange {
        region: Region::Data,
        off,
        len,
    }
}

fn dir_page(e: &BlockEntry) -> ByteRange {
    ByteRange {
        region: Region::Index,
        off: e.dir_off,
        len: e.dir_len as u64,
    }
}

fn filter_page(e: &BlockEntry) -> ByteRange {
    ByteRange {
        region: Region::Index,
        off: e.filter_off,
        len: e.filter_len as u64,
    }
}

/// A block's filter, or `None` if the source lacks it.
fn filter<'s>(src: &'s dyn Source, e: &BlockEntry) -> Result<Option<Filter<'s>>> {
    match src.get(filter_page(e)).map(Filter) {
        Some(f) if !f.valid() => Err(Error::Format("invalid block filter".into())),
        f => Ok(f),
    }
}

fn whole(e: &BlockEntry) -> ByteRange {
    data(e.off, e.len)
}

/// The parts of a block a read uses, by index.
struct Wanted {
    /// The variant ids.
    head: bool,
    streams: Vec<bool>,
    cols: Vec<bool>,
    dicts: Vec<bool>,
    pool: bool,
}

impl Wanted {
    /// What `need` names, with the dictionaries, pool and source columns its columns depend on.
    /// The variant ids are left out of a block whose rows are all of the model's one JSON
    /// variant.
    fn of(dir: &BlockDir, need: &Need, plan: &ModelPlan, kinds: u16) -> Self {
        let json = plan
            .variants
            .iter()
            .filter(|v| matches!(v, PlanVariant::JsonTree))
            .count();
        let mut w = Wanted {
            head: kinds != KIND_JSON || json != 1,
            streams: need.streams.clone(),
            cols: need.cols.clone(),
            dicts: vec![false; dir.dicts.len()],
            pool: false,
        };
        let mut stack: Vec<usize> = (0..w.cols.len()).filter(|&c| w.cols[c]).collect();
        while let Some(c) = stack.pop() {
            let deps = &dir.col(c).deps;
            w.pool |= deps.pool;
            if let Some(d) = deps.dict.filter(|&d| (d as usize) < w.dicts.len()) {
                w.dicts[d as usize] = true;
            }
            for &s in &deps.cols {
                if (s as usize) < w.cols.len() && !std::mem::replace(&mut w.cols[s as usize], true)
                {
                    stack.push(s as usize);
                }
            }
        }
        w
    }

    /// The parts as `(wanted, offset in the block, length)`: the structure's head, then by kind.
    fn spans<'d>(&'d self, dir: &'d BlockDir) -> impl Iterator<Item = (bool, u32, u32)> + 'd {
        let streams = dir
            .stream_spans()
            .zip(&self.streams)
            .map(|((off, len), &on)| (on, off, len));
        let dicts = dir
            .dicts
            .iter()
            .zip(&self.dicts)
            .map(|(&(off, len), &on)| (on, off, len));
        let pool = dir.pool.iter().map(|&(off, len)| (self.pool, off, len));
        let cols = dir
            .col_spans()
            .zip(&self.cols)
            .map(|((off, len), &on)| (on, off, len));
        std::iter::once((self.head, 0, dir.struct_head))
            .chain(streams)
            .chain(dicts)
            .chain(pool)
            .chain(cols)
    }

    fn ranges(&self, e: &BlockEntry, dir: &BlockDir) -> Vec<ByteRange> {
        self.spans(dir)
            .filter(|s| s.0)
            .map(|(_, off, len)| data(e.off + off as u64, len as u64))
            .collect()
    }
}

/// A block's parts: all of them if the source has the whole block, and else those wanted.
fn load_parts<'s>(
    src: &'s dyn Source,
    e: &BlockEntry,
    dir: &BlockDir,
    plan: &ModelPlan,
    want: &Wanted,
    budget: &DecodeBudget,
) -> Result<Option<(BlockParts<'s>, bool)>> {
    if let Some(bytes) = src.get(whole(e)) {
        return Ok(Some((BlockParts::of_block(bytes, plan, budget)?, true)));
    }
    if dir.n_cols() != plan.columns.len() {
        return Err(Error::Format(
            "block directory column count != model".into(),
        ));
    }
    let mut parts = want.spans(dir).map(|(on, off, len)| match on {
        true => src.get(data(e.off + off as u64, len as u64)),
        false => None,
    });
    let head = parts.next().unwrap();
    if want.head && head.is_none() {
        return Ok(None);
    }
    Ok(Some((
        BlockParts {
            structure: Structure::Parts {
                head,
                streams: parts.by_ref().take(dir.streams.len()).collect(),
            },
            dicts: parts.by_ref().take(dir.dicts.len()).collect(),
            pool: dir.pool.and_then(|_| parts.next().unwrap()),
            cols: parts.collect(),
        },
        false,
    )))
}

/// The records of a verbatim block.
fn verbatim_records(
    bytes: &[u8],
    e: &BlockEntry,
    budget: &DecodeBudget,
) -> Result<(Vec<u8>, Vec<(u32, u32)>)> {
    let BlockKind::Verbatim { framing } = e.kind else {
        unreachable!()
    };
    let raw = decompress_leaf(&mut Cursor::new(bytes), budget)?;
    let fc = split_passthrough(&raw, framing);
    let spans = if fc.framing == framing {
        fc.spans
    } else {
        Vec::new()
    };
    if spans.len() as u64 != e.rows {
        return Err(Error::Format("verbatim block record count != index".into()));
    }
    Ok((raw, spans))
}

/// False when a verbatim block's filter proves no record satisfies `pred`.
fn verbatim_may_match(pred: &Predicate, filter: Filter<'_>) -> bool {
    match pred {
        Predicate::And(list) => list.iter().all(|p| verbatim_may_match(p, filter)),
        Predicate::Or(list) => list.iter().any(|p| verbatim_may_match(p, filter)),
        Predicate::Contains(needle) => plain_may_contain(needle, false, filter),
        Predicate::Word(needle) => plain_may_contain(needle, true, filter),
        _ => true,
    }
}

/// Rows of a block as a mask.
fn mask(sel: impl Iterator<Item = (u32, u32)>, n: usize) -> Vec<bool> {
    let mut m = vec![false; n];
    for (s, e) in sel {
        m[s as usize..e as usize].fill(true);
    }
    m
}

impl<'f> Session<'f> {
    /// The file's dictionary in scope for the blocks about to be decoded, or its range added to
    /// `missing`.
    fn dict(&self, src: &dyn Source, missing: &mut Vec<ByteRange>) -> Result<Option<Scope>> {
        Ok(match self.file.dict_scope(src)? {
            Ok(scope) => Some(scope),
            Err(range) => {
                missing.push(range);
                None
            }
        })
    }

    /// Keeps the values a run that came back `Missing` collected, when this is the same request.
    fn begin_values(&mut self, key: String) {
        if key != self.value_key {
            self.value_key = key;
            self.values.clear();
        }
    }

    /// A block's directory. `None` if the source lacks its page.
    fn dir(&mut self, src: &dyn Source, b: usize) -> Result<Option<Rc<BlockDir>>> {
        if self.dirs[b].is_none() {
            let e = &self.file.root.blocks[b];
            let dir = match e.kind {
                BlockKind::Structured { .. } => match src.get(dir_page(e)) {
                    Some(page) => BlockDir::decode(page, &self.file.budget)?,
                    None => return Ok(None),
                },
                BlockKind::Verbatim { .. } => BlockDir::default(),
            };
            self.dirs[b] = Some(Rc::new(dir));
        }
        Ok(self.dirs[b].clone())
    }

    /// Leaves the filter pages unread. They rule out little for a caller whose own index chose
    /// the rows it asks about.
    pub fn without_filters(mut self) -> Self {
        self.filters = false;
        self
    }

    /// Rows of `within`, or of the file, that satisfy `pred`. No record is built unless its
    /// columns cannot answer.
    pub fn matches(
        &mut self,
        src: &dyn Source,
        pred: &Predicate,
        within: Option<&RowSet>,
    ) -> Result<Step<RowSet>> {
        let file = self.file;
        if matches!(pred, Predicate::All) {
            let all = RowSet::range(0, file.rows());
            return Ok(Step::Ready(
                within.map_or(all.clone(), |w| w.intersect(&all)),
            ));
        }
        let key = format!("{pred:?}");
        if key != self.match_key {
            self.match_key = key;
            self.matched.clear();
        }
        let blocks: Vec<usize> = file
            .blocks_in(within)
            .into_iter()
            .filter(|b| !self.matched.contains_key(b))
            .collect();

        // The directories, with the filters when nothing else can rule a block out.
        let (uses_filter, only_filter) = filter_use(pred);
        let uses_filter = uses_filter && self.filters;
        let mut missing = Vec::new();
        let mut dirs = Vec::with_capacity(blocks.len());
        for &b in &blocks {
            let e = &file.root.blocks[b];
            match self.dir(src, b)? {
                Some(dir) => dirs.push(dir),
                None => missing.push(dir_page(e)),
            }
            if uses_filter && only_filter && src.get(filter_page(e)).is_none() {
                missing.push(filter_page(e));
            }
        }
        if !missing.is_empty() {
            return Ok(Step::Missing(missing));
        }

        // Rule blocks out, and find what the rest need.
        let mut todo = Vec::new();
        for (&b, dir) in blocks.iter().zip(&dirs) {
            let e = &file.root.blocks[b];
            let held = if uses_filter { filter(src, e)? } else { None };
            let filter = held.unwrap_or(Filter::NONE);
            let ranges = match e.kind {
                BlockKind::Verbatim { .. } => {
                    verbatim_may_match(pred, filter).then(|| vec![whole(e)])
                }
                BlockKind::Structured { model, kinds } => {
                    let m = BlockMeta {
                        model,
                        plan: &file.plans[model as usize],
                        kinds,
                        dir,
                        filter,
                    };
                    may_match(pred, &m, &mut self.plans).then(|| {
                        let mut need = Need::new(dir);
                        needs(pred, &m, &mut self.plans, &mut need);
                        if need.whole {
                            vec![whole(e)]
                        } else {
                            Wanted::of(dir, &need, m.plan, kinds).ranges(e, dir)
                        }
                    })
                }
            };
            let Some(mut ranges) = ranges else {
                self.stats.blocks += 1;
                self.stats.blocks_pruned += 1;
                self.matched.insert(b, Rc::new(Vec::new()));
                continue;
            };
            ranges.retain(|r| src.get(whole(e)).is_none() && src.get(*r).is_none());
            // A filter is worth fetching when it is smaller than what it may save.
            if uses_filter
                && held.is_none()
                && ranges.iter().map(|r| r.len).sum::<u64>() > e.filter_len as u64
            {
                missing.push(filter_page(e));
                continue;
            }
            missing.extend(ranges);
            todo.push((b, dir.clone()));
        }
        let _dict = match todo.is_empty() {
            true => None,
            false => self.dict(src, &mut missing)?,
        };
        if !missing.is_empty() {
            return Ok(Step::Missing(missing));
        }

        for (b, dir) in todo {
            let e = &file.root.blocks[b];
            match self.match_block(src, pred, e, &dir) {
                Ok(Some(hits)) => {
                    self.stats.blocks += 1;
                    self.matched.insert(b, Rc::new(hits));
                }
                Ok(None) | Err(Error::NotLoaded) => missing.push(whole(e)),
                Err(err) => return Err(err),
            }
        }
        if !missing.is_empty() {
            return Ok(Step::Missing(missing));
        }

        let mut out = RowSet::default();
        for b in file.blocks_in(within) {
            let first = file.starts[b];
            for &(s, e) in self.matched[&b].iter() {
                out.push_range(first + s as u64, first + e as u64);
            }
        }
        Ok(Step::Ready(
            within.map_or(out.clone(), |w| out.intersect(w)),
        ))
    }

    /// The rows of one block that satisfy `pred`, or `None` when that takes the whole block and
    /// the source has only parts of it.
    fn match_block(
        &mut self,
        src: &dyn Source,
        pred: &Predicate,
        e: &BlockEntry,
        dir: &BlockDir,
    ) -> Result<Option<Vec<(u32, u32)>>> {
        let budget = &self.file.budget;
        let mut hits = Vec::<(u32, u32)>::new();
        let mut hit = |r: usize| match hits.last_mut() {
            Some(last) if last.1 as usize == r => last.1 += 1,
            _ => hits.push((r as u32, r as u32 + 1)),
        };
        let (model, kinds) = match e.kind {
            BlockKind::Structured { model, kinds } => (model, kinds),
            BlockKind::Verbatim { .. } => {
                let Some(bytes) = src.get(whole(e)) else {
                    return Ok(None);
                };
                let (raw, spans) = verbatim_records(bytes, e, budget)?;
                self.stats.blocks_whole += 1;
                self.stats.blocks_verbatim += 1;
                self.stats.rebuilt += spans.len();
                for (r, &(s, t)) in spans.iter().enumerate() {
                    if eval_raw_record(&raw[s as usize..t as usize], pred) {
                        hit(r);
                    }
                }
                return Ok(Some(hits));
            }
        };
        let plan = &self.file.plans[model as usize];
        let m = BlockMeta {
            model,
            plan,
            kinds,
            dir,
            filter: Filter::NONE,
        };
        let mut need = Need::new(dir);
        needs(pred, &m, &mut self.plans, &mut need);
        let Some((parts, is_whole)) = load_parts(
            src,
            e,
            dir,
            plan,
            &Wanted::of(dir, &need, plan, kinds),
            budget,
        )?
        else {
            return Ok(None);
        };
        let mut rows = BlockRows::of_parts(parts, e.rows as usize, plan, budget)?;
        let tri = eval(pred, &m, &mut rows, &mut self.plans, budget)?;
        if !is_whole && tri.contains(&UNKNOWN) {
            return Ok(None);
        }
        let mut rec = Vec::new();
        for (r, &t) in tri.iter().enumerate() {
            if t == UNKNOWN {
                rows.seek(r, budget)?;
                rec.clear();
                rows.emit(&mut rec, budget)?;
                self.stats.rebuilt += 1;
            }
            if t == YES || (t == UNKNOWN && eval_raw_record(&rec, pred)) {
                hit(r);
            }
        }
        if is_whole {
            self.stats.blocks_whole += 1;
        } else {
            self.stats.blocks_partial += 1;
        }
        Ok(Some(hits))
    }

    /// Calls `f(row, record)` for each row of `rows`, byte-exact and in row order.
    pub fn records(
        &mut self,
        src: &dyn Source,
        rows: &RowSet,
        mut f: impl FnMut(u64, &[u8]) -> Result<()>,
    ) -> Result<Step<()>> {
        let file = self.file;
        let budget = &file.budget;
        let blocks = file.blocks_in(Some(rows));
        let mut missing: Vec<_> = blocks
            .iter()
            .map(|&b| whole(&file.root.blocks[b]))
            .filter(|r| src.get(*r).is_none())
            .collect();
        let _dict = match blocks.is_empty() {
            true => None,
            false => self.dict(src, &mut missing)?,
        };
        if !missing.is_empty() {
            return Ok(Step::Missing(missing));
        }
        let mut rec = Vec::new();
        for b in blocks {
            let e = &file.root.blocks[b];
            let first = file.starts[b];
            let bytes = src.get(whole(e)).unwrap();
            let sel = rows.within(first, first + e.rows).flat_map(|(s, t)| s..t);
            self.stats.blocks_whole += 1;
            match e.kind {
                BlockKind::Verbatim { .. } => {
                    let (raw, spans) = verbatim_records(bytes, e, budget)?;
                    for r in sel {
                        let (s, t) = spans[r as usize];
                        f(first + r as u64, &raw[s as usize..t as usize])?;
                    }
                }
                BlockKind::Structured { model, .. } => {
                    let plan = &file.plans[model as usize];
                    let mut block = BlockRows::open(bytes, e.rows as usize, plan, budget)?;
                    for r in sel {
                        block.seek(r as usize, budget)?;
                        rec.clear();
                        block.emit(&mut rec, budget)?;
                        f(first + r as u64, &rec)?;
                    }
                }
            }
        }
        Ok(Step::Ready(()))
    }

    /// Collects the values of `paths` on the rows of `rows` in every block that holds some, into
    /// `self.values`.
    fn collect_values(
        &mut self,
        src: &dyn Source,
        paths: &[&str],
        rows: &RowSet,
    ) -> Result<Step<Vec<usize>>> {
        let file = self.file;
        let all = file.blocks_in(Some(rows));
        let blocks: Vec<usize> = all
            .iter()
            .copied()
            .filter(|b| !self.values.contains_key(b))
            .collect();
        let mut missing = Vec::new();
        let mut dirs = Vec::with_capacity(blocks.len());
        for &b in &blocks {
            match self.dir(src, b)? {
                Some(dir) => dirs.push(dir),
                None => missing.push(dir_page(&file.root.blocks[b])),
            }
        }
        if !missing.is_empty() {
            return Ok(Step::Missing(missing));
        }
        for (&b, dir) in blocks.iter().zip(&dirs) {
            let e = &file.root.blocks[b];
            if src.get(whole(e)).is_some() {
                continue;
            }
            let ranges = match e.kind {
                BlockKind::Structured { model, kinds } if kinds & !KIND_JSON == 0 => {
                    self.value_parts(paths, model, kinds, dir).ranges(e, dir)
                }
                _ => vec![whole(e)],
            };
            missing.extend(ranges.into_iter().filter(|r| src.get(*r).is_none()));
        }
        let _dict = match blocks.is_empty() {
            true => None,
            false => self.dict(src, &mut missing)?,
        };
        if !missing.is_empty() {
            return Ok(Step::Missing(missing));
        }
        for (&b, dir) in blocks.iter().zip(&dirs) {
            let e = &file.root.blocks[b];
            let first = file.starts[b];
            let sel = mask(rows.within(first, first + e.rows), e.rows as usize);
            match self.block_values(src, paths, e, dir, &sel) {
                Ok(Some(v)) => {
                    self.values.insert(b, v);
                }
                Ok(None) | Err(Error::NotLoaded) => missing.push(whole(e)),
                Err(err) => return Err(err),
            }
        }
        if !missing.is_empty() {
            return Ok(Step::Missing(missing));
        }
        Ok(Step::Ready(all))
    }

    /// The parts of a block that hold the values of `paths`.
    fn value_parts(&mut self, paths: &[&str], model: u32, kinds: u16, dir: &BlockDir) -> Wanted {
        let m = BlockMeta {
            model,
            plan: &self.file.plans[model as usize],
            kinds,
            dir,
            filter: Filter::NONE,
        };
        let mut need = Need::new(dir);
        for p in paths {
            self.plans.field(&m, p).needs(kinds, dir, &mut need);
        }
        Wanted::of(dir, &need, m.plan, kinds)
    }

    /// The values of `paths` on the rows `sel` of one block, or `None` when some row has to be
    /// rebuilt and the source has only parts of the block.
    fn block_values(
        &mut self,
        src: &dyn Source,
        paths: &[&str],
        e: &BlockEntry,
        dir: &BlockDir,
        sel: &[bool],
    ) -> Result<Option<BlockVals>> {
        let budget = &self.file.budget;
        let mut out = BlockVals {
            arena: Vec::new(),
            vals: vec![Vec::new(); paths.len()],
        };
        let add = |out: &mut BlockVals, p: usize, row: u32, v: &[u8]| {
            let s = out.arena.len() as u32;
            out.arena.extend_from_slice(v);
            out.vals[p].push((row, s, out.arena.len() as u32));
        };
        let (model, kinds) = match e.kind {
            BlockKind::Structured { model, kinds } => (model, kinds),
            BlockKind::Verbatim { .. } => {
                let Some(bytes) = src.get(whole(e)) else {
                    return Ok(None);
                };
                let (raw, spans) = verbatim_records(bytes, e, budget)?;
                for (r, &(s, t)) in spans.iter().enumerate().filter(|(r, _)| sel[*r]) {
                    self.stats.rebuilt += 1;
                    for (p, path) in paths.iter().enumerate() {
                        raw_field_values(&raw[s as usize..t as usize], path, &mut |v| {
                            add(&mut out, p, r as u32, v)
                        });
                    }
                }
                return Ok(Some(out));
            }
        };
        let plan = &self.file.plans[model as usize];
        let want = self.value_parts(paths, model, kinds, dir);
        let Some((parts, is_whole)) = load_parts(src, e, dir, plan, &want, budget)? else {
            return Ok(None);
        };
        let mut rows = BlockRows::of_parts(parts, e.rows as usize, plan, budget)?;
        let m = BlockMeta {
            model,
            plan,
            kinds,
            dir,
            filter: Filter::NONE,
        };
        // Rows the columns cannot answer: those outside the JSON tree, and opaque values.
        let jvi = json_variant(plan);
        let mut rebuild: Vec<bool> = rows
            .variant_ids
            .iter()
            .zip(sel)
            .map(|(&v, &s)| s && Some(v) != jvi)
            .collect();
        for (p, path) in paths.iter().enumerate() {
            let fp = self.plans.field(&m, path);
            field_events(
                &fp,
                kinds,
                dir,
                &mut rows,
                Some(sel),
                budget,
                &mut |row, ev| match ev {
                    FieldEv::Opaque => rebuild[row as usize] = true,
                    FieldEv::Val { lex, .. } => add(&mut out, p, row, lex),
                },
            )?;
        }
        if !rebuild.contains(&true) {
            self.stats.blocks_partial += usize::from(!is_whole);
            self.stats.blocks_whole += usize::from(is_whole);
            return Ok(Some(out));
        }
        if !is_whole {
            return Ok(None);
        }
        self.stats.blocks_whole += 1;
        let mut rec = Vec::new();
        for v in &mut out.vals {
            v.retain(|&(row, _, _)| !rebuild[row as usize]);
        }
        for r in (0..rebuild.len()).filter(|&r| rebuild[r]) {
            rows.seek(r, budget)?;
            rec.clear();
            rows.emit(&mut rec, budget)?;
            self.stats.rebuilt += 1;
            for (p, path) in paths.iter().enumerate() {
                raw_field_values(&rec, path, &mut |v| add(&mut out, p, r as u32, v));
            }
        }
        for v in &mut out.vals {
            v.sort_by_key(|&(row, _, _)| row);
        }
        Ok(Some(out))
    }

    /// Calls `f(row, path, value)` for the rows of `rows` in order and, within a row, for each of
    /// `paths` in order. A value is the field's bytes as the record writes them; a path with
    /// several values on a row gives each in turn, and one with none gives `None` once.
    pub fn fields(
        &mut self,
        src: &dyn Source,
        paths: &[&str],
        rows: &RowSet,
        mut f: impl FnMut(u64, usize, Option<&[u8]>) -> Result<()>,
    ) -> Result<Step<()>> {
        self.begin_values(format!("f{paths:?}{:?}", rows.ranges()));
        let blocks = match self.collect_values(src, paths, rows)? {
            Step::Ready(b) => b,
            Step::Missing(m) => return Ok(Step::Missing(m)),
        };
        for b in blocks {
            let e = &self.file.root.blocks[b];
            let first = self.file.starts[b];
            let v = &self.values[&b];
            let mut at = vec![0usize; paths.len()];
            for r in rows.within(first, first + e.rows).flat_map(|(s, t)| s..t) {
                for (p, vals) in v.vals.iter().enumerate() {
                    let from = at[p];
                    while vals.get(at[p]).is_some_and(|x| x.0 == r) {
                        let (_, s, t) = vals[at[p]];
                        f(first + r as u64, p, Some(&v.arena[s as usize..t as usize]))?;
                        at[p] += 1;
                    }
                    if at[p] == from {
                        f(first + r as u64, p, None)?;
                    }
                }
            }
        }
        self.values.clear();
        Ok(Step::Ready(()))
    }

    /// Calls `f(value, rows)` for each distinct value of `path` in each block that holds rows of
    /// `within`, with the rows of `within` in that block that have it. A value that occurs in
    /// several blocks is reported once per block.
    pub fn distinct(
        &mut self,
        src: &dyn Source,
        path: &str,
        within: &RowSet,
        mut f: impl FnMut(&[u8], &[u64]) -> Result<()>,
    ) -> Result<Step<()>> {
        self.begin_values(format!("d{path:?}{:?}", within.ranges()));
        let blocks = match self.collect_values(src, &[path], within)? {
            Step::Ready(b) => b,
            Step::Missing(m) => return Ok(Step::Missing(m)),
        };
        for b in blocks {
            let first = self.file.starts[b];
            let v = &self.values[&b];
            let mut groups = Vec::<(&[u8], Vec<u64>)>::new();
            let mut by_value = HashMap::<&[u8], usize>::new();
            for &(row, s, t) in &v.vals[0] {
                let value = &v.arena[s as usize..t as usize];
                let g = *by_value.entry(value).or_insert_with(|| {
                    groups.push((value, Vec::new()));
                    groups.len() - 1
                });
                let row = first + row as u64;
                if groups[g].1.last() != Some(&row) {
                    groups[g].1.push(row);
                }
            }
            for (value, rows) in groups {
                f(value, &rows)?;
            }
        }
        self.values.clear();
        Ok(Step::Ready(()))
    }
}

// ============================================================================
// Writing
// ============================================================================

/// Writes a file from records in their final order. The data region goes to `out` block by
/// block; `finish` returns the index region.
pub struct Writer<W: Write> {
    enc: StreamEncoder,
    out: W,
    /// The open block: each record behind its length.
    buf: Vec<u8>,
    buf_rows: usize,
    buf_bytes: usize,
    block_rows: usize,
    block_bytes: usize,
    rows: u64,
}

impl<W: Write> Writer<W> {
    /// `opts.framing`, `opts.chunk_bytes` and `opts.sync_interval_frames` are not used: records
    /// arrive already split, a frame is one block, and a model is written when it changes.
    pub fn new(out: W, mut opts: PackOptions) -> Self {
        opts.framing = FramingSpec::LengthU32Le;
        opts.index = true;
        opts.sync_interval_frames = usize::MAX;
        let (block_rows, block_bytes) = (opts.block_rows.max(1), opts.block_bytes.max(1));
        Self {
            enc: StreamEncoder::new(opts),
            out,
            buf: Vec::new(),
            buf_rows: 0,
            buf_bytes: 0,
            block_rows,
            block_bytes,
            rows: 0,
        }
    }

    /// Rows pushed so far; the next record's row number.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Bytes of the data region written so far.
    pub fn data_len(&self) -> u64 {
        self.enc.data_len()
    }

    /// The sink, which holds the data region's bytes in order. A host that buffers them takes
    /// them from here between calls.
    pub fn sink_mut(&mut self) -> &mut W {
        &mut self.out
    }

    /// The model blocks are being written with, once there is one.
    pub fn model(&self) -> Option<&ModelPlan> {
        self.enc.active_plan()
    }

    /// Adds a record. Returns the fit of a block this completed.
    pub fn push(&mut self, rec: &[u8]) -> Result<Option<FrameFit>> {
        if rec.len() > u32::MAX as usize {
            return Err(Error::InvalidArgument("record longer than 4 GiB".into()));
        }
        let full = self.buf_rows == self.block_rows
            || (self.buf_rows > 0 && self.buf_bytes + rec.len() > self.block_bytes);
        let fit = if full { self.cut()? } else { None };
        self.buf
            .extend_from_slice(&(rec.len() as u32).to_le_bytes());
        self.buf.extend_from_slice(rec);
        self.buf_rows += 1;
        self.buf_bytes += rec.len();
        self.rows += 1;
        Ok(fit)
    }

    /// Ends the open block, if it has rows, and returns its fit.
    pub fn cut(&mut self) -> Result<Option<FrameFit>> {
        if self.buf_rows == 0 {
            return Ok(None);
        }
        let fit = self.enc.encode_frame(&self.buf, &mut self.out)?;
        self.buf.clear();
        (self.buf_rows, self.buf_bytes) = (0, 0);
        Ok(Some(fit))
    }

    /// Ends the file: returns the sink and the index region.
    pub fn finish(mut self) -> Result<(W, Vec<u8>)> {
        self.cut()?;
        let index = self.enc.finish_index();
        Ok((self.out, index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_sets_combine() {
        let a = RowSet::from_sorted([1, 2, 3, 7, 9, 10]);
        assert_eq!(a.ranges(), &[(1, 4), (7, 8), (9, 11)]);
        assert_eq!(
            (a.len(), a.contains(3), a.contains(4), a.contains(10)),
            (6, true, false, true)
        );
        let b = RowSet::range(3, 10);
        assert_eq!(a.intersect(&b).ranges(), &[(3, 4), (7, 8), (9, 10)]);
        assert_eq!(a.union(&b).ranges(), &[(1, 11)]);
        assert_eq!(
            a.within(2, 10).collect::<Vec<_>>(),
            [(0, 2), (5, 6), (7, 8)]
        );
        assert!(a.intersect(&RowSet::default()).is_empty());
    }

    #[test]
    fn range_cache_serves_what_it_holds() {
        let r = |off, len| ByteRange {
            region: Region::Data,
            off,
            len,
        };
        let mut c = RangeCache::default();
        c.insert(r(200, 50), vec![1; 50]);
        c.insert(r(500, 10), vec![3; 10]);
        assert_eq!(c.gaps(r(100, 450)), [r(100, 100), r(250, 250), r(510, 40)]);
        assert_eq!(c.gaps(r(210, 30)), []);
        c.insert(r(100, 300), (0..300).map(|i| i as u8).collect());
        c.insert(r(350, 100), vec![2; 100]);
        assert_eq!(
            c.get(r(210, 5)),
            Some(&[110u8, 111, 112, 113, 114][..]),
            "the wider range replaced it"
        );
        assert_eq!(
            c.get(r(398, 4)),
            Some(&[2u8, 2, 2, 2][..]),
            "the later range overwrote the overlap"
        );
        assert_eq!(
            c.get(r(100, 350)).map(|b| b.len()),
            Some(350),
            "ranges that touch are one"
        );
        assert_eq!(c.get(r(440, 20)), None, "past what is held");
        assert_eq!(c.gaps(r(440, 80)), [r(450, 50), r(510, 10)]);
        c.insert(r(450, 50), vec![4; 50]);
        assert_eq!(
            c.get(r(100, 410)).map(|b| (b.len(), b[409])),
            Some((410, 3))
        );
        assert_eq!(c.len(), 410);
        assert_eq!(
            c.get(ByteRange {
                region: Region::Index,
                off: 100,
                len: 1
            }),
            None
        );
        let merged = coalesce(vec![r(10, 5), r(0, 10), r(20, 1), r(12, 2)]);
        assert_eq!(merged, [r(0, 15), r(20, 1)]);
    }
}
