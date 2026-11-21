//! The index region: a root, and for every block a directory of its parts and a substring filter.
//! All of it is derived from the data region.

use crate::codecs::{ColDeps, ColSummary};
use crate::framing::ResolvedFraming;
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, compress_leaf, crc32, decompress_leaf,
    put_bytes, put_ivarint, put_uvarint,
};

pub const INDEX_MAGIC: &[u8; 4] = b"ZFIX";
/// Its last byte counts the changes that would make a reader misread an older region; this one
/// is to what the filters hold.
pub const INDEX_END_MAGIC: &[u8; 4] = b"ZFI2";
/// `ZFIX` and the body length that follows it.
pub const INDEX_HEAD_LEN: usize = 12;
/// Root CRC32, root length, region length, `ZFI2`.
pub const INDEX_TAIL_LEN: usize = 24;

// What a block's rows are made of. A reader plans from these before it has any of the block.
/// Variant 0, the record verbatim.
pub const KIND_RESIDUAL: u16 = 1 << 0;
pub const KIND_JSON: u16 = 1 << 1;
/// Any other `Shape` variant.
pub const KIND_SHAPE: u16 = 1 << 2;
pub const KIND_SHAPE_CELL: u16 = 1 << 3;

// ============================================================================
// Substring filter
// ============================================================================

// A longer run is known by its grams of five letters. Measured on 8 MiB blocks of GitHub
// events, about 34,000 distinct tokens each, with needles taken from them: most blocks hold
// every gram of four letters a needle has, and a gram of six to eight cannot test a shorter
// needle, which then reads every block. Each token sets two bits, which at these sizes passes
// fewer blocks than one.
const WORD_MIN: usize = 3;
const GRAM: usize = 5;

/// Every bit of the result depends on every bit of `h`: a token's two filter bits come from its
/// two halves.
#[inline]
fn mix(h: u64) -> u64 {
    let h = (h ^ (h >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    let h = (h ^ (h >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

/// Calls `f(token, complete)` for the tokens of `text`: each run of ASCII letters of 3 or 4
/// letters, and every 5 consecutive letters of a longer run, case kept. `complete` is false for a
/// run that touches an end of `text`, where the text it was cut from may continue the run.
pub fn tokens(text: &[u8], mut f: impl FnMut(u64, bool)) {
    spellings(text, |w, complete| f(mix(w), complete))
}

/// `tokens`, with each token as the letters it stands for, read as one little-endian word.
#[inline]
fn spellings(text: &[u8], mut f: impl FnMut(u64, bool)) {
    let n = text.len();
    let mut run = |start: usize, end: usize| {
        let run = &text[start..end];
        if run.len() < WORD_MIN {
            return;
        }
        let (head, rest) = run.split_at(run.len().min(GRAM));
        let mut w = head.iter().rev().fold(0u64, |w, &l| (w << 8) | l as u64);
        f(w, run.len() >= GRAM || (start > 0 && end < n));
        for &l in rest {
            w = (w >> 8) | ((l as u64) << (8 * GRAM - 8));
            f(w, true);
        }
    };
    // Eight bytes at a time: a byte's top bit marks a letter, and a run's ends are where the
    // mark changes. Bytes past the text read as zero, which is not a letter.
    const HI: u64 = 0x8080_8080_8080_8080;
    let mut open = None;
    let mut scan = |at: usize, w: u64| {
        let x = w | 0x2020_2020_2020_2020;
        let y = x & !HI;
        let m = (y + 0x1f1f_1f1f_1f1f_1f1f) & !(y + 0x0505_0505_0505_0505) & !x & HI;
        let mut edges = m ^ ((m << 8) | if open.is_some() { 0x80 } else { 0 });
        while edges != 0 {
            let p = at + edges.trailing_zeros() as usize / 8;
            edges &= edges - 1;
            match open.take() {
                Some(s) => run(s, p),
                None => open = Some(p),
            }
        }
    };
    let chunks = text.chunks_exact(8);
    let tail = chunks.remainder();
    for (at, c) in (0..).step_by(8).zip(chunks) {
        scan(at, u64::from_le_bytes(c.try_into().unwrap()));
    }
    let mut b = [0u8; 8];
    b[..tail.len()].copy_from_slice(tail);
    scan(n - tail.len(), u64::from_le_bytes(b));
    if let Some(s) = open {
        run(s, n);
    }
}

/// The tokens every text containing `needle` must have, or `None` when there are none and the
/// needle cannot be pruned. `whole` says the needle is the entire text, so its end runs are words.
pub fn needle_tokens(needle: &[u8], whole: bool) -> Option<Vec<u64>> {
    let mut out = Vec::new();
    tokens(needle, |t, complete| {
        // A gram is always complete; a word is only when the needle bounds it on both sides.
        if complete || whole {
            out.push(t);
        }
    });
    out.sort_unstable();
    out.dedup();
    (!out.is_empty()).then_some(out)
}

pub struct FilterBuilder {
    bits: Vec<u64>,
    /// Spellings `add_text` has added, each where its hash puts it. Records repeat their words,
    /// and most tokens of a text are found here.
    recent: Box<[u64; RECENT]>,
}

const RECENT: usize = 1 << 12;

impl FilterBuilder {
    /// `max_bits` is rounded up to a power of two of at least 64.
    pub fn new(max_bits: usize) -> Self {
        let n = max_bits.max(64).next_power_of_two();
        Self {
            bits: vec![0u64; n / 64],
            recent: Box::new([0; RECENT]),
        }
    }

    /// Adds the tokens of `text`.
    pub fn add_text(&mut self, text: &[u8]) {
        spellings(text, |w, _| {
            let slot =
                (w.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - RECENT.trailing_zeros())) as usize;
            if self.recent[slot] != w {
                self.recent[slot] = w;
                self.add(mix(w));
            }
        })
    }

    #[inline]
    pub fn add(&mut self, token: u64) {
        // A power-of-two word count masks the index, which also spares its bounds check.
        let words = self.bits.len() - 1;
        for p in [token, token >> 32] {
            self.bits[(p >> 6) as usize & words] |= 1u64 << (p & 63);
        }
    }

    /// The filter's bytes, halved while under three tenths of its bits are set.
    pub fn finish(mut self) -> Vec<u8> {
        while self.bits.len() > 1 {
            let set: u32 = self.bits.iter().map(|w| w.count_ones()).sum();
            if set as usize * 10 >= self.bits.len() * 64 * 3 {
                break;
            }
            let half = self.bits.len() / 2;
            for i in 0..half {
                self.bits[i] |= self.bits[half + i];
            }
            self.bits.truncate(half);
        }
        self.bits.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A pool page's filter: halved while at most `PAGE_FILL` of the half's bits would be set.
    pub fn finish_page(mut self) -> Vec<u8> {
        while self.bits.len() > 1 {
            let half = self.bits.len() / 2;
            let (lo, hi) = self.bits.split_at(half);
            let set: u32 = lo.iter().zip(hi).map(|(a, b)| (a | b).count_ones()).sum();
            if set as usize * 1000 > PAGE_FILL * half * 64 {
                break;
            }
            for i in 0..half {
                self.bits[i] |= self.bits[half + i];
            }
            self.bits.truncate(half);
        }
        self.bits.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

/// Most pages a text pool is cut into, so a row page map value fits a byte.
pub const MAX_PAGES: usize = 255;

/// Most bits a page filter has set, per mille.
const PAGE_FILL: usize = 780;

/// A paged pool's index page: each page filter's size as `log2` of its bytes, the row page map,
/// then the filters back to back. The map gives row `r` the value `page + 1`, or `0` for a row
/// with no text in the pool, in `row_map_width` bits from bit `r * width`, least significant
/// first.
pub fn page_filter_page(filters: &[Vec<u8>], row_pages: &[u32]) -> Vec<u8> {
    let mut out: Vec<u8> = filters
        .iter()
        .map(|f| f.len().trailing_zeros() as u8)
        .collect();
    let w = row_map_width(filters.len());
    let mut map = vec![0u8; row_pages.len() * w / 8 + 1];
    for (r, &v) in row_pages.iter().enumerate() {
        for k in (0..w).filter(|k| v >> k & 1 != 0) {
            map[(r * w + k) / 8] |= 1 << ((r * w + k) % 8);
        }
    }
    map.truncate((row_pages.len() * w).div_ceil(8));
    out.extend_from_slice(&map);
    filters.iter().for_each(|f| out.extend_from_slice(f));
    out
}

/// Bits per row in the row page map of a pool of `n_pages` pages.
pub fn row_map_width(n_pages: usize) -> usize {
    (usize::BITS - n_pages.leading_zeros()) as usize
}

/// Where a block's row page map lies in its page filter page, as `(offset, length)`.
pub fn row_map_span(n_pages: usize, rows: u64) -> (u64, u64) {
    (
        n_pages as u64,
        rows.saturating_mul(row_map_width(n_pages) as u64)
            .div_ceil(8),
    )
}

/// The pool page row `r` of a row page map reads, if any.
pub fn row_page(map: &[u8], n_pages: usize, r: usize) -> Result<Option<usize>> {
    let w = row_map_width(n_pages);
    let mut v = 0usize;
    for k in 0..w {
        let bit = r * w + k;
        let byte = map
            .get(bit / 8)
            .ok_or_else(|| Error::Format("row page map".into()))?;
        v |= ((byte >> (bit % 8)) as usize & 1) << k;
    }
    match v {
        0 => Ok(None),
        v if v <= n_pages => Ok(Some(v - 1)),
        _ => Err(Error::Format("row page map value".into())),
    }
}

/// The `n` filters of a page filter page for a block of `rows` rows.
pub fn page_filters(page: &[u8], n: usize, rows: u64) -> Result<Vec<Filter<'_>>> {
    let bad = || Error::Format("page filter page".into());
    let (sizes, rest) = page.split_at_checked(n).ok_or_else(bad)?;
    let map_len = usize::try_from(row_map_span(n, rows).1).map_err(|_| bad())?;
    let (_, mut rest) = rest.split_at_checked(map_len).ok_or_else(bad)?;
    let mut out = Vec::with_capacity(n);
    for &log in sizes {
        let len = 1usize
            .checked_shl(log as u32)
            .filter(|&l| l >= 8)
            .ok_or_else(bad)?;
        let (f, more) = rest.split_at_checked(len).ok_or_else(bad)?;
        out.push(Filter(f));
        rest = more;
    }
    if !rest.is_empty() {
        return Err(bad());
    }
    Ok(out)
}

/// A block's filter as stored: a power-of-two number of 8-byte words. One with no bytes is a
/// filter not at hand, which holds every token.
#[derive(Clone, Copy)]
pub struct Filter<'a>(pub &'a [u8]);

impl Filter<'_> {
    pub const NONE: Filter<'static> = Filter(&[]);

    pub fn valid(&self) -> bool {
        self.0.len() >= 8 && self.0.len().is_power_of_two()
    }

    #[inline]
    pub fn has(&self, token: u64) -> bool {
        let mask = (self.0.len() as u64 * 8).wrapping_sub(1);
        self.0.is_empty()
            || [token & mask, (token >> 32) & mask]
                .iter()
                .all(|p| self.0[(p >> 3) as usize] & (1u8 << (p & 7)) != 0)
    }
}

// ============================================================================
// Block directory
// ============================================================================

/// In a directory, in place of a column's flags: a count of empty one-byte columns follows.
const EMPTY_RUN: u8 = 0x80;

/// How long one column's bytes are in its block and what they hold.
#[derive(Debug, Clone, Default)]
pub struct ColEntry {
    pub len: u32,
    pub summary: ColSummary,
    pub deps: ColDeps,
}

/// The parts of one block, as byte ranges from the block's start.
#[derive(Debug, Clone, Default)]
pub struct BlockDir {
    /// Length of the row structure, which leads the block: the variant ids up to `struct_head`,
    /// then the structure streams back to back, each of the length `streams` gives.
    pub struct_len: u32,
    pub struct_head: u32,
    pub streams: Vec<u32>,
    pub dicts: Vec<(u32, u32)>,
    pub pool: Option<(u32, u32)>,
    /// A paged pool's pages as `(compressed, raw)` lengths, lying back to back over `pool`.
    /// Empty for a pool of one leaf.
    pub pages: Vec<(u32, u32)>,
    /// Where the first column starts. The columns lie back to back.
    cols_off: u32,
    /// The columns some row has a value in, by number. A wide model leaves most of a block's
    /// columns empty, and a reader should not pay for those.
    listed: Vec<(u16, ColEntry)>,
    /// By column: its place in `listed` plus one, or zero for an empty column, which is one byte.
    slot: Vec<u16>,
}

static EMPTY: ColEntry = ColEntry {
    len: 1,
    summary: ColSummary {
        numeric_kind: 0,
        min_val: 0,
        max_val: 0,
        all_null: true,
        per_row: false,
        letters: 0,
    },
    deps: ColDeps {
        pool: false,
        dict: None,
        cols: Vec::new(),
    },
};

fn is_empty(c: &ColEntry) -> bool {
    c.summary.all_null && c.len == 1 && c.summary.numeric_kind == 0 && c.deps == ColDeps::default()
}

impl BlockDir {
    /// Each structure stream's place in the block, as `(offset, length)`.
    pub fn stream_spans(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.streams.iter().scan(self.struct_head, |at, &len| {
            let off = *at;
            *at += len;
            Some((off, len))
        })
    }

    /// Sets the block's columns, which start at `off` and follow one another.
    pub fn set_cols(&mut self, off: u32, cols: Vec<ColEntry>) {
        assert!(
            cols.len() < 1 << 16,
            "a model numbers its columns in 16 bits"
        );
        self.cols_off = off;
        self.slot = vec![0; cols.len()];
        self.listed.clear();
        for (c, e) in cols.into_iter().enumerate().filter(|(_, e)| !is_empty(e)) {
            self.listed.push((c as u16, e));
            self.slot[c] = self.listed.len() as u16;
        }
    }

    pub fn n_cols(&self) -> usize {
        self.slot.len()
    }

    pub fn col(&self, c: usize) -> &ColEntry {
        match self.slot[c] {
            0 => &EMPTY,
            s => &self.listed[s as usize - 1].1,
        }
    }

    /// The columns some row has a value in, with their numbers.
    pub fn cols(&self) -> impl Iterator<Item = (usize, &ColEntry)> + '_ {
        self.listed.iter().map(|(c, e)| (*c as usize, e))
    }

    /// Every column's place in the block, as `(offset, length)`.
    pub fn col_spans(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.slot.iter().scan(self.cols_off, |at, &s| {
            let off = *at;
            *at += if s == 0 {
                1
            } else {
                self.listed[s as usize - 1].1.len
            };
            Some((off, *at - off))
        })
    }

    pub fn encode(&self, backend: EntropyBackend, level: i32, out: &mut Vec<u8>) {
        let mut raw = Vec::with_capacity(self.listed.len() * 8 + self.streams.len() + 32);
        put_uvarint(&mut raw, self.struct_len as u64);
        put_uvarint(&mut raw, self.struct_head as u64);
        put_uvarint(&mut raw, self.streams.len() as u64);
        for &len in &self.streams {
            put_uvarint(&mut raw, len as u64);
        }
        put_uvarint(&mut raw, self.dicts.len() as u64);
        for &(off, len) in &self.dicts {
            put_uvarint(&mut raw, off as u64);
            put_uvarint(&mut raw, len as u64);
        }
        let (pool_off, pool_len) = self.pool.unwrap_or((0, 0));
        put_uvarint(&mut raw, pool_len as u64);
        if pool_len > 0 {
            put_uvarint(&mut raw, pool_off as u64);
            put_uvarint(&mut raw, self.pages.len() as u64);
            for &(comp, len) in &self.pages {
                put_uvarint(&mut raw, comp as u64);
                put_uvarint(&mut raw, len as u64);
            }
        }
        put_uvarint(&mut raw, self.slot.len() as u64);
        put_uvarint(&mut raw, self.cols_off as u64);
        let empty_run = |raw: &mut Vec<u8>, from: usize, to: usize| {
            if to > from {
                raw.push(EMPTY_RUN);
                put_uvarint(raw, (to - from) as u64);
            }
        };
        let mut next = 0usize;
        for (id, c) in self.cols() {
            empty_run(&mut raw, next, id);
            next = id + 1;
            let s = &c.summary;
            let flags = u8::from(!s.all_null)
                | (s.numeric_kind & 3) << 1
                | u8::from(c.deps.pool) << 3
                | u8::from(c.deps.dict.is_some()) << 4
                | u8::from(!c.deps.cols.is_empty()) << 5
                | u8::from(s.per_row) << 6;
            raw.push(flags);
            put_uvarint(&mut raw, c.len as u64);
            if !s.all_null {
                put_uvarint(&mut raw, s.letters as u64);
            }
            if s.numeric_kind != 0 {
                put_ivarint(&mut raw, s.min_val);
                put_ivarint(&mut raw, s.max_val);
            }
            if let Some(d) = c.deps.dict {
                put_uvarint(&mut raw, d as u64);
            }
            if !c.deps.cols.is_empty() {
                put_uvarint(&mut raw, c.deps.cols.len() as u64);
                for &s in &c.deps.cols {
                    put_uvarint(&mut raw, s as u64);
                }
            }
        }
        empty_run(&mut raw, next, self.slot.len());
        compress_leaf(out, &raw, backend, level);
    }

    pub fn decode(page: &[u8], budget: &DecodeBudget) -> Result<Self> {
        let raw = decompress_leaf(&mut Cursor::new(page), budget)?;
        let mut cur = Cursor::new(&raw);
        let u32v = |cur: &mut Cursor<'_>| -> Result<u32> {
            u32::try_from(cur.uvar()?).map_err(|_| Error::Format("block directory offset".into()))
        };
        let struct_len = u32v(&mut cur)?;
        let struct_head = u32v(&mut cur)?;
        let n_streams = cur.uvar()? as usize;
        if n_streams > raw.len() {
            return Err(Error::Format("block directory stream count".into()));
        }
        let mut streams = Vec::with_capacity(n_streams);
        let mut end = struct_head as u64;
        for _ in 0..n_streams {
            streams.push(u32v(&mut cur)?);
            end += *streams.last().unwrap() as u64;
        }
        if end > struct_len as u64 {
            return Err(Error::Format(
                "block directory streams past the structure".into(),
            ));
        }
        let n_dicts = cur.uvar()? as usize;
        if n_dicts > 4096 {
            return Err(Error::Format("too many block shared dicts".into()));
        }
        let mut dicts = Vec::with_capacity(n_dicts);
        for _ in 0..n_dicts {
            dicts.push((u32v(&mut cur)?, u32v(&mut cur)?));
        }
        let pool_len = u32v(&mut cur)?;
        let pool = if pool_len > 0 {
            Some((u32v(&mut cur)?, pool_len))
        } else {
            None
        };
        let n_pages = if pool.is_some() {
            cur.uvar()? as usize
        } else {
            0
        };
        if n_pages == 1 || n_pages > cur.remaining().min(MAX_PAGES) {
            return Err(Error::Format("block directory page count".into()));
        }
        let mut pages = Vec::with_capacity(n_pages);
        let (mut comp, mut raw_len) = (0u64, 0u64);
        for _ in 0..n_pages {
            pages.push((u32v(&mut cur)?, u32v(&mut cur)?));
            comp += pages.last().unwrap().0 as u64;
            raw_len += pages.last().unwrap().1 as u64;
        }
        let over = raw_len > budget.max_single_alloc_bytes as u64;
        if n_pages > 0 && (comp != pool_len as u64 || over) {
            return Err(Error::Format(
                "block directory pages do not cover the pool".into(),
            ));
        }
        // A model numbers its columns in 16 bits.
        let n_cols = cur.uvar()? as usize;
        if n_cols >= 1 << 16 {
            return Err(Error::Format("block directory column count".into()));
        }
        let cols_off = u32v(&mut cur)?;
        let mut end = cols_off;
        let mut slot = vec![0u16; n_cols];
        let mut listed = Vec::new();
        let overflow = || Error::Format("block directory overflow".into());
        let mut c = 0usize;
        while c < n_cols {
            let flags = cur.u8()?;
            if flags == EMPTY_RUN {
                let run = u32v(&mut cur)?;
                if run as usize > n_cols - c {
                    return Err(Error::Format("block directory column run".into()));
                }
                end = end.checked_add(run).ok_or_else(overflow)?;
                c += run as usize;
                continue;
            }
            let len = u32v(&mut cur)?;
            let mut summary = ColSummary {
                all_null: flags & 1 == 0,
                numeric_kind: (flags >> 1) & 3,
                per_row: flags & 0x40 != 0,
                ..ColSummary::default()
            };
            if !summary.all_null {
                summary.letters = u32v(&mut cur)?;
            }
            if summary.numeric_kind != 0 {
                summary.min_val = cur.ivar()?;
                summary.max_val = cur.ivar()?;
            }
            let mut deps = ColDeps {
                pool: flags & 8 != 0,
                ..ColDeps::default()
            };
            if flags & 16 != 0 {
                deps.dict = Some(u32v(&mut cur)?);
            }
            if flags & 32 != 0 {
                for _ in 0..cur.uvar()?.min(8) {
                    deps.cols.push(u32v(&mut cur)?);
                }
            }
            end = end.checked_add(len).ok_or_else(overflow)?;
            listed.push((c as u16, ColEntry { len, summary, deps }));
            c += 1;
            slot[c - 1] = listed.len() as u16;
        }
        Ok(Self {
            struct_len,
            struct_head,
            streams,
            dicts,
            pool,
            pages,
            cols_off,
            listed,
            slot,
        })
    }
}

// ============================================================================
// Root
// ============================================================================

#[derive(Debug, Clone)]
pub enum BlockKind {
    /// Columns under `models[model]`.
    Structured { model: u32, kinds: u16 },
    /// One compressed leaf holding the records as they were framed.
    Verbatim { framing: ResolvedFraming },
    /// Columns under `models[model]` whose rows the reader sorts back into order; read whole.
    Sorted { model: u32 },
}

#[derive(Debug, Clone)]
pub struct BlockEntry {
    pub kind: BlockKind,
    /// The block's bytes in the data region.
    pub off: u64,
    pub len: u64,
    pub rows: u64,
    /// The filter page and the directory page (empty for a verbatim block) in the index region.
    pub filter_off: u64,
    pub filter_len: u32,
    /// The page filter page, empty unless the block's text pool is paged.
    pub pages_off: u64,
    pub pages_len: u32,
    pub dir_off: u64,
    pub dir_len: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Root {
    pub data_len: u64,
    /// Encoded models, in the order blocks refer to them.
    pub models: Vec<Vec<u8>>,
    pub blocks: Vec<BlockEntry>,
    pub dict: Option<DictEntry>,
}

/// Where the data region holds the leaf of the dictionary its blocks were written with.
#[derive(Debug, Clone, Copy)]
pub struct DictEntry {
    pub off: u64,
    pub len: u64,
    /// The XXH64 of the dictionary's content.
    pub id: u64,
}

/// Collects blocks as they are written and lays out the index region: the page filter pages,
/// the filter pages, the directory pages, the root. The directories and the root, which every
/// query reads, are its end.
#[derive(Default)]
pub struct IndexBuilder {
    root: Root,
    filters: Vec<u8>,
    page_filters: Vec<u8>,
    dirs: Vec<u8>,
}

impl IndexBuilder {
    /// The index of `model`, added if no earlier block used the same bytes.
    pub fn model(&mut self, encoded: &[u8]) -> u32 {
        if let Some(i) = self.root.models.iter().rposition(|m| m == encoded) {
            return i as u32;
        }
        self.root.models.push(encoded.to_vec());
        self.root.models.len() as u32 - 1
    }

    pub fn block(
        &mut self,
        kind: BlockKind,
        off: u64,
        len: u64,
        rows: u64,
        dir: &[u8],
        filter: &[u8],
        pages: &[u8],
    ) {
        // The page offsets are the reader's to work out from the lengths.
        self.filters.extend_from_slice(filter);
        self.page_filters.extend_from_slice(pages);
        self.dirs.extend_from_slice(dir);
        self.root.blocks.push(BlockEntry {
            kind,
            off,
            len,
            rows,
            filter_off: 0,
            filter_len: filter.len() as u32,
            pages_off: 0,
            pages_len: pages.len() as u32,
            dir_off: 0,
            dir_len: dir.len() as u32,
        });
    }

    pub fn dictionary(&mut self, off: u64, len: u64, id: u64) {
        self.root.dict = Some(DictEntry { off, len, id });
    }

    pub fn is_empty(&self) -> bool {
        self.root.blocks.is_empty()
    }

    /// The index region for a data region of `data_len` bytes.
    pub fn finish(mut self, data_len: u64, backend: EntropyBackend, level: i32) -> Vec<u8> {
        self.root.data_len = data_len;
        let mut raw = Vec::new();
        put_uvarint(&mut raw, data_len);
        put_uvarint(&mut raw, self.root.models.len() as u64);
        for m in &self.root.models {
            put_bytes(&mut raw, m);
        }
        put_uvarint(&mut raw, self.root.blocks.len() as u64);
        let mut end = 0u64;
        for b in &self.root.blocks {
            match &b.kind {
                BlockKind::Structured { model, kinds } => {
                    raw.push(if b.pages_len > 0 { 2 } else { 0 });
                    put_uvarint(&mut raw, *model as u64);
                    put_uvarint(&mut raw, *kinds as u64);
                }
                BlockKind::Verbatim { framing } => {
                    raw.push(1);
                    framing.encode(&mut raw);
                }
                BlockKind::Sorted { model } => {
                    raw.push(3);
                    put_uvarint(&mut raw, *model as u64);
                }
            }
            put_uvarint(&mut raw, b.off - end);
            put_uvarint(&mut raw, b.len);
            put_uvarint(&mut raw, b.rows);
            put_uvarint(&mut raw, b.dir_len as u64);
            put_uvarint(&mut raw, b.filter_len as u64);
            if b.pages_len > 0 {
                put_uvarint(&mut raw, b.pages_len as u64);
            }
            end = b.off + b.len;
        }
        put_uvarint(&mut raw, self.root.dict.map_or(0, |d| d.len));
        if let Some(d) = self.root.dict {
            put_uvarint(&mut raw, d.off);
            raw.extend_from_slice(&d.id.to_le_bytes());
        }
        let mut root = Vec::new();
        compress_leaf(&mut root, &raw, backend, level.max(3));

        let body_len = self.filters.len()
            + self.page_filters.len()
            + self.dirs.len()
            + root.len()
            + INDEX_TAIL_LEN;
        let mut out = Vec::with_capacity(INDEX_HEAD_LEN + body_len);
        out.extend_from_slice(INDEX_MAGIC);
        out.extend_from_slice(&(body_len as u64).to_le_bytes());
        out.extend_from_slice(&self.page_filters);
        out.extend_from_slice(&self.filters);
        out.extend_from_slice(&self.dirs);
        out.extend_from_slice(&root);
        out.extend_from_slice(&crc32(&root).to_le_bytes());
        out.extend_from_slice(&(root.len() as u64).to_le_bytes());
        out.extend_from_slice(&((INDEX_HEAD_LEN + body_len) as u64).to_le_bytes());
        out.extend_from_slice(INDEX_END_MAGIC);
        out
    }
}

/// The root's place in an index region, read from the region's last `INDEX_TAIL_LEN` bytes:
/// `(offset, length, crc32)`.
pub fn locate_root(tail: &[u8], region_len: u64) -> Result<(u64, u64, u32)> {
    let bad = || Error::Format("not a zform index region".into());
    let t: &[u8; INDEX_TAIL_LEN] = tail.try_into().map_err(|_| bad())?;
    if &t[20..24] != INDEX_END_MAGIC
        || u64::from_le_bytes(t[12..20].try_into().unwrap()) != region_len
    {
        return Err(bad());
    }
    let root_len = u64::from_le_bytes(t[4..12].try_into().unwrap());
    let end = region_len - INDEX_TAIL_LEN as u64;
    let off = end
        .checked_sub(root_len)
        .filter(|&o| o >= INDEX_HEAD_LEN as u64)
        .ok_or_else(bad)?;
    Ok((
        off,
        root_len,
        u32::from_le_bytes(t[0..4].try_into().unwrap()),
    ))
}

impl Root {
    pub fn decode(root: &[u8], crc: u32, budget: &DecodeBudget) -> Result<Self> {
        if budget.verify_checksums && crc32(root) != crc {
            return Err(Error::ChecksumMismatch("index root CRC32"));
        }
        let raw = decompress_leaf(&mut Cursor::new(root), budget)?;
        let mut cur = Cursor::new(&raw);
        let data_len = cur.uvar()?;
        let n_models = cur.uvar()? as usize;
        let mut models = Vec::with_capacity(n_models.min(1024));
        for _ in 0..n_models {
            models.push(cur.bytes()?.to_vec());
        }
        let n_blocks = cur.uvar()? as usize;
        if n_blocks > raw.len() {
            return Err(Error::Format("index root block count".into()));
        }
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut end = 0u64;
        let mut pages_off = INDEX_HEAD_LEN as u64;
        for _ in 0..n_blocks {
            let tag = cur.u8()?;
            let kind = match tag {
                0 | 2 => {
                    let model = cur.uvar()? as u32;
                    if model as usize >= models.len() {
                        return Err(Error::Format("index root model reference".into()));
                    }
                    BlockKind::Structured {
                        model,
                        kinds: cur.uvar()? as u16,
                    }
                }
                1 => BlockKind::Verbatim {
                    framing: ResolvedFraming::decode(&mut cur)?,
                },
                3 => {
                    let model = cur.uvar()? as u32;
                    if model as usize >= models.len() {
                        return Err(Error::Format("index root model reference".into()));
                    }
                    BlockKind::Sorted { model }
                }
                k => return Err(Error::Format(format!("unknown index block kind {k}"))),
            };
            let off = end.checked_add(cur.uvar()?);
            let len = cur.uvar()?;
            let (Some(off), Some(new_end)) = (off, off.and_then(|o| o.checked_add(len))) else {
                return Err(Error::Format("index root block range".into()));
            };
            if new_end > data_len {
                return Err(Error::Format(
                    "index root block past the data region".into(),
                ));
            }
            end = new_end;
            let rows = cur.uvar()?;
            let dir_len = cur.uvar()? as u32;
            let filter_len = cur.uvar()? as u32;
            let pages_len = if tag == 2 { cur.uvar()? as u32 } else { 0 };
            blocks.push(BlockEntry {
                kind,
                off,
                len,
                rows,
                filter_off: 0,
                filter_len,
                pages_off,
                pages_len,
                dir_off: 0,
                dir_len,
            });
            pages_off += pages_len as u64;
        }
        let mut filter_off = pages_off;
        for b in &mut blocks {
            b.filter_off = filter_off;
            filter_off += b.filter_len as u64;
        }
        let mut dir_off = filter_off;
        for b in &mut blocks {
            b.dir_off = dir_off;
            dir_off += b.dir_len as u64;
        }
        let dict = match cur.uvar()? {
            0 => None,
            len => {
                let off = cur.uvar()?;
                if off.checked_add(len).is_none_or(|end| end > data_len) {
                    return Err(Error::Format("index root dictionary range".into()));
                }
                Some(DictEntry {
                    off,
                    len,
                    id: cur.u64_le()?,
                })
            }
        };
        Ok(Self {
            data_len,
            models,
            blocks,
            dict,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The letter-at-a-time form `tokens` must match: the tokens are stored in every filter.
    fn tokens_reference(text: &[u8], mut f: impl FnMut(u64, bool)) {
        let n = text.len();
        let mut i = 0usize;
        while i < n {
            let start = i;
            while i < n && text[i].is_ascii_alphabetic() {
                i += 1;
            }
            let run = &text[start..i];
            let spelled = |w: &[u8]| {
                let mut b = [0u8; 8];
                b[..w.len()].copy_from_slice(w);
                mix(u64::from_le_bytes(b))
            };
            if run.len() >= GRAM {
                run.windows(GRAM).for_each(|w| f(spelled(w), true));
            } else if run.len() >= WORD_MIN {
                f(spelled(run), start > 0 && i < n);
            }
            i += usize::from(run.is_empty());
        }
    }

    #[test]
    fn tokens_match_the_reference() {
        let mut state = 7u64;
        for len in 0..300 {
            let text: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    b"abcXYZ  .9@[`{zA\xc1\xe1"[(state % 18) as usize]
                })
                .collect();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            tokens(&text, |t, c| a.push((t, c)));
            tokens_reference(&text, |t, c| b.push((t, c)));
            assert_eq!(a, b);
        }
    }

    #[test]
    fn text_adds_its_tokens_once_seen_or_not() {
        let text = b"the parser parses; the Parser parsed the parse, then the parser again";
        let (mut by_token, mut by_text) =
            (FilterBuilder::new(1 << 12), FilterBuilder::new(1 << 12));
        tokens(text, |t, _| by_token.add(t));
        by_text.add_text(text);
        by_text.add_text(text);
        assert_eq!(by_token.bits, by_text.bits);
    }

    #[test]
    fn filter_keeps_what_a_needle_needs() {
        let mut b = FilterBuilder::new(1 << 12);
        let mut held = std::collections::HashSet::new();
        for text in ["ClickHouse/ClickHouse", "fix: the parser, again", "x"] {
            tokens(text.as_bytes(), |t, _| {
                b.add(t);
                held.insert(t);
            });
        }
        let bytes = b.finish();
        let f = Filter(&bytes);
        assert!(
            f.valid() && bytes.len() < 1 << 9,
            "a sparse filter shrinks: {}",
            bytes.len()
        );
        assert!(held.iter().all(|&t| f.has(t)), "folding loses nothing");

        // A filter can pass a token it never took, so what a needle needs is checked against the tokens.
        let all = |needle: &str, whole: bool| {
            needle_tokens(needle.as_bytes(), whole).map(|ts| ts.iter().all(|t| held.contains(t)))
        };
        assert_eq!(all("ClickHouse", false), Some(true));
        assert_eq!(all("ickHo", false), Some(true));
        assert_eq!(all("clickhouse", false), Some(false), "case is kept");
        assert_eq!(
            all("ClickHouse/", true),
            Some(true),
            "a long word is known by its grams"
        );
        assert_eq!(all("postgresql", false), Some(false));
        // A word inside the needle is a token; one at its edge may be part of a longer run.
        assert_eq!(all(" fix:", false), Some(true));
        assert_eq!(all(" the ", false), Some(true));
        assert_eq!(all(" pars,", false), Some(false));
        assert_eq!(all("pars", false), None);
        assert_eq!(all("pars", true), Some(false));
        assert_eq!(all("parse", false), Some(true), "part of a longer run");
        assert_eq!(all("42", true), None);
    }

    #[test]
    fn root_and_directory_roundtrip() {
        let budget = DecodeBudget::default();
        let mut dir = BlockDir {
            struct_len: 40,
            struct_head: 9,
            streams: vec![4, 0, 20],
            dicts: vec![(44, 10)],
            pool: Some((56, 100)),
            ..Default::default()
        };
        let deps = ColDeps {
            pool: true,
            dict: Some(0),
            cols: vec![0],
        };
        let cols = vec![
            ColEntry {
                len: 1,
                summary: ColSummary {
                    all_null: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ColEntry {
                len: 30,
                summary: ColSummary {
                    numeric_kind: 1,
                    min_val: -5,
                    max_val: 900,
                    letters: 0x400_0021,
                    per_row: true,
                    ..Default::default()
                },
                deps: deps.clone(),
            },
        ];
        dir.set_cols(160, cols.clone());
        let mut page = Vec::new();
        dir.encode(EntropyBackend::Zstd, 3, &mut page);
        let got = BlockDir::decode(&page, &budget).unwrap();
        assert_eq!(
            (got.struct_len, &got.dicts, got.pool),
            (40, &dir.dicts, dir.pool)
        );
        assert_eq!(
            got.stream_spans().collect::<Vec<_>>(),
            [(9, 4), (13, 0), (13, 20)]
        );
        assert_eq!(got.col_spans().collect::<Vec<_>>(), [(160, 1), (161, 30)]);
        assert_eq!(got.cols().map(|(c, _)| c).collect::<Vec<_>>(), [1]);
        assert_eq!((got.col(1).len, &got.col(1).deps), (30, &deps));
        assert_eq!(
            (got.col(1).summary.min_val, got.col(1).summary.max_val),
            (-5, 900)
        );
        assert_eq!(
            (got.col(0).summary.letters, got.col(1).summary.letters),
            (0, 0x400_0021)
        );
        assert!(got.col(0).summary.all_null && !got.col(1).summary.all_null);
        assert!(!got.col(0).summary.per_row && got.col(1).summary.per_row);

        // A paged pool's pages must cover it exactly.
        let paged = |pages: Vec<(u32, u32)>| {
            let mut d = BlockDir {
                pages,
                ..dir.clone()
            };
            d.set_cols(160, Vec::new());
            let mut page = Vec::new();
            d.encode(EntropyBackend::Zstd, 3, &mut page);
            BlockDir::decode(&page, &budget).map(|d| d.pages)
        };
        assert_eq!(
            paged(vec![(60, 900), (40, 70)]).unwrap(),
            [(60, 900), (40, 70)]
        );
        assert!(
            paged(vec![(60, 900), (41, 70)]).is_err(),
            "pages past the pool"
        );
        assert!(paged(vec![(100, 900)]).is_err(), "one page is a leaf");
        assert!(
            paged(vec![(60, u32::MAX), (40, u32::MAX)]).is_err(),
            "over budget"
        );

        // A wide model whose block uses few columns: the empty ones cost the page almost nothing.
        let mut wide_cols = cols;
        wide_cols.resize(
            5002,
            ColEntry {
                len: 1,
                summary: ColSummary {
                    all_null: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        wide_cols.push(ColEntry {
            len: 7,
            ..Default::default()
        });
        dir.set_cols(160, wide_cols);
        let mut wide_page = Vec::new();
        dir.encode(EntropyBackend::Zstd, 3, &mut wide_page);
        assert!(
            wide_page.len() < page.len() + 16,
            "{} vs {}",
            wide_page.len(),
            page.len()
        );
        let back = BlockDir::decode(&wide_page, &budget).unwrap();
        assert_eq!((back.n_cols(), back.cols().count()), (5003, 2));
        assert_eq!(
            back.col_spans().skip(5001).collect::<Vec<_>>(),
            [(5190, 1), (5191, 7)]
        );
        assert!(back.col(5001).summary.all_null && !back.col(5002).summary.all_null);

        let mut ib = IndexBuilder::default();
        let m = ib.model(b"model-a");
        assert_eq!((ib.model(b"model-a"), ib.model(b"model-b")), (m, 1));
        ib.block(
            BlockKind::Structured {
                model: 1,
                kinds: KIND_JSON,
            },
            30,
            200,
            7,
            &page,
            &[0u8; 8],
            &[2u8; 5],
        );
        ib.block(
            BlockKind::Verbatim {
                framing: ResolvedFraming::Lf { open_end: true },
            },
            260,
            50,
            3,
            &[],
            &[1u8; 16],
            &[],
        );
        let region = ib.finish(400, EntropyBackend::Zstd, 3);
        let len = region.len() as u64;
        let (off, rlen, crc) = locate_root(&region[region.len() - INDEX_TAIL_LEN..], len).unwrap();
        let root =
            Root::decode(&region[off as usize..(off + rlen) as usize], crc, &budget).unwrap();
        assert_eq!(
            (root.data_len, root.models.len(), root.blocks.len()),
            (400, 2, 2)
        );
        let (a, b) = (&root.blocks[0], &root.blocks[1]);
        // The page filter pages, both filters, then both directories.
        assert_eq!(
            (
                a.off,
                a.len,
                a.rows,
                a.pages_off,
                a.pages_len,
                a.filter_off,
                a.dir_off,
                a.dir_len as usize
            ),
            (30, 200, 7, 12, 5, 17, 41, page.len())
        );
        assert_eq!(
            (
                b.off,
                b.rows,
                b.pages_len,
                b.filter_off,
                b.filter_len,
                b.dir_len
            ),
            (260, 3, 0, 25, 16, 0)
        );
        assert_eq!(&region[a.pages_off as usize..][..5], &[2u8; 5]);
        assert_eq!(&region[b.filter_off as usize..][..16], &[1u8; 16]);
        assert_eq!(&region[a.dir_off as usize..][..page.len()], &page[..]);
        assert_eq!(crate::file::directories_offset(&region).unwrap(), a.dir_off);
        assert!(locate_root(&region[region.len() - INDEX_TAIL_LEN..], len + 1).is_err());
    }
}
