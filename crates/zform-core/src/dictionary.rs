//! A file's dictionary: the values that recur across its blocks, stored once. Column leaves are
//! compressed against it, so a block spells out only what the rest of the file does not hold.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use zstd_safe::{CCtx, CParameter, DCtx, DictAttachPref, ResetDirective};

use crate::container::PackOptions;
use crate::grammar::{ModelPlan, split_records_with_plan};
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, decompress_leaf, put_leaf, window_log,
    xxh64,
};

/// Seed of a dictionary's id, as of a frame's content checksum.
const ID_SEED: u64 = 0x5a46_4d31;
/// zstd reads content that starts with these bytes as its own table format, not as text.
const ZSTD_DICT_MAGIC: [u8; 4] = 0xEC30_A437u32.to_le_bytes();
/// The level the content itself is stored at: it is written once and read whole.
const STORE_LEVEL: i32 = 16;

pub struct Dictionary {
    bytes: Vec<u8>,
    id: u64,
    /// A compressor that has digested the content, with the level it did so at.
    enc: Mutex<Option<(i32, CCtx<'static>)>>,
}

impl fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dictionary({} bytes, {:016x})", self.len(), self.id)
    }
}

impl Dictionary {
    /// A dictionary to write a file with. [`DictionaryBuilder`] chooses the content.
    pub fn new(mut bytes: Vec<u8>) -> Self {
        if bytes.starts_with(&ZSTD_DICT_MAGIC) {
            bytes.insert(0, 0);
        }
        Self { id: xxh64(&bytes, ID_SEED), bytes, enc: Mutex::new(None) }
    }

    /// Bytes of content, which is what a dictionary holds in memory until it is written with.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The XXH64 of the content. A host that keeps dictionaries across opens keys them by it.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The content as a leaf.
    pub(crate) fn store(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let len = self.len();
        let mut tmp = vec![0u8; zstd_safe::compress_bound(len)];
        let mut ctx = CCtx::create();
        let _ = ctx.set_parameter(CParameter::CompressionLevel(STORE_LEVEL));
        let _ = ctx.set_parameter(CParameter::WindowLog(window_log(len)));
        let _ = ctx.set_parameter(CParameter::EnableLongDistanceMatching(true));
        match ctx.compress2(&mut tmp[..], &self.bytes) {
            Ok(n) if n < len => put_leaf(&mut out, EntropyBackend::Zstd, len, &tmp[..n]),
            _ => put_leaf(&mut out, EntropyBackend::Raw, len, &self.bytes),
        }
        out
    }

    /// The dictionary stored as `leaf`, whose content has the id `id`.
    pub(crate) fn read(leaf: &[u8], id: u64, budget: &DecodeBudget) -> Result<Self> {
        let bytes = decompress_leaf(&mut Cursor::new(leaf), budget)?;
        if budget.verify_checksums && xxh64(&bytes, ID_SEED) != id {
            return Err(Error::ChecksumMismatch("dictionary XXH64"));
        }
        if bytes.starts_with(&ZSTD_DICT_MAGIC) {
            return Err(Error::Format("dictionary content reads as zstd tables".into()));
        }
        Ok(Self { bytes, id, enc: Mutex::new(None) })
    }

    /// Compresses `src` into `dst` as a frame that decodes only against this dictionary.
    pub(crate) fn compress(&self, dst: &mut [u8], src: &[u8], level: i32) -> Option<usize> {
        let mut enc = self.enc.lock().unwrap();
        if enc.as_ref().is_none_or(|e| e.0 != level) {
            let mut ctx = CCtx::try_create()?;
            // The tables hold a position per byte of content, so a value is found wherever it
            // sits. Attached, they are searched in place: a leaf pays for its own bytes only.
            let log = self.len().max(2).ilog2().clamp(20, 30);
            ctx.set_parameter(CParameter::CompressionLevel(level)).ok()?;
            ctx.set_parameter(CParameter::ChecksumFlag(false)).ok()?;
            ctx.set_parameter(CParameter::HashLog(log)).ok()?;
            ctx.set_parameter(CParameter::ChainLog(log)).ok()?;
            ctx.set_parameter(CParameter::ForceAttachDict(DictAttachPref::ForceAttach)).ok()?;
            ctx.load_dictionary(&self.bytes).ok()?;
            *enc = Some((level, ctx));
        }
        let ctx = &mut enc.as_mut()?.1;
        ctx.reset(ResetDirective::SessionOnly).ok()?;
        // The content stays in reach only within a frame's first window.
        ctx.set_parameter(CParameter::WindowLog(window_log(src.len()))).ok()?;
        ctx.compress2(dst, src).ok()
    }

    /// Expands a frame `compress` wrote. The content is referred to, not copied.
    pub(crate) fn decompress(
        &self,
        dctx: &mut DCtx<'_>,
        dst: &mut [u8],
        src: &[u8],
    ) -> zstd_safe::SafeResult {
        dctx.decompress_using_dict(dst, src, &self.bytes)
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Dictionary>>> = const { RefCell::new(None) };
}

/// While it lives, the column leaves this thread writes and reads are compressed against `dict`.
pub struct Scope(Option<Arc<Dictionary>>);

impl Scope {
    pub fn enter(dict: Option<Arc<Dictionary>>) -> Self {
        Self(CURRENT.replace(dict))
    }

    /// Changes the dictionary for the rest of the scope.
    pub(crate) fn set(&mut self, dict: Option<Arc<Dictionary>>) {
        CURRENT.set(dict);
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.set(self.0.take());
    }
}

/// Runs `f` with the dictionary in scope.
pub(crate) fn with_current<R>(f: impl FnOnce(Option<&Dictionary>) -> R) -> R {
    CURRENT.with_borrow(|d| f(d.as_deref()))
}

/// Values shorter than this are left to a block's own tables.
const MIN_VALUE: usize = 16;
/// Values are kept while they come to under this many times the dictionary's budget.
const KEEP_FACTOR: usize = 4;

/// Chooses a file's dictionary in a pass over its records before they are written: the values
/// that two or more blocks hold, those in the most blocks first, up to a byte budget. Its memory
/// is the open block, 4 bytes per distinct value up to 64 MiB, and four budgets of values.
pub struct DictionaryBuilder {
    plan: ModelPlan,
    block_rows: usize,
    block_bytes: usize,
    /// The open block's records back to back, and where each ends.
    buf: Vec<u8>,
    ends: Vec<usize>,
    tally: Tally,
}

#[derive(Default)]
struct Tally {
    max_bytes: usize,
    block: u32,
    /// Values of the blocks before the open one, by hash.
    seen: Seen,
    /// Values the open block is the first to hold.
    fresh: Vec<u64>,
    /// Values some earlier block held too.
    found: HashMap<u64, Found>,
    arena: Vec<u8>,
}

struct Found {
    off: usize,
    len: usize,
    blocks: u32,
    last: u32,
}

impl Tally {
    fn note(&mut self, v: &[u8]) {
        let h = xxh64(v, 0);
        if let Some(f) = self.found.get_mut(&h) {
            f.blocks += u32::from(f.last != self.block);
            f.last = self.block;
        } else if !self.seen.has(h) {
            self.fresh.push(h);
        } else if self.arena.len() + v.len() <= self.max_bytes.saturating_mul(KEEP_FACTOR) {
            let (off, len) = (self.arena.len(), v.len());
            self.arena.extend_from_slice(v);
            self.found.insert(h, Found { off, len, blocks: 2, last: self.block });
        }
    }

    fn end_block(&mut self) {
        for h in self.fresh.drain(..) {
            self.seen.add(h);
        }
        self.block += 1;
    }
}

/// A Bloom filter that grows with what it holds, at 32 bits a value, to 64 MiB. Past the 16
/// million values that holds it fills, and takes more and more new values for ones seen.
#[derive(Default)]
struct Seen {
    layers: Vec<Vec<u64>>,
    /// Values the last layer still takes.
    room: usize,
}

impl Seen {
    const MAX_WORDS: usize = 1 << 22;

    fn bits(layer: &[u64], h: u64) -> [(usize, u64); 2] {
        let mask = layer.len() * 64 - 1;
        [h, h.rotate_left(32)].map(|x| ((x as usize & mask) >> 6, 1 << (x & 63)))
    }

    fn has(&self, h: u64) -> bool {
        self.layers
            .iter()
            .any(|l| Self::bits(l, h).iter().all(|&(w, b)| l[w] & b != 0))
    }

    fn add(&mut self, h: u64) {
        let words = self.layers.last().map_or(1 << 12, |l| l.len() * 2);
        if self.room == 0 && words <= Self::MAX_WORDS {
            self.layers.push(vec![0; words]);
            self.room = words * 2;
        }
        self.room = self.room.saturating_sub(1);
        let layer = self.layers.last_mut().unwrap();
        for (w, b) in Self::bits(layer, h) {
            layer[w] |= b;
        }
    }
}

impl DictionaryBuilder {
    /// For a file written with `opts`, which must name a profile: the values are those of its
    /// model's columns. The dictionary holds at most `max_bytes`.
    pub fn new(opts: &PackOptions, max_bytes: usize) -> Option<Self> {
        Some(Self {
            plan: opts.profile.as_ref()?.plan.clone(),
            block_rows: opts.block_rows.max(1),
            block_bytes: opts.block_bytes.max(1),
            buf: Vec::new(),
            ends: Vec::new(),
            tally: Tally { max_bytes, ..Tally::default() },
        })
    }

    /// Adds a record, in the order the writer is given them.
    pub fn push(&mut self, rec: &[u8]) {
        if self.ends.len() == self.block_rows
            || (!self.ends.is_empty() && self.buf.len() + rec.len() > self.block_bytes)
        {
            self.cut();
        }
        self.buf.extend_from_slice(rec);
        self.ends.push(self.buf.len());
    }

    fn cut(&mut self) {
        let mut start = 0;
        let recs: Vec<&[u8]> = self
            .ends
            .iter()
            .map(|&end| &self.buf[std::mem::replace(&mut start, end)..end])
            .collect();
        let split = split_records_with_plan(&recs, &self.plan);
        for occ in split.col_occs.iter().flatten() {
            let v = occ.as_slice(&split.scratch);
            if v.len() >= MIN_VALUE {
                self.tally.note(v);
            }
        }
        self.tally.end_block();
        self.buf.clear();
        self.ends.clear();
    }

    /// The dictionary, or `None` when no value recurs.
    pub fn finish(mut self) -> Option<Dictionary> {
        if !self.ends.is_empty() {
            self.cut();
        }
        let Tally { max_bytes, found, arena, .. } = self.tally;
        let value = |f: &Found| &arena[f.off..f.off + f.len];
        let mut picks: Vec<&Found> = found.values().collect();
        picks.sort_unstable_by(|a, b| {
            (b.blocks, b.len, value(a)).cmp(&(a.blocks, a.len, value(b)))
        });
        let mut total = 0usize;
        picks.retain(|f| {
            let fits = total + f.len <= max_bytes;
            total += if fits { f.len } else { 0 };
            fits
        });
        // Like values side by side: the content is compressed too.
        picks.sort_unstable_by(|a, b| value(a).cmp(value(b)));
        let bytes: Vec<u8> = picks.iter().flat_map(|f| value(f)).copied().collect();
        (!bytes.is_empty()).then(|| Dictionary::new(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Profile, TrainOptions, train_from_bytes};

    /// The dictionary of `blocks`, each a list of `note` values, one record per value.
    fn build(blocks: &[&[&str]], max_bytes: usize) -> Option<Dictionary> {
        let rec = |v: &&str| format!("{{\"id\":7,\"note\":\"{v}\"}}");
        let rows = blocks.iter().map(|b| b.len()).max().unwrap();
        let sample: String = blocks.iter().flat_map(|b| b.iter()).map(|v| rec(v) + "\n").collect();
        let plan = train_from_bytes(sample.as_bytes(), &TrainOptions::default(), None);
        let opts = PackOptions {
            profile: Some(Profile::from_plan(plan)),
            block_rows: rows,
            ..PackOptions::default()
        };
        let mut b = DictionaryBuilder::new(&opts, max_bytes)?;
        for block in blocks {
            assert_eq!(block.len(), rows);
            block.iter().for_each(|v| b.push(rec(v).as_bytes()));
        }
        b.finish()
    }

    #[test]
    fn builder_keeps_what_blocks_share_most_shared_first() {
        let v = |c: char| c.to_string().repeat(24);
        let (a, b, c, d) = (v('a'), v('b'), v('c'), v('d'));
        let (u, w, x, short) = (v('u'), v('w'), v('x'), "s".repeat(MIN_VALUE - 3));
        let blocks: [&[&str]; 3] = [
            &[&a, &b, &d, &d, &short],
            &[&a, &b, &c, &u, &short],
            &[&a, &c, &w, &x, &short],
        ];
        let holds = |dict: &Dictionary, v: &str| dict.bytes.windows(v.len()).any(|w| w == v.as_bytes());
        let all = build(&blocks, 1 << 20).unwrap();
        assert!([&a, &b, &c].iter().all(|v| holds(&all, v)));
        assert!(
            [&d, &u, &w, &x, &short].iter().all(|v| !holds(&all, v)),
            "a value one block holds, however often, or a short one, stays out"
        );
        assert!(all.len() < 3 * 28, "{}", all.len());

        let one = build(&blocks, 40).unwrap();
        assert!(holds(&one, &a) && !holds(&one, &b) && !holds(&one, &c));
        assert_eq!(one.id(), build(&blocks, 40).unwrap().id(), "the choice is repeatable");
        assert!(build(&blocks[..1], 1 << 20).is_none() && build(&blocks, 8).is_none());
    }

    #[test]
    fn leaf_reaches_its_dictionary_past_the_default_window() {
        let mut state = 7u64;
        let mut noise = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect()
        };
        let shared = noise(128 << 10);
        let dict = Dictionary::new(shared.clone());
        // The shared bytes follow more new ones than a level-3 window spans.
        let mut raw = noise(3 << 20);
        raw.extend_from_slice(&shared);
        let mut leaf = Vec::new();
        crate::wire::compress_leaf_with(&mut leaf, &raw, EntropyBackend::Zstd, 3, Some(&dict));
        assert_eq!(leaf[0], EntropyBackend::ZstdDict as u8);
        assert!(leaf.len() < raw.len() - (64 << 10), "{}", leaf.len());
        let budget = DecodeBudget::default();
        let read = |d| crate::wire::decompress_leaf_with(&mut Cursor::new(&leaf), &budget, d);
        assert_eq!(read(Some(&dict)).unwrap(), raw);
        assert!(read(None).is_err());

        // A dictionary read back from its stored leaf is the same one.
        let stored = Dictionary::read(&dict.store(), dict.id(), &budget).unwrap();
        assert_eq!((stored.len(), stored.id()), (dict.len(), dict.id()));
        assert_eq!(read(Some(&stored)).unwrap(), raw);
        assert!(Dictionary::read(&dict.store(), dict.id() ^ 1, &budget).is_err());
        let mut again = Vec::new();
        crate::wire::compress_leaf_with(&mut again, &raw, EntropyBackend::Zstd, 3, Some(&stored));
        assert_eq!(again, leaf);
    }
}
