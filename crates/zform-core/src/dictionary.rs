//! A file's dictionary: the values that recur across its blocks, stored once. Column leaves are
//! compressed against it, so a block spells out only what the rest of the file does not hold.

use std::cell::RefCell;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use zstd_safe::zstd_sys as zs;
use zstd_safe::{CCtx, CParameter};

use crate::container::PackOptions;
use crate::grammar::{ModelPlan, split_records_with_plan};
use crate::packs::HexFixedPack;
use crate::payload::graph::FastMap;
use crate::tables::Tables;
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, decompress_leaf, put_leaf, window_log,
    xxh64, xxh64_pair,
};

/// Seed of a dictionary's id, as of a frame's content checksum.
const ID_SEED: u64 = 0x5a46_4d31;
/// zstd reads content that starts with these bytes as its own table format, not as text.
const ZSTD_DICT_MAGIC: [u8; 4] = 0xEC30_A437u32.to_le_bytes();
/// The level the content itself is stored at: it is written once and read whole.
const STORE_LEVEL: i32 = 16;
/// Levels a dictionary keeps its content digested at.
const KEPT_LEVELS: usize = 2;
/// The window the content is digested under, whatever leaf comes first. Above 14, so the lazy
/// levels search it by rows rather than chains, which finds more of it; the window also caps the
/// tables each leaf builds over its own bytes, and a larger one buys a fraction of a percent
/// for a hundred megabytes of them.
const DIGEST_WINDOW_LOG: u32 = 17;

pub struct Dictionary {
    /// The content digested, by level. A writer prices leaves at one level and keeps them at
    /// another, and digesting the content costs more than most leaves do. Declared first: a
    /// digest and `ddict` refer to `bytes`, so they must be dropped before them.
    digests: Mutex<Vec<(i32, Arc<Digest>)>>,
    ddict: OnceLock<Option<DDict>>,
    bytes: Vec<u8>,
    /// Computed when first asked for: a writer's tables may still be encoding their wire.
    id: OnceLock<u64>,
    tables: Option<Arc<Tables>>,
}

/// Content wrapped as a zero-copy `ZSTD_DDict` for decompression.
struct DDict(NonNull<zs::ZSTD_DDict>);

// zstd only reads a DDict once it is made.
unsafe impl Send for DDict {}
unsafe impl Sync for DDict {}

impl DDict {
    /// `content` must outlive the DDict, which refers to it.
    fn new(content: &[u8]) -> Option<Self> {
        let ptr =
            unsafe { zs::ZSTD_createDDict_byReference(content.as_ptr().cast(), content.len()) };
        NonNull::new(ptr).map(Self)
    }
}

impl Drop for DDict {
    fn drop(&mut self) {
        unsafe { zs::ZSTD_freeDDict(self.0.as_ptr()) };
    }
}

impl fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dictionary({} bytes, {:016x?})",
            self.len(),
            self.id.get()
        )
    }
}

impl Dictionary {
    /// A dictionary to write a file with. [`DictionaryBuilder`] chooses the content.
    pub fn new(mut bytes: Vec<u8>) -> Self {
        if bytes.starts_with(&ZSTD_DICT_MAGIC) {
            bytes.insert(0, 0);
        }
        Self {
            digests: Mutex::default(),
            ddict: OnceLock::new(),
            id: OnceLock::new(),
            bytes,
            tables: None,
        }
    }

    /// This dictionary carrying `tables` too.
    pub fn with_tables(mut self, tables: Tables) -> Self {
        self.id = OnceLock::new();
        self.tables = Some(Arc::new(tables));
        self
    }

    pub fn tables(&self) -> Option<&Arc<Tables>> {
        self.tables.as_ref()
    }

    /// Bytes of content, which is what a dictionary holds in memory until it is written with.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The XXH64 of the content and tables. A host that keeps dictionaries across opens keys
    /// them by it.
    /// Whether [`id`](Self::id) returns without waiting on the tables' wire.
    pub(crate) fn has_id(&self) -> bool {
        self.id.get().is_some()
    }

    pub fn id(&self) -> u64 {
        *self
            .id
            .get_or_init(|| content_id(&self.bytes, self.tables.as_deref()))
    }

    /// The content as a leaf, at `level` or [`STORE_LEVEL`], whichever is higher.
    pub(crate) fn store(&self, level: i32) -> Vec<u8> {
        let mut out = Vec::new();
        let len = self.len();
        let mut tmp = vec![0u8; zstd_safe::compress_bound(len)];
        let mut ctx = CCtx::create();
        let _ = ctx.set_parameter(CParameter::CompressionLevel(level.max(STORE_LEVEL)));
        let _ = ctx.set_parameter(CParameter::WindowLog(window_log(len)));
        let _ = ctx.set_parameter(CParameter::EnableLongDistanceMatching(true));
        match ctx.compress2(&mut tmp[..], &self.bytes) {
            Ok(n) if n < len => put_leaf(&mut out, EntropyBackend::Zstd, len, &tmp[..n]),
            _ => put_leaf(&mut out, EntropyBackend::Raw, len, &self.bytes),
        }
        if let Some(t) = &self.tables {
            out.extend_from_slice(t.wire());
        }
        out
    }

    /// The dictionary `store` wrote, whose content and tables have the id `id`.
    pub(crate) fn read(stored: &[u8], id: u64, budget: &DecodeBudget) -> Result<Self> {
        let mut cur = Cursor::new(stored);
        let bytes = decompress_leaf(&mut cur, budget)?;
        let rest = &stored[stored.len() - cur.remaining()..];
        let tables = (!rest.is_empty())
            .then(|| Tables::decode(rest, budget))
            .transpose()?;
        if budget.verify_checksums && content_id(&bytes, tables.as_ref()) != id {
            return Err(Error::ChecksumMismatch("dictionary XXH64"));
        }
        if bytes.starts_with(&ZSTD_DICT_MAGIC) {
            return Err(Error::Format(
                "dictionary content reads as zstd tables".into(),
            ));
        }
        let id = OnceLock::from(id);
        Ok(Self {
            digests: Mutex::default(),
            ddict: OnceLock::new(),
            bytes,
            id,
            tables: tables.map(Arc::new),
        })
    }

    /// Compresses `src` into `dst` as a frame that decodes only against this dictionary.
    pub(crate) fn compress(&self, dst: &mut [u8], src: &[u8], level: i32) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let digest = {
            let mut digests = self.digests.lock().unwrap();
            match digests.iter().find(|d| d.0 == level) {
                Some(d) => Arc::clone(&d.1),
                None => {
                    let d = Arc::new(Digest::new(&self.bytes, level)?);
                    if digests.len() == KEPT_LEVELS {
                        digests.remove(0);
                    }
                    digests.push((level, Arc::clone(&d)));
                    d
                }
            }
        };
        COMPRESSOR.with_borrow_mut(|c| {
            if c.is_none() {
                *c = NonNull::new(unsafe { zs::ZSTD_createCCtx() }).map(Compressor::new);
            }
            c.as_mut()?.compress(&digest, dst, src, level)
        })
    }

    /// Expands a frame `compress` wrote directly into `dst_ptr[..dst_cap]`.
    /// The content is referred to through a cached zero-copy `ZSTD_DDict`.
    pub(crate) fn decompress_raw(
        &self,
        dctx: *mut zs::ZSTD_DCtx,
        dst_ptr: *mut u8,
        dst_cap: usize,
        src: &[u8],
    ) -> usize {
        if let Some(dd) = self.ddict.get_or_init(|| DDict::new(&self.bytes)) {
            unsafe {
                zs::ZSTD_decompress_usingDDict(
                    dctx,
                    dst_ptr.cast(),
                    dst_cap,
                    src.as_ptr().cast(),
                    src.len(),
                    dd.0.as_ptr(),
                )
            }
        } else {
            unsafe {
                zs::ZSTD_decompress_usingDict(
                    dctx,
                    dst_ptr.cast(),
                    dst_cap,
                    src.as_ptr().cast(),
                    src.len(),
                    self.bytes.as_ptr().cast(),
                    self.bytes.len(),
                )
            }
        }
    }
}

fn content_id(bytes: &[u8], tables: Option<&Tables>) -> u64 {
    match tables {
        None => xxh64(bytes, ID_SEED),
        Some(t) => xxh64_pair(bytes, t.wire(), ID_SEED),
    }
}

/// Whether a zstd call succeeded.
fn zok(code: usize) -> Option<()> {
    (unsafe { zs::ZSTD_isError(code) } == 0).then_some(())
}

static NEXT_DIGEST_ID: AtomicU64 = AtomicU64::new(1);

/// Content digested at one level: tables that compressors on any thread search in place.
struct Digest {
    ptr: NonNull<zs::ZSTD_CDict>,
    id: u64,
}

// zstd only reads a digest once it is made.
unsafe impl Send for Digest {}
unsafe impl Sync for Digest {}

impl Digest {
    /// `content` must outlive the digest, which refers to it.
    fn new(content: &[u8], level: i32) -> Option<Self> {
        use zs::ZSTD_cParameter::*;
        // The tables hold a position per byte of content, so a value is found wherever it
        // sits. Attached, they are searched in place: a leaf pays for its own bytes only. The
        // binary tree of levels 13 and up holds half the positions its chain log spans, hence
        // one more. The double-fast levels spend it on a second table, for matches under eight
        // bytes; a quarter of the positions there costs 2e-5 of a file priced at such a level,
        // and 48 MiB less for 16 MiB of content.
        let log = content.len().max(2).ilog2().clamp(20, 29) as i32;
        let double_fast =
            unsafe { zs::ZSTD_getCParams(level, 0, 0) }.strategy == zs::ZSTD_strategy::ZSTD_dfast;
        let params = [
            (ZSTD_c_compressionLevel, level),
            (ZSTD_c_hashLog, log),
            (ZSTD_c_chainLog, if double_fast { log - 2 } else { log + 1 }),
            (ZSTD_c_windowLog, DIGEST_WINDOW_LOG as i32),
        ];
        unsafe {
            let p = NonNull::new(zs::ZSTD_createCCtxParams())?;
            let made = params
                .iter()
                .all(|&(k, v)| zok(zs::ZSTD_CCtxParams_setParameter(p.as_ptr(), k, v)).is_some())
                .then(|| {
                    zs::ZSTD_createCDict_advanced2(
                        content.as_ptr().cast(),
                        content.len(),
                        zs::ZSTD_dictLoadMethod_e::ZSTD_dlm_byRef,
                        zs::ZSTD_dictContentType_e::ZSTD_dct_rawContent,
                        p.as_ptr(),
                        // No allocator of its own: zstd's.
                        zs::ZSTD_customMem {
                            customAlloc: None,
                            customFree: None,
                            opaque: std::ptr::null_mut(),
                        },
                    )
                });
            zs::ZSTD_freeCCtxParams(p.as_ptr());
            NonNull::new(made?).map(|ptr| Self {
                ptr,
                id: NEXT_DIGEST_ID.fetch_add(1, Ordering::Relaxed),
            })
        }
    }
}

impl Drop for Digest {
    fn drop(&mut self) {
        unsafe { zs::ZSTD_freeCDict(self.ptr.as_ptr()) };
    }
}

/// A thread's compressor for frames against a digest.
struct Compressor {
    ctx: NonNull<zs::ZSTD_CCtx>,
    last_digest_id: u64,
    last_level: i32,
    last_strategy: i32,
    last_wlog: i32,
}

impl Compressor {
    fn new(ctx: NonNull<zs::ZSTD_CCtx>) -> Self {
        Self {
            ctx,
            last_digest_id: 0,
            last_level: i32::MIN,
            last_strategy: i32::MIN,
            last_wlog: i32::MIN,
        }
    }

    fn compress(
        &mut self,
        digest: &Digest,
        dst: &mut [u8],
        src: &[u8],
        level: i32,
    ) -> Option<usize> {
        use zs::ZSTD_cParameter::*;
        let c = self.ctx.as_ptr();
        let strategy = unsafe { zs::ZSTD_getCParams(level, src.len() as u64, 0).strategy as i32 };
        let wlog = window_log(src.len()) as i32;
        unsafe {
            if self.last_digest_id != digest.id || self.last_level != level {
                zok(zs::ZSTD_CCtx_reset(
                    c,
                    zs::ZSTD_ResetDirective::ZSTD_reset_session_and_parameters,
                ))?;
                zok(zs::ZSTD_CCtx_setParameter(
                    c,
                    ZSTD_c_compressionLevel,
                    level,
                ))?;
                zok(zs::ZSTD_CCtx_setParameter(c, ZSTD_c_strategy, strategy))?;
                zok(zs::ZSTD_CCtx_setParameter(c, ZSTD_c_checksumFlag, 0))?;
                zok(zs::ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, wlog))?;
                // `ZSTD_c_forceAttachDict`: the leaf searches the digest in place, never a copy.
                zok(zs::ZSTD_CCtx_setParameter(c, ZSTD_c_experimentalParam4, 1))?;
                zok(zs::ZSTD_CCtx_refCDict(c, digest.ptr.as_ptr()))?;
                self.last_digest_id = digest.id;
                self.last_level = level;
                self.last_strategy = strategy;
                self.last_wlog = wlog;
            } else {
                zok(zs::ZSTD_CCtx_reset(
                    c,
                    zs::ZSTD_ResetDirective::ZSTD_reset_session_only,
                ))?;
                if self.last_strategy != strategy {
                    zok(zs::ZSTD_CCtx_setParameter(c, ZSTD_c_strategy, strategy))?;
                    self.last_strategy = strategy;
                }
                if self.last_wlog != wlog {
                    zok(zs::ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, wlog))?;
                    self.last_wlog = wlog;
                }
            }
            let n = zs::ZSTD_compress2(
                c,
                dst.as_mut_ptr().cast(),
                dst.len(),
                src.as_ptr().cast(),
                src.len(),
            );
            zok(n).map(|_| n)
        }
    }
}

impl Drop for Compressor {
    fn drop(&mut self) {
        unsafe { zs::ZSTD_freeCCtx(self.ctx.as_ptr()) };
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Dictionary>>> = const { RefCell::new(None) };
    static COMPRESSOR: RefCell<Option<Compressor>> = const { RefCell::new(None) };
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

/// The most content a dictionary used at `level` is searched over: zstd loads only the last
/// 16 MiB at levels 4 and below.
fn max_content(level: i32) -> usize {
    if level <= 4 {
        (1 << 24) - 2
    } else {
        usize::MAX
    }
}

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
    found: FastMap<u64, Found>,
    arena: Vec<u8>,
}

struct Found {
    off: usize,
    len: usize,
    blocks: u32,
    last: u32,
}

/// A value the hex pack codes as the bytes it spells, as those bytes: its leaves hold them so.
fn hex_bytes(v: &[u8]) -> Option<Vec<u8>> {
    let inner = v
        .strip_prefix(b"\"")
        .and_then(|v| v.strip_suffix(b"\""))
        .unwrap_or(v);
    if !matches!(inner.len(), 16 | 32 | 40 | 64) {
        return None;
    }
    HexFixedPack::parse_hex_row(inner, inner.len()).map(|(bytes, _)| bytes)
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
            let off = self.arena.len();
            match hex_bytes(v) {
                Some(bytes) => self.arena.extend_from_slice(&bytes),
                None => self.arena.extend_from_slice(v),
            }
            let len = self.arena.len() - off;
            self.found.insert(
                h,
                Found {
                    off,
                    len,
                    blocks: 2,
                    last: self.block,
                },
            );
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
    /// model's columns. The dictionary holds at most `max_bytes`, and 16 MiB at levels 1 to 4.
    pub fn new(opts: &PackOptions, max_bytes: usize) -> Option<Self> {
        Some(Self {
            plan: opts.profile.as_ref()?.plan.clone(),
            block_rows: opts.block_rows.max(1),
            block_bytes: opts.block_bytes.max(1),
            buf: Vec::new(),
            ends: Vec::new(),
            tally: Tally {
                max_bytes: max_bytes.min(max_content(opts.level)),
                ..Tally::default()
            },
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
        for col in &split.col_occs {
            let mut prev: Option<&[u8]> = None;
            for occ in col {
                let v = occ.as_slice(&split.scratch);
                if v.len() < MIN_VALUE || prev.is_some_and(|p| crate::wire::fast_eq(p, v)) {
                    continue;
                }
                prev = Some(v);
                // A number is coded as one, not as text the dictionary could match.
                if !v.iter().all(|b| b.is_ascii_digit() || b"+-.".contains(b)) {
                    self.tally.note(v);
                }
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
        let Tally {
            max_bytes,
            found,
            arena,
            ..
        } = self.tally;
        let value = |f: &Found| &arena[f.off..f.off + f.len];
        let mut picks: Vec<&Found> = found.values().collect();
        picks
            .sort_unstable_by(|a, b| (b.blocks, b.len, value(a)).cmp(&(a.blocks, a.len, value(b))));
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
        let sample: String = blocks
            .iter()
            .flat_map(|b| b.iter())
            .map(|v| rec(v) + "\n")
            .collect();
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
        let holds =
            |dict: &Dictionary, v: &str| dict.bytes.windows(v.len()).any(|w| w == v.as_bytes());
        let all = build(&blocks, 1 << 20).unwrap();
        assert!([&a, &b, &c].iter().all(|v| holds(&all, v)));
        assert!(
            [&d, &u, &w, &x, &short].iter().all(|v| !holds(&all, v)),
            "a value one block holds, however often, or a short one, stays out"
        );
        assert!(all.len() < 3 * 28, "{}", all.len());

        let one = build(&blocks, 40).unwrap();
        assert!(holds(&one, &a) && !holds(&one, &b) && !holds(&one, &c));
        assert_eq!(
            one.id(),
            build(&blocks, 40).unwrap().id(),
            "the choice is repeatable"
        );
        assert!(build(&blocks[..1], 1 << 20).is_none() && build(&blocks, 8).is_none());
    }

    #[test]
    fn builder_holds_hex_as_its_bytes() {
        let sha = "0123456789abcdef".repeat(3)[..40].to_string();
        let other = "x".repeat(24);
        let dict = build(&[&[&sha, &other], &[&sha, &other]], 1 << 20).unwrap();
        let bytes: Vec<u8> = (0..20)
            .map(|i| u8::from_str_radix(&sha[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        assert!(
            dict.bytes.windows(20).any(|w| w == bytes),
            "{:?}",
            dict.bytes
        );
        assert_eq!(dict.len(), 20 + other.len() + 2);
    }

    #[test]
    fn deep_levels_reach_the_start_of_the_content() {
        // Values alike but for a random tail, sorted as the builder writes them.
        let mut state = 7u64;
        let mut vals: Vec<String> = (0..24 << 10)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                format!("https://example.com/users/{state:016x}")
            })
            .collect();
        vals.sort();
        let dict = Dictionary::new(vals.concat().into_bytes());
        // Values from the first quarter, which a tree spanning half the content misses.
        let src: String = vals[..8192]
            .iter()
            .rev()
            .step_by(4)
            .map(String::as_str)
            .collect();
        let mut leaf = Vec::new();
        let src = src.as_bytes();
        crate::wire::compress_leaf_with(&mut leaf, src, EntropyBackend::Zstd, 19, Some(&dict));
        assert!(leaf.len() < 2048 * 3, "{}", leaf.len());
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
        let stored = Dictionary::read(&dict.store(3), dict.id(), &budget).unwrap();
        assert_eq!((stored.len(), stored.id()), (dict.len(), dict.id()));
        assert_eq!(read(Some(&stored)).unwrap(), raw);
        assert!(Dictionary::read(&dict.store(3), dict.id() ^ 1, &budget).is_err());
        let mut again = Vec::new();
        crate::wire::compress_leaf_with(&mut again, &raw, EntropyBackend::Zstd, 3, Some(&stored));
        assert_eq!(again, leaf);
    }

    #[test]
    fn leaves_compress_alike_whatever_came_before() {
        let vals: Vec<String> = (0..4096)
            .map(|i| format!("value-{:06}-{}", i * 7919 % 100_003, i % 13))
            .collect();
        let text = |from: usize, n: usize| -> Vec<u8> {
            (0..n)
                .map(|i| vals[(from + i * 31) % vals.len()].as_str())
                .collect::<String>()
                .into_bytes()
        };
        let big = text(5, 12_000);
        let leaf = |dict: &Dictionary, src: &[u8], level| {
            let mut out = Vec::new();
            crate::wire::compress_leaf_with(&mut out, src, EntropyBackend::Zstd, level, Some(dict));
            out
        };
        let fresh = || Dictionary::new(vals.concat().into_bytes());
        let want = [3, 9].map(|level| leaf(&fresh(), &big, level));
        // A small leaf first, and the levels taken in turn, change nothing.
        let dict = fresh();
        leaf(&dict, &text(1, 20), 9);
        for _ in 0..2 {
            assert_eq!(leaf(&dict, &big, 3), want[0]);
            assert_eq!(leaf(&dict, &big, 9), want[1]);
        }
        assert!(want[1].len() < want[0].len() && want[0].len() < big.len() / 8);
    }
}
