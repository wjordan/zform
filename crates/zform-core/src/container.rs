//! The `ZFM1` framed stream, which is a file's data region. A frame carries a model, or reuses the
//! previous frame's, and self-contained blocks. The encoder collects the index region as it goes.

use crate::cells::{Cells, Col, underflow};
use crate::codecs::{
    ANY_LETTERS, BlockDecodeCtx, ColSummary, PoolParts, Seeds, column_deps, decode_present_slice,
    decode_u32_seq, encode_dense_block_columns, encode_present_slice, encode_u32_seq,
    ensure_dense_block_column, with_seeds,
};
use crate::dictionary::{Dictionary, DictionaryBuilder, Scope};
use crate::framing::{
    FramedChunk, FramingSpec, ResolvedFraming, emit_framed_record, split_chunk, split_json_array,
};
use crate::grammar::{
    BlockReconstructor, ModelPlan, OccRef, PlanVariant, SplitResult, split_records_with_plan,
};
use crate::index::{
    BlockDir, BlockKind, ColEntry, FilterBuilder, INDEX_MAGIC, IndexBuilder, KIND_JSON,
    KIND_RESIDUAL, KIND_SHAPE, KIND_SHAPE_CELL, page_filter_page, tokens,
};
use crate::json_tree::{decode_json_struct_streams, encode_json_struct_streams};
use crate::lex::parse_base64;
use crate::packs::{EncodedDenseBlock, PackRegistry, Paging};
use crate::payload::graph::{NESTED_COLUMN_TAG, PayloadGraph};
use crate::payload::library::{Harvest, Library, with_code_pricing};
use crate::plan::{ProbeBackoff, Profile, TrainOptions, sample_strided, train_from_records};
use crate::tables::Tables;
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, compress_leaf, crc32, decompress_leaf,
    leaf_raw_len, put_bytes, put_uvarint, xxh64,
};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::{Arc, Condvar, Mutex, mpsc};

pub const FRAME_MAGIC: &[u8; 4] = b"ZFM1";

pub const FLAG_HAS_CHECKSUM: u16 = 1 << 0;
pub const FLAG_PASSTHROUGH: u16 = 1 << 1;
pub const FLAG_REUSE_PREV_MODEL: u16 = 1 << 2;
pub const FLAG_SYNC_POINT: u16 = 1 << 3;
pub const FLAG_DICTIONARY: u16 = 1 << 4;
/// Each block stores its rows in any order; the reader restores them by sorting on
/// [`collation_key`]. The frame's records are its blocks' records in block order.
pub const FLAG_SORTED_ROWS: u16 = 1 << 5;

/// The key a sorted-rows block's records ascend by: ASCII letters and digits only, letters folded
/// to lowercase.
pub fn collation_key(rec: &[u8]) -> Vec<u8> {
    rec.iter()
        .filter(|b| b.is_ascii_alphanumeric())
        .map(|b| b.to_ascii_lowercase())
        .collect()
}

/// Whether sorting `recs` by [`collation_key`] gives them back: keys never fall, and records with
/// equal keys are equal.
fn collation_sorted(recs: &[&[u8]]) -> bool {
    let mut prev: Option<(Vec<u8>, &[u8])> = None;
    for &r in recs {
        let k = collation_key(r);
        if let Some((pk, pr)) = &prev
            && (pk > &k || (pk == &k && *pr != r))
        {
            return false;
        }
        prev = Some((k, r));
    }
    true
}

/// Restores a sorted-rows block's record order.
pub(crate) fn sort_records(recs: &mut [Vec<u8>]) {
    recs.sort_by_cached_key(|r| collation_key(r));
}

pub const DEFAULT_BLOCK_ROWS: usize = 65_536;
pub const DEFAULT_BLOCK_BYTES: usize = 8 << 20;
/// Largest substring filter of one block, in bits. A filter shrinks to fit what its block holds.
pub const DEFAULT_FILTER_BITS: usize = 1 << 17;
pub const DEFAULT_POOL_PAGE_BYTES: usize = 384 << 10;

#[derive(Debug, Clone)]
pub struct PackOptions {
    pub level: i32,
    pub backend: EntropyBackend,
    pub chunk_bytes: usize,
    pub block_bytes: usize,
    pub block_rows: usize,
    pub framing: FramingSpec,
    pub train: TrainOptions,
    pub profile: Option<Profile>,
    /// Values the stream's blocks share, written ahead of them. A reader needs it for any block.
    pub dictionary: Option<Arc<Dictionary>>,
    /// Allow online retraining when residual exceeds `drift_residual_ratio` (even if `profile` was supplied when `allow_profile_drift` is true).
    pub allow_drift_retrain: bool,
    pub drift_residual_ratio: f64,
    /// Number of consecutive frames allowed to reuse a model (`REUSE_PREV_MODEL`) before forcing an independent `SYNC_POINT`.
    pub sync_interval_frames: usize,
    pub checksum: bool,
    /// Build the index region; `StreamEncoder::finish` writes it after the last frame.
    pub index: bool,
    pub filter_bits: usize,
    /// Most raw bytes in a page of a block's text pool, which a reader fetches a page at a time.
    /// Zero keeps each pool whole.
    pub pool_page_bytes: usize,
    /// Pluggable writer-side codec packs evaluated on columns.
    pub packs: PackRegistry,
    /// Frames `pack_stream` encodes at once; the output is the same for any count.
    pub threads: usize,
}

impl Default for PackOptions {
    fn default() -> Self {
        Self {
            level: 3,
            backend: EntropyBackend::Zstd,
            chunk_bytes: 16 << 20,
            block_bytes: DEFAULT_BLOCK_BYTES,
            block_rows: DEFAULT_BLOCK_ROWS,
            framing: FramingSpec::Auto,
            train: TrainOptions::default(),
            profile: None,
            dictionary: None,
            allow_drift_retrain: true,
            drift_residual_ratio: 0.10,
            sync_interval_frames: 8,
            checksum: true,
            index: true,
            filter_bits: DEFAULT_FILTER_BITS,
            pool_page_bytes: DEFAULT_POOL_PAGE_BYTES,
            packs: PackRegistry::default(),
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct InspectReport {
    pub frames: usize,
    pub sync_frames: usize,
    pub passthrough_frames: usize,
    pub total_compressed_bytes: u64,
    pub total_uncompressed_bytes: u64,
    pub total_records: u64,
    pub index_bytes: u64,
    pub column_names: Vec<String>,
    pub column_bytes: Vec<(String, u64)>,
    pub variant_count: usize,
}

/// What one encoded frame was made of.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameFit {
    pub rows: usize,
    pub bytes: usize,
    /// Rows and bytes the model could not place. A frame stored verbatim counts in full.
    pub residual_rows: usize,
    pub residual_bytes: usize,
}

/// About 256 KiB of `samples`: a layout of a few percent there still clears the trainer's floor.
fn keep_rows(samples: &[&[u8]]) -> Vec<Vec<u8>> {
    sample_strided(samples, 4096, 256 << 10)
        .iter()
        .map(|r| r.to_vec())
        .collect()
}

// ============================================================================
// 1. Stateful Multi-Frame Stream Encoder
// ============================================================================

pub struct StreamEncoder {
    opts: PackOptions,
    active_plan: Option<ModelPlan>,
    frames_since_sync: usize,
    backoff: ProbeBackoff,
    index: IndexBuilder,
    model_idx: u32,
    frames: usize,
    comp_offset: u64,
    seeds: Seeds,
    /// A slice of the rows the plan was trained on: a retrain sees them too, so it keeps the
    /// layouts that the new chunk lacks.
    trained_rows: Vec<Vec<u8>>,
}

impl StreamEncoder {
    pub fn new(opts: PackOptions) -> Self {
        let active_plan = opts.profile.as_ref().map(|p| p.plan.clone());
        Self {
            opts,
            active_plan,
            frames_since_sync: usize::MAX,
            backoff: ProbeBackoff::default(),
            index: IndexBuilder::default(),
            model_idx: 0,
            frames: 0,
            comp_offset: 0,
            seeds: Seeds::default(),
            trained_rows: Vec::new(),
        }
    }

    pub fn active_plan(&self) -> Option<&ModelPlan> {
        self.active_plan.as_ref()
    }

    /// Frames written so far.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Bytes written so far: the length of the data region.
    pub fn data_len(&self) -> u64 {
        self.comp_offset
    }

    /// Compresses one chunk of raw bytes into a `ZFM1` frame written to `out`.
    pub fn encode_frame<W: Write>(&mut self, raw: &[u8], mut out: W) -> Result<FrameFit> {
        if raw.is_empty() {
            return Ok(FrameFit::default());
        }
        self.encode_framed(&split_chunk(raw, self.opts.framing), &mut out)
    }

    /// Writes the dictionary as a frame of its own.
    fn write_dictionary<W: Write>(&mut self, dict: &Dictionary, mut out: W) -> Result<()> {
        let payload = dict.store(self.opts.level);
        let hdr = frame_header(
            FLAG_DICTIONARY | FLAG_HAS_CHECKSUM,
            EntropyBackend::Zstd,
            dict.len(),
            0,
            ResolvedFraming::LengthU32Le,
            payload.len(),
        );
        out.write_all(&hdr)?;
        out.write_all(&payload)?;
        out.write_all(&dict.id().to_le_bytes())?;
        let off = self.comp_offset + hdr.len() as u64;
        self.index.dictionary(off, payload.len() as u64, dict.id());
        self.comp_offset = off + payload.len() as u64 + 8;
        Ok(())
    }

    /// Compresses one already-framed chunk into a `ZFM1` frame written to `out`.
    pub(crate) fn encode_framed<W: Write>(
        &mut self,
        fc: &FramedChunk<'_>,
        mut out: W,
    ) -> Result<FrameFit> {
        self.start(&mut out)?;
        let task = self.decide(fc);
        // A sync frame starts afresh, so frames of different groups encode at once.
        if !task.reuse {
            self.seeds = Seeds::default();
        }
        let mut seeds = self.seeds.clone();
        let mut enc = encode_task(&self.opts, &task, fc, self.active_plan.as_ref(), &mut seeds);
        if !task.reuse {
            self.seeds = seeds;
        }
        if let Some(p) = enc.plan.take() {
            self.active_plan = Some(p);
        }
        self.emit(&task, enc, out)
    }

    /// Writes what precedes the first frame.
    fn start<W: Write>(&mut self, out: W) -> Result<()> {
        if self.comp_offset == 0
            && let Some(dict) = self.opts.dictionary.clone()
        {
            self.write_dictionary(&dict, out)?;
        }
        Ok(())
    }

    /// Chooses the frame's model and kind, in stream order: the model may retrain on drift.
    fn decide(&mut self, fc: &FramedChunk<'_>) -> FrameTask {
        let all_recs: Vec<&[u8]> = (0..fc.len()).map(|i| fc.record(i)).collect();

        // Pilot probe (`<= 512` strided records) to detect drift BEFORE encoding the full chunk!
        let mut plan_changed = false;
        let pilot = sample_strided(&all_recs, 512, 256 << 10);
        // Lines that are one Base64 value each fit no template, but a column codec decodes them.
        let mut scratch = Vec::new();
        let coded = pilot.len() >= 4
            && pilot
                .iter()
                .all(|r| r.len() >= 16 && parse_base64(r, &mut scratch).is_some());

        if let Some(cur_plan) = &self.active_plan {
            if self.opts.allow_drift_retrain && !pilot.is_empty() {
                let ps = split_records_with_plan(&pilot, cur_plan);
                // A JSON object in a key layout the plan lacks is stored whole: drift as well.
                let missed = |ps: &SplitResult| ps.residual_bytes + ps.unshaped_bytes;
                let pilot_bytes: usize = pilot.iter().map(|r| r.len()).sum();
                let row_ratio = ps.residual_rows as f64 / pilot.len().max(1) as f64;
                let byte_ratio = missed(&ps) as f64 / pilot_bytes.max(1) as f64;
                if (row_ratio > self.opts.drift_residual_ratio
                    || byte_ratio > self.opts.drift_residual_ratio)
                    && self.backoff.should_probe()
                {
                    let samples = sample_strided(
                        &all_recs,
                        self.opts.train.max_sample_rows,
                        self.opts.train.max_sample_bytes,
                    );
                    let mut both = samples.clone();
                    both.extend(self.trained_rows.iter().map(|r| r.as_slice()));
                    let extended =
                        train_from_records(&both, fc.framing, &self.opts.train, Some(cur_plan));
                    let new_ps = split_records_with_plan(&pilot, &extended);
                    if missed(&new_ps) + 64 < missed(&ps) {
                        self.active_plan = Some(extended);
                        self.trained_rows = keep_rows(&samples);
                        plan_changed = true;
                        self.backoff.record_success();
                    } else {
                        self.backoff.record_failure();
                    }
                }
            }
        } else if self.backoff.should_probe() && all_recs.len() >= 4 {
            let samples = sample_strided(
                &all_recs,
                self.opts.train.max_sample_rows,
                self.opts.train.max_sample_bytes,
            );
            let trained = train_from_records(&samples, fc.framing, &self.opts.train, None);
            let ps = split_records_with_plan(&pilot, &trained);
            if coded || trained.variants.len() > 1 && ps.residual_rows * 10 < pilot.len() * 9 {
                self.active_plan = Some(trained);
                self.trained_rows = keep_rows(&samples);
                plan_changed = true;
                self.backoff.record_success();
            } else {
                self.backoff.record_failure();
            }
        }

        // Decide Passthrough vs Structural
        let passthrough = match &self.active_plan {
            None => true,
            Some(p) => {
                if all_recs.len() < 2 {
                    true
                } else if coded {
                    false
                } else if p.variants.len() <= 1 {
                    true
                } else {
                    let ps = split_records_with_plan(&pilot, p);
                    ps.residual_rows == pilot.len()
                }
            }
        };

        let sync = passthrough
            || plan_changed
            || self.opts.sync_interval_frames <= 1
            || self.frames_since_sync >= self.opts.sync_interval_frames - 1;
        let reuse = !passthrough && !sync;
        let code = if reuse {
            CodeSharing::Reuse
        } else {
            // A chunk well short of full is taken for the stream's last.
            let goes_on = fc.raw.len() * 2 >= self.opts.chunk_bytes;
            let frames = if goes_on {
                self.opts.sync_interval_frames.max(1)
            } else {
                1
            };
            CodeSharing::Build { frames }
        };
        // `emit` corrects this when the encoded frame falls back to passthrough.
        if passthrough {
            self.frames_since_sync = usize::MAX;
        } else if sync {
            self.frames_since_sync = 0;
        } else {
            self.frames_since_sync += 1;
        }
        FrameTask {
            passthrough,
            sync,
            reuse,
            code,
            plan_changed,
        }
    }

    /// Writes an encoded frame and indexes it, in stream order.
    fn emit<W: Write>(
        &mut self,
        task: &FrameTask,
        enc: EncodedFrame,
        mut out: W,
    ) -> Result<FrameFit> {
        let EncodedFrame {
            payload,
            blocks,
            sorted,
            final_passthrough,
            raw_plan,
            fit,
            raw_len,
            records,
            framing,
            checksum,
            verbatim_filter,
            ..
        } = enc;
        if let Some(raw_plan) = &raw_plan
            && self.opts.index
        {
            self.model_idx = self.index.model(raw_plan);
        }
        let actual_sync = final_passthrough || task.sync;
        let actual_reuse = !final_passthrough && task.reuse;
        if final_passthrough && !task.passthrough {
            // The next structural frame re-syncs; it is decided after this one is written.
            self.frames_since_sync = usize::MAX;
        }

        let mut flags = 0u16;
        if self.opts.checksum {
            flags |= FLAG_HAS_CHECKSUM;
        }
        if final_passthrough {
            flags |= FLAG_PASSTHROUGH;
        }
        if actual_reuse {
            flags |= FLAG_REUSE_PREV_MODEL;
        }
        if actual_sync {
            flags |= FLAG_SYNC_POINT;
        }
        if sorted && !final_passthrough {
            flags |= FLAG_SORTED_ROWS;
        }

        let hdr = frame_header(
            flags,
            self.opts.backend,
            raw_len,
            records,
            framing,
            payload.len(),
        );

        let tail_len = if self.opts.checksum { 8usize } else { 0usize };
        let total_frame_bytes = (hdr.len() + payload.len() + tail_len) as u64;

        out.write_all(&hdr)?;
        out.write_all(&payload)?;
        if let Some(csum) = checksum {
            out.write_all(&csum.to_le_bytes())?;
        }

        let base = self.comp_offset + hdr.len() as u64;
        if !self.opts.index {
        } else if !final_passthrough {
            for b in &blocks {
                let Some((dir, filter, pages, kinds)) = &b.index else {
                    continue;
                };
                // A sorted block is read whole, so it needs no directory or pages.
                let (kind, dir, pages): (_, &[u8], &[u8]) = match sorted {
                    true => (
                        BlockKind::Sorted {
                            model: self.model_idx,
                        },
                        &[],
                        &[],
                    ),
                    false => {
                        let kind = BlockKind::Structured {
                            model: self.model_idx,
                            kinds: *kinds,
                        };
                        (kind, dir, pages)
                    }
                };
                self.index.block(
                    kind,
                    base + b.off as u64,
                    b.len as u64,
                    b.rows as u64,
                    dir,
                    filter,
                    pages,
                );
            }
        } else if let Some(filter) = verbatim_filter {
            let kind = BlockKind::Verbatim { framing };
            let (len, rows) = (payload.len() as u64, records as u64);
            self.index.block(kind, base, len, rows, &[], &filter, &[]);
        }
        self.comp_offset += total_frame_bytes;
        self.frames += 1;
        Ok(fit)
    }

    /// The index region over every frame written, or nothing when there is none to write.
    pub fn finish_index(&mut self) -> Vec<u8> {
        let index = std::mem::take(&mut self.index);
        if !self.opts.index || index.is_empty() {
            return Vec::new();
        }
        index.finish(self.comp_offset, self.opts.backend, self.opts.level)
    }

    /// Writes the index region after the last frame.
    pub fn finish<W: Write>(&mut self, mut out: W) -> Result<()> {
        out.write_all(&self.finish_index())?;
        Ok(())
    }
}

/// A frame's kind and model sharing, decided in stream order.
#[derive(Clone)]
pub(crate) struct FrameTask {
    passthrough: bool,
    sync: bool,
    reuse: bool,
    code: CodeSharing,
    /// The model was trained for this frame.
    plan_changed: bool,
}

/// A frame's payload, encoded on any thread, to be written in stream order.
pub(crate) struct EncodedFrame {
    payload: Vec<u8>,
    blocks: Vec<FrameBlock>,
    sorted: bool,
    final_passthrough: bool,
    /// The model with the library a sync frame's blocks made.
    plan: Option<ModelPlan>,
    raw_len: usize,
    records: usize,
    framing: ResolvedFraming,
    checksum: Option<u64>,
    /// The token filter of a passthrough frame's records, when indexed.
    verbatim_filter: Option<Vec<u8>>,
    /// The model with that library, encoded, for a frame that writes its model.
    raw_plan: Option<Vec<u8>>,
    fit: FrameFit,
}

/// Encodes `fc` as `task` decided, with `plan` holding the library the frame starts from.
fn encode_task(
    opts: &PackOptions,
    task: &FrameTask,
    fc: &FramedChunk<'_>,
    plan: Option<&ModelPlan>,
    seeds: &mut Seeds,
) -> EncodedFrame {
    let raw = fc.raw;
    let all_recs: Vec<&[u8]> = (0..fc.len()).map(|i| fc.record(i)).collect();
    let rec_bytes: usize = all_recs.iter().map(|r| r.len()).sum();
    let mut fit = FrameFit {
        rows: all_recs.len(),
        bytes: rec_bytes,
        residual_rows: all_recs.len(),
        residual_bytes: rec_bytes,
    };
    let filter_bits = opts.index.then_some(opts.filter_bits);
    let mut payload = Vec::new();
    let mut blocks = Vec::new();
    let mut sorted = false;
    let (mut new_plan, mut raw_plan) = (None, None);
    if task.passthrough {
        compress_leaf(&mut payload, raw, opts.backend, opts.level);
    } else {
        // The blocks come first: a sync frame's model is written with the library they make.
        let plan = plan.expect("a structural frame has a model");
        let fresh;
        let plan = match task.reuse {
            true => plan,
            false => {
                fresh = ModelPlan {
                    library: Arc::default(),
                    ..plan.clone()
                };
                &fresh
            }
        };
        let mut body = Vec::new();
        let (frame_blocks, lib, frame_sorted) = {
            let _dict = Scope::enter(opts.dictionary.clone());
            with_seeds(seeds, || {
                encode_structural_blocks(
                    &all_recs,
                    plan,
                    &opts.packs,
                    opts.backend,
                    opts.level,
                    opts.block_rows,
                    opts.block_bytes,
                    filter_bits,
                    // Pages serve readers that come through the index.
                    if opts.index { opts.pool_page_bytes } else { 0 },
                    task.code,
                    opts.packs.orders_rows(),
                    &mut body,
                )
            })
        };
        blocks = frame_blocks;
        sorted = frame_sorted;
        if !task.reuse {
            let mut with_lib = plan.clone();
            with_lib.library = Arc::clone(&lib);
            let mut rp = Vec::new();
            with_lib.encode(&mut rp);
            compress_leaf(&mut payload, &rp, opts.backend, opts.level.max(3));
            raw_plan = Some(rp);
            new_plan = Some(with_lib);
        }
        if has_affixes(fc.framing) {
            compress_leaf(&mut payload, fc.prefix, opts.backend, opts.level);
            compress_leaf(&mut payload, fc.suffix, opts.backend, opts.level);
        }
        if fc.framing == ResolvedFraming::JsonArray {
            compress_leaf(&mut payload, fc.sep(), opts.backend, opts.level);
        }
        for b in &mut blocks {
            b.off += payload.len();
        }
        payload.extend_from_slice(&body);
        fit.residual_rows = blocks.iter().map(|b| b.residual_rows).sum();
        fit.residual_bytes = blocks.iter().map(|b| b.residual_bytes).sum();
    }

    // A structural payload larger than the plain leaf, as tiny or degenerate inputs give, is
    // dropped for the leaf.
    let mut final_passthrough = task.passthrough;
    if !task.passthrough && raw.len() <= SMALL_FRAME {
        let mut pt_trial = Vec::new();
        compress_leaf(&mut pt_trial, raw, opts.backend, opts.level);
        if pt_trial.len() + 32 < payload.len() {
            payload = pt_trial;
            final_passthrough = true;
            (fit.residual_rows, fit.residual_bytes) = (fit.rows, fit.bytes);
        }
    }
    let verbatim_filter = (opts.index && final_passthrough && !fc.is_empty()).then(|| {
        let mut fb = FilterBuilder::new(opts.filter_bits);
        for r in &all_recs {
            tokens(r, |t, _| fb.add(t));
        }
        fb.finish()
    });
    EncodedFrame {
        raw_len: raw.len(),
        records: fc.len(),
        framing: fc.framing,
        checksum: opts.checksum.then(|| xxh64(raw, 0x5a46_4d31)),
        verbatim_filter,
        payload,
        blocks,
        sorted,
        final_passthrough,
        plan: new_plan,
        raw_plan,
        fit,
    }
}

/// Raw bytes up to which a structural frame may fall back to passthrough once encoded.
const SMALL_FRAME: usize = 1 << 20;

fn frame_header(
    flags: u16,
    backend: EntropyBackend,
    raw_len: usize,
    records: usize,
    framing: ResolvedFraming,
    payload_len: usize,
) -> Vec<u8> {
    let mut hdr = Vec::with_capacity(48);
    hdr.extend_from_slice(FRAME_MAGIC);
    hdr.extend_from_slice(&flags.to_le_bytes());
    hdr.push(backend as u8);
    put_uvarint(&mut hdr, raw_len as u64);
    put_uvarint(&mut hdr, records as u64);
    framing.encode(&mut hdr);
    hdr.extend_from_slice(&(payload_len as u64).to_le_bytes());
    let crc = crc32(&hdr);
    hdr.extend_from_slice(&crc.to_le_bytes());
    hdr
}

/// One block inside a frame payload.
pub(crate) struct FrameBlock {
    pub rows: usize,
    pub off: usize,
    pub len: usize,
    pub residual_rows: usize,
    pub residual_bytes: usize,
    pub index: Option<(Vec<u8>, Vec<u8>, Vec<u8>, u16)>,
}

#[allow(clippy::too_many_arguments)]
fn encode_structural_blocks(
    all_recs: &[&[u8]],
    plan: &ModelPlan,
    packs: &PackRegistry,
    backend: EntropyBackend,
    level: i32,
    max_block_rows: usize,
    max_block_bytes: usize,
    filter_bits: Option<usize>,
    page_bytes: usize,
    code: CodeSharing,
    reorder: bool,
    out: &mut Vec<u8>,
) -> (Vec<FrameBlock>, Arc<Library>, bool) {
    let mut bounds = Vec::new();
    let mut s = 0usize;
    while s < all_recs.len() {
        let mut e = s;
        let mut b = 0usize;
        while e < all_recs.len() && (e - s) < max_block_rows {
            let rl = all_recs[e].len();
            if e > s && b + rl > max_block_bytes {
                break;
            }
            b += rl;
            e += 1;
        }
        bounds.push((s, e));
        s = e;
    }
    // Every block of the frame must sort back into its records for any to be stored reordered.
    let sorted = reorder && bounds.iter().all(|&(s, e)| collation_sorted(&all_recs[s..e]));
    // A library this frame makes is shared by its blocks and, when the stream goes on, by the
    // frames that reuse the model: code is priced as that many uses.
    let (keep_single, share) = match code {
        CodeSharing::Reuse => (false, 1),
        CodeSharing::Build { frames } => (frames > 1, bounds.len() * frames),
    };
    let mut pending: Vec<PendingBlock> =
        with_code_pricing(plan.library.clone(), share as u32, || {
            bounds
                .iter()
                .map(|&(s, e)| {
                    let recs = &all_recs[s..e];
                    let order = sorted.then(|| packs.row_order(recs)).flatten();
                    let permuted: Vec<&[u8]>;
                    let recs = match order {
                        Some(o) => {
                            permuted = o.iter().map(|&i| recs[i as usize]).collect();
                            &permuted[..]
                        }
                        None => recs,
                    };
                    encode_block_columns(recs, plan, packs, backend, level, filter_bits, page_bytes)
                })
                .collect()
        });
    let library = match code {
        CodeSharing::Reuse => plan.library.clone(),
        CodeSharing::Build { .. } => {
            let mut harvest = Harvest::default();
            for p in &mut pending {
                p.harvest(plan, &mut harvest);
            }
            Arc::new(harvest.into_library(keep_single))
        }
    };
    let encoded: Vec<(usize, EncodedBlock)> = bounds
        .iter()
        .zip(pending)
        .map(|(&(s, e), p)| (e - s, p.finish(plan, &library, backend, level)))
        .collect();

    put_uvarint(out, encoded.len() as u64);
    for (nrows, eb) in &encoded {
        put_uvarint(out, *nrows as u64);
        put_uvarint(out, eb.bytes.len() as u64);
    }
    let mut blocks = Vec::with_capacity(encoded.len());
    for (rows, eb) in encoded {
        blocks.push(FrameBlock {
            rows,
            off: out.len(),
            len: eb.bytes.len(),
            residual_rows: eb.residual_rows,
            residual_bytes: eb.residual_bytes,
            index: eb.index,
        });
        out.extend_from_slice(&eb.bytes);
    }
    (blocks, library, sorted)
}

/// Where a frame's blocks get their library.
#[derive(Clone, Copy)]
enum CodeSharing {
    /// From the model as it stands: the frame reuses the model of a frame before it.
    Reuse,
    /// From the frame's own blocks, to be written with its model and reused by `frames` frames.
    Build { frames: usize },
}

pub(crate) struct EncodedBlock {
    pub bytes: Vec<u8>,
    /// Rows and bytes the model left residual.
    pub residual_rows: usize,
    pub residual_bytes: usize,
    /// Directory, filter and page filter pages and row kinds, when a filter size was given.
    pub index: Option<(Vec<u8>, Vec<u8>, Vec<u8>, u16)>,
}

/// What kinds of row a block holds, as `index::KIND_*` bits.
fn row_kinds(plan: &ModelPlan, variant_ids: &[u32]) -> u16 {
    let mut seen = vec![false; plan.variants.len()];
    for &v in variant_ids {
        seen[v as usize] = true;
    }
    let mut kinds = 0u16;
    for (vi, v) in plan.variants.iter().enumerate().filter(|(vi, _)| seen[*vi]) {
        kinds |= match v {
            PlanVariant::Shape(_) if vi == 0 => KIND_RESIDUAL,
            PlanVariant::Shape(_) => KIND_SHAPE,
            PlanVariant::JsonTree => KIND_JSON,
            PlanVariant::ShapeCell { .. } => KIND_SHAPE_CELL,
        };
    }
    kinds
}

/// Encodes one block against the library its model already has. `filter_bits` asks for its
/// index pages as well.
pub(crate) fn encode_block_fit(
    recs: &[&[u8]],
    plan: &ModelPlan,
    packs: &PackRegistry,
    backend: EntropyBackend,
    level: i32,
    filter_bits: Option<usize>,
) -> EncodedBlock {
    with_code_pricing(plan.library.clone(), 1, || {
        encode_block_columns(recs, plan, packs, backend, level, filter_bits, 0)
    })
    .finish(plan, &plan.library, backend, level)
}

/// The library for a model whose blocks will look like `recs`: the code one block of them uses.
pub(crate) fn learn_library(
    recs: &[&[u8]],
    plan: &ModelPlan,
    packs: &PackRegistry,
    backend: EntropyBackend,
    level: i32,
) -> Arc<Library> {
    if recs.is_empty() {
        return plan.library.clone();
    }
    // Priced as shared by many blocks: the sample stands for all of them.
    let mut block = with_code_pricing(plan.library.clone(), 64, || {
        encode_block_columns(recs, plan, packs, backend, level, None, 0)
    });
    let mut harvest = Harvest::default();
    block.harvest(plan, &mut harvest);
    Arc::new(harvest.into_library(true))
}

/// A block whose columns are encoded with their code inline, before it is laid out. A frame holds
/// its blocks like this until it knows which code they share.
pub(crate) struct PendingBlock {
    /// The row structure, which leads the block.
    out: Vec<u8>,
    dir: BlockDir,
    enc: EncodedDenseBlock,
    per_row: Vec<bool>,
    residual_rows: usize,
    residual_bytes: usize,
    /// Filter page, row kinds and row count, when a filter size was given.
    index: Option<(Vec<u8>, u16, usize)>,
}

impl PendingBlock {
    /// The payloads that are column graphs: the shared dictionaries and every column but the
    /// nested ones, which hold blocks of their own.
    fn graphs<'a>(&'a mut self, plan: &'a ModelPlan) -> impl Iterator<Item = &'a mut Vec<u8>> {
        let cols = self
            .enc
            .col_blobs
            .iter_mut()
            .zip(&plan.columns)
            .filter(|(_, c)| c.nested_plan.is_none())
            .map(|(b, _)| b);
        self.enc.shared_dict_blobs.iter_mut().chain(cols)
    }

    fn harvest(&mut self, plan: &ModelPlan, into: &mut Harvest) {
        for blob in self.graphs(plan) {
            into.add_blob(blob);
        }
    }

    /// Lays the block out, naming the code `lib` holds instead of carrying it.
    fn finish(
        mut self,
        plan: &ModelPlan,
        lib: &Library,
        backend: EntropyBackend,
        level: i32,
    ) -> EncodedBlock {
        if !lib.is_empty() {
            let budget = DecodeBudget::default();
            for blob in self.graphs(plan) {
                if let Ok(g) = PayloadGraph::decode(blob, &budget)
                    && g.has_code()
                {
                    *blob = g.encode_linked(Some(lib));
                }
            }
        }
        let Self {
            mut out,
            mut dir,
            enc,
            per_row,
            residual_rows,
            residual_bytes,
            index,
        } = self;
        put_uvarint(&mut out, enc.shared_dict_blobs.len() as u64);
        for db in &enc.shared_dict_blobs {
            put_uvarint(&mut out, db.len() as u64);
            dir.dicts.push((out.len() as u32, db.len() as u32));
            out.extend_from_slice(db);
        }
        match enc.pool_pages.as_slice() {
            [] => out.push(0),
            [pl] => {
                out.push(1);
                put_uvarint(&mut out, pl.len() as u64);
                dir.pool = Some((out.len() as u32, pl.len() as u32));
                out.extend_from_slice(pl);
            }
            pages => {
                out.push(2);
                put_uvarint(&mut out, pages.len() as u64);
                for p in pages {
                    put_uvarint(&mut out, p.len() as u64);
                }
                let off = out.len();
                for p in pages {
                    let raw = leaf_raw_len(p).expect("a page the writer made");
                    dir.pages.push((p.len() as u32, raw as u32));
                    out.extend_from_slice(p);
                }
                dir.pool = Some((off as u32, (out.len() - off) as u32));
            }
        }
        for blob in &enc.col_blobs {
            put_uvarint(&mut out, blob.len() as u64);
        }
        let cols_off = out.len() as u32;
        let mut cols = Vec::with_capacity(enc.col_blobs.len());
        for (((blob, mut summary), col), per_row) in enc
            .col_blobs
            .iter()
            .zip(enc.summaries)
            .zip(&plan.columns)
            .zip(per_row)
        {
            summary.per_row = per_row;
            // A nested cell is shown escaped, with letters its stored bytes lack.
            if col.nested_plan.is_some() && !summary.all_null {
                summary.letters = ANY_LETTERS;
            }
            cols.push(ColEntry {
                len: blob.len() as u32,
                summary,
                deps: column_deps(blob),
            });
            out.extend_from_slice(blob);
        }
        dir.set_cols(cols_off, cols);

        let index = index.map(|(filter, kinds, n_rows)| {
            let mut page = Vec::new();
            dir.encode(backend, level, &mut page);
            let pages = match enc.pool_filters.len() {
                0 | 1 => Vec::new(),
                _ => {
                    let mut rows = enc.pool_rows.clone();
                    rows.resize(n_rows, 0);
                    page_filter_page(&enc.pool_filters, &rows)
                }
            };
            (page, filter, pages, kinds)
        });
        EncodedBlock {
            bytes: out,
            residual_rows,
            residual_bytes,
            index,
        }
    }
}

/// Splits `recs` into columns and encodes each, under the code pricing in effect.
fn encode_block_columns(
    recs: &[&[u8]],
    plan: &ModelPlan,
    packs: &PackRegistry,
    backend: EntropyBackend,
    level: i32,
    filter_bits: Option<usize>,
    page_bytes: usize,
) -> PendingBlock {
    let _mixing = packs.encoding_scope();
    let split = split_records_with_plan(recs, plan);
    let n_cols = plan.columns.len();

    // Materialize dense `&[u8]` occurrence streams into `split` + `split.scratch`
    let mut occ_views: Vec<Vec<&[u8]>> = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        let mut v = Vec::with_capacity(split.col_occs[c].len());
        for &occ in &split.col_occs[c] {
            let slice = match occ {
                OccRef::Borrowed(b) => b,
                OccRef::Scratch(idx) => split.scratch.items[idx].as_slice(),
            };
            v.push(slice);
        }
        occ_views.push(v);
    }

    let mut preencoded_nested: Vec<Option<(ColSummary, Vec<u8>)>> = vec![None; n_cols];
    for cid in 0..n_cols {
        if let Some(child_plan) = &plan.columns[cid].nested_plan {
            let is_json_tagged = plan.columns[cid].name.starts_with("$.");
            preencoded_nested[cid] = Some(encode_nested_column_tagged(
                &occ_views[cid],
                is_json_tagged,
                child_plan,
                packs,
                backend,
                level,
            ));
        }
    }

    let dict_groups: Vec<&str> = plan.columns.iter().map(|c| c.dict_group.as_str()).collect();
    let horiz_cands: Vec<Vec<usize>> = plan
        .columns
        .iter()
        .map(|c| c.horiz_candidates.iter().map(|&x| x as usize).collect())
        .collect();

    // A column holding raw objects or arrays answers field queries below its path, so it must be
    // readable without the text pool.
    let mut standalone = vec![false; plan.columns.len()];
    if let Some(jt) = &plan.json_tree {
        for p in jt.paths.iter().filter(|p| p.value_col != u16::MAX) {
            let col = p.value_col as usize;
            standalone[col] = occ_views[col]
                .iter()
                .any(|v| matches!(v.first(), Some(b'{' | b'[')));
        }
    }

    let cell_rows = |want: &[bool]| {
        crate::query::cell_rows(plan, &split.json_struct_streams, &split.variant_ids, want).ok()
    };
    let paging = (page_bytes > 0).then_some(Paging {
        page_bytes,
        cell_rows: &cell_rows,
    });
    let enc = encode_dense_block_columns(
        &occ_views,
        &dict_groups,
        &horiz_cands,
        &mut preencoded_nested,
        &standalone,
        Some((plan, &split.variant_ids)),
        paging,
        packs,
        backend,
        level,
    );

    // Layout: row structure, shared dictionaries, text pool, column lengths, columns.
    let mut out = Vec::new();
    let mut dir = BlockDir::default();
    encode_u32_seq(&split.variant_ids, backend, level, &mut out);
    dir.struct_head = out.len() as u32;
    if plan.json_tree.is_some() {
        let mut js_buf = Vec::new();
        let (head, lens) =
            encode_json_struct_streams(&split.json_struct_streams, backend, level, &mut js_buf);
        put_uvarint(&mut out, js_buf.len() as u64);
        dir.struct_head = (out.len() + head) as u32;
        dir.streams = lens;
        out.extend_from_slice(&js_buf);
    }
    dir.struct_len = out.len() as u32;
    let index = filter_bits.map(|bits| {
        let mut fb = FilterBuilder::new(bits);
        for r in recs {
            tokens(r, |t, _| fb.add(t));
        }
        (fb.finish(), row_kinds(plan, &split.variant_ids), recs.len())
    });
    PendingBlock {
        out,
        dir,
        enc,
        per_row: split.per_row.clone(),
        residual_rows: split.residual_rows,
        residual_bytes: split.residual_bytes,
        index,
    }
}

/// Compresses a dense occurrence column carrying a recursive `nested_plan` (either JSON-unescaped
/// `[0|1, payload...]` or direct inner strings from CRI/Syslog envelopes!).
fn encode_nested_column_tagged(
    vals: &[&[u8]],
    is_json_tagged: bool,
    child_plan: &ModelPlan,
    packs: &PackRegistry,
    backend: EntropyBackend,
    level: i32,
) -> (ColSummary, Vec<u8>) {
    let logical: Vec<&[u8]> = if is_json_tagged {
        vals.iter().map(|&b| &b[1.min(b.len())..]).collect()
    } else {
        vals.to_vec()
    };
    let summary = ColSummary::of(&logical);
    let occ_len = vals.len();
    let mut out = Vec::new();
    put_uvarint(&mut out, occ_len as u64);
    if occ_len == 0 {
        return (summary, out);
    }
    out.push(NESTED_COLUMN_TAG);
    out.push(u8::from(is_json_tagged));

    // `row_modes[i]`: `1 = ChildPlan (no trailing \n)`, `2 = ChildPlan (had trailing \n)`, `3 = Fallback raw`
    let mut row_modes = Vec::with_capacity(occ_len);
    let mut child_recs = Vec::<&[u8]>::new();
    let mut fallbacks = Vec::<&[u8]>::new();

    for &b in vals {
        if is_json_tagged {
            if b.first() == Some(&1u8) {
                let inner = &b[1..];
                let (trimmed, had_nl) = if let Some(stripped) = inner.strip_suffix(b"\n") {
                    (stripped, true)
                } else {
                    (inner, false)
                };
                row_modes.push(if had_nl { 2 } else { 1 });
                child_recs.push(trimmed);
            } else {
                row_modes.push(3);
                fallbacks.push(&b[1.min(b.len())..]);
            }
        } else {
            row_modes.push(1);
            child_recs.push(b);
        }
    }

    encode_u32_seq(&row_modes, backend, level, &mut out);
    let child_blob = if !child_recs.is_empty() {
        encode_block_fit(&child_recs, child_plan, packs, backend, level, None).bytes
    } else {
        Vec::new()
    };
    put_bytes(&mut out, &child_blob);
    if !fallbacks.is_empty() {
        let (_, _, _, fb_enc) = encode_present_slice(&fallbacks, backend, level, 1);
        out.extend_from_slice(&fb_enc);
    }
    if packs.plain_nested_fallback && !is_json_tagged && !child_recs.is_empty() {
        // A child model can cost more than the whole messages, especially when one block
        // repeats only a few of them. The existing raw mode keeps that choice local to a block.
        let mut flat = Vec::new();
        put_uvarint(&mut flat, occ_len as u64);
        flat.extend([NESTED_COLUMN_TAG, 0]);
        encode_u32_seq(&vec![3; occ_len], backend, level, &mut flat);
        put_bytes(&mut flat, &[]);
        let (_, _, _, payload) = encode_present_slice(&logical, backend, level, 1);
        flat.extend_from_slice(&payload);
        if flat.len() < out.len() {
            out = flat;
        }
    }
    (summary, out)
}

fn decode_nested_column_tagged(
    blob: &[u8],
    child_plan: &ModelPlan,
    budget: &DecodeBudget,
) -> Result<Cells> {
    let mut cur = Cursor::new(blob);
    let occ_len = cur.uvar()? as usize;
    if occ_len == 0 {
        return Ok(Cells::default());
    }
    if occ_len > budget.max_records_per_block.saturating_mul(16) {
        return Err(Error::BudgetExceeded("nested column occurrence count"));
    }
    let h_mode = cur.u8()?;
    if h_mode != NESTED_COLUMN_TAG {
        return Err(Error::Format("expected nested column tag".into()));
    }
    let is_json_tagged = cur.u8()? != 0;
    let row_modes = decode_u32_seq(&mut cur, budget)?;
    if row_modes.len() != occ_len {
        return Err(Error::Format("nested row_modes count mismatch".into()));
    }
    let child_blob = cur.bytes()?;
    let n_fb = row_modes.iter().filter(|&&m| m == 3).count();
    let fallbacks = if n_fb > 0 {
        decode_present_slice(&mut cur, n_fb, budget)?
    } else {
        Col::Flat(Cells::default())
    };
    let n_child = row_modes.iter().filter(|&&m| m == 1 || m == 2).count();
    let mut child = if n_child > 0 {
        Some(BlockRows::open(child_blob, n_child, child_plan, budget)?)
    } else {
        None
    };

    let mut fi = 0usize;
    let mut out = Cells::with_capacity(occ_len, blob.len().saturating_mul(2));
    for &m in &row_modes {
        let buf = out.tail();
        match m {
            1 | 2 => {
                if is_json_tagged {
                    buf.push(1u8);
                }
                child
                    .as_mut()
                    .ok_or_else(|| Error::Format("nested child record underflow".into()))?
                    .emit(buf, budget)?;
                if is_json_tagged && m == 2 {
                    buf.push(b'\n');
                }
            }
            3 => {
                if is_json_tagged {
                    buf.push(0u8);
                }
                fallbacks.write(fi, buf)?;
                fi += 1;
            }
            other => return Err(Error::Format(format!("invalid nested row_mode {other}"))),
        }
        out.seal();
    }
    Ok(out)
}

// ============================================================================
// 2. Streaming Block-by-Block Frame Decoder & Header Parser
// ============================================================================

#[derive(Debug, Clone)]
pub struct ParsedFrameHeader {
    pub flags: u16,
    pub backend: EntropyBackend,
    pub raw_len: usize,
    pub record_count: usize,
    pub framing: ResolvedFraming,
    pub payload_len: usize,
}

#[derive(Debug, Clone)]
pub enum NextFrameKind {
    Data(ParsedFrameHeader),
    /// An index region; `skip_bytes` remain after the bytes already read.
    Index {
        skip_bytes: usize,
    },
    Eof,
}

/// Reads the next frame header from `reader`: a `ZFM1` data frame, an index region to skip, or a clean end.
pub fn read_next_frame_header<R: Read>(
    reader: &mut R,
    budget: &DecodeBudget,
) -> Result<NextFrameKind> {
    let mut magic = [0u8; 4];
    if !read_exact_or_eof(reader, &mut magic)? {
        return Ok(NextFrameKind::Eof);
    }
    if &magic == INDEX_MAGIC {
        let mut lb = [0u8; 8];
        reader.read_exact(&mut lb)?;
        return Ok(NextFrameKind::Index {
            skip_bytes: u64::from_le_bytes(lb) as usize,
        });
    }
    if &magic != FRAME_MAGIC {
        return Err(Error::Format("invalid ZFM1 frame magic".into()));
    }

    // Read remainder of variable-length header byte-by-byte into `hdr_buf` (max ~48 bytes)
    let mut hdr_buf = Vec::with_capacity(48);
    hdr_buf.extend_from_slice(&magic);
    let mut fixed3 = [0u8; 3];
    reader.read_exact(&mut fixed3)?;
    hdr_buf.extend_from_slice(&fixed3);
    let flags = u16::from_le_bytes([fixed3[0], fixed3[1]]);
    let backend = EntropyBackend::from_u8(fixed3[2])?;

    let raw_len = read_stream_uvarint(reader, &mut hdr_buf)? as usize;
    let record_count = read_stream_uvarint(reader, &mut hdr_buf)? as usize;

    let ftag = read_stream_u8(reader, &mut hdr_buf)?;
    let framing = match ftag {
        0 => ResolvedFraming::Lf {
            open_end: read_stream_u8(reader, &mut hdr_buf)? != 0,
        },
        1 => ResolvedFraming::Crlf {
            open_end: read_stream_u8(reader, &mut hdr_buf)? != 0,
        },
        2 => ResolvedFraming::Nul {
            open_end: read_stream_u8(reader, &mut hdr_buf)? != 0,
        },
        3 => ResolvedFraming::LengthU32Le,
        4 => {
            let stride = read_stream_uvarint(reader, &mut hdr_buf)? as u32;
            let header_bytes = read_stream_uvarint(reader, &mut hdr_buf)? as u32;
            let tail_bytes = read_stream_uvarint(reader, &mut hdr_buf)? as u32;
            if stride == 0 || stride > (16 << 20) {
                return Err(Error::Format("invalid FixedStride in frame header".into()));
            }
            ResolvedFraming::FixedStride {
                stride,
                header_bytes,
                tail_bytes,
            }
        }
        5 => ResolvedFraming::JsonArray,
        k => return Err(Error::Format(format!("unknown framing tag {k}"))),
    };

    let mut plen_b = [0u8; 8];
    reader.read_exact(&mut plen_b)?;
    hdr_buf.extend_from_slice(&plen_b);
    let payload_len = u64::from_le_bytes(plen_b) as usize;
    if payload_len > budget.max_frame_compressed_bytes {
        return Err(Error::BudgetExceeded("frame compressed payload length"));
    }

    let mut crc_b = [0u8; 4];
    reader.read_exact(&mut crc_b)?;
    let expected_crc = u32::from_le_bytes(crc_b);
    if budget.verify_checksums && crc32(&hdr_buf) != expected_crc {
        return Err(Error::ChecksumMismatch("ZFM1 frame header CRC32"));
    }

    Ok(NextFrameKind::Data(ParsedFrameHeader {
        flags,
        backend,
        raw_len,
        record_count,
        framing,
        payload_len,
    }))
}

fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut got = 0usize;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => {
                if got == 0 {
                    return Ok(false);
                }
                return Err(Error::Format("truncated frame header".into()));
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Io(e.to_string())),
        }
    }
    Ok(true)
}

fn read_stream_u8<R: Read>(r: &mut R, log: &mut Vec<u8>) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    log.push(b[0]);
    Ok(b[0])
}

fn read_stream_uvarint<R: Read>(r: &mut R, log: &mut Vec<u8>) -> Result<u64> {
    let mut x = 0u64;
    let mut shift = 0u32;
    for _ in 0..10 {
        let b = read_stream_u8(r, log)?;
        if shift == 63 && b > 1 {
            return Err(Error::Format("stream varint overflow".into()));
        }
        x |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            return Ok(x);
        }
        shift += 7;
    }
    Err(Error::Format("stream varint too long".into()))
}

/// Framings whose structural frames carry prefix and suffix leaves.
fn has_affixes(framing: ResolvedFraming) -> bool {
    matches!(
        framing,
        ResolvedFraming::FixedStride { .. } | ResolvedFraming::JsonArray
    )
}

/// Parsed structural block layout inside a frame payload.
pub(crate) struct BlockDirectory<'a> {
    pub prefix: Vec<u8>,
    pub suffix: Vec<u8>,
    /// Written between records; only `JsonArray` frames have one.
    pub sep: Vec<u8>,
    pub blocks: Vec<(usize, &'a [u8])>, // `(n_rows, block_slice)`
}

pub fn load_sync_frame_plan(
    hdr: &ParsedFrameHeader,
    payload: &[u8],
    prev_plan: &mut Option<ModelPlan>,
    budget: &DecodeBudget,
) -> Result<()> {
    let _ = parse_frame_blocks(hdr, payload, prev_plan, budget)?;
    Ok(())
}

pub(crate) fn parse_frame_blocks<'a>(
    hdr: &ParsedFrameHeader,
    payload: &'a [u8],
    prev_plan: &mut Option<ModelPlan>,
    budget: &DecodeBudget,
) -> Result<BlockDirectory<'a>> {
    let mut cur = Cursor::new(payload);
    if (hdr.flags & FLAG_REUSE_PREV_MODEL) == 0 {
        let raw_plan = decompress_leaf(&mut cur, budget)?;
        let decoded_plan = ModelPlan::decode(&mut Cursor::new(&raw_plan), 0)?;
        *prev_plan = Some(decoded_plan);
    } else if prev_plan.is_none() {
        return Err(Error::Format(
            "REUSE_PREV_MODEL set without a preceding sync frame".into(),
        ));
    }

    let (prefix, suffix) = if has_affixes(hdr.framing) {
        (
            decompress_leaf(&mut cur, budget)?,
            decompress_leaf(&mut cur, budget)?,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let sep = if hdr.framing == ResolvedFraming::JsonArray {
        decompress_leaf(&mut cur, budget)?
    } else {
        Vec::new()
    };

    let n_blocks = cur.uvar()? as usize;
    let mut lens = Vec::with_capacity(n_blocks.min(4096));
    let mut total_rows = 0usize;
    for _ in 0..n_blocks {
        let nrows = cur.uvar()? as usize;
        let blen = cur.uvar()? as usize;
        total_rows = total_rows.saturating_add(nrows);
        lens.push((nrows, blen));
    }
    if total_rows != hdr.record_count {
        return Err(Error::Format("block row sum != frame record_count".into()));
    }
    let mut blocks = Vec::with_capacity(n_blocks);
    for (nrows, blen) in lens {
        blocks.push((nrows, cur.slice(blen)?));
    }
    Ok(BlockDirectory {
        prefix,
        suffix,
        sep,
        blocks,
    })
}

/// Column `cid` of an open block, decoded on first use.
pub(crate) fn block_col<'c>(
    cid: usize,
    plan: &ModelPlan,
    ctx: &'c mut BlockDecodeCtx<'_>,
    budget: &DecodeBudget,
) -> Result<&'c Col> {
    if ctx.cache[cid].is_none()
        && let Some(child_plan) = &plan.columns[cid].nested_plan
    {
        let blob = ctx.col_blobs[cid].ok_or(Error::NotLoaded)?;
        ctx.cache[cid] = Some(Col::Flat(decode_nested_column_tagged(
            blob, child_plan, budget,
        )?));
    }
    ensure_dense_block_column(cid, ctx, budget)
}

/// A block's row structure: variant ids, then the JSON structure streams.
pub(crate) enum Structure<'a> {
    Whole(&'a [u8]),
    /// The variant ids, which lead the streams, and each stream. A block whose rows are all of
    /// the model's one JSON variant is read without its variant ids.
    Parts {
        head: Option<&'a [u8]>,
        streams: Vec<Option<&'a [u8]>>,
    },
}

/// The parts of a block a reader holds. A part it did not fetch is `None`.
pub(crate) struct BlockParts<'a> {
    pub structure: Structure<'a>,
    pub dicts: Vec<Option<&'a [u8]>>,
    pub pool: Option<PoolParts<'a>>,
    pub cols: Vec<Option<&'a [u8]>>,
}

impl<'a> BlockParts<'a> {
    /// Splits a whole block.
    pub fn of_block(bytes: &'a [u8], plan: &ModelPlan, budget: &DecodeBudget) -> Result<Self> {
        let mut cur = Cursor::new(bytes);
        decode_u32_seq(&mut cur, budget)?;
        if plan.json_tree.is_some() {
            cur.bytes()?;
        }
        let structure = Structure::Whole(&bytes[..bytes.len() - cur.remaining()]);
        let n_dicts = cur.uvar()? as usize;
        if n_dicts > 4096 {
            return Err(Error::Format("too many block shared dicts".into()));
        }
        let mut dicts = Vec::with_capacity(n_dicts);
        for _ in 0..n_dicts {
            dicts.push(Some(cur.bytes()?));
        }
        let pool = match cur.u8()? {
            0 => None,
            1 => Some(PoolParts::Leaf(cur.bytes()?)),
            2 => {
                let n = cur.uvar()? as usize;
                if n < 2 || n > cur.remaining() {
                    return Err(Error::Format("text pool page count".into()));
                }
                let lens = (0..n).map(|_| cur.uvar()).collect::<Result<Vec<_>>>()?;
                let mut pages = Vec::with_capacity(n);
                for len in lens {
                    let leaf = cur.slice(usize::try_from(len).map_err(|_| underflow())?)?;
                    let raw = u32::try_from(leaf_raw_len(leaf)?)
                        .map_err(|_| Error::BudgetExceeded("text pool"))?;
                    pages.push((Some(leaf), raw));
                }
                Some(PoolParts::Pages(pages))
            }
            m => return Err(Error::Format(format!("invalid block prose_pool flag {m}"))),
        };
        let n_cols = plan.columns.len();
        let mut col_lens = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            col_lens.push(cur.uvar()? as usize);
        }
        let mut cols = Vec::with_capacity(n_cols);
        for cl in col_lens {
            cols.push(Some(cur.slice(cl)?));
        }
        Ok(Self {
            structure,
            dicts,
            pool,
            cols,
        })
    }
}

/// An open block: its row structure, lazily decoded columns, and a cursor over its rows.
pub(crate) struct BlockRows<'a> {
    pub plan: &'a ModelPlan,
    pub variant_ids: Arc<[u32]>,
    pub ctx: BlockDecodeCtx<'a>,
    recon: BlockReconstructor,
    next: usize,
}

impl<'a> BlockRows<'a> {
    pub fn open(
        bytes: &'a [u8],
        n_rows: usize,
        plan: &'a ModelPlan,
        budget: &DecodeBudget,
    ) -> Result<Self> {
        Self::of_parts(
            BlockParts::of_block(bytes, plan, budget)?,
            n_rows,
            plan,
            budget,
        )
    }

    pub fn of_parts(
        parts: BlockParts<'a>,
        n_rows: usize,
        plan: &'a ModelPlan,
        budget: &DecodeBudget,
    ) -> Result<Self> {
        if parts.cols.len() != plan.columns.len() {
            return Err(Error::Format("block column count != model".into()));
        }
        let n_streams = plan.json_tree.as_ref().map_or(0, |jt| jt.n_struct_streams);
        let (variant_ids, json_streams) = match parts.structure {
            Structure::Whole(bytes) => {
                let mut cur = Cursor::new(bytes);
                let variant_ids = decode_u32_seq(&mut cur, budget)?;
                let streams = match &plan.json_tree {
                    Some(_) => decode_json_struct_streams(
                        &mut Cursor::new(cur.bytes()?),
                        n_streams,
                        budget,
                    )?,
                    None => Vec::new(),
                };
                (variant_ids, streams)
            }
            Structure::Parts { head, streams } => {
                if streams.len() != n_streams {
                    return Err(Error::Format(
                        "block structure stream count != model".into(),
                    ));
                }
                let mut out = Vec::with_capacity(n_streams);
                for s in streams {
                    out.push(match s {
                        Some(bytes) => decode_u32_seq(&mut Cursor::new(bytes), budget)?,
                        None => Vec::new(),
                    });
                }
                let variant_ids = match (head, plan.json_tree_variant_id()) {
                    (Some(head), _) => decode_u32_seq(&mut Cursor::new(head), budget)?,
                    (None, Some(vid)) => vec![vid; n_rows],
                    (None, None) => return Err(Error::NotLoaded),
                };
                (variant_ids, out)
            }
        };
        if variant_ids.len() != n_rows {
            return Err(Error::Format("block variant_ids length mismatch".into()));
        }
        if variant_ids
            .iter()
            .any(|&v| v as usize >= plan.variants.len())
        {
            return Err(Error::Format("block variant id out of range".into()));
        }
        let variant_ids: Arc<[u32]> = variant_ids.into();
        Ok(Self {
            plan,
            ctx: BlockDecodeCtx::new(parts.cols, parts.dicts, parts.pool, plan.library.clone())
                .with_layout(plan, Arc::clone(&variant_ids)),
            variant_ids,
            recon: BlockReconstructor::new(plan, json_streams),
            next: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.variant_ids.len()
    }

    pub fn json_streams(&self) -> &[Vec<u32>] {
        self.recon.json_streams()
    }

    pub fn col(&mut self, cid: usize, budget: &DecodeBudget) -> Result<&Col> {
        block_col(cid, self.plan, &mut self.ctx, budget)
    }

    /// Appends the row under the cursor to `out` and moves on.
    pub fn emit(&mut self, out: &mut Vec<u8>, budget: &DecodeBudget) -> Result<()> {
        let &vid = self
            .variant_ids
            .get(self.next)
            .ok_or_else(|| Error::Format("block row underflow".into()))?;
        self.next += 1;
        self.recon
            .reconstruct_next(vid, self.plan, &mut self.ctx, out, budget)
    }

    /// Calls `f(row, record)` for each of `rows`, ascending, from the cursor on. Columns are
    /// built only as far as these rows read them, which takes the cursors around each row first.
    pub fn emit_rows(
        &mut self,
        rows: &[usize],
        budget: &DecodeBudget,
        mut f: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        // Marks cost a copy of the cursors each; a reader of many rows reads most cells anyway.
        let marks = match rows.len() <= 64 {
            true => {
                let (mut marks, mut read) = (Vec::new(), Vec::new());
                for &r in rows {
                    self.seek(r, budget)?;
                    let m = self.recon.mark();
                    self.seek(r + 1, budget)?;
                    read.push((m.0.clone(), self.recon.mark().0));
                    marks.push(m);
                }
                self.ctx.read_only(read);
                marks
            }
            false => Vec::new(),
        };
        let mut marks = marks.into_iter();
        let mut rec = Vec::new();
        for &r in rows {
            match marks.next() {
                Some(m) => {
                    self.recon.reset(m);
                    self.next = r;
                }
                None => self.seek(r, budget)?,
            }
            rec.clear();
            self.emit(&mut rec, budget)?;
            f(r, &rec)?;
        }
        Ok(())
    }

    /// Moves the cursor forward to `row`.
    pub fn seek(&mut self, row: usize, budget: &DecodeBudget) -> Result<()> {
        if row <= self.next {
            return Ok(());
        }
        let vids = self
            .variant_ids
            .get(self.next..row)
            .ok_or_else(|| Error::Format("block row underflow".into()))?;
        self.next = row;
        self.recon.skip_rows(vids, self.plan, &mut self.ctx, budget)
    }
}

pub(crate) fn decode_single_block_records(
    block_bytes: &[u8],
    n_rows: usize,
    plan: &ModelPlan,
    budget: &DecodeBudget,
) -> Result<Vec<Vec<u8>>> {
    let mut rows = BlockRows::open(block_bytes, n_rows, plan, budget)?;
    let mut out = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let mut rec = Vec::with_capacity(128);
        rows.emit(&mut rec, budget)?;
        out.push(rec);
    }
    Ok(out)
}

/// Decodes a single `ZFM1` frame payload in block order and verifies its XXH64.
pub fn decode_frame_payload<W: Write>(
    hdr: &ParsedFrameHeader,
    payload: &[u8],
    expected_xxh64: Option<u64>,
    prev_plan: &mut Option<ModelPlan>,
    budget: &DecodeBudget,
    mut out: W,
) -> Result<()> {
    if (hdr.flags & FLAG_PASSTHROUGH) != 0 {
        let raw = decompress_leaf(&mut Cursor::new(payload), budget)?;
        if raw.len() != hdr.raw_len {
            return Err(Error::Format("passthrough raw_len mismatch".into()));
        }
        if let Some(exp) = expected_xxh64
            && budget.verify_checksums
            && xxh64(&raw, 0x5a46_4d31) != exp
        {
            return Err(Error::ChecksumMismatch("ZFM1 passthrough frame XXH64"));
        }
        out.write_all(&raw)?;
        return Ok(());
    }

    let bdir = parse_frame_blocks(hdr, payload, prev_plan, budget)?;
    let plan = prev_plan.as_ref().unwrap();

    let mut frame_out = budget.alloc_vec(0, "frame output")?;
    frame_out
        .try_reserve(hdr.raw_len.min(budget.max_single_alloc_bytes))
        .map_err(|_| Error::BudgetExceeded("frame output reserve"))?;

    frame_out.extend_from_slice(&bdir.prefix);
    let mut emitted_recs = 0usize;
    for (nrows, bslice) in &bdir.blocks {
        let mut recs = decode_single_block_records(bslice, *nrows, plan, budget)?;
        if (hdr.flags & FLAG_SORTED_ROWS) != 0 {
            sort_records(&mut recs);
        }
        for rec in &recs {
            if emitted_recs > 0 {
                frame_out.extend_from_slice(&bdir.sep);
            }
            emitted_recs += 1;
            let is_last = emitted_recs == hdr.record_count;
            emit_framed_record(&mut frame_out, rec, hdr.framing, is_last);
        }
    }
    frame_out.extend_from_slice(&bdir.suffix);
    if frame_out.len() != hdr.raw_len {
        return Err(Error::Format(format!(
            "frame decoded {} bytes != header raw_len {}",
            frame_out.len(),
            hdr.raw_len
        )));
    }
    if let Some(exp) = expected_xxh64
        && budget.verify_checksums
        && xxh64(&frame_out, 0x5a46_4d31) != exp
    {
        return Err(Error::ChecksumMismatch("ZFM1 frame content XXH64"));
    }
    out.write_all(&frame_out)?;
    Ok(())
}

// ============================================================================
// 3. High-Level Stream Pack / Unpack / Inspect / O(1) Seek Table Reader
// ============================================================================

pub fn pack_stream<R: Read, W: Write>(reader: R, mut writer: W, opts: &PackOptions) -> Result<()> {
    let mut enc = StreamEncoder::new(opts.clone());
    if opts.threads <= 1 {
        for_each_chunk(reader, opts, |fc| {
            enc.encode_framed(fc, &mut writer).map(drop)
        })?;
        return enc.finish(&mut writer);
    }
    // Frames encode `threads` at a time and are written in order. A reuse frame waits for its
    // sync frame's library and seeds, as `encode_framed` hands them on, so the bytes match.
    std::thread::scope(|s| {
        let (tx, rx) = mpsc::channel();
        let mut fl = Inflight {
            queue: VecDeque::new(),
            written: 0,
            running: 0,
            rx,
            dict: opts.dictionary.as_deref(),
        };
        let mut plan: Option<Arc<ModelPlan>> = None;
        let mut group: Option<Arc<Link>> = None;
        // Tables may still be encoding their wire, which only the dictionary frame needs.
        if let Some(d) = fl.dict {
            s.spawn(|| d.id());
        }
        for_each_chunk(reader, opts, |fc| {
            let task = enc.decide(fc);
            if task.plan_changed || plan.is_none() {
                plan = enc.active_plan.clone().map(Arc::new);
            }
            let (wait, publish) = if task.passthrough {
                (None, None)
            } else if task.reuse {
                (group.clone(), None)
            } else {
                let next = Arc::new(Link::default());
                group = Some(Arc::clone(&next));
                (None, Some(next))
            };
            let (c, t, plan, tx) = (OwnedChunk::new(fc), task.clone(), plan.clone(), tx.clone());
            let seq = fl.written + fl.queue.len();
            s.spawn(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let (mut plan, mut seeds) = (plan, Seeds::default());
                    if let Some(l) = wait {
                        let (p, sd) = l.get();
                        (plan, seeds) = (Some(p), sd);
                    }
                    let mut frame = encode_task(opts, &t, &c.view(), plan.as_deref(), &mut seeds);
                    if let Some(l) = publish {
                        let p = frame.plan.take().map(Arc::new).or(plan);
                        l.set(p.expect("a structural frame has a model"), seeds);
                    }
                    frame
                }));
                let _ = tx.send((seq, r));
            });
            // A small frame may fall back to passthrough, which changes the next frame's kind.
            let small = !task.passthrough && fc.raw.len() <= SMALL_FRAME;
            fl.queue.push_back((task, None));
            fl.running += 1;
            // Reuse frames wait on their sync frame, so more frames run than threads.
            let depth = opts.threads + opts.sync_interval_frames;
            loop {
                while let Ok(m) = fl.rx.try_recv() {
                    fl.land(m);
                }
                fl.flush(&mut enc, &mut writer)?;
                if fl.queue.is_empty() || fl.running < depth && !small {
                    return Ok(());
                }
                fl.wait();
            }
        })?;
        while !fl.queue.is_empty() {
            fl.wait();
            fl.flush(&mut enc, &mut writer)?;
        }
        enc.finish(&mut writer)
    })
}

/// Frames being encoded, in stream order from `written`, each with its encoding once done.
struct Inflight<'d> {
    queue: VecDeque<(FrameTask, Option<EncodedFrame>)>,
    written: usize,
    running: usize,
    rx: mpsc::Receiver<(usize, std::thread::Result<EncodedFrame>)>,
    dict: Option<&'d Dictionary>,
}

impl Inflight<'_> {
    fn land(&mut self, (seq, frame): (usize, std::thread::Result<EncodedFrame>)) {
        let frame = frame.unwrap_or_else(|p| std::panic::resume_unwind(p));
        self.queue[seq - self.written].1 = Some(frame);
        self.running -= 1;
    }

    /// Waits for a frame to finish or, with none running, for the dictionary.
    fn wait(&mut self) {
        match self.running {
            0 => drop(self.dict.map(Dictionary::id)),
            _ => self.land(self.rx.recv().expect("a running frame sends")),
        }
    }

    /// Writes the finished frames at the front once the dictionary frame can go first.
    fn flush<W: Write>(&mut self, enc: &mut StreamEncoder, mut out: W) -> Result<()> {
        while self.dict.is_none_or(Dictionary::has_id)
            && self.queue.front().is_some_and(|j| j.1.is_some())
        {
            let (task, frame) = self.queue.pop_front().unwrap();
            enc.start(&mut out)?;
            enc.emit(&task, frame.unwrap(), &mut out)?;
            self.written += 1;
        }
        Ok(())
    }
}

/// A sync frame's model and seeds once it is encoded, for the frames that start from them.
#[derive(Default)]
struct Link(Mutex<Option<(Arc<ModelPlan>, Seeds)>>, Condvar);

impl Link {
    fn set(&self, plan: Arc<ModelPlan>, seeds: Seeds) {
        *self.0.lock().unwrap() = Some((plan, seeds));
        self.1.notify_all();
    }

    fn get(&self) -> (Arc<ModelPlan>, Seeds) {
        let g = self.1.wait_while(self.0.lock().unwrap(), |v| v.is_none());
        let g = g.unwrap();
        let (p, s) = g.as_ref().unwrap();
        (Arc::clone(p), s.clone())
    }
}

/// A framed chunk that owns its bytes, to encode on another thread.
struct OwnedChunk {
    framing: ResolvedFraming,
    prefix: Vec<u8>,
    suffix: Vec<u8>,
    spans: Vec<(u32, u32)>,
    raw: Vec<u8>,
}

impl OwnedChunk {
    fn new(fc: &FramedChunk<'_>) -> Self {
        Self {
            framing: fc.framing,
            prefix: fc.prefix.to_vec(),
            suffix: fc.suffix.to_vec(),
            spans: fc.spans.clone(),
            raw: fc.raw.to_vec(),
        }
    }

    fn view(&self) -> FramedChunk<'_> {
        FramedChunk {
            framing: self.framing,
            prefix: &self.prefix,
            suffix: &self.suffix,
            spans: self.spans.clone(),
            raw: &self.raw,
        }
    }
}

/// The dictionary of what `pack_stream` would write from `reader`, within `max_bytes`: `None` when
/// `opts` names no profile or no value recurs across blocks.
pub fn build_dictionary<R: Read>(
    reader: R,
    opts: &PackOptions,
    max_bytes: usize,
) -> Result<Option<Dictionary>> {
    let Some(mut b) = DictionaryBuilder::new(opts, max_bytes) else {
        return Ok(None);
    };
    for_each_chunk(reader, opts, |fc| {
        (0..fc.len()).for_each(|i| b.push(fc.record(i)));
        Ok(())
    })?;
    Ok(b.finish())
}

/// The table `opts`' packs look values up in, from a pass over `reader`: `None` when no pack
/// keeps one or no record adds to it.
pub fn build_tables<R: Read>(reader: R, opts: &PackOptions) -> Result<Option<Tables>> {
    let Some(mut b) = opts.packs.table_builder() else {
        return Ok(None);
    };
    for_each_chunk(reader, opts, |fc| {
        b.push(fc.raw, &fc.spans);
        Ok(())
    })?;
    let cols = b.finish();
    if cols.first().is_none_or(|c| c.is_empty()) {
        return Ok(None);
    }
    Ok(Some(opts.packs.build_tables(cols)))
}

/// Cuts `reader` into the framed chunks `pack_stream` encodes, in order.
fn for_each_chunk<R: Read>(
    mut reader: R,
    opts: &PackOptions,
    mut f: impl FnMut(&FramedChunk<'_>) -> Result<()>,
) -> Result<()> {
    let chunk_cap = opts.chunk_bytes.max(64 << 10);
    let mut first = true;
    let mut carry = Vec::<u8>::new();
    let mut io_buf = vec![0u8; 256 << 10];
    // JSON document state: `Some(continued)` while the input may be, or is inside, a document's array.
    let mut json = matches!(opts.framing, FramingSpec::Auto | FramingSpec::Json).then_some(false);

    loop {
        while carry.len() < chunk_cap {
            let n = reader.read(&mut io_buf)?;
            if n == 0 {
                break;
            }
            carry.extend_from_slice(&io_buf[..n]);
        }
        if carry.is_empty() {
            break;
        }
        if let Some(continued) = json {
            json = None;
            if let Some(js) = split_json_array(&carry, continued) {
                // Cut after the last complete element; once the array or the input ends, take the rest as suffix.
                let more = !js.closed && carry.len() >= chunk_cap;
                let end = if more { js.cut() } else { carry.len() };
                json = more.then_some(true);
                f(&js.chunk(&carry, end))?;
                first = false;
                carry.drain(..end);
                continue;
            }
        }
        // Cut at last record boundary (`\n` or `\0`) within `chunk_cap` so records aren't split mid-line!
        let split_at = if carry.len() >= chunk_cap {
            let win = &carry[..chunk_cap];
            match opts.framing {
                FramingSpec::FixedStride {
                    stride,
                    header_bytes,
                } => {
                    let h = if first { header_bytes as usize } else { 0 };
                    let s = stride.max(1) as usize;
                    if win.len() > h + s {
                        h + ((win.len() - h) / s) * s
                    } else {
                        carry.len()
                    }
                }
                FramingSpec::Nul => win
                    .iter()
                    .rposition(|&b| b == 0)
                    .map(|p| p + 1)
                    .or_else(|| {
                        carry[chunk_cap..]
                            .iter()
                            .position(|&b| b == 0)
                            .map(|p| chunk_cap + p + 1)
                    })
                    .unwrap_or(carry.len()),
                FramingSpec::LengthU32Le => carry.len(),
                _ => win
                    .iter()
                    .rposition(|&b| b == b'\n')
                    .map(|p| p + 1)
                    .or_else(|| {
                        carry[chunk_cap..]
                            .iter()
                            .position(|&b| b == b'\n')
                            .map(|p| chunk_cap + p + 1)
                    })
                    .unwrap_or(carry.len()),
            }
        } else {
            carry.len()
        };

        f(&split_chunk(&carry[..split_at], opts.framing))?;
        first = false;
        carry.drain(..split_at);
    }
    Ok(())
}

pub fn unpack_stream<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    budget: &DecodeBudget,
) -> Result<()> {
    let mut prev_plan: Option<ModelPlan> = None;
    let mut dict = Scope::enter(None);
    loop {
        match read_next_frame_header(&mut reader, budget)? {
            NextFrameKind::Eof => return Ok(()),
            NextFrameKind::Index { skip_bytes } => {
                std::io::copy(
                    &mut (&mut reader).take(skip_bytes as u64),
                    &mut std::io::sink(),
                )?;
                // A stream concatenated after this one starts with its own model and dictionary.
                prev_plan = None;
                dict.set(None);
            }
            NextFrameKind::Data(hdr) => {
                let mut payload = budget.alloc_vec(hdr.payload_len, "frame payload")?;
                reader.read_exact(&mut payload)?;
                let exp_xxh = if (hdr.flags & FLAG_HAS_CHECKSUM) != 0 {
                    let mut cb = [0u8; 8];
                    reader.read_exact(&mut cb)?;
                    Some(u64::from_le_bytes(cb))
                } else {
                    None
                };
                if (hdr.flags & FLAG_DICTIONARY) != 0 {
                    let id = exp_xxh
                        .ok_or_else(|| Error::Format("dictionary frame has no id".into()))?;
                    dict.set(Some(Arc::new(Dictionary::read(&payload, id, budget)?)));
                    continue;
                }
                decode_frame_payload(&hdr, &payload, exp_xxh, &mut prev_plan, budget, &mut writer)?;
            }
        }
    }
}

pub use crate::inspect::inspect_stream;

#[cfg(test)]
mod nested_tests {
    use super::*;

    #[test]
    fn nested_messages_keep_plain_bytes_and_json_tags_when_model_is_unhelpful() {
        let train: Vec<Vec<u8>> = (0..8)
            .map(|i| format!("a={i} b={}", i * 3).into_bytes())
            .collect();
        let refs: Vec<&[u8]> = train.iter().map(Vec::as_slice).collect();
        let plan = crate::block::train(&refs, &TrainOptions::default(), None);
        for tagged in [false, true] {
            let rows: Vec<Vec<u8>> = (0..32)
                .map(|i| {
                    let mut row = Vec::new();
                    if tagged {
                        row.push((i % 2) as u8);
                    }
                    row.extend_from_slice(b"a wholly different message with \"quotes\"");
                    row
                })
                .collect();
            let vals: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
            let packs = PackRegistry::default().with_plain_nested_fallback();
            let (_, wire) =
                encode_nested_column_tagged(&vals, tagged, &plan, &packs, EntropyBackend::Zstd, 3);
            let back = decode_nested_column_tagged(&wire, &plan, &DecodeBudget::default()).unwrap();
            for (i, want) in rows.iter().enumerate() {
                assert_eq!(back.get(i).unwrap(), want);
            }
        }
    }
}
