//! The `ZFM1` framed stream, which is a file's data region. A frame carries a model, or reuses the
//! previous frame's, and self-contained blocks. The encoder collects the index region as it goes.

use crate::cells::{Cells, Col};
use crate::codecs::{
    ANY_LETTERS, BlockDecodeCtx, ColSummary, Seeds, column_deps, decode_present_slice,
    decode_u32_seq, encode_dense_block_columns, encode_present_slice, encode_u32_seq,
    ensure_dense_block_column, with_seeds,
};
use crate::framing::{
    FramedChunk, FramingSpec, ResolvedFraming, emit_framed_record, split_chunk, split_json_array,
};
use crate::grammar::{BlockReconstructor, ModelPlan, OccRef, PlanVariant, split_records_with_plan};
use crate::index::{
    BlockDir, BlockKind, ColEntry, FilterBuilder, INDEX_MAGIC, IndexBuilder, KIND_BINARY, KIND_CSV,
    KIND_ENVELOPE, KIND_JSON, KIND_LOGFMT, KIND_RESIDUAL, KIND_TEMPLATE, tokens,
};
use crate::json_tree::{decode_json_struct_streams, encode_json_struct_streams};
use crate::lex::parse_base64;
use crate::plan::{ProbeBackoff, Profile, TrainOptions, sample_strided, train_from_records};
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, compress_leaf, crc32, decompress_leaf,
    put_bytes, put_uvarint, xxh64,
};
use std::io::{Read, Write};

pub const FRAME_MAGIC: &[u8; 4] = b"ZFM1";

pub const FLAG_HAS_CHECKSUM: u16 = 1 << 0;
pub const FLAG_PASSTHROUGH: u16 = 1 << 1;
pub const FLAG_REUSE_PREV_MODEL: u16 = 1 << 2;
pub const FLAG_SYNC_POINT: u16 = 1 << 3;

pub const DEFAULT_BLOCK_ROWS: usize = 65_536;
pub const DEFAULT_BLOCK_BYTES: usize = 8 << 20;
/// Largest substring filter of one block, in bits. A filter shrinks to fit what its block holds.
pub const DEFAULT_FILTER_BITS: usize = 1 << 17;

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
    /// Allow online retraining when residual exceeds `drift_residual_ratio` (even if `profile` was supplied when `allow_profile_drift` is true).
    pub allow_drift_retrain: bool,
    pub drift_residual_ratio: f64,
    /// Number of consecutive frames allowed to reuse a model (`REUSE_PREV_MODEL`) before forcing an independent `SYNC_POINT`.
    pub sync_interval_frames: usize,
    pub checksum: bool,
    /// Build the index region; `StreamEncoder::finish` writes it after the last frame.
    pub index: bool,
    pub filter_bits: usize,
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
            allow_drift_retrain: true,
            drift_residual_ratio: 0.10,
            sync_interval_frames: 8,
            checksum: true,
            index: true,
            filter_bits: DEFAULT_FILTER_BITS,
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

    /// Compresses one already-framed chunk into a `ZFM1` frame written to `out`.
    pub(crate) fn encode_framed<W: Write>(
        &mut self,
        fc: &FramedChunk<'_>,
        mut out: W,
    ) -> Result<FrameFit> {
        let raw = fc.raw;
        let all_recs: Vec<&[u8]> = (0..fc.len()).map(|i| fc.record(i)).collect();
        let rec_bytes: usize = all_recs.iter().map(|r| r.len()).sum();
        let mut fit = FrameFit {
            rows: all_recs.len(),
            bytes: rec_bytes,
            residual_rows: all_recs.len(),
            residual_bytes: rec_bytes,
        };

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
                let pilot_bytes: usize = pilot.iter().map(|r| r.len()).sum();
                let row_ratio = ps.residual_rows as f64 / pilot.len().max(1) as f64;
                let byte_ratio = ps.residual_bytes as f64 / pilot_bytes.max(1) as f64;
                if (row_ratio > self.opts.drift_residual_ratio
                    || byte_ratio > self.opts.drift_residual_ratio)
                    && self.backoff.should_probe()
                {
                    let samples = sample_strided(
                        &all_recs,
                        self.opts.train.max_sample_rows,
                        self.opts.train.max_sample_bytes,
                    );
                    let extended =
                        train_from_records(&samples, fc.framing, &self.opts.train, Some(cur_plan));
                    let new_ps = split_records_with_plan(&pilot, &extended);
                    if new_ps.residual_bytes + 64 < ps.residual_bytes {
                        self.active_plan = Some(extended);
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
                plan_changed = true;
                self.backoff.record_success();
            } else {
                self.backoff.record_failure();
            }
        }

        // Decide Passthrough vs Structural
        let use_passthrough = match &self.active_plan {
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

        let is_sync = use_passthrough
            || plan_changed
            || self.opts.sync_interval_frames <= 1
            || self.frames_since_sync >= self.opts.sync_interval_frames - 1;
        let reuse_prev = !use_passthrough && !is_sync;
        let filter_bits = self.opts.index.then_some(self.opts.filter_bits);

        let mut payload = Vec::new();
        let mut blocks = Vec::new();
        if use_passthrough {
            compress_leaf(&mut payload, raw, self.opts.backend, self.opts.level);
        } else {
            let plan = self.active_plan.as_ref().unwrap();
            if !reuse_prev {
                let mut raw_plan = Vec::new();
                plan.encode(&mut raw_plan);
                compress_leaf(
                    &mut payload,
                    &raw_plan,
                    self.opts.backend,
                    self.opts.level.max(3),
                );
                if self.opts.index {
                    self.model_idx = self.index.model(&raw_plan);
                }
            }
            if has_affixes(fc.framing) {
                compress_leaf(&mut payload, fc.prefix, self.opts.backend, self.opts.level);
                compress_leaf(&mut payload, fc.suffix, self.opts.backend, self.opts.level);
            }
            if fc.framing == ResolvedFraming::JsonArray {
                compress_leaf(&mut payload, fc.sep(), self.opts.backend, self.opts.level);
            }
            blocks = with_seeds(&mut self.seeds, || {
                encode_structural_blocks(
                    &all_recs,
                    plan,
                    self.opts.backend,
                    self.opts.level,
                    self.opts.block_rows,
                    self.opts.block_bytes,
                    filter_bits,
                    &mut payload,
                )
            });
            fit.residual_rows = blocks.iter().map(|b| b.residual_rows).sum();
            fit.residual_bytes = blocks.iter().map(|b| b.residual_bytes).sum();
        }

        // A structural payload larger than the plain leaf, as tiny or degenerate inputs give, is
        // dropped for the leaf.
        let mut final_passthrough = use_passthrough;
        if !use_passthrough && raw.len() <= (1 << 20) {
            let mut pt_trial = Vec::new();
            compress_leaf(&mut pt_trial, raw, self.opts.backend, self.opts.level);
            if pt_trial.len() + 32 < payload.len() {
                payload = pt_trial;
                final_passthrough = true;
                (fit.residual_rows, fit.residual_bytes) = (fit.rows, fit.bytes);
            }
        }

        let actual_sync = if final_passthrough { true } else { is_sync };
        let actual_reuse = if final_passthrough { false } else { reuse_prev };
        if final_passthrough {
            // Don't reset `active_plan`, just force next structural frame to re-sync
            self.frames_since_sync = usize::MAX;
        } else if actual_sync {
            self.frames_since_sync = 0;
        } else {
            self.frames_since_sync += 1;
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

        let mut hdr = Vec::with_capacity(48);
        hdr.extend_from_slice(FRAME_MAGIC);
        hdr.extend_from_slice(&flags.to_le_bytes());
        hdr.push(self.opts.backend as u8);
        put_uvarint(&mut hdr, raw.len() as u64);
        put_uvarint(&mut hdr, all_recs.len() as u64);
        fc.framing.encode(&mut hdr);
        hdr.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let h_crc = crc32(&hdr);
        hdr.extend_from_slice(&h_crc.to_le_bytes());

        let tail_len = if self.opts.checksum { 8usize } else { 0usize };
        let total_frame_bytes = (hdr.len() + payload.len() + tail_len) as u64;

        out.write_all(&hdr)?;
        out.write_all(&payload)?;
        if self.opts.checksum {
            let csum = xxh64(raw, 0x5a46_4d31);
            out.write_all(&csum.to_le_bytes())?;
        }

        let base = self.comp_offset + hdr.len() as u64;
        if !self.opts.index {
        } else if !final_passthrough {
            for b in &blocks {
                let Some((dir, filter, kinds)) = &b.index else {
                    continue;
                };
                let kind = BlockKind::Structured {
                    model: self.model_idx,
                    kinds: *kinds,
                };
                self.index.block(
                    kind,
                    base + b.off as u64,
                    b.len as u64,
                    b.rows as u64,
                    dir,
                    filter,
                );
            }
        } else if !all_recs.is_empty() {
            let mut fb = FilterBuilder::new(self.opts.filter_bits);
            for r in &all_recs {
                tokens(r, |t, _| fb.add(t));
            }
            let kind = BlockKind::Verbatim {
                framing: fc.framing,
            };
            self.index.block(
                kind,
                base,
                payload.len() as u64,
                all_recs.len() as u64,
                &[],
                &fb.finish(),
            );
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

/// One block inside a frame payload.
pub(crate) struct FrameBlock {
    pub rows: usize,
    pub off: usize,
    pub len: usize,
    pub residual_rows: usize,
    pub residual_bytes: usize,
    pub index: Option<(Vec<u8>, Vec<u8>, u16)>,
}

#[allow(clippy::too_many_arguments)]
fn encode_structural_blocks(
    all_recs: &[&[u8]],
    plan: &ModelPlan,
    backend: EntropyBackend,
    level: i32,
    max_block_rows: usize,
    max_block_bytes: usize,
    filter_bits: Option<usize>,
    out: &mut Vec<u8>,
) -> Vec<FrameBlock> {
    let mut encoded = Vec::new();
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
        encoded.push((
            e - s,
            encode_block_fit(&all_recs[s..e], plan, backend, level, filter_bits),
        ));
        s = e;
    }

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
    blocks
}

pub(crate) struct EncodedBlock {
    pub bytes: Vec<u8>,
    /// Rows and bytes the model left residual.
    pub residual_rows: usize,
    pub residual_bytes: usize,
    /// Directory page, filter page and row kinds, when a filter size was given.
    pub index: Option<(Vec<u8>, Vec<u8>, u16)>,
}

/// What kinds of row a block holds, as `index::KIND_*` bits.
fn row_kinds(plan: &ModelPlan, variant_ids: &[u32]) -> u16 {
    let mut seen = vec![false; plan.variants.len()];
    for &v in variant_ids {
        seen[v as usize] = true;
    }
    let mut kinds = 0u16;
    for (_, v) in plan.variants.iter().enumerate().filter(|(vi, _)| seen[*vi]) {
        kinds |= match v {
            PlanVariant::Residual { .. } => KIND_RESIDUAL,
            PlanVariant::JsonTree => KIND_JSON,
            PlanVariant::Envelope(_) => KIND_ENVELOPE,
            PlanVariant::Logfmt => KIND_LOGFMT,
            PlanVariant::Template(_) => KIND_TEMPLATE,
            PlanVariant::Csv(_) => KIND_CSV,
            PlanVariant::BinaryStride(_) => KIND_BINARY,
        };
    }
    kinds
}

/// Encodes one block. `filter_bits` asks for its index pages as well.
pub(crate) fn encode_block_fit(
    recs: &[&[u8]],
    plan: &ModelPlan,
    backend: EntropyBackend,
    level: i32,
    filter_bits: Option<usize>,
) -> EncodedBlock {
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

    let enc = encode_dense_block_columns(
        &occ_views,
        &dict_groups,
        &horiz_cands,
        &mut preencoded_nested,
        &standalone,
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
    put_uvarint(&mut out, enc.shared_dict_blobs.len() as u64);
    for db in &enc.shared_dict_blobs {
        put_uvarint(&mut out, db.len() as u64);
        dir.dicts.push((out.len() as u32, db.len() as u32));
        out.extend_from_slice(db);
    }
    if let Some(pl) = &enc.prose_pool_leaf {
        out.push(1);
        put_uvarint(&mut out, pl.len() as u64);
        dir.pool = Some((out.len() as u32, pl.len() as u32));
        out.extend_from_slice(pl);
    } else {
        out.push(0);
    }
    for blob in &enc.col_blobs {
        put_uvarint(&mut out, blob.len() as u64);
    }
    let cols_off = out.len() as u32;
    let mut cols = Vec::with_capacity(enc.col_blobs.len());
    let per_row = split.per_row.iter();
    for (((blob, mut summary), col), &per_row) in enc
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

    let index = filter_bits.map(|bits| {
        let mut fb = FilterBuilder::new(bits);
        for r in recs {
            tokens(r, |t, _| fb.add(t));
        }
        let mut page = Vec::new();
        dir.encode(backend, level, &mut page);
        (page, fb.finish(), row_kinds(plan, &split.variant_ids))
    });
    EncodedBlock {
        bytes: out,
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
    out.push(4u8); // h_mode = 4 (NestedPlan)
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
        encode_block_fit(&child_recs, child_plan, backend, level, None).bytes
    } else {
        Vec::new()
    };
    put_bytes(&mut out, &child_blob);
    if !fallbacks.is_empty() {
        let (_, _, _, fb_enc) = encode_present_slice(&fallbacks, backend, level, 1);
        out.extend_from_slice(&fb_enc);
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
    if h_mode != 4 {
        return Err(Error::Format("expected h_mode=4 for nested column".into()));
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
    pub pool: Option<&'a [u8]>,
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
            1 => Some(cur.bytes()?),
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
    pub variant_ids: Vec<u32>,
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
        Ok(Self {
            plan,
            variant_ids,
            ctx: BlockDecodeCtx::new(parts.cols, parts.dicts, parts.pool),
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

    /// Moves past the row under the cursor.
    pub fn skip(&mut self, budget: &DecodeBudget) -> Result<()> {
        let &vid = self
            .variant_ids
            .get(self.next)
            .ok_or_else(|| Error::Format("block row underflow".into()))?;
        self.next += 1;
        self.recon.skip_next(vid, self.plan, &mut self.ctx, budget)
    }

    /// Moves the cursor forward to `row`.
    pub fn seek(&mut self, row: usize, budget: &DecodeBudget) -> Result<()> {
        while self.next < row {
            self.skip(budget)?;
        }
        Ok(())
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
    let mut rec_scratch = Vec::with_capacity(512);
    for (nrows, bslice) in &bdir.blocks {
        let mut rows = BlockRows::open(bslice, *nrows, plan, budget)?;
        for _ in 0..*nrows {
            rec_scratch.clear();
            rows.emit(&mut rec_scratch, budget)?;
            if emitted_recs > 0 {
                frame_out.extend_from_slice(&bdir.sep);
            }
            emitted_recs += 1;
            let is_last = emitted_recs == hdr.record_count;
            emit_framed_record(&mut frame_out, &rec_scratch, hdr.framing, is_last);
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

pub fn pack_stream<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    opts: &PackOptions,
) -> Result<()> {
    let chunk_cap = opts.chunk_bytes.max(64 << 10);
    let mut enc = StreamEncoder::new(opts.clone());
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
                enc.encode_framed(&js.chunk(&carry, end), &mut writer)?;
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
                    let h = if enc.frames() == 0 {
                        header_bytes as usize
                    } else {
                        0
                    };
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

        enc.encode_frame(&carry[..split_at], &mut writer)?;
        carry.drain(..split_at);
    }
    enc.finish(&mut writer)?;
    Ok(())
}

pub fn unpack_stream<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    budget: &DecodeBudget,
) -> Result<()> {
    let mut prev_plan: Option<ModelPlan> = None;
    loop {
        match read_next_frame_header(&mut reader, budget)? {
            NextFrameKind::Eof => return Ok(()),
            NextFrameKind::Index { skip_bytes } => {
                std::io::copy(
                    &mut (&mut reader).take(skip_bytes as u64),
                    &mut std::io::sink(),
                )?;
                // A stream concatenated after this one starts with its own model.
                prev_plan = None;
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
                decode_frame_payload(&hdr, &payload, exp_xxh, &mut prev_plan, budget, &mut writer)?;
            }
        }
    }
}

pub fn inspect_stream<R: Read>(mut reader: R, budget: &DecodeBudget) -> Result<InspectReport> {
    let mut rep = InspectReport::default();
    let mut prev_plan: Option<ModelPlan> = None;
    loop {
        match read_next_frame_header(&mut reader, budget)? {
            NextFrameKind::Eof => break,
            NextFrameKind::Index { skip_bytes } => {
                rep.index_bytes += skip_bytes as u64 + 12;
                std::io::copy(
                    &mut (&mut reader).take(skip_bytes as u64),
                    &mut std::io::sink(),
                )?;
                prev_plan = None;
            }
            NextFrameKind::Data(hdr) => {
                rep.frames += 1;
                if (hdr.flags & FLAG_SYNC_POINT) != 0 {
                    rep.sync_frames += 1;
                }
                if (hdr.flags & FLAG_PASSTHROUGH) != 0 {
                    rep.passthrough_frames += 1;
                }
                rep.total_uncompressed_bytes += hdr.raw_len as u64;
                rep.total_records += hdr.record_count as u64;
                let mut payload = budget.alloc_vec(hdr.payload_len, "inspect payload")?;
                reader.read_exact(&mut payload)?;
                let mut extra = 24u64;
                if (hdr.flags & FLAG_HAS_CHECKSUM) != 0 {
                    let mut cb = [0u8; 8];
                    reader.read_exact(&mut cb)?;
                    extra += 8;
                }
                rep.total_compressed_bytes += hdr.payload_len as u64 + extra;
                if (hdr.flags & FLAG_PASSTHROUGH) == 0 {
                    let bdir = parse_frame_blocks(&hdr, &payload, &mut prev_plan, budget)?;
                    if let Some(p) = &prev_plan {
                        rep.variant_count = rep.variant_count.max(p.variants.len());
                        for c in &p.columns {
                            if !rep.column_names.contains(&c.name) {
                                rep.column_names.push(c.name.clone());
                            }
                        }
                        let add_stat = |rep: &mut InspectReport, name: &str, nb: u64| {
                            if nb == 0 {
                                return;
                            }
                            if let Some(ent) = rep.column_bytes.iter_mut().find(|x| x.0 == name) {
                                ent.1 += nb;
                            } else {
                                rep.column_bytes.push((name.to_string(), nb));
                            }
                        };
                        for &(nrows, bslice) in &bdir.blocks {
                            let _ = nrows;
                            if let Ok(pb) = BlockParts::of_block(bslice, p, budget) {
                                if let Structure::Whole(s) = pb.structure {
                                    add_stat(&mut rep, "$structure", s.len() as u64);
                                }
                                let sd_bytes: u64 =
                                    pb.dicts.iter().flatten().map(|b| b.len() as u64).sum();
                                add_stat(&mut rep, "$shared_dicts", sd_bytes);
                                add_stat(
                                    &mut rep,
                                    "$prose_pool",
                                    pb.pool.map_or(0, |b| b.len() as u64),
                                );
                                for (cid, blob) in pb.cols.iter().enumerate() {
                                    add_stat(
                                        &mut rep,
                                        &p.columns[cid].name,
                                        blob.map_or(0, |b| b.len() as u64),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(rep)
}
