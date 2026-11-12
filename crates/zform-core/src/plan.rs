//! Strided sampler, multi-arm `ModelPlan` trainer (`JsonTreePlan`, Envelopes, `logfmt`,
//! CSV/TSV, Fixed-Stride Binary, and Rich Plain-Text Templates with constant pinning,
//! context-keyed semantic slots, and variable-word refinement), dense horizontal candidate
//! discovery, `ProbeBackoff`, and `ZFP1` profile serialization.

use crate::codecs::encode_present_slice;
use crate::framing::{FramingSpec, ResolvedFraming, split_chunk};
use crate::grammar::{
    BinaryStrideSpec, CsvSpec, EnvelopeKind, EnvelopeSpec, LogfmtSpec, ModelPlan, PlanVariant,
    SlotClass, TemplateVariant, match_template, parse_k8s_cri, parse_logfmt_line,
    parse_syslog_envelope, split_csv_exact, split_records_with_plan, tokenize_template_line,
};
use crate::json_tree::{JsonDoc, train_json_tree};
use crate::lex::parse_exact_int;
use crate::wire::{
    Cursor, DecodeBudget, EntropyBackend, Error, Result, compress_leaf, decompress_leaf,
    sha256_hex, xxh64,
};
use std::collections::{BTreeMap, HashMap, HashSet};

pub const PROFILE_MAGIC: &[u8; 4] = b"ZFP1";
pub const SAMPLE_RUN_ROWS: usize = 16;

#[derive(Debug, Clone)]
pub struct TrainOptions {
    pub max_sample_rows: usize,
    pub max_sample_bytes: usize,
    pub max_json_paths: usize,
    pub max_json_shapes: usize,
    pub max_templates: usize,
    pub framing: FramingSpec,
    /// Recursion depth (`0` = top-level, `1` = nested JSON/envelope body).
    pub depth: u8,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            max_sample_rows: 16_384,
            max_sample_bytes: 16 << 20,
            max_json_paths: 1024,
            max_json_shapes: 64,
            max_templates: 128,
            framing: FramingSpec::Auto,
            depth: 0,
        }
    }
}

/// Self-describing serialized `ZFP1` profile.
#[derive(Debug, Clone)]
pub struct Profile {
    pub plan: ModelPlan,
    pub bytes: Vec<u8>,
    pub id_sha256: String,
}

impl Profile {
    pub fn from_plan(plan: ModelPlan) -> Self {
        let mut raw_plan = Vec::new();
        plan.encode(&mut raw_plan);
        let mut bytes = Vec::with_capacity(raw_plan.len() + 24);
        bytes.extend_from_slice(PROFILE_MAGIC);
        bytes.extend_from_slice(&1u16.to_le_bytes());
        compress_leaf(&mut bytes, &raw_plan, EntropyBackend::Zstd, 9);
        let csum = xxh64(&raw_plan, 0x5a46_5031);
        bytes.extend_from_slice(&csum.to_le_bytes());
        let id_sha256 = sha256_hex(&bytes);
        Self {
            plan,
            bytes,
            id_sha256,
        }
    }

    pub fn from_bytes(bytes: &[u8], budget: &DecodeBudget) -> Result<Self> {
        if bytes.len() < 16 || &bytes[0..4] != PROFILE_MAGIC {
            return Err(Error::Format("invalid ZFP1 profile magic".into()));
        }
        let mut cur = Cursor::new(&bytes[4..]);
        let ver = cur.u16_le()?;
        if ver != 1 {
            return Err(Error::Format(format!("unsupported ZFP1 version {ver}")));
        }
        let raw_plan = decompress_leaf(&mut cur, budget)?;
        let expected_xxh = cur.u64_le()?;
        if budget.verify_checksums && xxh64(&raw_plan, 0x5a46_5031) != expected_xxh {
            return Err(Error::ChecksumMismatch("ZFP1 profile"));
        }
        let plan = ModelPlan::decode(&mut Cursor::new(&raw_plan), 0)?;
        let id_sha256 = sha256_hex(bytes);
        Ok(Self {
            plan,
            bytes: bytes.to_vec(),
            id_sha256,
        })
    }
}

/// Exponential backoff state (`1, 2, 4, 8, 16` chunks) when online training finds unstructured/random streams.
#[derive(Debug, Clone, Default)]
pub struct ProbeBackoff {
    skip_remaining: u32,
    streak: u32,
}

impl ProbeBackoff {
    pub fn should_probe(&mut self) -> bool {
        if self.skip_remaining > 0 {
            self.skip_remaining -= 1;
            false
        } else {
            true
        }
    }

    pub fn record_failure(&mut self) {
        self.streak = (self.streak + 1).min(5);
        self.skip_remaining = (1u32 << self.streak) - 1;
    }

    pub fn record_success(&mut self) {
        self.streak = 0;
        self.skip_remaining = 0;
    }
}

/// Selects strided 16-row micro-runs across `all_records` (avoiding head-of-file bias!).
pub fn sample_strided<'a>(
    all_records: &[&'a [u8]],
    max_rows: usize,
    max_bytes: usize,
) -> Vec<&'a [u8]> {
    if all_records.is_empty() || max_rows == 0 {
        return Vec::new();
    }
    if all_records.len() <= max_rows {
        let mut out = Vec::with_capacity(all_records.len());
        let mut b = 0usize;
        for &r in all_records {
            if b + r.len() > max_bytes && !out.is_empty() {
                break;
            }
            b += r.len();
            out.push(r);
        }
        return out;
    }
    let n_runs = (max_rows / SAMPLE_RUN_ROWS).max(1);
    let max_start = all_records.len().saturating_sub(SAMPLE_RUN_ROWS);
    let mut out = Vec::with_capacity(max_rows);
    let mut total_b = 0usize;
    for r_idx in 0..n_runs {
        let start = if n_runs == 1 {
            0
        } else {
            (r_idx * max_start) / (n_runs - 1)
        };
        for &rec in &all_records[start..(start + SAMPLE_RUN_ROWS).min(all_records.len())] {
            if out.len() >= max_rows || (total_b + rec.len() > max_bytes && !out.is_empty()) {
                return out;
            }
            total_b += rec.len();
            out.push(rec);
        }
    }
    out
}

/// Trains a `ModelPlan` on `raw` bytes (or extends `prior`, evicting stale unused variants if needed).
pub fn train_from_bytes(raw: &[u8], opts: &TrainOptions, prior: Option<&ModelPlan>) -> ModelPlan {
    let fc = split_chunk(raw, opts.framing);
    let recs: Vec<&[u8]> = (0..fc.len()).map(|i| fc.record(i)).collect();
    let samples = sample_strided(&recs, opts.max_sample_rows, opts.max_sample_bytes);
    train_from_records(&samples, fc.framing, opts, prior)
}

pub fn train_from_records(
    samples: &[&[u8]],
    framing: ResolvedFraming,
    opts: &TrainOptions,
    prior: Option<&ModelPlan>,
) -> ModelPlan {
    if samples.is_empty() {
        return prior.cloned().unwrap_or_else(ModelPlan::residual_only);
    }

    let mut plan = ModelPlan::residual_only();

    // 1. Fixed-Stride Binary Arm (when framing is FixedStride)
    if let ResolvedFraming::FixedStride { stride, .. } = framing {
        train_binary_stride(samples, stride, &mut plan);
        plan.rebuild_indexes();
        return plan;
    }

    // Retain active prior Template variants when extending across a drifting stream
    if let Some(prev) = prior {
        let prev_split = split_records_with_plan(samples, prev);
        let mut var_hits = vec![0usize; prev.variants.len()];
        for &vid in &prev_split.variant_ids {
            var_hits[vid as usize] += 1;
        }
        let keep_all_prior = prev.template_indices.len() < opts.max_templates / 2;
        for (vi, v) in prev.variants.iter().enumerate().skip(1) {
            if !keep_all_prior && var_hits[vi] == 0 {
                continue;
            }
            if let PlanVariant::Template(tv) = v {
                let slots: Vec<u16> = tv
                    .slots
                    .iter()
                    .map(|&old_c| {
                        let oc = &prev.columns[old_c as usize];
                        plan.add_or_get_col_with_group(&oc.name, &oc.dict_group)
                    })
                    .collect();
                let tv = TemplateVariant::new(tv.literals.clone(), slots);
                plan.variants.push(PlanVariant::Template(tv));
            }
        }
    }

    // Partition sample rows into candidate buckets simultaneously (Zero 90% JSON cliff!)
    let mut jdoc = JsonDoc::default();
    let mut json_rows = Vec::<&[u8]>::new();
    let mut cri_inners = Vec::<&[u8]>::new();
    let mut syslog_inners = Vec::<&[u8]>::new();
    let mut logfmt_rows = Vec::<&[u8]>::new();
    let mut plain_rows = Vec::<&[u8]>::new();

    for &line in samples {
        let first = line.iter().copied().find(|b| !b.is_ascii_whitespace());
        if first == Some(b'{') && jdoc.parse_root_object(line) {
            json_rows.push(line);
        } else if opts.depth == 0
            && let Some((_, _, _, inner)) = parse_k8s_cri(line)
        {
            cri_inners.push(inner);
        } else if opts.depth == 0
            && let Some((_, _, _, inner)) = parse_syslog_envelope(line)
        {
            syslog_inners.push(inner);
        } else if parse_logfmt_line(line).is_some() {
            logfmt_rows.push(line);
        } else {
            plain_rows.push(line);
        }
    }

    // 2. Train Hierarchical JSON Tree (`JsonTreePlan` + cross-path derivations + nested JSON-in-string!)
    if json_rows.len() >= 2 {
        if let Some(jt) = train_json_tree(&json_rows, opts, &mut plan) {
            for pdef in &jt.paths {
                if pdef.value_col != u16::MAX && !pdef.dict_group.is_empty() {
                    plan.columns[pdef.value_col as usize].dict_group = pdef.dict_group.clone();
                }
            }
            plan.json_tree = Some(jt);
            plan.variants.push(PlanVariant::JsonTree);
        } else if let Some(prev) = prior
            && let Some(prev_jt) = &prev.json_tree
        {
            // Carry over prior JsonTree if sample had <2 JSON rows during a temporary non-JSON burst
            let mut cloned_jt = prev_jt.clone();
            for pdef in &mut cloned_jt.paths {
                if pdef.value_col != u16::MAX {
                    let old_c = &prev.columns[pdef.value_col as usize];
                    pdef.value_col = plan.add_or_get_col_with_group(&old_c.name, &old_c.dict_group);
                }
            }
            cloned_jt.rebuild_runtime_index();
            plan.json_tree = Some(cloned_jt);
            plan.variants.push(PlanVariant::JsonTree);
        }
    } else if let Some(prev) = prior
        && let Some(prev_jt) = &prev.json_tree
    {
        let mut cloned_jt = prev_jt.clone();
        for pdef in &mut cloned_jt.paths {
            if pdef.value_col != u16::MAX {
                let old_c = &prev.columns[pdef.value_col as usize];
                pdef.value_col = plan.add_or_get_col_with_group(&old_c.name, &old_c.dict_group);
            }
        }
        cloned_jt.rebuild_runtime_index();
        plan.json_tree = Some(cloned_jt);
        plan.variants.push(PlanVariant::JsonTree);
    }

    // 3. Train Kubernetes CRI & Syslog Envelope Arms (with recursive inner ModelPlan!)
    if cri_inners.len() >= 4 {
        let c_ts = plan.add_or_get_col("$cri.ts");
        let c_stream = plan.add_or_get_col("$cri.stream");
        let c_flag = plan.add_or_get_col("$cri.flag");
        let c_inner = plan.add_or_get_col("$cri.inner");
        let mut child_opts = opts.clone();
        child_opts.depth = opts.depth + 1;
        let inner_plan = train_from_records(
            &cri_inners,
            ResolvedFraming::Lf { open_end: true },
            &child_opts,
            None,
        );
        if inner_plan.variants.len() > 1 {
            plan.columns[c_inner as usize].nested_plan = Some(Box::new(inner_plan));
        }
        plan.variants.push(PlanVariant::Envelope(EnvelopeSpec {
            kind: EnvelopeKind::KubernetesCri,
            cols: [c_ts, c_stream, c_flag, c_inner],
        }));
    }
    if syslog_inners.len() >= 4 {
        let c_pri = plan.add_or_get_col("$syslog.pri");
        let c_ts = plan.add_or_get_col("$syslog.ts");
        let c_ht = plan.add_or_get_col("$syslog.host_tag");
        let c_inner = plan.add_or_get_col("$syslog.inner");
        let mut child_opts = opts.clone();
        child_opts.depth = opts.depth + 1;
        let inner_plan = train_from_records(
            &syslog_inners,
            ResolvedFraming::Lf { open_end: true },
            &child_opts,
            None,
        );
        if inner_plan.variants.len() > 1 {
            plan.columns[c_inner as usize].nested_plan = Some(Box::new(inner_plan));
        }
        plan.variants.push(PlanVariant::Envelope(EnvelopeSpec {
            kind: EnvelopeKind::Syslog,
            cols: [c_pri, c_ts, c_ht, c_inner],
        }));
    }

    // 4. Train `logfmt` (`key=value`) Arm
    if logfmt_rows.len() >= 4 {
        train_logfmt_arm(&logfmt_rows, &mut plan);
    }

    // 5. Train CSV/TSV or Rich Plain-Text Templates (with Constant Pinning + Variable-Word Refinement!)
    if plain_rows.len() >= 4 {
        let trained_csv = train_csv_if_tabular(&plain_rows, &mut plan);
        if !trained_csv {
            train_templates_arm(&plain_rows, opts, &mut plan);
        }
    }

    plan.rebuild_indexes();
    discover_horizontal_candidates(samples, &mut plan);
    plan
}

// ============================================================================
// Arm Trainers
// ============================================================================

fn train_binary_stride(samples: &[&[u8]], stride: u32, plan: &mut ModelPlan) {
    let s = stride as usize;
    if s == 0 || s > 4096 {
        return;
    }
    // Fewer trial rows for wider records, so training cost does not grow with the stride.
    let whole = samples.iter().copied().filter(|r| r.len() == s);
    let rows: Vec<&[u8]> = whole.take(((1 << 18) / s).max(64)).collect();
    let mut slices = Vec::new();
    let mut off = 0usize;
    while off < s {
        let w = [8, 4, 2, 1].into_iter().find(|w| off + w <= s).unwrap();
        for (off, w) in binary_fields(&rows, off, w).1 {
            let cid = plan.add_or_get_col(&format!("$bin.{off}_{w}"));
            slices.push((off as u16, w as u16, cid));
        }
        off += w;
    }
    plan.variants
        .push(PlanVariant::BinaryStride(BinaryStrideSpec {
            stride,
            slices,
        }));
}

/// The fields in the `w` bytes at `off` and their encoded size over `rows`: one field, or the
/// fields of each half where those encode an eighth smaller. A narrower gain is not worth
/// the relations between rows that only the whole value shows.
fn binary_fields(rows: &[&[u8]], off: usize, w: usize) -> (usize, Vec<(usize, usize)>) {
    let col: Vec<&[u8]> = rows.iter().map(|r| &r[off..off + w]).collect();
    let whole = match col.is_empty() {
        true => 0,
        false => encode_present_slice(&col, EntropyBackend::Zstd, 3, 0)
            .3
            .len(),
    };
    if w > 2 {
        let (a, mut fields) = binary_fields(rows, off, w / 2);
        let (b, right) = binary_fields(rows, off + w / 2, w / 2);
        if (a + b) * 8 < whole * 7 {
            fields.extend(right);
            return (a + b, fields);
        }
    }
    (whole, vec![(off, w)])
}

fn train_logfmt_arm(rows: &[&[u8]], plan: &mut ModelPlan) {
    let mut key_counts: HashMap<Vec<u8>, usize> = HashMap::new();
    for &r in rows {
        if let Some(pairs) = parse_logfmt_line(r) {
            for p in pairs {
                *key_counts.entry(p.key.to_vec()).or_insert(0) += 1;
            }
        }
    }
    let mut keys: Vec<(Vec<u8>, usize)> = key_counts.into_iter().filter(|(_, c)| *c >= 2).collect();
    keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    keys.truncate(128);
    if keys.is_empty() {
        return;
    }
    let mut key_to_col = HashMap::with_capacity(keys.len());
    for (k, _) in keys {
        if let Ok(ks) = std::str::from_utf8(&k) {
            let group = if ks.contains("ip") || ks.contains("host") {
                "ip"
            } else if ks.contains("msg") || ks.contains("err") {
                "text"
            } else {
                ks
            };
            let cid = plan.add_or_get_col_with_group(&format!("logfmt.{ks}"), group);
            key_to_col.insert(k, cid);
        }
    }
    let recipe_col = plan.add_or_get_col("$logfmt.recipe");
    let overflow_col = plan.add_or_get_col_with_group("$logfmt.overflow", "text");
    plan.logfmt = Some(LogfmtSpec {
        key_to_col,
        recipe_col,
        overflow_col,
    });
    plan.variants.push(PlanVariant::Logfmt);
}

fn train_csv_if_tabular(rows: &[&[u8]], plan: &mut ModelPlan) -> bool {
    for delim in [b',', b'\t'] {
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        let probe = rows.len().min(128);
        for &r in rows.iter().take(probe) {
            let approx_n = r.iter().filter(|&&b| b == delim).count() + 1;
            if (3..=48).contains(&approx_n) && split_csv_exact(r, delim, approx_n).is_some() {
                *counts.entry(approx_n).or_insert(0) += 1;
            }
        }
        if let Some((&n_fields, &hits)) = counts.iter().max_by_key(|e| e.1)
            && hits * 4 >= probe * 3
        {
            let mut total_spaces = 0usize;
            for &r in rows.iter().take(probe) {
                total_spaces += r.iter().filter(|&&b| b == b' ').count();
            }
            if delim == b',' && total_spaces > probe * n_fields * 2 {
                continue;
            }
            let quote_mask_col = plan.add_or_get_col("$csv.qmask");
            let mut field_cols = Vec::with_capacity(n_fields);
            for fi in 0..n_fields {
                field_cols.push(plan.add_or_get_col(&format!("$csv.{fi}")));
            }
            plan.variants.push(PlanVariant::Csv(CsvSpec {
                delim,
                field_cols,
                quote_mask_col,
            }));
            return true;
        }
    }
    false
}

/// Pins slots that remain constant across all matched sample lines of a template directly into its literals!
fn pin_constant_slots(
    literals: Vec<Vec<u8>>,
    classes: Vec<SlotClass>,
    matched_lines: &[&[u8]],
) -> (Vec<Vec<u8>>, Vec<SlotClass>) {
    let k = classes.len();
    if k == 0 || matched_lines.len() < 6 {
        return (literals, classes);
    }
    let tv = TemplateVariant::new(literals.clone(), (0..k as u16).collect());

    let mut first_vals: Vec<Option<&[u8]>> = vec![None; k];
    let mut varies = vec![false; k];
    let mut parsed_rows = 0usize;
    let mut tmp = Vec::with_capacity(k);

    for &line in matched_lines {
        if !match_template(line, &tv, &mut tmp) || tmp.len() != k {
            continue;
        }
        parsed_rows += 1;
        for (si, &(_, val)) in tmp.iter().enumerate() {
            if varies[si] {
                continue;
            }
            match first_vals[si] {
                None => first_vals[si] = Some(val),
                Some(fv) if fv != val => varies[si] = true,
                _ => {}
            }
        }
    }
    if parsed_rows < 6 || varies.iter().all(|&v| v) {
        return (literals, classes);
    }

    let mut new_lits = Vec::with_capacity(k + 1);
    let mut new_classes = Vec::with_capacity(k);
    let mut cur_lit = literals[0].clone();
    for si in 0..k {
        if !varies[si]
            && let Some(cv) = first_vals[si]
        {
            cur_lit.extend_from_slice(cv);
            cur_lit.extend_from_slice(&literals[si + 1]);
        } else {
            new_lits.push(cur_lit);
            new_classes.push(classes[si]);
            cur_lit = literals[si + 1].clone();
        }
    }
    new_lits.push(cur_lit);
    (new_lits, new_classes)
}

fn sanitize_ctx_bytes(slice: &[u8]) -> String {
    let mut out = String::with_capacity(slice.len());
    for &b in slice {
        if b.is_ascii_alphanumeric() {
            out.push((b as char).to_ascii_lowercase());
        } else if matches!(b, b'_' | b':' | b'/' | b'=' | b'-') {
            out.push('_');
        }
    }
    out
}

fn train_templates_arm<'a>(rows: &[&'a [u8]], opts: &TrainOptions, plan: &mut ModelPlan) {
    let min_cluster = if rows.len() >= 64 { 4usize } else { 2usize };
    // Ordered maps here and below: the plan must not depend on hash iteration order.
    let mut clusters: BTreeMap<(Vec<Vec<u8>>, Vec<SlotClass>), Vec<&'a [u8]>> = BTreeMap::new();
    let mut unclustered = Vec::<&'a [u8]>::new();

    for &line in rows {
        if line.is_empty() || line.len() > 16384 {
            continue;
        }
        let raw_spans = tokenize_template_line(line);
        let mut spans = Vec::with_capacity(raw_spans.len());
        let mut prev_end = 0usize;
        for (s, e, cls) in raw_spans {
            if !spans.is_empty() && s == prev_end {
                let last: &mut (usize, usize, SlotClass) = spans.last_mut().unwrap();
                last.1 = e;
                last.2 = SlotClass::Word;
            } else {
                spans.push((s, e, cls));
            }
            prev_end = e;
        }
        if spans.is_empty() {
            unclustered.push(line);
            continue;
        }
        let mut lits = Vec::with_capacity(spans.len() + 1);
        let mut classes = Vec::with_capacity(spans.len());
        let mut p = 0usize;
        for &(s, e, cls) in &spans {
            lits.push(line[p..s].to_vec());
            classes.push(cls);
            p = e;
        }
        lits.push(line[p..].to_vec());
        clusters.entry((lits, classes)).or_default().push(line);
    }

    let mut accepted: Vec<((Vec<Vec<u8>>, Vec<SlotClass>), Vec<&'a [u8]>)> = Vec::new();
    let mut singleton_lines = unclustered;
    for (key, member_lines) in clusters {
        if member_lines.len() >= min_cluster {
            accepted.push((key, member_lines));
        } else {
            singleton_lines.extend(member_lines);
        }
    }

    if singleton_lines.len() >= min_cluster * 2 {
        for refined in refine_variable_word_templates(&singleton_lines, min_cluster) {
            accepted.push(refined);
        }
    }

    accepted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for ((raw_lits, raw_classes), member_lines) in accepted.into_iter().take(opts.max_templates) {
        let (literals, classes) = pin_constant_slots(raw_lits, raw_classes, &member_lines);

        // Context-keyed semantic slot assignment (shares matching header/entity slots across templates,
        // separates distinct integer domains like PID vs port vs size, and keeps per-template slots distinct
        // so same-row `src_ip == dest_ip` `ExactCopy` and `SharedDictPool` both fire!).
        let mut used_in_tpl: HashSet<u16> = HashSet::with_capacity(classes.len());
        let mut ctx_ord: HashMap<String, usize> = HashMap::new();
        let mut slots = Vec::with_capacity(classes.len());

        for (si, &cls) in classes.iter().enumerate() {
            let prev_lit = &literals[si];
            let next_lit = &literals[si + 1];
            let prev_tail = &prev_lit[prev_lit.len().saturating_sub(8)..];
            // Normalize log severity after PID/thread slot (`" INFO "` / `" WARN "` / `" ERROR "`) so thread IDs share one column!
            let next_head_raw = &next_lit[..next_lit.len().min(6)];
            let next_ctx = if next_lit.starts_with(b" INFO ")
                || next_lit.starts_with(b" WARN ")
                || next_lit.starts_with(b" ERROR ")
                || next_lit.starts_with(b" DEBUG ")
            {
                "lvl".to_string()
            } else {
                sanitize_ctx_bytes(next_head_raw)
            };

            let (base_key, group) = if cls == SlotClass::Int && prev_lit.ends_with(b"blk_") {
                ("$blk_id".to_string(), "blk_id")
            } else if cls == SlotClass::Timestamp && si == 0 {
                ("$ts.0".to_string(), "")
            } else if cls == SlotClass::IpPort {
                let pk = sanitize_ctx_bytes(prev_tail);
                (format!("$ip.{pk}"), "ip")
            } else {
                let pk = sanitize_ctx_bytes(prev_tail);
                (
                    format!("{}.{pk}_{next_ctx}", cls.canonical_col_prefix()),
                    "",
                )
            };

            let mut ord = *ctx_ord.get(&base_key).unwrap_or(&0);
            let cid = loop {
                let col_name = if ord == 0 {
                    base_key.clone()
                } else {
                    format!("{base_key}#{ord}")
                };
                let cand_cid = plan.add_or_get_col_with_group(&col_name, group);
                if used_in_tpl.insert(cand_cid) {
                    ctx_ord.insert(base_key.clone(), ord + 1);
                    break cand_cid;
                }
                ord += 1;
            };
            slots.push(cid);
        }

        let already = plan.variants.iter().any(|v| match v {
            PlanVariant::Template(tv) => tv.literals == literals,
            _ => false,
        });
        if !already {
            plan.variants
                .push(PlanVariant::Template(TemplateVariant::new(literals, slots)));
        }
    }
}

fn refine_variable_word_templates<'a>(
    lines: &[&'a [u8]],
    min_cluster: usize,
) -> Vec<((Vec<Vec<u8>>, Vec<SlotClass>), Vec<&'a [u8]>)> {
    let mut by_len: BTreeMap<usize, Vec<(&'a [u8], Vec<&'a [u8]>)>> = BTreeMap::new();
    for &line in lines {
        if line.contains(&b'\t') || line.len() > 1024 {
            continue;
        }
        let words: Vec<&[u8]> = line.split(|&b| b == b' ').collect();
        if (3..=24).contains(&words.len()) && words.iter().all(|w| !w.is_empty()) {
            by_len.entry(words.len()).or_default().push((line, words));
        }
    }
    let mut out = Vec::new();
    for (nw, group) in by_len {
        if group.len() < min_cluster {
            continue;
        }
        let mut subgroups: BTreeMap<Vec<u8>, Vec<&(&'a [u8], Vec<&'a [u8]>)>> = BTreeMap::new();
        for item in &group {
            let w = &item.1;
            let mut key = Vec::new();
            if !w[0][0].is_ascii_digit() {
                key.extend_from_slice(w[0]);
            }
            key.push(0);
            if !w[nw / 2][0].is_ascii_digit() {
                key.extend_from_slice(w[nw / 2]);
            }
            subgroups.entry(key).or_default().push(item);
        }
        for (_k, sg) in subgroups {
            if sg.len() < min_cluster {
                continue;
            }
            let mut is_var = vec![false; nw];
            let mut const_count = 0usize;
            for pos in 0..nw {
                let first_w = sg[0].1[pos];
                if sg.iter().all(|r| r.1[pos] == first_w) {
                    const_count += 1;
                } else {
                    is_var[pos] = true;
                }
            }
            if const_count * 2 < nw || !is_var.iter().any(|&v| v) {
                continue;
            }
            let mut literals = Vec::<Vec<u8>>::new();
            let mut classes = Vec::<SlotClass>::new();
            let mut cur_lit = Vec::<u8>::new();
            let mut i = 0usize;
            while i < nw {
                if !is_var[i] {
                    if i > 0 {
                        cur_lit.push(b' ');
                    }
                    cur_lit.extend_from_slice(sg[0].1[i]);
                    i += 1;
                } else {
                    if i > 0 {
                        cur_lit.push(b' ');
                    }
                    literals.push(std::mem::take(&mut cur_lit));
                    let start_var = i;
                    while i < nw && is_var[i] {
                        i += 1;
                    }
                    let cls = if i == start_var + 1 && parse_exact_int(sg[0].1[start_var]).is_some()
                    {
                        SlotClass::Int
                    } else {
                        SlotClass::Word
                    };
                    classes.push(cls);
                }
            }
            literals.push(cur_lit);
            if literals[1..literals.len() - 1]
                .iter()
                .all(|l| !l.is_empty())
            {
                let raw_lines: Vec<&'a [u8]> = sg.iter().map(|x| x.0).collect();
                out.push(((literals, classes), raw_lines));
            }
        }
    }
    out
}

#[inline]
fn unquote_lexeme(b: &[u8]) -> &[u8] {
    if b.len() >= 2 && b[0] == b'"' && b[b.len() - 1] == b'"' {
        &b[1..b.len() - 1]
    } else {
        b
    }
}

fn discover_horizontal_candidates(samples: &[&[u8]], plan: &mut ModelPlan) {
    let probe_vec = sample_strided(samples, 512, 4 << 20);
    let probe = probe_vec.as_slice();
    if probe.len() < 8 || plan.columns.len() <= 2 {
        return;
    }
    let split = split_records_with_plan(probe, plan);
    let n_cols = plan.columns.len();

    let skip_col: Vec<bool> = plan
        .columns
        .iter()
        .map(|c| c.name.starts_with("$logfmt.") || c.nested_plan.is_some())
        .collect();

    let views: Vec<Vec<&[u8]>> = (0..n_cols)
        .map(|c| {
            split.col_occs[c]
                .iter()
                .map(|o| o.as_slice(&split.scratch))
                .collect()
        })
        .collect();

    let is_int_col: Vec<bool> = (0..n_cols)
        .map(|c| {
            views[c].len() >= 4
                && views[c]
                    .iter()
                    .take(32)
                    .all(|b| parse_exact_int(unquote_lexeme(b)).is_some())
        })
        .collect();

    for target in 2..n_cols {
        let t_vals = &views[target];
        if skip_col[target] || t_vals.len() < 4 {
            continue;
        }
        let check_n = t_vals.len().min(128);
        let mut cands = Vec::<(u16, usize)>::new();
        for src in 1..target {
            let s_vals = &views[src];
            if skip_col[src] || s_vals.len() != t_vals.len() {
                continue;
            }
            let mut score = 0usize;
            for r in 0..check_n {
                let tb = t_vals[r];
                let sb = s_vals[r];
                if tb == sb {
                    score += 3;
                } else {
                    let sb_u = unquote_lexeme(sb);
                    if sb_u.len() >= 3
                        && tb.len() > sb_u.len()
                        && memchr::memmem::find(tb, sb_u).is_some()
                    {
                        score += 2;
                    } else if is_int_col[target] && is_int_col[src] {
                        score += 1;
                    }
                }
            }
            if score >= 2 && score * 2 >= check_n {
                cands.push((src as u16, score));
            }
        }
        cands.sort_by(|a, b| b.1.cmp(&a.1));
        cands.truncate(6);
        plan.columns[target].horiz_candidates = cands.into_iter().map(|x| x.0).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_fields_split_where_halves_encode_smaller() {
        // A slowly rising double, then a tag, a counter and a float from a small set.
        let rows: Vec<Vec<u8>> = (0..2000u32)
            .map(|i| {
                let mix = i.wrapping_mul(2_654_435_761);
                let mut r = (1.0 + f64::from(i) * 1e-5 + f64::from(mix) * 1e-16)
                    .to_le_bytes()
                    .to_vec();
                r.extend([b'G', b'0' + (i % 8) as u8]);
                r.extend((400 + (i % 650) as u16).to_le_bytes());
                r.extend((f32::from((mix >> 24) as u8) * 1e-8).to_le_bytes());
                r
            })
            .collect();
        let rows: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        assert_eq!(binary_fields(&rows, 0, 8).1, [(0, 8)]);
        assert_eq!(binary_fields(&rows, 8, 8).1, [(8, 4), (12, 4)]);

        let mut plan = ModelPlan::default();
        train_binary_stride(&rows, 16, &mut plan);
        let Some(PlanVariant::BinaryStride(spec)) = plan.variants.last() else {
            panic!("no stride variant");
        };
        let fields = spec.slices.iter().map(|&(off, w, _)| (off, w));
        assert_eq!(fields.collect::<Vec<_>>(), [(0, 8), (8, 4), (12, 4)]);
    }
}
