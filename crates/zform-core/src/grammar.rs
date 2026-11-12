//! Unified Multi-Arm Structural Grammar (`ModelPlan`):
//! Simultaneously routes and losslessly decomposes:
//! - Hierarchical JSON trees (`JsonTreePlan`: recursive objects, arrays, inline constants, cross-path derivations, nested JSON-in-string)
//! - Kubernetes CRI & Syslog envelopes (with recursive inner `ModelPlan`)
//! - First-class `logfmt` (`key=value` with dynamic per-row layout recipes)
//! - Rich plain-text templates (with pinned constants, context-keyed slots, and variable-word refinement)
//! - Dialect-exact CSV/TSV
//! - Fixed-stride binary structs (`sao`-class)
//! - Verbatim residuals (`Variant 0`), all into **dense occurrence streams** (zero `n_rows × n_cols` sparse matrix!).

use crate::codecs::BlockDecodeCtx;
use crate::container::block_col;
use crate::json_tree::{JsonTreeBlockReconstructor, JsonTreeBlockSplitter, JsonTreePlan};
use crate::lex::{
    parse_exact_decimal, parse_exact_int, parse_hex_exact, parse_ipv4_or_port, parse_timestamp,
    parse_traceparent, parse_ulid, parse_uuid, split_email,
};
use crate::payload::library::Library;
use crate::wire::{Cursor, DecodeBudget, Error, Result, put_bytes, put_uvarint};
use memchr::memmem::Finder;
use std::collections::HashMap;
use std::sync::Arc;

/// Column metadata in a `ModelPlan`.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    /// Canonical column/field identifier (e.g. `"$.actor.login"`, `"logfmt.level"`, `"$blk_id"`, `"$csv.0"`).
    pub name: String,
    /// Optional semantic dictionary/pool group (`"login"`, `"ip"`, `"text"`, etc.).
    pub dict_group: String,
    /// Pre-selected preceding column indices to test for horizontal derivations (`ExactCopy`, `SubstringInterp`, `AffineInt`).
    pub horiz_candidates: Vec<u16>,
    /// Optional recursive child `ModelPlan` for JSON/envelope string columns wrapping inner structured text!
    pub nested_plan: Option<Box<ModelPlan>>,
}

/// Plain-text template (`literals.len() == slots.len() + 1`).
#[derive(Debug, Clone)]
pub struct TemplateVariant {
    pub literals: Vec<Vec<u8>>,
    pub slots: Vec<u16>,
    /// Index of the longest interior literal, searched first to reject a line early; 0 if none.
    anchor_idx: usize,
    min_len: usize,
    /// One searcher per literal, built once: building one costs more than searching a line.
    finders: Vec<Finder<'static>>,
}

impl TemplateVariant {
    pub fn new(literals: Vec<Vec<u8>>, slots: Vec<u16>) -> Self {
        let interior = 1..literals.len().saturating_sub(1);
        let anchor_idx = interior
            .rev()
            .max_by_key(|&i| literals[i].len())
            .unwrap_or(0);
        Self {
            anchor_idx,
            min_len: literals.iter().map(Vec::len).sum(),
            finders: literals
                .iter()
                .map(|l| Finder::new(l).into_owned())
                .collect(),
            literals,
            slots,
        }
    }
}

/// Envelope prefix wrapper (e.g. Kubernetes CRI `<ts> <stream> <tag> <inner>` or Syslog `<ts> <host> <app>: <inner>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EnvelopeKind {
    /// `<iso_ts> <stdout|stderr> <F|P> <inner>`
    KubernetesCri = 0,
    /// `<Optional <PRI>><syslog_or_iso_ts> <host> <app_tag>: <inner>`
    Syslog = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct EnvelopeSpec {
    pub kind: EnvelopeKind,
    /// Prefix columns (`[ts_col, col1, col2]`) + `inner_col` (which carries a `nested_plan`!).
    pub cols: [u16; 4],
}

#[derive(Debug, Clone)]
pub struct LogfmtSpec {
    /// Known key names -> `col_id`.
    pub key_to_col: HashMap<Vec<u8>, u16>,
    /// Stores exact per-row layout recipe (`sep + quote_flag + known/key` sequence) so key order & spacing are 100% lossless!
    pub recipe_col: u16,
    /// Stores values of rare/unknown keys concatenated with `\x1f` separator (or falls back to Residual if `\x1f` present).
    pub overflow_col: u16,
}

#[derive(Debug, Clone)]
pub struct CsvSpec {
    pub delim: u8,
    pub field_cols: Vec<u16>,
    /// Per-row bitmask of quoted fields (`"..."` without interior `""` escapes; rows with `""` fall back to Residual).
    pub quote_mask_col: u16,
}

#[derive(Debug, Clone)]
pub struct BinaryStrideSpec {
    pub stride: u32,
    /// `(offset, len, col_id)` partitioning `0..stride`.
    pub slices: Vec<(u16, u16, u16)>,
}

/// Variant tag dispatch entry (`0` is always `Residual` into column `0`).
#[derive(Debug, Clone)]
pub enum PlanVariant {
    Residual { col: u16 },
    JsonTree,
    Envelope(EnvelopeSpec),
    Logfmt,
    Template(TemplateVariant),
    Csv(CsvSpec),
    BinaryStride(BinaryStrideSpec),
}

/// Complete, self-contained structural model embedded in `ZFM1` frames and `ZFP1` profile files.
#[derive(Debug, Clone)]
pub struct ModelPlan {
    pub columns: Vec<ColumnDef>,
    pub variants: Vec<PlanVariant>,
    pub json_tree: Option<JsonTreePlan>,
    pub logfmt: Option<LogfmtSpec>,
    /// Cached variant indices for fast template matching.
    pub template_indices: Vec<u32>,
    /// The programs and tables blocks under this model name instead of carrying.
    pub library: Arc<Library>,
}

impl Default for ModelPlan {
    fn default() -> Self {
        Self::residual_only()
    }
}

impl ModelPlan {
    pub fn residual_only() -> Self {
        Self {
            columns: vec![ColumnDef {
                name: "$residual".into(),
                dict_group: "text".into(),
                horiz_candidates: Vec::new(),
                nested_plan: None,
            }],
            variants: vec![PlanVariant::Residual { col: 0 }],
            json_tree: None,
            logfmt: None,
            template_indices: Vec::new(),
            library: Arc::default(),
        }
    }

    pub fn rebuild_indexes(&mut self) {
        self.template_indices.clear();
        for (vi, v) in self.variants.iter().enumerate() {
            if matches!(v, PlanVariant::Template(_)) {
                self.template_indices.push(vi as u32);
            }
        }
        if let Some(jt) = &mut self.json_tree {
            jt.rebuild_runtime_index();
        }
    }

    pub fn add_or_get_col(&mut self, name: &str) -> u16 {
        self.add_or_get_col_with_group(name, "")
    }

    pub fn add_or_get_col_with_group(&mut self, name: &str, dict_group: &str) -> u16 {
        if let Some(idx) = self.columns.iter().position(|c| c.name == name) {
            if self.columns[idx].dict_group.is_empty() && !dict_group.is_empty() {
                self.columns[idx].dict_group = dict_group.to_string();
            }
            return idx as u16;
        }
        let idx = self.columns.len() as u16;
        self.columns.push(ColumnDef {
            name: name.to_string(),
            dict_group: dict_group.to_string(),
            horiz_candidates: Vec::new(),
            nested_plan: None,
        });
        idx
    }

    #[inline]
    pub fn json_tree_variant_id(&self) -> Option<u32> {
        self.variants
            .iter()
            .position(|v| matches!(v, PlanVariant::JsonTree))
            .map(|i| i as u32)
    }

    /// Serializes `ModelPlan` into compact binary metadata.
    pub fn encode(&self, out: &mut Vec<u8>) {
        put_uvarint(out, self.columns.len() as u64);
        for c in &self.columns {
            put_bytes(out, c.name.as_bytes());
            put_bytes(out, c.dict_group.as_bytes());
            put_uvarint(out, c.horiz_candidates.len() as u64);
            for &hc in &c.horiz_candidates {
                put_uvarint(out, hc as u64);
            }
            if let Some(np) = &c.nested_plan {
                out.push(1);
                let mut child = Vec::new();
                np.encode(&mut child);
                put_bytes(out, &child);
            } else {
                out.push(0);
            }
        }
        if let Some(jt) = &self.json_tree {
            out.push(1);
            jt.encode(out);
        } else {
            out.push(0);
        }
        if let Some(lf) = &self.logfmt {
            out.push(1);
            put_uvarint(out, lf.recipe_col as u64);
            put_uvarint(out, lf.overflow_col as u64);
            let mut pairs: Vec<(&Vec<u8>, &u16)> = lf.key_to_col.iter().collect();
            pairs.sort_by_key(|p| *p.1);
            put_uvarint(out, pairs.len() as u64);
            for (k, &cid) in pairs {
                put_bytes(out, k);
                put_uvarint(out, cid as u64);
            }
        } else {
            out.push(0);
        }
        put_uvarint(out, self.variants.len() as u64);
        for v in &self.variants {
            match v {
                PlanVariant::Residual { col } => {
                    out.push(0);
                    put_uvarint(out, *col as u64);
                }
                PlanVariant::JsonTree => out.push(1),
                PlanVariant::Envelope(env) => {
                    out.push(3);
                    out.push(env.kind as u8);
                    for c in env.cols {
                        put_uvarint(out, c as u64);
                    }
                }
                PlanVariant::Logfmt => out.push(4),
                PlanVariant::Template(tv) => {
                    out.push(5);
                    put_uvarint(out, tv.slots.len() as u64);
                    for l in &tv.literals {
                        put_bytes(out, l);
                    }
                    for &cid in &tv.slots {
                        put_uvarint(out, cid as u64);
                    }
                }
                PlanVariant::Csv(cs) => {
                    out.push(6);
                    out.push(cs.delim);
                    put_uvarint(out, cs.quote_mask_col as u64);
                    put_uvarint(out, cs.field_cols.len() as u64);
                    for &cid in &cs.field_cols {
                        put_uvarint(out, cid as u64);
                    }
                }
                PlanVariant::BinaryStride(bs) => {
                    out.push(7);
                    put_uvarint(out, bs.stride as u64);
                    put_uvarint(out, bs.slices.len() as u64);
                    for &(off, len, cid) in &bs.slices {
                        put_uvarint(out, off as u64);
                        put_uvarint(out, len as u64);
                        put_uvarint(out, cid as u64);
                    }
                }
            }
        }
        self.library.encode(out);
    }

    pub fn decode(cur: &mut Cursor<'_>, depth: u8) -> Result<Self> {
        if depth > 3 {
            return Err(Error::Format("ModelPlan nesting depth > 3".into()));
        }
        let n_cols = cur.uvar()? as usize;
        if n_cols == 0 || n_cols > 16384 {
            return Err(Error::Format(format!(
                "invalid ModelPlan col count {n_cols}"
            )));
        }
        let mut columns = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            let name = cur.str_utf8()?.to_string();
            let dict_group = cur.str_utf8()?.to_string();
            let nh = cur.uvar()? as usize;
            let mut horiz_candidates = Vec::with_capacity(nh.min(32));
            for _ in 0..nh {
                horiz_candidates.push(cur.uvar()? as u16);
            }
            let nested_plan = if cur.u8()? != 0 {
                let b = cur.bytes()?;
                Some(Box::new(Self::decode(&mut Cursor::new(b), depth + 1)?))
            } else {
                None
            };
            columns.push(ColumnDef {
                name,
                dict_group,
                horiz_candidates,
                nested_plan,
            });
        }
        let json_tree = if cur.u8()? != 0 {
            Some(JsonTreePlan::decode(cur)?)
        } else {
            None
        };
        let logfmt = if cur.u8()? != 0 {
            let recipe_col = cur.uvar()? as u16;
            let overflow_col = cur.uvar()? as u16;
            let nk = cur.uvar()? as usize;
            let mut key_to_col = HashMap::with_capacity(nk.min(4096));
            for _ in 0..nk {
                let k = cur.bytes()?.to_vec();
                let cid = cur.uvar()? as u16;
                key_to_col.insert(k, cid);
            }
            Some(LogfmtSpec {
                key_to_col,
                recipe_col,
                overflow_col,
            })
        } else {
            None
        };
        let n_vars = cur.uvar()? as usize;
        if n_vars == 0 || n_vars > 8192 {
            return Err(Error::Format("invalid variant count".into()));
        }
        let mut variants = Vec::with_capacity(n_vars);
        for _ in 0..n_vars {
            match cur.u8()? {
                0 => variants.push(PlanVariant::Residual {
                    col: cur.uvar()? as u16,
                }),
                1 => variants.push(PlanVariant::JsonTree),
                3 => {
                    let kind = match cur.u8()? {
                        0 => EnvelopeKind::KubernetesCri,
                        1 => EnvelopeKind::Syslog,
                        x => return Err(Error::Format(format!("bad envelope kind {x}"))),
                    };
                    let cols = [
                        cur.uvar()? as u16,
                        cur.uvar()? as u16,
                        cur.uvar()? as u16,
                        cur.uvar()? as u16,
                    ];
                    variants.push(PlanVariant::Envelope(EnvelopeSpec { kind, cols }));
                }
                4 => variants.push(PlanVariant::Logfmt),
                5 => {
                    let k = cur.uvar()? as usize;
                    let mut literals = Vec::with_capacity(k + 1);
                    for _ in 0..=k {
                        literals.push(cur.bytes()?.to_vec());
                    }
                    let mut slots = Vec::with_capacity(k);
                    for _ in 0..k {
                        slots.push(cur.uvar()? as u16);
                    }
                    variants.push(PlanVariant::Template(TemplateVariant::new(literals, slots)));
                }
                6 => {
                    let delim = cur.u8()?;
                    let quote_mask_col = cur.uvar()? as u16;
                    let nf = cur.uvar()? as usize;
                    let mut field_cols = Vec::with_capacity(nf);
                    for _ in 0..nf {
                        field_cols.push(cur.uvar()? as u16);
                    }
                    variants.push(PlanVariant::Csv(CsvSpec {
                        delim,
                        field_cols,
                        quote_mask_col,
                    }));
                }
                7 => {
                    let stride = cur.uvar()? as u32;
                    let ns = cur.uvar()? as usize;
                    let mut slices = Vec::with_capacity(ns);
                    for _ in 0..ns {
                        slices.push((cur.uvar()? as u16, cur.uvar()? as u16, cur.uvar()? as u16));
                    }
                    variants.push(PlanVariant::BinaryStride(BinaryStrideSpec {
                        stride,
                        slices,
                    }));
                }
                tag => return Err(Error::Format(format!("unknown PlanVariant tag {tag}"))),
            }
        }
        let mut plan = Self {
            columns,
            variants,
            json_tree,
            logfmt,
            template_indices: Vec::new(),
            library: Arc::new(Library::decode(cur)?),
        };
        plan.rebuild_indexes();
        Ok(plan)
    }
}

// ============================================================================
// 1. Envelope Prefix Parsers (Kubernetes CRI & Syslog)
// ============================================================================

/// Matches Kubernetes CRI log lines: `<iso_ts> <stdout|stderr> <F|P> <inner>`.
pub fn parse_k8s_cri(line: &[u8]) -> Option<(&[u8], &[u8], &[u8], &[u8])> {
    let sp1 = line.iter().position(|&b| b == b' ')?;
    if !(20..=35).contains(&sp1) {
        return None;
    }
    let ts = &line[..sp1];
    parse_timestamp(ts)?;
    let rest1 = &line[sp1 + 1..];
    let (stream, rest2) = if let Some(r) = rest1.strip_prefix(b"stdout ") {
        (&b"stdout"[..], r)
    } else if let Some(r) = rest1.strip_prefix(b"stderr ") {
        (&b"stderr"[..], r)
    } else {
        return None;
    };
    if rest2.len() < 2 || rest2[1] != b' ' || !matches!(rest2[0], b'F' | b'P') {
        return None;
    }
    Some((ts, stream, &rest2[0..1], &rest2[2..]))
}

/// Matches Syslog lines: `[<PRI>]<Mmm [ D]D HH:MM:SS> <host> <tag>: <inner>`.
pub fn parse_syslog_envelope(line: &[u8]) -> Option<(&[u8], &[u8], &[u8], &[u8])> {
    let (pri, rem) = if line.starts_with(b"<") {
        let gt = line[..line.len().min(6)].iter().position(|&b| b == b'>')?;
        let p = &line[..=gt];
        if p.len() < 3 || !p[1..gt].iter().all(|b| b.is_ascii_digit()) {
            return None;
        }
        (p, &line[gt + 1..])
    } else {
        (&b""[..], line)
    };
    if rem.len() < 22 || rem.get(15) != Some(&b' ') {
        return None;
    }
    let ts = &rem[..15];
    parse_timestamp(ts)?;
    let after_ts = &rem[16..];
    let sp_host = after_ts.iter().position(|&b| b == b' ')?;
    if sp_host == 0 || sp_host > 64 {
        return None;
    }
    let after_host = &after_ts[sp_host + 1..];
    let colon = memchr::memmem::find(after_host, b": ")?;
    if colon == 0 || colon > 64 || after_host[..colon].contains(&b' ') {
        return None;
    }
    let host_and_tag = &after_ts[..sp_host + 1 + colon];
    let inner = &after_host[colon + 2..];
    Some((pri, ts, host_and_tag, inner))
}

// ============================================================================
// 2. First-Class `logfmt` (`key=value`) Lossless Parser
// ============================================================================

#[derive(Debug, Clone)]
pub struct LogfmtPair<'a> {
    pub sep_before: &'a [u8],
    pub key: &'a [u8],
    pub quoted: bool,
    pub val_inner: &'a [u8],
}

fn scan_quoted_span(b: &[u8], pos: &mut usize) -> Option<(usize, usize)> {
    if *pos >= b.len() || b[*pos] != b'"' {
        return None;
    }
    *pos += 1;
    let start = *pos;
    while *pos < b.len() {
        let c = b[*pos];
        if c == b'"' {
            let end = *pos;
            *pos += 1;
            return Some((start, end));
        }
        if c == b'\\' {
            *pos += 2;
            if *pos > b.len() {
                return None;
            }
        } else if c < 0x20 {
            return None;
        } else {
            *pos += 1;
        }
    }
    None
}

/// Parses a `logfmt` line (`ts=... level=info msg="hello world" dur=12ms`) if it has `>= 2` valid pairs
/// and covers the entire line byte-for-byte with restorable quoting!
pub fn parse_logfmt_line(line: &[u8]) -> Option<Vec<LogfmtPair<'_>>> {
    if line.len() < 7 || line.starts_with(b"{") || line.starts_with(b"[") {
        return None;
    }
    let mut pos = 0usize;
    let mut pairs = Vec::with_capacity(12);
    while pos < line.len() {
        let ws_start = pos;
        while pos < line.len() && (line[pos] == b' ' || line[pos] == b'\t') {
            pos += 1;
        }
        if pairs.is_empty() && pos > ws_start {
            return None;
        }
        if !pairs.is_empty() && pos == ws_start {
            return None;
        }
        if pos >= line.len() {
            return None;
        }
        let sep_before = &line[ws_start..pos];
        let k_start = pos;
        while pos < line.len()
            && (line[pos].is_ascii_alphanumeric() || matches!(line[pos], b'_' | b'.' | b'-' | b'/'))
        {
            pos += 1;
        }
        let key = &line[k_start..pos];
        if key.is_empty() || key.len() > 64 || pos >= line.len() || line[pos] != b'=' {
            return None;
        }
        pos += 1;
        if pos < line.len() && line[pos] == b'"' {
            let (is, ie) = scan_quoted_span(line, &mut pos)?;
            pairs.push(LogfmtPair {
                sep_before,
                key,
                quoted: true,
                val_inner: &line[is..ie],
            });
        } else {
            let v_start = pos;
            while pos < line.len() && line[pos] != b' ' && line[pos] != b'\t' {
                if line[pos] == b'"' {
                    return None;
                }
                pos += 1;
            }
            pairs.push(LogfmtPair {
                sep_before,
                key,
                quoted: false,
                val_inner: &line[v_start..pos],
            });
        }
    }
    if pairs.len() >= 2 { Some(pairs) } else { None }
}

// ============================================================================
// 3. Rich Observability Template Tokenizer & Matcher
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SlotClass {
    Timestamp,
    Traceparent,
    Uuid,
    Ulid,
    IpPort,
    Email,
    Path,
    Quoted,
    Hex,
    Decimal,
    Int,
    Word,
}

impl SlotClass {
    pub fn canonical_col_prefix(self) -> &'static str {
        match self {
            Self::Timestamp => "$ts",
            Self::Traceparent => "$traceparent",
            Self::Uuid => "$uuid",
            Self::Ulid => "$ulid",
            Self::IpPort => "$ip",
            Self::Email => "$email",
            Self::Path => "$url",
            Self::Quoted => "$qstr",
            Self::Hex => "$hex",
            Self::Decimal => "$dec",
            Self::Int => "$int",
            Self::Word => "$word",
        }
    }
}

/// Scans a plain-text line and identifies typed variable spans `(start, end, SlotClass)`.
pub fn tokenize_template_line(line: &[u8]) -> Vec<(usize, usize, SlotClass)> {
    let mut spans = Vec::new();
    let len = line.len();
    let mut i = 0usize;
    while i < len {
        // 1. ISO-8601 / CLF / Syslog timestamp at boundary
        if (i == 0 || !line[i - 1].is_ascii_alphanumeric())
            && let Some((end, cls)) = try_scan_timestamp_at(line, i)
        {
            spans.push((i, end, cls));
            i = end;
            continue;
        }
        // 2. Quoted string `"..."` (excluding empty `""` or >256B blobs)
        if line[i] == b'"'
            && let Some(close) = scan_simple_quoted(line, i)
            && close > i + 2
            && close - i <= 258
        {
            spans.push((i + 1, close - 1, SlotClass::Quoted));
            i = close;
            continue;
        }
        // 3. Token-boundary structured atoms (traceparent, UUID, ULID, standalone IPv4, Email, http/https URL)
        if is_token_start(line, i) {
            let tok_end = scan_token_end(line, i);
            let tok = &line[i..tok_end];
            if tok.len() == 55 && parse_traceparent(tok).is_some() {
                spans.push((i, tok_end, SlotClass::Traceparent));
                i = tok_end;
                continue;
            }
            if tok.len() == 36 && parse_uuid(tok).is_some() {
                spans.push((i, tok_end, SlotClass::Uuid));
                i = tok_end;
                continue;
            }
            if tok.len() == 26 && parse_ulid(tok).is_some() {
                spans.push((i, tok_end, SlotClass::Ulid));
                i = tok_end;
                continue;
            }
            // Standalone IPv4 (stops before `:port` so host IPs hit MTF-4096 and constant ports like `:50010` pin to literal!)
            let ip_end = line[i..tok_end]
                .iter()
                .position(|&c| c == b':')
                .map(|p| i + p)
                .unwrap_or(tok_end);
            if ip_end > i && parse_ipv4_or_port(&line[i..ip_end]).is_some_and(|x| x.1 == 0) {
                spans.push((i, ip_end, SlotClass::IpPort));
                i = ip_end;
                continue;
            }
            if tok.starts_with(b"/")
                && ip_end > i + 1
                && parse_ipv4_or_port(&line[i + 1..ip_end]).is_some_and(|x| x.1 == 0)
            {
                spans.push((i + 1, ip_end, SlotClass::IpPort));
                i = ip_end;
                continue;
            }
            if split_email(tok).is_some() {
                spans.push((i, tok_end, SlotClass::Email));
                i = tok_end;
                continue;
            }
            if (tok.starts_with(b"http://") || tok.starts_with(b"https://")) && tok.len() <= 256 {
                spans.push((i, tok_end, SlotClass::Path));
                i = tok_end;
                continue;
            }
        }
        // 4. Alphanumeric/numeric sub-tokens (keeps `blk_` and `/mnt/hadoop/.../job_` literal, masking only varying IDs!)
        let b = line[i];
        if (b == b'-' || b == b'+')
            && i + 1 < len
            && line[i + 1].is_ascii_digit()
            && (i == 0 || !line[i - 1].is_ascii_digit())
        {
            let mut j = i + 1;
            while j < len && line[j].is_ascii_digit() {
                j += 1;
            }
            if j < len && line[j] == b'.' && j + 1 < len && line[j + 1].is_ascii_digit() {
                j += 1;
                while j < len && line[j].is_ascii_digit() {
                    j += 1;
                }
                if parse_exact_decimal(&line[i..j]).is_some() {
                    spans.push((i, j, SlotClass::Decimal));
                    i = j;
                    continue;
                }
            }
            if parse_exact_int(&line[i..j]).is_some() {
                spans.push((i, j, SlotClass::Int));
                i = j;
                continue;
            }
        }
        // Check standalone IPv4 after non-whitespace delimiters like `/` or `:` (e.g. `src:/10.250.19.102:54106`)
        if b.is_ascii_digit() && (i == 0 || !line[i - 1].is_ascii_alphanumeric()) {
            let mut k = i;
            let mut dots = 0usize;
            while k < len && (line[k].is_ascii_digit() || line[k] == b'.') {
                if line[k] == b'.' {
                    dots += 1;
                }
                k += 1;
            }
            // Trim trailing dot (e.g. sentence-ending `10.250.14.224.`)
            if k > i && line[k - 1] == b'.' {
                k -= 1;
                dots = dots.saturating_sub(1);
            }
            if dots == 3 && parse_ipv4_or_port(&line[i..k]).is_some_and(|x| x.1 == 0) {
                spans.push((i, k, SlotClass::IpPort));
                i = k;
                continue;
            }
        }
        if b.is_ascii_hexdigit() && (i == 0 || !line[i - 1].is_ascii_alphanumeric()) {
            let mut j = i;
            while j < len && line[j].is_ascii_hexdigit() {
                j += 1;
            }
            if j == len || !line[j].is_ascii_alphanumeric() {
                let slice = &line[i..j];
                let all_digits = slice.iter().all(|c| c.is_ascii_digit());
                if all_digits {
                    if j < len && line[j] == b'.' && j + 1 < len && line[j + 1].is_ascii_digit() {
                        let mut k = j + 1;
                        while k < len && line[k].is_ascii_digit() {
                            k += 1;
                        }
                        if (k == len || line[k] != b'.')
                            && parse_exact_decimal(&line[i..k]).is_some()
                        {
                            spans.push((i, k, SlotClass::Decimal));
                            i = k;
                            continue;
                        }
                    }
                    if parse_exact_int(slice).is_some() {
                        spans.push((i, j, SlotClass::Int));
                        i = j;
                        continue;
                    }
                } else if slice.len() >= 8
                    && slice.len().is_multiple_of(2)
                    && parse_hex_exact(slice).is_some()
                {
                    spans.push((i, j, SlotClass::Hex));
                    i = j;
                    continue;
                }
            }
        }
        if b.is_ascii_digit() {
            let mut j = i;
            while j < len && line[j].is_ascii_digit() {
                j += 1;
            }
            if parse_exact_int(&line[i..j]).is_some() {
                spans.push((i, j, SlotClass::Int));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    spans
}

fn try_scan_timestamp_at(line: &[u8], i: usize) -> Option<(usize, SlotClass)> {
    let rem = &line[i..];
    if rem.len() >= 19 && rem[4] == b'-' && rem[7] == b'-' && rem[13] == b':' && rem[16] == b':' {
        let mut e = 19usize;
        if e < rem.len() && (rem[e] == b'.' || rem[e] == b',') {
            e += 1;
            while e < rem.len() && rem[e].is_ascii_digit() {
                e += 1;
            }
        }
        if e < rem.len() && (rem[e] == b'Z' || rem[e] == b'z') {
            e += 1;
        } else if e + 6 <= rem.len() && matches!(rem[e], b'+' | b'-') && rem[e + 3] == b':' {
            e += 6;
        } else if e + 5 <= rem.len() && matches!(rem[e], b'+' | b'-') && rem[e + 1].is_ascii_digit()
        {
            e += 5;
        }
        if parse_timestamp(&rem[..e]).is_some() {
            return Some((i + e, SlotClass::Timestamp));
        }
    }
    if rem.len() >= 26
        && rem[2] == b'/'
        && rem[6] == b'/'
        && rem[11] == b':'
        && parse_timestamp(&rem[..26]).is_some()
    {
        return Some((i + 26, SlotClass::Timestamp));
    }
    if rem.len() >= 15
        && rem[3] == b' '
        && rem[6] == b' '
        && rem[9] == b':'
        && parse_timestamp(&rem[..15]).is_some()
    {
        return Some((i + 15, SlotClass::Timestamp));
    }
    None
}

fn scan_simple_quoted(line: &[u8], start: usize) -> Option<usize> {
    let mut j = start + 1;
    while j < line.len() {
        if line[j] == b'\\' {
            j += 2;
        } else if line[j] == b'"' {
            return Some(j + 1);
        } else {
            j += 1;
        }
    }
    None
}

#[inline]
fn is_token_start(line: &[u8], i: usize) -> bool {
    i == 0 || matches!(line[i - 1], b' ' | b'\t' | b'[' | b'(' | b'=' | b',' | b'<')
}

#[inline]
fn scan_token_end(line: &[u8], start: usize) -> usize {
    let mut j = start;
    while j < line.len()
        && !matches!(
            line[j],
            b' ' | b'\t' | b']' | b')' | b',' | b';' | b'>' | b'"' | b'\''
        )
    {
        j += 1;
    }
    j
}

/// Matches `line` against a `TemplateVariant`, extracting `slots.len()` slices into `out_vals`.
pub fn match_template<'a>(
    line: &'a [u8],
    tv: &TemplateVariant,
    out_vals: &mut Vec<(u16, &'a [u8])>,
) -> bool {
    if line.len() < tv.min_len {
        return false;
    }
    let k = tv.slots.len();
    let prefix = &tv.literals[0];
    let suffix = &tv.literals[k];
    if !line.starts_with(prefix) || !line.ends_with(suffix) {
        return false;
    }
    if line.len() < prefix.len() + suffix.len() {
        return false;
    }
    if k == 0 {
        if line.len() == prefix.len() {
            out_vals.clear();
            return true;
        }
        return false;
    }
    if tv.anchor_idx > 0 && tv.anchor_idx < k {
        let body = &line[prefix.len()..line.len() - suffix.len()];
        if tv.finders[tv.anchor_idx].find(body).is_none() {
            return false;
        }
    }
    let body_end = line.len() - suffix.len();
    let mut pos = prefix.len();
    out_vals.clear();
    for s in 0..k {
        if s + 1 == k {
            out_vals.push((tv.slots[s], &line[pos..body_end]));
            return true;
        }
        let next_lit = &tv.literals[s + 1];
        if next_lit.is_empty() {
            return false;
        }
        let Some(rel) = tv.finders[s + 1].find(&line[pos..body_end]) else {
            return false;
        };
        let match_at = pos + rel;
        out_vals.push((tv.slots[s], &line[pos..match_at]));
        pos = match_at + next_lit.len();
    }
    true
}

// ============================================================================
// 4. Dialect-Exact CSV / TSV Row Splitter
// ============================================================================

pub fn split_csv_exact(
    line: &[u8],
    delim: u8,
    expected_fields: usize,
) -> Option<(Vec<&[u8]>, u64)> {
    if expected_fields == 0 || expected_fields > 62 {
        return None;
    }
    let mut cells = Vec::with_capacity(expected_fields);
    let mut mask = 0u64;
    let mut pos = 0usize;
    for f in 0..expected_fields {
        if pos < line.len() && line[pos] == b'"' {
            let start = pos + 1;
            let mut end = start;
            while end < line.len() && line[end] != b'"' {
                end += 1;
            }
            if end >= line.len() {
                return None;
            }
            let after = end + 1;
            if f + 1 < expected_fields {
                if after >= line.len() || line[after] != delim {
                    return None;
                }
                pos = after + 1;
            } else if after != line.len() {
                return None;
            } else {
                pos = after;
            }
            mask |= 1u64 << f;
            cells.push(&line[start..end]);
        } else {
            let start = pos;
            while pos < line.len() && line[pos] != delim {
                if line[pos] == b'"' {
                    return None;
                }
                pos += 1;
            }
            cells.push(&line[start..pos]);
            if f + 1 < expected_fields {
                if pos >= line.len() || line[pos] != delim {
                    return None;
                }
                pos += 1;
            } else if pos != line.len() {
                return None;
            }
        }
    }
    Some((cells, mask))
}

// ============================================================================
// 5. Dense Occurrence Block Splitter & Streaming Block Reconstructor
// ============================================================================

/// Scratch arena holding synthesized per-row bytes (recipes, quote masks, unescaped nested JSON strings).
#[derive(Default)]
pub struct RowScratch {
    pub items: Vec<Vec<u8>>,
}

impl RowScratch {
    #[inline]
    pub fn alloc(&mut self, b: Vec<u8>) -> usize {
        let idx = self.items.len();
        self.items.push(b);
        idx
    }
}

#[derive(Clone, Copy)]
pub enum OccRef<'a> {
    Borrowed(&'a [u8]),
    Scratch(usize),
}

impl<'a> OccRef<'a> {
    #[inline]
    pub fn as_slice<'b>(&'b self, scratch: &'b RowScratch) -> &'b [u8]
    where
        'a: 'b,
    {
        match *self {
            OccRef::Borrowed(b) => b,
            OccRef::Scratch(idx) => scratch.items[idx].as_slice(),
        }
    }
}

pub struct SplitResult<'a> {
    pub variant_ids: Vec<u32>,
    /// Dense value occurrences per column (`0` unused presence cells!).
    pub col_occs: Vec<Vec<OccRef<'a>>>,
    /// By column: every record put exactly one occurrence in it.
    pub per_row: Vec<bool>,
    /// Native `u32` structural streams (`shape`, `count`, `sel`) when `plan.json_tree` is present.
    pub json_struct_streams: Vec<Vec<u32>>,
    pub scratch: RowScratch,
    pub residual_rows: usize,
    pub residual_bytes: usize,
}

/// Routes every record in `records` through `plan`, appending only present occurrences to `col_occs[cid]`!
pub fn split_records_with_plan<'a>(records: &[&'a [u8]], plan: &ModelPlan) -> SplitResult<'a> {
    let n = records.len();
    let n_cols = plan.columns.len();
    let mut variant_ids = Vec::with_capacity(n);
    let mut col_occs: Vec<Vec<OccRef<'a>>> = (0..n_cols).map(|_| Vec::new()).collect();
    let mut scratch = RowScratch::default();
    let mut residual_rows = 0usize;
    let mut residual_bytes = 0usize;
    let mut tmp_vals = Vec::<(u16, &'a [u8])>::with_capacity(32);

    let has_nested: Vec<bool> = plan
        .columns
        .iter()
        .map(|c| c.nested_plan.is_some())
        .collect();
    let json_vi = plan.json_tree_variant_id();
    let mut json_splitter = plan.json_tree.as_ref().map(JsonTreeBlockSplitter::new);

    let logfmt_vi = plan
        .variants
        .iter()
        .position(|v| matches!(v, PlanVariant::Logfmt))
        .map(|i| i as u32);
    let cri_var = plan.variants.iter().enumerate().find_map(|(i, v)| match v {
        PlanVariant::Envelope(e) if e.kind == EnvelopeKind::KubernetesCri => Some((i as u32, *e)),
        _ => None,
    });
    let syslog_var = plan.variants.iter().enumerate().find_map(|(i, v)| match v {
        PlanVariant::Envelope(e) if e.kind == EnvelopeKind::Syslog => Some((i as u32, *e)),
        _ => None,
    });
    let csv_var = plan.variants.iter().enumerate().find_map(|(i, v)| match v {
        PlanVariant::Csv(c) => Some((i as u32, c)),
        _ => None,
    });
    let bin_var = plan.variants.iter().enumerate().find_map(|(i, v)| match v {
        PlanVariant::BinaryStride(b) => Some((i as u32, b)),
        _ => None,
    });

    // Columns holding one occurrence of each record so far.
    let mut one_each: Vec<u16> = (0..n_cols as u16).collect();
    for (r, &line) in records.iter().enumerate() {
        one_each.retain(|&c| col_occs[c as usize].len() == r);
        // 1. Fixed-Stride Binary
        if let Some((vi, bs)) = bin_var
            && line.len() == bs.stride as usize
        {
            variant_ids.push(vi);
            for &(off, len, cid) in &bs.slices {
                let s = off as usize;
                col_occs[cid as usize].push(OccRef::Borrowed(&line[s..s + len as usize]));
            }
            continue;
        }

        // 2. Hierarchical JSON Tree (`JsonTreePlan`)
        if let (Some(vi), Some(jt), Some(js)) = (json_vi, &plan.json_tree, json_splitter.as_mut())
            && line.iter().copied().find(|b| !b.is_ascii_whitespace()) == Some(b'{')
            && js.split_row(line, jt, &has_nested, &mut col_occs, &mut scratch)
        {
            variant_ids.push(vi);
            continue;
        }

        // 3. Envelopes (Kubernetes CRI & Syslog)
        if let Some((vi, env)) = cri_var
            && let Some((ts, stream, flag, inner)) = parse_k8s_cri(line)
        {
            variant_ids.push(vi);
            col_occs[env.cols[0] as usize].push(OccRef::Borrowed(ts));
            col_occs[env.cols[1] as usize].push(OccRef::Borrowed(stream));
            col_occs[env.cols[2] as usize].push(OccRef::Borrowed(flag));
            col_occs[env.cols[3] as usize].push(OccRef::Borrowed(inner));
            continue;
        }
        if let Some((vi, env)) = syslog_var
            && let Some((pri, ts, host_tag, inner)) = parse_syslog_envelope(line)
        {
            variant_ids.push(vi);
            col_occs[env.cols[0] as usize].push(OccRef::Borrowed(pri));
            col_occs[env.cols[1] as usize].push(OccRef::Borrowed(ts));
            col_occs[env.cols[2] as usize].push(OccRef::Borrowed(host_tag));
            col_occs[env.cols[3] as usize].push(OccRef::Borrowed(inner));
            continue;
        }

        // 4. `logfmt` (`key=value`)
        if let (Some(vi), Some(lf)) = (logfmt_vi, &plan.logfmt)
            && let Some(pairs) = parse_logfmt_line(line)
        {
            let mut recipe = Vec::with_capacity(pairs.len() * 8);
            put_uvarint(&mut recipe, pairs.len() as u64);
            let mut overflow_vals = Vec::<&[u8]>::new();
            let mut assigned = Vec::<(u16, &[u8])>::with_capacity(pairs.len());
            let mut ok = true;
            for p in &pairs {
                put_bytes(&mut recipe, p.sep_before);
                recipe.push(u8::from(p.quoted));
                if let Some(&cid) = lf.key_to_col.get(p.key)
                    && !assigned.iter().any(|&(c, _)| c == cid)
                {
                    recipe.push(1);
                    put_uvarint(&mut recipe, cid as u64);
                    assigned.push((cid, p.val_inner));
                } else {
                    if p.val_inner.contains(&0x1f) {
                        ok = false;
                        break;
                    }
                    recipe.push(0);
                    put_bytes(&mut recipe, p.key);
                    overflow_vals.push(p.val_inner);
                }
            }
            if ok {
                // Push overflow flag byte (`1` if any overflow values on this row, else `0`) so dense overflow stream stays exact!
                recipe.insert(0, u8::from(!overflow_vals.is_empty()));
                variant_ids.push(vi);
                let r_idx = scratch.alloc(recipe);
                col_occs[lf.recipe_col as usize].push(OccRef::Scratch(r_idx));
                if !overflow_vals.is_empty() {
                    let mut ov = Vec::new();
                    for (oi, v) in overflow_vals.iter().enumerate() {
                        if oi > 0 {
                            ov.push(0x1f);
                        }
                        ov.extend_from_slice(v);
                    }
                    let o_idx = scratch.alloc(ov);
                    col_occs[lf.overflow_col as usize].push(OccRef::Scratch(o_idx));
                }
                for (cid, val) in assigned {
                    col_occs[cid as usize].push(OccRef::Borrowed(val));
                }
                continue;
            }
        }

        // 5. CSV / TSV Dialect
        if let Some((vi, cs)) = csv_var
            && let Some((cells, qmask)) = split_csv_exact(line, cs.delim, cs.field_cols.len())
        {
            variant_ids.push(vi);
            let q_idx = scratch.alloc(qmask.to_string().into_bytes());
            col_occs[cs.quote_mask_col as usize].push(OccRef::Scratch(q_idx));
            for (fi, &cid) in cs.field_cols.iter().enumerate() {
                col_occs[cid as usize].push(OccRef::Borrowed(cells[fi]));
            }
            continue;
        }

        // 6. Plain-Text Templates
        let mut matched_tpl = None;
        for &vi in &plan.template_indices {
            if let PlanVariant::Template(tv) = &plan.variants[vi as usize]
                && match_template(line, tv, &mut tmp_vals)
            {
                matched_tpl = Some(vi);
                break;
            }
        }
        if let Some(vi) = matched_tpl {
            variant_ids.push(vi);
            for &(cid, val) in &tmp_vals {
                col_occs[cid as usize].push(OccRef::Borrowed(val));
            }
            continue;
        }

        // 7. Safe Verbatim Residual (Variant 0)
        variant_ids.push(0);
        col_occs[0].push(OccRef::Borrowed(line));
        residual_rows += 1;
        residual_bytes += line.len();
    }

    let json_struct_streams = json_splitter
        .map(|js| js.struct_streams)
        .unwrap_or_default();

    let mut per_row = vec![false; n_cols];
    for c in one_each
        .into_iter()
        .filter(|&c| n > 0 && col_occs[c as usize].len() == n)
    {
        per_row[c as usize] = true;
    }

    SplitResult {
        variant_ids,
        col_occs,
        per_row,
        json_struct_streams,
        scratch,
        residual_rows,
        residual_bytes,
    }
}

/// Stateful per-block row reconstructor advancing dense column cursors (`col_pos`) in O(occurrences)!
pub struct BlockReconstructor {
    col_pos: Vec<usize>,
    json_rec: Option<JsonTreeBlockReconstructor>,
}

impl BlockReconstructor {
    pub fn new(plan: &ModelPlan, json_struct_streams: Vec<Vec<u32>>) -> Self {
        let json_rec = plan
            .json_tree
            .as_ref()
            .map(|jt| JsonTreeBlockReconstructor::new(json_struct_streams, jt.paths.len()));
        Self {
            col_pos: vec![0usize; plan.columns.len()],
            json_rec,
        }
    }

    pub fn json_streams(&self) -> &[Vec<u32>] {
        self.json_rec
            .as_ref()
            .map_or(&[], |j| j.struct_streams.as_slice())
    }

    /// Appends the next unread cell of column `cid` to `out`.
    #[inline]
    fn emit(
        col_pos: &mut [usize],
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        budget: &DecodeBudget,
        out: &mut Vec<u8>,
        cid: u16,
    ) -> Result<()> {
        let c = cid as usize;
        let p = col_pos
            .get_mut(c)
            .ok_or_else(|| Error::Format(format!("col_id {c} out of range")))?;
        *p += 1;
        block_col(c, plan, ctx, budget)?.write(*p - 1, out)
    }

    pub fn reconstruct_next(
        &mut self,
        variant_id: u32,
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        out: &mut Vec<u8>,
        budget: &DecodeBudget,
    ) -> Result<()> {
        let var = plan
            .variants
            .get(variant_id as usize)
            .ok_or_else(|| Error::Format(format!("variant_id {variant_id} out of range")))?;

        match var {
            PlanVariant::Residual { col } => {
                Self::emit(&mut self.col_pos, plan, ctx, budget, out, *col)?;
            }
            PlanVariant::JsonTree => {
                let jt = plan
                    .json_tree
                    .as_ref()
                    .ok_or_else(|| Error::Format("JsonTree variant without JsonTreePlan".into()))?;
                let jr = self
                    .json_rec
                    .as_mut()
                    .ok_or_else(|| Error::Format("missing JsonTreeBlockReconstructor".into()))?;
                jr.reconstruct_row(jt, plan, ctx, &mut self.col_pos, out, budget)?;
            }
            PlanVariant::Envelope(env) => match env.kind {
                EnvelopeKind::KubernetesCri => {
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, env.cols[0])?;
                    out.push(b' ');
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, env.cols[1])?;
                    out.push(b' ');
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, env.cols[2])?;
                    out.push(b' ');
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, env.cols[3])?;
                }
                EnvelopeKind::Syslog => {
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, env.cols[0])?;
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, env.cols[1])?;
                    out.push(b' ');
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, env.cols[2])?;
                    out.extend_from_slice(b": ");
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, env.cols[3])?;
                }
            },
            PlanVariant::Logfmt => {
                let lf = plan
                    .logfmt
                    .as_ref()
                    .ok_or_else(|| Error::Format("Logfmt variant without config".into()))?;
                let (mut recipe, mut overflow) = (Vec::new(), Vec::new());
                Self::emit(
                    &mut self.col_pos,
                    plan,
                    ctx,
                    budget,
                    &mut recipe,
                    lf.recipe_col,
                )?;
                let mut rc = Cursor::new(&recipe);
                let has_ov = rc.u8()? != 0;
                let k = rc.uvar()? as usize;
                if has_ov {
                    Self::emit(
                        &mut self.col_pos,
                        plan,
                        ctx,
                        budget,
                        &mut overflow,
                        lf.overflow_col,
                    )?;
                }
                let mut ov_iter = has_ov.then(|| overflow.split(|&c| c == 0x1f));
                for _ in 0..k {
                    let sep = rc.bytes()?;
                    let quoted = rc.u8()? != 0;
                    let is_known = rc.u8()? != 0;
                    out.extend_from_slice(sep);
                    if is_known {
                        let cid = rc.uvar()? as u16;
                        let col_name = &plan.columns[cid as usize].name;
                        let key_name = col_name.strip_prefix("logfmt.").unwrap_or(col_name);
                        out.extend_from_slice(key_name.as_bytes());
                        out.push(b'=');
                        if quoted {
                            out.push(b'"');
                        }
                        Self::emit(&mut self.col_pos, plan, ctx, budget, out, cid)?;
                        if quoted {
                            out.push(b'"');
                        }
                    } else {
                        let key_b = rc.bytes()?;
                        let val = ov_iter
                            .as_mut()
                            .and_then(|it| it.next())
                            .ok_or_else(|| Error::Format("Logfmt overflow value missing".into()))?;
                        out.extend_from_slice(key_b);
                        out.push(b'=');
                        if quoted {
                            out.push(b'"');
                            out.extend_from_slice(val);
                            out.push(b'"');
                        } else {
                            out.extend_from_slice(val);
                        }
                    }
                }
            }
            PlanVariant::Template(tv) => {
                out.extend_from_slice(&tv.literals[0]);
                for (i, &cid) in tv.slots.iter().enumerate() {
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, cid)?;
                    out.extend_from_slice(&tv.literals[i + 1]);
                }
            }
            PlanVariant::Csv(cs) => {
                let mut mask_str = Vec::new();
                Self::emit(
                    &mut self.col_pos,
                    plan,
                    ctx,
                    budget,
                    &mut mask_str,
                    cs.quote_mask_col,
                )?;
                let mask: u64 = std::str::from_utf8(&mask_str)
                    .ok()
                    .and_then(|m| m.parse().ok())
                    .ok_or_else(|| Error::Format("invalid CSV quote mask".into()))?;
                for (fi, &cid) in cs.field_cols.iter().enumerate() {
                    if fi > 0 {
                        out.push(cs.delim);
                    }
                    let quoted = (mask & (1u64 << fi)) != 0;
                    if quoted {
                        out.push(b'"');
                    }
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, cid)?;
                    if quoted {
                        out.push(b'"');
                    }
                }
            }
            PlanVariant::BinaryStride(bs) => {
                for &(_off, _len, cid) in &bs.slices {
                    Self::emit(&mut self.col_pos, plan, ctx, budget, out, cid)?;
                }
            }
        }
        Ok(())
    }

    /// Advances occurrence cursors (`col_pos` and `json_rec.struct_pos`) across a skipped row without
    /// touching value columns (except `lf.recipe_col` when skipping a `Logfmt` row)!
    pub fn skip_next(
        &mut self,
        variant_id: u32,
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        budget: &DecodeBudget,
    ) -> Result<()> {
        let var = plan
            .variants
            .get(variant_id as usize)
            .ok_or_else(|| Error::Format(format!("variant_id {variant_id} out of range")))?;
        match var {
            PlanVariant::Residual { col } => {
                self.col_pos[*col as usize] += 1;
            }
            PlanVariant::JsonTree => {
                let jt = plan
                    .json_tree
                    .as_ref()
                    .ok_or_else(|| Error::Format("JsonTree variant without JsonTreePlan".into()))?;
                let jr = self
                    .json_rec
                    .as_mut()
                    .ok_or_else(|| Error::Format("missing JsonTreeBlockReconstructor".into()))?;
                jr.skip_row(jt, &mut self.col_pos, budget)?;
            }
            PlanVariant::Envelope(env) => {
                for c in env.cols {
                    self.col_pos[c as usize] += 1;
                }
            }
            PlanVariant::Logfmt => {
                let lf = plan
                    .logfmt
                    .as_ref()
                    .ok_or_else(|| Error::Format("Logfmt variant without config".into()))?;
                let mut recipe = Vec::new();
                Self::emit(
                    &mut self.col_pos,
                    plan,
                    ctx,
                    budget,
                    &mut recipe,
                    lf.recipe_col,
                )?;
                let mut rc = Cursor::new(&recipe);
                let has_ov = rc.u8()? != 0;
                let k = rc.uvar()? as usize;
                if has_ov {
                    self.col_pos[lf.overflow_col as usize] += 1;
                }
                for _ in 0..k {
                    let _ = rc.bytes()?;
                    let _ = rc.u8()?;
                    if rc.u8()? != 0 {
                        let cid = rc.uvar()? as usize;
                        self.col_pos[cid] += 1;
                    } else {
                        let _ = rc.bytes()?;
                    }
                }
            }
            PlanVariant::Template(tv) => {
                for &cid in &tv.slots {
                    self.col_pos[cid as usize] += 1;
                }
            }
            PlanVariant::Csv(cs) => {
                self.col_pos[cs.quote_mask_col as usize] += 1;
                for &cid in &cs.field_cols {
                    self.col_pos[cid as usize] += 1;
                }
            }
            PlanVariant::BinaryStride(bs) => {
                for &(_off, _len, cid) in &bs.slices {
                    self.col_pos[cid as usize] += 1;
                }
            }
        }
        Ok(())
    }
}
