//! Structural model (`ModelPlan`). A reader rebuilds every record from one of three variants:
//! `Shape` (literals around column cells), `ShapeCell` (a per-row shape stored in a column) and
//! `JsonTree`. Record syntax lives in the writer's front ends (templates, CRI and syslog
//! prefixes, logfmt, CSV, fixed-stride binary), which split records into those variants.

use crate::cells::{Col, underflow};
use crate::codecs::BlockDecodeCtx;
use crate::container::block_col;
use crate::json_tree::{JsonTreeBlockReconstructor, JsonTreeBlockSplitter, JsonTreePlan};
use crate::lex::{
    parse_exact_decimal, parse_exact_int, parse_hex_exact, parse_ipv4_or_port, parse_timestamp,
    parse_traceparent, parse_ulid, parse_uuid, short_eq, split_email,
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

/// A record layout: `literals[0] cell literals[1] ... literals[n]`, each cell the next unread
/// one of its slot's column.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Shape {
    pub literals: Vec<Vec<u8>>,
    pub slots: Vec<u16>,
}

impl Shape {
    pub fn encode(&self, out: &mut Vec<u8>) {
        put_uvarint(out, self.slots.len() as u64);
        for l in &self.literals {
            put_bytes(out, l);
        }
        for &c in &self.slots {
            put_uvarint(out, c as u64);
        }
    }

    /// Decodes a shape whose slots must name one of `n_cols` columns.
    pub fn decode(cur: &mut Cursor<'_>, n_cols: usize) -> Result<Self> {
        let n = cur.uvar()? as usize;
        // Every literal and slot takes a byte or more, so this bounds the allocations.
        if n > cur.remaining() / 2 {
            return Err(Error::Format("shape slot count exceeds its bytes".into()));
        }
        let literals = (0..=n)
            .map(|_| cur.bytes().map(<[u8]>::to_vec))
            .collect::<Result<_>>()?;
        let slots = (0..n)
            .map(|_| match cur.uvar()? {
                c if (c as usize) < n_cols => Ok(c as u16),
                c => Err(Error::Format(format!("shape slot column {c} out of range"))),
            })
            .collect::<Result<_>>()?;
        Ok(Self { literals, slots })
    }
}

/// The CRI and syslog front ends' literals around `[ts, stream, flag, inner]` and
/// `[pri, ts, host_tag, inner]`.
const CRI_LITS: [&[u8]; 5] = [b"", b" ", b" ", b" ", b""];
const SYSLOG_LITS: [&[u8]; 5] = [b"", b"", b" ", b": ", b""];

/// The shape an envelope front end writes over its four columns.
pub fn envelope_shape(syslog: bool, cols: [u16; 4]) -> Shape {
    let lits = if syslog { SYSLOG_LITS } else { CRI_LITS };
    Shape {
        literals: lits.iter().map(|l| l.to_vec()).collect(),
        slots: cols.to_vec(),
    }
}

/// Matcher for a template variant, built once: building a searcher costs more than a search.
#[derive(Debug, Clone)]
pub struct TemplateVariant {
    pub literals: Vec<Vec<u8>>,
    pub slots: Vec<u16>,
    /// Index of the longest interior literal, searched first to reject a line early; 0 if none.
    anchor_idx: usize,
    min_len: usize,
    finders: Vec<LitFinder>,
}

/// Finds a literal by its last [`PACKED`] bytes, which `memchr` searches with packed compares,
/// then checks the head: a longer needle would be verified a byte at a time, and template
/// literals often share a long head.
#[derive(Debug, Clone)]
struct LitFinder {
    tail: Finder<'static>,
    head: usize,
}

const PACKED: usize = 32;

impl LitFinder {
    fn new(lit: &[u8]) -> Self {
        let head = lit.len().saturating_sub(PACKED);
        Self {
            tail: Finder::new(&lit[head..]).into_owned(),
            head,
        }
    }

    /// The first occurrence of `lit`, the literal this was built from, in `hay`.
    #[inline]
    fn find(&self, lit: &[u8], hay: &[u8]) -> Option<usize> {
        if let [b] = lit {
            return memchr::memchr(*b, hay);
        }
        let mut from = self.head;
        loop {
            let at = from + self.tail.find(hay.get(from..)?)?;
            if hay[at - self.head..at] == lit[..self.head] {
                return Some(at - self.head);
            }
            from = at + 1;
        }
    }
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
            finders: literals.iter().map(|l| LitFinder::new(l)).collect(),
            literals,
            slots,
        }
    }
}

/// What a reader rebuilds a record with. Variant `0` is a one-slot shape over `$residual`.
#[derive(Debug, Clone)]
pub enum PlanVariant {
    Shape(Shape),
    JsonTree,
    /// The next cell of `col` holds the record's shape, whose slots name only `cols`.
    ShapeCell {
        col: u16,
        cols: Vec<u16>,
    },
}

/// The writer front end that made a variant, with what it needs to route records there again.
/// Stored as hints a reader skips.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrontEnd {
    /// Variant 0, the JSON tree, or a hint this writer does not know.
    None,
    Template,
    Cri,
    Syslog,
    /// Each slot is a field `width` bytes wide; the record is the fields back to back.
    Binary {
        widths: Vec<u16>,
    },
    /// A logfmt layout seen in training.
    Logfmt,
    /// Every other logfmt layout. The variant's other columns hold keys, a key's text being
    /// its column's name after `logfmt.`; a value whose key has none is a cell of `overflow`.
    LogfmtCell {
        overflow: u16,
    },
    /// A CSV quote pattern seen in training.
    Csv,
    /// Every other CSV quote pattern; the variant's columns are the fields in order.
    CsvCell {
        delim: u8,
    },
}

impl FrontEnd {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            FrontEnd::None => out.push(0),
            FrontEnd::Template => out.push(1),
            FrontEnd::Cri => out.push(2),
            FrontEnd::Syslog => out.push(3),
            FrontEnd::Binary { widths } => {
                out.push(4);
                put_uvars(out, widths);
            }
            FrontEnd::Logfmt => out.push(5),
            FrontEnd::LogfmtCell { overflow } => {
                out.push(6);
                put_uvarint(out, *overflow as u64);
            }
            FrontEnd::Csv => out.push(7),
            FrontEnd::CsvCell { delim } => out.extend([8, *delim]),
        }
    }

    fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        Ok(match cur.u8()? {
            0 => FrontEnd::None,
            1 => FrontEnd::Template,
            2 => FrontEnd::Cri,
            3 => FrontEnd::Syslog,
            4 => FrontEnd::Binary {
                widths: uvars(cur)?,
            },
            5 => FrontEnd::Logfmt,
            6 => FrontEnd::LogfmtCell {
                overflow: cur.uvar()? as u16,
            },
            7 => FrontEnd::Csv,
            8 => FrontEnd::CsvCell { delim: cur.u8()? },
            t => return Err(Error::Format(format!("unknown front end {t}"))),
        })
    }
}

fn put_uvars(out: &mut Vec<u8>, v: &[u16]) {
    put_uvarint(out, v.len() as u64);
    v.iter().for_each(|&x| put_uvarint(out, x as u64));
}

fn uvars(cur: &mut Cursor<'_>) -> Result<Vec<u16>> {
    let n = cur.uvar()? as usize;
    if n > cur.remaining() {
        return Err(underflow());
    }
    (0..n).map(|_| Ok(cur.uvar()? as u16)).collect()
}

/// Writer routing built from `fronts`. A hint that does not fit its variant routes nothing.
#[derive(Debug, Clone, Default)]
pub struct Routes {
    /// Template variants in variant order.
    pub templates: Vec<(u32, TemplateVariant)>,
    cri: Option<u32>,
    syslog: Option<u32>,
    /// Variant, record length, and `(offset, width, column)` of each field.
    pub(crate) binary: Option<(u32, usize, Vec<(usize, usize, u16)>)>,
    logfmt: Option<LogfmtRoute>,
    csv: Option<CsvRoute>,
}

#[derive(Debug, Clone)]
pub struct LogfmtRoute {
    /// The `ShapeCell` variant and its column.
    cell: u32,
    shape_col: u16,
    overflow: u16,
    keys: HashMap<Vec<u8>, u16>,
    /// Encoded shape of each layout variant.
    layouts: HashMap<Vec<u8>, u32>,
}

#[derive(Debug, Clone)]
pub struct CsvRoute {
    cell: u32,
    shape_col: u16,
    delim: u8,
    fields: Vec<u16>,
    /// Quote mask of each layout variant.
    layouts: HashMap<Vec<u64>, u32>,
}

/// Complete, self-contained structural model embedded in `ZFM1` frames and `ZFP1` profile files.
#[derive(Debug, Clone)]
pub struct ModelPlan {
    pub columns: Vec<ColumnDef>,
    pub variants: Vec<PlanVariant>,
    /// By variant.
    pub fronts: Vec<FrontEnd>,
    pub json_tree: Option<JsonTreePlan>,
    pub routes: Routes,
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
            variants: vec![PlanVariant::Shape(Shape {
                literals: vec![Vec::new(), Vec::new()],
                slots: vec![0],
            })],
            fronts: vec![FrontEnd::None],
            json_tree: None,
            routes: Routes::default(),
            library: Arc::default(),
        }
    }

    pub fn push_variant(&mut self, v: PlanVariant, front: FrontEnd) {
        self.variants.push(v);
        self.fronts.push(front);
    }

    pub fn rebuild_indexes(&mut self) {
        let mut r = Routes::default();
        for (vi, (v, front)) in self.variants.iter().zip(&self.fronts).enumerate() {
            let (PlanVariant::ShapeCell { col, cols }, vi) = (v, vi as u32) else {
                continue;
            };
            match front {
                FrontEnd::LogfmtCell { overflow }
                    if r.logfmt.is_none() && cols.contains(overflow) =>
                {
                    let keys = cols.iter().filter(|&c| c != overflow);
                    let key = |c: u16| {
                        let name = &self.columns[c as usize].name;
                        name.strip_prefix("logfmt.")
                            .unwrap_or(name)
                            .as_bytes()
                            .to_vec()
                    };
                    r.logfmt = Some(LogfmtRoute {
                        cell: vi,
                        shape_col: *col,
                        overflow: *overflow,
                        keys: keys.map(|&c| (key(c), c)).collect(),
                        layouts: HashMap::new(),
                    });
                }
                FrontEnd::CsvCell { delim }
                    if r.csv.is_none() && (1..=MAX_CSV_FIELDS).contains(&cols.len()) =>
                {
                    r.csv = Some(CsvRoute {
                        cell: vi,
                        shape_col: *col,
                        delim: *delim,
                        fields: cols.clone(),
                        layouts: HashMap::new(),
                    });
                }
                _ => {}
            }
        }
        for (vi, (v, front)) in self.variants.iter().zip(&self.fronts).enumerate() {
            let (PlanVariant::Shape(s), vi) = (v, vi as u32) else {
                continue;
            };
            let lits_are = |want: [&[u8]; 5]| s.slots.len() == 4 && s.literals == want;
            match front {
                FrontEnd::Template => r.templates.push((
                    vi,
                    TemplateVariant::new(s.literals.clone(), s.slots.clone()),
                )),
                FrontEnd::Cri if lits_are(CRI_LITS) => _ = r.cri.get_or_insert(vi),
                FrontEnd::Syslog if lits_are(SYSLOG_LITS) => _ = r.syslog.get_or_insert(vi),
                FrontEnd::Binary { widths }
                    if r.binary.is_none()
                        && widths.len() == s.slots.len()
                        && s.literals.iter().all(Vec::is_empty) =>
                {
                    let mut off = 0usize;
                    let fields = widths
                        .iter()
                        .zip(&s.slots)
                        .map(|(&w, &c)| {
                            off += w as usize;
                            (off - w as usize, w as usize, c)
                        })
                        .collect();
                    r.binary = Some((vi, off, fields));
                }
                FrontEnd::Logfmt if r.logfmt.is_some() => {
                    let mut enc = Vec::new();
                    s.encode(&mut enc);
                    let lf = r.logfmt.as_mut().unwrap();
                    lf.layouts.entry(enc).or_insert(vi);
                }
                FrontEnd::Csv if r.csv.is_some() => {
                    let cs = r.csv.as_mut().unwrap();
                    let quoted = |i: usize| s.literals.get(i).is_some_and(|l| l.ends_with(b"\""));
                    let mut mask = vec![0u64; cs.fields.len().div_ceil(64)];
                    for i in 0..cs.fields.len() {
                        mask[i / 64] |= u64::from(quoted(i)) << (i % 64);
                    }
                    if csv_shape(cs.delim, &cs.fields, &mask) == *s {
                        cs.layouts.entry(mask).or_insert(vi);
                    }
                }
                _ => {}
            }
        }
        self.routes = r;
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
        put_uvarint(out, self.variants.len() as u64);
        for v in &self.variants {
            match v {
                PlanVariant::Shape(s) => {
                    out.push(0);
                    s.encode(out);
                }
                PlanVariant::JsonTree => out.push(1),
                PlanVariant::ShapeCell { col, cols } => {
                    out.push(2);
                    put_uvarint(out, *col as u64);
                    put_uvars(out, cols);
                }
            }
        }
        let mut hints = Vec::new();
        self.fronts.iter().for_each(|f| f.encode(&mut hints));
        put_bytes(out, &hints);
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
        let n_vars = cur.uvar()? as usize;
        if n_vars == 0 || n_vars > 8192 {
            return Err(Error::Format("invalid variant count".into()));
        }
        let mut variants = Vec::with_capacity(n_vars);
        for _ in 0..n_vars {
            match cur.u8()? {
                0 => variants.push(PlanVariant::Shape(Shape::decode(cur, n_cols)?)),
                1 => variants.push(PlanVariant::JsonTree),
                2 => {
                    let col = cur.uvar()? as usize;
                    let cols = uvars(cur)?;
                    if col >= n_cols
                        || cols
                            .iter()
                            .any(|&c| c as usize >= n_cols || c as usize == col)
                    {
                        return Err(Error::Format("ShapeCell columns out of range".into()));
                    }
                    variants.push(PlanVariant::ShapeCell {
                        col: col as u16,
                        cols,
                    })
                }
                tag => return Err(Error::Format(format!("unknown PlanVariant tag {tag}"))),
            }
        }
        // The writer stores what it cannot place in variant 0.
        if !matches!(variants.first(), Some(PlanVariant::Shape(s))
            if s.slots == [0] && s.literals.iter().all(Vec::is_empty))
        {
            return Err(Error::Format("variant 0 is not the residual shape".into()));
        }
        // Hints only steer a writer, so one that does not parse is dropped, not an error.
        let mut hc = Cursor::new(cur.bytes()?);
        let fronts = (0..n_vars)
            .map(|_| FrontEnd::decode(&mut hc))
            .collect::<Result<_>>()
            .unwrap_or_else(|_| vec![FrontEnd::None; n_vars]);
        let mut plan = Self {
            columns,
            variants,
            fronts,
            json_tree,
            routes: Routes::default(),
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
    } else {
        let r = rest1.strip_prefix(b"stderr ")?;
        (&b"stderr"[..], r)
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

/// Encodes the shape of a logfmt line into `out` and lists its values in `vals`. Separators,
/// keys, `=` and quotes are literals; a key's first value fills the key's column, and any other
/// value is a cell of `overflow`.
pub fn logfmt_shape<'a>(
    pairs: &[LogfmtPair<'a>],
    keys: &HashMap<Vec<u8>, u16>,
    overflow: u16,
    out: &mut Vec<u8>,
    vals: &mut Vec<(u16, &'a [u8])>,
) {
    out.clear();
    vals.clear();
    put_uvarint(out, pairs.len() as u64);
    let mut lit = Vec::with_capacity(64);
    for p in pairs {
        let quote = if p.quoted { &b"\""[..] } else { b"" };
        lit.extend_from_slice(p.sep_before);
        lit.extend_from_slice(p.key);
        lit.push(b'=');
        lit.extend_from_slice(quote);
        put_bytes(out, &lit);
        lit.clear();
        lit.extend_from_slice(quote);
        let col = match keys.get(p.key) {
            Some(&c) if !vals.iter().any(|v| v.0 == c) => c,
            _ => overflow,
        };
        vals.push((col, p.val_inner));
    }
    put_bytes(out, &lit);
    vals.iter().for_each(|v| put_uvarint(out, v.0 as u64));
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
        let b = line[i];
        // 1. ISO-8601 / CLF / Syslog timestamp at boundary (only starts with digit or month initial).
        if (b.is_ascii_digit()
            || matches!(b, b'J' | b'F' | b'M' | b'A' | b'S' | b'O' | b'N' | b'D'))
            && (i == 0 || !line[i - 1].is_ascii_alphanumeric())
            && let Some((end, cls)) = try_scan_timestamp_at(line, i)
        {
            spans.push((i, end, cls));
            i = end;
            continue;
        }
        // 2. Quoted string `"..."` (excluding empty `""` or >256B blobs)
        if b == b'"'
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
            // Standalone IPv4 (only if token starts with digit or `/digit`).
            if b.is_ascii_digit() || (b == b'/' && tok.get(1).is_some_and(u8::is_ascii_digit)) {
                let ip_end = memchr::memchr(b':', tok).map(|p| i + p).unwrap_or(tok_end);
                if b.is_ascii_digit()
                    && ip_end > i
                    && parse_ipv4_or_port(&line[i..ip_end]).is_some_and(|x| x.1 == 0)
                {
                    spans.push((i, ip_end, SlotClass::IpPort));
                    i = ip_end;
                    continue;
                }
                if b == b'/'
                    && ip_end > i + 1
                    && parse_ipv4_or_port(&line[i + 1..ip_end]).is_some_and(|x| x.1 == 0)
                {
                    spans.push((i + 1, ip_end, SlotClass::IpPort));
                    i = ip_end;
                    continue;
                }
            }
            if memchr::memchr(b'@', tok).is_some() && split_email(tok).is_some() {
                spans.push((i, tok_end, SlotClass::Email));
                i = tok_end;
                continue;
            }
            if b == b'h'
                && tok.len() <= 256
                && (tok.starts_with(b"http://") || tok.starts_with(b"https://"))
            {
                spans.push((i, tok_end, SlotClass::Path));
                i = tok_end;
                continue;
            }
            // Pure alphabetic word with at least one non-hex letter (`g..=z`/`G..=Z`) cannot hold
            // any numeric, hex, IP, or sub-token slot: skip the entire word in one jump.
            if b.is_ascii_alphabetic() {
                let mut all_alpha = true;
                let mut has_non_hex = false;
                for &c in tok {
                    if !c.is_ascii_alphabetic() {
                        all_alpha = false;
                        break;
                    }
                    if !c.is_ascii_hexdigit() {
                        has_non_hex = true;
                    }
                }
                if all_alpha && (has_non_hex || tok.len() < 8 || !tok.len().is_multiple_of(2)) {
                    i = tok_end;
                    continue;
                }
            }
        }
        // 4. Alphanumeric/numeric sub-tokens (keeps `blk_` and `/mnt/hadoop/.../job_` literal, masking only varying IDs!)
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
    if rem.len() >= 19
        && rem[4] == b'-'
        && rem[7] == b'-'
        && matches!(rem[13], b':' | b'.')
        && rem[16] == rem[13]
    {
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
    if line.len() < prefix.len().max(suffix.len())
        || !short_eq(&line[..prefix.len()], prefix)
        || !short_eq(&line[line.len() - suffix.len()..], suffix)
    {
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
    let body_end = line.len() - suffix.len();
    let a = tv.anchor_idx;
    // The anchor's first place in the body, which is its place in the walk below once the walk
    // has not passed it.
    let mut anchor_at = usize::MAX;
    if a > 0 && a < k {
        let body = &line[prefix.len()..body_end];
        match tv.finders[a].find(&tv.literals[a], body) {
            Some(at) => anchor_at = prefix.len() + at,
            None => return false,
        }
    }
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
        let match_at = if s + 1 == a && anchor_at >= pos {
            anchor_at
        } else {
            match tv.finders[s + 1].find(next_lit, &line[pos..body_end]) {
                Some(rel) => pos + rel,
                None => return false,
            }
        };
        out_vals.push((tv.slots[s], &line[pos..match_at]));
        pos = match_at + next_lit.len();
    }
    true
}

// ============================================================================
// 4. Dialect-Exact CSV / TSV Row Splitter
// ============================================================================

/// The shape of a CSV row whose fields `mask` marks are quoted.
pub fn csv_shape(delim: u8, fields: &[u16], mask: &[u64]) -> Shape {
    let q = |i: usize| {
        if mask[i / 64] >> (i % 64) & 1 != 0 {
            &b"\""[..]
        } else {
            b""
        }
    };
    let n = fields.len();
    let mut literals = vec![q(0).to_vec()];
    literals.extend((1..n).map(|i| [q(i - 1), &[delim], q(i)].concat()));
    literals.push(q(n - 1).to_vec());
    Shape {
        literals,
        slots: fields.to_vec(),
    }
}

pub fn split_csv_exact(
    line: &[u8],
    delim: u8,
    expected_fields: usize,
) -> Option<(Vec<&[u8]>, Vec<u64>)> {
    split_csv_fields(line, delim, Some(expected_fields))
}

pub(crate) const MAX_CSV_FIELDS: usize = 1024;

pub(crate) fn split_csv_fields(
    line: &[u8],
    delim: u8,
    expected_fields: Option<usize>,
) -> Option<(Vec<&[u8]>, Vec<u64>)> {
    if expected_fields.is_some_and(|n| n == 0 || n > MAX_CSV_FIELDS) {
        return None;
    }
    let capacity = expected_fields.unwrap_or(16);
    let mut cells = Vec::with_capacity(capacity);
    let mut mask = Vec::with_capacity(capacity.div_ceil(64));
    let mut pos = 0usize;
    loop {
        let f = cells.len();
        if f == MAX_CSV_FIELDS || expected_fields.is_some_and(|n| f == n) {
            return None;
        }
        if f.is_multiple_of(64) {
            mask.push(0u64);
        }
        if pos < line.len() && line[pos] == b'"' {
            let start = pos + 1;
            let mut end = start;
            while end < line.len() {
                if line[end] != b'"' {
                    end += 1;
                } else if line.get(end + 1) == Some(&b'"') {
                    end += 2;
                } else {
                    break;
                }
            }
            if end >= line.len() {
                return None;
            }
            pos = end + 1;
            if pos < line.len() && line[pos] != delim {
                return None;
            }
            mask[f / 64] |= 1u64 << (f % 64);
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
        }
        if pos == line.len() {
            break;
        }
        pos += 1;
    }
    if expected_fields.is_some_and(|n| cells.len() != n) {
        return None;
    }
    Some((cells, mask))
}

// ============================================================================
// 5. Dense Occurrence Block Splitter & Streaming Block Reconstructor
// ============================================================================

/// Scratch arena holding synthesized per-row bytes (shape cells, unescaped nested JSON strings).
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
    /// JSON objects stored whole because their key layout is not in the plan.
    pub unshaped_bytes: usize,
    /// Extra raw slices needed for bit-exact structural substring filtering (when requested).
    pub extra_filter_slices: Vec<&'a [u8]>,
    /// Set if any used `JsonCand` requires falling back to raw-record substring filtering.
    pub json_filter_unsafe: bool,
}

/// Routes every record in `records` through `plan`, appending only present occurrences to `col_occs[cid]`!
pub fn split_records_with_plan<'a>(records: &[&'a [u8]], plan: &ModelPlan) -> SplitResult<'a> {
    split_records_with_plan_opts(records, plan, false)
}

/// [`split_records_with_plan`], optionally capturing raw slices needed by structural filtering.
pub(crate) fn split_records_with_plan_opts<'a>(
    records: &[&'a [u8]],
    plan: &ModelPlan,
    collect_filter: bool,
) -> SplitResult<'a> {
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
    let mut json_splitter = plan
        .json_tree
        .as_ref()
        .map(|jt| JsonTreeBlockSplitter::with_filter(jt, collect_filter));

    let routes = &plan.routes;
    let slots_of = |vi: u32| match &plan.variants[vi as usize] {
        PlanVariant::Shape(s) => &s.slots[..],
        _ => &[],
    };
    let mut shape_buf = Vec::new();

    // Columns holding one occurrence of each record so far.
    let mut one_each: Vec<u16> = (0..n_cols as u16).collect();
    for (r, &line) in records.iter().enumerate() {
        one_each.retain(|&c| col_occs[c as usize].len() == r);
        // 1. Fixed-Stride Binary
        if let Some((vi, stride, fields)) = &routes.binary
            && line.len() == *stride
        {
            variant_ids.push(*vi);
            for &(off, len, cid) in fields {
                col_occs[cid as usize].push(OccRef::Borrowed(&line[off..off + len]));
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
        let envelope = routes
            .cri
            .and_then(|vi| Some((vi, parse_k8s_cri(line)?)))
            .or_else(|| Some((routes.syslog?, parse_syslog_envelope(line)?)));
        if let Some((vi, (a, b, c, d))) = envelope {
            variant_ids.push(vi);
            for (&cid, v) in slots_of(vi).iter().zip([a, b, c, d]) {
                col_occs[cid as usize].push(OccRef::Borrowed(v));
            }
            continue;
        }

        // 4. `logfmt` (`key=value`) and 5. CSV / TSV: a layout seen in training has a variant of
        // its own, any other is stored as a shape cell.
        let mut layout = None;
        if let Some(lf) = &routes.logfmt
            && let Some(pairs) = parse_logfmt_line(line)
        {
            logfmt_shape(&pairs, &lf.keys, lf.overflow, &mut shape_buf, &mut tmp_vals);
            layout = Some((lf.layouts.get(&shape_buf).copied(), lf.cell, lf.shape_col));
        } else if let Some(cs) = &routes.csv
            && let Some((cells, mask)) = split_csv_exact(line, cs.delim, cs.fields.len())
        {
            tmp_vals.clear();
            tmp_vals.extend(cs.fields.iter().copied().zip(cells));
            let fixed = cs.layouts.get(&mask).copied();
            if fixed.is_none() {
                shape_buf.clear();
                csv_shape(cs.delim, &cs.fields, &mask).encode(&mut shape_buf);
            }
            layout = Some((fixed, cs.cell, cs.shape_col));
        }
        if let Some((fixed, cell, shape_col)) = layout {
            match fixed {
                Some(vi) => variant_ids.push(vi),
                None => {
                    variant_ids.push(cell);
                    let i = scratch.alloc(shape_buf.clone());
                    col_occs[shape_col as usize].push(OccRef::Scratch(i));
                }
            }
            for &(cid, val) in &tmp_vals {
                col_occs[cid as usize].push(OccRef::Borrowed(val));
            }
            continue;
        }

        // 6. Plain-Text Templates
        let matched_tpl = routes
            .templates
            .iter()
            .find(|(_, tv)| match_template(line, tv, &mut tmp_vals));
        if let Some(&(vi, _)) = matched_tpl {
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

    let (json_struct_streams, unshaped_bytes, extra_filter_slices, json_filter_unsafe) =
        json_splitter
            .map(|js| {
                (
                    js.struct_streams,
                    js.unshaped_bytes,
                    js.extra_filter_slices,
                    js.filter_unsafe,
                )
            })
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
        unshaped_bytes,
        extra_filter_slices,
        json_filter_unsafe,
    }
}

/// Stateful per-block row reconstructor advancing dense column cursors (`col_pos`) in O(occurrences)!
pub struct BlockReconstructor {
    col_pos: Vec<usize>,
    json_rec: Option<JsonTreeBlockReconstructor>,
    cell_shapes: Vec<CellShapes>,
}

/// The shapes one `ShapeCell` column holds, each parsed once per block.
struct CellShapes {
    variant: u32,
    shapes: Vec<Shape>,
    /// By entry of a dictionary-coded column: index into `shapes` plus one, 0 until parsed.
    by_entry: Vec<u32>,
    by_bytes: HashMap<Vec<u8>, u32>,
}

impl BlockReconstructor {
    pub fn new(plan: &ModelPlan, json_struct_streams: Vec<Vec<u32>>) -> Self {
        let json_rec = plan
            .json_tree
            .as_ref()
            .map(|jt| JsonTreeBlockReconstructor::new(json_struct_streams, jt, plan));
        Self {
            col_pos: vec![0usize; plan.columns.len()],
            json_rec,
            cell_shapes: Vec::new(),
        }
    }

    /// Moves column `cid`'s cursor past `n` cells read elsewhere.
    pub(crate) fn advance(&mut self, cid: u16, n: usize) {
        self.col_pos[cid as usize] += n;
    }

    pub fn json_streams(&self) -> &[Vec<u32>] {
        self.json_rec
            .as_ref()
            .map_or(&[], |j| j.struct_streams.as_slice())
    }

    pub(crate) fn json_rec_mut(&mut self) -> Option<&mut JsonTreeBlockReconstructor> {
        self.json_rec.as_mut()
    }

    /// Appends the next unread cell of column `cid` to `out`.
    #[inline(always)]
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

    #[inline(always)]
    fn emit_shape(
        col_pos: &mut [usize],
        s: &Shape,
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        budget: &DecodeBudget,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        crate::cells::put_lit(out, &s.literals[0]);
        for (&cid, l) in s.slots.iter().zip(&s.literals[1..]) {
            Self::emit(col_pos, plan, ctx, budget, out, cid)?;
            crate::cells::put_lit(out, l);
        }
        Ok(())
    }

    /// Reads the next cell of `col` as the shape of a row of `ShapeCell` variant `variant`:
    /// `(cell_shapes index, shape index)`.
    fn cell_shape(
        &mut self,
        (variant, col, cols): (u32, u16, &[u16]),
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        budget: &DecodeBudget,
    ) -> Result<(usize, usize)> {
        let ci = match self.cell_shapes.iter().position(|c| c.variant == variant) {
            Some(ci) => ci,
            None => {
                self.cell_shapes.push(CellShapes {
                    variant,
                    shapes: Vec::new(),
                    by_entry: Vec::new(),
                    by_bytes: HashMap::new(),
                });
                self.cell_shapes.len() - 1
            }
        };
        let pos = self.col_pos[col as usize];
        self.col_pos[col as usize] += 1;
        let cs = &mut self.cell_shapes[ci];
        let mut parse = |cell: &[u8]| -> Result<u32> {
            let mut cur = Cursor::new(cell);
            let s = Shape::decode(&mut cur, plan.columns.len())?;
            if cur.remaining() != 0 || !s.slots.iter().all(|c| cols.contains(c)) {
                return Err(Error::Format("invalid shape cell".into()));
            }
            cs.shapes.push(s);
            Ok(cs.shapes.len() as u32)
        };
        let k = match block_col(col as usize, plan, ctx, budget)? {
            Col::Dict { dict, ords } => {
                let o = *ords.get(pos).ok_or_else(underflow)? as usize;
                if cs.by_entry.len() < dict.len() {
                    cs.by_entry.resize(dict.len(), 0);
                }
                if cs.by_entry[o] == 0 {
                    cs.by_entry[o] = parse(dict.get(o).ok_or_else(underflow)?)?;
                }
                cs.by_entry[o]
            }
            other => {
                let mut scratch = Vec::new();
                let cell = other.get(pos, &mut scratch)?;
                match cs.by_bytes.get(cell) {
                    Some(&k) => k,
                    None => {
                        let k = parse(cell)?;
                        cs.by_bytes.insert(cell.to_vec(), k);
                        k
                    }
                }
            }
        };
        Ok((ci, k as usize - 1))
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
            PlanVariant::Shape(s) => {
                Self::emit_shape(&mut self.col_pos, s, plan, ctx, budget, out)?
            }
            PlanVariant::ShapeCell { col, cols } => {
                let (ci, si) = self.cell_shape((variant_id, *col, cols), plan, ctx, budget)?;
                let start = out.len();
                let s = &self.cell_shapes[ci].shapes[si];
                Self::emit_shape(&mut self.col_pos, s, plan, ctx, budget, out)?;
                if out.len() - start > budget.max_block_decoded_bytes {
                    return Err(Error::BudgetExceeded("shape cell record size"));
                }
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
        }
        Ok(())
    }

    /// Advances occurrence cursors (`col_pos` and `json_rec.struct_pos`) across a skipped row,
    /// reading no value column but a `ShapeCell` row's shape.
    pub fn skip_next(
        &mut self,
        variant_id: u32,
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        budget: &DecodeBudget,
    ) -> Result<()> {
        self.skip_rows(&[variant_id], plan, ctx, budget)
    }

    /// `skip_next` across rows of variants `vids`; a run of JSON rows is skipped at once.
    pub fn skip_rows(
        &mut self,
        vids: &[u32],
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        budget: &DecodeBudget,
    ) -> Result<()> {
        let mut json = 0;
        for &variant_id in vids {
            let var = plan
                .variants
                .get(variant_id as usize)
                .ok_or_else(|| Error::Format(format!("variant_id {variant_id} out of range")))?;
            match var {
                PlanVariant::JsonTree => json += 1,
                PlanVariant::Shape(s) => {
                    s.slots.iter().for_each(|&c| self.col_pos[c as usize] += 1)
                }
                PlanVariant::ShapeCell { col, cols } => {
                    // The shape cell is read at its place among the rows before it.
                    self.skip_json(std::mem::take(&mut json), plan, budget)?;
                    let (ci, si) = self.cell_shape((variant_id, *col, cols), plan, ctx, budget)?;
                    for &c in &self.cell_shapes[ci].shapes[si].slots {
                        self.col_pos[c as usize] += 1;
                    }
                }
            }
        }
        self.skip_json(json, plan, budget)
    }

    /// The column and structure cursors, to `reset` to.
    pub fn mark(&self) -> (Vec<usize>, Vec<usize>) {
        let streams = self.json_rec.as_ref().map(|j| j.struct_pos.clone());
        (self.col_pos.clone(), streams.unwrap_or_default())
    }

    pub fn reset(&mut self, (cols, streams): (Vec<usize>, Vec<usize>)) {
        self.col_pos = cols;
        if let Some(j) = self.json_rec.as_mut() {
            j.struct_pos = streams;
        }
    }

    fn skip_json(&mut self, n: usize, plan: &ModelPlan, budget: &DecodeBudget) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let jt = plan
            .json_tree
            .as_ref()
            .ok_or_else(|| Error::Format("JsonTree variant without JsonTreePlan".into()))?;
        let jr = self
            .json_rec
            .as_mut()
            .ok_or_else(|| Error::Format("missing JsonTreeBlockReconstructor".into()))?;
        jr.skip_rows(n, jt, &mut self.col_pos, budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packs::PackRegistry;

    #[test]
    fn literals_are_found_first_as_memmem_finds_them() {
        let mut state = 7u64;
        for len in [1usize, 5, 32, 33, 40] {
            let lit: Vec<u8> = (0..len).map(|i| b'a' + (i % 5) as u8).collect();
            let f = LitFinder::new(&lit);
            // Pieces that hold the literal's tail without its head.
            let cut = len.saturating_sub(PACKED);
            let pieces: [&[u8]; 4] = [&lit, &lit[cut..], &lit[len / 2..], b"x"];
            for _ in 0..200 {
                let mut hay = Vec::new();
                for _ in 0..state % 6 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    hay.extend_from_slice(pieces[(state % 4) as usize]);
                }
                assert_eq!(f.find(&lit, &hay), memchr::memmem::find(&hay, &lit));
            }
        }
    }

    fn shape(lits: &[&[u8]], slots: &[u16]) -> Shape {
        let literals = lits.iter().map(|l| l.to_vec()).collect();
        Shape {
            literals,
            slots: slots.to_vec(),
        }
    }

    fn cell(s: &Shape) -> Vec<u8> {
        let mut b = Vec::new();
        s.encode(&mut b);
        b
    }

    fn reload(p: &ModelPlan) -> Result<ModelPlan> {
        let mut wire = Vec::new();
        p.encode(&mut wire);
        ModelPlan::decode(&mut Cursor::new(&wire), 0)
    }

    /// Columns `$residual`, `a`, `b`, `$shape`, `c`; variants: residual, `a=<a> b=<b>`, cell
    /// shapes over `a` and `b`.
    fn plan() -> ModelPlan {
        let mut p = ModelPlan::residual_only();
        for name in ["a", "b", "$shape", "c"] {
            p.add_or_get_col(name);
        }
        let ab = shape(&[b"a=", b" b=", b""], &[1, 2]);
        p.push_variant(PlanVariant::Shape(ab), FrontEnd::None);
        let cell = PlanVariant::ShapeCell {
            col: 3,
            cols: vec![1, 2],
        };
        p.push_variant(cell, FrontEnd::None);
        reload(&p).unwrap()
    }

    /// Rebuilds rows `vids` from `cols`, skipping the rows `skip` marks.
    fn rebuild(
        p: &ModelPlan,
        vids: &[u32],
        skip: &[bool],
        cols: &[&[&[u8]]],
    ) -> Result<Vec<Vec<u8>>> {
        let blobs: Vec<Vec<u8>> = cols
            .iter()
            .map(|c| PackRegistry::default().encode_best(c))
            .collect();
        let col_blobs = blobs.iter().map(|b| Some(&b[..])).collect();
        let mut ctx = BlockDecodeCtx::new(col_blobs, vec![], None, p.library.clone());
        let mut rec = BlockReconstructor::new(p, vec![]);
        let budget = DecodeBudget::default();
        let mut rows = Vec::new();
        for (&v, &skip) in vids.iter().zip(skip) {
            if skip {
                rec.skip_next(v, p, &mut ctx, &budget)?;
            } else {
                let mut out = Vec::new();
                rec.reconstruct_next(v, p, &mut ctx, &mut out, &budget)?;
                rows.push(out);
            }
        }
        Ok(rows)
    }

    #[test]
    fn shape_and_shape_cell_round_trip() {
        let p = plan();
        let ba = cell(&shape(&[b"b=", b" a=\"", b"\""], &[2, 1]));
        let aa = cell(&shape(&[b"", b",", b""], &[1, 1]));
        let cols: [&[&[u8]]; 5] = [
            &[b"x y"],
            &[b"1", b"2", b"3", b"4", b"5"],
            &[b"p", b"q", b"r"],
            &[&ba, &aa, &ba],
            &[],
        ];
        let vids = [1, 2, 0, 2, 2];
        let rows = rebuild(&p, &vids, &[false; 5], &cols).unwrap();
        let want: [&[u8]; 5] = [b"a=1 b=p", b"b=q a=\"2\"", b"x y", b"3,4", b"b=r a=\"5\""];
        assert_eq!(rows, want);
        // Skipping a cell row still advances the columns its shape names.
        let skip = [false, true, false, true, false];
        let rows = rebuild(&p, &vids, &skip, &cols).unwrap();
        assert_eq!(rows, [want[0], want[2], want[4]]);
    }

    #[test]
    fn rejects_bad_slots() {
        let p = plan();
        let mut bad = p.clone();
        bad.push_variant(PlanVariant::Shape(shape(&[b"", b""], &[5])), FrontEnd::None);
        assert!(reload(&bad).is_err());
        for (col, cols) in [(9, vec![1]), (3, vec![1, 5]), (3, vec![1, 3])] {
            let mut bad = p.clone();
            bad.push_variant(PlanVariant::ShapeCell { col, cols }, FrontEnd::None);
            assert!(reload(&bad).is_err());
        }

        // A cell shape may name only its variant's declared columns.
        let own = cell(&shape(&[b"", b""], &[3]));
        let wide = cell(&shape(&[b"", b""], &[5]));
        let undeclared = cell(&shape(&[b"", b""], &[4]));
        for c in [own, wide, undeclared] {
            let cols: [&[&[u8]]; 5] = [&[], &[b"1"], &[], &[&c], &[b"z"]];
            assert!(rebuild(&p, &[2], &[false], &cols).is_err());
            assert!(rebuild(&p, &[2], &[true], &cols).is_err());
        }
    }

    /// Trains on `train`, then packs `rows` and checks they come back whole; returns the variant
    /// each row of `rows` took.
    fn round_trip(train: &[&[u8]], rows: &[&[u8]]) -> (ModelPlan, Vec<PlanVariant>) {
        use crate::block::{BlockOptions, decode_block, encode_block};
        use crate::plan::TrainOptions;
        let model = crate::block::train(train, &TrainOptions::default(), None);
        let (bytes, fit) = encode_block(rows, &model, &BlockOptions::default());
        assert_eq!(fit.residual_rows, 0);
        let back = decode_block(&model, &bytes, rows.len(), &DecodeBudget::default()).unwrap();
        assert_eq!(back, rows);
        let split = split_records_with_plan(rows, &model);
        let vars = split.variant_ids.iter();
        let vars = vars.map(|&v| model.variants[v as usize].clone()).collect();
        (model, vars)
    }

    #[test]
    fn first_timestamp_shares_a_column_after_variable_prefixes() {
        let rows: Vec<Vec<u8>> = (0..16)
            .map(|i| {
                let msg = if i < 8 { "boot task" } else { "exit code" };
                format!("{} 2024-02-29-23.59.{i:02}.123456 {msg} {}", 100 + i, i * 3).into_bytes()
            })
            .collect();
        let rows: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let (plan, _) = round_trip(&rows, &rows);
        let timestamps: Vec<_> = plan
            .columns
            .iter()
            .filter(|c| c.name.starts_with("$ts."))
            .collect();
        assert_eq!(timestamps.len(), 1);
        assert_eq!(timestamps[0].name, "$ts.0");
    }

    #[test]
    fn logfmt_new_key_order_takes_a_shape_cell() {
        let train: Vec<Vec<u8>> = (0..8)
            .map(|i| {
                format!("level=info ts=2024-05-0{i}T00:00:00Z msg=\"hi {i}\" n={i}").into_bytes()
            })
            .collect();
        let train: Vec<&[u8]> = train.iter().map(|r| r.as_slice()).collect();
        let rows: [&[u8]; 3] = [
            train[0],
            b"msg=\"new order\" level=warn  n=7 ts=2024-05-09T00:00:00Z extra=\"a b\" n=8",
            train[1],
        ];
        let (_, vars) = round_trip(&train, &rows);
        assert!(matches!(vars[0], PlanVariant::Shape(_)));
        assert!(matches!(vars[1], PlanVariant::ShapeCell { .. }));
    }

    #[test]
    fn wide_csv_keeps_quoted_commas_escaped_quotes_and_new_layouts() {
        let row = |n: usize, extra_quotes: bool| {
            (0..130)
                .map(|i| match i {
                    65 => "\"city, district\"".to_string(),
                    128 => "\"a\"\"b\"".to_string(),
                    129 => String::new(),
                    64 | 127 if extra_quotes => format!("\"{}\"", i + n),
                    _ => (i + n).to_string(),
                })
                .collect::<Vec<_>>()
                .join(",")
                .into_bytes()
        };
        let training: Vec<Vec<u8>> = (0..8).map(|n| row(n, false)).collect();
        let train: Vec<&[u8]> = training.iter().map(Vec::as_slice).collect();
        let changed = row(9, true);
        let (model, variants) = round_trip(&train, &[changed.as_slice(), train[0]]);
        assert!(model.routes.csv.is_some());
        assert!(matches!(variants[0], PlanVariant::ShapeCell { .. }));
        assert!(matches!(variants[1], PlanVariant::Shape(_)));
        assert!(reload(&model).unwrap().routes.csv.is_some());
    }

    #[test]
    fn csv_splitter_bounds_fields_and_refuses_malformed_quotes() {
        let (fields, _) = split_csv_exact(b",\"\",", b',', 3).unwrap();
        assert_eq!(fields, [b""; 3]);
        for row in [b"\"a,b,c".as_slice(), b"a\"b,c,d", b"\"a\"x,b,c", b"a,b"] {
            assert!(split_csv_exact(row, b',', 3).is_none());
        }
        let limit = vec![b','; MAX_CSV_FIELDS - 1];
        assert_eq!(
            split_csv_fields(&limit, b',', None).unwrap().0.len(),
            MAX_CSV_FIELDS
        );
        assert!(split_csv_fields(&vec![b','; MAX_CSV_FIELDS], b',', None).is_none());
    }

    #[test]
    fn csv_new_quote_pattern_takes_a_shape_cell() {
        let train: Vec<Vec<u8>> = (0..8)
            .map(|i| format!("{i},name{i},\"ab\",{}", i * 3).into_bytes())
            .collect();
        let train: Vec<&[u8]> = train.iter().map(|r| r.as_slice()).collect();
        let rows: [&[u8]; 2] = [b"\"9\",x,y,\"\"", train[2]];
        let (_, vars) = round_trip(&train, &rows);
        assert!(matches!(vars[0], PlanVariant::ShapeCell { .. }));
        assert!(matches!(vars[1], PlanVariant::Shape(_)));
    }
}
