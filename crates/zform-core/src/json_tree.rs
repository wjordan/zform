//! Hierarchical JSON Tree Grammar (`JsonTreePlan`):
//! - Zero-allocation SWAR-accelerated JSON arena parser (`JsonDoc`)
//! - Recursive decomposition of nested objects (`SlotKind::Object`), arrays (`SlotKind::Array`,
//!   e.g. `$.payload.commits[]`), varying scalars (`SlotKind::Scalar`), and wide/drifted JSON
//!   subtrees (`SlotKind::Text`), while keeping constant fields inline in shape literals (`0` bytes)
//! - Plan-level cross-path derivations (`JsonCand`: `Copy`, `Affix`, `MultiConcat`) across ancestor
//!   and array-element scopes; a `MultiConcat` may hold one free part stored in the path's own
//!   column, or spell its value as Base64 of the concatenation
//! - Native `u32` structural streams (`shape`, `count`, `sel`) and compressed-domain query mapping.

use crate::codecs::{BlockDecodeCtx, decode_u32_seq, encode_u32_seq};
use crate::container::block_col;
use crate::grammar::ModelPlan;
use crate::lex::{escape_json_str_exact, unescape_json_str_exact};
use crate::wire::{Cursor, DecodeBudget, EntropyBackend, Error, Result, put_bytes, put_uvarint};
use std::collections::{HashMap, HashSet};

pub const J_NONE: u8 = 0;
pub const J_OBJECT: u8 = 1;
pub const J_ARRAY: u8 = 2;
pub const J_STRING: u8 = 3;
pub const J_NUMBER: u8 = 4;
pub const J_LITERAL: u8 = 5;

const MAX_PARSE_DEPTH: usize = 64;
const MAX_PLAN_PATHS: usize = 1024;
const MAX_PATH_SHAPES: usize = 64;
const MAX_SHAPE_PARTS: usize = 128;
const MAX_PATH_NAME_LEN: usize = 1024;
const TRAIN_WIDE_KEYS: usize = 256;
const TRAIN_SHAPE_MIN: f64 = 0.001;
const TRAIN_SHAPE_COVER: f64 = 0.99;

const DERIV_MIN_OCC: usize = 8;
const DERIV_MAX_OCC: usize = 512;
const DERIV_MAX_AFFIX: usize = 128;
const DERIV_MIN_REF: usize = 3;
const DERIV_MAX_REFS: usize = 4;
/// A candidate is kept if it fits at least this many sampled values, and one in this many.
const DERIV_MIN_HITS: usize = 4;
const DERIV_MIN_SHARE: usize = 10;
/// Proposals scored per path, and decompositions per ref list paired to find free parts.
const DERIV_PROPOSALS: usize = 12;
const DERIV_PAIR_SAMPLE: usize = 16;
const MAX_CANDS_PER_PATH: usize = 4;

// ============================================================================
// 1. SWAR-Accelerated Zero-Alloc JSON Arena Parser (`JsonDoc`)
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct JVal {
    pub kind: u8,
    pub start: u32,
    pub end: u32,
    pub first: u32,
    pub n: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct JKid {
    pub key_start: u32,
    pub key_end: u32,
    pub key_esc: bool,
    pub node: u32,
}

#[derive(Default, Clone)]
pub struct JsonDoc {
    pub nodes: Vec<JVal>,
    pub kids: Vec<JKid>,
    scratch_kids: Vec<JKid>,
}

#[inline]
fn skip_ws(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

#[inline]
fn is_hex(c: u8) -> bool {
    c.is_ascii_hexdigit()
}

/// Scans a JSON string starting at `s[i] == b'"'`, returning `(end_pos_after_quote, had_escape)`.
#[inline]
pub fn scan_json_str(s: &[u8], i: usize) -> Option<(usize, bool)> {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    const Q: u64 = LO * (b'"' as u64);
    const BS: u64 = LO * (b'\\' as u64);

    if s.get(i) != Some(&b'"') {
        return None;
    }
    let mut j = i + 1;
    let mut esc = false;
    while j < s.len() {
        while j + 8 <= s.len() {
            let w = u64::from_le_bytes(s[j..j + 8].try_into().unwrap());
            let q = (w ^ Q).wrapping_sub(LO) & !(w ^ Q) & HI;
            let b = (w ^ BS).wrapping_sub(LO) & !(w ^ BS) & HI;
            let ctrl = w.wrapping_sub(LO * 0x20) & !w & HI;
            let mask = q | b | ctrl;
            if mask == 0 {
                j += 8;
                continue;
            }
            j += (mask.trailing_zeros() >> 3) as usize;
            break;
        }
        if j >= s.len() {
            return None;
        }
        let c = s[j];
        if c == b'"' {
            return Some((j + 1, esc));
        }
        if c == b'\\' {
            esc = true;
            let &e = s.get(j + 1)?;
            match e {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => j += 2,
                b'u' => {
                    if j + 6 > s.len()
                        || !is_hex(s[j + 2])
                        || !is_hex(s[j + 3])
                        || !is_hex(s[j + 4])
                        || !is_hex(s[j + 5])
                    {
                        return None;
                    }
                    j += 6;
                }
                _ => return None,
            }
            continue;
        }
        if c < 0x20 {
            return None;
        }
        j += 1;
    }
    None
}

fn scan_json_num(s: &[u8], i: usize) -> Option<usize> {
    let mut j = i;
    if j < s.len() && s[j] == b'-' {
        j += 1;
    }
    if j >= s.len() {
        return None;
    }
    if s[j] == b'0' {
        j += 1;
    } else if matches!(s[j], b'1'..=b'9') {
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }
    } else {
        return None;
    }
    if j < s.len() && s[j] == b'.' {
        j += 1;
        let d0 = j;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }
        if j == d0 {
            return None;
        }
    }
    if j < s.len() && (s[j] == b'e' || s[j] == b'E') {
        j += 1;
        if j < s.len() && (s[j] == b'+' || s[j] == b'-') {
            j += 1;
        }
        let d0 = j;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
        }
        if j == d0 {
            return None;
        }
    }
    Some(j)
}

impl JsonDoc {
    #[inline]
    pub fn kid_range(&self, ni: usize) -> std::ops::Range<usize> {
        let v = self.nodes[ni];
        v.first as usize..(v.first + v.n) as usize
    }

    /// Parses `s` as a root JSON object spanning `0..s.len()` exactly (`s[0] == b'{'` and `s[last] == b'}'`),
    /// guaranteeing 100% byte-exact root reconstruction with zero leading/trailing whitespace ambiguity.
    pub fn parse_root_object(&mut self, s: &[u8]) -> bool {
        self.nodes.clear();
        self.kids.clear();
        self.scratch_kids.clear();
        if s.len() < 2 || s[0] != b'{' || s[s.len() - 1] != b'}' || s.len() > (u32::MAX as usize) {
            return false;
        }
        let Some((end, root)) = self.parse_value(s, 0, 0) else {
            return false;
        };
        root == 0 && end == s.len() && self.nodes[0].kind == J_OBJECT
    }

    /// Parses `s` as one JSON value with optional whitespace around it; returns its node.
    pub fn parse_any(&mut self, s: &[u8]) -> Option<usize> {
        self.nodes.clear();
        self.kids.clear();
        self.scratch_kids.clear();
        if s.len() > (u32::MAX as usize) {
            return None;
        }
        let (end, root) = self.parse_value(s, skip_ws(s, 0), 0)?;
        (skip_ws(s, end) == s.len()).then_some(root)
    }

    fn parse_value(&mut self, s: &[u8], i: usize, depth: usize) -> Option<(usize, usize)> {
        if i >= s.len() || depth > MAX_PARSE_DEPTH {
            return None;
        }
        match s[i] {
            b'"' => {
                let (end, _) = scan_json_str(s, i)?;
                let idx = self.nodes.len();
                self.nodes.push(JVal {
                    kind: J_STRING,
                    start: i as u32,
                    end: end as u32,
                    first: 0,
                    n: 0,
                });
                Some((end, idx))
            }
            b'{' => self.parse_object(s, i, depth),
            b'[' => self.parse_array(s, i, depth),
            b'-' | b'0'..=b'9' => {
                let end = scan_json_num(s, i)?;
                let idx = self.nodes.len();
                self.nodes.push(JVal {
                    kind: J_NUMBER,
                    start: i as u32,
                    end: end as u32,
                    first: 0,
                    n: 0,
                });
                Some((end, idx))
            }
            b't' if s[i..].starts_with(b"true") => {
                let idx = self.nodes.len();
                self.nodes.push(JVal {
                    kind: J_LITERAL,
                    start: i as u32,
                    end: (i + 4) as u32,
                    first: 0,
                    n: 0,
                });
                Some((i + 4, idx))
            }
            b'f' if s[i..].starts_with(b"false") => {
                let idx = self.nodes.len();
                self.nodes.push(JVal {
                    kind: J_LITERAL,
                    start: i as u32,
                    end: (i + 5) as u32,
                    first: 0,
                    n: 0,
                });
                Some((i + 5, idx))
            }
            b'n' if s[i..].starts_with(b"null") => {
                let idx = self.nodes.len();
                self.nodes.push(JVal {
                    kind: J_LITERAL,
                    start: i as u32,
                    end: (i + 4) as u32,
                    first: 0,
                    n: 0,
                });
                Some((i + 4, idx))
            }
            _ => None,
        }
    }

    fn parse_object(&mut self, s: &[u8], start: usize, depth: usize) -> Option<(usize, usize)> {
        let idx = self.nodes.len();
        self.nodes.push(JVal {
            kind: J_OBJECT,
            start: start as u32,
            end: 0,
            first: 0,
            n: 0,
        });
        let mut i = skip_ws(s, start + 1);
        if i < s.len() && s[i] == b'}' {
            self.nodes[idx].end = (i + 1) as u32;
            return Some((i + 1, idx));
        }
        let mark = self.scratch_kids.len();
        loop {
            if i >= s.len() || s[i] != b'"' {
                self.scratch_kids.truncate(mark);
                return None;
            }
            let (kend, kesc) = scan_json_str(s, i)?;
            let ks = (i + 1) as u32;
            let ke = (kend - 1) as u32;
            i = skip_ws(s, kend);
            if i >= s.len() || s[i] != b':' {
                self.scratch_kids.truncate(mark);
                return None;
            }
            i = skip_ws(s, i + 1);
            let Some((vend, child)) = self.parse_value(s, i, depth + 1) else {
                self.scratch_kids.truncate(mark);
                return None;
            };
            self.scratch_kids.push(JKid {
                key_start: ks,
                key_end: ke,
                key_esc: kesc,
                node: child as u32,
            });
            i = skip_ws(s, vend);
            if i >= s.len() {
                self.scratch_kids.truncate(mark);
                return None;
            }
            if s[i] == b',' {
                i = skip_ws(s, i + 1);
                continue;
            }
            if s[i] == b'}' {
                let first = self.kids.len() as u32;
                let n = (self.scratch_kids.len() - mark) as u32;
                self.kids.extend_from_slice(&self.scratch_kids[mark..]);
                self.scratch_kids.truncate(mark);
                self.nodes[idx].end = (i + 1) as u32;
                self.nodes[idx].first = first;
                self.nodes[idx].n = n;
                return Some((i + 1, idx));
            }
            self.scratch_kids.truncate(mark);
            return None;
        }
    }

    fn parse_array(&mut self, s: &[u8], start: usize, depth: usize) -> Option<(usize, usize)> {
        let idx = self.nodes.len();
        self.nodes.push(JVal {
            kind: J_ARRAY,
            start: start as u32,
            end: 0,
            first: 0,
            n: 0,
        });
        let mut i = skip_ws(s, start + 1);
        if i < s.len() && s[i] == b']' {
            self.nodes[idx].end = (i + 1) as u32;
            return Some((i + 1, idx));
        }
        let mark = self.scratch_kids.len();
        loop {
            let Some((vend, child)) = self.parse_value(s, i, depth + 1) else {
                self.scratch_kids.truncate(mark);
                return None;
            };
            self.scratch_kids.push(JKid {
                key_start: 0,
                key_end: 0,
                key_esc: false,
                node: child as u32,
            });
            i = skip_ws(s, vend);
            if i >= s.len() {
                self.scratch_kids.truncate(mark);
                return None;
            }
            if s[i] == b',' {
                i = skip_ws(s, i + 1);
                continue;
            }
            if s[i] == b']' {
                let first = self.kids.len() as u32;
                let n = (self.scratch_kids.len() - mark) as u32;
                self.kids.extend_from_slice(&self.scratch_kids[mark..]);
                self.scratch_kids.truncate(mark);
                self.nodes[idx].end = (i + 1) as u32;
                self.nodes[idx].first = first;
                self.nodes[idx].n = n;
                return Some((i + 1, idx));
            }
            self.scratch_kids.truncate(mark);
            return None;
        }
    }
}

/// Decodes a JSON key (only invoked when `kid.key_esc == true`).
pub fn decode_json_key(raw_inner: &[u8]) -> Option<Vec<u8>> {
    unescape_json_str_exact(raw_inner)
}

#[inline]
pub fn append_path_key(out: &mut Vec<u8>, key: &[u8]) {
    let plain = !key.is_empty()
        && key
            .iter()
            .all(|&c| !matches!(c, b'.' | b'[' | b']' | b'\\' | b'"') && c >= 0x20);
    if plain {
        out.push(b'.');
        out.extend_from_slice(key);
        return;
    }
    out.extend_from_slice(b"[\"");
    for &c in key {
        match c {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            _ => out.push(c),
        }
    }
    out.extend_from_slice(b"\"]");
}

// ============================================================================
// 2. Serializable `JsonTreePlan` Data Structures
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SlotKind {
    None = 0,
    Scalar = 1,
    Object = 2,
    Text = 3,
    Array = 4,
}

impl SlotKind {
    fn from_u8(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::None),
            1 => Ok(Self::Scalar),
            2 => Ok(Self::Object),
            3 => Ok(Self::Text),
            4 => Ok(Self::Array),
            _ => Err(Error::Format(format!("invalid JsonTree SlotKind {b}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonSlotPart {
    pub literal: Vec<u8>,
    pub kind: SlotKind,
    /// Target path index in `JsonTreePlan::paths` (unused when `kind == SlotKind::None`).
    pub path_idx: u16,
    /// Array element separator (e.g. `b","`), only non-empty when `kind == SlotKind::Array`.
    pub sep: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum JsonCand {
    /// Exact copy of `paths[ref_path]` (`sel > 0` emits 0 value bytes!).
    Copy { ref_path: u16 },
    /// Single-ref affix `prefix + unquote(paths[ref_path]) + suffix`.
    Affix {
        prefix: Vec<u8>,
        ref_path: u16,
        suffix: Vec<u8>,
    },
    /// Multi-ref template `lits[0] + unquote(paths[refs[0]]) + lits[1] + ... + lits[k]`. A ref to
    /// the path itself is its free part: the path's next stored cell, as the record has it. With
    /// `b64`, the value is `"` + Base64 of the template (which has no free part) + `"`.
    MultiConcat {
        lits: Vec<Vec<u8>>,
        refs: Vec<u16>,
        b64: bool,
    },
}

impl JsonCand {
    /// Whether the candidate stores a cell of `own`, the path it belongs to.
    pub fn has_free(&self, own: usize) -> bool {
        matches!(self, JsonCand::MultiConcat { refs, .. } if refs.iter().any(|&r| r as usize == own))
    }
}

#[derive(Debug, Clone)]
pub struct JsonPathDef {
    pub name: String,
    pub is_text: bool,
    pub shapes: Vec<Vec<JsonSlotPart>>,
    pub cands: Vec<JsonCand>,
    /// Optional vocabulary/prose group hint (`"text"`, `"login"`, `"hex40"`, etc.).
    pub dict_group: String,
    /// Index into `JsonTreePlan` structural `u32` streams (`u16::MAX` if none).
    pub shape_stream: u16,
    pub count_stream: u16,
    pub sel_stream: u16,
    /// Column index in `ModelPlan::columns` holding emitted literal/fallback values (`u16::MAX` if none).
    pub value_col: u16,
    /// Array element path index in `JsonTreePlan::paths` (`u16::MAX` if not an array path).
    pub elem_path: u16,
}

#[derive(Clone, Default)]
pub struct PathTrieNode {
    pub path_idx: i32,
    pub kids: HashMap<Vec<u8>, usize>,
    pub elem_node: i32,
}

/// Precomputed fast-matching index for `JsonTreePlan` (rebuilt via `rebuild_runtime_index`).
#[derive(Clone, Default)]
pub struct JsonTreeRuntime {
    pub trie: Vec<PathTrieNode>,
    /// Per-path shape key lookup `shape_key -> 1-based shape_id`.
    pub shape_maps: Vec<HashMap<Vec<u8>, u32>>,
}

impl std::fmt::Debug for JsonTreeRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonTreeRuntime")
            .field("trie_nodes", &self.trie.len())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct JsonTreePlan {
    pub paths: Vec<JsonPathDef>,
    pub n_struct_streams: usize,
    pub runtime: JsonTreeRuntime,
}

impl JsonTreePlan {
    pub fn encode(&self, out: &mut Vec<u8>) {
        put_uvarint(out, self.paths.len() as u64);
        put_uvarint(out, self.n_struct_streams as u64);
        for p in &self.paths {
            put_bytes(out, p.name.as_bytes());
            out.push(u8::from(p.is_text));
            put_bytes(out, p.dict_group.as_bytes());
            put_uvarint(out, p.shape_stream as u64);
            put_uvarint(out, p.count_stream as u64);
            put_uvarint(out, p.sel_stream as u64);
            put_uvarint(out, p.value_col as u64);
            put_uvarint(out, p.elem_path as u64);

            put_uvarint(out, p.shapes.len() as u64);
            for sh in &p.shapes {
                put_uvarint(out, sh.len() as u64);
                for part in sh {
                    put_bytes(out, &part.literal);
                    out.push(part.kind as u8);
                    if part.kind != SlotKind::None {
                        put_uvarint(out, part.path_idx as u64);
                        if part.kind == SlotKind::Array {
                            put_bytes(out, &part.sep);
                        }
                    }
                }
            }

            put_uvarint(out, p.cands.len() as u64);
            for c in &p.cands {
                match c {
                    JsonCand::Copy { ref_path } => {
                        out.push(0);
                        put_uvarint(out, *ref_path as u64);
                    }
                    JsonCand::Affix {
                        prefix,
                        ref_path,
                        suffix,
                    } => {
                        out.push(1);
                        put_bytes(out, prefix);
                        put_uvarint(out, *ref_path as u64);
                        put_bytes(out, suffix);
                    }
                    JsonCand::MultiConcat { lits, refs, b64 } => {
                        out.push(if *b64 { 3 } else { 2 });
                        put_uvarint(out, refs.len() as u64);
                        for l in lits {
                            put_bytes(out, l);
                        }
                        for &r in refs {
                            put_uvarint(out, r as u64);
                        }
                    }
                }
            }
        }
    }

    pub fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        let np = cur.uvar()? as usize;
        if np == 0 || np > 2048 {
            return Err(Error::Format(format!(
                "invalid JsonTreePlan path count {np}"
            )));
        }
        let n_struct_streams = cur.uvar()? as usize;
        if n_struct_streams > 8192 {
            return Err(Error::Format(
                "invalid JsonTreePlan struct stream count".into(),
            ));
        }
        let mut paths = Vec::with_capacity(np);
        for _ in 0..np {
            let name = cur.str_utf8()?.to_string();
            let is_text = cur.u8()? != 0;
            let dict_group = cur.str_utf8()?.to_string();
            let shape_stream = cur.uvar()? as u16;
            let count_stream = cur.uvar()? as u16;
            let sel_stream = cur.uvar()? as u16;
            let value_col = cur.uvar()? as u16;
            let elem_path = cur.uvar()? as u16;

            let n_shapes = cur.uvar()? as usize;
            if n_shapes > 256 {
                return Err(Error::Format("too many shapes in JsonPathDef".into()));
            }
            let mut shapes = Vec::with_capacity(n_shapes);
            for _ in 0..n_shapes {
                let n_parts = cur.uvar()? as usize;
                if n_parts == 0 || n_parts > 256 {
                    return Err(Error::Format("invalid shape part count".into()));
                }
                let mut sh = Vec::with_capacity(n_parts);
                for _ in 0..n_parts {
                    let literal = cur.bytes()?.to_vec();
                    let kind = SlotKind::from_u8(cur.u8()?)?;
                    let mut path_idx = 0u16;
                    let mut sep = Vec::new();
                    if kind != SlotKind::None {
                        path_idx = cur.uvar()? as u16;
                        if (path_idx as usize) >= np {
                            return Err(Error::Format(
                                "JsonSlotPart path_idx out of bounds".into(),
                            ));
                        }
                        if kind == SlotKind::Array {
                            sep = cur.bytes()?.to_vec();
                        }
                    }
                    sh.push(JsonSlotPart {
                        literal,
                        kind,
                        path_idx,
                        sep,
                    });
                }
                shapes.push(sh);
            }

            let n_cands = cur.uvar()? as usize;
            if n_cands > 16 {
                return Err(Error::Format("too many cands in JsonPathDef".into()));
            }
            let mut cands = Vec::with_capacity(n_cands);
            for _ in 0..n_cands {
                match cur.u8()? {
                    0 => {
                        let ref_path = cur.uvar()? as u16;
                        if (ref_path as usize) >= np {
                            return Err(Error::Format("JsonCand ref_path out of bounds".into()));
                        }
                        cands.push(JsonCand::Copy { ref_path });
                    }
                    1 => {
                        let prefix = cur.bytes()?.to_vec();
                        let ref_path = cur.uvar()? as u16;
                        let suffix = cur.bytes()?.to_vec();
                        if (ref_path as usize) >= np {
                            return Err(Error::Format("JsonCand ref_path out of bounds".into()));
                        }
                        cands.push(JsonCand::Affix {
                            prefix,
                            ref_path,
                            suffix,
                        });
                    }
                    tag @ (2 | 3) => {
                        let k = cur.uvar()? as usize;
                        if k == 0 || k > 8 {
                            return Err(Error::Format("invalid MultiConcat arity".into()));
                        }
                        let mut lits = Vec::with_capacity(k + 1);
                        for _ in 0..=k {
                            lits.push(cur.bytes()?.to_vec());
                        }
                        let mut refs = Vec::with_capacity(k);
                        for _ in 0..k {
                            let r = cur.uvar()?;
                            if r >= np as u64 {
                                return Err(Error::Format("JsonCand ref out of bounds".into()));
                            }
                            refs.push(r as u16);
                        }
                        let b64 = tag == 3;
                        let own = paths.len();
                        let free = refs.iter().filter(|&&r| r as usize == own).count();
                        if b64 && !lits.iter().all(|l| b64_text_ok(l)) {
                            return Err(Error::Format("Base64 candidate text not printable".into()));
                        }
                        if free > usize::from(!b64) {
                            return Err(Error::Format("invalid MultiConcat free part".into()));
                        }
                        cands.push(JsonCand::MultiConcat { lits, refs, b64 });
                    }
                    t => return Err(Error::Format(format!("unknown JsonCand tag {t}"))),
                }
            }

            paths.push(JsonPathDef {
                name,
                is_text,
                shapes,
                cands,
                dict_group,
                shape_stream,
                count_stream,
                sel_stream,
                value_col,
                elem_path,
            });
        }
        let mut plan = Self {
            paths,
            n_struct_streams,
            runtime: JsonTreeRuntime::default(),
        };
        plan.rebuild_runtime_index();
        Ok(plan)
    }

    pub fn rebuild_runtime_index(&mut self) {
        let names: Vec<&[u8]> = self.paths.iter().map(|p| p.name.as_bytes()).collect();
        let trie = build_path_trie(&names);
        let mut shape_maps = Vec::with_capacity(self.paths.len());
        for p in &self.paths {
            let mut m = HashMap::with_capacity(p.shapes.len());
            for (si, sh) in p.shapes.iter().enumerate() {
                m.insert(encode_shape_match_key(sh), (si + 1) as u32);
            }
            shape_maps.push(m);
        }
        self.runtime = JsonTreeRuntime { trie, shape_maps };
    }
}

// ============================================================================
// 3. Path Trie & Canonical Shape Key Encoding
// ============================================================================

fn parse_path_step(p: &[u8], i: usize) -> Option<(Option<Vec<u8>>, usize)> {
    match *p.get(i)? {
        b'.' => {
            let start = i + 1;
            let mut j = start;
            while j < p.len() && p[j] != b'.' && p[j] != b'[' {
                j += 1;
            }
            (j > start).then(|| (Some(p[start..j].to_vec()), j))
        }
        b'[' if p.get(i + 1) == Some(&b']') => Some((None, i + 2)),
        b'[' if p.get(i + 1) == Some(&b'"') => {
            let mut key = Vec::new();
            let mut j = i + 2;
            while j < p.len() {
                if p[j] == b'\\' && j + 1 < p.len() {
                    key.push(p[j + 1]);
                    j += 2;
                } else if p[j] == b'"' && p.get(j + 1) == Some(&b']') {
                    return Some((Some(key), j + 2));
                } else {
                    key.push(p[j]);
                    j += 1;
                }
            }
            None
        }
        _ => None,
    }
}

fn build_path_trie(paths: &[&[u8]]) -> Vec<PathTrieNode> {
    let mut nodes = vec![PathTrieNode {
        path_idx: -1,
        kids: HashMap::new(),
        elem_node: -1,
    }];
    for (idx, &p) in paths.iter().enumerate() {
        if p.first() != Some(&b'$') {
            continue;
        }
        let mut cur = 0usize;
        let mut i = 1usize;
        let mut ok = true;
        while i < p.len() {
            let Some((step, next_i)) = parse_path_step(p, i) else {
                ok = false;
                break;
            };
            i = next_i;
            cur = match step {
                Some(key) => {
                    if let Some(&existing) = nodes[cur].kids.get(&key) {
                        existing
                    } else {
                        let next = nodes.len();
                        nodes.push(PathTrieNode {
                            path_idx: -1,
                            kids: HashMap::new(),
                            elem_node: -1,
                        });
                        nodes[cur].kids.insert(key, next);
                        next
                    }
                }
                None => {
                    if nodes[cur].elem_node >= 0 {
                        nodes[cur].elem_node as usize
                    } else {
                        let next = nodes.len();
                        nodes.push(PathTrieNode {
                            path_idx: -1,
                            kids: HashMap::new(),
                            elem_node: -1,
                        });
                        nodes[cur].elem_node = next as i32;
                        next
                    }
                }
            };
        }
        if ok {
            nodes[cur].path_idx = idx as i32;
        }
    }
    nodes
}

fn encode_shape_match_key(parts: &[JsonSlotPart]) -> Vec<u8> {
    let mut key = Vec::with_capacity(parts.len() * 12);
    for p in parts {
        put_bytes(&mut key, &p.literal);
        key.push(p.kind as u8);
        if p.kind != SlotKind::None {
            put_uvarint(&mut key, p.path_idx as u64);
            if p.kind == SlotKind::Array {
                put_bytes(&mut key, &p.sep);
            }
        }
    }
    key
}

// ============================================================================
// 4. Row Splitting (Encoder) & Exact Reconstruction (Decoder)
// ============================================================================

#[derive(Clone, Copy)]
struct ChildSlot {
    path_idx: u16,
    kind: SlotKind,
    node: u32,
    trie_node: u32,
}

#[inline]
fn unquote_json_lexeme(b: &[u8]) -> &[u8] {
    if b.len() >= 2 && b[0] == b'"' && b[b.len() - 1] == b'"' {
        &b[1..b.len() - 1]
    } else {
        b
    }
}

/// Whether `target`, a value of path `own`, is what `cand` builds from the values `val` gives
/// for its refs: `None` if not, else the span of the free part in `target`, if it has one.
/// `tmp` holds the bytes under a Base64 value.
#[inline]
fn cand_fit<'v>(
    cand: &JsonCand,
    own: usize,
    target: &[u8],
    val: impl Fn(u16) -> Option<&'v [u8]>,
    tmp: &mut Vec<u8>,
) -> Option<Option<(usize, usize)>> {
    match cand {
        JsonCand::Copy { ref_path } => (val(*ref_path)? == target).then_some(None),
        JsonCand::Affix {
            prefix,
            ref_path,
            suffix,
        } => {
            let rv = unquote_json_lexeme(val(*ref_path)?);
            (target.len() == prefix.len() + rv.len() + suffix.len()
                && target.starts_with(prefix)
                && target[prefix.len()..].starts_with(rv)
                && target.ends_with(suffix))
            .then_some(None)
        }
        JsonCand::MultiConcat { lits, refs, b64 } => {
            if !b64 {
                return concat_fit(lits, refs, own, target, val);
            }
            let inner = target.strip_prefix(b"\"")?.strip_suffix(b"\"")?;
            tmp.clear();
            crate::lex::parse_base64(inner, tmp)?;
            if !b64_text_ok(tmp) {
                return None;
            }
            concat_fit(lits, refs, usize::MAX, tmp, val).map(|_| None)
        }
    }
}

/// Whether `text` may be spelled in Base64 by a candidate. Holding it to printable ASCII lets a
/// search rule out most needles for every Base64 value at once.
pub(crate) fn b64_text_ok(text: &[u8]) -> bool {
    text.iter().all(|c| (0x20..0x7f).contains(c))
}

/// `cand_fit` for a plain template. Every part but the free one has a known length, so the free
/// part is what is left.
#[inline]
fn concat_fit<'v>(
    lits: &[Vec<u8>],
    refs: &[u16],
    own: usize,
    t: &[u8],
    val: impl Fn(u16) -> Option<&'v [u8]>,
) -> Option<Option<(usize, usize)>> {
    let mut fixed = lits.iter().map(|l| l.len()).sum::<usize>();
    let mut has_free = false;
    for &r in refs {
        if r as usize == own {
            has_free = true;
        } else {
            fixed += unquote_json_lexeme(val(r)?).len();
        }
    }
    if t.len() < fixed || (!has_free && t.len() != fixed) {
        return None;
    }
    let mut free = None;
    let mut pos = 0;
    let take = |piece: &[u8], pos: &mut usize| {
        let ok = t[*pos..].starts_with(piece);
        *pos += piece.len();
        ok
    };
    if !take(&lits[0], &mut pos) {
        return None;
    }
    for (i, &r) in refs.iter().enumerate() {
        if r as usize == own {
            let end = pos + t.len() - fixed;
            free = Some((pos, end));
            pos = end;
        } else if !take(unquote_json_lexeme(val(r)?), &mut pos) {
            return None;
        }
        if !take(&lits[i + 1], &mut pos) {
            return None;
        }
    }
    Some(free)
}

/// Reusable encoder state for `JsonTreePlan` within a block.
pub struct JsonTreeBlockSplitter {
    pub doc: JsonDoc,
    pub struct_streams: Vec<Vec<u32>>,
    key_buf: Vec<u8>,
    child_slots: Vec<ChildSlot>,
    scope: Vec<Option<(usize, usize)>>,
    /// Bytes of objects whose key layout no kept shape matched, stored whole.
    pub unshaped_bytes: usize,
    tmp: Vec<u8>,
}

impl JsonTreeBlockSplitter {
    pub fn new(jt: &JsonTreePlan) -> Self {
        Self {
            doc: JsonDoc::default(),
            struct_streams: vec![Vec::new(); jt.n_struct_streams],
            key_buf: Vec::with_capacity(256),
            child_slots: Vec::with_capacity(64),
            scope: vec![None; jt.paths.len()],
            unshaped_bytes: 0,
            tmp: Vec::new(),
        }
    }

    /// Attempts to split `line` using `jt`, pushing `u32` structural tokens to `self.struct_streams`
    /// and pushing value occurrences into `col_occs[cid]` (with `scratch` for nested-plan unescaping).
    /// Returns `false` (mutating nothing!) if `line` is not a valid root JSON object.
    pub fn split_row<'a>(
        &mut self,
        line: &'a [u8],
        jt: &JsonTreePlan,
        has_nested: &[bool],
        col_occs: &mut [Vec<crate::grammar::OccRef<'a>>],
        scratch: &mut crate::grammar::RowScratch,
    ) -> bool {
        if !self.doc.parse_root_object(line) {
            return false;
        }
        self.scope.fill(None);
        self.emit_node(0, 0, 0, line, jt, has_nested, col_occs, scratch);
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_node<'a>(
        &mut self,
        ni: usize,
        path_idx: usize,
        trie_node: usize,
        line: &'a [u8],
        jt: &JsonTreePlan,
        has_nested: &[bool],
        col_occs: &mut [Vec<crate::grammar::OccRef<'a>>],
        scratch: &mut crate::grammar::RowScratch,
    ) {
        let pdef = &jt.paths[path_idx];
        let v = self.doc.nodes[ni];

        // Leaf / Text path (`shapes.is_empty()`): check derivations, else emit scalar/text slice!
        if pdef.shapes.is_empty() {
            let val_slice = &line[v.start as usize..v.end as usize];
            let mut sel = 0u32;
            let mut stored = Some(val_slice);
            if !pdef.cands.is_empty() {
                let scope = &self.scope;
                let val = |r: u16| scope[r as usize].map(|(s, e)| &line[s..e]);
                for (ci, cand) in pdef.cands.iter().enumerate() {
                    if let Some(free) = cand_fit(cand, path_idx, val_slice, val, &mut self.tmp) {
                        sel = (ci + 1) as u32;
                        stored = free.map(|(s, e)| &val_slice[s..e]);
                        break;
                    }
                }
                self.struct_streams[pdef.sel_stream as usize].push(sel);
            }
            if let Some(cell) = stored
                && pdef.value_col != u16::MAX
            {
                push_json_val_occ(pdef.value_col as usize, cell, has_nested, col_occs, scratch);
            }
            self.scope[path_idx] = Some((v.start as usize, v.end as usize));
            return;
        }

        // Tabled path (`!pdef.shapes.is_empty()`): build canonical shape key + child list
        let mark = self.child_slots.len();
        let shape_id = if self.build_shape_key(ni, trie_node, line, jt) {
            jt.runtime.shape_maps[path_idx]
                .get(&self.key_buf)
                .copied()
                .unwrap_or(0)
        } else {
            0
        };
        self.struct_streams[pdef.shape_stream as usize].push(shape_id);

        if shape_id == 0 {
            self.child_slots.truncate(mark);
            if v.kind == J_OBJECT {
                self.unshaped_bytes += (v.end - v.start) as usize;
            }
            if pdef.value_col != u16::MAX {
                let raw_sub = &line[v.start as usize..v.end as usize];
                push_json_val_occ(
                    pdef.value_col as usize,
                    raw_sub,
                    has_nested,
                    col_occs,
                    scratch,
                );
            }
            return;
        }

        // Recurse into matched shape's child slots in exact left-to-right appearance order!
        let end = self.child_slots.len();
        for idx in mark..end {
            let cs = self.child_slots[idx];
            let c_path = cs.path_idx as usize;
            let c_trie = cs.trie_node as usize;
            match cs.kind {
                SlotKind::Scalar | SlotKind::Text | SlotKind::Object => {
                    self.emit_node(
                        cs.node as usize,
                        c_path,
                        c_trie,
                        line,
                        jt,
                        has_nested,
                        col_occs,
                        scratch,
                    );
                }
                SlotKind::Array => {
                    let av = self.doc.nodes[cs.node as usize];
                    self.struct_streams[jt.paths[c_path].count_stream as usize].push(av.n);
                    let elem_path = jt.paths[c_path].elem_path as usize;
                    let elem_trie = jt.runtime.trie[c_trie].elem_node.max(0) as usize;
                    for ek in self.doc.kid_range(cs.node as usize) {
                        let elem_ni = self.doc.kids[ek].node as usize;
                        self.emit_node(
                            elem_ni, elem_path, elem_trie, line, jt, has_nested, col_occs, scratch,
                        );
                    }
                }
                SlotKind::None => {}
            }
        }
        self.child_slots.truncate(mark);
    }

    fn build_shape_key(
        &mut self,
        ni: usize,
        trie_node: usize,
        s: &[u8],
        jt: &JsonTreePlan,
    ) -> bool {
        self.key_buf.clear();
        let v = self.doc.nodes[ni];
        if v.kind != J_OBJECT {
            // Tabled non-object (e.g. scalar array element or mixed-type node): classify `v` itself
            if v.kind == J_ARRAY {
                return false;
            }
            return false;
        }

        let mut pos = v.start as usize;
        for k in self.doc.kid_range(ni) {
            let kid = self.doc.kids[k];
            let raw_k = &s[kid.key_start as usize..kid.key_end as usize];
            let child_trie = if kid.key_esc {
                let Some(dec) = decode_json_key(raw_k) else {
                    return false;
                };
                jt.runtime.trie[trie_node].kids.get(&dec).copied()
            } else {
                jt.runtime.trie[trie_node].kids.get(raw_k).copied()
            };
            let Some(ct) = child_trie else {
                continue; // Unplanned/constant-inline member stays inside literal!
            };
            let cp_i32 = jt.runtime.trie[ct].path_idx;
            let cv = self.doc.nodes[kid.node as usize];

            // Determine slot kind for this child member
            let (kind, target_path, sep_slice): (SlotKind, usize, &[u8]) = if cp_i32 >= 0 {
                let cp = cp_i32 as usize;
                let cp_def = &jt.paths[cp];
                if cp_def.is_text {
                    (SlotKind::Text, cp, &[])
                } else if !cp_def.shapes.is_empty() {
                    // A `null` here is the child's to store, so it does not double the parent's shapes.
                    (SlotKind::Object, cp, &[])
                } else if cp_def.elem_path != u16::MAX {
                    if cv.kind != J_ARRAY || (cv.n == 0 && cv.end - cv.start != 2) {
                        continue; // `null` or a spaced `[ ]` stays inline in parent shape literal!
                    }
                    let Some(sep) = uniform_array_sep(&self.doc, s, kid.node as usize) else {
                        return false;
                    };
                    (SlotKind::Array, cp, sep)
                } else if cv.kind >= J_STRING {
                    (SlotKind::Scalar, cp, &[])
                } else {
                    return false; // Unexpected container at scalar leaf -> local `shape_id = 0` fallback!
                }
            } else {
                continue;
            };

            if kind == SlotKind::Array {
                let (elems_start, elems_end) = array_elems_span(&self.doc, &cv);
                put_bytes(&mut self.key_buf, &s[pos..elems_start]);
                self.key_buf.push(SlotKind::Array as u8);
                put_uvarint(&mut self.key_buf, target_path as u64);
                put_bytes(&mut self.key_buf, sep_slice);
                self.child_slots.push(ChildSlot {
                    path_idx: target_path as u16,
                    kind: SlotKind::Array,
                    node: kid.node,
                    trie_node: ct as u32,
                });
                pos = elems_end;
            } else {
                put_bytes(&mut self.key_buf, &s[pos..cv.start as usize]);
                self.key_buf.push(kind as u8);
                put_uvarint(&mut self.key_buf, target_path as u64);
                self.child_slots.push(ChildSlot {
                    path_idx: target_path as u16,
                    kind,
                    node: kid.node,
                    trie_node: ct as u32,
                });
                pos = cv.end as usize;
            }
        }

        put_bytes(&mut self.key_buf, &s[pos..v.end as usize]);
        self.key_buf.push(SlotKind::None as u8);
        true
    }
}

/// Where an array's elements start and end. An empty `[]` is an array of no elements, the same
/// shape as a full one.
#[inline]
fn array_elems_span(doc: &JsonDoc, av: &JVal) -> (usize, usize) {
    if av.n == 0 {
        return (av.start as usize + 1, av.end as usize - 1);
    }
    let first = doc.nodes[doc.kids[av.first as usize].node as usize];
    let last = doc.nodes[doc.kids[(av.first + av.n - 1) as usize].node as usize];
    (first.start as usize, last.end as usize)
}

#[inline]
fn uniform_array_sep<'a>(doc: &JsonDoc, s: &'a [u8], arr_ni: usize) -> Option<&'a [u8]> {
    let av = doc.nodes[arr_ni];
    if av.n <= 1 {
        return Some(b",");
    }
    let e0 = doc.nodes[doc.kids[av.first as usize].node as usize];
    let e1 = doc.nodes[doc.kids[(av.first + 1) as usize].node as usize];
    let sep = &s[e0.end as usize..e1.start as usize];
    for idx in 1..(av.n as usize - 1) {
        let a = doc.nodes[doc.kids[av.first as usize + idx].node as usize];
        let b = doc.nodes[doc.kids[av.first as usize + idx + 1].node as usize];
        if &s[a.end as usize..b.start as usize] != sep {
            return None;
        }
    }
    Some(sep)
}

#[inline]
fn push_json_val_occ<'a>(
    cid: usize,
    raw_lexeme: &'a [u8],
    has_nested: &[bool],
    col_occs: &mut [Vec<crate::grammar::OccRef<'a>>],
    scratch: &mut crate::grammar::RowScratch,
) {
    if has_nested[cid] {
        if raw_lexeme.len() >= 2
            && raw_lexeme[0] == b'"'
            && raw_lexeme[raw_lexeme.len() - 1] == b'"'
            && let Some(mut unesc) = unescape_json_str_exact(&raw_lexeme[1..raw_lexeme.len() - 1])
        {
            let mut tagged = Vec::with_capacity(1 + unesc.len());
            tagged.push(1u8);
            tagged.append(&mut unesc);
            let idx = scratch.alloc(tagged);
            col_occs[cid].push(crate::grammar::OccRef::Scratch(idx));
            return;
        }
        let mut tagged = Vec::with_capacity(1 + raw_lexeme.len());
        tagged.push(0u8);
        tagged.extend_from_slice(raw_lexeme);
        let idx = scratch.alloc(tagged);
        col_occs[cid].push(crate::grammar::OccRef::Scratch(idx));
    } else {
        col_occs[cid].push(crate::grammar::OccRef::Borrowed(raw_lexeme));
    }
}

/// Appends the streams to `out`; returns where the first starts there and the length of each.
pub fn encode_json_struct_streams(
    streams: &[Vec<u32>],
    backend: EntropyBackend,
    level: i32,
    out: &mut Vec<u8>,
) -> (usize, Vec<u32>) {
    put_uvarint(out, streams.len() as u64);
    let head = out.len();
    let mut lens = Vec::with_capacity(streams.len());
    for s in streams {
        let at = out.len();
        encode_u32_seq(s, backend, level, out);
        lens.push((out.len() - at) as u32);
    }
    (head, lens)
}

pub fn decode_json_struct_streams(
    cur: &mut Cursor<'_>,
    expected: usize,
    budget: &DecodeBudget,
) -> Result<Vec<Vec<u32>>> {
    let n = cur.uvar()? as usize;
    if n != expected {
        return Err(Error::Format(format!(
            "json struct stream count {n} != expected {expected}"
        )));
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(decode_u32_seq(cur, budget)?);
    }
    Ok(out)
}

/// Stateful streaming decoder for `JsonTreePlan` rows within a block!
pub struct JsonTreeBlockReconstructor {
    pub struct_streams: Vec<Vec<u32>>,
    pub struct_pos: Vec<usize>,
    /// By path: where its value stands in the output, valid on the row numbered `row`.
    scope_ranges: Vec<(u32, usize, usize)>,
    row: u32,
    tmp: Vec<u8>,
    /// `skip_order` of the tree, once a skip needs it.
    order: Option<Option<Vec<u16>>>,
    /// By path: value visits and array visits, for `skip_rows`.
    visits: (Vec<usize>, Vec<usize>),
}

impl JsonTreeBlockReconstructor {
    pub fn new(struct_streams: Vec<Vec<u32>>, n_paths: usize) -> Self {
        let ns = struct_streams.len();
        Self {
            struct_streams,
            struct_pos: vec![0usize; ns],
            scope_ranges: vec![(0, 0, 0); n_paths],
            row: 0,
            tmp: Vec::new(),
            order: None,
            visits: Default::default(),
        }
    }

    #[inline(always)]
    fn next_u32(&mut self, stream_idx: u16) -> Result<u32> {
        let si = stream_idx as usize;
        if let (Some(pos), Some(s)) = (self.struct_pos.get_mut(si), self.struct_streams.get(si))
            && let Some(&v) = s.get(*pos)
        {
            *pos += 1;
            return Ok(v);
        }
        Err(stream_underflow())
    }

    pub fn reconstruct_row(
        &mut self,
        jt: &JsonTreePlan,
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        col_pos: &mut [usize],
        out: &mut Vec<u8>,
        budget: &DecodeBudget,
    ) -> Result<()> {
        self.row = self.row.wrapping_add(1);
        if self.row == 0 {
            self.scope_ranges.fill((0, 0, 0));
            self.row = 1;
        }
        let start_len = out.len();
        self.emit_child(0, jt, plan, ctx, col_pos, out, budget, 0)?;
        if out.len() - start_len > budget.max_block_decoded_bytes {
            return Err(Error::BudgetExceeded("json reconstructed row size"));
        }
        Ok(())
    }

    /// Where in `out` the value of `r` on this row stands.
    #[inline]
    fn scope_ref(&self, r: u16) -> Result<(usize, usize)> {
        match self.scope_ranges[r as usize] {
            (row, rs, re) if row == self.row => Ok((rs, re)),
            _ => Err(Error::Format("JsonTree candidate missing scope ref".into())),
        }
    }

    /// Where in `out` the value of `r` on this row stands, without its quotes.
    #[inline]
    fn unquoted_ref(&self, r: u16, out: &[u8]) -> Result<(usize, usize)> {
        let (rs, re) = self.scope_ref(r)?;
        if re >= rs + 2 && out[rs] == b'"' && out[re - 1] == b'"' {
            return Ok((rs + 1, re - 1));
        }
        Ok((rs, re))
    }

    /// Emits path `p`: a leaf in place, which is most paths, any other through `emit_path`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn emit_child(
        &mut self,
        p: usize,
        jt: &JsonTreePlan,
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        col_pos: &mut [usize],
        out: &mut Vec<u8>,
        budget: &DecodeBudget,
        depth: usize,
    ) -> Result<()> {
        let pdef = &jt.paths[p];
        if !pdef.shapes.is_empty() {
            return self.emit_path(p, jt, plan, ctx, col_pos, out, budget, depth);
        }
        let start = out.len();
        let sel = match pdef.cands.is_empty() {
            true => 0,
            false => self.next_u32(pdef.sel_stream)? as usize,
        };
        if sel == 0 {
            emit_json_stored_cell(pdef.value_col as usize, plan, ctx, col_pos, out, budget)?;
        } else {
            self.emit_cand(p, sel, jt, plan, ctx, col_pos, out, budget)?;
        }
        self.scope_ranges[p] = (self.row, start, out.len());
        Ok(())
    }

    /// Emits leaf `path_idx` from its candidate `sel`.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn emit_cand(
        &mut self,
        path_idx: usize,
        sel: usize,
        jt: &JsonTreePlan,
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        col_pos: &mut [usize],
        out: &mut Vec<u8>,
        budget: &DecodeBudget,
    ) -> Result<()> {
        let pdef = &jt.paths[path_idx];
        let cand = pdef
            .cands
            .get(sel - 1)
            .ok_or_else(|| Error::Format("JsonTree sel out of range".into()))?;
        match cand {
            JsonCand::Copy { ref_path } => {
                let (rs, re) = self.scope_ref(*ref_path)?;
                out.extend_from_within(rs..re);
            }
            JsonCand::Affix {
                prefix,
                ref_path,
                suffix,
            } => {
                let (rs, re) = self.unquoted_ref(*ref_path, out)?;
                out.extend_from_slice(prefix);
                out.extend_from_within(rs..re);
                out.extend_from_slice(suffix);
            }
            JsonCand::MultiConcat {
                lits,
                refs,
                b64: false,
            } => {
                out.extend_from_slice(&lits[0]);
                for (i, &r) in refs.iter().enumerate() {
                    if r as usize == path_idx {
                        let cid = pdef.value_col as usize;
                        emit_json_stored_cell(cid, plan, ctx, col_pos, out, budget)?;
                    } else {
                        let (rs, re) = self.unquoted_ref(r, out)?;
                        out.extend_from_within(rs..re);
                    }
                    out.extend_from_slice(&lits[i + 1]);
                }
            }
            JsonCand::MultiConcat { lits, refs, .. } => {
                let mut text = std::mem::take(&mut self.tmp);
                text.clear();
                text.extend_from_slice(&lits[0]);
                for (i, &r) in refs.iter().enumerate() {
                    let (rs, re) = self.unquoted_ref(r, out)?;
                    text.extend_from_slice(&out[rs..re]);
                    text.extend_from_slice(&lits[i + 1]);
                }
                if !b64_text_ok(&text) {
                    return Err(Error::Format("Base64 candidate text not printable".into()));
                }
                out.push(b'"');
                crate::lex::format_base64(&text, out);
                out.push(b'"');
                self.tmp = text;
            }
        }
        Ok(())
    }

    /// Emits path `path_idx`, which has shapes.
    #[allow(clippy::too_many_arguments)]
    fn emit_path(
        &mut self,
        path_idx: usize,
        jt: &JsonTreePlan,
        plan: &ModelPlan,
        ctx: &mut BlockDecodeCtx<'_>,
        col_pos: &mut [usize],
        out: &mut Vec<u8>,
        budget: &DecodeBudget,
        depth: usize,
    ) -> Result<()> {
        if depth > MAX_PARSE_DEPTH + 4 {
            return Err(Error::Format("JsonTree decode recursion limit".into()));
        }
        let pdef = &jt.paths[path_idx];
        let shape_id = self.next_u32(pdef.shape_stream)? as usize;
        if shape_id == 0 {
            emit_json_stored_cell(pdef.value_col as usize, plan, ctx, col_pos, out, budget)?;
            return Ok(());
        }
        let shape = pdef
            .shapes
            .get(shape_id - 1)
            .ok_or_else(|| Error::Format("JsonTree shape_id out of bounds".into()))?;

        for part in shape {
            out.extend_from_slice(&part.literal);
            match part.kind {
                SlotKind::None => {}
                SlotKind::Scalar | SlotKind::Text | SlotKind::Object => {
                    let p = part.path_idx as usize;
                    self.emit_child(p, jt, plan, ctx, col_pos, out, budget, depth + 1)?;
                }
                SlotKind::Array => {
                    let arr_p = part.path_idx as usize;
                    let count = self.next_u32(jt.paths[arr_p].count_stream)? as usize;
                    if count > budget.max_records_per_block {
                        return Err(Error::BudgetExceeded("JsonTree array count"));
                    }
                    let elem_p = jt.paths[arr_p].elem_path as usize;
                    for ei in 0..count {
                        if ei > 0 {
                            out.extend_from_slice(&part.sep);
                        }
                        self.emit_child(elem_p, jt, plan, ctx, col_pos, out, budget, depth + 1)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Advances `struct_pos` and `col_pos` across `n` skipped JSON rows, reading no column.
    ///
    /// Paths go parent first, each taking its visits' worth of its own stream at once: a shape
    /// stream is counted by shape and each shape's parts handed their share, a count stream is
    /// summed into its elements' visits. Row by row, every part of every shape is touched.
    pub fn skip_rows(
        &mut self,
        n: usize,
        jt: &JsonTreePlan,
        col_pos: &mut [usize],
        budget: &DecodeBudget,
    ) -> Result<()> {
        let order = self.order.take().unwrap_or_else(|| skip_order(jt));
        let res = match &order {
            Some(o) => self.skip_counted(n, o, jt, col_pos, budget),
            None => (0..n).try_for_each(|_| self.skip_path(0, jt, col_pos, budget, 0)),
        };
        self.order = Some(order);
        res
    }

    fn skip_counted(
        &mut self,
        n: usize,
        order: &[u16],
        jt: &JsonTreePlan,
        col_pos: &mut [usize],
        budget: &DecodeBudget,
    ) -> Result<()> {
        let np = jt.paths.len();
        // By path: times its value is visited, and times it is visited as an array.
        let (visits, arrays) = &mut self.visits;
        visits.clear();
        visits.resize(np, 0);
        arrays.clear();
        arrays.resize(np, 0);
        visits[0] = n;
        let mut hist = [0usize; 257];
        for &p in order {
            let p = p as usize;
            let (v, a) = (visits[p], arrays[p]);
            let pdef = &jt.paths[p];
            let mut stored = 0;
            if v > 0 && pdef.shapes.is_empty() {
                stored = v;
                if !pdef.cands.is_empty() {
                    let sels = take(&self.struct_streams, &mut self.struct_pos, pdef.sel_stream, v)?;
                    for &s in sels {
                        let s = s as usize;
                        if s > pdef.cands.len() {
                            return Err(Error::Format("JsonTree sel out of range".into()));
                        }
                        hist[s] += 1;
                    }
                    stored = hist[0];
                    for (c, h) in pdef.cands.iter().zip(&mut hist[1..]) {
                        stored += if c.has_free(p) { *h } else { 0 };
                        *h = 0;
                    }
                    hist[0] = 0;
                }
            } else if v > 0 {
                let ids = take(&self.struct_streams, &mut self.struct_pos, pdef.shape_stream, v)?;
                for &s in ids {
                    let s = s as usize;
                    if s > pdef.shapes.len() {
                        return Err(Error::Format("JsonTree shape_id out of bounds".into()));
                    }
                    hist[s] += 1;
                }
                stored = std::mem::take(&mut hist[0]);
                for (shape, h) in pdef.shapes.iter().zip(&mut hist[1..]) {
                    let h = std::mem::take(h);
                    if h == 0 {
                        continue;
                    }
                    for part in shape {
                        let kid = part.path_idx as usize;
                        match part.kind {
                            SlotKind::None => {}
                            SlotKind::Array => arrays[kid] += h,
                            _ => visits[kid] += h,
                        }
                    }
                }
            }
            if stored > 0
                && let Some(c) = col_pos.get_mut(pdef.value_col as usize)
            {
                *c += stored;
            }
            if a > 0 {
                let counts = take(&self.struct_streams, &mut self.struct_pos, pdef.count_stream, a)?;
                let mut elems = 0usize;
                for &c in counts {
                    if c as usize > budget.max_records_per_block {
                        return Err(Error::BudgetExceeded("JsonTree array count"));
                    }
                    elems += c as usize;
                }
                if elems > 0 {
                    let e = visits
                        .get_mut(pdef.elem_path as usize)
                        .ok_or_else(|| Error::Format("JsonTree array without elements".into()))?;
                    *e += elems;
                }
            }
        }
        Ok(())
    }

    fn skip_path(
        &mut self,
        path_idx: usize,
        jt: &JsonTreePlan,
        col_pos: &mut [usize],
        budget: &DecodeBudget,
        depth: usize,
    ) -> Result<()> {
        if depth > MAX_PARSE_DEPTH + 4 {
            return Err(Error::Format("JsonTree skip recursion limit".into()));
        }
        let pdef = &jt.paths[path_idx];
        if pdef.shapes.is_empty() {
            let sel = if !pdef.cands.is_empty() {
                self.next_u32(pdef.sel_stream)? as usize
            } else {
                0
            };
            let stores = match sel {
                0 => true,
                _ => pdef
                    .cands
                    .get(sel - 1)
                    .ok_or_else(|| Error::Format("JsonTree sel out of range".into()))?
                    .has_free(path_idx),
            };
            if stores && let Some(p) = col_pos.get_mut(pdef.value_col as usize) {
                *p += 1;
            }
            return Ok(());
        }
        let shape_id = self.next_u32(pdef.shape_stream)? as usize;
        if shape_id == 0 {
            if pdef.value_col != u16::MAX {
                col_pos[pdef.value_col as usize] += 1;
            }
            return Ok(());
        }
        let shape = pdef
            .shapes
            .get(shape_id - 1)
            .ok_or_else(|| Error::Format("JsonTree shape_id out of bounds".into()))?;
        for part in shape {
            match part.kind {
                SlotKind::None => {}
                SlotKind::Scalar | SlotKind::Text | SlotKind::Object => {
                    // Most slots are plain stored values: one cell and nothing to read.
                    let kid = &jt.paths[part.path_idx as usize];
                    if kid.shapes.is_empty() && kid.cands.is_empty() {
                        if let Some(p) = col_pos.get_mut(kid.value_col as usize) {
                            *p += 1;
                        }
                    } else {
                        self.skip_path(part.path_idx as usize, jt, col_pos, budget, depth + 1)?;
                    }
                }
                SlotKind::Array => {
                    let arr_p = part.path_idx as usize;
                    let count = self.next_u32(jt.paths[arr_p].count_stream)? as usize;
                    if count > budget.max_records_per_block {
                        return Err(Error::BudgetExceeded("JsonTree array count"));
                    }
                    let elem_p = jt.paths[arr_p].elem_path as usize;
                    for _ in 0..count {
                        self.skip_path(elem_p, jt, col_pos, budget, depth + 1)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The next `n` values of structure stream `si`.
fn take<'s>(streams: &'s [Vec<u32>], pos: &mut [usize], si: u16, n: usize) -> Result<&'s [u32]> {
    let si = si as usize;
    let (Some(s), Some(at)) = (streams.get(si), pos.get_mut(si)) else {
        return Err(Error::Format("json struct stream index out of bounds".into()));
    };
    let vals = s
        .get(*at..*at + n)
        .ok_or_else(|| Error::Format("json struct stream underflow".into()))?;
    *at += n;
    Ok(vals)
}

/// Paths in an order that puts each after every path that can lead to it, for counting visits;
/// `None` when the tree allows none, or two paths read one stream and so share its order.
fn skip_order(jt: &JsonTreePlan) -> Option<Vec<u16>> {
    let n = jt.paths.len();
    let mut kids: Vec<Vec<u16>> = vec![Vec::new(); n];
    let mut array = vec![false; n];
    for (p, pdef) in jt.paths.iter().enumerate() {
        for part in pdef.shapes.iter().flatten() {
            if part.kind != SlotKind::None {
                kids[p].push(part.path_idx);
                array[part.path_idx as usize] |= part.kind == SlotKind::Array;
            }
        }
    }
    let mut owner = vec![false; jt.n_struct_streams];
    let mut claim = |s: u16| match owner.get_mut(s as usize) {
        Some(o) => !std::mem::replace(o, true),
        None => true,
    };
    for (p, pdef) in jt.paths.iter().enumerate() {
        let own = match pdef.shapes.is_empty() {
            true => pdef.cands.is_empty() || claim(pdef.sel_stream),
            false => claim(pdef.shape_stream),
        };
        if !own || (array[p] && !claim(pdef.count_stream)) {
            return None;
        }
        if array[p] && (pdef.elem_path as usize) < n {
            kids[p].push(pdef.elem_path);
        }
    }
    let mut ins = vec![0u32; n];
    kids.iter().flatten().for_each(|&k| ins[k as usize] += 1);
    let mut order: Vec<u16> = (0..n as u16).filter(|&p| ins[p as usize] == 0).collect();
    let mut i = 0;
    while i < order.len() {
        for &k in &kids[order[i] as usize] {
            ins[k as usize] -= 1;
            if ins[k as usize] == 0 {
                order.push(k);
            }
        }
        i += 1;
    }
    (order.len() == n).then_some(order)
}

#[cold]
fn stream_underflow() -> Error {
    Error::Format("json struct stream underflow".into())
}

/// Appends the next unread cell of value column `cid` as it appears in the record.
#[inline(always)]
fn emit_json_stored_cell(
    cid: usize,
    plan: &ModelPlan,
    ctx: &mut BlockDecodeCtx<'_>,
    col_pos: &mut [usize],
    out: &mut Vec<u8>,
    budget: &DecodeBudget,
) -> Result<()> {
    let p = col_pos
        .get_mut(cid)
        .ok_or_else(|| Error::Format("JsonTree col index out of bounds".into()))?;
    *p += 1;
    let nested = plan.columns[cid].nested_plan.is_some();
    let col = block_col(cid, plan, ctx, budget)?;
    if !nested {
        return col.write(*p - 1, out);
    }
    let mut cell = Vec::new();
    col.write(*p - 1, &mut cell)?;
    emit_nested_cell(&cell, out)
}

/// Writes a nested column's cell as its record bytes: tag `1` is a string holding the inner
/// record, tag `0` any other token.
pub(crate) fn emit_nested_cell(cell: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let (&tag, inner) = cell
        .split_first()
        .ok_or_else(|| Error::Format("empty nested JSON cell".into()))?;
    if tag == 1 {
        out.push(b'"');
        escape_json_str_exact(inner, out);
        out.push(b'"');
    } else {
        out.extend_from_slice(inner);
    }
    Ok(())
}

// ============================================================================
// 5. Compressed-Domain Query Occurrence Mapper (Decodes 0 Value Columns!)
// ============================================================================

/// Per-path occurrence action recorded by walking ONLY the tiny `u32` structural streams!
#[derive(Clone, Debug)]
pub enum PathRowEntry {
    /// Stored directly in `value_col` at dense occurrence index `occ_idx`.
    LiteralOcc { col_id: u16, occ_idx: u32 },
    /// Synthesized via `cands[cand_idx]` using the latest occurrence of referenced paths on this row!
    Derived { path_idx: u16, cand_idx: u8 },
}

/// One value of a tracked path on one row.
#[derive(Debug, Clone, Copy)]
pub enum PathVal {
    /// Cell `occ` of column `col`.
    Occ { col: u16, occ: u32 },
    /// Built by candidate `cand` of the path from the events listed at `JsonBlockMap::srcs[srcs..]`.
    Derived { cand: u8, srcs: u32 },
    /// A hook the walk was given for the shape the path took.
    Hook(u32),
}

#[derive(Debug, Clone, Copy)]
pub struct PathEvent {
    pub row: u32,
    pub path: u16,
    pub val: PathVal,
}

/// What a walk over a block's row structure reports.
#[derive(Default)]
pub struct JsonWalk<'a> {
    /// By column: record the row of each occurrence.
    pub want_cols: &'a [bool],
    /// By path: report each value. The paths a tracked path derives from must be tracked too.
    pub track: &'a [bool],
    /// `hooks[path][shape - 1]` lists `(part, id)` by part: `id` is reported when the path takes
    /// that shape, before the slot of `part`.
    pub hooks: &'a [Vec<Vec<(u16, u32)>>],
}

#[derive(Default)]
pub struct JsonBlockMap {
    /// For each wanted column, the row of each of its occurrences, in order.
    pub col_occ_rows: Vec<Vec<u32>>,
    /// Values of tracked paths in record order.
    pub events: Vec<PathEvent>,
    pub srcs: Vec<u32>,
}

/// By path: whether a walk visits it, as a path asked for or one above such a path.
fn walk_paths(jt: &JsonTreePlan, walk: &JsonWalk<'_>) -> Vec<bool> {
    let n = jt.paths.len();
    let flag = |v: &[bool], i: usize| v.get(i).copied().unwrap_or(false);
    let mut parent = vec![usize::MAX; n];
    for (pi, p) in jt.paths.iter().enumerate() {
        for part in p
            .shapes
            .iter()
            .flatten()
            .filter(|part| part.kind != SlotKind::None)
        {
            parent[part.path_idx as usize] = pi;
        }
        if p.elem_path != u16::MAX {
            parent[p.elem_path as usize] = pi;
        }
    }
    let mut needed = vec![false; n];
    for (pi, p) in jt.paths.iter().enumerate() {
        let own = flag(walk.track, pi)
            || walk.hooks.get(pi).is_some_and(|h| !h.is_empty())
            || (p.value_col != u16::MAX && flag(walk.want_cols, p.value_col as usize));
        let mut at = pi;
        while own && at != usize::MAX && !needed[at] {
            needed[at] = true;
            at = parent[at];
        }
    }
    needed
}

/// The part of a tree a walk visits. It depends on the walk alone, so one serves every block.
#[derive(Default)]
pub struct WalkRoute {
    paths: Vec<bool>,
    /// By path and shape: the parts that lead to something asked for. A shape can have a hundred
    /// parts and a walk wants a few.
    stops: Vec<Vec<Vec<u16>>>,
}

impl WalkRoute {
    pub fn of(jt: &JsonTreePlan, walk: &JsonWalk<'_>) -> Self {
        let paths = walk_paths(jt, walk);
        let stop =
            |part: &JsonSlotPart| part.kind != SlotKind::None && paths[part.path_idx as usize];
        let stops = jt
            .paths
            .iter()
            .zip(&paths)
            .map(|(p, &on)| {
                let shapes = p.shapes.iter().filter(|_| on);
                shapes
                    .map(|s| {
                        (0..s.len() as u16)
                            .filter(|&i| stop(&s[i as usize]))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self { paths, stops }
    }

    /// By structure stream: whether the walk reads it.
    pub fn streams(&self, jt: &JsonTreePlan) -> Vec<bool> {
        let mut out = vec![false; jt.n_struct_streams];
        for (p, _) in jt.paths.iter().zip(&self.paths).filter(|(_, on)| **on) {
            for s in [p.shape_stream, p.count_stream, p.sel_stream] {
                if let Some(o) = out.get_mut(s as usize) {
                    *o = true;
                }
            }
        }
        out
    }
}

/// Walks the rows of variant `json_vi`, visiting only the subtrees that hold something asked for.
/// A stream the reader did not fetch is empty, and reading it is `Error::NotLoaded`.
pub fn map_json_block_structure(
    jt: &JsonTreePlan,
    struct_streams: &[Vec<u32>],
    variant_ids: &[u32],
    json_vi: u32,
    walk: &JsonWalk<'_>,
    route: &WalkRoute,
) -> Result<JsonBlockMap> {
    let n = jt.paths.len();
    if route.paths.len() != n {
        return Err(Error::Format("walk route of another tree".into()));
    }
    let mut w = StructWalk {
        jt,
        streams: struct_streams,
        spos: vec![0usize; struct_streams.len()],
        occ: vec![0u32; n],
        scope: vec![(u32::MAX, 0u32); n],
        stops: &route.stops,
        walk,
        map: JsonBlockMap {
            col_occ_rows: vec![Vec::new(); walk.want_cols.len()],
            ..Default::default()
        },
    };
    if !route.paths[0] {
        return Ok(w.map);
    }
    for (r, &vid) in variant_ids.iter().enumerate() {
        if vid == json_vi {
            w.row(0, r as u32)?;
        }
    }
    Ok(w.map)
}

struct StructWalk<'a> {
    jt: &'a JsonTreePlan,
    streams: &'a [Vec<u32>],
    spos: Vec<usize>,
    /// By path: stored occurrences seen so far.
    occ: Vec<u32>,
    /// By path: the row and event of its latest value.
    scope: Vec<(u32, u32)>,
    /// By path and shape: the parts that lead to something asked for.
    stops: &'a [Vec<Vec<u16>>],
    walk: &'a JsonWalk<'a>,
    map: JsonBlockMap,
}

impl StructWalk<'_> {
    #[inline]
    fn next(&mut self, stream: u16) -> Result<usize> {
        let si = stream as usize;
        let s = self
            .streams
            .get(si)
            .ok_or_else(|| Error::Format("struct stream out of range".into()))?;
        let v = s.get(self.spos[si]).ok_or_else(|| match s.is_empty() {
            true => Error::NotLoaded,
            false => Error::Format("struct stream underflow in query walk".into()),
        })?;
        self.spos[si] += 1;
        Ok(*v as usize)
    }

    #[inline]
    fn tracked(&self, path_idx: usize) -> bool {
        self.walk.track.get(path_idx).copied().unwrap_or(false)
    }

    fn event(&mut self, path_idx: usize, row: u32, val: PathVal) {
        self.scope[path_idx] = (row, self.map.events.len() as u32);
        self.map.events.push(PathEvent {
            row,
            path: path_idx as u16,
            val,
        });
    }

    /// Records one stored occurrence of `path_idx` on `row`.
    #[inline]
    fn stored(&mut self, path_idx: usize, row: u32) {
        let col = self.jt.paths[path_idx].value_col;
        if col == u16::MAX {
            return;
        }
        let occ = self.count_cell(path_idx, row);
        if self.tracked(path_idx) {
            self.event(path_idx, row, PathVal::Occ { col, occ });
        }
    }

    /// Counts a cell of `path_idx`'s column on `row`; returns its occurrence.
    #[inline]
    fn count_cell(&mut self, path_idx: usize, row: u32) -> u32 {
        let col = self.jt.paths[path_idx].value_col as usize;
        let occ = self.occ[path_idx];
        self.occ[path_idx] += 1;
        if self.walk.want_cols.get(col).copied().unwrap_or(false) {
            self.map.col_occ_rows[col].push(row);
        }
        occ
    }

    fn row(&mut self, path_idx: usize, row: u32) -> Result<()> {
        let jt = self.jt;
        let pdef = &jt.paths[path_idx];
        if pdef.shapes.is_empty() {
            let sel = if pdef.cands.is_empty() {
                0
            } else {
                self.next(pdef.sel_stream)?
            };
            if sel == 0 {
                self.stored(path_idx, row);
                return Ok(());
            }
            let cand = pdef
                .cands
                .get(sel - 1)
                .ok_or_else(|| Error::Format("JsonTree sel out of range".into()))?;
            let free = cand.has_free(path_idx).then(|| self.count_cell(path_idx, row));
            if self.tracked(path_idx) {
                let srcs = self.map.srcs.len() as u32;
                let one;
                let refs: &[u16] = match cand {
                    JsonCand::Copy { ref_path } | JsonCand::Affix { ref_path, .. } => {
                        one = [*ref_path];
                        &one
                    }
                    JsonCand::MultiConcat { refs, .. } => refs,
                };
                for &r in refs {
                    // The free part's source is its cell, given by occurrence.
                    if let (true, Some(occ)) = (r as usize == path_idx, free) {
                        self.map.srcs.push(occ);
                        continue;
                    }
                    let (at, ev) = self.scope[r as usize];
                    if at != row {
                        return Err(Error::Format("derived value without its source".into()));
                    }
                    self.map.srcs.push(ev);
                }
                self.event(
                    path_idx,
                    row,
                    PathVal::Derived {
                        cand: (sel - 1) as u8,
                        srcs,
                    },
                );
            }
            return Ok(());
        }
        let shape_id = self.next(pdef.shape_stream)?;
        if shape_id == 0 {
            self.stored(path_idx, row);
            return Ok(());
        }
        let shape = pdef
            .shapes
            .get(shape_id - 1)
            .ok_or_else(|| Error::Format("shape_id out of bounds in query walk".into()))?;
        let hooks: &[(u16, u32)] = self
            .walk
            .hooks
            .get(path_idx)
            .and_then(|h| h.get(shape_id - 1))
            .map_or(&[], |h| h.as_slice());
        let mut hi = 0usize;
        let stops = self.stops;
        for &i in &stops[path_idx][shape_id - 1] {
            while hi < hooks.len() && hooks[hi].0 <= i {
                self.event(path_idx, row, PathVal::Hook(hooks[hi].1));
                hi += 1;
            }
            let part = &shape[i as usize];
            let target = part.path_idx as usize;
            if part.kind == SlotKind::Array {
                let arr = &jt.paths[target];
                for _ in 0..self.next(arr.count_stream)? {
                    self.row(arr.elem_path as usize, row)?;
                }
            } else {
                self.row(target, row)?;
            }
        }
        for &(_, id) in &hooks[hi..] {
            self.event(path_idx, row, PathVal::Hook(id));
        }
        Ok(())
    }
}

/// Calls `f(start, end)` with the span in `text` of each value at `target`, a path relative to the
/// value `text` spells, written as `JsonPathDef::name` writes keys. Arrays are transparent: a path
/// names the elements of an array it reaches or passes through. Returns false if `text` is not
/// one JSON value.
pub fn json_values_at(
    doc: &mut JsonDoc,
    text: &[u8],
    target: &[u8],
    f: &mut dyn FnMut(usize, usize),
) -> bool {
    fn visit(
        doc: &JsonDoc,
        ni: usize,
        text: &[u8],
        path: &mut Vec<u8>,
        target: &[u8],
        f: &mut dyn FnMut(usize, usize),
    ) {
        let v = doc.nodes[ni];
        if v.kind == J_ARRAY {
            for ki in doc.kid_range(ni) {
                visit(doc, doc.kids[ki].node as usize, text, path, target, f);
            }
        } else if path.len() == target.len() {
            f(v.start as usize, v.end as usize);
        } else if v.kind == J_OBJECT {
            let base = path.len();
            for ki in doc.kid_range(ni) {
                let kid = doc.kids[ki];
                let raw_k = &text[kid.key_start as usize..kid.key_end as usize];
                path.truncate(base);
                match kid.key_esc.then(|| decode_json_key(raw_k)) {
                    None => append_path_key(path, raw_k),
                    Some(Some(k)) => append_path_key(path, &k),
                    Some(None) => continue,
                }
                let boundary = matches!(target.get(path.len()), None | Some(b'.' | b'['));
                if boundary && target.starts_with(path) {
                    visit(doc, kid.node as usize, text, path, target, f);
                }
            }
            path.truncate(base);
        }
    }
    let Some(root) = doc.parse_any(text) else {
        return false;
    };
    visit(doc, root, text, &mut Vec::new(), target, f);
    true
}

// ============================================================================
// 6. Edition-2 Recursive `JsonTreePlan` Trainer + Cross-Path Derivation Finder
// ============================================================================

const ROLE_INLINE: u8 = 0;
const ROLE_LEAF: u8 = 1;
const ROLE_ARRAY: u8 = 2;
const ROLE_TABLED: u8 = 3;
const ROLE_TEXT: u8 = 4;

#[derive(Default)]
struct TPath {
    name: Vec<u8>,
    is_elem: bool,
    kinds: [u32; 6],
    first_lexeme: Vec<u8>,
    varies: bool,
    non_empty_containers: u32,
    total_str_bytes: u64,
    kids: HashMap<Vec<u8>, usize>,
    elem_idx: i32,
    wide: bool,
    role: u8,
    prov_id: i32,
    kept_shape_keys: Vec<Vec<u8>>,
    kept_shapes: HashMap<Vec<u8>, Vec<RawPart>>,
}

#[derive(Clone)]
struct RawPart {
    literal: Vec<u8>,
    kind: SlotKind,
    target_name: Vec<u8>,
    sep: Vec<u8>,
}

impl TPath {
    fn decide_role(&self) -> u8 {
        let k = &self.kinds;
        let classes = k[1..].iter().filter(|&&n| n > 0).count();
        if self.wide || (self.is_elem && k[J_ARRAY as usize] > 0) {
            ROLE_TEXT
        } else if k[J_OBJECT as usize] > 0
            && self.non_empty_containers > 0
            && (self.is_elem || classes == 1)
        {
            ROLE_TABLED
        } else if k[J_OBJECT as usize] > 0
            && self.non_empty_containers > 0
            && classes == 2
            && k[J_LITERAL as usize] > 0
        {
            // Nullable object (`{...}` or `null`): keep object `ROLE_TABLED`, `null` stays inline in parent!
            ROLE_TABLED
        } else if classes > 1 && k[J_OBJECT as usize] == 0 && k[J_ARRAY as usize] == 0 {
            // Nullable scalar (`"abc"` or `null`, or `123` or `null`) -> `ROLE_LEAF`!
            if self.varies { ROLE_LEAF } else { ROLE_INLINE }
        } else if k[J_ARRAY as usize] > 0 && self.non_empty_containers > 0 {
            ROLE_ARRAY
        } else if (k[J_STRING as usize] > 0
            || k[J_NUMBER as usize] > 0
            || k[J_LITERAL as usize] > 0)
            && (self.varies || self.is_elem)
        {
            ROLE_LEAF
        } else if classes > 1 {
            ROLE_TEXT
        } else {
            ROLE_INLINE
        }
    }
}

#[derive(Default)]
struct TreeTrainer {
    paths: Vec<TPath>,
    index: HashMap<Vec<u8>, usize>,
}

impl TreeTrainer {
    fn add_path(&mut self, name: Vec<u8>, is_elem: bool) -> usize {
        if let Some(&idx) = self.index.get(&name) {
            return idx;
        }
        let idx = self.paths.len();
        self.index.insert(name.clone(), idx);
        self.paths.push(TPath {
            name,
            is_elem,
            elem_idx: -1,
            prov_id: -1,
            ..Default::default()
        });
        idx
    }

    fn observe(&mut self, doc: &JsonDoc, s: &[u8], ni: usize, p: usize) {
        let v = doc.nodes[ni];
        {
            let tp = &mut self.paths[p];
            tp.kinds[v.kind as usize] += 1;
            if v.kind >= J_STRING {
                let lx = &s[v.start as usize..v.end as usize];
                if v.kind == J_STRING {
                    tp.total_str_bytes += lx.len() as u64;
                }
                let scalar_seen = tp.kinds[3] + tp.kinds[4] + tp.kinds[5];
                if scalar_seen == 1 {
                    tp.first_lexeme = lx.to_vec();
                } else if !tp.varies && lx != tp.first_lexeme.as_slice() {
                    tp.varies = true;
                }
            }
            if v.n > 0 {
                tp.non_empty_containers += 1;
            }
        }
        match v.kind {
            J_OBJECT => {
                // Skip duplicate keys inside the same object instance
                let range = doc.kid_range(ni);
                for k in range.clone() {
                    let kid = doc.kids[k];
                    let raw_k = &s[kid.key_start as usize..kid.key_end as usize];
                    let dup = doc.kids[range.start..k]
                        .iter()
                        .any(|prev| &s[prev.key_start as usize..prev.key_end as usize] == raw_k);
                    if dup {
                        continue;
                    }
                    let decoded_k = if kid.key_esc {
                        match decode_json_key(raw_k) {
                            Some(d) => d,
                            None => continue,
                        }
                    } else {
                        raw_k.to_vec()
                    };
                    let existing = self.paths[p].kids.get(&decoded_k).copied();
                    let child_p = match existing {
                        Some(cp) => cp,
                        None => {
                            let mut cname = self.paths[p].name.clone();
                            append_path_key(&mut cname, &decoded_k);
                            if self.paths[p].wide
                                || self.paths[p].kids.len() >= TRAIN_WIDE_KEYS
                                || cname.len() > MAX_PATH_NAME_LEN
                                || std::str::from_utf8(&cname).is_err()
                            {
                                self.paths[p].wide = true;
                                continue;
                            }
                            let cp = self.add_path(cname, false);
                            self.paths[p].kids.insert(decoded_k, cp);
                            cp
                        }
                    };
                    self.observe(doc, s, kid.node as usize, child_p);
                }
            }
            J_ARRAY => {
                if v.n > 0 && self.paths[p].elem_idx < 0 {
                    let mut ename = self.paths[p].name.clone();
                    ename.extend_from_slice(b"[]");
                    if ename.len() <= MAX_PATH_NAME_LEN {
                        self.paths[p].elem_idx = self.add_path(ename, true) as i32;
                    }
                }
                let ei = self.paths[p].elem_idx;
                if ei >= 0 {
                    for k in doc.kid_range(ni) {
                        self.observe(doc, s, doc.kids[k].node as usize, ei as usize);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Trains a `JsonTreePlan` on sampled JSON records, allocates value columns on `model_plan`,
/// discovers cross-path `JsonCand` derivations, and optionally trains depth-1 nested string plans!
pub fn train_json_tree(
    json_rows: &[&[u8]],
    opts: &crate::plan::TrainOptions,
    model_plan: &mut crate::grammar::ModelPlan,
) -> Option<JsonTreePlan> {
    let mut tr = TreeTrainer::default();
    let root = tr.add_path(b"$".to_vec(), false);
    let mut doc = JsonDoc::default();
    let mut parsed_count = 0usize;

    for &row in json_rows {
        if doc.parse_root_object(row) {
            parsed_count += 1;
            tr.observe(&doc, row, 0, root);
        }
    }
    if parsed_count < 2 {
        return None;
    }

    // A scalar key that varies under one parent varies under every parent: the sample may have
    // missed the rare value here, and a value baked into a layout costs an opaque row later.
    let scalar = |tp: &TPath| tp.kinds[J_OBJECT as usize] == 0 && tp.kinds[J_ARRAY as usize] == 0;
    let mut varying: HashSet<&[u8]> = HashSet::new();
    for tp in &tr.paths {
        for (key, &c) in &tp.kids {
            if tr.paths[c].varies && scalar(&tr.paths[c]) {
                varying.insert(key);
            }
        }
    }
    let spread: Vec<usize> = tr
        .paths
        .iter()
        .flat_map(|tp| tp.kids.iter())
        .filter(|&(key, &c)| varying.contains(key.as_slice()) && scalar(&tr.paths[c]))
        .map(|(_, &c)| c)
        .collect();
    for c in spread {
        tr.paths[c].varies = true;
    }

    // Decide initial path roles
    for i in 0..tr.paths.len() {
        tr.paths[i].role = tr.paths[i].decide_role();
    }
    tr.paths[root].role = ROLE_TABLED;

    // If an array's element path ended up `ROLE_INLINE` (e.g. `[]` or constant element), keep array inline too
    for i in 0..tr.paths.len() {
        if tr.paths[i].role == ROLE_ARRAY {
            let ei = tr.paths[i].elem_idx;
            if ei < 0 || tr.paths[ei as usize].role == ROLE_INLINE {
                tr.paths[i].role = ROLE_INLINE;
            }
        }
    }

    // Iterative census + shape selection + demotion loop (at most 3 rounds if paths > MAX_PLAN_PATHS)
    let kept_order = loop {
        // Build provisional lookup trie over all non-inline paths
        let mut prov_names: Vec<Vec<u8>> = Vec::new();
        let mut prov_to_tp: Vec<usize> = Vec::new();
        for (i, tp) in tr.paths.iter_mut().enumerate() {
            tp.prov_id = -1;
            if tp.role >= ROLE_LEAF {
                tp.prov_id = prov_names.len() as i32;
                prov_names.push(tp.name.clone());
                prov_to_tp.push(i);
            }
        }
        let name_refs: Vec<&[u8]> = prov_names.iter().map(|v| v.as_slice()).collect();
        let prov_trie = build_path_trie(&name_refs);

        // Census shape occurrences for every `ROLE_TABLED` path
        let mut census: Vec<HashMap<Vec<u8>, u32>> = vec![HashMap::new(); tr.paths.len()];
        for &row in json_rows {
            if doc.parse_root_object(row) {
                census_node(
                    &doc,
                    row,
                    0,
                    root,
                    0,
                    &tr,
                    &prov_trie,
                    &prov_to_tp,
                    &mut census,
                );
            }
        }

        // Keep top shapes per `ROLE_TABLED` path; demote low-coverage sub-objects to `ROLE_TEXT`
        let mut any_role_flipped = false;
        for i in 0..tr.paths.len() {
            if tr.paths[i].role != ROLE_TABLED {
                continue;
            }
            tr.paths[i].kept_shape_keys.clear();
            tr.paths[i].kept_shapes.clear();
            let mut list: Vec<(&Vec<u8>, u32)> = census[i].iter().map(|(k, &c)| (k, c)).collect();
            let total: u64 = list.iter().map(|x| x.1 as u64).sum();
            list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
            let min_cnt = 1u64.max((TRAIN_SHAPE_MIN * total as f64 + 0.999) as u64);
            let mut covered = 0u64;
            for (kbytes, cnt) in list {
                if (cnt as u64) < min_cnt || tr.paths[i].kept_shape_keys.len() >= MAX_PATH_SHAPES {
                    continue;
                }
                let Some(raw_parts) = decode_census_key(kbytes, &prov_names) else {
                    continue;
                };
                tr.paths[i].kept_shape_keys.push(kbytes.clone());
                tr.paths[i].kept_shapes.insert(kbytes.clone(), raw_parts);
                covered += cnt as u64;
            }
            if i > 0
                && (tr.paths[i].kept_shape_keys.is_empty()
                    || (covered as f64) < TRAIN_SHAPE_COVER * (total as f64))
            {
                tr.paths[i].role = ROLE_TEXT;
                tr.paths[i].kept_shape_keys.clear();
                tr.paths[i].kept_shapes.clear();
                any_role_flipped = true;
            }
        }
        if any_role_flipped {
            continue;
        }
        if tr.paths[root].kept_shape_keys.is_empty() {
            return None;
        }

        // Compute reachable DFS order from `$`
        let order = reachable_dfs_order(&tr, root);
        if order.len() <= MAX_PLAN_PATHS.min(opts.max_json_paths.max(64)) {
            break order;
        }
        if !demote_widest_subtrees(
            &mut tr,
            &order,
            MAX_PLAN_PATHS.min(opts.max_json_paths.max(64)),
        ) {
            return None;
        }
    };

    // Map kept `TPath` indices -> `0..kept_order.len()` in `JsonTreePlan::paths`
    let mut tp_to_plan: HashMap<usize, u16> = HashMap::with_capacity(kept_order.len());
    for (pi, &tpi) in kept_order.iter().enumerate() {
        tp_to_plan.insert(tpi, pi as u16);
    }

    let mut paths = Vec::with_capacity(kept_order.len());
    let mut n_struct_streams = 0usize;

    for &tpi in &kept_order {
        let tp = &tr.paths[tpi];
        let name_str = String::from_utf8_lossy(&tp.name).into_owned();
        let is_text = tp.role == ROLE_TEXT;
        let elem_path = if tp.role == ROLE_ARRAY && tp.elem_idx >= 0 {
            tp_to_plan
                .get(&(tp.elem_idx as usize))
                .copied()
                .unwrap_or(u16::MAX)
        } else {
            u16::MAX
        };

        let mut shapes = Vec::new();
        if tp.role == ROLE_TABLED {
            for k in &tp.kept_shape_keys {
                let rparts = &tp.kept_shapes[k];
                let mut parts = Vec::with_capacity(rparts.len());
                let mut valid = true;
                for rp in rparts {
                    if rp.kind == SlotKind::None {
                        parts.push(JsonSlotPart {
                            literal: rp.literal.clone(),
                            kind: SlotKind::None,
                            path_idx: 0,
                            sep: Vec::new(),
                        });
                    } else {
                        let Some(&orig_tp) = tr.index.get(&rp.target_name) else {
                            valid = false;
                            break;
                        };
                        let Some(&target_pi) = tp_to_plan.get(&orig_tp) else {
                            valid = false;
                            break;
                        };
                        let actual_kind = if matches!(rp.kind, SlotKind::Object | SlotKind::Text) {
                            if tr.paths[orig_tp].role == ROLE_TEXT {
                                SlotKind::Text
                            } else {
                                SlotKind::Object
                            }
                        } else {
                            rp.kind
                        };
                        parts.push(JsonSlotPart {
                            literal: rp.literal.clone(),
                            kind: actual_kind,
                            path_idx: target_pi,
                            sep: rp.sep.clone(),
                        });
                    }
                }
                if valid {
                    shapes.push(parts);
                }
            }
        }

        let shape_stream = if !shapes.is_empty() {
            let s = n_struct_streams as u16;
            n_struct_streams += 1;
            s
        } else {
            u16::MAX
        };
        let count_stream = if elem_path != u16::MAX {
            let s = n_struct_streams as u16;
            n_struct_streams += 1;
            s
        } else {
            u16::MAX
        };

        // Determine whether this path needs a `value_col` (all `Leaf`, `Text`, and `Tabled` paths do; pure `Array` container paths do not!)
        let value_col = if tp.role != ROLE_ARRAY {
            model_plan.add_or_get_col(&name_str)
        } else {
            u16::MAX
        };

        // Heuristic group hint: prose `"text"` when `ROLE_TEXT` or average string lexeme > 72 bytes, else field basename
        let avg_str_len = if tp.kinds[J_STRING as usize] > 0 {
            tp.total_str_bytes / (tp.kinds[J_STRING as usize] as u64)
        } else {
            0
        };
        let dict_group = if is_text || avg_str_len > 64 {
            "text".to_string()
        } else {
            basename_group(&name_str)
        };
        if value_col != u16::MAX {
            model_plan.columns[value_col as usize].dict_group = dict_group.clone();
        }

        paths.push(JsonPathDef {
            name: name_str,
            is_text,
            shapes,
            cands: Vec::new(),
            dict_group,
            shape_stream,
            count_stream,
            sel_stream: u16::MAX,
            value_col,
            elem_path,
        });
    }

    let mut jt = JsonTreePlan {
        paths,
        n_struct_streams,
        runtime: JsonTreeRuntime::default(),
    };
    jt.rebuild_runtime_index();

    // Pass 2: Discover Cross-Path `JsonCand` Derivations (`Copy`, `Affix`, `MultiConcat`) across ancestor + array scopes!
    discover_json_derivations(json_rows, &mut jt);

    // Pass 3: Depth-1 Nested String Unwrapping (`{"log":"{\"level\":...}\n"}`) on non-derived string leaves!
    if opts.depth == 0 {
        discover_nested_json_strings(json_rows, opts, &jt, model_plan);
    }

    Some(jt)
}

fn basename_group(path: &str) -> String {
    let clean = path.strip_suffix("[]").unwrap_or(path);
    clean.rsplit('.').next().unwrap_or(clean).to_string()
}

#[allow(clippy::too_many_arguments)]
fn census_node(
    doc: &JsonDoc,
    s: &[u8],
    ni: usize,
    tp_idx: usize,
    trie_node: usize,
    tr: &TreeTrainer,
    prov_trie: &[PathTrieNode],
    prov_to_tp: &[usize],
    census: &mut [HashMap<Vec<u8>, u32>],
) {
    let role = tr.paths[tp_idx].role;
    let v = doc.nodes[ni];
    if role != ROLE_TABLED || v.kind != J_OBJECT {
        return;
    }

    let mut key_buf = Vec::<u8>::with_capacity(128);
    let mut children_to_visit = Vec::<(usize, usize, usize)>::new();
    let mut pos = v.start as usize;

    for k in doc.kid_range(ni) {
        let kid = doc.kids[k];
        let raw_k = &s[kid.key_start as usize..kid.key_end as usize];
        let ct_opt = if kid.key_esc {
            decode_json_key(raw_k).and_then(|d| prov_trie[trie_node].kids.get(&d).copied())
        } else {
            prov_trie[trie_node].kids.get(raw_k).copied()
        };
        let Some(ct) = ct_opt else {
            continue;
        };
        let pid = prov_trie[ct].path_idx;
        if pid < 0 {
            continue;
        }
        let child_tp = prov_to_tp[pid as usize];
        let child_role = tr.paths[child_tp].role;
        let cv = doc.nodes[kid.node as usize];

        let (kind, sep): (SlotKind, &[u8]) = match child_role {
            ROLE_TEXT => (SlotKind::Text, &[]),
            ROLE_TABLED => {
                if cv.kind == J_OBJECT {
                    children_to_visit.push((kid.node as usize, child_tp, ct));
                }
                (SlotKind::Object, &[])
            }
            ROLE_ARRAY => {
                if cv.kind != J_ARRAY || (cv.n == 0 && cv.end - cv.start != 2) {
                    continue; // `null` or a spaced `[ ]` stays inline!
                }
                let Some(sp) = uniform_array_sep(doc, s, kid.node as usize) else {
                    return;
                };
                let elem_tp = tr.paths[child_tp].elem_idx as usize;
                let elem_ct = prov_trie[ct].elem_node;
                if elem_ct >= 0 && tr.paths[elem_tp].role == ROLE_TABLED {
                    for ek in doc.kid_range(kid.node as usize) {
                        children_to_visit.push((
                            doc.kids[ek].node as usize,
                            elem_tp,
                            elem_ct as usize,
                        ));
                    }
                }
                (SlotKind::Array, sp)
            }
            ROLE_LEAF => {
                if cv.kind >= J_STRING {
                    (SlotKind::Scalar, &[])
                } else {
                    return;
                }
            }
            _ => continue,
        };

        if kind == SlotKind::Array {
            let (elems_start, elems_end) = array_elems_span(doc, &cv);
            put_bytes(&mut key_buf, &s[pos..elems_start]);
            key_buf.push(SlotKind::Array as u8);
            put_uvarint(&mut key_buf, pid as u64);
            put_bytes(&mut key_buf, sep);
            pos = elems_end;
        } else {
            put_bytes(&mut key_buf, &s[pos..cv.start as usize]);
            key_buf.push(kind as u8);
            put_uvarint(&mut key_buf, pid as u64);
            pos = cv.end as usize;
        }
    }
    put_bytes(&mut key_buf, &s[pos..v.end as usize]);
    key_buf.push(SlotKind::None as u8);

    *census[tp_idx].entry(key_buf).or_insert(0) += 1;

    for (c_ni, c_tp, c_trie) in children_to_visit {
        census_node(
            doc, s, c_ni, c_tp, c_trie, tr, prov_trie, prov_to_tp, census,
        );
    }
}

fn decode_census_key(key: &[u8], prov_names: &[Vec<u8>]) -> Option<Vec<RawPart>> {
    let mut cur = Cursor::new(key);
    let mut parts = Vec::new();
    while cur.remaining() > 0 {
        if parts.len() >= MAX_SHAPE_PARTS {
            return None;
        }
        let lit = cur.bytes().ok()?.to_vec();
        let kind = SlotKind::from_u8(cur.u8().ok()?).ok()?;
        let mut target_name = Vec::new();
        let mut sep = Vec::new();
        if kind != SlotKind::None {
            let pid = cur.uvar().ok()? as usize;
            target_name = prov_names.get(pid)?.clone();
            if kind == SlotKind::Array {
                sep = cur.bytes().ok()?.to_vec();
            }
        }
        parts.push(RawPart {
            literal: lit,
            kind,
            target_name,
            sep,
        });
    }
    Some(parts)
}

fn reachable_dfs_order(tr: &TreeTrainer, root: usize) -> Vec<usize> {
    let mut seen = vec![false; tr.paths.len()];
    let mut out = Vec::new();
    seen[root] = true;
    out.push(root);
    walk_dfs(tr, root, &mut seen, &mut out);
    out
}

fn walk_dfs(tr: &TreeTrainer, p: usize, seen: &mut [bool], out: &mut Vec<usize>) {
    let tp = &tr.paths[p];
    if tp.role == ROLE_ARRAY && tp.elem_idx >= 0 {
        let ei = tp.elem_idx as usize;
        if !seen[ei] {
            seen[ei] = true;
            out.push(ei);
            walk_dfs(tr, ei, seen, out);
        }
        return;
    }
    if tp.role != ROLE_TABLED {
        return;
    }
    for k in &tp.kept_shape_keys {
        for rp in &tp.kept_shapes[k] {
            if rp.kind == SlotKind::None {
                continue;
            }
            let Some(&ci) = tr.index.get(&rp.target_name) else {
                continue;
            };
            if !seen[ci] {
                seen[ci] = true;
                out.push(ci);
                walk_dfs(tr, ci, seen, out);
            }
        }
    }
}

fn demote_widest_subtrees(tr: &mut TreeTrainer, order: &[usize], limit: usize) -> bool {
    let np = tr.paths.len();
    let mut parent = vec![usize::MAX; np];
    for (i, tp) in tr.paths.iter().enumerate() {
        for &c in tp.kids.values() {
            parent[c] = i;
        }
        if tp.elem_idx >= 0 {
            parent[tp.elem_idx as usize] = i;
        }
    }
    let mut below = vec![0usize; np];
    for &q in &order[1..] {
        let mut a = parent[q];
        while a != usize::MAX {
            below[a] += 1;
            a = parent[a];
        }
    }
    let mut cands: Vec<usize> = order[1..]
        .iter()
        .copied()
        .filter(|&p| tr.paths[p].role == ROLE_TABLED && below[p] > 0)
        .collect();
    cands.sort_by(|&a, &b| below[b].cmp(&below[a]));
    let mut gone = vec![false; np];
    let mut rem = order.len();
    let mut demoted = false;
    for p in cands {
        if rem <= limit {
            break;
        }
        let mut anc_gone = false;
        let mut a = parent[p];
        while a != usize::MAX {
            if gone[a] {
                anc_gone = true;
                break;
            }
            a = parent[a];
        }
        if anc_gone {
            continue;
        }
        gone[p] = true;
        demoted = true;
        tr.paths[p].role = ROLE_TEXT;
        tr.paths[p].kept_shape_keys.clear();
        tr.paths[p].kept_shapes.clear();
        rem = rem.saturating_sub(below[p]);
    }
    demoted
}

// ============================================================================
// 7. Cross-Path Derivation Discovery (`Copy`, `Affix`, `MultiConcat`)
// ============================================================================

struct ScopeOcc<'a> {
    target: &'a [u8],
    /// The values in scope, by path, ascending.
    refs: Vec<(u16, &'a [u8])>,
}

impl<'a> ScopeOcc<'a> {
    fn val(&self, r: u16) -> Option<&'a [u8]> {
        let i = self.refs.binary_search_by_key(&r, |&(p, _)| p).ok()?;
        Some(self.refs[i].1)
    }
}

/// `0..n` stepped by a stride near `n / φ` that is coprime to `n`: every prefix is spread evenly
/// over the range, without the power-of-two aliasing of a bit-reversed order.
fn spread_order(n: usize) -> impl Iterator<Item = usize> {
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let mut step = (n as f64 * 0.618) as usize;
    while gcd(step, n) != 1 && step < n {
        step += 1;
    }
    (0..n).map(move |k| k * step % n)
}

// Values live in the sampled input. Cache their traversal order once per row. On
// overflow the input itself backs replay, including for a single oversized row.
const DERIV_CACHE_EVENTS: usize = 1 << 19;

#[derive(Clone, Copy)]
struct FieldEvent {
    path: u16,
    start: u32,
    end: u32,
}

impl FieldEvent {
    fn value(self, row: &[u8]) -> &[u8] {
        &row[self.start as usize..self.end as usize]
    }
}

#[derive(Clone, Copy)]
struct Observation {
    row: usize,
    event: usize,
}

struct SampleRow<'a> {
    bytes: &'a [u8],
    cached: Option<std::ops::Range<usize>>,
}

struct DerivationSample<'a> {
    rows: Vec<SampleRow<'a>>,
    events: Vec<FieldEvent>,
    observations: Vec<Vec<Observation>>,
}

impl<'a> DerivationSample<'a> {
    fn collect(input: &[&'a [u8]], jt: &JsonTreePlan, cache_limit: usize) -> Self {
        let mut sample = Self {
            rows: Vec::new(),
            events: Vec::new(),
            observations: vec![Vec::new(); jt.paths.len()],
        };
        let mut doc = JsonDoc::default();
        let mut active = vec![false; jt.paths.len()];
        let strided: Vec<_> = input
            .iter()
            .step_by(input.len().div_ceil(2048).max(1))
            .copied()
            .collect();
        for bytes in spread_order(strided.len()).map(|i| strided[i]) {
            if !doc.parse_root_object(bytes) {
                continue;
            }
            active.fill(false);
            let mut active_count = 0;
            let start = sample.events.len();
            let mut count = 0;
            let mut needed = false;
            walk_deriv_fields(0, 0, 0, bytes, &doc, jt, &mut |id, v| {
                let event = FieldEvent {
                    path: id as u16,
                    start: v.start,
                    end: v.end,
                };
                if sample.events.len() < cache_limit {
                    sample.events.push(event);
                }
                let path = &mut sample.observations[id];
                if path.len() < DERIV_MAX_OCC
                    && !jt.paths[id].is_text
                    && v.kind >= J_STRING
                    && active_count > usize::from(active[id])
                {
                    path.push(Observation {
                        row: sample.rows.len(),
                        event: count,
                    });
                    needed = true;
                }
                let len = v.end - v.start;
                let usable = len > 0 && len <= 256;
                active_count = active_count - usize::from(active[id]) + usize::from(usable);
                active[id] = usable;
                count += 1;
            });
            let cached = if needed && count == sample.events.len() - start {
                Some(start..sample.events.len())
            } else {
                sample.events.truncate(start);
                None
            };
            if needed {
                sample.rows.push(SampleRow { bytes, cached });
            }
        }
        sample
    }

    fn expand(&self, observations: &[Observation], jt: &JsonTreePlan) -> Vec<ScopeOcc<'a>> {
        let mut list = Vec::with_capacity(observations.len());
        let mut scope = FieldScope::new(jt.paths.len());
        let mut doc = JsonDoc::default();
        for group in observations.chunk_by(|a, b| a.row == b.row) {
            let row = &self.rows[group[0].row];
            scope.present.fill(0);
            if let Some(range) = &row.cached {
                let events = &self.events[range.clone()];
                let mut start = 0;
                for obs in group {
                    for &event in &events[start..obs.event] {
                        scope.set(event);
                    }
                    list.push(scope.snapshot(events[obs.event], row.bytes));
                    start = obs.event;
                }
            } else {
                // This row was validated during collection.
                let valid = doc.parse_root_object(row.bytes);
                debug_assert!(valid);
                let mut wanted = group.iter().peekable();
                let mut pos = 0;
                walk_deriv_fields(0, 0, 0, row.bytes, &doc, jt, &mut |path, v| {
                    let event = FieldEvent {
                        path: path as u16,
                        start: v.start,
                        end: v.end,
                    };
                    if wanted.next_if(|obs| obs.event == pos).is_some() {
                        list.push(scope.snapshot(event, row.bytes));
                    }
                    scope.set(event);
                    pos += 1;
                });
            }
        }
        list
    }
}

// Keep only the paths present in this row when materializing a predecessor list.
// Iterating the set bits preserves the path order used to break ties in training.
struct FieldScope {
    present: Vec<u64>,
    values: Vec<(u32, u32)>,
}

impl FieldScope {
    fn new(paths: usize) -> Self {
        Self {
            present: vec![0; paths.div_ceil(64)],
            values: vec![(0, 0); paths],
        }
    }

    fn set(&mut self, event: FieldEvent) {
        let id = event.path as usize;
        let mask = 1u64 << (id % 64);
        let word = &mut self.present[id / 64];
        if (1..=256).contains(&(event.end - event.start)) {
            *word |= mask;
            self.values[id] = (event.start, event.end);
        } else {
            *word &= !mask;
        }
    }

    fn snapshot<'a>(&self, target: FieldEvent, row: &'a [u8]) -> ScopeOcc<'a> {
        let count: usize = self
            .present
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum();
        let mut refs = Vec::with_capacity(count);
        for (block, &word) in self.present.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let id = 64 * block + remaining.trailing_zeros() as usize;
                if id != target.path as usize {
                    let (start, end) = self.values[id];
                    refs.push((id as u16, &row[start as usize..end as usize]));
                }
                remaining &= remaining - 1;
            }
        }
        ScopeOcc {
            target: target.value(row),
            refs,
        }
    }
}

fn discover_json_derivations(json_rows: &[&[u8]], jt: &mut JsonTreePlan) {
    let sample = DerivationSample::collect(json_rows, jt, DERIV_CACHE_EVENTS);
    let mut tmp = Vec::new();
    for (p, observations) in sample.observations.iter().enumerate().skip(1) {
        if observations.len() < DERIV_MIN_OCC {
            continue;
        }
        let list = sample.expand(observations, jt);
        let props = propose_cands(&list, p, &mut tmp);
        let cands = pick_cands(&list, p, props, &mut tmp);
        if !cands.is_empty() {
            let sel_s = jt.n_struct_streams as u16;
            jt.n_struct_streams += 1;
            jt.paths[p].sel_stream = sel_s;
            jt.paths[p].cands = cands;
        }
    }
}

/// Candidates for path `own` read off its sampled values, most frequent first: copies of a value
/// in scope, templates over values in scope (also under Base64), and templates that differ
/// between two values in one literal only, with the part that differs left free.
fn propose_cands(list: &[ScopeOcc<'_>], own: usize, tmp: &mut Vec<u8>) -> Vec<JsonCand> {
    let mut counts: HashMap<JsonCand, usize> = HashMap::new();
    let mut by_refs: HashMap<Vec<u16>, Vec<Vec<Vec<u8>>>> = HashMap::new();
    let mut add = |c: JsonCand| *counts.entry(c).or_insert(0) += 1;
    for oc in list {
        // A `null`, `true` or `false` costs its column next to nothing.
        let literal = matches!(oc.target.first(), Some(b'n' | b't' | b'f'));
        for &(rp, rv) in oc.refs.iter().filter(|_| !literal) {
            if rv == oc.target {
                add(JsonCand::Copy { ref_path: rp });
            }
        }
        let t = oc.target;
        if t.len() < 6 || t[0] != b'"' || t[t.len() - 1] != b'"' {
            continue;
        }
        if let Some((lits, refs)) = decompose_target(t, &oc.refs) {
            let seen = by_refs.entry(refs.clone()).or_default();
            if seen.len() < DERIV_PAIR_SAMPLE {
                seen.push(lits.clone());
            }
            add(JsonCand::MultiConcat {
                lits,
                refs,
                b64: false,
            });
        }
        tmp.clear();
        if crate::lex::parse_base64(&t[1..t.len() - 1], tmp).is_some()
            && b64_text_ok(tmp)
            && let Some((lits, refs)) = decompose_target(tmp, &oc.refs)
        {
            add(JsonCand::MultiConcat {
                lits,
                refs,
                b64: true,
            });
        }
    }
    for (refs, seen) in &by_refs {
        for (i, a) in seen.iter().enumerate() {
            for b in &seen[i + 1..] {
                if let Some(c) = free_template(a, b, refs, own) {
                    add(c);
                }
            }
        }
    }
    let mut ranked: Vec<(JsonCand, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
        .into_iter()
        .take(DERIV_PROPOSALS)
        .map(|(c, _)| c)
        .collect()
}

/// The template two decompositions over the same refs share when they differ in one literal:
/// that literal becomes its common prefix, a free part, and its common suffix.
fn free_template(a: &[Vec<u8>], b: &[Vec<u8>], refs: &[u16], own: usize) -> Option<JsonCand> {
    let mut diff = (0..a.len()).filter(|&j| a[j] != b[j]);
    let j = diff.next()?;
    if diff.next().is_some() || refs.len() >= DERIV_MAX_REFS + 1 {
        return None;
    }
    let (x, y) = (&a[j], &b[j]);
    // The free part holds whole runs of letters or digits: `/issues/1` + `234` would make the
    // literal fit only numbers that start with 1.
    let class = |c: u8| (c.is_ascii_alphabetic() as u8) | ((c.is_ascii_digit() as u8) << 1);
    let cuts = |s: &[u8], i: usize| i == 0 || i == s.len() || class(s[i - 1]) & class(s[i]) == 0;
    let mut pre = x.iter().zip(y).take_while(|(p, q)| p == q).count();
    while !(cuts(x, pre) && cuts(y, pre)) {
        pre -= 1;
    }
    let mut suf = x[pre..]
        .iter()
        .rev()
        .zip(y[pre..].iter().rev())
        .take_while(|(p, q)| p == q)
        .count();
    while !(cuts(x, x.len() - suf) && cuts(y, y.len() - suf)) {
        suf -= 1;
    }
    // A free part right against a ref would leave no text to tell where one ends.
    if (pre == 0 && j > 0) || (suf == 0 && j < refs.len()) {
        return None;
    }
    let mut lits = a.to_vec();
    lits[j] = x[..pre].to_vec();
    lits.insert(j + 1, x[x.len() - suf..].to_vec());
    let mut refs = refs.to_vec();
    refs.insert(j, own as u16);
    Some(JsonCand::MultiConcat {
        lits,
        refs,
        b64: false,
    })
}

/// Greedily keeps the proposal that saves the most bytes on the values no kept one fits, while it
/// fits enough of them. Order is precedence: the encoder takes the first that fits.
fn pick_cands(
    list: &[ScopeOcc<'_>],
    own: usize,
    props: Vec<JsonCand>,
    tmp: &mut Vec<u8>,
) -> Vec<JsonCand> {
    // Bytes each proposal saves on each value: the value less its free part, or 0 if it misfits.
    let gains: Vec<Vec<usize>> = props
        .iter()
        .map(|c| {
            list.iter()
                .map(|oc| match cand_fit(c, own, oc.target, |r| oc.val(r), tmp) {
                    Some(free) => oc.target.len() - free.map_or(0, |(s, e)| e - s),
                    None => 0,
                })
                .collect()
        })
        .collect();
    // Bytes saved on each value by the best candidate kept so far.
    let mut best = vec![0usize; list.len()];
    let mut kept = Vec::new();
    while kept.len() < MAX_CANDS_PER_PATH {
        let score = |g: &Vec<usize>| {
            let more = g.iter().zip(&best).filter(|&(&x, &b)| x > b);
            more.fold((0usize, 0usize), |(n, s), (&x, &b)| (n + 1, s + x - b))
        };
        let top = (0..props.len())
            .filter(|i| !kept.contains(i))
            .map(|i| (score(&gains[i]), i))
            .max_by(|a, b| (a.0).1.cmp(&(b.0).1).then_with(|| b.1.cmp(&a.1)));
        let Some(((hits, saved), i)) = top else {
            break;
        };
        if hits < DERIV_MIN_HITS || hits * DERIV_MIN_SHARE < list.len() || saved < 32 {
            break;
        }
        kept.push(i);
        for (b, &g) in best.iter_mut().zip(&gains[i]) {
            *b = (*b).max(g);
        }
    }
    // The encoder takes the first that fits, so a candidate with no free part goes before one
    // with, and a free part with more literal text around it before one with less.
    let free_rank = |c: &JsonCand| match c {
        JsonCand::MultiConcat { lits, .. } if c.has_free(own) => {
            usize::MAX - lits.iter().map(Vec::len).sum::<usize>()
        }
        _ => 0,
    };
    kept.sort_by_key(|&i| free_rank(&props[i]));
    let mut out = Vec::new();
    for i in kept {
        out.push(match &props[i] {
            // One ref and no free part is an affix, which the plan writes shorter.
            JsonCand::MultiConcat { lits, refs, b64 } if !b64 && refs.len() == 1 && refs[0] as usize != own => {
                JsonCand::Affix {
                    prefix: lits[0].clone(),
                    ref_path: refs[0],
                    suffix: lits[1].clone(),
                }
            }
            c => c.clone(),
        });
    }
    out
}

/// Splits `t` into literals and the in-scope values it contains: longest first, the latest path on
/// a tie, at most `DERIV_MAX_REFS`.
fn decompose_target(t: &[u8], scope: &[(u16, &[u8])]) -> Option<(Vec<Vec<u8>>, Vec<u16>)> {
    if t.len() > 384 {
        return None;
    }
    // Most in-scope strings cannot occur in the target. Index its three-byte
    // prefixes once and verify only the positions that share a bucket.
    let bucket = |s: &[u8]| usize::from(s[0].wrapping_mul(13) ^ s[1].rotate_left(3) ^ s[2]);
    let mut head = [0u16; 256];
    let mut next = [0u16; 384];
    for (pos, prefix) in t.windows(3).enumerate().rev() {
        let h = bucket(prefix);
        next[pos] = head[h];
        head[h] = pos as u16 + 1;
    }
    let mut spans: Vec<(usize, usize, u16)> = Vec::new();
    for &(rp, raw_rv) in scope {
        let rv = unquote_json_lexeme(raw_rv);
        if rv.len() < DERIV_MIN_REF || rv.len() >= t.len() {
            continue;
        }
        let mut link = head[bucket(rv)];
        let mut end = 0;
        while link != 0 {
            let pos = usize::from(link - 1);
            if pos >= end && t[pos..].starts_with(rv) {
                end = pos + rv.len();
                spans.push((pos, end, rp));
            }
            link = next[pos];
        }
    }
    if spans.is_empty() {
        return None;
    }
    spans.sort_by(|a, b| {
        (b.1 - b.0)
            .cmp(&(a.1 - a.0))
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| b.2.cmp(&a.2))
    });
    let mut chosen: Vec<(usize, usize, u16)> = Vec::new();
    for sp in spans {
        if chosen.len() >= DERIV_MAX_REFS {
            break;
        }
        if chosen.iter().all(|c| sp.1 <= c.0 || sp.0 >= c.1) {
            chosen.push(sp);
        }
    }
    chosen.sort_by_key(|c| c.0);

    let mut lits = Vec::with_capacity(chosen.len() + 1);
    let mut refs = Vec::with_capacity(chosen.len());
    let mut pos = 0usize;
    let mut ref_bytes = 0usize;
    for (s, e, rp) in chosen {
        let lit = &t[pos..s];
        if lit.len() > DERIV_MAX_AFFIX {
            return None;
        }
        lits.push(lit.to_vec());
        refs.push(rp);
        ref_bytes += e - s;
        pos = e;
    }
    let tail = &t[pos..];
    if tail.len() > DERIV_MAX_AFFIX || ref_bytes < 4 {
        return None;
    }
    lits.push(tail.to_vec());
    Some((lits, refs))
}

fn walk_deriv_fields(
    ni: usize,
    path_idx: usize,
    trie_node: usize,
    s: &[u8],
    doc: &JsonDoc,
    jt: &JsonTreePlan,
    visit: &mut impl FnMut(usize, JVal),
) {
    let pdef = &jt.paths[path_idx];
    let v = doc.nodes[ni];
    if pdef.shapes.is_empty() {
        visit(path_idx, v);
        return;
    }
    if v.kind != J_OBJECT {
        return;
    }
    for k in doc.kid_range(ni) {
        let kid = doc.kids[k];
        let raw_k = &s[kid.key_start as usize..kid.key_end as usize];
        let ct_opt = if kid.key_esc {
            decode_json_key(raw_k).and_then(|d| jt.runtime.trie[trie_node].kids.get(&d).copied())
        } else {
            jt.runtime.trie[trie_node].kids.get(raw_k).copied()
        };
        let Some(ct) = ct_opt else {
            continue;
        };
        let pid = jt.runtime.trie[ct].path_idx;
        if pid < 0 {
            continue;
        }
        let cp = pid as usize;
        let cv = doc.nodes[kid.node as usize];
        if jt.paths[cp].elem_path != u16::MAX {
            if cv.kind == J_ARRAY && cv.n > 0 {
                let elem_p = jt.paths[cp].elem_path as usize;
                let elem_ct = jt.runtime.trie[ct].elem_node.max(0) as usize;
                for ek in doc.kid_range(kid.node as usize) {
                    walk_deriv_fields(
                        doc.kids[ek].node as usize,
                        elem_p,
                        elem_ct,
                        s,
                        doc,
                        jt,
                        visit,
                    );
                }
            }
        } else {
            walk_deriv_fields(kid.node as usize, cp, ct, s, doc, jt, visit);
        }
    }
}

fn discover_nested_json_strings(
    json_rows: &[&[u8]],
    opts: &crate::plan::TrainOptions,
    jt: &JsonTreePlan,
    model_plan: &mut crate::grammar::ModelPlan,
) {
    let mut doc = JsonDoc::default();
    let mut string_samples: HashMap<u16, Vec<Vec<u8>>> = HashMap::new();

    for &row in json_rows.iter().step_by(json_rows.len().div_ceil(512).max(1)) {
        if !doc.parse_root_object(row) {
            continue;
        }
        collect_nested_string_samples(0, 0, row, &doc, jt, &mut string_samples);
    }

    for (cid, unesc_vec) in string_samples {
        if unesc_vec.len() < 8 {
            continue;
        }
        let structured_hits = unesc_vec
            .iter()
            .filter(|s| {
                let t = s.trim_ascii();
                (t.starts_with(b"{") && t.ends_with(b"}"))
                    || crate::grammar::parse_logfmt_line(t).is_some()
            })
            .count();
        if structured_hits * 2 >= unesc_vec.len() {
            let refs: Vec<&[u8]> = unesc_vec.iter().map(|v| v.as_slice()).collect();
            let mut child_opts = opts.clone();
            child_opts.depth = 1;
            child_opts.max_json_paths = 64;
            child_opts.max_templates = 32;
            let child = crate::plan::train_from_records(
                &refs,
                crate::framing::ResolvedFraming::Lf { open_end: false },
                &child_opts,
                None,
            );
            if child.variants.len() > 1 {
                model_plan.columns[cid as usize].nested_plan = Some(Box::new(child));
            }
        }
    }
}

fn collect_nested_string_samples(
    ni: usize,
    trie_node: usize,
    s: &[u8],
    doc: &JsonDoc,
    jt: &JsonTreePlan,
    samples: &mut HashMap<u16, Vec<Vec<u8>>>,
) {
    let v = doc.nodes[ni];
    if v.kind != J_OBJECT {
        return;
    }
    for k in doc.kid_range(ni) {
        let kid = doc.kids[k];
        let raw_k = &s[kid.key_start as usize..kid.key_end as usize];
        let Some(&ct) = jt.runtime.trie[trie_node].kids.get(raw_k) else {
            continue;
        };
        let pid = jt.runtime.trie[ct].path_idx;
        if pid < 0 {
            continue;
        }
        let cp = pid as usize;
        let pdef = &jt.paths[cp];
        let cv = doc.nodes[kid.node as usize];
        if !pdef.shapes.is_empty() && cv.kind == J_OBJECT {
            collect_nested_string_samples(kid.node as usize, ct, s, doc, jt, samples);
        } else if pdef.shapes.is_empty()
            && pdef.cands.is_empty()
            && pdef.value_col != u16::MAX
            && cv.kind == J_STRING
        {
            let inner = &s[cv.start as usize + 1..cv.end as usize - 1];
            if inner.len() >= 16
                && inner.len() <= 8192
                && (inner.contains(&b'\\') || inner.contains(&b'='))
            {
                let e = samples.entry(pdef.value_col).or_default();
                if e.len() < 256
                    && let Some(u) = unescape_json_str_exact(inner)
                {
                    e.push(u);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records with a Base64 id of a type and the sibling `id`, an avatar with one varying digit,
    /// and a url whose tail no earlier field holds. Some rows fit none of these.
    fn derivable_rows() -> Vec<String> {
        (0..120u64)
            .map(|i| {
                let id = 1000 + i * 7919 % 90_000;
                let login = format!("dev{}", i % 23);
                let kind = if i % 5 == 0 { "Bot" } else { "User" };
                let mut node = Vec::new();
                let text = format!("0{}:{kind}{id}", kind.len());
                crate::lex::format_base64(text.as_bytes(), &mut node);
                let node = match i % 17 {
                    3 => format!("U_kgDO{i:06}"),
                    _ => String::from_utf8(node).unwrap(),
                };
                let avatar = match i % 19 {
                    7 => format!("https://cdn.test/x/{i}.png"),
                    _ => format!("https://img{}.cdn.test/u/{id}?v=4", i % 4),
                };
                let n = 1 + i * 31 % 5000;
                let url = match i % 13 {
                    2 => format!("https://api.test/misc/{i}"),
                    _ => format!("https://api.test/repos/{login}/issues/{n}"),
                };
                format!(
                    "{{\"login\":\"{login}\",\"id\":{id},\"node_id\":\"{node}\",\"avatar_url\":\"{avatar}\",\"url\":\"{url}\",\"number\":{n}}}"
                )
            })
            .collect()
    }

    #[test]
    fn predecessor_snapshots_keep_only_live_paths_in_order() {
        let row = [b'x'; 260];
        let mut scope = FieldScope::new(100);
        let event = |path, start, end| FieldEvent { path, start, end };
        scope.set(event(71, 1, 2));
        scope.set(event(64, 0, 1));
        scope.set(event(1, 2, 3));
        scope.set(event(0, 0, 256));
        scope.set(event(71, 0, 257));
        let snapshot = scope.snapshot(event(64, 3, 4), &row);
        assert_eq!(snapshot.target, &row[3..4]);
        assert_eq!(snapshot.refs, [(0, &row[..256]), (1, &row[2..3])]);
    }

    #[test]
    fn replay_from_input_matches_cached_field_history() {
        let rows: Vec<_> = (0..32).map(|i| format!(
            r#"{{"id":{i},"items":[{{"name":"a{i}","url":"a{i}"}},{{"name":"b{i}","url":"b{i}"}}],"last":"b{i}"}}"#
        )).collect();
        let refs: Vec<_> = rows.iter().map(|row| row.as_bytes()).collect();
        let mut model = crate::grammar::ModelPlan::residual_only();
        let jt = train_json_tree(&refs, &Default::default(), &mut model).unwrap();
        let cached = DerivationSample::collect(&refs, &jt, DERIV_CACHE_EVENTS);
        assert!(cached.rows.iter().all(|row| row.cached.is_some()));
        assert!(cached.observations.iter().any(|obs| obs.len() > rows.len()));
        for limit in [0, cached.events.len() / 2] {
            let sample = DerivationSample::collect(&refs, &jt, limit);
            assert!(sample.events.len() <= limit);
            assert!(sample.rows.iter().any(|row| row.cached.is_none()));
            for (a, b) in cached.observations.iter().zip(&sample.observations) {
                let left = cached.expand(a, &jt);
                let right = sample.expand(b, &jt);
                assert_eq!(left.len(), right.len());
                for (x, y) in left.iter().zip(right) {
                    assert_eq!((x.target, &x.refs), (y.target, &y.refs));
                }
            }
        }
    }

    #[test]
    fn trains_base64_and_free_part_candidates() {
        let rows = derivable_rows();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_bytes()).collect();
        let mut model = crate::grammar::ModelPlan::residual_only();
        let opts = crate::plan::TrainOptions::default();
        let jt = train_json_tree(&refs, &opts, &mut model).unwrap();
        let path = |name: &str| jt.paths.iter().position(|p| p.name == name).unwrap();
        let b64 = |c: &&JsonCand| matches!(c, JsonCand::MultiConcat { b64: true, .. });
        let node = &jt.paths[path("$.node_id")].cands;
        assert_eq!(node.iter().filter(b64).count(), 2, "{node:?}");
        for name in ["$.avatar_url", "$.url"] {
            let p = path(name);
            assert!(jt.paths[p].cands.iter().any(|c| c.has_free(p)), "{name}");
        }
        let mut buf = Vec::new();
        jt.encode(&mut buf);
        let back = JsonTreePlan::decode(&mut Cursor::new(&buf)).unwrap();
        assert!(back.paths.iter().zip(&jt.paths).all(|(a, b)| a.cands == b.cands));
    }

    #[test]
    fn derivations_judge_rows_from_the_whole_sample() {
        // `b` copies `a` in the first 8% of rows only: under the 1-in-10 bar overall, though
        // over it among the head's occurrences.
        let rows: Vec<String> = (0..3000u64)
            .map(|i| {
                let b = if i < 240 { 5000 + i } else { 90_000 + i * 7 };
                format!("{{\"a\":{},\"b\":{b}}}", 5000 + i)
            })
            .collect();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_bytes()).collect();
        let mut model = crate::grammar::ModelPlan::residual_only();
        let jt = train_json_tree(&refs, &crate::plan::TrainOptions::default(), &mut model).unwrap();
        let b = jt.paths.iter().find(|p| p.name == "$.b").unwrap();
        assert!(b.cands.is_empty(), "{:?}", b.cands);
        assert_eq!(spread_order(10).collect::<Vec<_>>(), [0, 7, 4, 1, 8, 5, 2, 9, 6, 3]);
    }

    #[test]
    fn refuses_a_free_part_it_cannot_place() {
        // A plan of one path, `$`, with one candidate: `tag` over `refs`.
        let plan = |tag: u8, refs: &[u64], lit: &[u8]| {
            let mut b = Vec::new();
            put_uvarint(&mut b, 1);
            put_uvarint(&mut b, 1);
            put_bytes(&mut b, b"$");
            b.push(0);
            put_bytes(&mut b, b"");
            for v in [0, u16::MAX as u64, 0, 0, u16::MAX as u64] {
                put_uvarint(&mut b, v);
            }
            put_uvarint(&mut b, 0);
            put_uvarint(&mut b, 1);
            b.push(tag);
            put_uvarint(&mut b, refs.len() as u64);
            for _ in 0..=refs.len() {
                put_bytes(&mut b, lit);
            }
            refs.iter().for_each(|&r| put_uvarint(&mut b, r));
            JsonTreePlan::decode(&mut Cursor::new(&b))
        };
        assert!(plan(2, &[0], b"x").is_ok());
        assert!(plan(2, &[0, 0], b"x").is_err(), "two free parts");
        assert!(plan(3, &[0], b"x").is_err(), "a free part under Base64");
        assert!(plan(3, &[1], b"x").is_err(), "a ref past the paths");
        let err = plan(3, &[0], b"\xff").err().map(|e| e.to_string());
        assert!(err.is_some_and(|e| e.contains("printable")), "Base64 of text not printable");
    }
}
