//! Predicates, and their evaluation on a raw record and on a block's columns.
//!
//! The raw record defines what a predicate means. A block answers from its columns for the rows
//! whose model makes that exact, and reports the rest as unknown for the caller to rebuild.

use crate::cells::{Col, NOT_HELD, NULL_CELL, underflow};
use crate::codecs::{ANY_LETTERS, ColSummary, ESCAPES, letters_of};
use crate::container::BlockRows;
use crate::grammar::{ModelPlan, PlanVariant, parse_logfmt_line};
use crate::index::{
    BlockDir, Filter, KIND_JSON, KIND_RESIDUAL, KIND_SHAPE, KIND_SHAPE_CELL, needle_tokens,
};
use crate::json_tree::{
    JsonBlockMap, JsonCand, JsonDoc, JsonTreePlan, JsonWalk, PathEvent, PathVal, SlotKind,
    WalkRoute, emit_nested_cell, json_values_at, map_json_block_structure,
};
use crate::lex::{parse_exact_int, parse_timestamp, unescape_json_str_exact};
use crate::wire::{DecodeBudget, Error, Result};
use memchr::memmem::Finder;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub enum Predicate {
    All,
    /// The record's bytes contain these bytes.
    Contains(Vec<u8>),
    /// The record contains these bytes with no ASCII letter, digit or `_` right before them or
    /// right after them.
    Word(Vec<u8>),
    FieldEq {
        field: String,
        value: Vec<u8>,
    },
    FieldIn {
        field: String,
        values: Vec<Vec<u8>>,
    },
    FieldContains {
        field: String,
        needle: Vec<u8>,
    },
    FieldIntRange {
        field: String,
        min: Option<i64>,
        max: Option<i64>,
    },
    FieldTimeRange {
        field: String,
        min_ns: Option<i64>,
        max_ns: Option<i64>,
    },
    FieldExists(String),
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Not(Box<Predicate>),
}

// ============================================================================
// Raw records
// ============================================================================

/// A field name as a path from the record's root, in the spelling `JsonPathDef::name` uses below
/// its `$`. Array steps are dropped: a path names the elements of any array on its way.
pub(crate) fn field_target(field: &str) -> Vec<u8> {
    let clean = field.strip_prefix('$').unwrap_or(field).replace("[]", "");
    let dot = if clean.starts_with(['.', '[']) {
        ""
    } else {
        "."
    };
    format!("{dot}{clean}").into_bytes()
}

/// What a value's bytes spell: a JSON string without its quotes and escapes, anything else as
/// written.
pub fn logical(lexeme: &[u8]) -> Cow<'_, [u8]> {
    if lexeme.len() < 2 || lexeme[0] != b'"' || lexeme[lexeme.len() - 1] != b'"' {
        return Cow::Borrowed(lexeme);
    }
    let inner = &lexeme[1..lexeme.len() - 1];
    if memchr::memchr(b'\\', inner).is_none() {
        return Cow::Borrowed(inner);
    }
    unescape_json_str_exact(inner).map_or(Cow::Borrowed(inner), Cow::Owned)
}

/// Calls `f` with each value of `field` in `rec`, as the record writes it: the values at that
/// path of a JSON record, or the value of that key of a logfmt line.
pub fn raw_field_values(rec: &[u8], field: &str, f: &mut dyn FnMut(&[u8])) {
    let target = field_target(field);
    let mut doc = JsonDoc::default();
    if json_values_at(&mut doc, rec, &target, &mut |s, e| f(&rec[s..e])) {
        return;
    }
    if let Some(pairs) = parse_logfmt_line(rec) {
        for p in pairs
            .iter()
            .filter(|p| target.strip_prefix(b".") == Some(p.key))
        {
            f(p.val_inner);
        }
    }
}

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Whether `text` holds the needle of `finder`, `len` bytes long. With `word` it must have no
/// word byte on either side; the ends of `text` count as none.
fn found(finder: &Finder<'_>, len: usize, word: bool, text: &[u8]) -> bool {
    let mut pos = 0;
    while let Some(p) = finder.find(&text[pos..]) {
        let at = pos + p;
        let open = |b: Option<&u8>| !b.is_some_and(|&b| is_word(b));
        if !word || len == 0 || (open(text[..at].last()) && open(text.get(at + len))) {
            return true;
        }
        pos = at + 1;
    }
    false
}

/// The test a field predicate applies to each value of its field.
pub(crate) enum Test<'a> {
    Eq(&'a [u8]),
    In(&'a [Vec<u8>]),
    Contains(&'a [u8]),
    Int(Option<i64>, Option<i64>),
    Time(Option<i64>, Option<i64>),
    Exists,
}

impl<'a> Test<'a> {
    pub fn of(pred: &'a Predicate) -> Option<(&'a str, Self)> {
        Some(match pred {
            Predicate::FieldEq { field, value } => (field.as_str(), Test::Eq(value)),
            Predicate::FieldIn { field, values } => (field.as_str(), Test::In(values)),
            Predicate::FieldContains { field, needle } => (field.as_str(), Test::Contains(needle)),
            Predicate::FieldIntRange { field, min, max } => (field.as_str(), Test::Int(*min, *max)),
            Predicate::FieldTimeRange {
                field,
                min_ns,
                max_ns,
            } => (field.as_str(), Test::Time(*min_ns, *max_ns)),
            Predicate::FieldExists(field) => (field.as_str(), Test::Exists),
            _ => return None,
        })
    }

    pub fn ok(&self, v: &[u8]) -> bool {
        let within = |x: i64, lo: &Option<i64>, hi: &Option<i64>| {
            lo.is_none_or(|lo| x >= lo) && hi.is_none_or(|hi| x <= hi)
        };
        match self {
            Test::Eq(want) => v == *want,
            Test::In(wants) => wants.iter().any(|w| w == v),
            Test::Contains(needle) => memchr::memmem::find(v, needle).is_some(),
            Test::Int(lo, hi) => parse_exact_int(v).is_some_and(|(x, _)| within(x, lo, hi)),
            Test::Time(lo, hi) => parse_timestamp(v).is_some_and(|(x, _)| within(x, lo, hi)),
            Test::Exists => true,
        }
    }

    /// False when no cell of a column with this summary, in a block with this filter, can pass.
    fn may_hold(&self, s: &ColSummary, filter: Filter<'_>) -> bool {
        // Without escapes a cell spells its value, so the value's letters and tokens are the
        // record's.
        let tokens_ok = |text: &[u8], whole: bool| {
            s.letters & ESCAPES != 0
                || (letters_of(text) & !s.letters == 0
                    && needle_tokens(text, whole)
                        .is_none_or(|ts| ts.iter().all(|&t| filter.has(t))))
        };
        let eq = |want: &[u8]| {
            let in_bounds = s.numeric_kind != 1
                || parse_exact_int(want).is_none_or(|(x, _)| x >= s.min_val && x <= s.max_val);
            in_bounds && tokens_ok(want, true)
        };
        match self {
            Test::Eq(want) => eq(want),
            Test::In(wants) => wants.iter().any(|w| eq(w)),
            Test::Contains(needle) => tokens_ok(needle, false),
            Test::Int(lo, hi) => {
                s.numeric_kind != 1
                    || !(lo.is_some_and(|lo| s.max_val < lo) || hi.is_some_and(|hi| s.min_val > hi))
            }
            Test::Time(lo, hi) => {
                s.numeric_kind != 2
                    || !(lo.is_some_and(|lo| s.max_val < lo) || hi.is_some_and(|hi| s.min_val > hi))
            }
            Test::Exists => true,
        }
    }
}

/// Evaluates `pred` on one record. A field test holds when any value of the field passes it.
pub fn eval_raw_record(rec: &[u8], pred: &Predicate) -> bool {
    match pred {
        Predicate::All => true,
        Predicate::Contains(needle) => memchr::memmem::find(rec, needle).is_some(),
        Predicate::Word(needle) => found(&Finder::new(needle), needle.len(), true, rec),
        Predicate::And(v) => v.iter().all(|p| eval_raw_record(rec, p)),
        Predicate::Or(v) => v.iter().any(|p| eval_raw_record(rec, p)),
        Predicate::Not(inner) => !eval_raw_record(rec, inner),
        _ => {
            let (field, test) = Test::of(pred).unwrap();
            let mut hit = false;
            raw_field_values(rec, field, &mut |v| hit = hit || test.ok(&logical(v)));
            hit
        }
    }
}

// ============================================================================
// What a model says about a field or a needle
// ============================================================================

const OPAQUE: u32 = u32::MAX;

/// How to find a field's values in the JSON rows of blocks under one model.
pub(crate) struct FieldPlan {
    /// By path, for `JsonWalk`.
    track: Vec<bool>,
    hooks: Vec<Vec<Vec<(u16, u32)>>>,
    /// By path: where the field sits below a value stored for that path. `Some` for the paths that
    /// spell the field and for those above it.
    rem: Vec<Option<Vec<u8>>>,
    /// Hook payloads: values a shape's literal text holds.
    consts: Vec<Vec<u8>>,
    /// Columns a walk reads cells from.
    pub cols: Vec<u16>,
    /// Columns of the scalar paths that spell the field.
    leaf_cols: Vec<u16>,
    /// The `leaf_cols` whose every cell is a value of the field.
    own_cols: Vec<u16>,
    /// Columns that may hold a raw subtree with the field inside, each with the letters of the
    /// keys that lead to the field from there.
    subtree_cols: Vec<(u16, u32)>,
    /// Columns of the paths a leaf copies values from: a leaf value with no cell in `leaf_cols`
    /// is one of theirs.
    copy_cols: Vec<u16>,
    /// Every value comes from `leaf_cols`, `copy_cols` or `subtree_cols`.
    prunable: bool,
    /// By structure stream: a walk reads it.
    streams: Vec<bool>,
    route: WalkRoute,
}

impl FieldPlan {
    fn build(plan: &ModelPlan, field: &str) -> Self {
        let n = plan.json_tree.as_ref().map_or(0, |jt| jt.paths.len());
        let mut fp = FieldPlan {
            track: vec![false; n],
            hooks: vec![Vec::new(); n],
            rem: vec![None; n],
            consts: Vec::new(),
            cols: Vec::new(),
            leaf_cols: Vec::new(),
            own_cols: Vec::new(),
            subtree_cols: Vec::new(),
            copy_cols: Vec::new(),
            prunable: true,
            streams: Vec::new(),
            route: WalkRoute::default(),
        };
        let Some(jt) = &plan.json_tree else {
            return fp;
        };
        let target = field_target(field);
        for (pi, p) in jt.paths.iter().enumerate() {
            let name = p
                .name
                .strip_prefix('$')
                .unwrap_or(&p.name)
                .replace("[]", "");
            let name = name.as_bytes();
            let above = matches!(target.get(name.len()), None | Some(b'.' | b'['));
            if above && target.starts_with(name) {
                fp.rem[pi] = Some(target[name.len()..].to_vec());
            }
        }
        for pi in 0..n {
            let Some(rem) = fp.rem[pi].clone() else {
                continue;
            };
            let p = &jt.paths[pi];
            fp.add_tracked(jt, pi);
            if p.shapes.is_empty() && rem.is_empty() {
                fp.prunable &= fp.add_copy_cols(jt, pi, &mut vec![false; n]);
                fp.leaf_cols
                    .extend((p.value_col != u16::MAX).then_some(p.value_col));
            } else {
                let keys = Some(letters_of(&rem))
                    .filter(|l| l & ESCAPES == 0)
                    .unwrap_or(0);
                fp.subtree_cols
                    .extend((p.value_col != u16::MAX).then_some((p.value_col, keys)));
            }
            fp.hooks[pi] = p
                .shapes
                .iter()
                .map(|shape| fp.shape_hooks(shape, &rem))
                .collect();
            fp.prunable &= fp.hooks[pi].iter().all(|h| h.is_empty());
        }
        fp.cols.sort_unstable();
        fp.cols.dedup();
        let own = |&c: &u16| {
            plan.columns[c as usize].nested_plan.is_none()
                && jt.paths.iter().filter(|p| p.value_col == c).count() == 1
        };
        fp.own_cols = fp.leaf_cols.iter().copied().filter(own).collect();
        fp.route = WalkRoute::of(
            jt,
            &JsonWalk {
                want_cols: &[],
                track: &fp.track,
                hooks: &fp.hooks,
            },
        );
        fp.streams = fp.route.streams(jt);
        fp
    }

    /// A plan that walks to the values of `paths` and of the paths they derive from, and reports
    /// `hooks`.
    fn of_paths(jt: &JsonTreePlan, paths: &[usize], hooks: Vec<Vec<Vec<(u16, u32)>>>) -> Self {
        let n = jt.paths.len();
        let mut fp = FieldPlan {
            track: vec![false; n],
            hooks,
            rem: vec![None; n],
            consts: Vec::new(),
            cols: Vec::new(),
            leaf_cols: Vec::new(),
            own_cols: Vec::new(),
            subtree_cols: Vec::new(),
            copy_cols: Vec::new(),
            prunable: false,
            streams: Vec::new(),
            route: WalkRoute::default(),
        };
        paths.iter().for_each(|&pi| fp.add_tracked(jt, pi));
        fp.cols.sort_unstable();
        fp.cols.dedup();
        fp.route = WalkRoute::of(
            jt,
            &JsonWalk {
                track: &fp.track,
                hooks: &fp.hooks,
                ..JsonWalk::default()
            },
        );
        fp.streams = fp.route.streams(jt);
        fp
    }

    /// Adds to `copy_cols` the columns path `pi` copies its values from, and theirs in turn.
    /// False when it derives values any other way.
    fn add_copy_cols(&mut self, jt: &JsonTreePlan, pi: usize, seen: &mut [bool]) -> bool {
        if std::mem::replace(&mut seen[pi], true) {
            return true;
        }
        jt.paths[pi].cands.iter().all(|c| {
            let &JsonCand::Copy { ref_path } = c else {
                return false;
            };
            let r = &jt.paths[ref_path as usize];
            self.copy_cols
                .extend((r.value_col != u16::MAX).then_some(r.value_col));
            r.shapes.is_empty() && self.add_copy_cols(jt, ref_path as usize, seen)
        })
    }

    /// Whether a block's leaf values may come from other columns than `leaf_cols`. A copy takes a
    /// value of its source, so with no source cell there is none.
    fn borrows(&self, dir: &BlockDir) -> bool {
        let live = |c: u16| !dir.col(c as usize).summary.all_null;
        self.subtree_cols.iter().any(|&(c, _)| live(c)) || self.copy_cols.iter().any(|&c| live(c))
    }

    /// The column whose cell `i` is the one value the field has in row `i`, in a block where
    /// that holds for every row.
    fn row_col(&self, kinds: u16, dir: &BlockDir) -> Option<u16> {
        let live = |c: u16| !dir.col(c as usize).summary.all_null;
        if !self.prunable || kinds != KIND_JSON || self.borrows(dir) {
            return None;
        }
        let mut cols = self.leaf_cols.iter().copied().filter(|&c| live(c));
        let (c, None) = (cols.next()?, cols.next()) else {
            return None;
        };
        (dir.col(c as usize).summary.per_row && self.own_cols.contains(&c)).then_some(c)
    }

    /// Adds what `field_events` reads of a block to `need`.
    pub fn needs(&self, kinds: u16, dir: &BlockDir, need: &mut Need) {
        if self.cols.iter().any(|&c| dir.col(c as usize).deps.pool) {
            need.pages = None;
        }
        if let Some(c) = self.row_col(kinds, dir) {
            need.cols[c as usize] = true;
            return;
        }
        for &c in &self.cols {
            need.cols[c as usize] |= !dir.col(c as usize).summary.all_null;
        }
        need.streams
            .iter_mut()
            .zip(&self.streams)
            .for_each(|(n, s)| *n |= s);
    }

    /// Tracks a path and the paths its values derive from.
    fn add_tracked(&mut self, jt: &JsonTreePlan, pi: usize) {
        if std::mem::replace(&mut self.track[pi], true) {
            return;
        }
        let p = &jt.paths[pi];
        self.cols
            .extend((p.value_col != u16::MAX).then_some(p.value_col));
        for cand in &p.cands {
            match cand {
                JsonCand::Copy { ref_path } | JsonCand::Affix { ref_path, .. } => {
                    self.add_tracked(jt, *ref_path as usize)
                }
                JsonCand::MultiConcat { refs, .. } => {
                    refs.iter().for_each(|&r| self.add_tracked(jt, r as usize))
                }
            }
        }
    }

    /// The field's values written in a shape's literal text, each placed before the slot that
    /// follows it. A value made of both literal text and slots is `OPAQUE`.
    fn shape_hooks(
        &mut self,
        shape: &[crate::json_tree::JsonSlotPart],
        rem: &[u8],
    ) -> Vec<(u16, u32)> {
        // The shape's text with `0` standing for each slot and nothing for an array's elements.
        let mut text = Vec::new();
        let mut marks = Vec::<(usize, u16, bool)>::new();
        for (i, part) in shape.iter().enumerate() {
            text.extend_from_slice(&part.literal);
            match part.kind {
                SlotKind::None => {}
                SlotKind::Array => marks.push((text.len(), i as u16, false)),
                _ => {
                    marks.push((text.len(), i as u16, true));
                    text.push(b'0');
                }
            }
        }
        let mut hooks = Vec::new();
        let mut doc = JsonDoc::default();
        let parsed = json_values_at(&mut doc, &text, rem, &mut |s, e| {
            let inside = marks.iter().find(|m| m.0 >= s && m.0 < e);
            let part = marks
                .iter()
                .find(|m| m.0 >= s)
                .map_or(shape.len() as u16, |m| m.1);
            match inside {
                // The slot is the value: its own path reports it, if this plan tracks that path.
                Some(&(at, part, true)) if at == s => {
                    if self.rem[shape[part as usize].path_idx as usize].is_none() {
                        hooks.push((part, OPAQUE));
                    }
                }
                Some(_) => hooks.push((part, OPAQUE)),
                None => {
                    hooks.push((part, self.consts.len() as u32));
                    self.consts.push(text[s..e].to_vec());
                }
            }
        });
        if !parsed {
            hooks.push((0, OPAQUE));
        }
        hooks
    }
}

fn b64_digit(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// How `needle` can lie inside the Base64 spelling of the printable ASCII text a Base64 candidate
/// holds. `None` when it cannot: at none of the four offsets into a 4-character group does every
/// byte the needle's bits touch agree with a printable byte. Otherwise, by offset where it can,
/// the bytes the needle spells whole, one of which the text then holds; empty when an offset
/// spells none, or the needle has padding, so no text can be ruled out.
fn b64_cores(needle: &[u8]) -> Option<Vec<Vec<u8>>> {
    // Padding ends a value; such a needle is rare enough to keep without a test.
    if needle.is_empty() || needle.contains(&b'=') {
        return needle.iter().all(|&c| c == b'=' || b64_digit(c).is_some()).then(Vec::new);
    }
    let v: Vec<u8> = needle.iter().map(|&c| b64_digit(c)).collect::<Option<_>>()?;
    let (mut cores, mut screens) = (Vec::new(), true);
    for a in 0..4 {
        let (lo, hi) = (6 * a, 6 * (a + v.len()));
        let last = hi.div_ceil(8) - 1;
        let mut core = Vec::new();
        let fits = (lo / 8..=last).all(|b| {
            let (mut mask, mut val) = (0u8, 0u8);
            for i in (0..8).filter(|i| (lo..hi).contains(&(8 * b + i))) {
                let g = 8 * b + i - lo;
                mask |= 0x80 >> i;
                val |= ((v[g / 6] >> (5 - g % 6)) & 1) << (7 - i);
            }
            if mask == 0xff {
                core.push(val);
            }
            // Zero bits that end the needle may be the padding after a value's last byte.
            (b == last && hi % 8 != 0 && val == 0) || (0x20..0x7fu8).any(|x| x & mask == val)
        });
        if fits {
            screens &= !core.is_empty();
            cores.push(core);
        }
    }
    match (cores.is_empty(), screens) {
        (true, _) => None,
        (false, true) => Some(cores),
        (false, false) => Some(Vec::new()),
    }
}

/// What a variant's rows need to answer a substring test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VarNeedle {
    /// Every row holds the needle in its literal text.
    AllTrue,
    /// The needle may lie across literal text and a value, so the row has to be rebuilt.
    Unknown,
    /// The needle is in the row exactly when it is in one of the row's cells.
    Scan,
}

pub(crate) struct NeedlePlan {
    vars: Vec<VarNeedle>,
    /// `KIND_*` bits of the variants that are `Unknown`.
    unknown_kinds: u16,
    /// By column: its cells appear in records verbatim and one variant alone writes it.
    scan_cols: Vec<bool>,
    /// `KIND_*` bits of the variants whose rows can match with the needle in no cell.
    open_kinds: u16,
    /// The needle's letters, as `ColSummary::letters` holds them.
    letters: u32,
    finder: Finder<'static>,
    /// The needle must stand as a word.
    word: bool,
    /// The needle can match a `"` or the bare token `null`, which a quoted column adds to cells.
    quote_sensitive: bool,
    /// When JSON rows are scanned and the needle can lie outside every cell, in a derived value
    /// or in a shape's text: the walk that finds those rows.
    derived: Option<DerivedWalk>,
}

/// JSON paths whose values a substring test rebuilds, and shapes whose text holds the needle.
struct DerivedWalk {
    /// By path, by shape: fires on rows that take a shape holding the needle.
    hooks: Vec<Vec<Vec<(u16, u32)>>>,
    /// By path, by candidate: the values it derives are rebuilt.
    cands: Vec<Vec<bool>>,
    /// By path: every value is rebuilt, and its column is not scanned.
    every: Vec<bool>,
    /// `b64_cores` of the needle.
    b64_cores: Vec<Finder<'static>>,
    /// By the letters a block's path columns hold: its walk, built once.
    walks: Mutex<HashMap<Vec<u32>, Option<Arc<FieldPlan>>>>,
}

impl DerivedWalk {
    /// The walk for one block: to the paths whose rebuilt values can hold the needle, or one of
    /// its Base64 cores, in a way no cell does, given the letters the block's columns hold.
    fn for_block(&self, plan: &ModelPlan, dir: &BlockDir, needle: &[u8]) -> Option<Arc<FieldPlan>> {
        let jt = plan.json_tree.as_ref()?;
        let col = |c: u16| match plan.columns.get(c as usize) {
            None => 0,
            Some(_) if dir.col(c as usize).summary.all_null => 0,
            Some(def) if def.nested_plan.is_some() => ANY_LETTERS,
            Some(_) => dir.col(c as usize).summary.letters,
        };
        let rebuilds = self.every.contains(&true) || self.cands.iter().flatten().any(|&c| c);
        // Nothing below reads the block but these letters.
        let key: Vec<u32> = match rebuilds {
            true => jt.paths.iter().map(|p| col(p.value_col)).collect(),
            false => Vec::new(),
        };
        let mut walks = self.walks.lock().unwrap();
        if let Some(fp) = walks.get(&key) {
            return fp.clone();
        }
        let mut paths = Vec::new();
        let mut memo = vec![None; key.len()];
        for (pi, p) in jt.paths.iter().enumerate().filter(|_| rebuilds) {
            let mut walk = self.every.get(pi) == Some(&true);
            for (c, &on) in p.cands.iter().zip(self.cands.get(pi).into_iter().flatten()) {
                if !on {
                    continue;
                }
                let (lits, refs) = cand_parts(c);
                let gaps: Vec<u32> = refs
                    .iter()
                    .map(|&r| path_letters(jt, &key, &mut memo, r as usize, pi, 0))
                    .collect();
                walk |= match c {
                    JsonCand::MultiConcat { b64: true, .. } => {
                        self.b64_cores.is_empty()
                            || self.b64_cores.iter().any(|core| {
                                may_lie_in(core.needle(), &lits, &gaps, false)
                            })
                    }
                    _ => may_lie_in(needle, &lits, &gaps, true),
                };
            }
            if walk {
                paths.push(pi);
            }
        }
        let fp = (!(paths.is_empty() && self.hooks.iter().all(Vec::is_empty)))
            .then(|| Arc::new(FieldPlan::of_paths(jt, &paths, self.hooks.clone())));
        walks.insert(key, fp.clone());
        fp
    }
}

/// A candidate as the text `lits[0] ref[0] lits[1] … lits[k]` it builds, before any Base64.
fn cand_parts(c: &JsonCand) -> (Vec<&[u8]>, &[u16]) {
    match c {
        JsonCand::Copy { ref_path } => (vec![&[], &[]], std::slice::from_ref(ref_path)),
        JsonCand::Affix { prefix, ref_path, suffix } => {
            (vec![prefix, suffix], std::slice::from_ref(ref_path))
        }
        JsonCand::MultiConcat { lits, refs, .. } => (lits.iter().map(Vec::as_slice).collect(), refs),
    }
}

/// The letters the values of path `r`, as a part of a value of path `own`, may show, given `cols`,
/// the letters of each path's column.
fn path_letters(
    jt: &JsonTreePlan,
    cols: &[u32],
    memo: &mut [Option<u32>],
    r: usize,
    own: usize,
    depth: usize,
) -> u32 {
    // A path's own part is a cell of its column.
    if r == own {
        return cols[r];
    }
    if let Some(l) = memo[r] {
        return l;
    }
    // As deep as `materialize` follows refs.
    if depth > 8 {
        return ANY_LETTERS;
    }
    let mut l = cols[r];
    for c in &jt.paths[r].cands {
        l |= match c {
            JsonCand::MultiConcat { b64: true, .. } => ANY_LETTERS,
            _ => {
                let (lits, refs) = cand_parts(c);
                lits.iter().fold(0, |a, x| a | letters_of(x))
                    | refs
                        .iter()
                        .fold(0, |a, &x| a | path_letters(jt, cols, memo, x as usize, r, depth + 1))
            }
        };
    }
    memo[r] = Some(l);
    l
}

/// Whether `n` can lie in a text `lits[0] gap lits[1] … gap lits[k]`, where gap `i` is any run of
/// bytes whose letters are among `gaps[i]`. With `crossing`, only a placement that takes a
/// literal byte or reaches into two gaps counts: one no single gap holds.
fn may_lie_in(n: &[u8], lits: &[&[u8]], gaps: &[u32], crossing: bool) -> bool {
    // A state is `(part, at, from)`: literal `part / 2` before its byte `at`, or gap `part / 2`
    // for odd `part`; `from` is the gap a placement has kept to so far, `NONE` before its first
    // byte, or `MIXED`.
    const NONE: u32 = u32::MAX;
    const MIXED: u32 = u32::MAX - 1;
    let allowed = |class: u32, b: u8| {
        let l = (b | 0x20).wrapping_sub(b'a');
        match (l < 26, b == b'\\') {
            (true, _) => class & (1 << l) != 0,
            (_, true) => class & ESCAPES != 0,
            _ => true,
        }
    };
    // Moves past literal ends and into the literal after a gap, which may be empty.
    let settle = |mut st: (usize, usize, u32), out: &mut Vec<(usize, usize, u32)>| loop {
        let (part, at, from) = st;
        if part % 2 == 1 {
            out.push(st);
            st = (part + 1, 0, from);
        } else if at < lits[part / 2].len() {
            out.push(st);
            return;
        } else if part / 2 < gaps.len() {
            st = (part + 1, 0, from);
        } else {
            return;
        }
    };
    let mut cur = Vec::new();
    for i in 0..lits.len() {
        (0..lits[i].len()).for_each(|at| settle((2 * i, at, NONE), &mut cur));
    }
    (0..gaps.len()).for_each(|i| settle((2 * i + 1, 0, NONE), &mut cur));
    let mut next = Vec::new();
    for &b in n {
        next.clear();
        for &(part, at, from) in &cur {
            if part % 2 == 0 && lits[part / 2][at] == b {
                settle((part, at + 1, MIXED), &mut next);
            } else if part % 2 == 1 && allowed(gaps[part / 2], b) {
                let gap = (part / 2) as u32;
                let from = if from == NONE || from == gap { gap } else { MIXED };
                settle((part, 0, from), &mut next);
            }
        }
        next.sort_unstable();
        next.dedup();
        std::mem::swap(&mut cur, &mut next);
        if cur.is_empty() {
            return false;
        }
    }
    !crossing || cur.iter().any(|s| s.2 == MIXED)
}

/// True when a match of `needle` could lie partly in one of `lits` and partly outside it, or two
/// values could meet with nothing between them.
fn literals_can_overlap_needle<'a>(lits: impl Iterator<Item = &'a [u8]>, needle: &[u8]) -> bool {
    for lit in lits {
        if lit.is_empty() {
            return true;
        }
        let max_p = (needle.len() - 1).min(lit.len());
        for p in 1..=max_p {
            if lit.ends_with(&needle[..p]) || lit.starts_with(&needle[needle.len() - p..]) {
                return true;
            }
        }
        if lit.len() < needle.len() && memchr::memmem::find(needle, lit).is_some() {
            return true;
        }
    }
    false
}

impl NeedlePlan {
    fn build(plan: &ModelPlan, needle: &[u8], word: bool) -> Self {
        let n_cols = plan.columns.len();
        let finder = Finder::new(needle);
        let has = |lit: &[u8]| finder.find(lit).is_some();
        // A literal holds the needle whatever the values beside it are.
        let sure = |lit: &[u8]| match word {
            false => has(lit),
            // As a word it needs a byte of the literal on each side.
            true => finder.find_iter(lit).any(|at| {
                let end = at + needle.len();
                at > 0 && end < lit.len() && !is_word(lit[at - 1]) && !is_word(lit[end])
            }),
        };
        // A value's ends are where a word may end only if what the record has beside them is no
        // word byte.
        let joins = |lit: &[u8]| {
            !word
                || ![lit.first(), lit.last()]
                    .into_iter()
                    .flatten()
                    .any(|&b| is_word(b))
        };
        // Columns each variant writes, to find the ones exactly one variant writes verbatim.
        let mut users = vec![0u8; n_cols];
        let mut verbatim = vec![false; n_cols];
        let mut use_cols = |cols: &mut dyn Iterator<Item = u16>, plain: bool| {
            for c in cols {
                users[c as usize] = users[c as usize].saturating_add(1);
                verbatim[c as usize] = plain;
            }
        };
        let mut vars = Vec::with_capacity(plan.variants.len());
        let mut unknown_kinds = 0u16;
        let mut derived = None;
        let mut unscanned = Vec::new();
        for (vi, v) in plan.variants.iter().enumerate() {
            let (var, kind) = match v {
                PlanVariant::Shape(s) => {
                    let mut slots = s.slots.clone();
                    slots.sort_unstable();
                    slots.dedup();
                    for c in slots {
                        // `scan_col` reads a nested column's cells as JSON-tagged, which a
                        // shape's are not.
                        let nested = plan.columns[c as usize].nested_plan.is_some();
                        use_cols(&mut std::iter::once(c), !nested);
                    }
                    let n = s.literals.len();
                    let inner = s
                        .literals
                        .iter()
                        .enumerate()
                        .filter(|(i, l)| !(l.is_empty() && (*i == 0 || *i + 1 == n)));
                    let var = if s.literals.iter().any(|l| sure(l)) {
                        VarNeedle::AllTrue
                    } else if s.literals.iter().any(|l| has(l) || !joins(l))
                        || literals_can_overlap_needle(inner.map(|(_, l)| l.as_slice()), needle)
                    {
                        VarNeedle::Unknown
                    } else {
                        VarNeedle::Scan
                    };
                    (var, if vi == 0 { KIND_RESIDUAL } else { KIND_SHAPE })
                }
                PlanVariant::ShapeCell { cols, .. } => {
                    use_cols(&mut cols.iter().copied(), false);
                    (VarNeedle::Unknown, KIND_SHAPE_CELL)
                }
                PlanVariant::JsonTree => {
                    let Some(jt) = &plan.json_tree else {
                        vars.push(VarNeedle::Unknown);
                        unknown_kinds |= KIND_JSON;
                        continue;
                    };
                    use_cols(
                        &mut jt
                            .paths
                            .iter()
                            .map(|p| p.value_col)
                            .filter(|&c| c != u16::MAX),
                        true,
                    );
                    let mut lits = Vec::<&[u8]>::new();
                    // A value built from others repeats their cells, which are tested where
                    // they stand, so only the text between stored values has to join.
                    let mut joined = true;
                    let b64_needle = b64_cores(needle);
                    let overlaps = |l: &[u8]| {
                        has(l) || literals_can_overlap_needle(std::iter::once(l), needle)
                    };
                    // A derived value may hold the needle in no cell when the needle meets
                    // its literal text, or when it is spelled in Base64.
                    let rebuilt = |c: &JsonCand| match c {
                        JsonCand::Copy { .. } => false,
                        JsonCand::Affix { prefix, suffix, .. } => [prefix, suffix]
                            .into_iter()
                            .any(|l| !l.is_empty() && overlaps(l)),
                        JsonCand::MultiConcat { b64: true, .. } => {
                            overlaps(b"\"") || b64_needle.is_some()
                        }
                        JsonCand::MultiConcat { lits: ls, .. } => {
                            ls.iter().enumerate().any(|(i, l)| {
                                !(l.is_empty() && (i == 0 || i + 1 == ls.len())) && overlaps(l)
                            })
                        }
                    };
                    // A free part's cell stands as a word only where the text beside it in the
                    // value is no word byte.
                    let free_joins = |c: &JsonCand, own: usize| match c {
                        JsonCand::MultiConcat { lits: ls, refs, .. } => {
                            refs.iter().position(|&r| r as usize == own).is_none_or(|j| {
                                let open = |b: Option<&u8>| b.is_some_and(|&b| !is_word(b));
                                open(ls[j].last()) && open(ls[j + 1].first())
                            })
                        }
                        _ => true,
                    };
                    let mut hooks = vec![Vec::new(); jt.paths.len()];
                    for (pi, p) in jt.paths.iter().enumerate() {
                        // Every row that takes a shape whose text holds the needle matches.
                        let sure_shape = |sh: &[crate::json_tree::JsonSlotPart]| {
                            sh.iter().any(|t| sure(&t.literal))
                        };
                        let any_sure = p.shapes.iter().any(|sh| sure_shape(sh));
                        if any_sure {
                            hooks[pi] = p
                                .shapes
                                .iter()
                                .map(|sh| if sure_shape(sh) { vec![(0, 0)] } else { Vec::new() })
                                .collect();
                        }
                        for shape in p.shapes.iter().filter(|sh| !any_sure || !sure_shape(sh)) {
                            for (i, part) in shape.iter().enumerate() {
                                // An empty literal at the start or end of a shape joins nothing.
                                if !(part.literal.is_empty()
                                    && (i == 0 || part.kind == SlotKind::None))
                                {
                                    lits.push(&part.literal);
                                }
                                if part.kind == SlotKind::Array {
                                    lits.push(&part.sep);
                                }
                                joined &= joins(&part.literal) && joins(&part.sep);
                            }
                        }
                    }
                    // Outside word mode `sure` is `has`, so no literal left here holds the needle.
                    let exact = joined
                        && !(word && lits.iter().any(|l| has(l)))
                        && !literals_can_overlap_needle(lits.into_iter(), needle);
                    let every = |(pi, p): (usize, &crate::json_tree::JsonPathDef)| {
                        word && !p.cands.iter().all(|c| free_joins(c, pi))
                    };
                    let rebuilds = || {
                        jt.paths.iter().enumerate().any(|(pi, p)| {
                            every((pi, p)) || p.cands.iter().any(rebuilt)
                        })
                    };
                    if exact && (hooks.iter().any(|h| !h.is_empty()) || rebuilds()) {
                        let cands = jt.paths.iter().map(|p| p.cands.iter().map(rebuilt).collect()).collect();
                        let every = jt.paths.iter().enumerate().map(every).collect::<Vec<_>>();
                        unscanned.extend(
                            (0..jt.paths.len())
                                .filter(|&pi| every[pi])
                                .map(|pi| jt.paths[pi].value_col),
                        );
                        derived = Some(DerivedWalk {
                            hooks,
                            cands,
                            every,
                            b64_cores: b64_needle
                                .iter()
                                .flatten()
                                .map(|c| Finder::new(c).into_owned())
                                .collect(),
                            walks: Mutex::default(),
                        });
                    }
                    (
                        if exact {
                            VarNeedle::Scan
                        } else {
                            VarNeedle::Unknown
                        },
                        KIND_JSON,
                    )
                }
            };
            if var == VarNeedle::Unknown {
                unknown_kinds |= kind;
            }
            vars.push(var);
        }
        // A column two variants write has occurrences this code cannot place on rows.
        let mut scan_cols: Vec<bool> = (0..n_cols).map(|c| users[c] == 1 && verbatim[c]).collect();
        for c in unscanned.into_iter().filter(|&c| c != u16::MAX) {
            scan_cols[c as usize] = false;
        }
        for (vi, v) in plan.variants.iter().enumerate() {
            let shared = |c: u16| users[c as usize] > 1;
            let (bad, kind) = match v {
                PlanVariant::Shape(s) => (
                    s.slots.iter().any(|&c| !scan_cols[c as usize]),
                    if vi == 0 { KIND_RESIDUAL } else { KIND_SHAPE },
                ),
                PlanVariant::JsonTree => (
                    plan.json_tree.as_ref().is_some_and(|jt| {
                        jt.paths
                            .iter()
                            .any(|p| p.value_col != u16::MAX && shared(p.value_col))
                    }),
                    KIND_JSON,
                ),
                _ => (false, 0),
            };
            if bad && vars[vi] != VarNeedle::AllTrue {
                vars[vi] = VarNeedle::Unknown;
                unknown_kinds |= kind;
            }
        }
        let all_true = vars.contains(&VarNeedle::AllTrue);
        let derived_open = if derived.is_some() { KIND_JSON } else { 0 };
        NeedlePlan {
            vars,
            unknown_kinds,
            open_kinds: unknown_kinds | derived_open | if all_true { KIND_SHAPE } else { 0 },
            scan_cols,
            letters: letters_of(needle),
            finder: finder.into_owned(),
            word,
            quote_sensitive: needle.contains(&b'"')
                || memchr::memmem::find(b"null", needle).is_some(),
            derived,
        }
    }

    /// Whether `text`, a value whole, holds the needle.
    fn found(&self, text: &[u8]) -> bool {
        found(&self.finder, self.finder.needle().len(), self.word, text)
    }

    /// The columns of a block whose cells may hold the needle.
    fn candidates(&self, m: &BlockMeta<'_>) -> Vec<usize> {
        let may_hold = |s: &ColSummary| !s.all_null && self.letters & !s.letters == 0;
        m.dir
            .cols()
            .filter(|(c, e)| self.scan_cols[*c] && may_hold(&e.summary))
            .map(|(c, _)| c)
            .collect()
    }
}

/// Per-model plans, kept for the length of a query.
#[derive(Default)]
pub(crate) struct Plans {
    fields: HashMap<(u32, String), Arc<FieldPlan>>,
    needles: HashMap<(u32, Vec<u8>, bool), Arc<NeedlePlan>>,
    /// By model and the columns a walk wants, the structure streams it reads.
    streams: HashMap<(u32, Vec<bool>), Vec<bool>>,
    /// Plans kept for every session of a file, so each is built once.
    shared: Option<Arc<Mutex<Plans>>>,
}

impl Plans {
    pub fn sharing(shared: &Arc<Mutex<Plans>>) -> Self {
        Plans {
            shared: Some(shared.clone()),
            ..Plans::default()
        }
    }

    pub fn field(&mut self, m: &BlockMeta<'_>, field: &str) -> Arc<FieldPlan> {
        let key = (m.model, field.to_string());
        if let Some(fp) = self.fields.get(&key) {
            return fp.clone();
        }
        let fp = match &self.shared {
            Some(s) => s.lock().unwrap().field(m, field),
            None => Arc::new(FieldPlan::build(m.plan, field)),
        };
        self.fields.insert(key, fp.clone());
        fp
    }

    fn needle(&mut self, m: &BlockMeta<'_>, needle: &[u8], word: bool) -> Arc<NeedlePlan> {
        let key = (m.model, needle.to_vec(), word);
        if let Some(np) = self.needles.get(&key) {
            return np.clone();
        }
        let np = match &self.shared {
            Some(s) => s.lock().unwrap().needle(m, needle, word),
            None => Arc::new(NeedlePlan::build(m.plan, needle, word)),
        };
        self.needles.insert(key, np.clone());
        np
    }
}

// ============================================================================
// Blocks
// ============================================================================

/// What the index says about one structured block.
pub(crate) struct BlockMeta<'a> {
    pub model: u32,
    pub plan: &'a ModelPlan,
    pub kinds: u16,
    pub dir: &'a BlockDir,
    pub filter: Filter<'a>,
    /// The filters of the text pool's pages, when at hand.
    pub pages: &'a [Filter<'a>],
}

/// False when the index proves no row of the block satisfies `pred`.
pub(crate) fn may_match(pred: &Predicate, m: &BlockMeta<'_>, plans: &mut Plans) -> bool {
    match pred {
        Predicate::All | Predicate::Not(_) => true,
        Predicate::And(list) => list.iter().all(|p| may_match(p, m, plans)),
        Predicate::Or(list) => list.iter().any(|p| may_match(p, m, plans)),
        Predicate::Contains(needle) | Predicate::Word(needle) => {
            let word = matches!(pred, Predicate::Word(_));
            plain_may_contain(needle, word, m.filter) && {
                let np = plans.needle(m, needle, word);
                m.kinds & np.open_kinds != 0 || !np.candidates(m).is_empty()
            }
        }
        _ => {
            let (field, test) = Test::of(pred).unwrap();
            // Rows outside the JSON tree are tested as raw records, which the index does not see.
            if m.kinds & !KIND_JSON != 0 {
                return true;
            }
            let fp = plans.field(m, field);
            let live = |c: &u16| !m.dir.col(*c as usize).summary.all_null;
            // A subtree spells the keys it holds, unless it escapes them.
            let holds = |&(c, keys): &(u16, u32)| {
                let s = &m.dir.col(c as usize).summary;
                !s.all_null && (s.letters & ESCAPES != 0 || keys & !s.letters == 0)
            };
            if !fp.prunable || fp.subtree_cols.iter().any(holds) {
                return true;
            }
            fp.leaf_cols
                .iter()
                .chain(&fp.copy_cols)
                .filter(|c| live(c))
                .any(|&c| test.may_hold(&m.dir.col(c as usize).summary, m.filter))
        }
    }
}

/// False when a filter proves no record of its block contains `needle`. As a word, the runs of
/// letters at its ends are whole runs of the record's.
pub(crate) fn plain_may_contain(needle: &[u8], word: bool, filter: Filter<'_>) -> bool {
    needle_tokens(needle, word).is_none_or(|ts| ts.iter().all(|&t| filter.has(t)))
}

/// The parts of a block an evaluation reads.
pub(crate) struct Need {
    pub whole: bool,
    /// By column.
    pub cols: Vec<bool>,
    /// By structure stream.
    pub streams: Vec<bool>,
    /// The text pool pages a search may find its needles in, or `None` for every page.
    pub pages: Option<Vec<bool>>,
}

impl Need {
    pub fn new(dir: &BlockDir) -> Self {
        Need {
            whole: false,
            cols: vec![false; dir.n_cols()],
            streams: vec![false; dir.streams.len()],
            pages: None,
        }
    }

    /// As `new`, with no text pool page needed yet when the page filters are at hand.
    pub fn paged(m: &BlockMeta<'_>) -> Self {
        let n = m.dir.pages.len();
        Need {
            pages: (n > 0 && m.pages.len() == n).then(|| vec![false; n]),
            ..Need::new(m.dir)
        }
    }
}

/// Whether a filter can rule a block out for `pred`, and whether only a filter can.
pub(crate) fn filter_use(pred: &Predicate) -> (bool, bool) {
    match pred {
        Predicate::All | Predicate::Not(_) => (false, true),
        Predicate::And(list) | Predicate::Or(list) => list
            .iter()
            .map(filter_use)
            .fold((false, true), |a, b| (a.0 || b.0, a.1 && b.1)),
        Predicate::Contains(needle) => (needle_tokens(needle, false).is_some(), true),
        Predicate::Word(needle) => (needle_tokens(needle, true).is_some(), true),
        Predicate::FieldEq { value, .. } => (needle_tokens(value, true).is_some(), false),
        Predicate::FieldIn { values, .. } => (
            values.iter().all(|v| needle_tokens(v, true).is_some()),
            false,
        ),
        Predicate::FieldContains { needle, .. } => (needle_tokens(needle, false).is_some(), false),
        _ => (false, false),
    }
}

/// Adds what `eval` reads for `pred` to `need`.
pub(crate) fn needs(pred: &Predicate, m: &BlockMeta<'_>, plans: &mut Plans, need: &mut Need) {
    match pred {
        Predicate::All => {}
        Predicate::Not(inner) => needs(inner, m, plans, need),
        Predicate::And(list) | Predicate::Or(list) => {
            list.iter().for_each(|p| needs(p, m, plans, need))
        }
        Predicate::Contains(needle) | Predicate::Word(needle) if needle.is_empty() => {}
        Predicate::Contains(needle) | Predicate::Word(needle) => {
            let np = plans.needle(m, needle, matches!(pred, Predicate::Word(_)));
            need.whole |= m.kinds & np.unknown_kinds != 0;
            if let Some(fp) = np.derived.as_ref().and_then(|d| d.for_block(m.plan, m.dir, np.finder.needle())) {
                fp.needs(m.kinds, m.dir, need);
            }
            // A column with a cell for every row needs no structure to place its cells.
            let mut want = vec![false; need.cols.len()];
            let mut pooled = false;
            for c in np.candidates(m) {
                (need.cols[c], want[c]) = (true, !m.dir.col(c).summary.per_row);
                pooled |= m.dir.col(c).deps.pool;
            }
            // The pages whose filters hold every token of the needle.
            if pooled && let Some(pages) = &mut need.pages {
                let word = matches!(pred, Predicate::Word(_));
                match needle_tokens(needle, word) {
                    Some(ts) => pages
                        .iter_mut()
                        .zip(m.pages)
                        .for_each(|(p, f)| *p |= ts.iter().all(|&t| f.has(t))),
                    None => pages.fill(true),
                }
            }
            if let Some(jt) = &m.plan.json_tree {
                let streams = plans
                    .streams
                    .entry((m.model, want))
                    .or_insert_with_key(|(_, want)| {
                        let walk = JsonWalk {
                            want_cols: want,
                            ..JsonWalk::default()
                        };
                        WalkRoute::of(jt, &walk).streams(jt)
                    });
                need.streams
                    .iter_mut()
                    .zip(streams.iter())
                    .for_each(|(n, s)| *n |= s);
            }
        }
        _ => {
            let (field, _) = Test::of(pred).unwrap();
            need.whole |= m.kinds & !KIND_JSON != 0;
            plans.field(m, field).needs(m.kinds, m.dir, need);
        }
    }
}

pub(crate) const NO: u8 = 0;
pub(crate) const YES: u8 = 1;
/// The columns cannot say; the record has to be rebuilt and tested.
pub(crate) const UNKNOWN: u8 = 2;

/// Evaluates `pred` on every row of a block from its columns: `NO`, `YES` or `UNKNOWN` per row.
pub(crate) fn eval(
    pred: &Predicate,
    m: &BlockMeta<'_>,
    rows: &mut BlockRows<'_>,
    plans: &mut Plans,
    budget: &DecodeBudget,
) -> Result<Vec<u8>> {
    let n = rows.len();
    match pred {
        Predicate::All => Ok(vec![YES; n]),
        Predicate::Not(inner) => {
            let mut t = eval(inner, m, rows, plans, budget)?;
            t.iter_mut()
                .for_each(|v| *v = [YES, NO, UNKNOWN][*v as usize]);
            Ok(t)
        }
        Predicate::And(list) => {
            let mut acc = vec![YES; n];
            for sub in list {
                let t = eval(sub, m, rows, plans, budget)?;
                for (a, b) in acc.iter_mut().zip(t) {
                    *a = if *a == NO || b == NO { NO } else { (*a).max(b) };
                }
            }
            Ok(acc)
        }
        Predicate::Or(list) => {
            let mut acc = vec![NO; n];
            for sub in list {
                let t = eval(sub, m, rows, plans, budget)?;
                for (a, b) in acc.iter_mut().zip(t) {
                    *a = if *a == YES || b == YES {
                        YES
                    } else {
                        (*a).max(b)
                    };
                }
            }
            Ok(acc)
        }
        Predicate::Contains(needle) | Predicate::Word(needle) if needle.is_empty() => {
            Ok(vec![YES; n])
        }
        Predicate::Contains(needle) | Predicate::Word(needle) => eval_contains(
            needle,
            matches!(pred, Predicate::Word(_)),
            m,
            rows,
            plans,
            budget,
        ),
        _ => {
            let (field, test) = Test::of(pred).unwrap();
            let jvi = json_variant(m.plan);
            let mut tri: Vec<u8> = rows
                .variant_ids
                .iter()
                .map(|&v| if Some(v) == jvi { NO } else { UNKNOWN })
                .collect();
            let fp = plans.field(m, field);
            if let Some(tri) = eval_leaf_cells(&fp, &test, m, rows, &tri, budget)? {
                return Ok(tri);
            }
            // By column: each dictionary entry's verdict, 0 until tested.
            let mut verdicts: Vec<Vec<u8>> = vec![Vec::new(); m.plan.columns.len()];
            field_events(&fp, m.kinds, m.dir, rows, None, budget, &mut |row, ev| {
                let t = &mut tri[row as usize];
                if *t == YES {
                    return;
                }
                let pass = match ev {
                    FieldEv::Opaque => {
                        *t = UNKNOWN;
                        return;
                    }
                    FieldEv::Val { lex, entry: None } => test.ok(&logical(lex)),
                    FieldEv::Val {
                        lex,
                        entry: Some((col, ord, n_entries)),
                    } => {
                        let v = &mut verdicts[col as usize];
                        if v.is_empty() {
                            v.resize(n_entries as usize, 0);
                        }
                        if v[ord as usize] == 0 {
                            v[ord as usize] = 1 + u8::from(test.ok(&logical(lex)));
                        }
                        v[ord as usize] == 2
                    }
                };
                if pass {
                    *t = YES;
                }
            })?;
            Ok(tri)
        }
    }
}

/// By cell of `col`, whether its value passes `test`. `None` when a cell holds an array, whose
/// elements are the values. `escapes` is false when no cell has a backslash.
fn cell_verdicts(col: &Col, escapes: bool, test: &Test<'_>) -> Result<Option<Vec<bool>>> {
    let mut buf = Vec::new();
    let by_entry =
        |ords: &[u32], entries: Vec<bool>| ords.iter().map(|&o| entries[o as usize]).collect();
    // `ok` of each cell, without a call per cell where the cells lie back to back.
    let mut each = |c: &Col, ok: &mut dyn FnMut(&[u8]) -> bool| -> Result<Vec<bool>> {
        let Some((bytes, ends)) = c.laid() else {
            return (0..c.len())
                .map(|i| c.get(i, &mut buf).map(&mut *ok))
                .collect();
        };
        let mut start = 0usize;
        Ok(ends
            .iter()
            .map(|&e| ok(&bytes[std::mem::replace(&mut start, e as usize)..e as usize]))
            .collect())
    };
    if let Col::Quoted { inner, map } = col {
        // What `logical` makes of the cell `"e"`.
        let mut ok = |e: &[u8]| match escapes.then(|| unescape_json_str_exact(e)).flatten() {
            Some(plain) => test.ok(&plain),
            None => test.ok(e),
        };
        let cells: Vec<bool> = match &**inner {
            Col::Dict { dict, ords } => by_entry(ords, dict.iter().map(&mut ok).collect()),
            c => each(c, &mut ok)?,
        };
        if map.is_empty() {
            return Ok(Some(cells));
        }
        let null = test.ok(b"null");
        let cell = |&j: &u32| match cells.get(j as usize) {
            _ if j == NULL_CELL => Ok(null),
            Some(&pass) => Ok(pass),
            None => Err(underflow()),
        };
        return Ok(Some(map.iter().map(cell).collect::<Result<_>>()?));
    }
    let mut array = false;
    let mut ok = |lex: &[u8]| {
        array |= lex.first() == Some(&b'[');
        test.ok(&logical(lex))
    };
    let cells: Vec<bool> = match col {
        Col::Dict { dict, ords } => by_entry(ords, dict.iter().map(&mut ok).collect()),
        _ => each(col, &mut ok)?,
    };
    Ok((!array).then_some(cells))
}

/// A field test on a block whose every value of the field is a whole cell of a column: the
/// cells are tested as columns, and the rows are walked only if a cell passes and the cells are
/// not one to a row. `None` when the field has values elsewhere.
fn eval_leaf_cells(
    fp: &FieldPlan,
    test: &Test<'_>,
    m: &BlockMeta<'_>,
    rows: &mut BlockRows<'_>,
    unmatched: &[u8],
    budget: &DecodeBudget,
) -> Result<Option<Vec<u8>>> {
    let plan = rows.plan;
    // An empty column is not among the block's loaded parts.
    let summary = |c: u16| &m.dir.col(c as usize).summary;
    if !fp.prunable || fp.subtree_cols.iter().any(|&(c, _)| !summary(c).all_null) {
        return Ok(None);
    }
    let mut passing: Vec<(u16, Vec<bool>)> = Vec::with_capacity(fp.leaf_cols.len());
    let cols = fp.leaf_cols.iter().chain(&fp.copy_cols);
    for &c in cols.filter(|&&c| !summary(c).all_null) {
        if plan.columns[c as usize].nested_plan.is_some() {
            return Ok(None);
        }
        let escapes = summary(c).letters & ESCAPES != 0;
        match cell_verdicts(rows.col(c as usize, budget)?, escapes, test)? {
            Some(v) => passing.push((c, v)),
            None => return Ok(None),
        }
    }
    let mut tri = unmatched.to_vec();
    if fp.row_col(m.kinds, m.dir).is_some() {
        let cells = &passing[0].1;
        if cells.len() != tri.len() {
            return Err(underflow());
        }
        tri.iter_mut()
            .zip(cells)
            .for_each(|(t, &pass)| *t = if pass { YES } else { NO });
        return Ok(Some(tri));
    }
    let (Some(jt), Some(jvi)) = (&plan.json_tree, json_variant(plan)) else {
        return Ok(Some(tri));
    };
    if passing.iter().all(|(_, v)| !v.contains(&true)) {
        return Ok(Some(tri));
    }
    let walk = JsonWalk {
        want_cols: &[],
        track: &fp.track,
        hooks: &fp.hooks,
    };
    let map = map_json_block_structure(
        jt,
        rows.json_streams(),
        &rows.variant_ids,
        jvi,
        &walk,
        &fp.route,
    )?;
    for ev in &map.events {
        let Some(rem) = &fp.rem[ev.path as usize] else {
            continue;
        };
        // A copied value is its source's.
        let mut src = *ev;
        while let PathVal::Derived { cand, srcs } = src.val {
            let JsonCand::Copy { .. } = jt.paths[src.path as usize].cands[cand as usize] else {
                return Ok(None);
            };
            src = map.events[map.srcs[srcs as usize] as usize];
        }
        let PathVal::Occ { col, occ } = src.val else {
            return Ok(None);
        };
        let Some((_, cells)) = passing
            .iter()
            .find(|(c, _)| *c == col)
            .filter(|_| rem.is_empty())
        else {
            return Ok(None);
        };
        if *cells.get(occ as usize).ok_or_else(underflow)? {
            tri[ev.row as usize] = YES;
        }
    }
    Ok(Some(tri))
}

pub(crate) fn json_variant(plan: &ModelPlan) -> Option<u32> {
    plan.variants
        .iter()
        .position(|v| matches!(v, PlanVariant::JsonTree))
        .map(|i| i as u32)
}

fn eval_contains(
    needle: &[u8],
    word: bool,
    m: &BlockMeta<'_>,
    rows: &mut BlockRows<'_>,
    plans: &mut Plans,
    budget: &DecodeBudget,
) -> Result<Vec<u8>> {
    let plan = m.plan;
    let np = plans.needle(m, needle, word);
    let mut tri: Vec<u8> = rows
        .variant_ids
        .iter()
        .map(|&v| match np.vars[v as usize] {
            VarNeedle::AllTrue => YES,
            VarNeedle::Unknown => UNKNOWN,
            VarNeedle::Scan => NO,
        })
        .collect();
    // By candidate column with a cell that holds the needle: those cells.
    let mut hits = Vec::<(usize, Vec<u32>)>::new();
    let mut want = vec![false; plan.columns.len()];
    for cid in np.candidates(m) {
        let nested = plan.columns[cid].nested_plan.is_some();
        let col = rows.col(cid, budget)?;
        let mut occs = Vec::new();
        scan_col(col, nested, needle, &np, &mut |occ| occs.push(occ as u32))?;
        if occs.is_empty() {
            continue;
        }
        if m.dir.col(cid).summary.per_row {
            if col.len() != tri.len() {
                return Err(underflow());
            }
            occs.iter().for_each(|&occ| tri[occ as usize] = YES);
        } else {
            want[cid] = true;
            hits.push((cid, occs));
        }
    }
    if !hits.is_empty() {
        let occ_rows = cell_rows(plan, rows.json_streams(), &rows.variant_ids, &want)?;
        for (cid, occs) in hits {
            let at = &occ_rows[cid];
            if rows.col(cid, budget)?.len() != at.len() {
                return Err(Error::Format(
                    "column occurrences do not match the row structure".into(),
                ));
            }
            occs.iter()
                .for_each(|&occ| tri[at[occ as usize] as usize] = YES);
        }
    }
    // Only rows no cell has settled need the walk.
    let walked = match &np.derived {
        Some(d) if tri.contains(&NO) => d.for_block(plan, m.dir, np.finder.needle()).map(|fp| (d, fp)),
        _ => None,
    };
    if let (Some((d, fp)), Some(jt), Some(jvi)) = (walked, &plan.json_tree, json_variant(plan)) {
        let walk = JsonWalk {
            track: &fp.track,
            hooks: &fp.hooks,
            ..JsonWalk::default()
        };
        let map = map_json_block_structure(
            jt,
            rows.json_streams(),
            &rows.variant_ids,
            jvi,
            &walk,
            &fp.route,
        )?;
        let mut text = Vec::new();
        for ev in &map.events {
            let p = ev.path as usize;
            if tri[ev.row as usize] != NO {
                continue;
            }
            let (test, b64) = match ev.val {
                PathVal::Hook(_) => {
                    tri[ev.row as usize] = YES;
                    continue;
                }
                PathVal::Derived { cand, .. } => (
                    d.cands[p].get(cand as usize) == Some(&true),
                    matches!(
                        jt.paths[p].cands.get(cand as usize),
                        Some(JsonCand::MultiConcat { b64: true, .. })
                    ),
                ),
                _ => (false, false),
            };
            if !(test || d.every[p]) {
                continue;
            }
            // Text whose Base64 holds the needle holds one of its cores; most values do not.
            if b64 && !d.b64_cores.is_empty() {
                text.clear();
                b64_text(&map, ev, rows, &mut text, budget)?;
                let has = |c: &Finder<'_>| c.find(&text).is_some();
                if !d.b64_cores.iter().any(has) {
                    continue;
                }
            }
            text.clear();
            materialize(&map, ev, rows, &mut text, budget, 0)?;
            if np.found(&text) {
                tri[ev.row as usize] = YES;
            }
        }
    }
    Ok(tri)
}

/// Appends the text a Base64 candidate's value encodes.
fn b64_text(
    map: &JsonBlockMap,
    ev: &PathEvent,
    rows: &mut BlockRows<'_>,
    out: &mut Vec<u8>,
    budget: &DecodeBudget,
) -> Result<()> {
    let jt = rows.plan.json_tree.as_ref().ok_or_else(underflow)?;
    let (PathVal::Derived { cand, srcs }, Some(p)) = (ev.val, jt.paths.get(ev.path as usize))
    else {
        return Err(underflow());
    };
    let Some(JsonCand::MultiConcat { lits, refs, .. }) = p.cands.get(cand as usize) else {
        return Err(underflow());
    };
    out.extend_from_slice(&lits[0]);
    for k in 0..refs.len() {
        let src = map.srcs[srcs as usize + k];
        let at = out.len();
        materialize(map, &map.events[src as usize], rows, out, budget, 1)?;
        if out.len() - at >= 2 && out[at] == b'"' && out[out.len() - 1] == b'"' {
            out.pop();
            out.remove(at);
        }
        out.extend_from_slice(&lits[k + 1]);
    }
    Ok(())
}

/// By column of `want`, the row of each of its cells, as JSON and shape rows place them. Cells
/// of other variants are left out, so a caller compares the counts with the column's.
pub(crate) fn cell_rows(
    plan: &ModelPlan,
    json_streams: &[Vec<u32>],
    variant_ids: &[u32],
    want: &[bool],
) -> Result<Vec<Vec<u32>>> {
    let mut occ_rows = match (&plan.json_tree, json_variant(plan)) {
        (Some(jt), Some(jvi)) => {
            let walk = JsonWalk {
                want_cols: want,
                ..JsonWalk::default()
            };
            let route = WalkRoute::of(jt, &walk);
            map_json_block_structure(jt, json_streams, variant_ids, jvi, &walk, &route)?
                .col_occ_rows
        }
        _ => vec![Vec::new(); plan.columns.len()],
    };
    for (r, &v) in variant_ids.iter().enumerate() {
        if let Some(PlanVariant::Shape(s)) = plan.variants.get(v as usize) {
            for &c in s.slots.iter().filter(|&&c| want[c as usize]) {
                occ_rows[c as usize].push(r as u32);
            }
        }
    }
    Ok(occ_rows)
}

/// Calls `hit` with each cell of `col` whose bytes, as a record shows them, contain the needle.
fn scan_col(
    col: &Col,
    nested: bool,
    needle: &[u8],
    np: &NeedlePlan,
    hit: &mut dyn FnMut(usize),
) -> Result<()> {
    // Finds matches in cells laid back to back, skipping those that cross from one cell to the next.
    let scan_buf = |buf: &[u8], ends: &[u32], hit: &mut dyn FnMut(usize)| {
        let mut pos = 0usize;
        while let Some(p) = np.finder.find(&buf[pos..]) {
            let (at, fin) = (pos + p, pos + p + needle.len());
            let cell = ends.partition_point(|&e| e as usize <= at);
            let (start, end) = (
                if cell == 0 {
                    0
                } else {
                    ends[cell - 1] as usize
                },
                ends[cell] as usize,
            );
            let stands = || {
                !np.word
                    || ((at == start || !is_word(buf[at - 1]))
                        && (fin == end || !is_word(buf[fin])))
            };
            if fin <= end && stands() {
                hit(cell);
                pos = end;
            } else {
                pos = at + 1;
            }
        }
    };
    match col {
        Col::Const { val, n } => {
            if np.found(val) {
                (0..*n).for_each(hit);
            }
        }
        Col::Dict { dict, ords } => {
            let mut verdict = vec![0u8; dict.len()];
            for (i, &o) in ords.iter().enumerate() {
                let entry = dict.get(o as usize).ok_or_else(underflow)?;
                let v = &mut verdict[o as usize];
                if *v == 0 {
                    *v = 1 + u8::from(np.found(entry));
                }
                if *v == 2 {
                    hit(i);
                }
            }
        }
        Col::Flat(cells) if !nested => scan_buf(cells.bytes(), cells.ends(), hit),
        Col::Pool { pool, base, ends } => {
            let len = ends.last().map_or(0, |&e| e as usize);
            scan_buf(&pool.bytes()[*base..*base + len], ends, hit)
        }
        Col::Paged { pool, spans } => {
            // A page holds its cells of the column back to back, in order: scan each run. The
            // pages not held are those the page filters ruled out.
            let bytes = pool.bytes();
            let held = |i: &u32| spans[*i as usize] != NOT_HELD;
            let mut order: Vec<u32> = (0..spans.len() as u32).filter(held).collect();
            order.sort_by_key(|&i| spans[i as usize].0);
            let mut ends = Vec::new();
            let mut a = 0;
            while a < order.len() {
                let from = spans[order[a] as usize].0;
                ends.clear();
                let mut b = a;
                while let Some(&(s, e)) = order.get(b).map(|&i| &spans[i as usize])
                    && s == from + ends.last().copied().unwrap_or(0)
                {
                    ends.push(e - from);
                    b += 1;
                }
                let run = &order[a..b];
                let to = from as usize + *ends.last().unwrap() as usize;
                scan_buf(&bytes[from as usize..to], &ends, &mut |k| hit(run[k] as usize));
                a = b;
            }
        }
        Col::Quoted { inner, map } if !np.quote_sensitive => {
            if map.is_empty() {
                return scan_col(inner, false, needle, np, hit);
            }
            let mut inner_hit = vec![false; inner.len()];
            scan_col(inner, false, needle, np, &mut |j| inner_hit[j] = true)?;
            for (i, &j) in map.iter().enumerate() {
                if j != NULL_CELL && inner_hit[j as usize] {
                    hit(i);
                }
            }
        }
        _ => {
            let (mut cell, mut shown) = (Vec::new(), Vec::new());
            for i in 0..col.len() {
                cell.clear();
                col.write(i, &mut cell)?;
                let text = if nested {
                    shown.clear();
                    emit_nested_cell(&cell, &mut shown)?;
                    &shown
                } else {
                    &cell
                };
                if np.found(text) {
                    hit(i);
                }
            }
        }
    }
    Ok(())
}

/// One value of a field on a JSON row.
pub(crate) enum FieldEv<'a> {
    /// The value is spread over literal text and columns; the record has to be rebuilt.
    Opaque,
    /// `entry` is `(column, ordinal, entries)` when the value is one entry of a dictionary.
    Val {
        lex: &'a [u8],
        entry: Option<(u16, u32, u32)>,
    },
}

/// Appends the bytes a record shows for one event.
fn materialize(
    map: &JsonBlockMap,
    ev: &PathEvent,
    rows: &mut BlockRows<'_>,
    out: &mut Vec<u8>,
    budget: &DecodeBudget,
    depth: usize,
) -> Result<()> {
    match ev.val {
        PathVal::Occ { col, occ } => {
            let nested = rows.plan.columns[col as usize].nested_plan.is_some();
            let c = rows.col(col as usize, budget)?;
            if !nested {
                return c.write(occ as usize, out);
            }
            let mut cell = Vec::new();
            c.write(occ as usize, &mut cell)?;
            emit_nested_cell(&cell, out)
        }
        PathVal::Derived { cand, srcs } => {
            let plan = rows.plan;
            let jt = plan.json_tree.as_ref().unwrap();
            let cand = &jt.paths[ev.path as usize].cands[cand as usize];
            if depth > 8 {
                return Err(Error::Format("derived value nests too deep".into()));
            }
            let own = ev.path as usize;
            let mut stem = |k: usize, out: &mut Vec<u8>, unquote: bool| -> Result<()> {
                let src = map.srcs[srcs as usize + k];
                if matches!(cand, JsonCand::MultiConcat { refs, .. } if refs[k] as usize == own) {
                    // The free part: `src` is the occurrence of its cell.
                    let col = jt.paths[own].value_col as usize;
                    return rows.col(col, budget)?.write(src as usize, out);
                }
                let src = map.events[src as usize];
                let mut t = Vec::new();
                materialize(map, &src, rows, &mut t, budget, depth + 1)?;
                let quoted = unquote && t.len() >= 2 && t[0] == b'"' && t[t.len() - 1] == b'"';
                out.extend_from_slice(if quoted { &t[1..t.len() - 1] } else { &t });
                Ok(())
            };
            match cand {
                JsonCand::Copy { .. } => stem(0, out, false),
                JsonCand::Affix { prefix, suffix, .. } => {
                    out.extend_from_slice(prefix);
                    stem(0, out, true)?;
                    out.extend_from_slice(suffix);
                    Ok(())
                }
                JsonCand::MultiConcat { lits, refs, b64 } => {
                    let mut text = Vec::new();
                    let dst = if *b64 { &mut text } else { &mut *out };
                    dst.extend_from_slice(&lits[0]);
                    for k in 0..refs.len() {
                        stem(k, dst, true)?;
                        dst.extend_from_slice(&lits[k + 1]);
                    }
                    if *b64 {
                        if !crate::json_tree::b64_text_ok(&text) {
                            return Err(Error::Format("Base64 candidate text not printable".into()));
                        }
                        out.push(b'"');
                        crate::lex::format_base64(&text, out);
                        out.push(b'"');
                    }
                    Ok(())
                }
            }
        }
        PathVal::Hook(_) => Err(Error::Format("derived value from literal text".into())),
    }
}

/// Calls `f(row, value)` for each value of the planned field on the block's JSON rows, in record
/// order; `only` limits it to some rows.
pub(crate) fn field_events(
    fp: &FieldPlan,
    kinds: u16,
    dir: &BlockDir,
    rows: &mut BlockRows<'_>,
    only: Option<&[bool]>,
    budget: &DecodeBudget,
    f: &mut dyn FnMut(u32, FieldEv<'_>),
) -> Result<()> {
    let plan = rows.plan;
    let (Some(jt), Some(jvi)) = (&plan.json_tree, json_variant(plan)) else {
        return Ok(());
    };
    let (mut doc, mut cell_doc) = (JsonDoc::default(), JsonDoc::default());
    let (mut buf, mut cell_buf) = (Vec::new(), Vec::new());
    // The values of the cell `occ` of a column.
    let mut cell = |c: &Col,
                    col: u16,
                    occ: u32,
                    row: u32,
                    f: &mut dyn FnMut(u32, FieldEv<'_>)|
     -> Result<()> {
        let (doc, buf) = (&mut cell_doc, &mut cell_buf);
        let entry = match c {
            Col::Dict { dict, ords } => Some((col, ords[occ as usize], dict.len() as u32)),
            _ => None,
        };
        let lex = c.get(occ as usize, buf)?;
        if lex.first() != Some(&b'[') {
            f(row, FieldEv::Val { lex, entry });
        } else if !json_values_at(doc, lex, b"", &mut |s, e| {
            f(
                row,
                FieldEv::Val {
                    lex: &lex[s..e],
                    entry: None,
                },
            )
        }) {
            f(row, FieldEv::Opaque);
        }
        Ok(())
    };
    if let Some(col) = fp.row_col(kinds, dir) {
        let n = rows.variant_ids.len();
        let c = rows.col(col as usize, budget)?;
        if c.len() != n {
            return Err(underflow());
        }
        for row in (0..n as u32).filter(|&r| only.is_none_or(|o| o[r as usize])) {
            cell(c, col, row, row, f)?;
        }
        return Ok(());
    }
    let walk = JsonWalk {
        want_cols: &[],
        track: &fp.track,
        hooks: &fp.hooks,
    };
    let map = map_json_block_structure(
        jt,
        rows.json_streams(),
        &rows.variant_ids,
        jvi,
        &walk,
        &fp.route,
    )?;
    for ev in &map.events {
        let Some(rem) = &fp.rem[ev.path as usize] else {
            continue;
        };
        if only.is_some_and(|o| !o[ev.row as usize]) {
            continue;
        }
        match ev.val {
            PathVal::Hook(OPAQUE) => f(ev.row, FieldEv::Opaque),
            PathVal::Hook(id) => f(
                ev.row,
                FieldEv::Val {
                    lex: &fp.consts[id as usize],
                    entry: None,
                },
            ),
            PathVal::Occ { col, occ }
                if rem.is_empty() && plan.columns[col as usize].nested_plan.is_none() =>
            {
                cell(rows.col(col as usize, budget)?, col, occ, ev.row, f)?
            }
            _ => {
                buf.clear();
                materialize(&map, ev, rows, &mut buf, budget, 0)?;
                if !json_values_at(&mut doc, &buf, rem, &mut |s, e| {
                    f(
                        ev.row,
                        FieldEv::Val {
                            lex: &buf[s..e],
                            entry: None,
                        },
                    )
                }) {
                    f(ev.row, FieldEv::Opaque);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_rule_keeps_every_substring_of_printable_text() {
        for text in [&b"04:User583"[..], b"012:PullRequest99", b" ~{}|", b"a"] {
            let mut enc = Vec::new();
            crate::lex::format_base64(text, &mut enc);
            let body = enc.split(|&c| c == b'=').next().unwrap();
            for s in 0..body.len() {
                for e in s + 1..=body.len() {
                    assert!(b64_cores(&body[s..e]).is_some(), "{:?}", &body[s..e]);
                }
            }
            assert!(b64_cores(&enc).is_some());
        }
        for n in ["clickhouse", "terminating", "kgDOexc", "2024", "main"] {
            assert!(b64_cores(n.as_bytes()).is_none(), "{n}");
        }
        assert!(b64_cores(b"a.b").is_none());
    }

    #[test]
    fn raw_field_values_follow_paths_through_arrays() {
        let rec = br#"{"a":{"b":[{"c":1},{"c":"x\ny"},{"d":2}],"e":[1,[2,3]]},"k.z":true}"#;
        let vals = |field: &str| {
            let mut out = Vec::new();
            raw_field_values(rec, field, &mut |v| {
                out.push(String::from_utf8_lossy(v).into_owned())
            });
            out
        };
        assert_eq!(vals("a.b.c"), ["1", r#""x\ny""#]);
        assert_eq!(vals("$.a.b[].c"), ["1", r#""x\ny""#]);
        assert_eq!(vals("a.e"), ["1", "2", "3"]);
        assert_eq!(vals(r#"["k.z"]"#), ["true"]);
        assert_eq!(vals("a.b").len(), 3);
        assert!(vals("a.c").is_empty() && vals("a.b.c.d").is_empty());

        let eq = |field: &str, value: &str| Predicate::FieldEq {
            field: field.into(),
            value: value.into(),
        };
        assert!(eval_raw_record(rec, &eq("a.b.c", "x\ny")));
        assert!(eval_raw_record(rec, &eq("a.e", "3")) && !eval_raw_record(rec, &eq("a.e", "4")));
        assert!(
            eval_raw_record(b"  {\"a\": 1}\n", &eq("a", "1")),
            "whitespace around a record"
        );
        assert!(eval_raw_record(
            b"level=info msg=\"it broke\" n=3",
            &eq("msg", "it broke")
        ));
        assert!(!eval_raw_record(
            b"plain text",
            &Predicate::FieldExists("a".into())
        ));
    }

    #[test]
    fn needle_and_literal_overlap() {
        let lits: [&[u8]; 2] = [b"\",\"repo\":{\"id\":", b"}"];
        let over = |n: &str| literals_can_overlap_needle(lits.iter().copied(), n.as_bytes());
        assert!(!over("clickhouse"));
        assert!(over("x\",\"re") && over("d\":12") && over("3}") && over("a}b"));
        assert!(literals_can_overlap_needle(
            [b"".as_slice()].into_iter(),
            b"ab"
        ));
    }
}
