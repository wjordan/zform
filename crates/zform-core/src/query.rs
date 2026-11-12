//! Predicates, and their evaluation on a raw record and on a block's columns.
//!
//! The raw record defines what a predicate means. A block answers from its columns for the rows
//! whose model makes that exact, and reports the rest as unknown for the caller to rebuild.

use crate::cells::{Col, NULL_CELL, underflow};
use crate::codecs::{ColSummary, ESCAPES, letters_of};
use crate::container::BlockRows;
use crate::grammar::{ModelPlan, PlanVariant, parse_logfmt_line};
use crate::index::{
    BlockDir, Filter, KIND_BINARY, KIND_CSV, KIND_ENVELOPE, KIND_JSON, KIND_LOGFMT, KIND_RESIDUAL,
    KIND_TEMPLATE, needle_tokens,
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
    /// Every value comes from `leaf_cols` or `subtree_cols`.
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
                fp.prunable &= p.cands.is_empty();
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

    /// The column whose cell `i` is the one value the field has in row `i`, in a block where
    /// that holds for every row.
    fn row_col(&self, kinds: u16, dir: &BlockDir) -> Option<u16> {
        let live = |c: u16| !dir.col(c as usize).summary.all_null;
        if !self.prunable || kinds != KIND_JSON || self.subtree_cols.iter().any(|&(c, _)| live(c)) {
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
        for v in &plan.variants {
            let (var, kind) = match v {
                PlanVariant::Residual { col } => {
                    use_cols(&mut std::iter::once(*col), true);
                    (VarNeedle::Scan, 0)
                }
                PlanVariant::Template(tv) => {
                    let mut slots = tv.slots.clone();
                    slots.sort_unstable();
                    slots.dedup();
                    use_cols(&mut slots.into_iter(), true);
                    let n = tv.literals.len();
                    let inner = tv
                        .literals
                        .iter()
                        .enumerate()
                        .filter(|(i, l)| !(l.is_empty() && (*i == 0 || *i + 1 == n)));
                    let var = if tv.literals.iter().any(|l| sure(l)) {
                        VarNeedle::AllTrue
                    } else if tv.literals.iter().any(|l| has(l) || !joins(l))
                        || literals_can_overlap_needle(inner.map(|(_, l)| l.as_slice()), needle)
                    {
                        VarNeedle::Unknown
                    } else {
                        VarNeedle::Scan
                    };
                    (var, KIND_TEMPLATE)
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
                    for p in &jt.paths {
                        for shape in &p.shapes {
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
                        for cand in &p.cands {
                            match cand {
                                JsonCand::Copy { .. } => {}
                                JsonCand::Affix { prefix, suffix, .. } => {
                                    lits.extend(
                                        [prefix.as_slice(), suffix.as_slice()]
                                            .into_iter()
                                            .filter(|l| !l.is_empty()),
                                    );
                                }
                                JsonCand::MultiConcat { lits: ls, .. } => {
                                    let n = ls.len();
                                    lits.extend(
                                        ls.iter()
                                            .enumerate()
                                            .filter(|(i, l)| {
                                                !(l.is_empty() && (*i == 0 || *i + 1 == n))
                                            })
                                            .map(|(_, l)| l.as_slice()),
                                    );
                                }
                            }
                        }
                    }
                    let exact = joined
                        && !lits.iter().any(|l| has(l))
                        && !literals_can_overlap_needle(lits.into_iter(), needle);
                    (
                        if exact {
                            VarNeedle::Scan
                        } else {
                            VarNeedle::Unknown
                        },
                        KIND_JSON,
                    )
                }
                PlanVariant::Envelope(env) => {
                    use_cols(&mut env.cols.into_iter(), false);
                    (VarNeedle::Unknown, KIND_ENVELOPE)
                }
                PlanVariant::Logfmt => {
                    let lf_cols = plan.columns.iter().enumerate().filter(|(_, c)| {
                        c.name.starts_with("logfmt.") || c.name.starts_with("$logfmt")
                    });
                    use_cols(&mut lf_cols.map(|(i, _)| i as u16), false);
                    if let Some(lf) = &plan.logfmt {
                        use_cols(&mut [lf.recipe_col, lf.overflow_col].into_iter(), false);
                    }
                    (VarNeedle::Unknown, KIND_LOGFMT)
                }
                PlanVariant::Csv(cs) => {
                    use_cols(
                        &mut cs.field_cols.iter().copied().chain([cs.quote_mask_col]),
                        false,
                    );
                    (VarNeedle::Unknown, KIND_CSV)
                }
                PlanVariant::BinaryStride(bs) => {
                    use_cols(&mut bs.slices.iter().map(|s| s.2), false);
                    (VarNeedle::Unknown, KIND_BINARY)
                }
            };
            if var == VarNeedle::Unknown {
                unknown_kinds |= kind;
            }
            vars.push(var);
        }
        // A column two variants write has occurrences this code cannot place on rows.
        let scan_cols: Vec<bool> = (0..n_cols).map(|c| users[c] == 1 && verbatim[c]).collect();
        for (vi, v) in plan.variants.iter().enumerate() {
            let shared = |c: u16| users[c as usize] > 1;
            let (bad, kind) = match v {
                PlanVariant::Residual { col } => (shared(*col), KIND_RESIDUAL),
                PlanVariant::Template(tv) => (tv.slots.iter().any(|&c| shared(c)), KIND_TEMPLATE),
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
        NeedlePlan {
            vars,
            unknown_kinds,
            open_kinds: unknown_kinds | if all_true { KIND_TEMPLATE } else { 0 },
            scan_cols,
            letters: letters_of(needle),
            finder: finder.into_owned(),
            word,
            quote_sensitive: needle.contains(&b'"')
                || memchr::memmem::find(b"null", needle).is_some(),
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
}

impl Need {
    pub fn new(dir: &BlockDir) -> Self {
        Need {
            whole: false,
            cols: vec![false; dir.n_cols()],
            streams: vec![false; dir.streams.len()],
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
            // A column with a cell for every row needs no structure to place its cells.
            let mut want = vec![false; need.cols.len()];
            for c in np.candidates(m) {
                (need.cols[c], want[c]) = (true, !m.dir.col(c).summary.per_row);
            }
            if let Some(jt) = &m.plan.json_tree {
                let streams = WalkRoute::of(
                    jt,
                    &JsonWalk {
                        want_cols: &want,
                        ..JsonWalk::default()
                    },
                )
                .streams(jt);
                need.streams
                    .iter_mut()
                    .zip(streams)
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
    for &c in fp.leaf_cols.iter().filter(|&&c| !summary(c).all_null) {
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
        let PathVal::Occ { col, occ } = ev.val else {
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
    if hits.is_empty() {
        return Ok(tri);
    }

    // The row of each occurrence of each column left.
    let mut occ_rows = match (&plan.json_tree, json_variant(plan)) {
        (Some(jt), Some(jvi)) => {
            let walk = JsonWalk {
                want_cols: &want,
                ..JsonWalk::default()
            };
            let route = WalkRoute::of(jt, &walk);
            map_json_block_structure(
                jt,
                rows.json_streams(),
                &rows.variant_ids,
                jvi,
                &walk,
                &route,
            )?
            .col_occ_rows
        }
        _ => vec![Vec::new(); plan.columns.len()],
    };
    for (r, &v) in rows.variant_ids.iter().enumerate() {
        match &plan.variants[v as usize] {
            PlanVariant::Residual { col } if want[*col as usize] => {
                occ_rows[*col as usize].push(r as u32)
            }
            PlanVariant::Template(tv) => {
                for &c in tv.slots.iter().filter(|&&c| want[c as usize]) {
                    occ_rows[c as usize].push(r as u32);
                }
            }
            _ => {}
        }
    }
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
    Ok(tri)
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
            scan_buf(&pool[*base..*base + len], ends, hit)
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
            let mut stem = |k: usize, out: &mut Vec<u8>, unquote: bool| -> Result<()> {
                let src = map.events[map.srcs[srcs as usize + k] as usize];
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
                JsonCand::MultiConcat { lits, refs } => {
                    out.extend_from_slice(&lits[0]);
                    for k in 0..refs.len() {
                        stem(k, out, true)?;
                        out.extend_from_slice(&lits[k + 1]);
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
