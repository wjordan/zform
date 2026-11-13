//! The programs, recurrences and byte tables the blocks of one model share.
//!
//! A block names a library entry instead of carrying the code itself, and gives a program's
//! block-specific constants as overrides to the entry's values. A reader resolves the names while
//! it parses a column's graph; past the parser only the engine looks at where a program came
//! from, to run the entry it has already compiled.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::graph::{FastMap, GraphNode, PayloadGraph, unpack_leaf};
use super::ir::{Len, Program, Width};
use super::recur::RecurrenceSpec;
use crate::wire::{Cursor, DecodeBudget, Error, Result, put_ivarint, put_uvarint};

const MAX_ENTRIES: usize = 4096;
const MAX_TABLE: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibEntry {
    Program(Arc<Program>),
    Recur(RecurrenceSpec),
    Bytes(Vec<u8>),
}

/// A program as a graph holds it. One read from the library remembers its entry: overrides
/// change only constants, so the engine can reuse the plan it made for the entry.
#[derive(Clone, Debug)]
pub struct Code {
    prog: Arc<Program>,
    entry: Option<Arc<Program>>,
}

impl Code {
    /// The library entry this is.
    pub(crate) fn entry(&self) -> Option<&Arc<Program>> {
        self.entry.as_ref()
    }
}

impl From<Program> for Code {
    fn from(prog: Program) -> Self {
        Self {
            prog: Arc::new(prog),
            entry: None,
        }
    }
}

impl std::ops::Deref for Code {
    type Target = Program;
    fn deref(&self) -> &Program {
        &self.prog
    }
}

impl PartialEq for Code {
    fn eq(&self, other: &Self) -> bool {
        self.prog == other.prog
    }
}

impl Eq for Code {}

/// The values a block may override: each instruction's immediate, then each fixed sink width.
pub fn program_values(p: &Program) -> Vec<i64> {
    let imms = p.ops.iter().filter(|i| i.op.has_imm()).map(|i| i.imm);
    let widths = p.sinks.iter().filter_map(|&(_, len)| match len {
        Len::Fixed(w) => Some(i64::from(w)),
        Len::Lane(_) => None,
    });
    imms.chain(widths).collect()
}

/// `p` with `values` in place of its own, in the order of [`program_values`].
fn with_values(p: &Program, values: &[i64]) -> Program {
    let mut p = p.clone();
    let mut vals = values.iter().copied();
    for ins in p.ops.iter_mut().filter(|i| i.op.has_imm()) {
        ins.imm = vals.next().unwrap_or(0);
    }
    for (_, len) in &mut p.sinks {
        if let Len::Fixed(w) = len {
            *w = vals.next().unwrap_or(0) as u8;
        }
    }
    p
}

/// What is left of `p` when its values are taken out: programs with one skeleton differ only in
/// constants.
fn skeleton(p: &Program) -> Vec<u8> {
    let zeros = vec![0i64; program_values(p).len()];
    with_values(p, &zeros).encode()
}

#[derive(Clone, Debug, Default)]
pub struct Library {
    entries: Vec<LibEntry>,
    /// Entry ids by program skeleton, recurrence bytes, and table bytes.
    by_skeleton: HashMap<Vec<u8>, Vec<u32>>,
    by_recur: HashMap<Vec<u8>, u32>,
    by_bytes: HashMap<Vec<u8>, u32>,
    /// What [`Library::link`] found for programs seen before: an encoder asks about the same few
    /// programs for every column of every block.
    links: Arc<Mutex<FastMap<Program, Link>>>,
}

/// A program's inline size and, when an entry has its shape, the reference that stands for it.
type Link = (usize, Option<Vec<u8>>);

impl PartialEq for Library {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl Library {
    pub fn new(entries: Vec<LibEntry>) -> Self {
        let mut lib = Self::default();
        for (id, e) in entries.iter().enumerate() {
            let id = id as u32;
            match e {
                LibEntry::Program(p) => lib.by_skeleton.entry(skeleton(p)).or_default().push(id),
                LibEntry::Recur(s) => {
                    lib.by_recur.entry(s.encode()).or_insert(id);
                }
                LibEntry::Bytes(b) => {
                    lib.by_bytes.entry(b.clone()).or_insert(id);
                }
            }
        }
        lib.entries = entries;
        lib
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[LibEntry] {
        &self.entries
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        put_uvarint(out, self.entries.len() as u64);
        for e in &self.entries {
            let (kind, body) = match e {
                LibEntry::Program(p) => (0u8, p.encode()),
                LibEntry::Recur(s) => (1, s.encode()),
                LibEntry::Bytes(b) => (2, b.clone()),
            };
            out.push(kind);
            put_uvarint(out, body.len() as u64);
            out.extend_from_slice(&body);
        }
    }

    pub fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        let n = cur.uvar()? as usize;
        if n > MAX_ENTRIES {
            return Err(Error::Format("program library too large".into()));
        }
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            let kind = cur.u8()?;
            let body = cur.bytes()?;
            entries.push(match kind {
                0 => LibEntry::Program(Arc::new(Program::decode(body)?)),
                1 => LibEntry::Recur(RecurrenceSpec::decode(body)?),
                2 => LibEntry::Bytes(body.to_vec()),
                _ => return Err(Error::Format("unknown library entry kind".into())),
            });
        }
        Ok(Self::new(entries))
    }

    fn entry(&self, id: u64) -> Result<&LibEntry> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.entries.get(i))
            .ok_or_else(|| Error::Format("library entry out of bounds".into()))
    }

    /// The program a reference names: entry `id` with the overrides that follow in `cur`.
    pub(crate) fn read_program(&self, id: u64, cur: &mut Cursor<'_>) -> Result<Code> {
        let LibEntry::Program(base) = self.entry(id)? else {
            return Err(Error::Format("library entry is not a program".into()));
        };
        let n = cur.uvar()? as usize;
        let n_imm = base.ops.iter().filter(|i| i.op.has_imm()).count();
        let mut prog = base.clone();
        let mut at = 0usize;
        for i in 0..n {
            let skip = cur.uvar()? as usize;
            at = at
                .checked_add(skip + usize::from(i > 0))
                .ok_or_else(|| Error::Format("program override out of bounds".into()))?;
            let value = cur.ivar()?;
            // An override gets the checks an inline program gets.
            let p = Arc::make_mut(&mut prog);
            let (width, mut imms) = (p.width, p.ops.iter_mut().filter(|i| i.op.has_imm()));
            let ok = if let Some(ins) = imms.nth(at) {
                ins.imm = value;
                ins.op.imm_ok(width, value)
            } else {
                let fixed = p.sinks.iter_mut().filter_map(|(_, len)| match len {
                    Len::Fixed(w) => Some(w),
                    Len::Lane(_) => None,
                });
                let max = if width == Width::W32 { 4 } else { 8 };
                match fixed.into_iter().nth(at - n_imm) {
                    Some(w) => {
                        *w = value as u8;
                        (1..=max).contains(&value)
                    }
                    None => false,
                }
            };
            if !ok {
                return Err(Error::Format("invalid program override".into()));
            }
        }
        Ok(Code {
            prog,
            entry: Some(base.clone()),
        })
    }

    pub(crate) fn read_recur(&self, id: u64) -> Result<RecurrenceSpec> {
        match self.entry(id)? {
            LibEntry::Recur(s) => Ok(s.clone()),
            _ => Err(Error::Format("library entry is not a recurrence".into())),
        }
    }

    pub(crate) fn read_bytes(&self, id: u64) -> Result<&[u8]> {
        match self.entry(id)? {
            LibEntry::Bytes(b) => Ok(b),
            _ => Err(Error::Format("library entry is not a table".into())),
        }
    }

    pub(crate) fn link(&self, p: &Program) -> Link {
        let mut links = self.links.lock().unwrap();
        if let Some(hit) = links.get(p) {
            return hit.clone();
        }
        let mut reference = Vec::new();
        let found = self.reference(p, &mut reference);
        // Inline, a program follows its length tag.
        let mut inline = p.encode();
        let len = inline.len() as u64;
        put_uvarint(&mut inline, len << 1);
        let smaller = found && reference.len() < inline.len();
        let link = (inline.len(), smaller.then_some(reference));
        if links.len() >= 4096 {
            links.clear();
        }
        links.insert(p.clone(), link.clone());
        link
    }

    /// Writes a smaller reference to the nearest entry, or returns false to keep code inline.
    pub(crate) fn write_program(&self, p: &Program, out: &mut Vec<u8>) -> bool {
        match self.link(p).1 {
            Some(reference) => {
                out.extend_from_slice(&reference);
                true
            }
            None => false,
        }
    }

    fn reference(&self, p: &Program, out: &mut Vec<u8>) -> bool {
        let Some(ids) = self.by_skeleton.get(&skeleton(p)) else {
            return false;
        };
        let values = program_values(p);
        let diffs = |id: u32| -> Vec<(usize, i64)> {
            let LibEntry::Program(base) = &self.entries[id as usize] else {
                unreachable!()
            };
            let base = program_values(base);
            (0..values.len())
                .filter(|&i| values[i] != base[i])
                .map(|i| (i, values[i]))
                .collect()
        };
        let (id, over) = ids
            .iter()
            .map(|&id| (id, diffs(id)))
            .min_by_key(|(_, d)| d.len())
            .unwrap();
        put_uvarint(out, (u64::from(id) << 1) | 1);
        put_uvarint(out, over.len() as u64);
        let mut prev = None;
        for (i, v) in over {
            put_uvarint(out, prev.map_or(i, |p: usize| i - p - 1) as u64);
            put_ivarint(out, v);
            prev = Some(i);
        }
        true
    }

    pub(crate) fn find_recur(&self, wire: &[u8]) -> Option<u32> {
        self.by_recur.get(wire).copied()
    }

    pub(crate) fn find_bytes(&self, raw: &[u8]) -> Option<u32> {
        self.by_bytes.get(raw).copied()
    }

    pub(crate) fn has_tables(&self) -> bool {
        !self.by_bytes.is_empty()
    }
}

/// The bytes of a `Bytes` node that can be a shared table: small and stored uncompressed, as
/// writers store constants. A block's own data is compressed, or too short to matter.
pub(crate) fn table_bytes(node: &GraphNode) -> Option<Vec<u8>> {
    match node {
        GraphNode::Bytes { leaf } if leaf.first() == Some(&0) && leaf.len() <= MAX_TABLE + 16 => {
            unpack_leaf(leaf, &DecodeBudget::default()).ok()
        }
        _ => None,
    }
}

/// The code a set of column payloads uses, counted, from which a library is drawn.
#[derive(Default)]
pub struct Harvest {
    /// Each distinct program with its skeleton and use count, in first-use order.
    programs: Vec<(Program, Vec<u8>, u32)>,
    program_ids: HashMap<Vec<u8>, usize>,
    /// Each distinct recurrence or table by its bytes, in first-use order.
    others: Vec<(LibEntry, u32)>,
    other_ids: HashMap<(u8, Vec<u8>), usize>,
}

impl Harvest {
    /// Counts the code in a column payload written without a library.
    pub fn add_blob(&mut self, blob: &[u8]) {
        let Ok(graph) = PayloadGraph::decode(blob, &DecodeBudget::default()) else {
            return;
        };
        for node in &graph.nodes {
            match node {
                GraphNode::ProgramI64 { prog, bufs, .. }
                | GraphNode::ProgramCells { prog, bufs, .. } => {
                    let at = *self.program_ids.entry(prog.encode()).or_insert_with(|| {
                        self.programs
                            .push((Program::clone(prog), skeleton(prog), 0));
                        self.programs.len() - 1
                    });
                    self.programs[at].2 += 1;
                    for &b in bufs {
                        if let Some(raw) = table_bytes(&graph.nodes[b as usize]) {
                            self.add_other(2, raw.clone(), || LibEntry::Bytes(raw));
                        }
                    }
                }
                GraphNode::RecurBuf { spec, .. } => {
                    self.add_other(1, spec.encode(), || LibEntry::Recur(spec.clone()));
                }
                _ => {}
            }
        }
    }

    fn add_other(&mut self, kind: u8, key: Vec<u8>, entry: impl FnOnce() -> LibEntry) {
        let at = *self.other_ids.entry((kind, key)).or_insert_with(|| {
            self.others.push((entry(), 0));
            self.others.len() - 1
        });
        self.others[at].1 += 1;
    }

    /// The library worth storing. Code used once is left inline unless `keep_single`, which says
    /// later blocks are expected to use it too.
    pub fn into_library(self, keep_single: bool) -> Library {
        let min_uses = if keep_single { 1 } else { 2 };
        let mut by_shape: HashMap<&[u8], u32> = HashMap::new();
        for (_, skel, n) in &self.programs {
            *by_shape.entry(skel).or_default() += n;
        }
        // A program blocks repeat exactly goes in as it is. A shape whose constants change every
        // time goes in once, and each use overrides the constants.
        let mut covered: HashMap<&[u8], bool> = HashMap::new();
        for (_, skel, n) in &self.programs {
            *covered.entry(skel).or_default() |= *n >= 2;
        }
        let mut entries = Vec::new();
        let mut placed: HashMap<&[u8], bool> = HashMap::new();
        for (prog, skel, n) in &self.programs {
            let repeats = *n >= 2;
            let stands_in = !covered[skel.as_slice()]
                && by_shape[skel.as_slice()] >= min_uses
                && !placed.contains_key(skel.as_slice());
            if repeats || stands_in {
                placed.insert(skel, true);
                entries.push(LibEntry::Program(Arc::new(prog.clone())));
            }
        }
        for (entry, n) in self.others {
            if n >= min_uses {
                entries.push(entry);
            }
        }
        entries.truncate(MAX_ENTRIES);
        Library::new(entries)
    }
}

thread_local! {
    static PRICING: RefCell<Option<(Arc<Library>, u32)>> = const { RefCell::new(None) };
}

/// Runs `f` pricing code for an encoder that will share it: code in `lib` costs its reference,
/// and other code a `1 / share` part of its size, `share` being the blocks expected to use it.
pub fn with_code_pricing<R>(lib: Arc<Library>, share: u32, f: impl FnOnce() -> R) -> R {
    struct Guard(Option<(Arc<Library>, u32)>);
    impl Drop for Guard {
        fn drop(&mut self) {
            PRICING.with(|p| *p.borrow_mut() = self.0.take());
        }
    }
    let active = (!lib.is_empty() || share > 1).then_some((lib, share.max(1)));
    let _guard = Guard(PRICING.with(|p| p.replace(active)));
    f()
}

/// How many of `graph`'s inline bytes the active pricing does not charge.
pub(crate) fn code_discount(graph: &PayloadGraph) -> usize {
    PRICING.with(|p| {
        let p = p.borrow();
        let Some((lib, share)) = p.as_ref() else {
            return 0;
        };
        let share = *share as usize;
        let mut saved = 0usize;
        let mut tables = Vec::new();
        for node in &graph.nodes {
            let (inline, shared) = match node {
                GraphNode::ProgramI64 { prog, bufs, .. }
                | GraphNode::ProgramCells { prog, bufs, .. } => {
                    tables.extend(bufs.iter().copied());
                    let (inline, reference) = lib.link(prog);
                    (inline, reference.map(|r| r.len()))
                }
                GraphNode::RecurBuf { spec, .. } => {
                    let wire = spec.encode();
                    (wire.len(), lib.find_recur(&wire).map(|_| 2))
                }
                _ => continue,
            };
            saved += inline.saturating_sub(shared.unwrap_or(inline.div_ceil(share)));
        }
        tables.sort_unstable();
        tables.dedup();
        for t in tables {
            if let Some(raw) = table_bytes(&graph.nodes[t as usize]) {
                let shared = lib.find_bytes(&raw).map(|_| 2);
                saved += raw
                    .len()
                    .saturating_sub(shared.unwrap_or(raw.len().div_ceil(share)));
            }
        }
        saved
    })
}
