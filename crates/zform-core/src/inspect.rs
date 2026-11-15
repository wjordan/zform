//! Where a stream's bytes go. [`inspect_breakdown`] attributes every byte to a column, a block
//! part, a nested model's parts, or framing, and records the graph each column was coded with.
//! It only parses the wire: no cells are decoded.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Read;
use std::sync::Arc;

use crate::tables::TABLE_PEER;
use crate::container::{
    BlockParts, FLAG_DICTIONARY, FLAG_HAS_CHECKSUM, FLAG_PASSTHROUGH, FLAG_REUSE_PREV_MODEL,
    FLAG_SYNC_POINT, InspectReport, NextFrameKind, Structure, parse_frame_blocks,
    read_next_frame_header,
};
use crate::grammar::{ColumnDef, ModelPlan};
use crate::payload::graph::{
    GraphNode, IntStream, NESTED_COLUMN_TAG, PayloadGraph, decode_linked_sizes,
};
use crate::payload::ir::{Len, Program};
use crate::payload::library::{Code, LibEntry, Library};
use crate::wire::{Cursor, DecodeBudget, Error, Result};

/// Frame and column totals, without the per-column graph parse [`inspect_breakdown`] does.
pub fn inspect_stream<R: Read>(reader: R, budget: &DecodeBudget) -> Result<InspectReport> {
    walk(reader, budget, None)
}

/// [`inspect_stream`], plus a tree under which every byte read adds up. Top-level children are
/// the report's `column_bytes` plus `$model`, `$index`, `$passthrough` and `$overhead` (frame
/// headers, block directories, block framing).
pub fn inspect_breakdown<R: Read>(
    reader: R,
    budget: &DecodeBudget,
) -> Result<(InspectReport, SizeNode)> {
    let mut root = SizeNode::default();
    let mut counted = Counted {
        inner: reader,
        n: 0,
    };
    let rep = walk(&mut counted, budget, Some(&mut root))?;
    // The report takes every frame header as 24 bytes; the tree counts what was read.
    root.bytes = counted.n;
    let rest = root
        .bytes
        .saturating_sub(root.children.iter().map(|c| c.bytes).sum());
    if rest > 0 {
        root.child("$overhead").bytes += rest;
    }
    Ok((rep, root))
}

struct Counted<R> {
    inner: R,
    n: u64,
}

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let k = self.inner.read(buf)?;
        self.n += k as u64;
        Ok(k)
    }
}

fn walk<R: Read>(
    mut reader: R,
    budget: &DecodeBudget,
    mut tree: Option<&mut SizeNode>,
) -> Result<InspectReport> {
    let mut rep = InspectReport::default();
    let mut prev_plan: Option<ModelPlan> = None;
    loop {
        match read_next_frame_header(&mut reader, budget)? {
            NextFrameKind::Eof => break,
            NextFrameKind::Index { skip_bytes } => {
                rep.index_bytes += skip_bytes as u64 + 12;
                if let Some(t) = tree.as_deref_mut() {
                    t.child("$index").bytes += skip_bytes as u64 + 12;
                }
                std::io::copy(
                    &mut (&mut reader).take(skip_bytes as u64),
                    &mut std::io::sink(),
                )?;
                prev_plan = None;
            }
            NextFrameKind::Data(hdr) if (hdr.flags & FLAG_DICTIONARY) != 0 => {
                std::io::copy(
                    &mut (&mut reader).take(hdr.payload_len as u64 + 8),
                    &mut std::io::sink(),
                )?;
                rep.total_compressed_bytes += hdr.payload_len as u64 + 32;
                rep.column_bytes
                    .push(("$dictionary".into(), hdr.payload_len as u64));
                if let Some(t) = tree.as_deref_mut() {
                    t.child("$dictionary").bytes += hdr.payload_len as u64;
                }
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
                if (hdr.flags & FLAG_PASSTHROUGH) != 0 {
                    if let Some(t) = tree.as_deref_mut() {
                        t.child("$passthrough").bytes += hdr.payload_len as u64;
                    }
                    continue;
                }
                let bdir = parse_frame_blocks(&hdr, &payload, &mut prev_plan, budget)?;
                let Some(p) = &prev_plan else { continue };
                if let Some(t) = tree.as_deref_mut()
                    && (hdr.flags & FLAG_REUSE_PREV_MODEL) == 0
                {
                    t.child("$model").bytes += leaf_len(&payload).unwrap_or(0) as u64;
                }
                rep.variant_count = rep.variant_count.max(p.variants.len());
                for c in &p.columns {
                    if !rep.column_names.contains(&c.name) {
                        rep.column_names.push(c.name.clone());
                    }
                }
                for &(_, bslice) in &bdir.blocks {
                    let Ok(pb) = BlockParts::of_block(bslice, p, budget) else {
                        continue;
                    };
                    if let Structure::Whole(s) = pb.structure {
                        add_stat(&mut rep, "$structure", s.len() as u64);
                    }
                    let sd_bytes: u64 = pb.dicts.iter().flatten().map(|b| b.len() as u64).sum();
                    add_stat(&mut rep, "$shared_dicts", sd_bytes);
                    let pool = pb.pool.as_ref().map_or(0, |b| b.held_bytes() as u64);
                    add_stat(&mut rep, "$prose_pool", pool);
                    for (cid, blob) in pb.cols.iter().enumerate() {
                        let nb = blob.map_or(0, |b| b.len() as u64);
                        add_stat(&mut rep, &p.columns[cid].name, nb);
                    }
                    if let Some(t) = tree.as_deref_mut() {
                        add_parts(t, "", &pb, p, budget);
                    }
                }
            }
        }
    }
    Ok(rep)
}

fn add_stat(rep: &mut InspectReport, name: &str, nb: u64) {
    if nb == 0 {
        return;
    }
    if let Some(ent) = rep.column_bytes.iter_mut().find(|x| x.0 == name) {
        ent.1 += nb;
    } else {
        rep.column_bytes.push((name.to_string(), nb));
    }
}

/// The wire length of the leaf `b` starts with.
fn leaf_len(b: &[u8]) -> Option<usize> {
    let mut cur = Cursor::new(b);
    cur.u8().ok()?;
    cur.uvar().ok()?;
    let n = usize::try_from(cur.uvar().ok()?).ok()?;
    Some(cur.pos + n.min(cur.remaining()))
}

/// Byte totals for one part of the stream, and its parts.
#[derive(Debug, Clone, Default)]
pub struct SizeNode {
    /// Dotted path: a nested model's parts are `<column>.<part>`.
    pub name: String,
    pub bytes: u64,
    pub children: Vec<SizeNode>,
    /// The graphs this part was coded with, one entry per distinct shape.
    pub sigs: Vec<Signature>,
    index: HashMap<String, usize>,
}

/// A graph shape and the bytes every blob of that shape spent.
#[derive(Debug, Clone, Default)]
pub struct Signature {
    /// Nodes from the root down, `a<-b` or `a<-[b, c]`; each leaf ends in `\0`.
    pub shape: String,
    pub blobs: u64,
    /// Whole wire bytes: the leaves, plus graph header, operator nodes and code.
    pub bytes: u64,
    /// Each leaf's kind and wire bytes, in the order `shape` names them.
    pub leaves: Vec<(String, u64)>,
}

impl Signature {
    fn of(shape: String, bytes: usize, leaves: Vec<(String, u64)>) -> Self {
        Self {
            shape,
            blobs: 1,
            bytes: bytes as u64,
            leaves,
        }
    }

    /// The shape, each leaf followed by its bytes.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut leaves = self.leaves.iter();
        for part in self.shape.split_inclusive('\0') {
            match part.strip_suffix('\0') {
                Some(head) => {
                    let n = leaves.next().map_or(0, |l| l.1);
                    let _ = write!(out, "{head} {}", human(n));
                }
                None => out.push_str(part),
            }
        }
        let graph = self.bytes - self.leaves.iter().map(|l| l.1).sum::<u64>();
        if graph * 50 >= self.bytes && graph > 0 {
            let _ = write!(out, " +{} graph", human(graph));
        }
        out
    }
}

impl SizeNode {
    fn named(name: String) -> Self {
        Self {
            name,
            ..Self::default()
        }
    }

    /// The child named `name`, made empty if absent.
    pub fn child(&mut self, name: &str) -> &mut SizeNode {
        let i = match self.index.get(name) {
            Some(&i) => i,
            None => {
                self.index.insert(name.to_string(), self.children.len());
                self.children.push(Self::named(name.to_string()));
                self.children.len() - 1
            }
        };
        &mut self.children[i]
    }

    pub fn find(&self, name: &str) -> Option<&SizeNode> {
        self.index.get(name).map(|&i| &self.children[i])
    }

    fn add_sig(&mut self, s: Signature) {
        match self.sigs.iter_mut().find(|x| x.shape == s.shape) {
            Some(x) => {
                x.blobs += s.blobs;
                x.bytes += s.bytes;
                for (a, b) in x.leaves.iter_mut().zip(&s.leaves) {
                    a.1 += b.1;
                }
            }
            None => self.sigs.push(s),
        }
    }

    fn merge(&mut self, other: SizeNode) {
        self.bytes += other.bytes;
        for s in other.sigs {
            self.add_sig(s);
        }
        for c in other.children {
            let name = c.name.clone();
            self.child(&name).merge(c);
        }
    }

    fn sorted_children(&self) -> Vec<&SizeNode> {
        let mut v: Vec<&SizeNode> = self.children.iter().collect();
        v.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
        v
    }

    fn dominant(&self) -> Option<&Signature> {
        self.sigs.iter().max_by_key(|s| s.bytes)
    }

    /// The children as an indented table, largest first: bytes, share of `total`, bytes per
    /// record, and the signature that spent the most bytes.
    pub fn tree(&self, total: u64, records: u64) -> String {
        let mut rows = Vec::new();
        self.tree_rows(0, total, &mut rows);
        let width = rows.iter().map(|r| r.1.len()).max().unwrap_or(0).min(48);
        let mut out = format!(
            "{:>12} {:>6} {:>8}  {:<width$}  dominant signature\n",
            "bytes", "%", "B/rec", "part"
        );
        for (bytes, name, sig) in rows {
            let pct = 100.0 * bytes as f64 / total.max(1) as f64;
            let per = bytes as f64 / records.max(1) as f64;
            let _ = writeln!(
                out,
                "{:>12} {pct:>5.2}% {per:>8.3}  {name:<width$}  {sig}",
                commas(bytes)
            );
        }
        out
    }

    fn tree_rows(&self, depth: usize, total: u64, rows: &mut Vec<(u64, String, String)>) {
        let kids = self.sorted_children();
        let shown = kids
            .iter()
            .enumerate()
            .take_while(|(i, c)| *i < 40 && (c.bytes * 10_000 >= total || *i < 8))
            .count();
        for c in &kids[..shown] {
            let mut sig = c.dominant().map_or_else(String::new, Signature::render);
            if let Some(dom) = c.dominant().filter(|_| c.sigs.len() > 1) {
                let all: u64 = c.sigs.iter().map(|s| s.blobs).sum();
                let share = 100 * dom.bytes / c.bytes.max(1);
                let n = c.sigs.len();
                let _ = write!(
                    sig,
                    "  [{}/{all} blobs, {share}% of bytes; {n} shapes]",
                    dom.blobs
                );
            }
            rows.push((c.bytes, format!("{}{}", "  ".repeat(depth), c.name), sig));
            c.tree_rows(depth + 1, total, rows);
        }
        if shown < kids.len() {
            let rest = &kids[shown..];
            rows.push((
                rest.iter().map(|c| c.bytes).sum(),
                format!("{}({} more)", "  ".repeat(depth), rest.len()),
                String::new(),
            ));
        }
    }

    /// This node and everything under it as one JSON object.
    pub fn json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        out.push_str("{\"name\":");
        json_str(out, &self.name);
        let _ = write!(out, ",\"bytes\":{}", self.bytes);
        if !self.sigs.is_empty() {
            let mut sigs: Vec<&Signature> = self.sigs.iter().collect();
            sigs.sort_by(|a, b| b.bytes.cmp(&a.bytes));
            out.push_str(",\"signatures\":[");
            for (i, s) in sigs.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str("{\"shape\":");
                json_str(out, &s.shape.replace('\0', ""));
                let _ = write!(
                    out,
                    ",\"blobs\":{},\"bytes\":{},\"leaves\":[",
                    s.blobs, s.bytes
                );
                for (j, (kind, n)) in s.leaves.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    out.push_str("{\"kind\":");
                    json_str(out, kind);
                    let _ = write!(out, ",\"bytes\":{n}}}");
                }
                out.push_str("]}");
            }
            out.push(']');
        }
        if !self.children.is_empty() {
            out.push_str(",\"children\":[");
            for (i, c) in self.sorted_children().into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                c.write_json(out);
            }
            out.push(']');
        }
        out.push('}');
    }
}

/// Attributes the parts of one block to children of `node` named `prefix<part>`, and returns
/// the bytes attributed. The rest of the block is its framing.
fn add_parts(
    node: &mut SizeNode,
    prefix: &str,
    parts: &BlockParts<'_>,
    plan: &ModelPlan,
    budget: &DecodeBudget,
) -> u64 {
    let lib = &plan.library;
    let mut sum = 0u64;
    if let Structure::Whole(s) = parts.structure {
        let n = node.child(&format!("{prefix}$structure"));
        n.bytes += s.len() as u64;
        n.add_sig(structure_sig(s, plan.json_tree.is_some(), budget));
        sum += s.len() as u64;
    }
    for d in parts.dicts.iter().flatten() {
        let n = node.child(&format!("{prefix}$shared_dicts"));
        n.bytes += d.len() as u64;
        n.add_sig(graph_sig(d, lib, &plan.columns, budget));
        sum += d.len() as u64;
    }
    let pool = parts.pool.as_ref().map_or(0, |p| p.held_bytes() as u64);
    if pool > 0 {
        node.child(&format!("{prefix}$prose_pool")).bytes += pool;
        sum += pool;
    }
    for (col, blob) in plan.columns.iter().zip(&parts.cols) {
        let Some(blob) = blob.filter(|b| !b.is_empty()) else {
            continue;
        };
        let name = format!("{prefix}{}", col.name);
        sum += blob.len() as u64;
        match &col.nested_plan {
            Some(child) => {
                let mut one = SizeNode::named(name.clone());
                if add_nested(&mut one, blob, child, budget).is_err() {
                    one = SizeNode::named(name.clone());
                    one.bytes = blob.len() as u64;
                    one.child(&format!("{name}.$unparsed")).bytes = blob.len() as u64;
                }
                node.child(&name).merge(one);
            }
            None => {
                let n = node.child(&name);
                n.bytes += blob.len() as u64;
                n.add_sig(graph_sig(blob, lib, &plan.columns, budget));
            }
        }
    }
    sum
}

/// Splits a nested column, laid out as `container.rs` `decode_nested_column_tagged` reads it.
fn add_nested(
    node: &mut SizeNode,
    blob: &[u8],
    child_plan: &ModelPlan,
    budget: &DecodeBudget,
) -> Result<()> {
    let pre = format!("{}.", node.name);
    node.bytes += blob.len() as u64;
    let mut cur = Cursor::new(blob);
    let mut framing = 0usize;
    if cur.uvar()? == 0 {
        framing = blob.len();
    } else {
        if cur.u8()? != NESTED_COLUMN_TAG {
            return Err(Error::Format("expected nested column tag".into()));
        }
        cur.u8()?;
        framing += cur.pos;
        let at = cur.pos;
        let modes = IntStream::decode_wire(&mut cur, budget, 0)?;
        let n = cur.pos - at;
        let kind = format!("IntStream:{}", int_kind(&modes));
        let m = node.child(&format!("{pre}$row_modes"));
        m.bytes += n as u64;
        m.add_sig(Signature::of(
            format!("{kind}\0"),
            n,
            vec![(kind, n as u64)],
        ));

        let at = cur.pos;
        let child = cur.bytes()?;
        framing += cur.pos - at - child.len();
        if !child.is_empty() {
            let parts = BlockParts::of_block(child, child_plan, budget)?;
            framing += child.len() - add_parts(node, &pre, &parts, child_plan, budget) as usize;
        }
        let rest = &blob[cur.pos..];
        if !rest.is_empty() {
            let f = node.child(&format!("{pre}$fallback"));
            f.bytes += rest.len() as u64;
            let mut fc = Cursor::new(rest);
            let wire = fc.bytes()?;
            let mut sig = graph_sig(wire, &Library::default(), &[], budget);
            sig.bytes = rest.len() as u64;
            f.add_sig(sig);
        }
    }
    if framing > 0 {
        node.child(&format!("{pre}$overhead")).bytes += framing as u64;
    }
    Ok(())
}

/// The variant ids' stream, then the JSON structure streams if the model has a JSON tree.
fn structure_sig(s: &[u8], json: bool, budget: &DecodeBudget) -> Signature {
    let mut cur = Cursor::new(s);
    let Ok(ids) = IntStream::decode_wire(&mut cur, budget, 0) else {
        return Signature::of("unparsed".into(), s.len(), Vec::new());
    };
    let kind = format!("IntStream:{}", int_kind(&ids));
    let mut leaves = vec![(kind.clone(), cur.pos as u64)];
    if json {
        leaves.push(("JsonStreams".into(), cur.remaining() as u64));
        let shape = format!("[{kind}\0, JsonStreams\0]");
        return Signature::of(shape, s.len(), leaves);
    }
    Signature::of(format!("{kind}\0"), s.len(), leaves)
}

/// The shape of graph `wire`; `cols` name the columns its peer references point at.
fn graph_sig(wire: &[u8], lib: &Library, cols: &[ColumnDef], budget: &DecodeBudget) -> Signature {
    let Ok((g, sizes)) = decode_linked_sizes(wire, lib, budget) else {
        return Signature::of("unparsed".into(), wire.len(), Vec::new());
    };
    if g.nodes.is_empty() {
        return Signature::of("Empty".into(), wire.len(), Vec::new());
    }
    let mut r = Render {
        g: &g,
        sizes: &sizes,
        lib,
        cols,
        seen: vec![false; g.nodes.len()],
        shape: String::new(),
        leaves: Vec::new(),
    };
    r.node(g.root as usize);
    let unreached: Vec<usize> = (0..g.nodes.len()).filter(|&i| !r.seen[i]).collect();
    if !unreached.is_empty() {
        r.shape.push_str(" +[");
        for (k, i) in unreached.into_iter().enumerate() {
            if r.seen[i] {
                continue;
            }
            if k > 0 {
                r.shape.push_str(", ");
            }
            r.node(i);
        }
        r.shape.push(']');
    }
    let (shape, leaves) = (r.shape, r.leaves);
    Signature::of(shape, wire.len(), leaves)
}

struct Render<'a> {
    g: &'a PayloadGraph,
    sizes: &'a [usize],
    lib: &'a Library,
    cols: &'a [ColumnDef],
    seen: Vec<bool>,
    shape: String,
    leaves: Vec<(String, u64)>,
}

impl Render<'_> {
    fn node(&mut self, i: usize) {
        if self.seen[i] {
            let _ = write!(self.shape, "@{i}");
            return;
        }
        self.seen[i] = true;
        let n = &self.g.nodes[i];
        let size = self.sizes.get(i).copied().unwrap_or(0);
        let leaf = match n {
            GraphNode::Bytes { leaf } if size <= leaf.len() => Some("Bytes:lib".to_string()),
            GraphNode::Bytes { leaf } => Some(format!("Bytes/{}", backend(leaf))),
            GraphNode::IntStream(s) => Some(format!("IntStream:{}", int_kind(s))),
            GraphNode::ConstCells { .. } => Some("ConstCells".to_string()),
            _ => None,
        };
        if let Some(kind) = leaf {
            self.shape.push_str(&kind);
            self.shape.push('\0');
            self.leaves.push((kind, size as u64));
            return;
        }
        let label = self.label(n);
        self.shape.push_str(&label);
        let mut deps = Vec::new();
        n.for_each_dep(|d| deps.push(d as usize));
        match deps.as_slice() {
            [] => {}
            [d] => {
                self.shape.push_str("<-");
                self.node(*d);
            }
            _ => {
                self.shape.push_str("<-[");
                for (k, d) in deps.into_iter().enumerate() {
                    if k > 0 {
                        self.shape.push_str(", ");
                    }
                    self.node(d);
                }
                self.shape.push(']');
            }
        }
    }

    fn col(&self, slot: u16) -> String {
        match self.g.ext_peers.get(slot as usize) {
            Some(&c) if c & TABLE_PEER != 0 => format!("table.{}", c & !TABLE_PEER),
            Some(&c) => self.cols.get(c as usize).map_or("?", |c| c.name.as_str()).to_string(),
            None => "?".to_string(),
        }
    }

    fn label(&self, n: &GraphNode) -> String {
        match n {
            GraphNode::ProgramI64 { prog, .. } => format!("ProgramI64({})", code(prog, self.lib)),
            GraphNode::ProgramCells { prog, .. } => {
                format!("ProgramCells({})", code(prog, self.lib))
            }
            GraphNode::RecurBuf { spec, .. } => {
                let shared = (self.lib.entries().iter())
                    .any(|e| matches!(e, LibEntry::Recur(s) if s == spec));
                format!("RecurBuf({})", lib_name(shared, &spec.encode()))
            }
            GraphNode::ExtPool => "Pool".into(),
            GraphNode::ExtDict { .. } => "SharedDict".into(),
            GraphNode::ExtPeer { slot } => format!("Peer({})", self.col(*slot)),
            GraphNode::ExtAlign { key, .. } => format!("Align({})", self.col(*key)),
            GraphNode::ByteChunks { .. } => "ByteChunks".into(),
            GraphNode::MtfFixed { .. } => "MtfFixed".into(),
            GraphNode::PatchCells { .. } => "PatchCells".into(),
            GraphNode::MtfInts { .. } => "MtfInts".into(),
            GraphNode::SliceBuf { .. } => "SliceBuf".into(),
            GraphNode::PagedSlice { .. } => "PagedSlice".into(),
            GraphNode::IndexCells { .. } => "IndexCells".into(),
            GraphNode::TemplateCells { .. } => "TemplateCells".into(),
            GraphNode::QuoteOrNull { .. } => "QuoteOrNull".into(),
            GraphNode::ParseAsciiInt { .. } => "ParseAsciiInt".into(),
            GraphNode::MatchIndex { .. } => "MatchIndex".into(),
            GraphNode::Bytes { .. } | GraphNode::IntStream(_) | GraphNode::ConstCells { .. } => {
                unreachable!("leaves are labelled in Render::node")
            }
        }
    }
}

/// Where a program came from, the library or inline, and a hash of it with its constants
/// zeroed: one program's blocks share a shape across sync frames and constant overrides.
fn code(prog: &Code, lib: &Library) -> String {
    let shared = prog.entry().is_some_and(|e| {
        (lib.entries().iter()).any(|x| matches!(x, LibEntry::Program(p) if Arc::ptr_eq(p, e)))
    });
    let mut skeleton = Program::clone(prog);
    for ins in skeleton.ops.iter_mut().filter(|i| i.op.has_imm()) {
        ins.imm = 0;
    }
    for (_, len) in &mut skeleton.sinks {
        if let Len::Fixed(w) = len {
            *w = 0;
        }
    }
    lib_name(shared, &skeleton.encode())
}

fn lib_name(shared: bool, bytes: &[u8]) -> String {
    let h = bytes.iter().fold(0x811c_9dc5u32, |h, &b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    });
    let at = if shared { "lib" } else { "inline" };
    format!("{at}:{:06x}", h >> 8)
}

fn int_kind(s: &IntStream) -> String {
    match s {
        IntStream::ContextMix {
            bits, table_bits, ..
        } => format!("ContextMix/{bits}b/t{table_bits}"),
        IntStream::Const { .. } => "Const".into(),
        IntStream::Rice { .. } => "Rice".into(),
        IntStream::Rle { leaf, .. } => format!("Rle/{}", backend(leaf)),
        IntStream::ForBitpack { leaf, .. } => format!("ForBitpack/{}", backend(leaf)),
        IntStream::ZigZagVarint { leaf, .. } => format!("ZigZagVarint/{}", backend(leaf)),
        IntStream::ForPlanes { leaf, .. } => format!("ForPlanes/{}", backend(leaf)),
        IntStream::ZigZagBytes {
            head_leaf, tail, ..
        } => format!("ZigZagBytes/{}+{}", backend(head_leaf), int_kind(tail)),
        IntStream::ForTail255 {
            head_leaf, tail, ..
        } => format!("ForTail255/{}+{}", backend(head_leaf), int_kind(tail)),
    }
}

/// The entropy coder a leaf's tag names.
fn backend(leaf: &[u8]) -> &'static str {
    match leaf.first() {
        Some(0) => "raw",
        Some(1) => "zstd",
        Some(2) => "lz4",
        Some(3) => "zdict",
        _ => "?",
    }
}

fn human(n: u64) -> String {
    match n {
        0..1_000 => format!("{n}B"),
        1_000..1_000_000 => format!("{:.1}KB", n as f64 / 1e3),
        _ => format!("{:.1}MB", n as f64 / 1e6),
    }
}

fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block;
    use crate::container::{PackOptions, pack_stream};
    use crate::framing::FramingSpec;
    use crate::grammar::{FrontEnd, PlanVariant, Shape};
    use crate::plan::{Profile, TrainOptions};

    /// Literal `lits[0]`, column, `lits[1]`, ...: one shape over columns named `names`.
    fn shape_plan(names: &[&str], lits: &[&str]) -> ModelPlan {
        let mut plan = ModelPlan::default();
        let slots = names.iter().map(|n| plan.add_or_get_col(n)).collect();
        let literals = lits.iter().map(|l| l.as_bytes().to_vec()).collect();
        plan.push_variant(
            PlanVariant::Shape(Shape { literals, slots }),
            FrontEnd::Template,
        );
        plan.rebuild_indexes();
        plan
    }

    /// Every node's children, where it has any, add up to it; so do a graph column's signatures.
    fn check_sums(n: &SizeNode) {
        if !n.children.is_empty() {
            let sum: u64 = n.children.iter().map(|c| c.bytes).sum();
            assert_eq!(sum, n.bytes, "children of {:?}", n.name);
        } else if !n.sigs.is_empty() {
            let sum: u64 = n.sigs.iter().map(|s| s.bytes).sum();
            assert_eq!(sum, n.bytes, "signatures of {:?}", n.name);
        }
        for s in &n.sigs {
            assert!(s.leaves.iter().map(|l| l.1).sum::<u64>() <= s.bytes);
            assert_eq!(s.shape.matches('\0').count(), s.leaves.len());
        }
        n.children.iter().for_each(check_sums);
    }

    #[test]
    fn nested_parts_add_up_to_their_column_and_the_file() {
        let rows: Vec<String> = (0..3000)
            .map(|i| match i {
                _ if i % 97 == 0 => format!("{i} kernel panic"),
                _ if i >= 2700 => {
                    format!("{i} {:x}", (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                }
                _ => format!(
                    "{i} user=u{} said disk {} at {}% after {}ms",
                    i % 7,
                    i % 13,
                    i * 7 % 100,
                    i * 3
                ),
            })
            .collect();
        let details: Vec<&[u8]> = rows[..64]
            .iter()
            .filter_map(|r| r.split_once(" said ").map(|x| x.1.as_bytes()))
            .collect();
        let mut child = shape_plan(&["user", "detail"], &["user=", " said ", ""]);
        let c = child.add_or_get_col("detail") as usize;
        child.columns[c].nested_plan = Some(Box::new(block::train(
            &details,
            &TrainOptions::default(),
            None,
        )));
        let mut plan = shape_plan(&["id", "msg"], &["", " ", ""]);
        let c = plan.add_or_get_col("msg") as usize;
        plan.columns[c].nested_plan = Some(Box::new(child));

        let mut raw = rows.join("\n").into_bytes();
        raw.push(b'\n');
        let mut opts = PackOptions {
            profile: Some(Profile::from_plan(plan)),
            allow_drift_retrain: false,
            framing: FramingSpec::Lf,
            block_rows: 300,
            chunk_bytes: 64 << 10,
            ..PackOptions::default()
        };
        opts.packs = opts.packs.with_plain_nested_fallback();
        let mut zf = Vec::new();
        pack_stream(raw.as_slice(), &mut zf, &opts).unwrap();

        let budget = DecodeBudget::default();
        let rep = inspect_stream(zf.as_slice(), &budget).unwrap();
        let (deep, root) = inspect_breakdown(zf.as_slice(), &budget).unwrap();
        assert_eq!(deep.column_bytes, rep.column_bytes);
        assert!(rep.frames > 1);
        assert_eq!(root.bytes, zf.len() as u64);
        check_sums(&root);
        for (name, bytes) in &rep.column_bytes {
            assert_eq!(root.find(name).unwrap().bytes, *bytes, "{name}");
        }
        for c in &root.children {
            let extra = ["$model", "$index", "$passthrough", "$overhead"];
            assert!(
                rep.column_bytes.iter().any(|x| x.0 == c.name) || extra.contains(&c.name.as_str())
            );
        }

        let msg = root.find("msg").unwrap();
        for part in [
            "msg.$row_modes",
            "msg.$structure",
            "msg.user",
            "msg.detail",
            "msg.$fallback",
        ] {
            assert!(msg.find(part).is_some_and(|p| p.bytes > 0), "{part}");
        }
        let detail = msg.find("msg.detail").unwrap();
        assert!(detail.find("msg.detail.$structure").is_some());
        assert!(!root.find("id").unwrap().sigs.is_empty());
        let json = root.json();
        assert!(json.contains("\"name\":\"msg.detail.$row_modes\""));
        assert!(
            root.tree(root.bytes, rep.total_records)
                .contains("msg.$row_modes")
        );
    }
}
