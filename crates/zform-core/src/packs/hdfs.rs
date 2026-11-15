//! Writer-side HDFS relationships. Readers execute only ordinary payload operations.
use super::*;
use memchr::memmem::Finder;
use std::sync::LazyLock;

// Prebuilt searchers: these run once or more per record.
static BLK: LazyLock<Finder<'static>> = LazyLock::new(|| Finder::new(b"blk_"));
static ADDED_AT: LazyLock<Finder<'static>> = LazyLock::new(|| Finder::new(ADDED));
/// What a write pipeline's lines say, each with the offset of an `R` or `B` in it.
const WRITES: [(&[u8], usize); 4] = [
    (b"Receiving block", 0),
    (b"PacketResponder", 6),
    (b"addStoredBlock: blockMap", 9),
    (b"allocateBlock", 8),
];

/// Predicts output partitions from task numbers and repeated attributes from block IDs.
pub struct HdfsRelations;

impl RelationPack for HdfsRelations {
    fn name(&self) -> &'static str {
        "relation/hdfs"
    }

    fn try_encode(
        &self,
        ctx: &RelationContext<'_, '_>,
        packs: &PackRegistry,
    ) -> Option<PayloadGraph> {
        let cid = ctx.column;
        let mut task = None;
        let mut size = false;
        for variant in &ctx.plan.variants {
            let PlanVariant::Shape(shape) = variant else {
                continue;
            };
            if !shape.slots.contains(&(cid as u16)) {
                continue;
            }
            if !shape
                .literals
                .iter()
                .any(|l| memchr::memmem::find(l, b"dfs.").is_some())
            {
                return None;
            }
            for (i, &col) in shape.slots.iter().enumerate() {
                if col as usize != cid {
                    continue;
                }
                let before = &shape.literals[i];
                size |= before.ends_with(b"size ");
                if before.ends_with(b"part") || before.ends_with(b"part-") {
                    for (j, &src) in shape.slots[..i].iter().enumerate() {
                        let lit = &shape.literals[j];
                        if (src as usize) < cid && (lit.ends_with(b"_m_") || lit.ends_with(b"_r_"))
                        {
                            if task.is_some_and(|old| old != src as usize) {
                                return None;
                            }
                            task = Some(src as usize);
                        }
                    }
                }
            }
        }
        if let Some(src) = task {
            return task_partition(ctx, src);
        }
        if field(ctx.plan, cid) == Some(1) {
            return clocks(ctx);
        }
        let key = ctx.plan.columns[..cid]
            .iter()
            .position(|c| c.dict_group == "blk_id")?;
        if size {
            return block_attribute(ctx, key, packs);
        }
        if ctx.plan.columns[cid].dict_group == "ip" {
            return replica(ctx, key);
        }
        None
    }

    fn table_builder(&self) -> Option<Box<dyn TableBuilder>> {
        Some(Box::<Replicas>::default())
    }

    fn row_order(&self, recs: &[&[u8]]) -> Option<Vec<u32>> {
        // Rows sit together by block: first each block's write pipeline by thread id, so its
        // relations look a row or two back, then each block's other rows in file order. Block
        // ids are successive `java.util.Random` outputs, so blocks in allocation order make each
        // new id a short step from the one before. Rows naming no block follow in file order.
        let mut block_of: FastMap<&[u8], u32> = FastMap::default();
        // By block: its generator state, and its group among the writes and among the others.
        let (mut states, mut slots) = (Vec::new(), Vec::new());
        // Groups in order of first appearance: block, whether its rows are not writes, rows.
        let mut groups: Vec<(u32, bool, u32)> = Vec::new();
        // By row: its group, `u32::MAX` when it names no block, and a write's thread.
        let mut of_row = Vec::with_capacity(recs.len());
        for r in recs {
            let Some(blk) = block_token(r) else {
                of_row.push((u32::MAX, 0));
                continue;
            };
            let other = !is_write(r);
            let b = *block_of.entry(blk).or_insert_with(|| {
                states.push(token_state(blk));
                slots.push([u32::MAX; 2]);
                states.len() as u32 - 1
            });
            let g = &mut slots[b as usize][usize::from(other)];
            if *g == u32::MAX {
                *g = groups.len() as u32;
                groups.push((b, other, 0));
            }
            groups[*g as usize].2 += 1;
            of_row.push((*g, if other { 0 } else { thread_id(r) }));
        }
        if groups.is_empty() {
            return None;
        }
        // Groups go by the step from the first block's generator state to theirs; ids no
        // generator made go last.
        let anchor = (groups.iter().filter(|g| !g.1))
            .chain(groups.iter().filter(|g| g.1))
            .find_map(|g| states[g.0 as usize]);
        let steps: Vec<_> = states
            .iter()
            .map(|&s| match (anchor, s) {
                (Some(a), Some(s)) => (false, JavaLcgPack::distance(a, s)),
                _ => (true, 0),
            })
            .collect();
        let mut ranked: Vec<u32> = (0..groups.len() as u32).collect();
        ranked.sort_by_key(|&g| (groups[g as usize].1, steps[groups[g as usize].0 as usize]));
        // Each group's place in the order, then rows placed in file order.
        let mut end = vec![0u32; groups.len()];
        let mut at = 0;
        for &g in &ranked {
            end[g as usize] = at;
            at += groups[g as usize].2;
        }
        let mut order = vec![0u32; recs.len()];
        let mut rest = at as usize;
        for (i, &(g, _)) in of_row.iter().enumerate() {
            let to = match end.get_mut(g as usize) {
                Some(e) => {
                    *e += 1;
                    *e as usize - 1
                }
                None => {
                    rest += 1;
                    rest - 1
                }
            };
            order[to] = i as u32;
        }
        for (g, &(_, other, n)) in groups.iter().enumerate() {
            if !other {
                let e = end[g] as usize;
                order[e - n as usize..e].sort_by_key(|&i| of_row[i as usize].1);
            }
        }
        Some(order)
    }
}

/// The table of each block's replicas, from the NameNode's `addStoredBlock` lines: a row per
/// block id, in allocation order, and its first [`REPLICAS`] datanodes at a fixed stride, in the
/// order [`HdfsRelations::row_order`] stores them, so a block's rows find them in sequence.
#[derive(Default)]
struct Replicas {
    rows: Vec<(Option<u64>, u64, Vec<u8>, Vec<u8>)>,
}

const ADDED: &[u8] = b"addStoredBlock: blockMap updated: ";
/// HDFS's default replication: datanodes past it are patched where a block names them.
const REPLICAS: usize = 3;

impl TableBuilder for Replicas {
    fn push(&mut self, raw: &[u8], spans: &[(u32, u32)]) {
        // One search over the chunk: few records match.
        let mut next = 0;
        for at in ADDED_AT.find_iter(raw) {
            let i = next + spans[next..].partition_point(|&(_, e)| e as usize <= at);
            let Some(&(s, e)) = spans.get(i) else {
                break;
            };
            next = i + 1;
            let (s, e) = (s as usize, e as usize);
            if at < s || at + ADDED.len() > e {
                continue;
            }
            let rec = &raw[s..e];
            let ip = &raw[at + ADDED.len()..e];
            // The model holds the port in a template literal.
            let ip = &ip[..memchr::memchr2(b':', b' ', ip).unwrap_or(ip.len())];
            if let Some(blk) = block_token(rec) {
                let row = (lcg_state(rec), thread_id(rec), blk[4..].to_vec(), ip.to_vec());
                self.rows.push(row);
            }
        }
    }

    fn finish(self: Box<Self>) -> Vec<Vec<Vec<u8>>> {
        let mut rows = self.rows;
        let anchor = rows.iter().find_map(|r| r.0);
        rows.sort_by_cached_key(|(s, thread, blk, _)| match (anchor, s) {
            (Some(a), Some(s)) => (false, JavaLcgPack::distance(a, *s), Vec::new(), *thread),
            _ => (true, 0, blk.clone(), *thread),
        });
        let (mut keys, mut ips): (Vec<Vec<u8>>, Vec<Vec<u8>>) = (Vec::new(), Vec::new());
        for (_, _, blk, ip) in rows {
            if keys.last() != Some(&blk) {
                keys.push(blk);
                ips.resize(keys.len() * REPLICAS, Vec::new());
            }
            let at = (keys.len() - 1) * REPLICAS;
            if let Some(slot) = ips[at..].iter_mut().find(|v| v.is_empty()) {
                *slot = ip;
            }
        }
        vec![keys, ips]
    }
}

/// A datanode column looked up among its block's replicas in the file's table: each value is
/// coded as which replica it is, and values the table lacks are patched.
fn replica(ctx: &RelationContext<'_, '_>, key: usize) -> Option<PayloadGraph> {
    const KEY: u32 = TABLE_PEER;
    const IP: u32 = TABLE_PEER | 1;
    let tables = ctx.tables.filter(|t| t.len() == 2)?;
    let table_ips = tables.col(1).ok()?;
    let vals = &ctx.columns[ctx.column];
    let aligned = align_cells(ctx.plan, ctx.variant_ids, key, ctx.column).ok()?;
    if aligned.len() != vals.len() {
        return None;
    }
    let ids = tables.joint_ids(0, ctx.columns[key].iter().copied()).ok()?;
    let (mut codes, mut exc_idx, mut exc_vals) = (Vec::with_capacity(vals.len()), vec![], vec![]);
    for (i, (&at, &v)) in aligned.iter().zip(vals).enumerate() {
        let row = ids[at as usize] as usize * REPLICAS;
        match (row..row + REPLICAS).position(|r| table_ips.get(r) == Some(v)) {
            Some(c) => codes.push(c as i64 + 1),
            None => {
                codes.push(0);
                exc_idx.push(i);
                exc_vals.push(v);
            }
        }
    }
    if exc_idx.len() * 4 > vals.len() {
        return None;
    }
    let mut gb = GraphBuilder::new(vals.len());
    let source = gb.add_ext_peer(IP);
    let ids = gb.add_ext_align(key as u32, Some(KEY));
    let codes = gb.add_ints(&codes);
    // Table row `REPLICAS * id + code - 1`; code 0, a patched value, reads row 0.
    let b = Builder::new(2);
    let (id, code) = (b.input(0), b.input(1));
    let row = b.rr(Op::Add, b.ri(Op::MulI, id, REPLICAS as i64), b.ri(Op::AddI, code, -1));
    let row = b.select(code, row, b.konst(0));
    let idx = gb.add_prog_i64(b.finish(&[(row, 8)]), vec![ids, codes], vec![]);
    let base = gb.add_index_cells(source, idx);
    let root = gb.add_patch(base, &exc_idx, &exc_vals);
    Some(gb.finish(root))
}

/// The space-separated field column `cid` fills in every shape holding it, once each.
fn field(plan: &ModelPlan, cid: usize) -> Option<usize> {
    let mut at = None;
    for v in &plan.variants {
        let PlanVariant::Shape(s) = v else { continue };
        let mut slots = s.slots.iter().enumerate().filter(|&(_, &c)| c as usize == cid);
        let Some((i, _)) = slots.next() else { continue };
        let (before, after) = (&s.literals[i], &s.literals[i + 1]);
        let spaced = (before.is_empty() || before.ends_with(b" "))
            && (after.is_empty() || after.starts_with(b" "));
        if slots.next().is_some() || !spaced {
            return None;
        }
        // Earlier slots are this format's leading numbers, which hold no spaces.
        let f = s.literals[..=i].iter().map(|l| memchr::memchr_iter(b' ', l).count()).sum();
        if at.is_some_and(|a| a != f) {
            return None;
        }
        at = Some(f);
    }
    at
}

const DAY: i64 = 86_400;

/// A `HHMMSS` cell as seconds into the day.
fn seconds(v: &[u8]) -> Option<i64> {
    let n = (v.len() == 6).then(|| unsigned(v)).flatten()?;
    let (h, m, s) = (n / 10000, n / 100 % 100, n % 100);
    (h < 24 && m < 60 && s < 60).then_some(h * 3600 + m * 60 + s)
}

/// Times of day of a block's writes, each a step from the last row of its clock: one for rows
/// starting the write, one for rows finishing it. Other rows' times are coded as they are,
/// where recency finds them. Each template's values are a stream of their own. The reader
/// finds a row's clock and stream from its template.
fn clocks(ctx: &RelationContext<'_, '_>) -> Option<PayloadGraph> {
    let vals = &ctx.columns[ctx.column];
    let n_var = ctx.plan.variants.len();
    // By variant: its clock (`0`: none) and its stream, numbering the variants holding the
    // column; by stream, whether it is coded by recency.
    let (mut clock, mut stream, mut recency) = (vec![0u8; n_var], vec![0u8; n_var], vec![]);
    for (v, var) in ctx.plan.variants.iter().enumerate() {
        let PlanVariant::Shape(s) = var else { continue };
        if !s.slots.contains(&(ctx.column as u16)) {
            continue;
        }
        let text = s.literals.concat();
        let has = |w: &[u8]| memchr::memmem::find(&text, w).is_some();
        clock[v] = match WRITES.iter().position(|&(w, _)| has(w)) {
            // `Receiving block`, `allocateBlock`
            Some(0 | 3) => 1,
            Some(_) => 2,
            None => 0,
        };
        stream[v] = recency.len() as u8;
        recency.push(clock[v] == 0);
    }
    if n_var > 120 || recency.len() > 64 {
        return None;
    }
    let vids = crate::codecs::cell_variants(ctx.plan, ctx.variant_ids, ctx.column).ok()?;
    if vids.len() != vals.len() {
        return None;
    }
    // Clocks start at the block's first time, and steps wrap at midnight.
    let base = vals.iter().find_map(|v| seconds(v))?;
    let mut state = [base; 3];
    let (mut deltas, mut exc_idx, mut exc_vals) = (Vec::with_capacity(vals.len()), vec![], vec![]);
    for (i, (&v, &vid)) in vals.iter().zip(&vids).enumerate() {
        let k = clock[vid as usize] as usize;
        match seconds(v) {
            Some(t) => {
                deltas.push((t - state[k] + DAY / 2).rem_euclid(DAY) - DAY / 2);
                if k > 0 {
                    state[k] = t;
                }
            }
            None => {
                deltas.push(0);
                exc_idx.push(i);
                exc_vals.push(v);
            }
        }
    }
    if exc_idx.len() * 8 > vals.len() {
        return None;
    }
    let mut gb = GraphBuilder::new(vals.len());
    // Each variant's clock, then its stream.
    let mut luts = clock.clone();
    luts.extend(&stream);
    let lut = gb.add_raw_bytes(&luts);
    let vid = gb.push(GraphNode::ExtVariant);
    let keys = gb.add_prog_i64(lut_prog(0), vec![vid], vec![lut]);
    let sel = gb.add_prog_i64(lut_prog(n_var as i64), vec![vid], vec![lut]);
    let sels: Vec<usize> = vids.iter().map(|&v| stream[v as usize] as usize).collect();
    let deltas = interleave(&mut gb, sel, &sels, &deltas, &recency);
    let t = gb.push(GraphNode::KeyedSum { keys, deltas });
    let b = Builder::new(1);
    // Sums stay within a few days of the base.
    let t_in = b.ri(Op::AddI, b.input(0), base + 4 * DAY);
    let t_in = b.rr(Op::Add, t_in, b.ri(Op::MulI, b.div(t_in, DAY as u64, 20), -DAY));
    let q = b.div(t_in, 60, 17);
    let h = b.div(q, 60, 11);
    let sec = b.rr(Op::Add, t_in, b.ri(Op::MulI, q, -60));
    let min = b.rr(Op::Add, q, b.ri(Op::MulI, h, -60));
    let hm = b.rr(Op::Add, b.ri(Op::MulI, h, 10000), b.ri(Op::MulI, min, 100));
    let hms = b.rr(Op::Add, hm, sec);
    let hms = gb.add_prog_i64(b.finish(&[(hms, 8)]), vec![t], vec![]);
    let base = gb.add_prog_cells(prog_fmt_zero_padded(6, None), vec![hms], vec![]);
    let root = gb.add_patch(base, &exc_idx, &exc_vals);
    Some(gb.finish(root))
}

/// The byte at `offset + x` of the program's buffer.
fn lut_prog(offset: i64) -> Program {
    let b = Builder::new(1);
    let buf = b.add_buf();
    let v = b.gather(buf, 0, b.input(0), offset as u64);
    b.finish(&[(v, 8)])
}

/// `vals` split by `sel` into a stream each, which `sel_node` interleaves again; a stream
/// `recency` marks may be coded by recency. Streams go by selector, so a block and its sample
/// number theirs alike.
fn interleave(
    gb: &mut GraphBuilder,
    sel_node: u16,
    sel: &[usize],
    vals: &[i64],
    recency: &[bool],
) -> u16 {
    let mut streams = vec![Vec::new(); recency.len()];
    for (&s, &v) in sel.iter().zip(vals) {
        streams[s].push(v);
    }
    if streams.iter().filter(|s| !s.is_empty()).count() <= 1 {
        return gb.add_ints(vals);
    }
    // Empty streams still take a slot, so selectors index the list directly.
    let empty = gb.add_ints(&[]);
    let nodes = (streams.iter().zip(recency))
        .map(|(s, &r)| match (s.is_empty(), r) {
            (true, _) => empty,
            (false, true) => gb.add_mtf_ints(s),
            (false, false) => gb.add_ints(s),
        })
        .collect();
    gb.push(GraphNode::Interleave {
        sel: sel_node,
        streams: nodes,
    })
}

/// Whether a record says one of [`WRITES`], found from the letters they hold.
fn is_write(r: &[u8]) -> bool {
    memchr::memchr2_iter(b'R', b'B', r)
        .any(|p| WRITES.iter().any(|&(w, at)| p >= at && r[p - at..].starts_with(w)))
}

/// The first `blk_<id>` token of a record.
fn block_token(r: &[u8]) -> Option<&[u8]> {
    let s = BLK.find(r)?;
    let tail = &r[s + 4..];
    let n = usize::from(tail.first() == Some(&b'-'));
    let d = tail[n..].iter().take_while(|b| b.is_ascii_digit()).count();
    (d > 0).then(|| &r[s..s + 4 + n + d])
}

/// The `java.util.Random` state behind a record's first block id.
fn lcg_state(r: &[u8]) -> Option<u64> {
    token_state(block_token(r)?)
}

fn token_state(blk: &[u8]) -> Option<u64> {
    let t = &blk[4..];
    let (neg, d) = match t.strip_prefix(b"-") {
        Some(d) => (true, d),
        None => (false, t),
    };
    let v = std::str::from_utf8(d).ok()?.parse::<u64>().ok()?;
    let v = if neg { 0i64.checked_sub_unsigned(v)? } else { i64::try_from(v).ok()? };
    JavaLcgPack::state_of(v)
}

/// The third space-separated field of a record as a number: the logging thread in HDFS lines.
fn thread_id(r: &[u8]) -> u64 {
    r.split(|&b| b == b' ')
        .nth(2)
        .and_then(unsigned)
        .map_or(0, |v| v as u64)
}

fn unsigned(v: &[u8]) -> Option<i64> {
    if v.is_empty() || v.len() > 16 {
        return None;
    }
    v.iter().try_fold(0i64, |n, &c| {
        c.is_ascii_digit().then(|| n * 10 + i64::from(c - b'0'))
    })
}

fn task_partition(ctx: &RelationContext<'_, '_>, src: usize) -> Option<PayloadGraph> {
    let vals = &ctx.columns[ctx.column];
    let aligned = align_cells(ctx.plan, ctx.variant_ids, src, ctx.column).ok()?;
    let source = &ctx.columns[src];
    if aligned.len() != vals.len() || source.iter().any(|v| unsigned(v).is_none()) {
        return None;
    }
    // The template tokenizer can include the filename's hyphen in the numeric cell.
    let hyphen = vals[0].starts_with(b"-");
    let width = vals[0].len() - usize::from(hyphen);
    if !(1..=16).contains(&width) {
        return None;
    }
    let mut exc_idx = Vec::new();
    let mut exc_vals = Vec::new();
    for (i, (&at, &v)) in aligned.iter().zip(vals).enumerate() {
        let n = unsigned(source[at as usize])?;
        let digits = if hyphen {
            v.strip_prefix(b"-")
        } else {
            Some(v)
        };
        if digits.is_none_or(|d| d.len() != width || unsigned(d) != Some(n)) {
            exc_idx.push(i);
            exc_vals.push(v);
        }
    }
    if exc_idx.len() * 8 > vals.len() {
        return None;
    }
    let mut gb = GraphBuilder::new(vals.len());
    let peer = gb.add_ext_peer(src as u32);
    let identity =
        aligned.len() == source.len() && aligned.iter().enumerate().all(|(i, &a)| a == i as i64);
    let source = if identity {
        peer
    } else {
        let idx = gb.add_ints(&aligned);
        gb.add_index_cells(peer, idx)
    };
    let numbers = gb.add_parse_ascii_int(source);
    let base = gb.add_prog_cells(prog_fmt_zero_padded(width, None), vec![numbers], vec![]);
    let base = if hyphen {
        let template = gb.add_const_cells(b"-\x1f\0", vals.len());
        gb.push(GraphNode::TemplateCells {
            template,
            slots: vec![base],
        })
    } else {
        base
    };
    let root = gb.add_patch(base, &exc_idx, &exc_vals);
    Some(gb.finish(root))
}

fn block_attribute(
    ctx: &RelationContext<'_, '_>,
    key: usize,
    packs: &PackRegistry,
) -> Option<PayloadGraph> {
    // ExtAlign uses the first key cell of a record; multiple block IDs would make the
    // self-aligned haystack differ from the key column's occurrence order.
    if ctx.plan.variants.iter().any(|v| matches!(v, PlanVariant::Shape(s) if s.slots.iter().filter(|&&c| c as usize == key).count() > 1)) {
        return None;
    }
    let vals = &ctx.columns[ctx.column];
    let keys = &ctx.columns[key];
    let aligned = align_cells(ctx.plan, ctx.variant_ids, key, ctx.column).ok()?;
    if aligned.len() != vals.len() {
        return None;
    }
    let mut counts: FastMap<&[u8], usize> = FastMap::default();
    let mut fill = vals[0];
    let mut most = 0;
    for &v in vals {
        let count = counts.entry(v).or_default();
        *count += 1;
        if *count > most {
            fill = v;
            most = *count;
        }
    }
    let mut first = FastMap::default();
    for (i, &k) in keys.iter().enumerate() {
        first.entry(k).or_insert(i);
    }
    let mut table = vec![fill; keys.len()];
    let mut assigned = FastSet::default();
    let (mut exc_idx, mut exc_vals) = (Vec::new(), Vec::new());
    for (i, (&at, &v)) in aligned.iter().zip(vals).enumerate() {
        let pos = first[keys[at as usize]];
        if assigned.insert(pos) {
            table[pos] = v;
        } else if table[pos] != v {
            exc_idx.push(i);
            exc_vals.push(v);
        }
    }
    if assigned.len() * 2 > vals.len() || exc_idx.len() * 8 > vals.len() {
        return None;
    }
    let mut gb = GraphBuilder::new(vals.len());
    let source = gb.merge_graph(&packs.encode_best_graph_inner(&table, 0).0);
    let needles = gb.add_ext_align(key as u32, None);
    let hay = gb.add_ext_align(key as u32, Some(key as u32));
    let codes = gb.push(GraphNode::IntStream(IntStream::Const {
        val: 1,
        len: vals.len() as u32,
    }));
    let idx = gb.add_match_index(needles, hay, codes);
    let base = gb.add_index_cells(source, idx);
    let root = gb.add_patch(base, &exc_idx, &exc_vals);
    Some(gb.finish(root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codecs::{BlockDecodeCtx, ensure_dense_block_column};
    use crate::grammar::{FrontEnd, Shape};
    use crate::wire::DecodeBudget;

    fn shape(plan: &mut ModelPlan, literals: &[&[u8]], slots: &[u16]) {
        plan.push_variant(
            PlanVariant::Shape(Shape {
                literals: literals.iter().map(|v| v.to_vec()).collect(),
                slots: slots.to_vec(),
            }),
            FrontEnd::None,
        );
    }

    #[test]
    fn writes_are_found_by_their_letters() {
        let lines: [&[u8]; 6] = [
            b"INFO dfs.DataNode$PacketResponder: Received block blk_1",
            b"BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.0.0.1:50010",
            b"BLOCK* NameSystem.allocateBlock: /x. blk_2",
            b"Receiving block blk_3",
            b"BLOCK* NameSystem.addStoredBlock: Redundant",
            b"Deleting block blk_4 file /BlockReceiving block",
        ];
        for r in lines {
            let any = WRITES.iter().any(|(w, _)| memchr::memmem::find(r, w).is_some());
            assert_eq!(is_write(r), any, "{}", String::from_utf8_lossy(r));
        }
        for (w, at) in WRITES {
            assert!(matches!(w[at], b'R' | b'B'));
        }
    }

    fn check_graph(plan: &ModelPlan, vids: &[u32], cols: &[Vec<&[u8]>], graph: PayloadGraph) {
        for mixing in [false, true] {
            let packs = PackRegistry::default();
            let mut blobs: Vec<_> = cols.iter().map(|col| packs.encode_best(col)).collect();
            blobs[2] = if mixing {
                graph.clone().with_context_mixing()
            } else {
                graph.clone()
            }
            .encode();
            let budget = DecodeBudget::default();
            for selective in [false, true] {
                let mut ctx = BlockDecodeCtx::new(
                    blobs.iter().map(|v| Some(&v[..])).collect(),
                    vec![],
                    None,
                    Arc::default(),
                )
                .with_layout(plan, vids.to_vec().into());
                if selective {
                    ctx.read_only(vec![(vec![0, 0, 2], vec![0, 0, 3])]);
                }
                let got = ensure_dense_block_column(2, &mut ctx, &budget)
                    .unwrap()
                    .to_cells()
                    .unwrap();
                if selective {
                    assert_eq!(got.get(2).unwrap(), cols[2][2]);
                } else {
                    assert!(got.iter().eq(cols[2].iter().copied()));
                }
                assert!(
                    ctx.cache[0].is_none(),
                    "unrelated column must stay unloaded"
                );
                assert!(ctx.cache[1].is_some());
            }
        }
    }

    /// Column `cid` decoded from `graph`, the others coded on their own.
    fn decode_with(
        plan: &ModelPlan,
        vids: &[u32],
        cols: &[Vec<&[u8]>],
        cid: usize,
        graph: &PayloadGraph,
    ) -> Vec<Vec<u8>> {
        let packs = PackRegistry::default();
        let mut blobs: Vec<_> = cols.iter().map(|col| packs.encode_best(col)).collect();
        blobs[cid] = graph.clone().with_context_mixing().encode();
        let blobs = blobs.iter().map(|v| Some(&v[..])).collect();
        let mut ctx = BlockDecodeCtx::new(blobs, vec![], None, Arc::default())
            .with_layout(plan, vids.to_vec().into());
        let col = ensure_dense_block_column(cid, &mut ctx, &DecodeBudget::default()).unwrap();
        col.to_cells().unwrap().iter().map(<[u8]>::to_vec).collect()
    }

    #[test]
    fn write_times_run_on_start_and_finish_clocks() {
        let mut plan = ModelPlan::residual_only();
        plan.add_or_get_col("date");
        plan.add_or_get_col("time");
        let lines: [&[u8]; 4] = [
            b" 19 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /x. blk_1",
            b" 300 INFO dfs.DataNode$DataXceiver: Receiving block blk_1",
            b" 301 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_1 terminating",
            b" 13 INFO dfs.FSDataset: Deleting block blk_1",
        ];
        for l in lines {
            shape(&mut plan, &[b"", b" ", l], &[1, 2]);
        }
        let (mut date, mut time, mut vids) = (Vec::new(), Vec::new(), Vec::new());
        let mut t = 86_390;
        for i in 0..600u32 {
            let v = i % 4;
            t = match v {
                0 => t + i64::from(i % 3),
                2 => t + 30,
                _ => t,
            } % DAY;
            date.push(b"081109".to_vec());
            time.push(match i {
                7 => b"2461xx".to_vec(),
                _ => format!("{:02}{:02}{:02}", t / 3600, t / 60 % 60, t % 60).into_bytes(),
            });
            vids.push(1 + v);
        }
        let cols = vec![
            vec![],
            date.iter().map(|v| &v[..]).collect(),
            time.iter().map(|v| &v[..]).collect(),
        ];
        assert_eq!(field(&plan, 2), Some(1));
        assert_eq!(field(&plan, 1), Some(0));
        let ctx = RelationContext {
            tables: None,
            column: 2,
            columns: &cols,
            plan: &plan,
            variant_ids: &vids,
        };
        let graph = HdfsRelations.try_encode(&ctx, &PackRegistry::default()).unwrap();
        assert!(graph.nodes.iter().any(|n| matches!(n, GraphNode::KeyedSum { .. })));
        assert_eq!(decode_with(&plan, &vids, &cols, 2, &graph), time);
    }

    #[test]
    fn task_partition_preserves_padding_exceptions_and_sparse_alignment() {
        let mut plan = ModelPlan::residual_only();
        plan.add_or_get_col("task");
        plan.add_or_get_col("partition");
        shape(
            &mut plan,
            &[b"dfs.FSNamesystem: _task_job_m_", b"_0/part", b""],
            &[1, 2],
        );
        shape(&mut plan, &[b"dfs.FSNamesystem: _task_job_m_", b""], &[1]);
        let (mut source, mut target, mut vids) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..256 {
            let n = if i == 2 { 100_000 } else { (i * 37) % 997 };
            source.push(format!("{n:06}").into_bytes());
            target.push(match i {
                17 => b"-7".to_vec(),
                98 => b"\xffexception".to_vec(),
                _ => format!("-{n:05}").into_bytes(),
            });
            vids.push(1);
            if i % 7 == 0 {
                source.push(b"000001".to_vec());
                vids.push(2);
            }
        }
        let cols = vec![
            vec![],
            source.iter().map(|v| &v[..]).collect(),
            target.iter().map(|v| &v[..]).collect(),
        ];
        let ctx = RelationContext {
            tables: None,
            column: 2,
            columns: &cols,
            plan: &plan,
            variant_ids: &vids,
        };
        let graph = HdfsRelations
            .try_encode(&ctx, &PackRegistry::default())
            .unwrap();
        assert_eq!(graph.ext_peers, [1]);
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| matches!(n, GraphNode::PatchCells { .. }))
        );
        check_graph(&plan, &vids, &cols, graph);
        let PlanVariant::Shape(s) = &mut plan.variants[1] else {
            unreachable!()
        };
        s.literals[0] = b"unrelated task_m_".to_vec();
        let ctx = RelationContext {
            tables: None,
            column: 2,
            columns: &cols,
            plan: &plan,
            variant_ids: &vids,
        };
        assert!(
            HdfsRelations
                .try_encode(&ctx, &PackRegistry::default())
                .is_none()
        );
    }

    #[test]
    fn relation_registry_prices_candidates_and_enforces_dependency_rules() {
        struct Copy(bool);
        impl RelationPack for Copy {
            fn name(&self) -> &'static str {
                "test/copy"
            }
            fn try_encode(
                &self,
                ctx: &RelationContext<'_, '_>,
                _: &PackRegistry,
            ) -> Option<PayloadGraph> {
                if ctx.column != 2 {
                    return None;
                }
                let mut gb = GraphBuilder::new(ctx.columns[2].len());
                let root = gb.add_ext_peer(if self.0 { 2 } else { 1 });
                Some(gb.finish(root))
            }
        }
        let mut plan = ModelPlan::residual_only();
        plan.add_or_get_col("a");
        plan.add_or_get_col("b");
        shape(&mut plan, &[b"", b" ", b""], &[1, 2]);
        let values: Vec<_> = (0..128u64)
            .map(|i| format!("value-{:016x}", i.wrapping_mul(0x9e3779b97f4a7c15)))
            .collect();
        let col: Vec<_> = values.iter().map(|v| v.as_bytes()).collect();
        let cols = vec![vec![], col.clone(), col];
        let encode = |pack: PackRegistry, standalone| {
            pack.encode_dense_block(
                &cols,
                &["", "", ""],
                &vec![vec![]; 3],
                &mut vec![None; 3],
                &[false, false, standalone],
                Some((&plan, &[1; 128])),
                None,
                EntropyBackend::Zstd,
                3,
            )
        };
        let base = encode(PackRegistry::default(), false);
        let mut invalid = PackRegistry::default();
        invalid.register_relation(Copy(true));
        assert_eq!(encode(invalid, false).col_blobs, base.col_blobs);
        let mut valid = PackRegistry::default();
        valid.register_relation(Copy(false));
        let selected = encode(valid.clone(), false);
        assert!(selected.col_blobs[2].len() + 32 < base.col_blobs[2].len());
        assert_eq!(
            PayloadGraph::decode(&selected.col_blobs[2], &DecodeBudget::default())
                .unwrap()
                .ext_peers,
            [1]
        );
        assert_eq!(
            encode(valid, true).col_blobs,
            encode(PackRegistry::default(), true).col_blobs
        );
    }

    #[test]
    fn block_attributes_preserve_changes_and_keys_without_attributes() {
        let mut plan = ModelPlan::residual_only();
        plan.add_or_get_col("block");
        plan.columns[1].dict_group = "blk_id".into();
        plan.add_or_get_col("size");
        shape(
            &mut plan,
            &[b"dfs.DataNode: Received block blk_", b" of size ", b""],
            &[1, 2],
        );
        shape(
            &mut plan,
            &[b"dfs.DataNode: Receiving block blk_", b""],
            &[1],
        );
        let (mut keys, mut values, mut vids) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..160 {
            keys.push(format!("{i}"));
            vids.push(2);
        }
        for i in 0..512 {
            let key = (i * 37) % 128;
            keys.push(format!("{key}"));
            values.push(if i % 73 == 0 {
                "changed".to_string()
            } else {
                format!("{}", 10000 + key * 341)
            });
            vids.push(1);
        }
        let cols = vec![
            vec![],
            keys.iter().map(|v| v.as_bytes()).collect(),
            values.iter().map(|v| v.as_bytes()).collect(),
        ];
        let ctx = RelationContext {
            tables: None,
            column: 2,
            columns: &cols,
            plan: &plan,
            variant_ids: &vids,
        };
        let graph = HdfsRelations
            .try_encode(&ctx, &PackRegistry::default())
            .unwrap();
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| matches!(n, GraphNode::MatchIndex { .. }))
        );
        check_graph(&plan, &vids, &cols, graph);
        let PlanVariant::Shape(s) = &mut plan.variants[2] else {
            unreachable!()
        };
        s.slots.push(1);
        s.literals.push(vec![]);
        let ctx = RelationContext {
            tables: None,
            column: 2,
            columns: &cols,
            plan: &plan,
            variant_ids: &vids,
        };
        assert!(
            HdfsRelations
                .try_encode(&ctx, &PackRegistry::default())
                .is_none()
        );
    }
}
