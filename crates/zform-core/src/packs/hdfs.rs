//! Writer-side HDFS relationships. Readers execute only ordinary payload operations.
use super::*;
use crate::payload::graph::{SumSeed, parse_cell_ascii_i64};
use memchr::memmem::Finder;
use std::mem::take;
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
        let key = ctx.plan.columns[..cid]
            .iter()
            .position(|c| c.dict_group == "blk_id");
        if field(ctx.plan, cid) == Some(1) {
            return clocks(ctx, key);
        }
        let key = key?;
        if size {
            return block_attribute(ctx, key, packs);
        }
        if ctx.plan.columns[cid].dict_group == "ip" {
            return replica(ctx, key);
        }
        if field(ctx.plan, cid) == Some(2) {
            return threads(ctx, key);
        }
        None
    }

    fn table_builder(&self) -> Option<Box<dyn TableBuilder>> {
        Some(Box::<Replicas>::default())
    }

    fn row_order(&self, recs: &[&[u8]]) -> Option<Vec<u32>> {
        // Rows sit together by block: first each block's write pipeline by thread id, or in
        // [`in_pipeline_order`], then each block's other rows in file order. Block
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
                let rows = &mut order[e - n as usize..e];
                rows.sort_by_key(|&i| of_row[i as usize].1);
                in_pipeline_order(recs, rows);
            }
        }
        Some(order)
    }
}

/// A block's write rows `rows`, by thread, reordered where its pipeline is known: the
/// datanodes' rows by template, then by their node's place in the pipeline, so each template's
/// replica codes repeat block after block. The NameNode's rows stay first, by thread.
fn in_pipeline_order(recs: &[&[u8]], rows: &mut [u32]) {
    let mut reps = Vec::with_capacity(REPLICAS);
    let said: Vec<_> = (rows.iter())
        .map(|&i| {
            let r = recs[i as usize];
            let (kind, p) = write_at(r)?;
            let role = match kind {
                0 => said_at(r, true, p),
                1 => said_at(r, false, p),
                _ => {
                    if reps.len() < REPLICAS {
                        reps.extend(added_node(r, p));
                    }
                    None
                }
            };
            with_thread(r, role?)
        })
        .collect();
    // As the table finds it, from `PacketResponder` rows.
    let named: Vec<_> = (said.iter().flatten())
        .filter(|s| s.0 != Role::Receiving)
        .copied()
        .collect();
    let Some(chain) = pipeline(&reps, &named) else {
        return;
    };
    let at = |v: &[u8]| chain.iter().position(|&c| reps[c] == v);
    let mut keyed: Vec<_> = (rows.iter().zip(&said))
        .map(|(&i, s)| match *s {
            None => ((0, 0), i),
            Some((role, _, Said::Node(v))) => ((role as u8, at(v).map_or(0, |p| p as i64 + 1)), i),
            Some((role, _, Said::Responder(n))) => ((role as u8, REPLICAS as i64 - 1 - n), i),
        })
        .collect();
    // Stable: rows alike stay by thread.
    keyed.sort_by_key(|k| k.0);
    for (r, k) in rows.iter_mut().zip(keyed) {
        *r = k.1;
    }
}

/// The datanode an `addStoredBlock` record adds, `at` the offset of [`ADDED`] in it.
fn added_node(r: &[u8], at: usize) -> Option<&[u8]> {
    let ip = r[at..].strip_prefix(ADDED)?;
    Some(&ip[..memchr::memchr2(b':', b' ', ip).unwrap_or(ip.len())])
}

/// The table of each block's replicas, from the NameNode's `addStoredBlock` lines: a row per
/// block id, in allocation order, and its first [`REPLICAS`] datanodes at a fixed stride, in
/// pipeline order where the block's [`pipeline`] is clear and else as `addStoredBlock` lists them.
#[derive(Default)]
struct Replicas {
    rows: Vec<(Option<u64>, u64, Vec<u8>, Vec<u8>)>,
    /// Datanodes' write rows: block id and nodes by [`hash`].
    writes: Vec<(u64, Role, u64, Said<u64>)>,
}

const ADDED: &[u8] = b"addStoredBlock: blockMap updated: ";
/// HDFS's default replication: datanodes past it are patched where a block names them.
const REPLICAS: usize = 3;

/// Each record of `raw` holding a match `hits` finds, `w` bytes long, with its offset in it.
fn records_with<'a>(
    hits: impl Iterator<Item = usize>,
    raw: &'a [u8],
    spans: &[(u32, u32)],
    w: usize,
    mut each: impl FnMut(&'a [u8], usize),
) {
    // One search over the chunk; hits lie a few records apart.
    let mut next = 0;
    for at in hits {
        let i = next
            + spans[next..]
                .iter()
                .take_while(|&&(_, e)| e as usize <= at)
                .count();
        let Some(&(s, e)) = spans.get(i) else {
            break;
        };
        next = i + 1;
        let (s, e) = (s as usize, e as usize);
        if at >= s && at + w <= e {
            each(&raw[s..e], at - s);
        }
    }
}

/// Names compared by hash: a collision only misplaces a pipeline.
fn hash(v: &[u8]) -> u64 {
    use core::hash::Hasher;
    let mut h = crate::payload::graph::FastHasher::default();
    h.write(v);
    h.finish()
}

impl TableBuilder for Replicas {
    fn push(&mut self, raw: &[u8], spans: &[(u32, u32)]) {
        records_with(
            ADDED_AT.find_iter(raw),
            raw,
            spans,
            ADDED.len(),
            |rec, at| {
                let ip = &rec[at + ADDED.len()..];
                // The model holds the port in a template literal.
                let ip = &ip[..memchr::memchr2(b':', b' ', ip).unwrap_or(ip.len())];
                if let Some(blk) = block_token(rec) {
                    let row = (
                        lcg_state(rec),
                        thread_id(rec),
                        blk[4..].to_vec(),
                        ip.to_vec(),
                    );
                    self.rows.push(row);
                }
            },
        );
        // `Received` rows name every upstream `Receiving` rows do. `$` is rarer than the
        // needle's letters.
        let hits = memchr::memchr_iter(b'$', raw)
            .filter(|&p| raw[p + 1..].starts_with(RESPONDER))
            .map(|p| p + 1);
        records_with(hits, raw, spans, RESPONDER.len(), |rec, at| {
            let said = said_at(rec, false, at).and_then(|s| with_thread(rec, s));
            let (Some(blk), Some((role, t, s))) = (block_token(rec), said) else {
                return;
            };
            let s = match s {
                Said::Node(v) => Said::Node(hash(v)),
                Said::Responder(n) => Said::Responder(n),
            };
            self.writes.push((hash(&blk[4..]), role, t, s));
        });
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
        // Each block's replicas in pipeline order where its writes tell it.
        let mut writes = self.writes;
        writes.sort_unstable_by_key(|w| w.0);
        let mut rows = Vec::new();
        // Blocks in hash order meet their writes in one pass over them.
        let mut order: Vec<(u64, usize)> = keys
            .iter()
            .enumerate()
            .map(|(k, blk)| (hash(blk), k))
            .collect();
        order.sort_unstable();
        let mut s = 0;
        for (b, k) in order {
            s += writes[s..].iter().take_while(|w| w.0 < b).count();
            let e = s + writes[s..].iter().take_while(|w| w.0 == b).count();
            if s == e {
                continue;
            }
            rows.clear();
            rows.extend(writes[s..e].iter().map(|w| (w.1, w.2, w.3)));
            let reps = &mut ips[k * REPLICAS..(k + 1) * REPLICAS];
            // An empty replica is no node.
            let ids: [u64; REPLICAS] = std::array::from_fn(|i| {
                if reps[i].is_empty() {
                    i as u64
                } else {
                    hash(&reps[i])
                }
            });
            if let Some(c) = pipeline(&ids, &rows) {
                let mut old: [Vec<u8>; REPLICAS] = std::array::from_fn(|i| take(&mut reps[i]));
                for (r, &i) in reps.iter_mut().zip(&c) {
                    *r = take(&mut old[i]);
                }
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
    let row = b.rr(
        Op::Add,
        b.ri(Op::MulI, id, REPLICAS as i64),
        b.ri(Op::AddI, code, -1),
    );
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
        let mut slots = s
            .slots
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c as usize == cid);
        let Some((i, _)) = slots.next() else { continue };
        let (before, after) = (&s.literals[i], &s.literals[i + 1]);
        let spaced = (before.is_empty() || before.ends_with(b" "))
            && (after.is_empty() || after.starts_with(b" "));
        if slots.next().is_some() || !spaced {
            return None;
        }
        // Earlier slots are this format's leading numbers, which hold no spaces.
        let f = s.literals[..=i]
            .iter()
            .map(|l| memchr::memchr_iter(b' ', l).count())
            .sum();
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
/// starting the write, one for rows finishing it. Deletions are steps on their datanodes'
/// clocks, as [`deletions`] finds them. Other rows' times are coded as they are, where recency
/// finds them. Each template's values are a stream of their own. The reader finds a row's
/// clock and stream from its template.
fn clocks(ctx: &RelationContext<'_, '_>, key: Option<usize>) -> Option<PayloadGraph> {
    let vals = &ctx.columns[ctx.column];
    let n_var = ctx.plan.variants.len();
    // By variant: its clock (`0`: none), its stream, numbering the variants holding the
    // column, and whether it deletes; by stream, whether it is coded by recency.
    let (mut clock, mut stream, mut recency) = (vec![0u8; n_var], vec![0u8; n_var], vec![]);
    let mut deletes = vec![0u8; n_var];
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
            // A block is read soon after its neighbours.
            None if has(b"Served block") || has(b"exception while serving") => 3,
            None => 0,
        };
        deletes[v] = u8::from(has(DELETING));
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
    // Sums count from the block's first time, and steps wrap at midnight.
    let base = vals.iter().find_map(|v| seconds(v))?;
    let (mut state, mut last) = ([None; 4], base);
    let (mut deltas, mut exc_idx, mut exc_vals) = (Vec::with_capacity(vals.len()), vec![], vec![]);
    for (i, (&v, &vid)) in vals.iter().zip(&vids).enumerate() {
        let k = clock[vid as usize] as usize;
        match seconds(v) {
            Some(t) => {
                let from = if k > 0 {
                    state[k].unwrap_or(last)
                } else {
                    base
                };
                deltas.push((t - from + DAY / 2).rem_euclid(DAY) - DAY / 2);
                if k > 0 {
                    // The sum, not `t`: a wrapped step leaves it a day off.
                    last = from + deltas[i];
                    state[k] = Some(last);
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
    let marked: Vec<bool> = (vids.iter().zip(vals))
        .map(|(&v, &t)| deletes[v as usize] > 0 && seconds(t).is_some())
        .collect();
    let slots = key.and_then(|key| deletions(ctx, key, &marked, &mut deltas));
    if slots.is_some() {
        // Steps, not times.
        (0..n_var)
            .filter(|&v| deletes[v] > 0)
            .for_each(|v| recency[stream[v] as usize] = false);
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
    let mut t = gb.push(GraphNode::KeyedSum {
        keys,
        deltas,
        order: None,
        seed: SumSeed::Previous,
    });
    if let (Some(slots), Some(key)) = (slots, key) {
        t = deletion_sum(&mut gb, t, key, &slots, sel, &sels, recency.len());
    }
    let b = Builder::new(1);
    // Sums stay within a few days of the base.
    let t_in = b.ri(Op::AddI, b.input(0), base + 4 * DAY);
    let t_in = b.rr(
        Op::Add,
        t_in,
        b.ri(Op::MulI, b.div(t_in, DAY as u64, 20), -DAY),
    );
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

const DELETING: &[u8] = b"Deleting block";

/// A datanode works through the blocks it is told to delete in order of their ids, so one
/// node's deletions run on a clock of their own once the rows are taken in that order. Rows
/// do not name the node: infer one of the block's replicas from its clock and recent use.
/// Selected rows become steps on that replica's independent clock, starting at zero.
/// Return explicit replica slots, or `-1` for rows left unchanged.
fn deletions(
    ctx: &RelationContext<'_, '_>,
    key: usize,
    marked: &[bool],
    deltas: &mut [i64],
) -> Option<Vec<i64>> {
    let tables = ctx.tables.filter(|t| t.len() == 2)?;
    let (blocks, ips) = (tables.col(0).ok()?, tables.col(1).ok()?);
    let aligned = align_cells(ctx.plan, ctx.variant_ids, key, ctx.column).ok()?;
    if aligned.len() != marked.len() {
        return None;
    }
    let ids = tables.joint_ids(0, ctx.columns[key].iter().copied()).ok()?;
    let row_of = |i: usize| ids[aligned[i] as usize] as usize;
    let mut rows: Vec<(i64, usize)> = (0..marked.len())
        .filter(|&i| marked[i])
        .filter_map(|i| Some((parse_cell_ascii_i64(blocks.get(row_of(i))?), i)))
        .collect();
    rows.sort_unstable();
    let mut state = DeletionClocks::default();
    let (mut slots, mut steps) = (vec![-1i64; marked.len()], deltas.to_vec());
    let mut emitted = FastSet::default();
    let (mut left, mut all) = (Vec::new(), Vec::new());
    let mut j = 0;
    while j < rows.len() {
        let end = j + rows[j..].iter().take_while(|r| r.0 == rows[j].0).count();
        let group = &rows[j..end];
        j = end;
        let from = row_of(group[0].1) * REPLICAS;
        state.open(
            (from..from + REPLICAS).filter_map(|c| ips.get(c)),
            &mut left,
        );
        // One table row per order, with a replica for each of its deletions.
        if group.len() > left.len() || group.iter().any(|r| row_of(r.1) * REPLICAS != from) {
            continue;
        }
        all.clone_from(&left);
        let cost = |c: &&[usize]| -> u32 {
            (c.iter().zip(group))
                .map(|(&p, r)| bits(deltas[r.1] - state.sum(all[p])))
                .sum()
        };
        let chains = CHAINS.iter().map(|c| &c[..group.len()]);
        let chain = chains
            .filter(|c| c.iter().all(|&p| p < all.len()))
            .min_by_key(cost)?;
        for (&p, &(_, i)) in chain.iter().zip(group) {
            let at = left.iter().position(|&k| k == all[p])?;
            let k = left.remove(at);
            slots[i] = (from..from + REPLICAS)
                .position(|c| ips.get(c).and_then(|v| state.ids.get(v).copied()) == Some(k))?
                as i64;
            let step = deltas[i] - state.sum(k);
            // Inference may compare clocks across nodes; decoding keeps each key independent.
            steps[i] = if emitted.insert(k) { deltas[i] } else { step };
            state.step(k, step);
        }
        state.close(&all, &left);
    }
    // A clock pays once its node has rows to step between.
    let n = slots.iter().filter(|&&p| p >= 0).count();
    if n < 4 * state.keys().max(4) {
        return None;
    }
    deltas.copy_from_slice(&steps);
    Some(slots)
}

#[derive(Default)]
struct DeletionClocks<'a> {
    ids: FastMap<&'a [u8], u32>,
    keys: Vec<(Option<i64>, bool)>,
    last: i64,
}

impl<'a> DeletionClocks<'a> {
    /// A group's candidates, in pick order, as `left`.
    fn open(&mut self, cells: impl Iterator<Item = &'a [u8]>, left: &mut Vec<u32>) {
        left.clear();
        for v in cells.filter(|v| !v.is_empty()) {
            let next = self.keys.len() as u32;
            let k = *self.ids.entry(v).or_insert(next);
            if k == next {
                self.keys.push((None, false));
            }
            if !left.contains(&k) {
                left.push(k);
            }
        }
        // Stable: candidates alike stay in cell order.
        left.sort_by_key(|&k| !self.keys[k as usize].1);
    }

    fn keys(&self) -> usize {
        self.keys.len()
    }

    fn sum(&self, k: u32) -> i64 {
        self.keys[k as usize].0.unwrap_or(self.last)
    }

    fn step(&mut self, k: u32, delta: i64) -> i64 {
        self.last = self.sum(k).wrapping_add(delta);
        self.keys[k as usize].0 = Some(self.last);
        self.last
    }

    /// Ends a group of candidates `all`, `left` of them untaken.
    fn close(&mut self, all: &[u32], left: &[u32]) {
        for &k in all {
            self.keys[k as usize].1 = !left.contains(&k);
        }
    }
}

/// Explicit replica choices feed ordinary table lookups and an ordered sum.
fn deletion_sum(
    gb: &mut GraphBuilder,
    deltas: u16,
    key: usize,
    slots: &[i64],
    sel: u16,
    sels: &[usize],
    streams: usize,
) -> u16 {
    let at = gb.add_ext_align(key as u32, Some(TABLE_PEER));
    let slots = interleave(gb, sel, sels, slots, &vec![false; streams]);
    let b = Builder::new(2);
    let present = b.ri(Op::AddI, b.input(1), 1);
    let index = b.rr(
        Op::Add,
        b.ri(Op::MulI, b.input(0), REPLICAS as i64),
        b.input(1),
    );
    let index = b.select(present, index, b.konst(0));
    let index = gb.add_prog_i64(b.finish(&[(index, 8)]), vec![at, slots], vec![]);
    let ips = gb.add_ext_peer(TABLE_PEER | 1);
    let ips = gb.add_index_cells(ips, index);
    let ids = gb.push(GraphNode::CellIds { cells: ips });
    let b = Builder::new(2);
    let present = b.ri(Op::AddI, b.input(1), 1);
    let key = b.select(present, b.ri(Op::AddI, b.input(0), 1), b.konst(0));
    let keys = gb.add_prog_i64(b.finish(&[(key, 8)]), vec![ids, slots], vec![]);
    // Unselected rows may name absent table entries; their ordering value is immaterial.
    let b = Builder::new(2);
    let safe = b.select(b.input(1), b.input(0), b.konst(0));
    let at = gb.add_prog_i64(b.finish(&[(safe, 8)]), vec![at, keys], vec![]);
    let blocks = gb.add_ext_peer(TABLE_PEER);
    let blocks = gb.add_index_cells(blocks, at);
    let order = gb.add_parse_ascii_int(blocks);
    gb.push(GraphNode::KeyedSum {
        keys,
        deltas,
        order: Some(order),
        seed: SumSeed::Zero,
    })
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

/// What a datanode's row of a block's write says of its place in the pipeline.
#[derive(Clone, Copy, PartialEq)]
enum Role {
    /// A row of no write, or of the NameNode.
    None,
    /// `Receiving block`, naming its upstream.
    Receiving,
    /// `PacketResponder N`: the node `N` places before the pipeline's end.
    Responder,
    /// `Received block`, naming its upstream.
    Received,
}

/// By variant: the role of its rows, and the slot holding the upstream or responder number.
fn roles(plan: &ModelPlan) -> Vec<(Role, usize)> {
    (plan.variants.iter())
        .map(|v| {
            let PlanVariant::Shape(s) = v else {
                return (Role::None, 0);
            };
            let text = s.literals.concat();
            let has = |w: &[u8]| memchr::memmem::find(&text, w).is_some();
            let (role, before): (_, &[u8]) = if has(b"DataXceiver: Receiving block") {
                (Role::Receiving, b"src: /")
            } else if has(b"PacketResponder: Received block") {
                (Role::Received, b"from /")
            } else if has(b"terminating") {
                (Role::Responder, b"PacketResponder ")
            } else {
                return (Role::None, 0);
            };
            match (s.literals.iter()).position(|l| l.ends_with(before)) {
                Some(slot) if slot < s.slots.len() => (role, slot),
                _ => (Role::None, 0),
            }
        })
        .collect()
}

/// About the bits a step costs.
fn bits(step: i64) -> u32 {
    2 * (64 - step.unsigned_abs().leading_zeros()) + 1
}

/// The orders of up to [`REPLICAS`] replicas; fewer take the orders of their first indices.
const CHAINS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// What a write row says of its node: nothing, its place in the pipeline, or its upstream
/// (by replica index; `None` for the client).
#[derive(Clone, Copy)]
enum Hint {
    Any,
    At(usize),
    After(Option<usize>),
}

/// Each write row's replica, by index into a block's replicas in pipeline order `chain`, as
/// its hint allows and its node's last thread in `last` suits best (`prev`, the last keyed
/// thread, for a node yet to log one), and the bits spent.
fn place(
    chain: &[usize],
    rows: &[(Hint, i64)],
    last: &mut [Option<i64>],
    mut prev: i64,
) -> (u32, Vec<usize>) {
    let n = chain.len();
    let (mut cost, mut picks) = (0, Vec::with_capacity(rows.len()));
    for &(hint, thread) in rows {
        let at = |r: usize| chain.iter().position(|&c| c == r);
        let cands: &[usize] = match hint {
            Hint::At(p) if p < n => &chain[p..=p],
            // The client may run on the first replica itself.
            Hint::After(Some(r)) => match at(r) {
                Some(0) if n > 1 => &chain[..2],
                Some(p) if p + 1 < n => &chain[p + 1..p + 2],
                _ => &chain[..1],
            },
            Hint::After(None) => &chain[..1],
            _ => chain,
        };
        let step = |c: usize| thread - last[c].unwrap_or(prev);
        let &c = cands.iter().min_by_key(|&&c| bits(step(c))).unwrap();
        cost += bits(step(c));
        (last[c], prev) = (Some(thread), thread);
        picks.push(c);
    }
    (cost, picks)
}

/// Thread ids, which a datanode hands out from a counter: each row of a block's write is coded
/// as a step from the last thread its datanode logged. Which replica logged it follows from
/// the pipeline (client to first replica, and on), tried in each order the replicas could
/// take, keeping the order whose steps are smallest; the reader is told the replica and finds
/// the datanode in the replica table. Other rows' threads are coded as they are.
fn threads(ctx: &RelationContext<'_, '_>, key: usize) -> Option<PayloadGraph> {
    const KEY: u32 = TABLE_PEER;
    const IP: u32 = TABLE_PEER | 1;
    let tables = ctx.tables.filter(|t| t.len() == 2)?;
    let table_ips = tables.col(1).ok()?;
    let n_rows = table_ips.len() / REPLICAS;
    let vals = &ctx.columns[ctx.column];
    let aligned = align_cells(ctx.plan, ctx.variant_ids, key, ctx.column).ok()?;
    if aligned.len() != vals.len() {
        return None;
    }
    let ids = tables.joint_ids(0, ctx.columns[key].iter().copied()).ok()?;
    let roles = roles(ctx.plan);
    let n_var = roles.len();
    // By variant holding the column: its streams of codes, of rows without a replica and of
    // rows with one, and whether a stream is coded by recency.
    let (mut lut, mut recency) = (vec![0u8; 3 * n_var], Vec::new());
    for (v, var) in ctx.plan.variants.iter().enumerate() {
        if matches!(var, PlanVariant::Shape(s) if s.slots.contains(&(ctx.column as u16))) {
            lut[v] = recency.len() as u8;
            lut[n_var + v] = recency.len() as u8;
            recency.push(true);
            if roles[v].0 != Role::None {
                lut[n_var + v] = recency.len() as u8;
                recency.push(false);
            }
        }
    }
    if n_var > 80 || recency.len() > 64 {
        return None;
    }
    // By thread cell: its variant and what its role reads in the same record.
    let mut cells = Vec::with_capacity(vals.len());
    let mut cursor = vec![0usize; ctx.columns.len()];
    for &v in ctx.variant_ids {
        let PlanVariant::Shape(s) = &ctx.plan.variants[v as usize] else {
            return None;
        };
        let (role, slot) = roles[v as usize];
        let (mut held, mut has) = (None, false);
        for (j, &c) in s.slots.iter().enumerate() {
            let c = c as usize;
            if role != Role::None && j == slot {
                held = ctx.columns[c].get(cursor[c]).copied();
            }
            has |= c == ctx.column;
            cursor[c] += 1;
        }
        if has {
            cells.push((v as usize, held));
        }
    }
    let (mut exc_idx, mut exc_vals) = (Vec::new(), Vec::new());
    let mut thread = Vec::with_capacity(vals.len());
    for (i, &v) in vals.iter().enumerate() {
        match unsigned(v).filter(|n| v == n.to_string().as_bytes()) {
            Some(n) => thread.push(n),
            None => {
                thread.push(0);
                exc_idx.push(i);
                exc_vals.push(v);
            }
        }
    }
    if cells.len() != vals.len() || exc_idx.len() * 8 > vals.len() {
        return None;
    }
    // By datanode, the last thread it logged; and the last of any.
    let (mut last, mut prev): (FastMap<&[u8], i64>, i64) = (FastMap::default(), 0);
    let (mut codes, mut steps) = (vec![0i64; vals.len()], thread.clone());
    let row_of = |j: usize| ids[aligned[j] as usize] as usize;
    let mut i = 0;
    while i < vals.len() {
        // A block's write rows lie together, as `row_order` stores them.
        let row = row_of(i);
        let write = |j: usize| {
            roles[cells[j].0].0 != Role::None && row_of(j) == row && !exc_idx.contains(&j)
        };
        if row >= n_rows || !write(i) {
            i += 1;
            continue;
        }
        let end = (i..vals.len()).find(|&j| !write(j)).unwrap_or(vals.len());
        let reps: Vec<&[u8]> = (0..REPLICAS)
            .filter_map(|r| table_ips.get(row * REPLICAS + r))
            .filter(|ip| !ip.is_empty())
            .collect();
        let n = reps.len();
        if n == 0 {
            i = end;
            continue;
        }
        let hint = |j: usize| match (roles[cells[j].0].0, cells[j].1) {
            (Role::Responder, Some(h)) => unsigned(h)
                .and_then(|k| (n as i64 - 1 - k).try_into().ok())
                .map_or(Hint::Any, Hint::At),
            (_, Some(ip)) => Hint::After(reps.iter().position(|&r| r == ip)),
            _ => Hint::Any,
        };
        let rows: Vec<_> = (i..end).map(|j| (hint(j), thread[j])).collect();
        let start: Vec<Option<i64>> = reps.iter().map(|r| last.get(r).copied()).collect();
        let chains = CHAINS
            .iter()
            .map(|c| &c[..n])
            .filter(|c| c.iter().all(|&p| p < n));
        let (_, picks) = chains
            .map(|c| place(c, &rows, &mut start.clone(), prev))
            .min_by_key(|p| p.0)
            .unwrap();
        for (j, c) in (i..end).zip(picks) {
            steps[j] = thread[j] - last.get(reps[c]).copied().unwrap_or(prev);
            last.insert(reps[c], thread[j]);
            prev = thread[j];
            codes[j] = c as i64 + 1;
        }
        i = end;
    }
    let mut gb = GraphBuilder::new(vals.len());
    let lut_node = gb.add_raw_bytes(&lut);
    let vid = gb.push(GraphNode::ExtVariant);
    let vids: Vec<usize> = cells.iter().map(|c| c.0).collect();
    let code_sel = gb.add_prog_i64(lut_prog(0), vec![vid], vec![lut_node]);
    let code_sels: Vec<usize> = vids.iter().map(|&v| lut[v] as usize).collect();
    let codes_node = interleave(
        &mut gb,
        code_sel,
        &code_sels,
        &codes,
        &vec![false; recency.len()],
    );
    // A row with a replica reads the second table.
    let b = Builder::new(2);
    let (v, code) = (b.input(0), b.input(1));
    let keyed = b.ri(Op::ShrI, b.ri(Op::AddI, code, 3), 2);
    let at = b.rr(Op::Add, v, b.ri(Op::MulI, keyed, n_var as i64));
    let sel = b.gather(b.add_buf(), 0, at, 0);
    let sel = gb.add_prog_i64(b.finish(&[(sel, 8)]), vec![vid, codes_node], vec![lut_node]);
    let sels: Vec<usize> = (vids.iter().zip(&codes))
        .map(|(&v, &c)| lut[v + n_var * usize::from(c > 0)] as usize)
        .collect();
    let steps = interleave(&mut gb, sel, &sels, &steps, &recency);
    // Table row `REPLICAS * id + code - 1`; code 0 reads row 0 and keys no node.
    let source = gb.add_ext_peer(IP);
    let ids = gb.add_ext_align(key as u32, Some(KEY));
    let b = Builder::new(2);
    let (id, code) = (b.input(0), b.input(1));
    let row = b.rr(
        Op::Add,
        b.ri(Op::MulI, id, REPLICAS as i64),
        b.ri(Op::AddI, code, -1),
    );
    let row = b.select(code, row, b.konst(0));
    let idx = gb.add_prog_i64(b.finish(&[(row, 8)]), vec![ids, codes_node], vec![]);
    let nodes = gb.add_index_cells(source, idx);
    let nodes = gb.push(GraphNode::CellIds { cells: nodes });
    let b = Builder::new(2);
    let (node, code) = (b.input(0), b.input(1));
    let k = b.select(code, b.ri(Op::AddI, node, 1), b.konst(0));
    let keys = gb.add_prog_i64(b.finish(&[(k, 8)]), vec![nodes, codes_node], vec![]);
    let t = gb.push(GraphNode::KeyedSum {
        keys,
        deltas: steps,
        order: None,
        seed: SumSeed::Previous,
    });
    let base = gb.add_prog_cells(prog_fmt_int(false, 0), vec![t], vec![]);
    let root = gb.add_patch(base, &exc_idx, &exc_vals);
    Some(gb.finish(root))
}

const RESPONDER: &[u8] = b"PacketResponder: ";

/// What a datanode's write row names: an upstream node (`Receiving`, `Received`) or its
/// responder number (`Responder`). `at` is the offset of `Receiving block` or, unless
/// `receiving`, of `PacketResponder`.
fn said_at(r: &[u8], receiving: bool, at: usize) -> Option<(Role, &[u8])> {
    // The token from `s` to a `:` or space.
    let token = |s: usize| {
        let n = r[s..]
            .iter()
            .take_while(|&&b| b != b':' && b != b' ')
            .count();
        (n > 0).then(|| &r[s..s + n])
    };
    // The token after the first `/` past `at`: block ids hold none.
    let node = |at: usize, w: &[u8]| {
        let s = at + memchr::memchr(b'/', &r[at..])?;
        r[..s].ends_with(w).then(|| token(s + 1)).flatten()
    };
    if receiving {
        let dn = r[..at].ends_with(b"DataXceiver: ");
        return Some((Role::Receiving, dn.then(|| node(at, b"src: ")).flatten()?));
    }
    let at = at + 17;
    let t = r.get(at - 2..)?.strip_prefix(b": ")?;
    if t.starts_with(b"Received block ") {
        Some((Role::Received, node(at, b" from ")?))
    } else if t.starts_with(b"PacketResponder ") && r.ends_with(b"terminating") {
        Some((Role::Responder, token(at + 16)?))
    } else {
        None
    }
}

/// A block's replicas `reps` in pipeline order, as indices, from its write rows (role, thread,
/// what the row names): the last replica is no row's upstream, and `PacketResponder N`
/// shares its thread with its node's `Received` row naming the node before. `None` unless one
/// order fits best.
fn pipeline<T: PartialEq + Copy>(reps: &[T], rows: &[(Role, u64, Said<T>)]) -> Option<[usize; 3]> {
    if reps.len() != REPLICAS || reps[0] == reps[1] || reps[1] == reps[2] || reps[0] == reps[2] {
        return None;
    }
    let received = |t: u64, up: T| {
        (rows.iter()).any(|&(r, u, s)| r == Role::Received && u == t && s == Said::Node(up))
    };
    let score = |c: &[usize; 3]| {
        let mut s = 0i32;
        for &(_, t, said) in rows {
            match said {
                Said::Node(v) => s -= i32::from(v == reps[c[2]]),
                // Responders 1 and 0 run on the second and third nodes.
                Said::Responder(n @ 0..=1) => s += i32::from(received(t, reps[c[1 - n as usize]])),
                Said::Responder(_) => {}
            }
        }
        s
    };
    let (mut best, mut tied) = ((i32::MIN, 0), false);
    for (k, c) in CHAINS.iter().enumerate() {
        let s = score(c);
        if s == best.0 {
            tied = true;
        } else if s > best.0 {
            (best, tied) = ((s, k), false);
        }
    }
    (!tied).then_some(CHAINS[best.1])
}

/// What a write row names: a node, or its responder number.
#[derive(Clone, Copy, PartialEq)]
enum Said<T> {
    Node(T),
    Responder(i64),
}

/// A record's [`said_at`] with its thread.
fn with_thread<'a>(r: &[u8], (role, v): (Role, &'a [u8])) -> Option<(Role, u64, Said<&'a [u8]>)> {
    let s = match role {
        Role::Responder => Said::Responder(unsigned(v)?),
        _ => Said::Node(v),
    };
    Some((role, thread_id(r), s))
}

/// Whether a record says one of [`WRITES`], found from the letters they hold.
fn is_write(r: &[u8]) -> bool {
    write_at(r).is_some()
}

/// The first of [`WRITES`] a record says, and its offset.
fn write_at(r: &[u8]) -> Option<(usize, usize)> {
    memchr::memchr2_iter(b'R', b'B', r).find_map(|p| {
        let k = WRITES
            .iter()
            .position(|&(w, at)| p >= at && r[p - at..].starts_with(w))?;
        Some((k, p - WRITES[k].1))
    })
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
    let v = if neg {
        0i64.checked_sub_unsigned(v)?
    } else {
        i64::try_from(v).ok()?
    };
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
            let any = WRITES
                .iter()
                .any(|(w, _)| memchr::memmem::find(r, w).is_some());
            assert_eq!(is_write(r), any, "{}", String::from_utf8_lossy(r));
        }
        for (w, at) in WRITES {
            assert!(matches!(w[at], b'R' | b'B'));
        }
    }

    #[test]
    fn write_rows_go_in_pipeline_order() {
        let recs: [&[u8]; 13] = [
            b"081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
            b"081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /x/job.jar. blk_-16",
            b"081109 203519 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /10.250.10.6:40524 dest: /10.250.10.6:50010",
            b"081109 203519 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /10.250.14.224:42420 dest: /10.250.14.224:50010",
            b"081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-16 terminating",
            b"081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-16 terminating",
            b"081109 203519 145 INFO dfs.DataNode$PacketResponder: Received block blk_-16 of size 91178 from /10.250.10.6",
            b"081109 203519 145 INFO dfs.DataNode$PacketResponder: Received block blk_-16 of size 91178 from /10.250.19.102",
            b"081109 203519 147 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-16 terminating",
            b"081109 203519 147 INFO dfs.DataNode$PacketResponder: Received block blk_-16 of size 91178 from /10.250.14.224",
            b"081109 203519 29 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.10.6:50010 is added to blk_-16 size 91178",
            b"081109 203519 30 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.111.209:50010 is added to blk_-16 size 91178",
            b"081109 203519 31 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.14.224:50010 is added to blk_-16 size 91178",
        ];
        let mut b = Replicas::default();
        let raw = recs.join(&b"\n"[..]);
        let mut spans = Vec::new();
        for r in &recs {
            let s = spans.last().map_or(0, |&(_, e): &(u32, u32)| e + 1);
            spans.push((s, s + r.len() as u32));
        }
        b.push(&raw, &spans);
        let ips = Box::new(b).finish().remove(1);
        assert_eq!(
            ips,
            [&b"10.250.10.6"[..], b"10.250.14.224", b"10.251.111.209"]
        );
        let mut rows: Vec<u32> = (0..recs.len() as u32).collect();
        rows.sort_by_key(|&i| thread_id(recs[i as usize]));
        in_pipeline_order(&recs, &mut rows);
        assert_eq!(rows, [10, 11, 12, 1, 0, 2, 3, 5, 4, 8, 7, 6, 9]);
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
        let graph = HdfsRelations
            .try_encode(&ctx, &PackRegistry::default())
            .unwrap();
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| matches!(n, GraphNode::KeyedSum { .. }))
        );
        assert_eq!(decode_with(&plan, &vids, &cols, 2, &graph), time);
    }

    #[test]
    fn write_threads_step_from_their_datanodes_counters() {
        use crate::{
            cells::Cells,
            dictionary::{Dictionary, Scope},
            tables::Tables,
        };
        let mut plan = ModelPlan::residual_only();
        plan.add_or_get_col_with_group("blk", "blk_id");
        plan.add_or_get_col("thread");
        plan.add_or_get_col("ip");
        plan.add_or_get_col("n");
        let lines: [(&[&[u8]], &[u16]); 4] = [
            (
                &[b"0 0 ", b" INFO dfs.FSNamesystem: allocateBlock: blk_", b""],
                &[2, 1],
            ),
            (
                &[
                    b"0 0 ",
                    b" INFO dfs.DataNode$DataXceiver: Receiving block blk_",
                    b" src: /",
                    b"",
                ],
                &[2, 1, 3],
            ),
            (
                &[
                    b"0 0 ",
                    b" INFO dfs.DataNode$PacketResponder: PacketResponder ",
                    b" for block blk_",
                    b" terminating",
                ],
                &[2, 4, 1],
            ),
            (
                &[
                    b"0 0 ",
                    b" INFO dfs.DataNode$PacketResponder: Received block blk_",
                    b" from /",
                    b"",
                ],
                &[2, 1, 3],
            ),
        ];
        for (l, s) in lines {
            shape(&mut plan, l, s);
        }
        let node = |k: usize| format!("10.0.0.{k}").into_bytes();
        let mut counter = [1000i64, 1010, 1020, 1030, 1040];
        let (mut cols, mut vids) = (vec![Vec::<Vec<u8>>::new(); 5], Vec::new());
        let (mut keys, mut ips) = (Vec::new(), Vec::new());
        let mut row = |cols: &mut Vec<Vec<Vec<u8>>>, v: u32, vals: &[(usize, Vec<u8>)]| {
            vids.push(v);
            for (c, x) in vals {
                cols[*c].push(x.clone());
            }
        };
        for b in 0..40usize {
            let blk = format!("{}", 7000 + b).into_bytes();
            let pipe = [b % 5, (b + 1) % 5, (b + 3) % 5];
            keys.push(blk.clone());
            // The table holds the replicas out of pipeline order.
            ips.extend((0..3).map(|r| node(pipe[(r + b) % 3])));
            let nn = format!("{}", b % 10).into_bytes();
            row(&mut cols, 1, &[(2, nn), (1, blk.clone())]);
            let mut held = [0i64; 3];
            for p in 0..3 {
                counter[pipe[p]] += 1 + (b % 2) as i64;
                held[p] = counter[pipe[p]];
                let up = if p == 0 {
                    b"10.9.9.9".to_vec()
                } else {
                    node(pipe[p - 1])
                };
                let t = format!("{}", held[p]).into_bytes();
                row(&mut cols, 2, &[(2, t), (1, blk.clone()), (3, up)]);
            }
            for p in (0..3).rev() {
                counter[pipe[p]] += 1;
                let t = match b * 3 + p {
                    11 => b"0123".to_vec(),
                    _ => format!("{}", counter[pipe[p]]).into_bytes(),
                };
                row(
                    &mut cols,
                    3,
                    &[
                        (2, t),
                        (4, format!("{}", 2 - p).into_bytes()),
                        (1, blk.clone()),
                    ],
                );
                let up = if p == 0 {
                    b"10.9.9.9".to_vec()
                } else {
                    node(pipe[p - 1])
                };
                let t = format!("{}", counter[pipe[p]]).into_bytes();
                row(&mut cols, 4, &[(2, t), (1, blk.clone()), (3, up)]);
            }
        }
        let cells = |v: &[Vec<u8>]| {
            let lens: Vec<u32> = v.iter().map(|x| x.len() as u32).collect();
            Cells::from_lens(v.concat(), &lens).unwrap()
        };
        let table = || Tables::from_cells(vec![cells(&keys), cells(&ips)], Vec::new);
        let cols: Vec<Vec<&[u8]>> = cols
            .iter()
            .map(|c| c.iter().map(|v| &v[..]).collect())
            .collect();
        assert_eq!(field(&plan, 2), Some(2));
        let tables = table();
        let ctx = RelationContext {
            tables: Some(&tables),
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
                .any(|n| matches!(n, GraphNode::KeyedSum { .. }))
        );
        // Each pipeline's nodes are found: the client, then each row's upstream.
        let rows = [
            (Hint::After(None), 5),
            (Hint::After(Some(2)), 9),
            (Hint::At(0), 6),
        ];
        assert_eq!(
            place(&[2, 0, 1], &rows, &mut [Some(8), None, Some(4)], 0).1,
            [2, 0, 2]
        );
        let dict = Dictionary::new(b"content".to_vec()).with_tables(table());
        let _scope = Scope::enter(Some(Arc::new(dict)));
        let got = decode_with(&plan, &vids, &cols, 2, &graph);
        assert!(
            got.iter().eq(cols[2]
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>()
                .iter())
        );
    }

    #[test]
    fn deletions_step_on_their_datanodes_clocks() {
        use crate::{
            cells::Cells,
            dictionary::{Dictionary, Scope},
            tables::Tables,
        };
        let mut plan = ModelPlan::residual_only();
        plan.add_or_get_col_with_group("blk", "blk_id");
        plan.add_or_get_col("date");
        plan.add_or_get_col("time");
        shape(
            &mut plan,
            &[
                b"",
                b" ",
                b" 19 INFO dfs.FSDataset: Deleting block blk_",
                b"",
            ],
            &[2, 3, 1],
        );
        shape(
            &mut plan,
            &[
                b"",
                b" ",
                b" 9 INFO dfs.DataBlockScanner: Verification succeeded for blk_",
                b"",
            ],
            &[2, 3, 1],
        );
        // Each node deletes its blocks in id order, a second on every other one.
        let id = |b: usize| (b * 37 % 60) as i64 - 30;
        let reps = |b: usize| [b % 5, (b + 1) % 5, (b + 3) % 5];
        let mut by_id: Vec<usize> = (0..60).collect();
        by_id.sort_by_key(|&b| id(b));
        let (mut clock, mut seen) = ([1000i64, 1050, 1100, 1150, 1200], [0i64; 5]);
        let mut times = vec![Vec::new(); 60];
        for b in by_id {
            for (r, n) in reps(b).into_iter().enumerate() {
                // Some replicas are deleted in another block of the file.
                if b % 4 == 0 && r == 1 {
                    continue;
                }
                seen[n] += 1;
                clock[n] += seen[n] % 2;
                times[b].push(clock[n]);
            }
            times[b].sort_unstable();
        }
        let (mut cols, mut vids) = (vec![Vec::<Vec<u8>>::new(); 4], Vec::new());
        let (mut keys, mut ips) = (Vec::new(), Vec::new());
        let hms = |t: i64| format!("{:02}{:02}{:02}", t / 3600, t / 60 % 60, t % 60).into_bytes();
        for b in 0..60 {
            let blk = id(b).to_string().into_bytes();
            keys.push(blk.clone());
            ips.extend(reps(b).map(|n| format!("10.0.0.{n}").into_bytes()));
            let scan = (b % 5 == 0).then_some((2, 5000 + b as i64));
            for (v, t) in times[b].iter().map(|&t| (1, t)).chain(scan) {
                vids.push(v);
                cols[1].push(blk.clone());
                cols[2].push(b"081109".to_vec());
                cols[3].push(hms(t));
            }
        }
        for (variant, block, time) in [
            (2, b"999999".to_vec(), hms(6100)),
            (1, b"999998".to_vec(), hms(6200)),
            (1, keys[0].clone(), b"250000".to_vec()),
        ] {
            vids.push(variant);
            cols[1].push(block);
            cols[2].push(b"081109".to_vec());
            cols[3].push(time);
        }
        let cells = |v: &[Vec<u8>]| {
            let lens: Vec<u32> = v.iter().map(|x| x.len() as u32).collect();
            Cells::from_lens(v.concat(), &lens).unwrap()
        };
        let table = || Tables::from_cells(vec![cells(&keys), cells(&ips)], Vec::new);
        let cols: Vec<Vec<&[u8]>> = cols
            .iter()
            .map(|c| c.iter().map(|v| &v[..]).collect())
            .collect();
        assert_eq!(field(&plan, 3), Some(1));
        let tables = table();
        let ctx = RelationContext {
            tables: Some(&tables),
            column: 3,
            columns: &cols,
            plan: &plan,
            variant_ids: &vids,
        };
        let graph = HdfsRelations
            .try_encode(&ctx, &PackRegistry::default())
            .unwrap();
        assert!(graph.nodes.iter().any(|n| matches!(
            n,
            GraphNode::KeyedSum {
                order: Some(_),
                seed: SumSeed::Zero,
                ..
            }
        )));
        let dict = Dictionary::new(b"content".to_vec()).with_tables(table());
        let _scope = Scope::enter(Some(Arc::new(dict)));
        let got = decode_with(&plan, &vids, &cols, 3, &graph);
        assert!(
            got.iter().eq(cols[3]
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>()
                .iter())
        );
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
