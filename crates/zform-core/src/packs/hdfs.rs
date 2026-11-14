//! Writer-side HDFS relationships. Readers execute only ordinary payload operations.
use super::*;

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
        if size {
            let key = ctx.plan.columns[..cid]
                .iter()
                .position(|c| c.dict_group == "blk_id")?;
            return block_attribute(ctx, key, packs);
        }
        None
    }

    fn row_order(&self, recs: &[&[u8]]) -> Option<Vec<u32>> {
        // A block's write pipeline rows sit together by thread id, so its relations look one row
        // back; the other rows follow in file order.
        const WRITE: [&[u8]; 4] = [
            b"Receiving block",
            b"PacketResponder",
            b"addStoredBlock: blockMap",
            b"allocateBlock",
        ];
        let mut groups = std::collections::HashMap::new();
        let mut members: Vec<Vec<u32>> = Vec::new();
        let mut rest = Vec::new();
        for (i, r) in recs.iter().enumerate() {
            let blk = WRITE
                .iter()
                .any(|w| memchr::memmem::find(r, w).is_some())
                .then(|| block_token(r))
                .flatten();
            match blk {
                Some(b) => {
                    let g = *groups.entry(b).or_insert_with(|| {
                        members.push(Vec::new());
                        members.len() - 1
                    });
                    members[g].push(i as u32);
                }
                None => rest.push(i as u32),
            }
        }
        if members.is_empty() {
            return None;
        }
        let mut order = Vec::with_capacity(recs.len());
        for mut m in members {
            m.sort_by_key(|&i| thread_id(recs[i as usize]));
            order.extend(m);
        }
        order.extend(rest);
        Some(order)
    }
}

/// The first `blk_<id>` token of a record.
fn block_token(r: &[u8]) -> Option<&[u8]> {
    let s = memchr::memmem::find(r, b"blk_")?;
    let tail = &r[s + 4..];
    let n = usize::from(tail.first() == Some(&b'-'));
    let d = tail[n..].iter().take_while(|b| b.is_ascii_digit()).count();
    (d > 0).then(|| &r[s..s + 4 + n + d])
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
