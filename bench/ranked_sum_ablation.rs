// Included only in the benchmark's temporary copy of packs/hdfs.rs.
// Keep inference identical for a given record order; change what the reader reconstructs.

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeletionMode {
    Ranked,
    Off,
    KeyedRow,
    KeyedSorted,
    ScanGrouped,
    KeyedArgsort,
    ReplicaRow,
    ReplicaArgsort,
    ReplicaSplit,
    ReplicaOrdered,
    ReplicaOrderedSplit,
    ReplicaPhysical,
    ReplicaOrderedZero,
}

fn deletion_mode() -> DeletionMode {
    static MODE: std::sync::OnceLock<DeletionMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("ZFORM_RANKED_ABLATION").as_deref() {
        Ok("ranked") | Err(_) => DeletionMode::Ranked,
        Ok("off") => DeletionMode::Off,
        Ok("keyed-row") => DeletionMode::KeyedRow,
        Ok("keyed-sorted") => DeletionMode::KeyedSorted,
        Ok("scan-grouped") => DeletionMode::ScanGrouped,
        Ok("keyed-argsort") => DeletionMode::KeyedArgsort,
        Ok("replica-row") => DeletionMode::ReplicaRow,
        Ok("replica-argsort") => DeletionMode::ReplicaArgsort,
        Ok("replica-split") => DeletionMode::ReplicaSplit,
        Ok("replica-ordered") => DeletionMode::ReplicaOrdered,
        Ok("replica-ordered-split") => DeletionMode::ReplicaOrderedSplit,
        Ok("replica-physical") => DeletionMode::ReplicaPhysical,
        Ok("replica-ordered-zero") => DeletionMode::ReplicaOrderedZero,
        Ok(other) => panic!("unknown deletion ablation: {other}"),
    })
}

struct DeletionChoice {
    picks: Vec<i64>,
    keys: Vec<i64>,
    order: Vec<usize>,
    targets: Vec<i64>,
    slots: Vec<i64>,
}

fn prepare_deletions(choice: &DeletionChoice, deltas: &mut [i64], mode: DeletionMode) {
    match mode {
        DeletionMode::KeyedRow | DeletionMode::ReplicaRow | DeletionMode::ReplicaPhysical => {
            let mut state = FastMap::default();
            let mut last = 0i64;
            for (i, &key) in choice.keys.iter().enumerate().filter(|(_, k)| **k != 0) {
                let target = choice.targets[i];
                deltas[i] = target.wrapping_sub(*state.get(&key).unwrap_or(&last));
                state.insert(key, target);
                last = target;
            }
        }
        DeletionMode::KeyedSorted | DeletionMode::ScanGrouped => {
            // These rows are supplied by a separate stream, added to the ordinary clocks.
            for &i in &choice.order {
                deltas[i] = 0;
            }
        }
        DeletionMode::ReplicaOrderedZero => {
            let mut seen = FastSet::default();
            for &i in &choice.order {
                if seen.insert(choice.keys[i]) {
                    deltas[i] = choice.targets[i];
                }
            }
        }
        _ => {}
    }
}

fn simple_deletion_graph(
    gb: &mut GraphBuilder,
    base: u16,
    choice: &DeletionChoice,
    mode: DeletionMode,
) -> u16 {
    if mode == DeletionMode::KeyedRow {
        let keys = gb.add_ints(&choice.keys);
        return gb.push(GraphNode::KeyedSum { keys, deltas: base });
    }
    let mut order = choice.order.clone();
    if mode == DeletionMode::ScanGrouped {
        // Stable grouping retains block-id order within each inferred datanode.
        order.sort_by_key(|&i| choice.keys[i]);
    }
    let mut inverse = vec![-1i64; choice.keys.len()];
    let mut deltas = Vec::with_capacity(order.len());
    let mut state = FastMap::default();
    let mut last = 0i64;
    for (at, &i) in order.iter().enumerate() {
        inverse[i] = at as i64;
        let target = choice.targets[i];
        let from = if mode == DeletionMode::KeyedSorted {
            *state.get(&choice.keys[i]).unwrap_or(&last)
        } else {
            last
        };
        deltas.push(target.wrapping_sub(from));
        state.insert(choice.keys[i], target);
        last = target;
    }
    let deltas = gb.add_ints(&deltas);
    let sums = if mode == DeletionMode::KeyedSorted {
        let keys: Vec<_> = order.iter().map(|&i| choice.keys[i]).collect();
        let keys = gb.add_ints(&keys);
        gb.push(GraphNode::KeyedSum { keys, deltas })
    } else {
        let b = Builder::new(1);
        let sum = b.scan_add(b.input(0), 0);
        gb.add_prog_i64(b.finish(&[(sum, 8)]), vec![deltas], vec![])
    };
    let inverse = gb.add_ints(&inverse);
    let b = Builder::new(2);
    let buf = b.add_buf();
    // -1 addresses outside the buffer and yields zero for an untouched row.
    let offset = b.ri(Op::ShlI, b.input(1), 3);
    let restored = b.gather(buf, 3, offset, 0);
    let sum = b.rr(Op::Add, b.input(0), restored);
    gb.add_prog_i64(b.finish(&[(sum, 8)]), vec![base, inverse], vec![sums])
}

fn gather_ints(gb: &mut GraphBuilder, values: u16, indices: u16) -> u16 {
    let b = Builder::new(1);
    let buf = b.add_buf();
    let offset = b.ri(Op::ShlI, b.input(0), 3);
    let value = b.gather(buf, 3, offset, 0);
    gb.add_prog_i64(b.finish(&[(value, 8)]), vec![indices], vec![values])
}

fn replica_keys(gb: &mut GraphBuilder, choice: &DeletionChoice, key: usize, split: Option<(u16, &[usize], usize)>) -> u16 {
    let at = gb.add_ext_align(key as u32, Some(TABLE_PEER));
    let slots = match split {
        Some((sel, sels, count)) => interleave(gb, sel, sels, &choice.slots, &vec![false; count]),
        None => gb.add_ints(&choice.slots),
    };
    let b = Builder::new(2);
    let present = b.ri(Op::AddI, b.input(1), 1);
    let index = b.rr(Op::Add, b.ri(Op::MulI, b.input(0), REPLICAS as i64), b.input(1));
    let index = b.select(present, index, b.konst(0));
    let index = gb.add_prog_i64(b.finish(&[(index, 8)]), vec![at, slots], vec![]);
    let ips = gb.add_ext_peer(TABLE_PEER | 1);
    let ips = gb.add_index_cells(ips, index);
    let ids = gb.push(GraphNode::CellIds { cells: ips });
    let b = Builder::new(2);
    let present = b.ri(Op::AddI, b.input(1), 1);
    let key = b.select(present, b.ri(Op::AddI, b.input(0), 1), b.konst(0));
    gb.add_prog_i64(b.finish(&[(key, 8)]), vec![ids, slots], vec![])
}

fn argsort_deletion_graph(
    gb: &mut GraphBuilder,
    base: u16,
    choice: &DeletionChoice,
    key: usize,
    replicas: bool,
    split: Option<(u16, &[usize], usize)>,
    ordered: bool,
) -> u16 {
    let keys = if replicas { replica_keys(gb, choice, key, split) } else { gb.add_ints(&choice.keys) };
    let at = gb.add_ext_align(key as u32, Some(TABLE_PEER));
    // Unselected rows may name blocks absent from the file table. Their sort key is immaterial.
    let b = Builder::new(2);
    let safe = b.select(b.input(1), b.input(0), b.konst(0));
    let at = gb.add_prog_i64(b.finish(&[(safe, 8)]), vec![at, keys], vec![]);
    let blocks = gb.add_ext_peer(TABLE_PEER);
    let blocks = gb.add_index_cells(blocks, at);
    let values = gb.push(GraphNode::ParseAsciiInt { cells: blocks });
    if ordered {
        return gb.push(GraphNode::OrderedSum {
            keys, deltas: base, order: values, zero: deletion_mode() == DeletionMode::ReplicaOrderedZero,
        });
    }
    let order = gb.push(GraphNode::SortIndices { values });
    let keys = gather_ints(gb, keys, order);
    let deltas = gather_ints(gb, base, order);
    let sums = gb.push(GraphNode::KeyedSum { keys, deltas });
    // Sorting a permutation's indices produces its inverse.
    let inverse = gb.push(GraphNode::SortIndices { values: order });
    gather_ints(gb, sums, inverse)
}

fn order_deletions(recs: &[&[u8]], order: &mut [u32]) {
    let mut positions = Vec::new();
    let mut rows = Vec::new();
    for (at, &row) in order.iter().enumerate() {
        let rec = recs[row as usize];
        if memchr::memmem::find(rec, DELETING).is_none() {
            continue;
        }
        let id = block_token(rec)
            .and_then(|b| std::str::from_utf8(&b[4..]).ok())
            .and_then(|b| b.parse::<i64>().ok());
        if let Some(id) = id {
            positions.push(at);
            rows.push((id, at, row));
        }
    }
    rows.sort_unstable();
    for (at, (_, _, row)) in positions.into_iter().zip(rows) {
        order[at] = row;
    }
}

#[cfg(test)]
mod ablation_tests {
    use super::*;
    use crate::payload::graph::CoreDecoder;
    use crate::wire::DecodeBudget;

    #[test]
    fn ordered_sums_keep_passthrough_rows_out_of_the_clock_state() {
        let mut gb = GraphBuilder::new(6);
        let keys = gb.add_ints(&[1, 0, 2, 1, 2, 1]);
        let deltas = gb.add_ints(&[3, 999, 4, 10, 2, -1]);
        let order = gb.add_ints(&[9, -100, 0, -3, 0, 10]);
        let root = gb.push(GraphNode::OrderedSum { keys, deltas, order, zero: false });
        let wire = gb.finish(root).encode();
        let got = CoreDecoder::decode_ints_bytes(&wire, &DecodeBudget::default()).unwrap();
        assert_eq!(got, [13, 999, 14, 10, 16, 12]);
    }

    #[test]
    fn zero_seed_sums_are_independent_between_keys() {
        let mut gb = GraphBuilder::new(5);
        let keys = gb.add_ints(&[1, 0, 2, 1, 2]);
        let deltas = gb.add_ints(&[3, 999, 4, 10, 2]);
        let order = gb.add_ints(&[9, -100, 0, -3, 0]);
        let root = gb.push(GraphNode::OrderedSum { keys, deltas, order, zero: true });
        let wire = gb.finish(root).encode();
        let got = CoreDecoder::decode_ints_bytes(&wire, &DecodeBudget::default()).unwrap();
        assert_eq!(got, [13, 999, 4, 10, 6]);
    }

    #[test]
    fn sort_indices_are_stable_and_invert_permutations() {
        let mut gb = GraphBuilder::new(6);
        let values = gb.add_ints(&[i64::MAX, -9, 3, -9, i64::MIN, 3]);
        let order = gb.push(GraphNode::SortIndices { values });
        let graph = gb.finish(order);
        let budget = DecodeBudget::default();
        assert_eq!(CoreDecoder::decode_ints_bytes(&graph.encode(), &budget).unwrap(), [4, 1, 3, 2, 5, 0]);
        let mut gb = GraphBuilder::new(6);
        let values = gb.add_ints(&[i64::MAX, -9, 3, -9, i64::MIN, 3]);
        let order = gb.push(GraphNode::SortIndices { values });
        let sorted = gather_ints(&mut gb, values, order);
        let inverse = gb.push(GraphNode::SortIndices { values: order });
        let restored = gather_ints(&mut gb, sorted, inverse);
        let wire = gb.finish(restored).encode();
        assert_eq!(CoreDecoder::decode_ints_bytes(&wire, &budget).unwrap(), [i64::MAX, -9, 3, -9, i64::MIN, 3]);
        let tight = DecodeBudget { max_single_alloc_bytes: 8, ..budget };
        assert!(CoreDecoder::decode_ints_bytes(&wire, &tight).is_err());
    }

    #[test]
    fn simpler_graphs_preserve_sparse_rows_negative_times_and_first_key_seeds() {
        let choice = DeletionChoice {
            picks: vec![1, 1, 0, 1, 1, 0, 1],
            keys: vec![1, 2, 0, 1, 2, 0, 3],
            order: vec![3, 1, 4, 0, 6],
            targets: vec![12, -4, 70, 10, -3, 80, 19],
            slots: vec![],
        };
        for mode in [DeletionMode::KeyedRow, DeletionMode::KeyedSorted, DeletionMode::ScanGrouped] {
            let mut deltas = choice.targets.clone();
            prepare_deletions(&choice, &mut deltas, mode);
            let mut gb = GraphBuilder::new(deltas.len());
            let base = gb.add_ints(&deltas);
            let root = simple_deletion_graph(&mut gb, base, &choice, mode);
            let wire = gb.finish(root).encode();
            let got = CoreDecoder::decode_ints_bytes(&wire, &DecodeBudget::default()).unwrap();
            assert_eq!(got, choice.targets);
        }
    }
}
