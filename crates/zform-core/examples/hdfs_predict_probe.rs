//! Manual comparison of HDFS occurrence-order models using existing payload operations.
use std::{collections::HashMap, env, fs};
use zform_core::{
    packs::PackRegistry,
    payload::graph::{CoreDecoder, GraphBuilder, IntStream, PayloadGraph},
    wire::DecodeBudget,
};

fn grouped(rows: &[&[u8]], keys: &[&[u8]], packs: &PackRegistry) -> PayloadGraph {
    let mut ids = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        let next = ids.len();
        let id = *ids.entry(k).or_insert(next);
        if id == groups.len() {
            groups.push(Vec::new());
        }
        groups[id].push(i);
    }
    let order: Vec<i64> = groups.into_iter().flatten().map(|i| i as i64).collect();
    let values: Vec<_> = order.iter().map(|&i| rows[i as usize]).collect();
    let mut gb = GraphBuilder::new(rows.len());
    let source = gb.merge_graph(&packs.encode_best_graph(&values));
    let hay = gb.add_ints(&order);
    let needles = gb.add_row_index(rows.len());
    let codes = gb.push(zform_core::payload::graph::GraphNode::IntStream(
        IntStream::Const {
            val: 1,
            len: rows.len() as u32,
        },
    ));
    let idx = gb.add_match_index(needles, hay, codes);
    let root = gb.add_index_cells(source, idx);
    gb.finish(root)
}

fn keyed(rows: &[&[u8]], keys: &[&[u8]], width: usize, packs: &PackRegistry) -> PayloadGraph {
    let mut ids = HashMap::new();
    let mut dict = Vec::new();
    let mut caches: HashMap<&[u8], Vec<i64>> = HashMap::new();
    let mut codes = Vec::new();
    let mut misses = Vec::new();
    for (&v, &key) in rows.iter().zip(keys) {
        let next = ids.len() as i64;
        let id = *ids.entry(v).or_insert_with(|| {
            dict.push(v);
            next
        });
        let cache = caches.entry(key).or_default();
        if let Some(rank) = cache.iter().position(|&x| x == id) {
            codes.push(rank as i64 + 1);
            cache.remove(rank);
        } else {
            codes.push(0);
            misses.push(id);
            if cache.len() == width {
                cache.pop();
            }
        }
        cache.insert(0, id);
    }
    eprintln!(
        "keyed width={width} miss={}/{} keys={}",
        misses.len(),
        rows.len(),
        caches.len()
    );
    let mut gb = GraphBuilder::new(rows.len());
    gb.merge_graph(&packs.encode_best_graph(&dict));
    gb.add_ints(&codes);
    let root = gb.add_mtf_ints(&misses);
    gb.finish(root)
}

fn delta_probe(rows: &[&[u8]], keys: &[&[u8]], packs: &PackRegistry, name: &str) {
    let base = packs.encode_best_graph(rows);
    let base_cm = base.clone().with_context_mixing().encode().len();
    let mut best = usize::MAX;
    for dictionary in [false, true] {
        let mut ids = HashMap::new();
        let mut dict = Vec::new();
        let mut prev: HashMap<&[u8], i64> = HashMap::new();
        let mut delta = Vec::new();
        for (&v, &k) in rows.iter().zip(keys) {
            let val = if dictionary {
                let next = ids.len() as i64;
                *ids.entry(v).or_insert_with(|| {
                    dict.push(v);
                    next
                })
            } else if let Some(n) = std::str::from_utf8(v)
                .ok()
                .and_then(|s| s.parse::<i64>().ok())
            {
                n
            } else {
                return;
            };
            let last = prev.entry(k).or_default();
            delta.push(val.wrapping_sub(*last));
            *last = val;
        }
        let mut gb = GraphBuilder::new(rows.len());
        if dictionary {
            gb.merge_graph(&packs.encode_best_graph(&dict));
        }
        let root = gb.add_ints(&delta);
        let graph = gb.finish(root);
        let bytes = graph.encode().len();
        let cm = graph.with_context_mixing().encode().len();
        if cm < best {
            best = cm;
        }
        if bytes + 100 < base.encode().len() || cm + 100 < base_cm {
            println!(
                "DELTA {name} dict={dictionary} base={}/{} candidate={bytes}/{cm} rows={} keys={} (excludes new node/format)",
                base.encode().len(),
                base_cm,
                rows.len(),
                prev.len()
            );
        }
    }
}

fn main() {
    for path in env::args().skip(1) {
        let raw = fs::read(&path).unwrap();
        let mut blocks = Vec::new();
        let mut emitters = Vec::new();
        let mut operations = Vec::new();
        for line in raw.split(|&b| b == b'\n') {
            let fields: Vec<_> = line
                .split(|b| b.is_ascii_whitespace())
                .filter(|s| !s.is_empty())
                .collect();
            if fields.len() < 7 {
                continue;
            }
            if let Some(i) = line.windows(4).position(|b| b == b"blk_") {
                let tail = &line[i + 4..];
                let n = tail
                    .iter()
                    .take_while(|&&b| b == b'-' || b.is_ascii_digit())
                    .count();
                if n == 0 {
                    continue;
                }
                blocks.push(&tail[..n]);
                emitters.push(fields[2]);
                operations.push(if fields[5] == b"BLOCK*" {
                    fields[6]
                } else {
                    fields[5]
                });
            }
        }
        let plan = zform_core::plan::train_from_bytes(&raw, &Default::default(), None);
        let packs = PackRegistry::default();
        let fc = zform_core::framing::split_chunk(&raw, Default::default());
        let records: Vec<_> = (0..fc.len()).map(|i| fc.record(i)).collect();
        let split = zform_core::grammar::split_records_with_plan(&records, &plan);
        let occ: Vec<Vec<&[u8]>> = split
            .col_occs
            .iter()
            .map(|col| col.iter().map(|v| v.as_slice(&split.scratch)).collect())
            .collect();
        for cid in 1..occ.len() {
            if occ[cid].len() < 512 {
                continue;
            }
            for key in 1..cid {
                if let Ok(align) =
                    zform_core::codecs::align_cells(&plan, &split.variant_ids, key, cid)
                {
                    let keys: Vec<_> = align.iter().map(|&i| occ[key][i as usize]).collect();
                    delta_probe(
                        &occ[cid],
                        &keys,
                        &packs,
                        &format!(
                            "{}:{} <- {}",
                            path, plan.columns[cid].name, plan.columns[key].name
                        ),
                    );
                }
            }
        }
        delta_probe(&blocks, &operations, &packs, "blocks <- operations");
        for (key_name, keys) in [("emitter", &emitters), ("operation", &operations)] {
            for width in [1, 4, 16, 256, 4096] {
                let graph = keyed(&blocks, keys, width, &packs);
                println!(
                    "{path} {key_name}-keyed-{width} bytes={} cm_bytes={} (plus node framing)",
                    graph.encode().len(),
                    graph.with_context_mixing().encode().len()
                );
            }
        }
        for (name, graph) in [
            ("baseline", packs.encode_best_graph(&blocks)),
            ("by_block", grouped(&blocks, &blocks, &packs)),
            ("by_emitter", grouped(&blocks, &emitters, &packs)),
        ] {
            for mixing in [false, true] {
                let g = if mixing {
                    graph.clone().with_context_mixing()
                } else {
                    graph.clone()
                };
                println!(
                    "{path} {name} cm={mixing} rows={} bytes={}",
                    blocks.len(),
                    g.encode().len()
                );
                let got =
                    CoreDecoder::decode_col(&g, &Default::default(), &DecodeBudget::default())
                        .unwrap()
                        .to_cells()
                        .unwrap();
                assert!(got.iter().eq(blocks.iter().copied()));
            }
        }
    }
}
