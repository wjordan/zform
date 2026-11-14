//! Manual A/B of template-specific lanes versus shared semantic slots.
//! Whole-input profile search, not a streaming planner. Candidate prices omit peer savings;
//! compare complete archives before accepting a plan. `none` is the fixed-profile control.
use std::{
    env,
    fs::{self, File},
    time::Instant,
};
use zform_core::{
    PackOptions, Profile,
    grammar::{ModelPlan, PlanVariant},
    pack_stream, train_from_bytes,
};

fn adaptive(plan: &mut ModelPlan, raw: &[u8], fast: bool) {
    use std::collections::{BTreeMap, HashMap, HashSet};
    let fc = zform_core::framing::split_chunk(raw, Default::default());
    let recs: Vec<_> = (0..fc.len()).map(|i| fc.record(i)).collect();
    let split = zform_core::grammar::split_records_with_plan(&recs, plan);
    let vals: Vec<Vec<_>> = split
        .col_occs
        .iter()
        .map(|c| c.iter().map(|v| v.as_slice(&split.scratch)).collect())
        .collect();
    let mut owners = vec![Vec::new(); plan.columns.len()];
    for &vi in &split.variant_ids {
        if let PlanVariant::Shape(s) = &plan.variants[vi as usize] {
            for &c in &s.slots {
                owners[c as usize].push(vi);
            }
        }
    }
    let packs = zform_core::packs::PackRegistry::default().with_context_mixing();
    let price = |v: &[&[u8]]| {
        if fast {
            let sample = zform_core::plan::sample_strided(v, 4096, 1 << 20);
            packs.encode_best_graph(&sample).encode().len() * v.len() / sample.len().max(1)
        } else {
            packs.encode_best_graph(v).encode().len()
        }
    };
    let original = plan.columns.len();
    for cid in 1..original {
        if vals[cid].len() < 512
            || owners[cid].len() != vals[cid].len()
            || plan.columns[cid].nested_plan.is_some()
        {
            continue;
        }
        let mut groups: BTreeMap<u32, Vec<&[u8]>> = BTreeMap::new();
        for (&v, &vi) in vals[cid].iter().zip(&owners[cid]) {
            groups.entry(vi).or_default().push(v);
        }
        if groups.len() < 2 {
            continue;
        }
        if fast {
            let medians: Vec<_> = groups
                .values()
                .filter_map(|v| {
                    let mut bits: Vec<_> = v
                        .iter()
                        .take(64)
                        .filter_map(|s| {
                            std::str::from_utf8(s)
                                .ok()?
                                .parse::<i64>()
                                .ok()
                                .map(|n| 64 - n.unsigned_abs().leading_zeros())
                        })
                        .collect();
                    if bits.len() < v.len().min(64) {
                        return None;
                    }
                    bits.sort_unstable();
                    Some(bits[bits.len() / 2])
                })
                .collect();
            if medians.len() != groups.len()
                || medians.iter().max().unwrap() - medians.iter().min().unwrap() < 8
            {
                continue;
            }
        }
        let base = price(&vals[cid]);
        if base < 256 {
            continue;
        }
        let mut mapped: HashMap<u32, u32> = groups.keys().map(|&v| (v, v)).collect();
        let mut costs: BTreeMap<u32, usize> =
            groups.iter().map(|(&vi, v)| (vi, price(v))).collect();
        let keys: Vec<_> = groups.keys().copied().collect();
        for (j, &a) in keys.iter().enumerate() {
            if !groups.contains_key(&a) {
                continue;
            }
            for &b in &keys[j + 1..] {
                let Some(bvals) = groups.get(&b) else {
                    continue;
                };
                let aset: HashSet<_> = groups[&a].iter().take(256).copied().collect();
                let bset: HashSet<_> = bvals.iter().take(256).copied().collect();
                if aset.intersection(&bset).count() * 4 < aset.len().min(bset.len()) * 3 {
                    continue;
                }
                let joined: Vec<_> = vals[cid]
                    .iter()
                    .zip(&owners[cid])
                    .filter_map(|(&v, vi)| matches!(mapped[vi],x if x==a || x==b).then_some(v))
                    .collect();
                let cost = price(&joined);
                if cost < costs[&a] + costs[&b] {
                    for group in mapped.values_mut() {
                        if *group == b {
                            *group = a;
                        }
                    }
                    groups.insert(a, joined);
                    groups.remove(&b);
                    costs.insert(a, cost);
                    costs.remove(&b);
                }
            }
        }
        let cost =
            costs.values().sum::<usize>() + groups.len() * (plan.columns[cid].name.len() + 32);
        if cost * 10 >= base * 9 {
            continue;
        }
        eprintln!(
            "split {}: estimated {} -> {} bytes, {} groups",
            plan.columns[cid].name,
            base,
            cost,
            groups.len()
        );
        let mut newcols = HashMap::new();
        for &key in groups.keys() {
            let mut copy = plan.columns[cid].clone();
            copy.name = format!("{}.p{key}", copy.name);
            copy.horiz_candidates.clear();
            newcols.insert(key, plan.columns.len() as u16);
            plan.columns.push(copy);
        }
        for (vi, v) in plan.variants.iter_mut().enumerate() {
            if let PlanVariant::Shape(s) = v {
                for col in &mut s.slots {
                    if *col as usize == cid {
                        if let Some(group) = mapped.get(&(vi as u32)) {
                            *col = newcols[group];
                        }
                    }
                }
            }
        }
    }
    plan.rebuild_indexes();
}

fn partition(plan: &mut ModelPlan, mode: &str) {
    let originals = plan.columns.clone();
    for (vi, v) in plan.variants.iter_mut().enumerate() {
        if let PlanVariant::Shape(s) = v {
            for (si, col) in s.slots.iter_mut().enumerate() {
                let c = &originals[*col as usize];
                let select = match mode {
                    "all" => *col != 0,
                    "numbers" => {
                        c.name.starts_with("$int.")
                            || c.name.starts_with("$dec.")
                            || c.name.starts_with("$ts.")
                            || c.name == "$blk_id"
                    }
                    "body" => {
                        (c.name.starts_with("$int.") || c.name.starts_with("$dec."))
                            && s.literals[..=si].iter().any(|v| v.contains(&b'|'))
                    }
                    "block" => c.name == "$blk_id",
                    "clock" => c.name.starts_with("$ts."),
                    _ => panic!("unknown mode"),
                };
                if select {
                    let mut copy = c.clone();
                    copy.name = format!("{}.v{vi}", copy.name);
                    copy.horiz_candidates.clear();
                    *col = plan.columns.len() as u16;
                    plan.columns.push(copy);
                }
            }
        }
    }
    plan.rebuild_indexes();
}
fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 4, "partition_probe INPUT OUTPUT MODE");
    let start = Instant::now();
    let raw = fs::read(&args[1]).unwrap();
    let mut plan = train_from_bytes(&raw, &Default::default(), None);
    if args[3].starts_with("adaptive") {
        adaptive(&mut plan, &raw, args[3] == "adaptive-fast");
    } else if args[3] != "none" {
        partition(&mut plan, &args[3]);
    }
    let mut opts = PackOptions {
        profile: Some(Profile::from_plan(plan)),
        allow_drift_retrain: false,
        ..Default::default()
    };
    opts.packs = opts.packs.with_context_mixing();
    pack_stream(
        File::open(&args[1]).unwrap(),
        File::create(&args[2]).unwrap(),
        &opts,
    )
    .unwrap();
    println!(
        "bytes={} seconds={:.6}",
        fs::metadata(&args[2]).unwrap().len(),
        start.elapsed().as_secs_f64()
    );
}
