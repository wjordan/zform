//! Manual HealthApp clock/header experiment; ordinary payload graph and reader.
use std::{
    env,
    fs::{self, File},
    time::Instant,
};
use zform_core::{
    PackOptions, Profile, block,
    grammar::{FrontEnd, ModelPlan, PlanVariant, Shape},
    pack_stream,
    packs::{CodecPack, PackRegistry},
    payload::{
        graph::{GraphBuilder, PayloadGraph},
        ir::{Builder, Op},
    },
};

struct Clock;
impl CodecPack for Clock {
    fn name(&self) -> &'static str {
        "probe/health-clock"
    }
    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.is_empty() {
            return None;
        }
        let mut dates = Vec::new();
        let mut ticks = Vec::new();
        for &v in rows {
            if !(16..=21).contains(&v.len()) || v[8] != b'-' {
                return None;
            }
            let fields: Vec<_> = v[9..].split(|&b| b == b':').collect();
            if fields.len() != 4 {
                return None;
            }
            let n = |s: &[u8]| -> Option<i64> {
                if s.is_empty() || s.len() > 3 {
                    return None;
                }
                let n = s.iter().try_fold(0, |n, &c| {
                    c.is_ascii_digit().then(|| n * 10 + i64::from(c - b'0'))
                })?;
                (n.to_string().as_bytes() == s).then_some(n)
            };
            let (h, m, s, ms) = (n(fields[0])?, n(fields[1])?, n(fields[2])?, n(fields[3])?);
            if h > 23 || m > 59 || s > 59 || ms > 999 {
                return None;
            }
            dates.push(&v[..8]);
            ticks.push(((h * 60 + m) * 60 + s) * 1000 + ms);
        }
        let mut gb = GraphBuilder::new(rows.len());
        let date = gb.merge_graph(&PackRegistry::default().encode_best_graph(&dates));
        let deltas: Vec<_> = std::iter::once(0)
            .chain(ticks.windows(2).map(|v| v[1] - v[0]))
            .collect();
        let lo: Vec<_> = deltas.iter().map(|v| v & 255).collect();
        let hi: Vec<_> = deltas.iter().map(|v| v >> 8).collect();
        let lo = gb.add_ints(&lo);
        let hi = gb.add_ints(&hi);
        let b = Builder::new(2);
        let delta = b.rr(Op::Add, b.input(0), b.ri(Op::ShlI, b.input(1), 8));
        let value = b.scan_add(delta, ticks[0]);
        let tick = gb.add_prog_i64(b.finish(&[(value, 8)]), vec![lo, hi], vec![]);
        let mut slots = vec![date];
        for (div, modulo) in [(3600000, 24), (60000, 60), (1000, 60), (1, 1000)] {
            let b = Builder::new(1);
            let part = b.div_wide(b.input(0), div);
            let q = b.div_wide(part, modulo);
            let val = b.rr(Op::Sub, part, b.ri(Op::MulI, q, modulo as i64));
            let ints = gb.add_prog_i64(b.finish(&[(val, 8)]), vec![tick], vec![]);
            slots.push(gb.add_prog_cells(
                zform_core::packs::prog_fmt_int(false, 0),
                vec![ints],
                vec![],
            ));
        }
        let template =
            gb.add_const_cells(b"\x1f\0-\x1f\x02:\x1f\x04:\x1f\x06:\x1f\x08", rows.len());
        let root =
            gb.push(zform_core::payload::graph::GraphNode::TemplateCells { template, slots });
        Some(gb.finish(root))
    }
}

fn main() {
    let args: Vec<_> = env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "health_clock_probe INPUT OUTPUT [--split]"
    );
    let raw = fs::read(&args[1]).unwrap();
    let fc = zform_core::framing::split_chunk(&raw, Default::default());
    let messages: Vec<_> = (0..fc.len())
        .filter_map(|i| fc.record(i).splitn(4, |&b| b == b'|').nth(3))
        .collect();
    let mut plan = ModelPlan::default();
    let slots: Vec<_> = ["time", "component", "process", "message"]
        .iter()
        .map(|s| plan.add_or_get_col(s))
        .collect();
    plan.columns[4].nested_plan =
        Some(Box::new(block::train(&messages, &Default::default(), None)));
    if args.get(3).is_some_and(|s| s == "--split") {
        let body = plan.columns[4].nested_plan.as_mut().unwrap();
        let originals = body.columns.clone();
        for (vi, variant) in body.variants.iter_mut().enumerate() {
            if let PlanVariant::Shape(shape) = variant {
                for col in &mut shape.slots {
                    let original = &originals[*col as usize];
                    if original.name.starts_with("$int.") || original.name.starts_with("$dec.") {
                        let mut copy = original.clone();
                        copy.name = format!("{}.v{vi}", copy.name);
                        copy.horiz_candidates.clear();
                        *col = body.columns.len() as u16;
                        body.columns.push(copy);
                    }
                }
            }
        }
        body.rebuild_indexes();
    }
    plan.columns[4].dict_group = "text".into();
    plan.push_variant(
        PlanVariant::Shape(Shape {
            literals: vec![vec![], b"|".to_vec(), b"|".to_vec(), b"|".to_vec(), vec![]],
            slots,
        }),
        FrontEnd::Template,
    );
    plan.rebuild_indexes();
    let mut opts = PackOptions {
        profile: Some(Profile::from_plan(plan)),
        allow_drift_retrain: false,
        ..Default::default()
    };
    opts.packs = opts.packs.with_context_mixing();
    opts.packs.register(Clock);
    let start = Instant::now();
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
