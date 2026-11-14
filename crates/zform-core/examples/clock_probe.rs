//! Manual A/B: apply the existing combined clock pack to the ordinary record planner.
use std::{env, fs::File, time::Instant};
use zform_core::{PackOptions, pack_stream, packs::BglClockPack};

fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 3, "clock_probe INPUT OUTPUT");
    let mut opts = PackOptions::default();
    opts.packs = opts.packs.with_context_mixing();
    opts.packs.register(BglClockPack);
    let start = Instant::now();
    pack_stream(
        File::open(&args[1]).unwrap(),
        File::create(&args[2]).unwrap(),
        &opts,
    )
    .unwrap();
    println!(
        "bytes={} seconds={:.6}",
        std::fs::metadata(&args[2]).unwrap().len(),
        start.elapsed().as_secs_f64()
    );
}
