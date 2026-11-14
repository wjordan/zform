//! Manual comparison of clock lanes using the existing payload graph and decoder.
//! cargo run --release -p zform-core --example bgl_payload_probe -- ticks.i64
//! Timings cover payload construction/encoding and decoding, excluding input file I/O.
//! Also accepts `pack ticks.i64 clock.graph` and `unpack clock.graph ticks.i64` for bgl_cm.py.

use std::{env, fs, time::Instant};
use zform_core::payload::{
    graph::{CoreDecoder, GraphBuilder, with_encode_opts},
    ir::{Builder, Op},
};
use zform_core::wire::{DecodeBudget, EntropyBackend};

pub(crate) fn encode(ticks: &[i64], level: i32, bits: i64) -> Vec<u8> {
    with_encode_opts(EntropyBackend::Zstd, level, || {
        let mut graph = GraphBuilder::new(ticks.len());
        let mut deltas = Vec::with_capacity(ticks.len());
        deltas.push(0);
        deltas.extend(ticks.windows(2).map(|w| w[1].wrapping_sub(w[0])));
        let b = Builder::new(if bits == 0 { 1 } else { 2 });
        let (value, inputs) = if bits == 0 {
            (b.input(0), vec![graph.add_ints_fast(&deltas)])
        } else {
            let low: Vec<_> = deltas.iter().map(|v| v & ((1 << bits) - 1)).collect();
            let high: Vec<_> = deltas.iter().map(|v| v >> bits).collect();
            let lo = graph.add_ints_fast(&low);
            let hi = graph.add_ints_fast(&high);
            (
                b.rr(Op::Add, b.input(0), b.ri(Op::ShlI, b.input(1), bits)),
                vec![lo, hi],
            )
        };
        let value = b.scan_add(value, ticks[0]);
        let root = graph.add_prog_i64(b.finish(&[(value, 8)]), inputs, vec![]);
        graph.finish(root).encode()
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().collect();
    let budget = DecodeBudget {
        max_records_per_block: 16_000_000,
        max_single_alloc_bytes: 512 << 20,
        ..DecodeBudget::default()
    };
    if args.len() == 4 && args[1] == "unpack" {
        let wire = fs::read(&args[2])?;
        let ticks = CoreDecoder::decode_ints_bytes(&wire, &budget)?;
        let raw: Vec<_> = ticks.iter().flat_map(|t| t.to_le_bytes()).collect();
        fs::write(&args[3], raw)?;
        return Ok(());
    }
    let packing = args.len() == 4 && args[1] == "pack";
    if !packing && args.len() != 2 {
        return Err(
            "expected ticks.i64, pack ticks.i64 clock.graph, or unpack clock.graph ticks.i64"
                .into(),
        );
    }
    let raw = fs::read(&args[if packing { 2 } else { 1 }])?;
    if raw.is_empty() || raw.len() % 8 != 0 {
        return Err("expected a nonempty i64 lane".into());
    }
    let ticks: Vec<_> = raw
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if packing {
        fs::write(&args[3], encode(&ticks, 3, 10))?;
        return Ok(());
    }
    for level in [3, 9] {
        for bits in [0, 8, 10] {
            let start = Instant::now();
            let wire = encode(&ticks, level, bits);
            let encode = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let restored = CoreDecoder::decode_ints_bytes(&wire, &budget)?;
            let decode = start.elapsed().as_secs_f64();
            assert_eq!(restored, ticks, "payload clock roundtrip");
            println!(
                "{{\"rows\":{},\"level\":{level},\"split_bits\":{bits},\"bytes\":{},\"encode_seconds\":{encode:.6},\"decode_seconds\":{decode:.6},\"exact\":true}}",
                ticks.len(),
                wire.len()
            );
        }
    }
    Ok(())
}
