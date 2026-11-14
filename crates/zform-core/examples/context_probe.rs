//! Manual screening of context mixing on per-field dictionary IDs.
use std::{collections::HashMap, env, fs, time::Instant};
use zform_core::payload::graph::{IntStream, with_context_mixing};
use zform_core::wire::DecodeBudget;
fn main() {
    for path in env::args().skip(1) {
        let raw = fs::read(&path).unwrap();
        let rows: Vec<_> = raw
            .split(|&b| b == b'\n')
            .filter(|r| !r.is_empty())
            .take(65536)
            .collect();
        let mut maps: [HashMap<&[u8], i64>; 12] = std::array::from_fn(|_| HashMap::new());
        let mut cols: [Vec<i64>; 12] = std::array::from_fn(|_| Vec::new());
        for row in rows {
            for (i, field) in row
                .split(|b| b.is_ascii_whitespace() || *b == b'|')
                .filter(|s| !s.is_empty())
                .take(12)
                .enumerate()
            {
                let next = maps[i].len() as i64;
                cols[i].push(*maps[i].entry(field).or_insert(next));
            }
        }
        for (i, col) in cols.iter().enumerate() {
            let max = col.iter().copied().max().unwrap_or(0);
            let bits = (64 - (max as u64).leading_zeros()) as u8;
            if col.len() < 512 || !(1..=16).contains(&bits) {
                continue;
            }
            let start = Instant::now();
            let base = IntStream::auto_encode(col);
            let mut wire = Vec::new();
            base.encode(&mut wire);
            let base_ms = start.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            let selected = with_context_mixing(true, || IntStream::auto_encode(col));
            let mut encoded = Vec::new();
            selected.encode(&mut encoded);
            let cm_ms = start.elapsed().as_secs_f64() * 1000.;
            let start = Instant::now();
            let restored = selected.unpack(&DecodeBudget::default()).unwrap();
            let dec_ms = start.elapsed().as_secs_f64() * 1000.;
            assert_eq!(&restored, col);
            println!(
                "{{\"path\":{path:?},\"field\":{i},\"rows\":{},\"distinct\":{},\"bits\":{bits},\"base_bytes\":{},\"cm_bytes\":{},\"base_ms\":{base_ms:.3},\"cm_ms\":{cm_ms:.3},\"decode_ms\":{dec_ms:.3}}}",
                col.len(),
                max + 1,
                wire.len(),
                encoded.len()
            );
        }
    }
}
