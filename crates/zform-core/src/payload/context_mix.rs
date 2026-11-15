//! Integer-only mixture of three binary context models for bounded integer lanes.
//! This fixed preset is the kind-9 wire contract: changing its predictions needs a new kind.
use crate::wire::{DecodeBudget, Error, Result};

pub(super) fn validate(
    len: usize,
    bits: u8,
    tb: u8,
    bytes: usize,
    budget: &DecodeBudget,
) -> Result<()> {
    if len == 0 || !(1..=16).contains(&bits) || !(8..=14).contains(&tb) {
        return Err(Error::Format("invalid context mixing parameters".into()));
    }
    if bytes < 4
        || bytes
            > len
                .saturating_mul(bits as usize)
                .saturating_mul(2)
                .saturating_add(4)
    {
        return Err(Error::Format("invalid context mixing byte count".into()));
    }
    if bytes > budget.max_frame_compressed_bytes {
        return Err(Error::BudgetExceeded("context stream bytes"));
    }
    let tables = std::mem::size_of::<[[Counter; MAX_SLOTS]; K]>();
    budget.check_alloc(tables, "context mixing tables")?;
    budget.check_alloc(len.saturating_mul(8), "context mixing output")?;
    if tables
        .saturating_add(len.saturating_mul(8))
        .saturating_add(1024)
        > budget.max_block_decoded_bytes
    {
        return Err(Error::BudgetExceeded("context mixing working set"));
    }
    Ok(())
}
const K: usize = 3;
#[derive(Clone, Copy)]
struct Counter {
    p: u16,
    n: u8,
}
/// Tables are allocated at their largest size so masked indexes need no bounds checks.
const MAX_SLOTS: usize = 1 << 14;
struct Model {
    tables: Box<[[Counter; MAX_SLOTS]; K]>,
    weights: [[u32; K]; 16],
    rates: [i32; 64],
    mask: usize,
}
fn hash(mut x: u64) -> usize {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    (x ^ (x >> 31)) as usize
}
impl Model {
    fn new(tb: u8) -> Self {
        Self {
            tables: vec![[Counter { p: 32768, n: 0 }; MAX_SLOTS]; K]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            weights: [[21845, 21845, 21846]; 16],
            rates: std::array::from_fn(|n| 65536 / (n as i32 + 2)),
            mask: (1usize << tb) - 1,
        }
    }
    #[inline(always)]
    fn indexes(&self, prefix: u32, prev: u32, prev2: u32) -> [usize; K] {
        [
            prefix as usize & self.mask & (MAX_SLOTS - 1),
            hash((prev as u64) << 32 | prefix as u64) & self.mask & (MAX_SLOTS - 1),
            hash((prev2 as u64) << 48 | (prev as u64) << 16 | prefix as u64)
                & self.mask
                & (MAX_SLOTS - 1),
        ]
    }
    #[inline(always)]
    fn predict(&self, ids: &[usize; K], depth: usize) -> u32 {
        let mut sum = 0u64;
        for k in 0..K {
            sum += self.weights[depth & 15][k] as u64 * self.tables[k][ids[k]].p as u64;
        }
        ((sum >> 16) as u32).clamp(1, 65535)
    }
    #[inline(always)]
    fn update(&mut self, ids: &[usize; K], depth: usize, bit: bool) {
        let weights = &mut self.weights[depth & 15];
        let likelihood: [u64; K] = std::array::from_fn(|k| {
            let p = self.tables[k][ids[k]].p as u64;
            weights[k] as u64 * if bit { p } else { 65536 - p }
        });
        let total = likelihood.iter().sum::<u64>();
        for k in 0..K - 1 {
            weights[k] = (((likelihood[k] << 16) / total * 31 + 21845) >> 5) as u32;
        }
        weights[K - 1] = 65536 - weights[0] - weights[1];
        for k in 0..K {
            let c = &mut self.tables[k][ids[k]];
            let rate = self.rates[c.n as usize & 63];
            c.p = (c.p as i32 + (((if bit { 65536 } else { 0 }) - c.p as i32) * rate >> 16))
                .clamp(1, 65535) as u16;
            c.n = (c.n + 1).min(63);
        }
    }
}

pub(super) fn encode(vals: &[i64], min: i64, bits: u8, tb: u8, limit: usize) -> Option<Vec<u8>> {
    assert!((1..=16).contains(&bits) && (8..=14).contains(&tb));
    let mut model = Model::new(tb);
    let (mut lo, mut hi) = (0u32, u32::MAX);
    let (mut prev, mut prev2) = (0u32, 0u32);
    let mut out = Vec::new();
    for &value in vals {
        let x = value.wrapping_sub(min) as u32;
        assert!((x as u64) < (1u64 << bits));
        let mut prefix = 1u32;
        for depth in 0..bits as usize {
            let ids = model.indexes(prefix, prev, prev2);
            let p = model.predict(&ids, depth);
            let bit = x & (1 << (bits as usize - 1 - depth)) != 0;
            let mid = lo + (((hi - lo) as u64 * p as u64) >> 16) as u32;
            if bit {
                hi = mid;
            } else {
                lo = mid + 1;
            }
            while (lo ^ hi) < 1 << 24 {
                if out.len().saturating_add(5) > limit {
                    return None;
                }
                out.push((hi >> 24) as u8);
                lo <<= 8;
                hi = (hi << 8) | 255;
            }
            model.update(&ids, depth, bit);
            prefix = prefix * 2 + bit as u32;
        }
        prev2 = prev;
        prev = x;
    }
    if out.len().saturating_add(4) > limit {
        return None;
    }
    for _ in 0..4 {
        out.push((lo >> 24) as u8);
        lo <<= 8;
    }
    Some(out)
}

pub(super) fn decode(
    bytes: &[u8],
    len: usize,
    min: i64,
    bits: u8,
    tb: u8,
    budget: &DecodeBudget,
) -> Result<Vec<i64>> {
    validate(len, bits, tb, bytes.len(), budget)?;
    let mut at = 0;
    let mut next = || {
        let b = *bytes
            .get(at)
            .ok_or_else(|| Error::Format("truncated context stream".into()))?;
        at += 1;
        Ok::<u8, Error>(b)
    };
    let mut code = 0u32;
    for _ in 0..4 {
        code = (code << 8) | next()? as u32;
    }
    let mut model = Model::new(tb);
    let (mut lo, mut hi) = (0u32, u32::MAX);
    let (mut prev, mut prev2) = (0u32, 0u32);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let mut prefix = 1u32;
        for depth in 0..bits as usize {
            let ids = model.indexes(prefix, prev, prev2);
            let p = model.predict(&ids, depth);
            let mid = lo + (((hi - lo) as u64 * p as u64) >> 16) as u32;
            let bit = code <= mid;
            if bit {
                hi = mid;
            } else {
                lo = mid + 1;
            }
            while (lo ^ hi) < 1 << 24 {
                lo <<= 8;
                hi = (hi << 8) | 255;
                code = (code << 8) | next()? as u32;
            }
            model.update(&ids, depth, bit);
            prefix = prefix * 2 + bit as u32;
        }
        let x = prefix - (1 << bits);
        out.push(min.wrapping_add(x as i64));
        prev2 = prev;
        prev = x;
    }
    if at != bytes.len() {
        return Err(Error::Format("trailing context bytes".into()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::graph::{CoreDecoder, GraphBuilder, IntStream, with_context_mixing};

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn context_mixing_roundtrips_and_bounds_untrusted_parameters() {
        let budget = DecodeBudget::default();
        for (bits, min) in [(1, 0), (3, -9), (8, i64::MIN), (16, i64::MAX - 100)] {
            let mut rng = 123;
            let values: Vec<_> = (0..97)
                .map(|_| min.wrapping_add((next(&mut rng) & ((1 << bits) - 1)) as i64))
                .collect();
            let wire = encode(&values, min, bits, 8, usize::MAX).unwrap();
            assert_eq!(
                decode(&wire, values.len(), min, bits, 8, &budget).unwrap(),
                values
            );
            for cut in [0, 1, 3, wire.len() / 2, wire.len() - 1] {
                assert!(decode(&wire[..cut], values.len(), min, bits, 8, &budget).is_err());
            }
            let mut trailing = wire.clone();
            trailing.push(0);
            assert!(decode(&trailing, values.len(), min, bits, 8, &budget).is_err());
            assert!(encode(&values, min, bits, 8, 3).is_none());
            for (b, t) in [(0, 8), (17, 8), (1, 7), (1, 15), (255, 255)] {
                assert!(decode(&wire, values.len(), min, b, t, &budget).is_err());
            }
            let tiny = DecodeBudget {
                max_single_alloc_bytes: 1024,
                ..budget
            };
            assert!(matches!(
                decode(&wire, values.len(), min, bits, 8, &tiny),
                Err(Error::BudgetExceeded(_))
            ));
            let tiny = DecodeBudget {
                max_block_decoded_bytes: 1024,
                ..budget
            };
            assert!(matches!(
                decode(&wire, values.len(), min, bits, 8, &tiny),
                Err(Error::BudgetExceeded(_))
            ));
        }
    }

    #[test]
    fn context_mixing_selection_is_optional_and_payloads_are_self_contained() {
        let (mut rng, mut value) = (17, 0);
        let values: Vec<i64> = (0..8192)
            .map(|_| {
                if next(&mut rng) % 10 == 0 {
                    value ^= 1;
                }
                value
            })
            .collect();
        let plain = IntStream::auto_encode(&values);
        let mixed = with_context_mixing(true, || IntStream::auto_encode(&values));
        assert!(matches!(mixed, IntStream::ContextMix { .. }), "{mixed:?}");
        assert_eq!(
            plain,
            IntStream::auto_encode(&values),
            "option leaked out of scope"
        );
        let mut a = Vec::new();
        plain.encode(&mut a);
        let mut b = Vec::new();
        mixed.encode(&mut b);
        assert!(b.len() + 32 <= a.len());
        let budget = DecodeBudget::default();
        for _ in 0..2 {
            let mut gb = GraphBuilder::new(values.len());
            let root = gb.push(crate::payload::graph::GraphNode::IntStream(mixed.clone()));
            assert_eq!(
                CoreDecoder::decode_ints_bytes(&gb.finish(root).encode(), &budget).unwrap(),
                values
            );
        }
        let constant = vec![5; 2048];
        assert_eq!(
            with_context_mixing(true, || IntStream::auto_encode(&constant)),
            IntStream::auto_encode(&constant)
        );
        let noisy: Vec<i64> = (0..2048).map(|_| next(&mut rng) as i64).collect();
        assert_eq!(
            with_context_mixing(true, || IntStream::auto_encode(&noisy)),
            IntStream::auto_encode(&noisy)
        );
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use crate::cells::Col;
    use crate::payload::graph::{
        CoreDecoder, ExtInputs, GraphBuilder, GraphNode, IntStream, PayloadGraph,
    };

    #[test]
    fn context_mixing_wire_fixture() {
        let values = [-3, -2, -3, 4, -1, 4, 4, 0, -3, 4, 1, 2, -2, 4, -3, 3];
        let bytes = encode(&values, -3, 3, 8, 100).unwrap();
        assert_eq!(bytes, [241, 175, 227, 179, 9, 212, 128, 230, 255, 76]);
        assert_eq!(
            decode(&bytes, values.len(), -3, 3, 8, &DecodeBudget::default()).unwrap(),
            values
        );
    }

    #[test]
    fn context_mixing_graph_keeps_lazy_dictionary_reads() {
        let mut state = 7u64;
        let mut v = 0i64;
        let values: Vec<_> = (0..8192)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state % 10 == 0 {
                    v ^= 1;
                }
                v
            })
            .collect();
        let bytes = encode(&values, 0, 1, 14, usize::MAX).unwrap();
        let mut gb = GraphBuilder::new(values.len());
        let buf = gb.add_bytes(b"noyes");
        let dict = gb.add_slice_buf(buf, 0, &[2, 3]);
        let codes = gb.push(GraphNode::IntStream(IntStream::ContextMix {
            len: values.len() as u32,
            min: 0,
            bits: 1,
            table_bits: 14,
            bytes,
        }));
        let root = gb.add_index_cells(dict, codes);
        let budget = DecodeBudget::default();
        let graph = PayloadGraph::decode(&gb.finish(root).encode(), &budget).unwrap();
        let ext = ExtInputs::empty();
        let col =
            CoreDecoder::decode_col_reading(&graph, &ext, &budget, Some(&[(7, 8), (8191, 8192)]))
                .unwrap();
        assert!(matches!(col, Col::Dict { .. }));
        let mut scratch = Vec::new();
        for i in [7, 8191] {
            assert_eq!(
                col.get(i, &mut scratch).unwrap(),
                if values[i] == 0 {
                    b"no".as_slice()
                } else {
                    b"yes".as_slice()
                }
            );
        }
    }
}
