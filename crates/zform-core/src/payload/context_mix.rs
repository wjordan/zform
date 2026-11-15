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
type Tables = [[Counter; MAX_SLOTS]; K];
struct Model {
    tables: Box<Tables>,
    weights: [[u32; K]; 16],
    mask: usize,
}
/// The step of a counter seen `n` times; `n` stops at 63, and 256 entries spare a u8 index
/// its bounds check.
static RATES: [i32; 256] = {
    let mut rates = [0; 256];
    let mut n = 0;
    while n < 256 {
        rates[n] = 65536 / (if n < 63 { n } else { 63 } as i32 + 2);
        n += 1;
    }
    rates
};
/// The splitmix64 finalizer after its first xor-shift, which [`seed`] applies up front.
#[inline(always)]
fn mix(mut x: u64) -> usize {
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    (x ^ (x >> 31)) as usize
}
/// `x ^ x >> 30` of a hash input without its prefix bits. A prefix (below 2^17) fills
/// only zero bits of `x` and vanishes under `>> 30`, so it can be xored in afterwards.
fn seed(x: u64) -> u64 {
    x ^ x >> 30
}
/// `(l << 16) / total` given `scale = 65536.0 / total`, for `l <= total` and
/// `65536 <= total < 2^32`: one f64 division per update instead of two u64 ones.
///
/// Exact: the product is within 2^-36 of the quotient q (at most 2^16) and the subtraction
/// adds at most 2^-38. A fractional q is over 1/total > 2^-32 from an integer, so the value
/// sits strictly inside (floor(q) - 0.5, floor(q) + 0.5), which adding 1.5 * 2^52 rounds
/// to floor(q) in the low mantissa bits.
#[inline(always)]
fn share(l: u64, scale: f64) -> u32 {
    const MAGIC: f64 = 6755399441055744.0;
    ((l as i64 as f64 * scale - (0.5 - 1.0 / 17179869184.0)) + MAGIC).to_bits() as u32
}
/// What [`update`] needs from [`predict`]'s loads.
struct Slots {
    ids: [usize; K],
    ps: [u32; K],
    prods: [u64; 2],
    sum: u64,
}
impl Model {
    fn new(tb: u8) -> Self {
        Self {
            tables: vec![[Counter { p: 32768, n: 0 }; MAX_SLOTS]; K]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            weights: [[21845, 21845, 21846]; 16],
            mask: (1usize << tb) - 1,
        }
    }
    /// The tables, the weights and the index mask as separate borrows, so the coding
    /// loops keep the mask and table base in registers across table stores.
    fn parts(&mut self) -> (&mut Tables, &mut [[u32; K]; 16], usize) {
        (
            &mut self.tables,
            &mut self.weights,
            self.mask & (MAX_SLOTS - 1),
        )
    }
}
fn seeds(prev: u32, prev2: u32) -> [u64; 2] {
    [
        seed((prev as u64) << 32),
        seed((prev2 as u64) << 48 | (prev as u64) << 16),
    ]
}
/// The prediction at `prefix` under weights `ws`, and the slots it read.
#[inline(always)]
fn predict(tables: &Tables, ws: &[u32; K], m: usize, prefix: u32, seeds: [u64; 2]) -> (u32, Slots) {
    let ids = [
        prefix as usize & m,
        mix(seeds[0] ^ prefix as u64) & m,
        mix(seeds[1] ^ prefix as u64) & m,
    ];
    let ps: [u32; K] = std::array::from_fn(|k| tables[k][ids[k]].p as u32);
    let prods: [u64; K] = std::array::from_fn(|k| ws[k] as u64 * ps[k] as u64);
    let sum = prods.iter().sum::<u64>();
    let slots = Slots {
        ids,
        ps,
        prods: [prods[0], prods[1]],
        sum,
    };
    // Weights sum to 65536 and each p is in 1..=65535, so this needs no clamp.
    ((sum >> 16) as u32, slots)
}
#[inline(always)]
fn update(tables: &mut Tables, ws: &mut [u32; K], s: Slots, bit: bool) {
    // The likelihood of a 0 is w * (65536 - p), and the weights sum to 65536.
    let pick = |one, zero| std::hint::select_unpredictable(bit, one, zero);
    let l = |k: usize| pick(s.prods[k], ((ws[k] as u64) << 16) - s.prods[k]);
    let scale = 65536.0 / pick(s.sum, (1 << 32) - s.sum) as i64 as f64;
    let w0 = (share(l(0), scale) * 31 + 21845) >> 5;
    let w1 = (share(l(1), scale) * 31 + 21845) >> 5;
    *ws = [w0, w1, 65536 - w0 - w1];
    for k in 0..K {
        let c = &mut tables[k][s.ids[k]];
        let p = s.ps[k] as i32;
        // A rate of at most 1/2 keeps p below 65536; only a step to 0 needs the floor.
        c.p = (p + ((bit as i32 * 65536 - p) * RATES[c.n as usize] >> 16)).max(1) as u16;
        c.n += (c.n < 63) as u8;
    }
}

#[cfg(test)]
fn encode(vals: &[i64], min: i64, bits: u8, tb: u8, limit: usize) -> Option<Vec<u8>> {
    let mut enc = Encoder::new(min, bits, tb);
    enc.push(vals, limit).then_some(())?;
    enc.finish(limit)
}

/// An [`encode`] that can stop after a prefix and go on, so a trial on a prefix is not redone.
pub(super) struct Encoder {
    model: Model,
    min: i64,
    bits: u8,
    lo: u32,
    hi: u32,
    prev: u32,
    prev2: u32,
    out: Vec<u8>,
}

impl Encoder {
    pub(super) fn new(min: i64, bits: u8, tb: u8) -> Self {
        assert!((1..=16).contains(&bits) && (8..=14).contains(&tb));
        Self {
            model: Model::new(tb),
            min,
            bits,
            lo: 0,
            hi: u32::MAX,
            prev: 0,
            prev2: 0,
            out: Vec::new(),
        }
    }

    /// Whether the bytes so far and the final flush fit in `limit`. A prefix that fits was
    /// also written within it, so it may go on under that limit.
    pub(super) fn fits(&self, limit: usize) -> bool {
        self.out.len() + 4 <= limit
    }

    /// Codes `vals`; false once a byte would pass `limit`, after which the encoder is spent.
    pub(super) fn push(&mut self, vals: &[i64], limit: usize) -> bool {
        let (bits, min) = (self.bits, self.min);
        let (mut lo, mut hi, mut prev, mut prev2) = (self.lo, self.hi, self.prev, self.prev2);
        let ((tables, weights, m), out) = (self.model.parts(), &mut self.out);
        assert!(
            vals.iter()
                .all(|&v| (v.wrapping_sub(min) as u32 as u64) < (1u64 << bits))
        );
        for &value in vals {
            let x = value.wrapping_sub(min) as u32;
            let seeds = seeds(prev, prev2);
            let mut prefix = 1u32;
            for (depth, ws) in weights[..bits as usize].iter_mut().enumerate() {
                let (p, slots) = predict(tables, ws, m, prefix, seeds);
                let bit = x >> (bits as usize - 1 - depth) & 1;
                // The model never reads the coder, so updating first leaves only `p` live
                // across the byte output.
                update(tables, ws, slots, bit != 0);
                prefix = prefix * 2 + bit;
                let mid = lo + (((hi - lo) as u64 * p as u64) >> 16) as u32;
                if bit != 0 {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
                while (lo ^ hi) < 1 << 24 {
                    if out.len().saturating_add(5) > limit {
                        return false;
                    }
                    out.push((hi >> 24) as u8);
                    lo <<= 8;
                    hi = (hi << 8) | 255;
                }
            }
            prev2 = prev;
            prev = x;
        }
        (self.lo, self.hi, self.prev, self.prev2) = (lo, hi, prev, prev2);
        true
    }

    pub(super) fn finish(mut self, limit: usize) -> Option<Vec<u8>> {
        if !self.fits(limit) {
            return None;
        }
        for _ in 0..4 {
            self.out.push((self.lo >> 24) as u8);
            self.lo <<= 8;
        }
        Some(self.out)
    }
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
    let (tables, weights, m) = model.parts();
    let (mut lo, mut hi) = (0u32, u32::MAX);
    let (mut prev, mut prev2) = (0u32, 0u32);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let seeds = seeds(prev, prev2);
        let mut prefix = 1u32;
        for ws in &mut weights[..bits as usize] {
            let (p, slots) = predict(tables, ws, m, prefix, seeds);
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
            update(tables, ws, slots, bit);
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
    fn share_matches_integer_division() {
        let check = |l: u64, t: u64| {
            let scale = 65536.0 / t as i64 as f64;
            assert_eq!(share(l, scale) as u64, (l << 16) / t, "{l} / {t}");
        };
        for t in 1..300 {
            (0..=t).for_each(|l| check(l, t));
        }
        let mut rng = 99;
        for _ in 0..50_000 {
            let t = (next(&mut rng) >> 32).max(1) | [0, 0xffff_0000][rng as usize & 1];
            // Quotients at and next to an integer are where rounding could bite.
            let l = (next(&mut rng) % 65537 * t) >> 16;
            for l in [l.saturating_sub(1), l, l + 1, next(&mut rng) % (t + 1), t] {
                check(l.min(t), t);
            }
        }
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
