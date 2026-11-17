//! Kind 10: bounded integer lanes coded a nibble at a time from a mixture of two adaptive
//! 16-symbol models, with a multi-symbol range coder: one coding step per nibble. Its
//! predictions are the wire contract.
use crate::wire::{DecodeBudget, Error, Result};

/// The models' cumulative total. The coder adds one per symbol below `s`, so every symbol
/// keeps a nonzero frequency within `2^TOT_BITS`.
const MTOT: u16 = (1 << TOT_BITS) - 16;
const TOT_BITS: u32 = 15;

/// One context: lane 0 counts its visits (to 127), lanes 1–15 hold the cumulative
/// frequency below symbols 1–15 out of [`MTOT`].
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Node([u16; 16]);

static RATES: [u16; 128] = {
    let mut r = [0; 128];
    let mut n = 0;
    while n < 128 {
        let v = 131072 / (n as u32 * 2 + 4);
        r[n] = if v > 32767 { 32767 } else { v as u16 };
        n += 1;
    }
    r
};

#[inline(always)]
fn mulhi(a: u16, b: u16) -> u16 {
    ((a as u32 * b as u32) >> 16) as u16
}

/// The mixed cumulative frequency below `i` (0–16) of nodes `a` and `b` at weights `w`
/// and `65536 - w`.
#[inline(always)]
fn cum(a: &Node, b: &Node, w: u16, i: usize) -> u32 {
    if i == 0 {
        return 0;
    }
    if i == 16 {
        return 1 << TOT_BITS;
    }
    let mixed = mulhi(a.0[i], w).wrapping_add(mulhi(b.0[i], w.wrapping_neg()));
    mixed as u32 + i as u32
}

/// The model's frequency of `s`, plus one.
#[inline(always)]
fn freq1(n: &Node, s: usize) -> u64 {
    let lo = if s == 0 { 0 } else { n.0[s] as u64 };
    let hi = if s == 15 {
        MTOT as u64
    } else {
        n.0[(s + 1) & 15] as u64
    };
    hi - lo + 1
}

/// What the SIMD build accelerates; [`Scalar`] defines the results.
trait Ops {
    /// The symbol `s` with `cum(s) <= q < cum(s + 1)`, and those two bounds.
    fn find(a: &Node, b: &Node, w: u16, q: u32) -> (usize, u32, u32);
    /// Moves `n` toward having seen `s` and counts the visit.
    fn update(n: &mut Node, s: usize);
}

struct Scalar;

impl Ops for Scalar {
    #[inline(always)]
    fn find(a: &Node, b: &Node, w: u16, q: u32) -> (usize, u32, u32) {
        let s = (1..16).filter(|&i| cum(a, b, w, i) <= q).count();
        (s, cum(a, b, w, s), cum(a, b, w, s + 1))
    }
    #[inline(always)]
    fn update(n: &mut Node, s: usize) {
        let c = n.0[0];
        let r = RATES[c as usize & 127] as i32;
        for i in 1..16 {
            let t = if i > s { MTOT as i32 } else { 0 };
            let v = n.0[i] as i32;
            n.0[i] = (v + (((t - v) * r) >> 16)) as u16;
        }
        n.0[0] = c + (c < 127) as u16;
    }
}

#[cfg(target_arch = "x86_64")]
struct Avx2;

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{MTOT, Node, RATES};
    use core::arch::x86_64::*;

    static RAMP: Node = Node([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    static NOT0: Node = Node([
        0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0,
    ]);

    #[inline]
    #[target_feature(enable = "avx2")]
    fn ld(n: &Node) -> __m256i {
        // SAFETY: a Node is 32 aligned bytes.
        unsafe { _mm256_load_si256(n.0.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2,popcnt")]
    pub(super) fn find(a: &Node, b: &Node, w: u16, q: u32) -> (usize, u32, u32) {
        let mix = _mm256_add_epi16(
            _mm256_mulhi_epu16(ld(a), _mm256_set1_epi16(w as i16)),
            _mm256_mulhi_epu16(ld(b), _mm256_set1_epi16(w.wrapping_neg() as i16)),
        );
        let cum = _mm256_and_si256(_mm256_add_epi16(mix, ld(&RAMP)), ld(&NOT0));
        // Lanes are below 2^15, so a signed compare ranks them.
        let gt = _mm256_cmpgt_epi16(cum, _mm256_set1_epi16(q as i16));
        let s = 15 - (_mm256_movemask_epi8(gt) as u32).count_ones() as usize / 2;
        // Dwords s/2 and (s+1)/2 hold lanes s and s + 1; a lane-16 read wraps to 0.
        let ix = _mm256_set_epi32(0, 0, 0, 0, 0, 0, (s as i32 + 1) >> 1, s as i32 >> 1);
        let v = _mm_cvtsi128_si64(_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(cum, ix)));
        let sh = 16 * (s as u32 & 1);
        let lo = (v as u64 >> sh) as u32 & 0xffff;
        let hi = (v as u64 >> (sh + 16)) as u32 & 0xffff;
        (
            s,
            lo,
            std::hint::select_unpredictable(s == 15, 1 << super::TOT_BITS, hi),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn update(n: &mut Node, s: usize) {
        let v = ld(n);
        let c = n.0[0];
        let t = _mm256_and_si256(
            _mm256_cmpgt_epi16(ld(&RAMP), _mm256_set1_epi16(s as i16)),
            _mm256_set1_epi16(MTOT as i16),
        );
        let d = _mm256_and_si256(_mm256_sub_epi16(t, v), ld(&NOT0));
        let r = _mm256_set1_epi16(RATES[c as usize & 127] as i16);
        let v = _mm256_add_epi16(v, _mm256_mulhi_epi16(d, r));
        let v = _mm256_add_epi16(
            v,
            _mm256_zextsi128_si256(_mm_cvtsi32_si128((c < 127) as i32)),
        );
        // SAFETY: a Node is 32 aligned bytes.
        unsafe { _mm256_store_si256(n.0.as_mut_ptr().cast(), v) };
    }
}

#[cfg(target_arch = "x86_64")]
impl Ops for Avx2 {
    #[inline(always)]
    fn find(a: &Node, b: &Node, w: u16, q: u32) -> (usize, u32, u32) {
        // SAFETY: only instantiated under `simd()`.
        unsafe { avx2::find(a, b, w, q) }
    }
    #[inline(always)]
    fn update(n: &mut Node, s: usize) {
        // SAFETY: only instantiated under `simd()`.
        unsafe { avx2::update(n, s) }
    }
}

#[cfg(target_arch = "x86_64")]
fn simd() -> bool {
    static V3: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V3.get_or_init(|| {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("bmi1")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("lzcnt")
            && is_x86_feature_detected!("popcnt")
    })
}

/// How a value splits into nibbles. With `escape`, a first symbol codes values 0–14
/// directly and 15 escapes to `x - 15` in `bits` more bits.
#[derive(Clone, Copy)]
struct Plan {
    escape: bool,
    bits: u32,
    steps: u32,
    first: u32,
}

impl Plan {
    fn new(bits: u8, escape: bool) -> Self {
        let bits = if escape {
            32 - ((1u32 << bits) - 16).leading_zeros()
        } else {
            bits as u32
        };
        let steps = bits.div_ceil(4);
        Self {
            escape,
            bits,
            steps,
            first: bits - 4 * (steps - 1),
        }
    }
}

/// Whether escaping saves coding steps on `vals`, given as offsets from `min`.
pub(super) fn escapes(vals: &[i64], min: i64, bits: u8) -> bool {
    if bits <= 4 {
        return false;
    }
    let (plain, esc) = (Plan::new(bits, false).steps, Plan::new(bits, true).steps);
    let small = vals
        .iter()
        .filter(|&&v| (v.wrapping_sub(min) as u64) < 15)
        .count();
    small + (vals.len() - small) * (1 + esc as usize) < vals.len() * plain as usize
}

/// An order-0 table indexed by the nibble prefix and an order-2 table hashed from the two
/// preceding values. The children of a context share one 16-node bucket.
struct Model {
    nodes: Vec<Node>,
    tsize: usize,
    weights: [u16; 8],
}

impl Model {
    fn new(tb: u8) -> Self {
        let mut init = Node([0; 16]);
        for i in 1..16 {
            init.0[i] = (i as u32 * MTOT as u32 / 16) as u16;
        }
        let tsize = 1 << tb;
        Self {
            nodes: vec![init; 3 * tsize],
            tsize,
            weights: [32768; 8],
        }
    }

    /// The two nodes of the context whose parent prefix is `parent`, shifted by `wprev`
    /// bits, and whose last symbol is `last`.
    #[inline(always)]
    fn ids(&self, parent: u32, wprev: u32, last: usize, h: usize) -> (usize, usize) {
        let m = self.tsize - 1;
        let p = (parent as usize).wrapping_mul(0x9E3779B1);
        let i0 = ((parent << wprev) as usize & m) | last;
        (i0, self.tsize + ((h ^ p) & (2 * m + 1) & !15) + last)
    }

    #[inline(always)]
    fn nodes(&self, (i0, i1): (usize, usize)) -> (&Node, &Node) {
        debug_assert!(i0 < self.tsize && i1 < self.nodes.len());
        // SAFETY: `ids` masks i0 below tsize and i1 below 3 * tsize.
        unsafe { (self.nodes.get_unchecked(i0), self.nodes.get_unchecked(i1)) }
    }

    /// Learns `s` at depth `d`: reweights the models by how well each predicted it, then
    /// updates both nodes.
    #[inline(always)]
    fn learn<O: Ops>(&mut self, ids: (usize, usize), d: usize, s: usize) {
        let (a, b) = self.nodes(ids);
        let w = self.weights[d & 7] as u64;
        let la = w * freq1(a, s);
        let lb = (65536 - w) * freq1(b, s);
        // When a value takes one step, this is the decoder's serial path to the next symbol:
        // an integer division is shorter than a floating one with its two conversions.
        let share = ((la << 16) / (la + lb)) as u32;
        self.weights[d & 7] = ((share * 31 + 32768) >> 5) as u16;
        // SAFETY: as in `nodes`.
        unsafe {
            O::update(self.nodes.get_unchecked_mut(ids.0), s);
            O::update(self.nodes.get_unchecked_mut(ids.1), s);
        }
    }
}

fn hash(prev: u32, prev2: u32) -> usize {
    let mut x = (prev2 as u64) << 48 | (prev as u64) << 16;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    (x ^ (x >> 31)) as usize
}

/// The symbol steps of one value: `f(parent, wprev, last, depth, width)` codes one
/// nibble and returns it; the escape symbol, if any, comes first.
#[inline(always)]
fn value(plan: Plan, mut f: impl FnMut(u32, u32, usize, usize, u32) -> usize) -> u32 {
    let (mut parent, mut wprev, mut last, mut d, mut base) = (0, 0, 1, 0, 0);
    if plan.escape {
        let s = f(0, 0, 1, 0, 4);
        if s < 15 {
            return s as u32;
        }
        (parent, wprev, last, d, base) = (1, 1, 0, 1, 15);
    }
    let (mut width, mut out) = (plan.first, 0);
    for _ in 0..plan.steps {
        let s = f(parent, wprev, last, d, width);
        (parent, wprev, last) = (parent << wprev | last as u32, width, s);
        out = out << width | s as u32;
        width = 4;
        d += 1;
    }
    out + base
}

/// A carry-propagating range coder over `2^TOT_BITS` totals. It drops the leading byte,
/// which is always zero.
pub(super) struct Encoder {
    model: Model,
    plan: Plan,
    min: i64,
    bits: u8,
    low: u64,
    range: u32,
    cache: u8,
    pending: usize,
    started: bool,
    prev: u32,
    prev2: u32,
    out: Vec<u8>,
}

impl Encoder {
    pub(super) fn new(min: i64, bits: u8, tb: u8, escape: bool) -> Self {
        assert!((1..=16).contains(&bits) && (8..=14).contains(&tb) && (!escape || bits > 4));
        Self {
            model: Model::new(tb),
            plan: Plan::new(bits, escape),
            min,
            bits,
            low: 0,
            range: u32::MAX,
            cache: 0,
            pending: 0,
            started: false,
            prev: 0,
            prev2: 0,
            out: Vec::new(),
        }
    }

    /// The stream length if it finished now.
    fn finished_len(&self) -> usize {
        self.out.len() + self.pending + self.started as usize + 1
    }

    /// Whether the bytes so far and the final flush fit in `limit`.
    pub(super) fn fits(&self, limit: usize) -> bool {
        self.finished_len() <= limit
    }

    #[inline(always)]
    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low > 0xFFFF_FFFF {
            let carry = (self.low >> 32) as u8;
            if self.started {
                self.out.push(self.cache.wrapping_add(carry));
            }
            self.started = true;
            for _ in 0..self.pending {
                self.out.push(0xFFu8.wrapping_add(carry));
            }
            self.pending = 0;
            self.cache = (self.low >> 24) as u8;
        } else {
            self.pending += 1;
        }
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    #[inline(always)]
    fn put(&mut self, lo: u32, hi: u32) {
        let r = self.range >> TOT_BITS;
        self.low += (r * lo) as u64;
        self.range = if hi == 1 << TOT_BITS {
            self.range - r * lo
        } else {
            r * (hi - lo)
        };
        while self.range < 1 << 24 {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Codes `vals`; false once the stream would pass `limit`, after which the encoder is
    /// spent. A prefix that returns true also fits `limit`.
    pub(super) fn push(&mut self, vals: &[i64], limit: usize) -> bool {
        assert!(
            vals.iter()
                .all(|&v| (v.wrapping_sub(self.min) as u64) < (1u64 << self.bits))
        );
        #[cfg(target_arch = "x86_64")]
        if simd() {
            // SAFETY: the CPU has the features `push_v3` is compiled for.
            return unsafe { self.push_v3(vals, limit) };
        }
        self.push_with::<Scalar>(vals, limit)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    fn push_v3(&mut self, vals: &[i64], limit: usize) -> bool {
        self.push_with::<Avx2>(vals, limit)
    }

    #[inline(always)]
    fn push_with<O: Ops>(&mut self, vals: &[i64], limit: usize) -> bool {
        let plan = self.plan;
        for &v in vals {
            let x = v.wrapping_sub(self.min) as u32;
            let h = hash(self.prev, self.prev2);
            let y = if plan.escape && x >= 15 { x - 15 } else { x };
            let mut shift = plan.bits;
            // Inlined for the same reason as the decoder's step.
            value(
                plan,
                #[inline(always)]
                |parent, wprev, last, d, width| {
                    let s = if plan.escape && d == 0 {
                        x.min(15) as usize
                    } else {
                        shift -= width;
                        (y >> shift & ((1 << width) - 1)) as usize
                    };
                    let ids = self.model.ids(parent, wprev, last, h);
                    let (a, b) = self.model.nodes(ids);
                    let w = self.model.weights[d & 7];
                    let (lo, hi) = (cum(a, b, w, s), cum(a, b, w, s + 1));
                    self.put(lo, hi);
                    self.model.learn::<O>(ids, d, s);
                    s
                },
            );
            (self.prev2, self.prev) = (self.prev, x);
            if !self.fits(limit) {
                return false;
            }
        }
        true
    }

    pub(super) fn finish(mut self, limit: usize) -> Option<Vec<u8>> {
        if !self.fits(limit) {
            return None;
        }
        let want = self.finished_len();
        // Range is at least 2^24, so it holds a point whose low three bytes are zero; the
        // decoder reads those as padding.
        self.low = (self.low + 0xFF_FFFF) & !0xFF_FFFF;
        self.shift_low();
        self.shift_low();
        debug_assert_eq!(self.out.len(), want);
        Some(self.out)
    }
}

pub(super) fn validate(
    len: usize,
    bits: u8,
    tb: u8,
    escape: bool,
    bytes: usize,
    budget: &DecodeBudget,
) -> Result<()> {
    if len == 0 || !(1..=16).contains(&bits) || !(8..=14).contains(&tb) || (escape && bits <= 4) {
        return Err(Error::Format("invalid nibble mixing parameters".into()));
    }
    // A step codes at most 15 bits and a value takes at most `bits / 4 + 2` steps.
    if bytes == 0
        || bytes
            > len
                .saturating_mul(bits as usize)
                .saturating_mul(2)
                .saturating_add(4)
    {
        return Err(Error::Format("invalid nibble mixing byte count".into()));
    }
    if bytes > budget.max_frame_compressed_bytes {
        return Err(Error::BudgetExceeded("nibble stream bytes"));
    }
    let tables = 3usize << tb << 5;
    budget.check_alloc(tables, "nibble mixing tables")?;
    budget.check_alloc(len.saturating_mul(8), "nibble mixing output")?;
    if tables
        .saturating_add(len.saturating_mul(8))
        .saturating_add(bytes)
        .saturating_add(1024)
        > budget.max_block_decoded_bytes
    {
        return Err(Error::BudgetExceeded("nibble mixing working set"));
    }
    Ok(())
}

pub(super) fn decode(
    bytes: &[u8],
    len: usize,
    min: i64,
    bits: u8,
    tb: u8,
    escape: bool,
    budget: &DecodeBudget,
) -> Result<Vec<i64>> {
    validate(len, bits, tb, escape, bytes.len(), budget)?;
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the CPU has the features `decode_v3` is compiled for.
        return unsafe { decode_v3(bytes, len, min, bits, tb, escape) };
    }
    decode_with::<Scalar>(bytes, len, min, bits, tb, escape)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
fn decode_v3(
    bytes: &[u8],
    len: usize,
    min: i64,
    bits: u8,
    tb: u8,
    escape: bool,
) -> Result<Vec<i64>> {
    decode_with::<Avx2>(bytes, len, min, bits, tb, escape)
}

/// The zero bytes the final flush leaves implicit.
const FLUSH_ZEROS: usize = 3;
/// Bytes past the stream's end that a read may touch: the flush's zeros, two per step of a
/// value that starts there, and the four-byte word.
const PAD: usize = FLUSH_ZEROS + 2 * 5 + 4;

#[inline(always)]
fn decode_with<O: Ops>(
    bytes: &[u8],
    len: usize,
    min: i64,
    bits: u8,
    tb: u8,
    escape: bool,
) -> Result<Vec<i64>> {
    let plan = Plan::new(bits, escape);
    let mut model = Model::new(tb);
    let mut buf = Vec::with_capacity(bytes.len() + PAD);
    buf.extend_from_slice(bytes);
    buf.resize(bytes.len() + PAD, 0);
    let word = |at: usize| u32::from_be_bytes(buf[at..at + 4].try_into().unwrap());
    let (mut code, mut range, mut at) = (word(0), u32::MAX, 4);
    let (mut prev, mut prev2) = (0u32, 0u32);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        if at > bytes.len() + FLUSH_ZEROS {
            return Err(Error::Format("truncated nibble stream".into()));
        }
        let h = hash(prev, prev2);
        // Inlined so the step compiles with the caller's SIMD features; an outlined
        // closure calls `O::find` and `O::update` through memory.
        let x = value(
            plan,
            #[inline(always)]
            |parent, wprev, last, d, _| {
                let ids = model.ids(parent, wprev, last, h);
                let (a, b) = model.nodes(ids);
                let r = range >> TOT_BITS;
                // Normalization keeps range at least 2^24, so r is nonzero.
                let q = (code / r).min(32767);
                let (s, lo, hi) = O::find(a, b, model.weights[d & 7], q);
                code -= r * lo;
                range = if s == 15 {
                    range - r * lo
                } else {
                    r * (hi - lo)
                };
                let shift = range.leading_zeros() & 24;
                code = ((code as u64) << shift | (word(at) as u64) >> (32 - shift)) as u32;
                range <<= shift;
                at += (shift / 8) as usize;
                model.learn::<O>(ids, d, s);
                s
            },
        );
        // Only an escaped value can pass the width, and only in a corrupt stream.
        if x >> bits != 0 {
            return Err(Error::Format("nibble value out of range".into()));
        }
        out.push(min.wrapping_add(x as i64));
        (prev2, prev) = (prev, x);
    }
    if at != bytes.len() + FLUSH_ZEROS {
        return Err(Error::Format("truncated or trailing nibble bytes".into()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::graph::{
        CoreDecoder, GraphBuilder, GraphNode, IntStream, with_context_mixing,
    };

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn encode(
        vals: &[i64],
        min: i64,
        bits: u8,
        tb: u8,
        esc: bool,
        limit: usize,
    ) -> Option<Vec<u8>> {
        let mut enc = Encoder::new(min, bits, tb, esc);
        enc.push(vals, limit).then_some(())?;
        enc.finish(limit)
    }

    /// Offsets that are mostly below 8, a quarter spread across the full width.
    fn skewed(n: usize, min: i64, bits: u8, seed: u64) -> Vec<i64> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                let r = next(&mut rng);
                let x = if r % 4 == 0 { r >> 8 } else { r >> 8 & 7 };
                min.wrapping_add((x & ((1 << bits) - 1)) as i64)
            })
            .collect()
    }

    #[test]
    fn nibble_mixing_roundtrips_and_bounds_untrusted_parameters() {
        let budget = DecodeBudget::default();
        let cases = [
            (1, 0),
            (3, -9),
            (4, 7),
            (5, 0),
            (8, i64::MIN),
            (13, 1),
            (16, i64::MAX - 100),
        ];
        for (bits, min) in cases {
            for esc in [false, bits > 4] {
                let values = skewed(301, min, bits, 123 + bits as u64);
                let wire = encode(&values, min, bits, 8, esc, usize::MAX).unwrap();
                let dec =
                    |w: &[u8], b: &DecodeBudget| decode(w, values.len(), min, bits, 8, esc, b);
                assert_eq!(dec(&wire, &budget).unwrap(), values, "{bits} {esc}");
                assert!(dec(&[], &budget).is_err());
                let mut trailing = wire.clone();
                trailing.push(0);
                assert!(dec(&trailing, &budget).is_err());
                assert!(encode(&values, min, bits, 8, esc, wire.len() - 1).is_none());
                assert_eq!(
                    encode(&values, min, bits, 8, esc, wire.len()).unwrap(),
                    wire
                );
                for (b, t, e) in [
                    (0, 8, false),
                    (17, 8, false),
                    (1, 7, false),
                    (1, 15, false),
                    (4, 8, true),
                ] {
                    assert!(decode(&wire, values.len(), min, b, t, e, &budget).is_err());
                }
                let tiny = DecodeBudget {
                    max_single_alloc_bytes: 1024,
                    ..budget
                };
                assert!(matches!(dec(&wire, &tiny), Err(Error::BudgetExceeded(_))));
            }
        }
    }

    #[test]
    fn nibble_mixing_survives_corrupt_streams() {
        let values = skewed(500, 0, 12, 5);
        let wire = encode(&values, 0, 12, 9, true, usize::MAX).unwrap();
        let mut rng = 77;
        for _ in 0..200 {
            let mut bad = wire.clone();
            let i = next(&mut rng) as usize % bad.len();
            bad[i] ^= 1 << (next(&mut rng) % 8);
            if let Ok(out) = decode(&bad, values.len(), 0, 12, 9, true, &DecodeBudget::default()) {
                assert!(out.iter().all(|&v| (0..1 << 12).contains(&v)));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn nibble_mixing_simd_matches_scalar() {
        if !simd() {
            return;
        }
        for (bits, esc, seed) in [(2, false, 1), (7, true, 2), (11, false, 3), (16, true, 4)] {
            let values = skewed(4000, -5, bits, seed);
            let run = |v3: bool| {
                let mut enc = Encoder::new(-5, bits, 10, esc);
                let ok = if v3 {
                    // SAFETY: `simd()` holds.
                    unsafe { enc.push_v3(&values, usize::MAX) }
                } else {
                    enc.push_with::<Scalar>(&values, usize::MAX)
                };
                assert!(ok);
                enc.finish(usize::MAX).unwrap()
            };
            let wire = run(true);
            assert_eq!(run(false), wire);
            let n = values.len();
            assert_eq!(
                decode_with::<Scalar>(&wire, n, -5, bits, 10, esc).unwrap(),
                values
            );
            // SAFETY: `simd()` holds.
            assert_eq!(
                unsafe { decode_v3(&wire, n, -5, bits, 10, esc) }.unwrap(),
                values
            );
        }
    }

    #[test]
    fn nibble_mixing_wire_fixture() {
        let values = [-3, -2, -3, 4, -1, 4, 4, 0, -3, 4, 1, 2, -2, 4, -3, 3];
        let bytes = encode(&values, -3, 3, 8, false, 100).unwrap();
        assert_eq!(bytes, [8, 146, 231, 164, 30, 189, 251, 175, 139]);
        let budget = DecodeBudget::default();
        assert_eq!(
            decode(&bytes, 16, -3, 3, 8, false, &budget).unwrap(),
            values
        );
        let values = skewed(40, 0, 9, 9);
        let bytes = encode(&values, 0, 9, 8, true, 100).unwrap();
        let want = [
            40, 204, 118, 147, 164, 60, 240, 1, 124, 144, 77, 73, 229, 26, 120, 229, 116, 28, 243,
            106, 19, 7, 114, 84, 79, 183,
        ];
        assert_eq!(bytes, want);
        assert_eq!(decode(&bytes, 40, 0, 9, 8, true, &budget).unwrap(), values);
    }

    #[test]
    fn context_mixing_selects_kind_10_and_payloads_are_self_contained() {
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
        assert!(matches!(mixed, IntStream::NibbleMix { .. }), "{mixed:?}");
        assert_eq!(
            plain,
            IntStream::auto_encode(&values),
            "option leaked out of scope"
        );
        let (mut a, mut b) = (Vec::new(), Vec::new());
        plain.encode(&mut a);
        mixed.encode(&mut b);
        assert_eq!(b[0], 10);
        assert!(b.len() + 32 <= a.len());
        let mut gb = GraphBuilder::new(values.len());
        let root = gb.push(GraphNode::IntStream(mixed));
        let graph = gb.finish(root).encode();
        let budget = DecodeBudget::default();
        assert_eq!(
            CoreDecoder::decode_ints_bytes(&graph, &budget).unwrap(),
            values
        );
        let noisy: Vec<i64> = (0..2048).map(|_| next(&mut rng) as i64).collect();
        assert_eq!(
            with_context_mixing(true, || IntStream::auto_encode(&noisy)),
            IntStream::auto_encode(&noisy)
        );
    }
}
