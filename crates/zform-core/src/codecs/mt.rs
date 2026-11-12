//! Values made from the draws of a Mersenne Twister (MT19937, the generator behind Python's
//! `random`), stored as positions in its output stream.
//!
//! A [`Layout`] makes a record from a few consecutive 32-bit outputs: id bytes cut from them,
//! or a double stepped by them. The stream is named by 624 of its outputs, which fix every
//! later one, or by a small integer seed as `random.Random(seed)` expands it. Draws the leaf
//! does not hold only widen the gap between positions, so values interleaved with other uses
//! of the generator still cost a few bits each.

use super::*;
use crate::wire::put_bytes;
use std::cell::RefCell;

const N: usize = 624;
const M: usize = 397;
/// Leaf tag, after the entropy backends'.
const TAG: u8 = 3;
/// Words of a stream a leaf may address, which bounds what a decoder generates.
const MAX_WORDS: usize = 1 << 22;
/// Seeds tried when a leaf does not hold a stream's first 624 words itself.
const SEEDS: u32 = 256;
/// Words searched past the previous record for the next one.
const REACH: usize = 1024;
/// Words searched for a leaf's first record, and again before the furthest one found.
const RESCAN: usize = 1 << 21;

fn temper(mut y: u32) -> u32 {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c_5680;
    y ^= (y << 15) & 0xefc6_0000;
    y ^ (y >> 18)
}

fn untemper(mut y: u32) -> u32 {
    y ^= y >> 18;
    y ^= (y << 15) & 0xefc6_0000;
    // Each pass settles 7 more low bits, then 11 more high bits.
    let mut t = y;
    for _ in 0..4 {
        t = y ^ ((t << 7) & 0x9d2c_5680);
    }
    t ^ (t >> 11) ^ (t >> 22)
}

/// The outputs of one generator from a fixed starting point, made on demand.
struct Stream {
    state: [u32; N],
    words: Vec<u32>,
    /// Where leaves found in this stream stopped, newest last. One often continues another.
    ends: Vec<usize>,
}

impl Stream {
    /// `random.Random(seed)`: MT19937's `init_by_array` with the seed as a one-word key.
    fn seeded(seed: u32) -> Self {
        let mut s = [0u32; N];
        s[0] = 19_650_218;
        for i in 1..N {
            s[i] = (s[i - 1] ^ (s[i - 1] >> 30))
                .wrapping_mul(1_812_433_253)
                .wrapping_add(i as u32);
        }
        let mut i = 1;
        let mut mix = |s: &mut [u32; N], mult: u32, add: fn(u32, u32, u32) -> u32| {
            s[i] = add(
                s[i] ^ (s[i - 1] ^ (s[i - 1] >> 30)).wrapping_mul(mult),
                seed,
                i as u32,
            );
            i += 1;
            if i == N {
                s[0] = s[N - 1];
                i = 1;
            }
        };
        for _ in 0..N {
            mix(&mut s, 1_664_525, |x, seed, _| x.wrapping_add(seed));
        }
        for _ in 1..N {
            mix(&mut s, 1_566_083_941, |x, _, i| x.wrapping_sub(i));
        }
        s[0] = 1 << 31;
        Self {
            state: s,
            words: Vec::new(),
            ends: Vec::new(),
        }
    }

    /// The stream that begins with these 624 outputs.
    fn after(outputs: &[u32]) -> Self {
        let mut state = [0; N];
        for (s, &o) in state.iter_mut().zip(outputs) {
            *s = untemper(o);
        }
        Self {
            state,
            words: outputs[..N].to_vec(),
            ends: Vec::new(),
        }
    }

    fn grow(&mut self, len: usize) {
        let twist = |hi: u32, lo: u32, far: u32| {
            let y = (hi & (1 << 31)) | (lo & !(1 << 31));
            far ^ (y >> 1) ^ (0x9908_b0df * (y & 1))
        };
        while self.words.len() < len {
            let s = &mut self.state;
            // Split where the indices wrap, so each loop is branch-free.
            for i in 0..N - M {
                s[i] = twist(s[i], s[i + 1], s[i + M]);
            }
            for i in N - M..N - 1 {
                s[i] = twist(s[i], s[i + 1], s[i + M - N]);
            }
            s[N - 1] = twist(s[N - 1], s[0], s[M - 1]);
            self.words.extend(s.iter().map(|&x| temper(x)));
        }
    }
}

/// How a record's bytes are made from consecutive outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Layout {
    /// Each word little-endian in draw order: `randbytes(n)`.
    Le,
    /// One big-endian integer, first word lowest: `getrandbits(8 * n)` printed in hex.
    Be,
    /// The last ten bytes of a UUIDv7 whose `rand_a` is `getrandbits(12)` and whose `rand_b`
    /// is `getrandbits(62)`.
    Uuid7,
    /// A little-endian double that is the row before plus `uniform(a, b)`, which Python
    /// computes as `a + (b - a) * random()`. The first row has no row before and is not held.
    Walk { a: f64, b: f64 },
}

impl Layout {
    /// The steps of a column of doubles read as `uniform(a, b)` draws. The smallest and
    /// largest of n draws sit within a few `(b - a) / n` of the bounds, which are taken to be
    /// the shortest decimals that close, as a script would have written them.
    pub(super) fn walk(recs: &[u8]) -> Option<Self> {
        let rows = recs.chunks_exact(8);
        let v: Vec<f64> = rows
            .map(|r| f64::from_le_bytes(r.try_into().unwrap()))
            .collect();
        let mut steps: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).collect();
        let n = steps.len();
        if n < 64 {
            return None;
        }
        // Rows of other data, a header say, fall outside the middle 98% and a margin.
        let mut nth = |i: usize| *steps.select_nth_unstable_by(i, f64::total_cmp).1;
        let (p1, p50, p99) = (nth(n / 100), nth(n / 2), nth(n - 1 - n / 100));
        // Uniform draws have their median midway.
        if ((p50 - p1) - (p99 - p50)).abs() > (p99 - p1) / 4.0 {
            return None;
        }
        let fence = (p99 - p1) / 16.0;
        let inside = steps
            .iter()
            .filter(|&&d| p1 - fence <= d && d <= p99 + fence);
        let (lo, hi) = inside.fold((p1, p99), |(lo, hi), &d| (lo.min(d), hi.max(d)));
        let slack = 16.0 * (hi - lo) / n as f64;
        let short = |x: f64| {
            let near = |y: &f64| (y - x).abs() <= slack;
            let mut digits = (0..17).map(|d| format!("{x:.d$e}").parse().unwrap_or(x));
            if near(&0.0) {
                0.0
            } else {
                digits.find(near).unwrap_or(x)
            }
        };
        let (a, b) = (short(lo), short(hi));
        (a < b && (b - a).is_finite()).then_some(Self::Walk { a, b })
    }

    fn put(self, out: &mut Vec<u8>) {
        match self {
            Self::Le => out.push(0),
            Self::Be => out.push(1),
            Self::Uuid7 => out.push(2),
            Self::Walk { a, b } => {
                out.push(3);
                out.extend_from_slice(&a.to_le_bytes());
                out.extend_from_slice(&b.to_le_bytes());
            }
        }
    }

    fn get(cur: &mut Cursor<'_>) -> Result<Self> {
        let tag = cur.u8()?;
        let mut f64 =
            || -> Result<f64> { Ok(f64::from_le_bytes(cur.slice(8)?.try_into().unwrap())) };
        Ok(match tag {
            0 => Self::Le,
            1 => Self::Be,
            2 => Self::Uuid7,
            3 => Self::Walk {
                a: f64()?,
                b: f64()?,
            },
            _ => return Err(Error::Format("invalid generator leaf".into())),
        })
    }

    /// Words in a record of `width` bytes.
    fn words(self, width: usize) -> Option<usize> {
        match self {
            Self::Uuid7 => (width == 10).then_some(3),
            Self::Walk { .. } => (width == 8).then_some(2),
            _ => (width.is_multiple_of(4) && (4..=64).contains(&width)).then_some(width / 4),
        }
    }

    /// Writes the record that `w` makes after the record `prev`.
    fn render(self, w: &[u32], prev: &[u8], out: &mut [u8]) {
        match self {
            Self::Le => {
                for (o, x) in out.chunks_exact_mut(4).zip(w) {
                    o.copy_from_slice(&x.to_le_bytes());
                }
            }
            Self::Be => {
                for (o, x) in out.chunks_exact_mut(4).zip(w.iter().rev()) {
                    o.copy_from_slice(&x.to_be_bytes());
                }
            }
            Self::Uuid7 => {
                let a = w[0] >> 20;
                let b = 2 << 62 | u64::from(w[2] >> 2) << 32 | u64::from(w[1]);
                out[..2].copy_from_slice(&(0x7000 | a as u16).to_be_bytes());
                out[2..].copy_from_slice(&b.to_be_bytes());
            }
            Self::Walk { a, b } => out.copy_from_slice(&walk(a, b, w, prev)),
        }
    }

    /// The `i`th word of a record that holds its words whole.
    fn word(self, rec: &[u8], i: usize) -> u32 {
        let at = |o: usize| rec[o..o + 4].try_into().unwrap();
        match self {
            Self::Be => u32::from_be_bytes(at(rec.len() - 4 * (i + 1))),
            _ => u32::from_le_bytes(at(4 * i)),
        }
    }

    /// A word a record shows whole, and its index among the record's words.
    fn probe(self, rec: &[u8]) -> Option<(usize, u32)> {
        match self {
            Self::Le | Self::Be => Some((0, self.word(rec, 0))),
            Self::Uuid7 => Some((1, u32::from_be_bytes(rec[6..10].try_into().unwrap()))),
            Self::Walk { .. } => None,
        }
    }

    /// Whether `w` makes `rec` after `prev`.
    fn fits(self, w: &[u32], prev: &[u8], rec: &[u8]) -> bool {
        // A walk shows no word to look for, so every position tried gets this far.
        if let Self::Walk { a, b } = self {
            return rec.try_into().ok() == Some(walk(a, b, w, prev));
        }
        self.probe(rec).is_none_or(|(j, x)| w[j] == x) && {
            let mut buf = [0u8; 64];
            self.render(w, prev, &mut buf[..rec.len()]);
            buf[..rec.len()] == *rec
        }
    }
}

/// The double a `uniform(a, b)` step drawn from `w` makes after the double `prev`.
#[inline]
fn walk(a: f64, b: f64, w: &[u32], prev: &[u8]) -> [u8; 8] {
    // `random()`: 27 bits of one word above 26 of the next, over 2^53.
    let r = u64::from(w[0] >> 5) << 26 | u64::from(w[1] >> 6);
    let step = a + (b - a) * (r as f64 / (1u64 << 53) as f64);
    // A sum with NaN has bits that vary by platform, so no step follows one.
    let prev = f64::from_le_bytes(prev.try_into().unwrap());
    let prev = if prev.is_nan() { 0.0 } else { prev };
    (prev + step).to_le_bytes()
}

/// The record before row `row` of `recs`, or zeros before the first.
fn before(recs: &[u8], row: usize, width: usize) -> &[u8] {
    match row.checked_sub(1) {
        Some(r) => &recs[r * width..][..width],
        None => &[0; 64][..width],
    }
}

/// Where each record starts in `stream`, in row order, and the rows it does not hold.
fn locate(
    stream: &mut Stream,
    recs: &[u8],
    width: usize,
    layout: Layout,
) -> Option<(Vec<i64>, Vec<u32>)> {
    let k = layout.words(width)?;
    let n = recs.len() / width;
    let (mut at, mut missing) = (Vec::with_capacity(n), Vec::new());
    let mut next = 0; // one past the furthest record found
    let mut rescan = RESCAN;
    let ends = stream.ends.clone();
    for (row, rec) in recs.chunks_exact(width).enumerate() {
        // Other data almost never passes, so 32 records settle it.
        if row == 32 && missing.len() > 16 {
            return None;
        }
        if row == 0 && matches!(layout, Layout::Walk { .. }) {
            missing.push(0);
            continue;
        }
        let prev = before(recs, row, width);
        let mut find = |from: usize, to: usize| {
            (from..to.min(MAX_WORDS - k)).find(|&p| {
                stream.grow(p + k);
                layout.fits(&stream.words[p..p + k], prev, rec)
            })
        };
        let mut reach = REACH;
        let mut hit = None;
        if at.is_empty() {
            let mut ends = ends.iter().rev();
            hit = ends.find_map(|e| find(e.saturating_sub(REACH), e + REACH));
            // Short at first: a row after one of other data is never found.
            reach = (REACH << missing.len().min(11)).min(rescan);
            rescan -= reach;
        }
        hit = hit.or_else(|| find(next, next + reach));
        // Records out of draw order, as a list of first appearances is.
        if hit.is_none() && (1..=rescan).contains(&next) {
            rescan -= next;
            hit = find(0, next);
        }
        match hit {
            Some(p) => {
                at.push(p as i64);
                next = next.max(p + k);
            }
            None => missing.push(row as u32),
        }
    }
    if missing.len() * 4 > n {
        return None;
    }
    stream.ends.retain(|&e| e != next);
    stream.ends.push(next);
    let old = stream.ends.len().saturating_sub(8);
    stream.ends.drain(..old);
    Some((at, missing))
}

/// Seeds found while packing one stream, and how many more leaves may search for one.
pub struct Seeds {
    found: Vec<(u32, Stream)>,
    searches: u8,
}

impl Default for Seeds {
    fn default() -> Self {
        Self {
            found: Vec::new(),
            searches: 2,
        }
    }
}

thread_local! {
    static SHARED: RefCell<Option<Seeds>> = const { RefCell::new(None) };
}

/// Runs `f` with `seeds` as the seeds its leaves share. Without it leaves search for none, so
/// what one pack writes never depends on another.
pub(crate) fn with_seeds<R>(seeds: &mut Seeds, f: impl FnOnce() -> R) -> R {
    let outer = SHARED.replace(Some(std::mem::take(seeds)));
    let r = f();
    *seeds = SHARED.replace(outer).unwrap_or_default();
    r
}

/// The first small seed whose stream opens with one of the first rows of `recs`.
fn search(recs: &[u8], width: usize, layouts: &[(Layout, usize)]) -> Option<u32> {
    (0..SEEDS).find(|&seed| {
        let mut stream = Stream::seeded(seed);
        stream.grow(2 * N);
        let rows = recs.chunks_exact(width).take(4).enumerate();
        rows.flat_map(|row| layouts.iter().map(move |l| (row, l)))
            .any(|((row, rec), &(layout, k))| {
                let prev = before(recs, row, width);
                let mut starts = stream.words.windows(k);
                starts.any(|w| layout.fits(w, prev, rec))
            })
    })
}

fn body(
    stream: &mut Stream,
    anchor: &[u8],
    recs: &[u8],
    width: usize,
    layout: Layout,
    backend: EntropyBackend,
    level: i32,
) -> Option<Vec<u8>> {
    let (at, missing) = locate(stream, recs, width, layout)?;
    let mut out = Vec::new();
    layout.put(&mut out);
    put_uvarint(&mut out, width as u64);
    out.extend_from_slice(anchor);
    encode_u32_seq(&missing, backend, level, &mut out);
    encode_i64_seq_with(&at, backend, level, 0, &mut out);
    if !missing.is_empty() {
        let rows = missing
            .iter()
            .map(|&r| &recs[r as usize * width..][..width]);
        compress_leaf(
            &mut out,
            &rows.flatten().copied().collect::<Vec<_>>(),
            backend,
            level,
        );
    }
    out.extend_from_slice(&recs[recs.len() - recs.len() % width..]);
    (out.len() * 2 < recs.len()).then_some(out)
}

fn try_encode(
    recs: &[u8],
    width: usize,
    layouts: &[Layout],
    backend: EntropyBackend,
    level: i32,
) -> Option<Vec<u8>> {
    let n = recs.len() / width.max(1);
    let layouts = layouts.iter().filter_map(|&l| Some((l, l.words(width)?)));
    let layouts: Vec<(Layout, usize)> = layouts.collect();
    if n < 8 || layouts.is_empty() {
        return None;
    }
    // Records drawn one after another hold the state themselves.
    for &(layout, k) in &layouts {
        if !matches!(layout, Layout::Le | Layout::Be) || n * k <= N + 8 {
            continue;
        }
        let rows = recs.chunks_exact(width);
        let mut flat = rows.flat_map(|rec| (0..k).map(move |i| layout.word(rec, i)));
        let head: Vec<u32> = flat.by_ref().take(N).collect();
        let mut stream = Stream::after(&head);
        stream.grow(N + 1);
        if Some(stream.words[N]) == flat.next() {
            let mut anchor = vec![1];
            anchor.extend(head.iter().flat_map(|w| w.to_le_bytes()));
            return body(&mut stream, &anchor, recs, width, layout, backend, level);
        }
    }
    SHARED.with_borrow_mut(|seeds| {
        let seeds = seeds.as_mut()?;
        let encode = |seed: u32, stream: &mut Stream| {
            let mut anchor = vec![0];
            put_uvarint(&mut anchor, u64::from(seed));
            let mut fit = |&(layout, _)| body(stream, &anchor, recs, width, layout, backend, level);
            layouts.iter().find_map(&mut fit)
        };
        if let Some(out) = seeds
            .found
            .iter_mut()
            .find_map(|(seed, s)| encode(*seed, s))
        {
            return Some(out);
        }
        if seeds.searches == 0 {
            return None;
        }
        let found = search(recs, width, &layouts).and_then(|seed| {
            let mut stream = Stream::seeded(seed);
            let out = encode(seed, &mut stream)?;
            seeds.found.push((seed, stream));
            Some(out)
        });
        seeds.searches -= u8::from(found.is_none());
        found
    })
}

/// Writes rows of `width` bytes as stream positions if a layout fits, and says whether one did.
pub(super) fn try_leaf(
    out: &mut Vec<u8>,
    recs: &[u8],
    width: usize,
    layouts: &[Layout],
    backend: EntropyBackend,
    level: i32,
) -> bool {
    let body = try_encode(recs, width, layouts, backend, level);
    if let Some(body) = &body {
        out.push(TAG);
        put_uvarint(out, recs.len() as u64);
        put_bytes(out, body);
    }
    body.is_some()
}

/// Writes rows of `width` bytes as [`try_leaf`] does, else as a plain leaf.
pub(super) fn encode_leaf(
    out: &mut Vec<u8>,
    recs: &[u8],
    width: usize,
    layouts: &[Layout],
    backend: EntropyBackend,
    level: i32,
) {
    if !try_leaf(out, recs, width, layouts, backend, level) {
        compress_leaf(out, recs, backend, level);
    }
}

/// Reads a leaf written by [`encode_leaf`].
pub(super) fn decode_leaf(cur: &mut Cursor<'_>, budget: &DecodeBudget) -> Result<Vec<u8>> {
    if cur.buf.get(cur.pos) != Some(&TAG) {
        return decompress_leaf(cur, budget);
    }
    cur.pos += 1;
    let bad = || Error::Format("invalid generator leaf".into());
    let len = cur.uvar()? as usize;
    let mut cur = Cursor::new(cur.bytes()?);
    let layout = Layout::get(&mut cur)?;
    let width = cur.uvar()? as usize;
    let k = layout.words(width).ok_or_else(bad)?;
    let mut stream = match cur.u8()? {
        0 => Stream::seeded(u32::try_from(cur.uvar()?).map_err(|_| bad())?),
        1 => {
            let words = cur.slice(4 * N)?.chunks_exact(4);
            Stream::after(
                &words
                    .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )
        }
        _ => return Err(bad()),
    };
    let missing = decode_u32_seq(&mut cur, budget)?;
    let at = decode_i64_seq_at(&mut cur, budget, 2)?;
    let spare = match missing.is_empty() {
        true => Vec::new(),
        false => decompress_leaf(&mut cur, budget)?,
    };
    let tail = cur.slice(cur.remaining())?;
    let far = at.iter().copied().max().unwrap_or(0);
    if (at.len() + missing.len()).checked_mul(width) != len.checked_sub(tail.len())
        || tail.len() >= width
        || spare.len() != missing.len() * width
        || at.iter().any(|&p| p < 0)
        || far as usize > MAX_WORDS - k
    {
        return Err(bad());
    }
    budget.check_alloc(len + 4 * (far as usize + k + N), "generator leaf")?;
    stream.grow(far as usize + k);
    let mut out = vec![0; len];
    let (mut at, mut spare) = (at.into_iter(), spare.chunks_exact(width));
    let mut missing = missing.into_iter().peekable();
    for row in 0..(len - tail.len()) / width {
        let (done, rest) = out.split_at_mut(row * width);
        let rec = &mut rest[..width];
        match missing.next_if_eq(&(row as u32)) {
            Some(_) => rec.copy_from_slice(spare.next().unwrap()),
            // Rows out of order leave too few positions for the rows that remain.
            None => {
                let p = at.next().ok_or_else(bad)? as usize;
                layout.render(&stream.words[p..p + k], before(done, row, width), rec);
            }
        }
    }
    out[len - tail.len()..].copy_from_slice(tail);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(seed: u32, range: std::ops::Range<usize>) -> Vec<u32> {
        let mut s = Stream::seeded(seed);
        s.grow(range.end);
        s.words[range].to_vec()
    }

    /// Rows of `layout` cut from `w` at the starts `at`.
    fn rows(w: &[u32], layout: Layout, width: usize, at: impl Iterator<Item = usize>) -> Vec<u8> {
        let k = layout.words(width).unwrap();
        let mut out = Vec::new();
        for p in at {
            out.resize(out.len() + width, 0);
            let end = out.len();
            let (done, rec) = out.split_at_mut(end - width);
            layout.render(&w[p..p + k], before(done, done.len() / width, width), rec);
        }
        out
    }

    /// Encodes, checks the roundtrip, and returns the leaf.
    fn leaf(recs: &[u8], width: usize, layouts: &[Layout]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_leaf(&mut out, recs, width, layouts, EntropyBackend::Zstd, 3);
        let mut cur = Cursor::new(&out);
        assert_eq!(
            decode_leaf(&mut cur, &DecodeBudget::default()).unwrap(),
            recs
        );
        assert_eq!(cur.remaining(), 0);
        out
    }

    const BOTH: [Layout; 2] = [Layout::Le, Layout::Be];

    #[test]
    fn streams_and_layouts_match_python() {
        // `random.Random(seed).getrandbits(32)`, three times.
        assert_eq!(words(0, 0..3), [3626764237, 1654615998, 3255389356]);
        assert_eq!(words(42, 0..3), [2746317213, 478163327, 107420369]);
        assert_eq!(words(255, 0..3), [237503877, 1393327952, 2750896139]);
        assert_eq!(words(42, 1300..1301), [791835934]);

        // Any 624 outputs carry the stream on.
        let w = words(42, 100..2000);
        let mut s = Stream::after(&w);
        s.grow(w.len());
        assert_eq!(s.words[..w.len()], w);
        assert!((0..32).all(|b| untemper(temper(1 << b)) == 1 << b));

        // `Random(7)`: `randbytes(8)`, `'%016x' % getrandbits(64)`, and `getrandbits(12)`
        // then `getrandbits(62)` under a UUIDv7's version and variant bits.
        let w = words(7, 0..3);
        let hex = |layout, width| {
            let bytes = rows(&w, layout, width, 0..1);
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        assert_eq!(hex(Layout::Le, 8), "38b4e652e44da7f2");
        assert_eq!(hex(Layout::Be, 8), "f2a74de452e6b438");
        assert_eq!(hex(Layout::Uuid7, 10), "752e89a7834df2a74de4");
    }

    #[test]
    fn consecutive_draws_carry_their_own_state() {
        // A seed no search would find, and no shared seeds at all.
        let w = words(123_456, 5..4005);
        for layout in BOTH {
            let mut recs = rows(&w, layout, 16, (0..1000).map(|i| 4 * i));
            recs.extend([1, 2, 3]);
            let out = leaf(&recs, 16, &[Layout::Be, Layout::Le]);
            assert_eq!(out[0], TAG);
            assert!(out.len() < 4 * N + 100, "{}", out.len());
        }
        // Too short to check a word past the state, and bytes no generator made.
        assert_ne!(
            leaf(
                &rows(&w, Layout::Le, 16, (0..156).map(|i| 4 * i)),
                16,
                &BOTH
            )[0],
            TAG
        );
        let noise: Vec<u8> = (0..16_000u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        assert_ne!(leaf(&noise, 16, &BOTH)[0], TAG);
    }

    #[test]
    fn seeded_draws_cost_their_gaps() {
        let w = words(42, 0..60_000);
        // Ids with 9 to 15 other draws between them, and one row in 50 from elsewhere.
        let mut p = 1100;
        let starts = std::iter::from_fn(|| {
            p += 11 + (w[p] % 7) as usize;
            (p + 3 < w.len()).then_some(p)
        });
        let starts: Vec<usize> = starts.collect();
        let mut spans = rows(&w, Layout::Le, 8, starts.iter().copied());
        for rec in spans.chunks_exact_mut(8).step_by(50) {
            rec.fill(0xaa);
        }
        // Outside a pack no seed is searched for.
        assert_ne!(leaf(&spans, 8, &BOTH)[0], TAG);

        let mut seeds = Seeds::default();
        let out = with_seeds(&mut seeds, || leaf(&spans, 8, &BOTH));
        assert_eq!(out[0], TAG);
        assert!(
            out.len() * 6 < spans.len(),
            "{} of {}",
            out.len(),
            spans.len()
        );
        assert_eq!((seeds.found[0].0, seeds.searches), (42, 2));

        // A list of first appearances is in no draw order.
        let firsts = rows(
            &w,
            Layout::Be,
            16,
            (0..256).map(|i| 40 + 4 * (i * 97 % 256)),
        );
        let out = with_seeds(&mut seeds, || leaf(&firsts, 16, &BOTH));
        assert!(out[0] == TAG && out.len() < 600, "{}", out.len());

        let tails = rows(&w, Layout::Uuid7, 10, starts.iter().map(|p| p + 4));
        let out = with_seeds(&mut seeds, || leaf(&tails, 10, &[Layout::Uuid7]));
        assert!(
            out[0] == TAG && out.len() * 6 < tails.len(),
            "{}",
            out.len()
        );

        // Each failed search spends one of the pack's two.
        let other = rows(
            &words(100_000, 0..4000),
            Layout::Le,
            8,
            (0..300).map(|i| 13 * i),
        );
        let mut seeds = Seeds::default();
        for left in [1, 0, 0] {
            assert_ne!(with_seeds(&mut seeds, || leaf(&other, 8, &BOTH))[0], TAG);
            assert_eq!(seeds.searches, left);
        }
    }

    #[test]
    fn packs_reach_generators_through_columns() {
        use crate::container::{PackOptions, pack_stream, unpack_stream};
        let pack = |raw: &[u8]| {
            let opts = PackOptions {
                index: false,
                ..Default::default()
            };
            let mut out = Vec::new();
            pack_stream(raw, &mut out, &opts).unwrap();
            let mut back = Vec::new();
            unpack_stream(out.as_slice(), &mut back, &DecodeBudget::default()).unwrap();
            assert!(back == raw);
            out.len()
        };

        // Base64 lines of bytes drawn back to back, which no template splits.
        let w = words(123_456, 0..24_000);
        let mut text = Vec::new();
        for line in rows(&w, Layout::Le, 48, (0..2000).map(|i| 12 * i)).chunks(48) {
            crate::lex::format_base64(line, &mut text);
            text.push(b'\n');
        }
        let size = pack(&text);
        assert!(size < 4 * N + 400, "{size}");

        // JSON whose span ids are `Random(42)` draws, with 16 trace ids drawn before them.
        let w = words(42, 0..30_000);
        let hex = |layout, width, p| {
            let bytes = rows(&w, layout, width, p..p + 1);
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        let mut json = String::new();
        for i in 0..2000 {
            let p = 200 + 13 * i;
            let trace = hex(Layout::Be, 16, 40 + 4 * (w[p] as usize % 16));
            let span = hex(Layout::Le, 8, p + 1);
            json += &format!("{{\"seq\":{i},\"traceparent\":\"00-{trace}-{span}-01\"}}\n");
        }
        let size = pack(json.as_bytes());
        assert!(size < 3000, "{size}");

        // 24-byte records after a header: two doubles that walk by `Random(7)` draws, a tag,
        // a counter, and a float from a set of 64.
        let w = words(7, 0..24_002);
        let mut recs = b"A_HEADER_OF_24_BYTES____".to_vec();
        let (mut ra, mut dec) = (0.01f64.to_le_bytes(), (-1.2f64).to_le_bytes());
        for i in 0..3000 {
            ra = walk(0.0, 0.00003, &w[8 * i..], &ra);
            dec = walk(-0.0001, 0.00012, &w[8 * i + 2..], &dec);
            recs.extend(ra.iter().chain(&dec));
            recs.extend([b'G', b'0' + (i % 8) as u8]);
            recs.extend((400 + (i % 650) as u16).to_le_bytes());
            recs.extend((f32::from((w[8 * i + 4] >> 26) as u8) * 1e-8).to_le_bytes());
        }
        let size = pack(&recs);
        assert!(size < 4000, "{size}");
    }

    #[test]
    fn walks_of_uniform_steps_cost_no_rows() {
        // `v += rng.uniform(a, b)` on `random.Random(7)`, six more words drawn per row.
        let w = words(7, 0..33_000);
        let walk = |start: f64, layout: Layout, first: usize, n: usize| {
            let mut out = start.to_le_bytes().to_vec();
            for p in (first..).step_by(8).take(n) {
                let at = out.len();
                out.extend([0; 8]);
                let (done, rec) = out.split_at_mut(at);
                layout.render(&w[p..p + 2], &done[at - 8..], rec);
            }
            out
        };
        let hex = |b: &[u8]| b.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let (ra, dec) = (
            Layout::Walk { a: 0.0, b: 0.00003 },
            Layout::Walk {
                a: -0.0001,
                b: 0.00012,
            },
        );
        let cols = [walk(0.01, ra, 0, 4000), walk(-1.2, dec, 2, 4000)];
        // The first three values Python reaches from 0.01 and from -1.2.
        assert_eq!(
            hex(&cols[0][8..32]),
            "fac5f333f97f843fa189eaf26688843f708625edfd88843f"
        );
        assert_eq!(
            hex(&cols[1][8..32]),
            "87163a427933f3bf9d89b8c18d33f3bf4c3af9939233f3bf"
        );

        // The bounds come back from the steps, whatever a header row holds.
        let mut seeds = Seeds::default();
        for (mut col, layout) in cols.into_iter().zip([ra, dec]) {
            col[..8].copy_from_slice(b"SAO_STAR");
            assert_eq!(Layout::walk(&col), Some(layout));
            assert_ne!(leaf(&col, 8, &[layout])[0], TAG, "no seed outside a pack");
            let out = with_seeds(&mut seeds, || leaf(&col, 8, &[layout]));
            assert!(out[0] == TAG && out.len() < 120, "{}", out.len());
        }
        // A later leaf starts where an earlier one stopped.
        assert_eq!(seeds.found[0].0, 7);
        assert_eq!(seeds.found[0].1.ends, [31_994, 31_996]);
        let more = walk(1.0, ra, 32_000, 100);
        let out = with_seeds(&mut seeds, || leaf(&more, 8, &[ra]));
        assert!(out[0] == TAG && out.len() < 80, "{}", out.len());

        // Steps bunched at one end are no uniform draws, and NaN starts a walk over.
        let skewed: Vec<u8> = (0..500u32)
            .flat_map(|i| f64::from(i * i * i).to_le_bytes())
            .collect();
        assert_eq!(Layout::walk(&skewed), None);
        let (mut after_nan, mut after_zero) = ([0; 8], [0; 8]);
        ra.render(&w[..2], &f64::NAN.to_le_bytes(), &mut after_nan);
        ra.render(&w[..2], &[0; 8], &mut after_zero);
        assert_eq!(after_nan, after_zero);
    }

    #[test]
    fn rejects_malformed_leaves() {
        let build = |layout: &[u8],
                     width: u64,
                     anchor: &[u8],
                     missing: &[u32],
                     at: &[i64],
                     rest: &[u8],
                     len: u64| {
            let mut body = layout.to_vec();
            put_uvarint(&mut body, width);
            body.extend_from_slice(anchor);
            encode_u32_seq(missing, EntropyBackend::Raw, 3, &mut body);
            encode_i64_seq_with(at, EntropyBackend::Raw, 3, 0, &mut body);
            body.extend_from_slice(rest);
            let mut out = vec![TAG];
            put_uvarint(&mut out, len);
            put_bytes(&mut out, &body);
            decode_leaf(&mut Cursor::new(&out), &DecodeBudget::default())
        };
        let seed = [0, 42];
        // One spare row as a plain leaf: tag, raw length, stored length, bytes.
        let spare = [0, 4, 4, 9, 9, 9, 9];
        let w = words(42, 0..8);
        assert_eq!(
            build(
                &[0],
                4,
                &seed,
                &[1],
                &[7, 0],
                &[&spare[..], &[5, 6]].concat(),
                14
            )
            .unwrap(),
            [
                &w[7].to_le_bytes()[..],
                &[9, 9, 9, 9],
                &w[0].to_le_bytes(),
                &[5, 6]
            ]
            .concat()
        );

        assert!(build(&[4], 4, &seed, &[], &[0], &[], 4).is_err(), "layout");
        let walk = [&[3][..], &[0; 16]].concat();
        assert!(build(&walk, 8, &seed, &[], &[0], &[], 8).is_ok());
        assert!(
            build(&walk, 4, &seed, &[], &[0], &[], 4).is_err(),
            "walk width"
        );
        assert!(
            build(&walk[..9], 8, &seed, &[], &[0], &[], 8).is_err(),
            "bounds"
        );
        assert!(build(&[0], 6, &seed, &[], &[0], &[], 6).is_err(), "width");
        assert!(
            build(&[2], 8, &seed, &[], &[0], &[], 8).is_err(),
            "UUIDv7 width"
        );
        assert!(
            build(&[0], 4, &[2], &[], &[0], &[], 4).is_err(),
            "anchor kind"
        );
        assert!(
            build(&[0], 4, &[1, 0, 0], &[], &[0], &[], 4).is_err(),
            "short state"
        );
        assert!(
            build(&[0], 4, &seed, &[], &[-1], &[], 4).is_err(),
            "negative position"
        );
        assert!(
            build(&[0], 4, &seed, &[], &[MAX_WORDS as i64], &[], 4).is_err(),
            "far position"
        );
        assert!(
            build(&[0], 4, &seed, &[], &[0], &[], 8).is_err(),
            "row count"
        );
        assert!(
            build(&[0], 4, &seed, &[], &[0], &[1, 2, 3, 4], 8).is_err(),
            "tail of a whole row"
        );
        assert!(
            build(
                &[0],
                4,
                &seed,
                &[1, 0],
                &[],
                &[0, 8, 8, 1, 1, 1, 1, 2, 2, 2, 2],
                8
            )
            .is_err(),
            "row order"
        );
        assert!(
            build(&[0], 4, &seed, &[5], &[0], &spare, 8).is_err(),
            "row out of range"
        );
        assert!(
            build(&[0], 4, &seed, &[0], &[0], &[0, 2, 2, 9, 9], 8).is_err(),
            "spare length"
        );
        assert!(
            build(&[0], 4, &seed, &[0], &[0], &[], 8).is_err(),
            "no spare leaf"
        );
        // A spare leaf may not be a generator leaf itself.
        assert!(
            build(&[0], 4, &seed, &[0], &[0], &[TAG, 4, 0], 8).is_err(),
            "nesting"
        );
    }
}
