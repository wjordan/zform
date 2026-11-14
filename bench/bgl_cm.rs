//! Standalone port of the experimental row codec in bgl_cm.cpp.
//! Build: rustc --edition=2024 -C opt-level=3 bench/bgl_cm.rs -o /tmp/bgl_cm_rust
//! Use with bgl_cm.py --bin /tmp/bgl_cm_rust, or the same pack/unpack array CLI.
//! This keeps the reference model and stream; the Python frontend is shared.
//! Optional pack-external-clock/unpack-external-clock use ticks.i64 as a decoded side input.
//! Set BGL_CM_SCALAR=1 to disable the AVX2 weight-update path for comparisons.

use std::{
    collections::HashMap,
    fs,
    hash::{BuildHasherDefault, Hasher},
    path::Path,
};

const NONE: usize = u32::MAX as usize;

trait Key {
    fn key(self) -> u64;
}
macro_rules! keys { ($($t:ty),*) => { $(impl Key for $t { fn key(self) -> u64 { self as u64 } })* }; }
keys!(u64, u32, usize, i64, i32, bool);

#[inline]
fn h(a: impl Key, b: impl Key) -> u64 {
    let mut x = a
        .key()
        .wrapping_add(0x9E3779B97F4A7C15)
        .wrapping_mul(0xBF58476D1CE4E5B9)
        ^ b.key().wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 29;
    (x.wrapping_mul(0xD6E8FEB86659FD93) >> 32) as u32 as u64
}

#[inline]
fn nbits(x: u64) -> usize {
    (64 - x.leading_zeros()) as usize
}
#[inline]
fn width(n: usize) -> usize {
    if n > 1 { nbits((n - 1) as u64) } else { 0 }
}
#[inline]
fn q(x: u64, m: usize) -> u64 {
    let b = nbits(x);
    if b <= m + 1 {
        x
    } else {
        ((b as u64) << m) | ((x >> (b - 1 - m)) & ((1 << m) - 1))
    }
}
fn index<T: PartialEq>(v: &[T], x: T) -> usize {
    v.iter().position(|a| *a == x).unwrap_or(v.len())
}

// Mix the integer keys' high bits into the flat hash tables' bucket indices.
#[derive(Default)]
struct ModelHasher(u64);
impl Hasher for ModelHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write_u64(&mut self, x: u64) {
        let mut x = x ^ (x >> 30);
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        self.0 = x ^ (x >> 31);
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("model keys must be u64")
    }
}
type Map<V> = HashMap<u64, V, BuildHasherDefault<ModelHasher>>;

#[link(name = "m")]
unsafe extern "C" {
    fn logf(x: f32) -> f32;
    fn expf(x: f32) -> f32;
    fn lrintf(x: f32) -> std::ffi::c_long;
}

struct Coder<const ENC: bool> {
    enc: bool,
    x1: u32,
    x2: u32,
    x: u32,
    buf: Vec<u8>,
    pos: usize,
    costs: Vec<f64>,
}
impl<const ENC: bool> Coder<ENC> {
    fn new(buf: Vec<u8>) -> Self {
        let mut ac = Self {
            enc: ENC,
            x1: 0,
            x2: u32::MAX,
            x: 0,
            buf,
            pos: 0,
            costs: (0..65536).map(|i| -(i as f64 / 65536.0).log2()).collect(),
        };
        if !ENC {
            for _ in 0..4 {
                ac.x = (ac.x << 8) | ac.next() as u32;
            }
        }
        ac
    }
    fn next(&mut self) -> u8 {
        if let Some(&b) = self.buf.get(self.pos) {
            self.pos += 1;
            b
        } else {
            0
        }
    }
    fn flush(&mut self) {
        for _ in 0..4 {
            self.buf.push((self.x1 >> 24) as u8);
            self.x1 <<= 8;
        }
    }
    #[inline(always)]
    fn bit(&mut self, mut b: bool, p: u32, cost: &mut f64) -> bool {
        let xm = self.x1 + (((self.x2 - self.x1) as u64 * p as u64) >> 16) as u32;
        if !ENC {
            b = self.x <= xm;
        }
        if b {
            self.x2 = xm;
        } else {
            self.x1 = xm + 1;
        }
        *cost += self.costs[if b { p } else { 65536 - p } as usize];
        while (self.x1 ^ self.x2) < (1 << 24) {
            if ENC {
                self.buf.push((self.x2 >> 24) as u8);
            } else {
                self.x = (self.x << 8) | self.next() as u32;
            }
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
        b
    }
    fn raw(&mut self, v: u64, n: usize, cost: &mut f64) -> u64 {
        let mut r = 0;
        for j in (0..n).rev() {
            r = (r << 1) | self.bit((v >> j) & 1 != 0, 32768, cost) as u64;
        }
        r
    }
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct Counter {
    p: f32,
    n: u8,
}

#[inline(always)]
fn update_weights(ww: &mut [f32], st: &[f32], err: f32, avx2: bool) {
    debug_assert!(st.len() >= ww.len());
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // CM::new checks CPU support; both slices cover every active weight.
        unsafe {
            update_weights_avx2(ww, st, err);
        }
        return;
    }
    for i in 0..ww.len() {
        ww[i] += err * st[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_weights_avx2(ww: &mut [f32], st: &[f32], err: f32) {
    use std::arch::x86_64::*;
    let stop = ww.len() / 8 * 8;
    for i in (0..stop).step_by(8) {
        // Full eight-lane chunks only. Separate multiply/add preserves scalar rounding.
        unsafe {
            let w = _mm256_loadu_ps(ww.as_ptr().add(i));
            let s = _mm256_loadu_ps(st.as_ptr().add(i));
            let v = _mm256_add_ps(w, _mm256_mul_ps(_mm256_set1_ps(err), s));
            _mm256_storeu_ps(ww.as_mut_ptr().add(i), v);
        }
    }
    for i in stop..ww.len() {
        ww[i] += err * st[i];
    }
}

struct CM<const K: usize> {
    avx2: bool,
    tb: usize,
    lr: f32,
    tab: [Vec<Counter>; K],
    weights: Vec<f32>,
    rate: [f32; 256],
    cost: f64,
}
impl<const K: usize> CM<K> {
    fn new(tb: usize, nsel: usize, lr: f32) -> Self {
        assert!((1..=22).contains(&tb) && (1..=32).contains(&K));
        let avx2 = {
            #[cfg(target_arch = "x86_64")]
            {
                std::is_x86_feature_detected!("avx2") && std::env::var_os("BGL_CM_SCALAR").is_none()
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };
        Self {
            avx2,
            tb,
            lr,
            tab: std::array::from_fn(|_| vec![Counter { p: 0.5, n: 0 }; 1 << tb]),
            weights: vec![0.3; nsel * (K + 1)],
            rate: std::array::from_fn(|i| 1.0 / (i as f32 + 1.6)),
            cost: 0.0,
        }
    }
    #[inline]
    fn code<const ENC: bool>(
        &mut self,
        ac: &mut Coder<ENC>,
        bit: bool,
        ctx: &[u64; K],
        sel: usize,
    ) -> bool {
        // Each hash is below 2^tb. Tables never resize, and each pointer belongs to a
        // different table, so these slots remain valid and distinct throughout this call.
        let slots: [*mut Counter; K] = std::array::from_fn(|k| {
            let index = (h(ctx[k], k) >> (32 - self.tb)) as usize;
            debug_assert!(index < self.tab[k].len());
            let slot = unsafe { self.tab[k].as_mut_ptr().add(index) };
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::x86_64::_mm_prefetch(slot.cast(), std::arch::x86_64::_MM_HINT_T2);
            }
            slot
        });
        let mut st = [0.0f32; 33];
        for k in 0..K {
            let p = unsafe { (*slots[k]).p }.clamp(0.0005, 0.9995);
            // Match the reference's libm rounding instead of changing the probability model.
            st[k] = unsafe { logf(p / (1.0 - p)) };
        }
        st[K] = 0.3;
        let ww = &mut self.weights[sel * (K + 1)..(sel + 1) * (K + 1)];
        let mut dot = 0.0f32;
        for k in 0..=K {
            dot += ww[k] * st[k];
        }
        let p = (1.0 / (1.0 + unsafe { expf(-dot) })).clamp(0.0003, 0.9997);
        let bit = ac.bit(
            bit,
            unsafe { lrintf(p * 65536.0) }.clamp(1, 65535) as u32,
            &mut self.cost,
        );
        let err = (bit as u8 as f32 - p) * self.lr;
        update_weights(ww, &st, err, self.avx2);
        for k in 0..K {
            let s = unsafe { &mut *slots[k] };
            s.p += (bit as u8 as f32 - s.p) * self.rate[s.n as usize];
            if s.n < 60 {
                s.n += 1;
            }
        }
        bit
    }
}

fn rank<const N: usize>(scores: &[f32; N]) -> [usize; N] {
    let mut order = [0; N];
    for i in 0..N {
        let mut j = i;
        while j != 0 && scores[order[j - 1]] < scores[i] {
            order[j] = order[j - 1];
            j -= 1;
        }
        order[j] = i;
    }
    order
}

pub(crate) struct Dict {
    pub(crate) card: Vec<usize>,
    pub(crate) mid: Vec<usize>,
    pub(crate) rack: Vec<usize>,
    pub(crate) pos: Vec<usize>,
    pub(crate) lanes: Vec<usize>,
}
struct Node {
    mtf: Vec<usize>,
    last: Vec<usize>,
    succ: Vec<usize>,
    sp1: Vec<usize>,
    sp2: Vec<usize>,
    cardsucc: Vec<usize>,
    card_first: usize,
    nseen: usize,
    last_t: Vec<i64>,
    hist: Vec<usize>,
    seen: Vec<bool>,
    succt: Map<usize>,
    by_name: Map<usize>,
    cards_of: Vec<Vec<usize>>,
    nodes_on: Vec<Vec<usize>>,
    prank: usize,
    prank2: usize,
    pm: usize,
    rank: CM<7>,
    within: CM<7>,
    esc: CM<2>,
    rankraw: f64,
    rawc: f64,
}
impl Node {
    fn new(d: &Dict, tb: usize) -> Self {
        let nm = *d.mid.iter().max().unwrap() + 1;
        let np = *d.pos.iter().max().unwrap() + 1;
        let nc = *d.card.iter().max().unwrap() + 1;
        Self {
            mtf: (0..nm).collect(),
            last: vec![NONE; nm],
            succ: vec![NONE; d.card.len()],
            sp1: vec![NONE; np],
            sp2: vec![NONE; np],
            cardsucc: vec![NONE; nc],
            card_first: NONE,
            nseen: 0,
            last_t: vec![0; nm],
            hist: vec![0; nm],
            seen: vec![false; d.card.len()],
            succt: Map::default(),
            by_name: (0..d.card.len())
                .map(|x| (((d.card[x] as u64) << 32) | d.pos[x] as u64, x))
                .collect(),
            cards_of: vec![vec![]; nm],
            nodes_on: vec![vec![]; nc],
            prank: 0,
            prank2: 0,
            pm: NONE,
            rank: CM::new(tb, 16, 0.005),
            within: CM::new(tb, 8, 0.005),
            esc: CM::new(tb, 4, 0.005),
            rankraw: 0.0,
            rawc: 0.0,
        }
    }
    fn named(&self, card: usize, pos: usize) -> usize {
        *self
            .by_name
            .get(&((card as u64) << 32 | pos as u64))
            .unwrap_or(&NONE)
    }
    fn code<const ENC: bool>(
        &mut self,
        ac: &mut Coder<ENC>,
        d: &Dict,
        mut n: usize,
        t: usize,
        pt: usize,
        now: i64,
    ) -> usize {
        if self.pm != NONE {
            self.last_t[self.pm] = now;
        }
        let mut r = 12;
        let rt = index(&self.mtf, d.mid[n]);
        for k in 0..12.min(self.mtf.len()) {
            let cand = self.mtf[k];
            let ln = self.last[cand];
            let c = [
                k as u64,
                h(k, t),
                h(k, h(self.prank, self.prank2)),
                h(k, h(t, pt)),
                h(
                    k,
                    h(
                        q(now.wrapping_sub(self.last_t[cand]) as u64, 2),
                        q(now.wrapping_sub(self.last_t[self.mtf[0]]) as u64, 2),
                    ),
                ),
                h(k, h(self.hist[cand], t)),
                h(k, h(ln != NONE && self.succ[ln] != NONE, self.prank)),
            ];
            if self.rank.code(ac, k == rt, &c, k) {
                r = k;
                break;
            }
        }
        let m = if r < 12 {
            self.mtf[r]
        } else {
            ac.raw(d.mid[n] as u64, width(self.mtf.len()), &mut self.rankraw) as usize
        };
        if r == 12 {
            r = index(&self.mtf, m);
        }
        let ln = self.last[m];
        let s1 = if ln == NONE { NONE } else { self.succ[ln] };
        let s2 = if s1 == NONE { NONE } else { self.succ[s1] };
        let (mut st, mut spa, mut spb, mut sc) = (NONE, NONE, NONE, NONE);
        if ln != NONE {
            st = *self
                .succt
                .get(&((t as u64) << 32 | ln as u64))
                .unwrap_or(&NONE);
            if self.sp1[d.pos[ln]] != NONE {
                spa = self.named(d.card[ln], self.sp1[d.pos[ln]]);
            }
            if self.sp2[d.pos[ln]] != NONE {
                spb = self.named(d.card[ln], self.sp2[d.pos[ln]]);
            }
            if self.cardsucc[d.card[ln]] != NONE && self.card_first != NONE {
                sc = self.named(self.cardsucc[d.card[ln]], self.card_first);
            }
        }
        let cands = [st, s1, spa, ln, s2, spb, sc];
        let mut hit = 7;
        for k in 0..7 {
            if cands[k] == NONE || cands[..k].contains(&cands[k]) {
                continue;
            }
            let c = [
                k as u64,
                h(k, t),
                h(k, self.hist[m]),
                h(k, h(t, self.hist[m])),
                h(k, h(t, pt)),
                h(k, h(r, self.hist[m] & 3)),
                h(
                    k,
                    h(
                        st == s1,
                        h(s1 == ln, q(now.wrapping_sub(self.last_t[m]) as u64, 2)),
                    ),
                ),
            ];
            if self.within.code(ac, cands[k] == n, &c, k) {
                hit = k;
                n = cands[k];
                break;
            }
        }
        if hit == 7 {
            if self.esc.code(ac, !self.seen[n], &[0, t as u64], 0) {
                n = self.nseen;
            } else {
                let same =
                    self.esc
                        .code(ac, ln != NONE && d.card[ln] == d.card[n], &[1, h(t, 0)], 1);
                let cards = &self.cards_of[m];
                let card = if same {
                    d.card[ln]
                } else {
                    cards[ac.raw(
                        index(cards, d.card[n]) as u64,
                        width(cards.len()),
                        &mut self.rawc,
                    ) as usize]
                };
                let nodes = &self.nodes_on[card];
                n = nodes
                    [ac.raw(index(nodes, n) as u64, width(nodes.len()), &mut self.rawc) as usize];
            }
        }
        if !self.seen[n] {
            self.seen[n] = true;
            self.nseen += 1;
            self.nodes_on[d.card[n]].push(n);
            if !self.cards_of[m].contains(&d.card[n]) {
                self.cards_of[m].push(d.card[n]);
            }
        }
        if ln != NONE && ln != n {
            self.succ[ln] = n;
            if d.card[ln] == d.card[n] {
                let a = &mut self.sp1[d.pos[ln]];
                if *a != d.pos[n] {
                    self.sp2[d.pos[ln]] = *a;
                }
                *a = d.pos[n];
            } else {
                self.cardsucc[d.card[ln]] = d.card[n];
                self.card_first = d.pos[n];
            }
        }
        if ln != NONE {
            self.succt.insert((t as u64) << 32 | ln as u64, n);
        }
        self.hist[m] = (self.hist[m] << 3 | hit.min(7)) & 0x1ff;
        self.last[m] = n;
        self.pm = m;
        self.mtf.remove(r);
        self.mtf.insert(0, m);
        self.prank2 = self.prank;
        self.prank = r.min(12);
        n
    }
}

struct Sel {
    flag: CM<7>,
    idx: CM<3>,
    rawc: f64,
    s1: Map<usize>,
    s2: Map<usize>,
    s3: Map<usize>,
    n1: Map<usize>,
    n2: Map<usize>,
    m1: Map<usize>,
    last_node: Vec<usize>,
    last2_node: Vec<usize>,
    last_mid: Vec<usize>,
    score: [f32; 9],
    cache: Vec<usize>,
    hist: usize,
    p1: usize,
    p2: usize,
    p3: usize,
    any: bool,
}
fn prediction<V: Copy>(map: &Map<V>, key: u64, e: usize, values: &mut [V], have: &mut [bool]) {
    if let Some(&v) = map.get(&key) {
        values[e] = v;
        have[e] = true;
    }
}
impl Sel {
    fn new(d: &Dict, tb: usize) -> Self {
        Self {
            flag: CM::new(tb, 10, 0.004),
            idx: CM::new(tb, 8, 0.004),
            rawc: 0.0,
            s1: Map::default(),
            s2: Map::default(),
            s3: Map::default(),
            n1: Map::default(),
            n2: Map::default(),
            m1: Map::default(),
            last_node: vec![NONE; d.card.len()],
            last2_node: vec![NONE; d.card.len()],
            last_mid: vec![NONE; 256],
            score: [0.0; 9],
            cache: Vec::with_capacity(65),
            hist: 0,
            p1: 0,
            p2: 0,
            p3: 0,
            any: false,
        }
    }
    fn code<const ENC: bool>(
        &mut self,
        ac: &mut Coder<ENC>,
        d: &Dict,
        mut t: usize,
        nd: usize,
        pnd: usize,
    ) -> usize {
        let (ln, ln2, lm) = (
            self.last_node[nd],
            self.last2_node[nd],
            self.last_mid[d.mid[nd] & 255],
        );
        let (mut e, mut have) = ([0; 9], [false; 9]);
        let rel = if nd == pnd {
            0
        } else if d.card[nd] == d.card[pnd] {
            1
        } else if d.mid[nd] == d.mid[pnd] {
            2
        } else {
            3
        };
        e[0] = self.p1;
        have[0] = self.any;
        let k2 = (self.p2 as u64) << 32 | self.p1 as u64;
        let k3 = ((self.p3 as u64) << 42) ^ ((self.p2 as u64) << 21) ^ self.p1 as u64;
        prediction(&self.s1, self.p1 as u64, 1, &mut e, &mut have);
        prediction(&self.s2, k2, 2, &mut e, &mut have);
        prediction(&self.s3, k3, 3, &mut e, &mut have);
        if ln != NONE {
            e[4] = ln;
            have[4] = true;
            prediction(&self.n1, ln as u64, 5, &mut e, &mut have);
            prediction(
                &self.n2,
                (ln2 as u64) << 32 | ln as u64,
                6,
                &mut e,
                &mut have,
            );
        }
        if lm != NONE {
            e[7] = lm;
            have[7] = true;
            prediction(
                &self.m1,
                (lm as u64) << 32 | self.p1 as u64,
                8,
                &mut e,
                &mut have,
            );
        }
        let order = rank(&self.score);
        let (mut hit, mut tried) = (10, 0);
        let mut agree = 0usize;
        for j in 1..9 {
            agree |= ((have[j] && e[j] == e[0]) as usize) << j;
        }
        for oi in 0..9 {
            let j = order[oi];
            if !have[j] || order[..oi].iter().any(|&g| have[g] && e[g] == e[j]) {
                continue;
            }
            let c = [
                h(j, tried),
                h(h(j, tried), self.hist & 0xfff),
                h(h(j, tried), e[j]),
                h(h(j, e[j]), self.p1),
                h(h(j, rel), h(self.hist & 0xff, agree)),
                h(h(j, e[j]), h(self.p1, self.p2)),
                h(h(j, e[j]), h(rel, ln)),
            ];
            let ok = self.flag.code(ac, e[j] == t, &c, tried.min(8));
            tried += 1;
            if ok {
                hit = j;
                t = e[j];
                break;
            }
        }
        if hit > 9 {
            let at = index(&self.cache, t);
            let c = [
                1,
                h(1, self.hist & 0xfff),
                h(1, self.p1),
                h(1, h(self.p1, self.p2)),
                h(1, rel),
                h(1, ln),
                h(1, tried),
            ];
            if self.flag.code(ac, at < self.cache.len(), &c, 9) {
                hit = 9;
                let mut pre = 1;
                for b in (0..6).rev() {
                    let c = [pre as u64, h(pre, self.p1), h(pre, self.hist & 0xff)];
                    pre = pre * 2 + self.idx.code(ac, at >> b & 1 != 0, &c, 5 - b) as usize;
                }
                t = self.cache[pre - 64];
            } else {
                t = ac.raw(t as u64, width(d.lanes.len()), &mut self.rawc) as usize;
            }
        }
        for j in 0..9 {
            if have[j] {
                self.score[j] += ((e[j] == t) as u8 as f32 - self.score[j]) * 0.02;
            }
        }
        self.s1.insert(self.p1 as u64, t);
        self.s2.insert(k2, t);
        self.s3.insert(k3, t);
        if ln != NONE {
            self.n1.insert(ln as u64, t);
            self.n2.insert((ln2 as u64) << 32 | ln as u64, t);
        }
        if lm != NONE {
            self.m1.insert((lm as u64) << 32 | self.p1 as u64, t);
        }
        self.last2_node[nd] = ln;
        self.last_node[nd] = t;
        self.last_mid[d.mid[nd] & 255] = t;
        let at = index(&self.cache, t);
        if at < self.cache.len() {
            self.cache.remove(at);
        }
        self.cache.insert(0, t);
        self.cache.truncate(64);
        self.hist = (self.hist << 4 | hit.min(15)) & 0xffff;
        self.p3 = self.p2;
        self.p2 = self.p1;
        self.p1 = t;
        self.any = true;
        t
    }
}

struct Head {
    bits: usize,
    hd: CM<4>,
    rawc: f64,
    last: Map<usize>,
}
impl Head {
    fn new(nheads: usize, tb: usize) -> Self {
        Self {
            bits: width(nheads),
            hd: CM::new(tb, 4, 0.004),
            rawc: 0.0,
            last: Map::default(),
        }
    }
    fn code<const ENC: bool>(
        &mut self,
        ac: &mut Coder<ENC>,
        mut v: usize,
        t: usize,
        nd: usize,
        ph: usize,
    ) -> usize {
        let key = ((nd + 1) as u64) << 32 | t as u64;
        let h0 = *self.last.get(&(t as u64)).unwrap_or(&NONE);
        let h1 = *self.last.get(&key).unwrap_or(&NONE);
        if self.hd.code(
            ac,
            v == h0,
            &[h(t, 0), h(t, h0), h(h0, h1 == h0), h(t, ph)],
            0,
        ) {
            v = h0;
        } else {
            v = if self.hd.code(
                ac,
                h1 != NONE && v == h1,
                &[h(t, 1), h(h1, 1), h(t, h1), 7],
                1,
            ) {
                h1
            } else {
                ac.raw(v as u64, self.bits, &mut self.rawc) as usize
            };
        }
        self.last.insert(t as u64, v);
        self.last.insert(key, v);
        v
    }
}

struct Clock {
    cm: CM<18>,
    rawc: f64,
    last_mid: [i64; 256],
    gap_mid: [i64; 256],
    pd: [u64; 8],
    recent: [usize; 8],
}
impl Clock {
    fn new(tb: usize) -> Self {
        Self {
            cm: CM::new(tb, 64 + 41 * 20, 0.004),
            rawc: 0.0,
            last_mid: [0; 256],
            gap_mid: [0; 256],
            pd: [0; 8],
            recent: [0; 8],
        }
    }
    fn code<const ENC: bool>(
        &mut self,
        ac: &mut Coder<ENC>,
        d: &Dict,
        tick: i64,
        prev: i64,
        first: bool,
        n: usize,
        pn: usize,
        t: usize,
        pt: usize,
    ) -> i64 {
        if first {
            return ac.raw(tick as u64, 64, &mut self.rawc) as i64;
        }
        let dt = tick.wrapping_sub(prev) as u64;
        assert!(!ac.enc || dt >> 63 == 0, "ticks must not decrease");
        let m = d.mid[n];
        let r = if n == pn {
            0
        } else if d.card[n] == d.card[pn] {
            1
        } else if m == d.mid[pn] {
            2
        } else if d.rack[n] == d.rack[pn] {
            3
        } else {
            4
        };
        let sm = prev.wrapping_sub(self.last_mid[m & 255]) as u64;
        let act = (0..8)
            .filter(|&a| !self.recent[..a].contains(&self.recent[a]))
            .count();
        let mn3 = self.pd[0].min(self.pd[1]).min(self.pd[2]);
        let mn5 = mn3.min(self.pd[3]).min(self.pd[4]);
        let mut s5 = [self.pd[0], self.pd[1], self.pd[2], self.pd[3], self.pd[4]];
        s5.sort_unstable();
        let med5 = s5[2];
        let gm = self.gap_mid[m & 255] as u64;
        let es = [
            mn3,
            self.pd[0],
            med5,
            gm.saturating_sub(sm),
            self.pd[1],
            mn5,
        ];
        let length: [u64; 6] = std::array::from_fn(|e| h(q(es[e], 3), 12 + e));
        let base = [
            0,
            q(self.pd[0], 4),
            h(q(self.pd[0], 3), q(self.pd[1], 3)),
            h(t, r),
            h(h(t, pt), q(self.pd[0], 2)),
            h(r, q(self.pd[0], 3)),
            h(act, q(self.pd[0], 3)),
            h(q(sm, 3), act),
            h(q(mn3, 4), q(self.pd[0] - mn3, 2)),
            h(q(gm, 3), h(q(sm, 3), r == 0)),
            h(q(med5, 4), act),
            h(
                h(q(self.pd[0], 2), q(self.pd[1], 2)),
                h(q(self.pd[2], 2), r),
            ),
        ];
        let mut bit = |ac: &mut Coder<ENC>,
                       b: bool,
                       sel: usize,
                       shift: usize,
                       isnb: bool,
                       prefix: u64,
                       value: u64| {
            let mut c = [0; 18];
            for k in 0..12 {
                c[k] = h(base[k], prefix);
            }
            for e in 0..6 {
                if isnb {
                    c[12 + e] = h(length[e], prefix);
                    continue;
                }
                let df = (value as i64)
                    .wrapping_sub((es[e] >> shift) as i64)
                    .clamp(-3, 3);
                let nx = if shift >= 4 {
                    (es[e] >> (shift - 4)) & 15
                } else {
                    (es[e] << (4 - shift)) & 15
                };
                c[12 + e] = h(h(df + 3, nx), h(12 + e, sel));
            }
            self.cm.code(ac, b, &c, sel)
        };
        let (mut prefix, mut nb) = (1u64, 0usize);
        let nbt = nbits(dt);
        for j in (0..6).rev() {
            let b = bit(ac, nbt >> j & 1 != 0, 5 - j, 0, true, prefix, 0) as usize;
            prefix = prefix * 2 + b as u64;
            nb = nb * 2 + b;
        }
        prefix = 64 + nb as u64;
        let mut value = (nb > 0) as u64;
        for (k, j) in (0..nb.saturating_sub(1)).rev().enumerate() {
            if k >= 8 {
                value = value << (j + 1) | ac.raw(dt, j + 1, &mut self.rawc);
                break;
            }
            let b = bit(
                ac,
                dt >> j & 1 != 0,
                64 + nb.min(40) * 20 + k,
                j + 1,
                false,
                prefix,
                value,
            ) as u64;
            prefix = prefix * 2 + b;
            value = value * 2 + b;
        }
        self.pd.copy_within(..7, 1);
        self.pd[0] = value;
        let next = prev.wrapping_add(value as i64);
        self.gap_mid[m & 255] = next.wrapping_sub(self.last_mid[m & 255]);
        self.last_mid[m & 255] = next;
        self.recent.copy_within(..7, 1);
        self.recent[0] = m;
        next
    }
}

#[derive(Clone)]
struct Lane {
    hist: usize,
    res: usize,
    delta: i64,
    prev2: i64,
    err: [f32; 19],
    score: [f32; 19],
    cache: Vec<i64>,
}
impl Default for Lane {
    fn default() -> Self {
        Self {
            hist: 0,
            res: 0,
            delta: 0,
            prev2: 0,
            err: [8.0; 19],
            score: [0.0; 19],
            cache: Vec::with_capacity(65),
        }
    }
}
#[derive(Default)]
struct Template {
    rows: usize,
    pnd: usize,
    lane: Vec<Lane>,
    prev: Vec<i64>,
}
struct Num {
    tpl: Vec<Template>,
    base: Vec<u64>,
    flag: CM<6>,
    res: CM<7>,
    idx: CM<3>,
    last_node: Map<[i64; 2]>,
    pair_delta: Map<i64>,
    last_mid: Map<i64>,
    seq1: Map<i64>,
    seq2: Map<i64>,
    assoc: Map<i64>,
    assoc2: Map<i64>,
    nsucc: Map<i64>,
}
impl Num {
    fn new(d: &Dict, tb: usize) -> Self {
        let mut base = vec![0; d.lanes.len()];
        for s in 1..base.len() {
            base[s] = base[s - 1] + d.lanes[s - 1] as u64;
        }
        Self {
            tpl: (0..d.lanes.len()).map(|_| Template::default()).collect(),
            base,
            flag: CM::new(tb, 20, 0.004),
            res: CM::new(tb, 96, 0.004),
            idx: CM::new(tb, 6, 0.004),
            last_node: Map::default(),
            pair_delta: Map::default(),
            last_mid: Map::default(),
            seq1: Map::default(),
            seq2: Map::default(),
            assoc: Map::default(),
            assoc2: Map::default(),
            nsucc: Map::default(),
        }
    }
    fn code<const ENC: bool>(
        &mut self,
        ac: &mut Coder<ENC>,
        d: &Dict,
        s: usize,
        nd: usize,
        v: &mut [i64],
    ) {
        let p = d.lanes[s];
        if p == 0 {
            return;
        }
        let t = &mut self.tpl[s];
        if t.lane.is_empty() {
            t.lane.resize(p, Lane::default());
            t.prev.resize(p, 0);
        }
        let k = t.rows;
        let pnd = if k != 0 { t.pnd } else { nd };
        let rel = if nd == pnd {
            0
        } else if d.card[nd] == d.card[pnd] {
            1
        } else if d.mid[nd] == d.mid[pnd] {
            2
        } else {
            3
        };
        for j in 0..p {
            let pr = if j != 0 { t.lane[j - 1].res } else { 99 };
            let ln = &mut t.lane[j];
            let lane = self.base[s] + j as u64;
            let mut x = v[j];
            let prev = if k != 0 { t.prev[j] } else { 0 };
            let kn = lane << 32 | nd as u64;
            let kp = (lane << 24 | d.pos[pnd] as u64) << 20 | (d.pos[nd] as u64) << 2 | rel as u64;
            let km = lane << 32 | d.mid[nd] as u64;
            let last = self.last_node.get(&kn).copied();
            let known = last.is_some();
            let (mut es, mut have) = ([0i64; 19], [false; 19]);
            es[0] = prev;
            have[0] = k > 0;
            if let Some(last) = last {
                es[1] = last[0];
                have[1] = true;
                es[2] = last[1];
                have[2] = last[1] != last[0];
            }
            if k != 0 {
                if let Some(&delta) = self.pair_delta.get(&kp) {
                    es[3] = prev.wrapping_add(delta);
                    have[3] = true;
                }
            }
            if k > 1 {
                es[4] = prev.wrapping_add(ln.delta);
                have[4] = true;
            }
            for l in 1..=4 {
                if j >= l {
                    es[4 + l] = v[j - l].wrapping_add(if k != 0 {
                        t.prev[j].wrapping_sub(t.prev[j - l])
                    } else {
                        0
                    });
                    have[4 + l] = true;
                }
            }
            prediction(&self.last_mid, km, 9, &mut es, &mut have);
            for l in 1..=3 {
                if j >= 2 * l {
                    es[9 + l] = v[j - l].wrapping_mul(2).wrapping_sub(v[j - 2 * l]);
                    have[9 + l] = true;
                }
            }
            if j != 0 {
                es[13] = v[j - 1];
                have[13] = true;
            }
            let up = prev as u64;
            let u2 = ln.prev2 as u64;
            let ul = if j != 0 { v[j - 1] as u64 } else { 0 };
            let un = last.map_or(0, |a| a[0] as u64);
            let k1 = h(lane, up) | h(up, 5) << 32;
            let k2 = h(lane ^ u2.wrapping_mul(31), up) | h(u2, up) << 32;
            let ka = h(lane, ul) | h(ul, 9) << 32;
            let ka2 = h(lane ^ up.wrapping_mul(131), ul) | h(ul, up) << 32;
            let kns = h(lane, un) | h(un, 13) << 32;
            if k != 0 {
                prediction(&self.seq1, k1, 14, &mut es, &mut have);
                if k > 1 {
                    prediction(&self.seq2, k2, 15, &mut es, &mut have);
                }
            }
            if j != 0 {
                prediction(&self.assoc, ka, 16, &mut es, &mut have);
                if k != 0 {
                    prediction(&self.assoc2, ka2, 18, &mut es, &mut have);
                }
            }
            if known {
                prediction(&self.nsucc, kns, 17, &mut es, &mut have);
            }
            let order = rank(&ln.score);
            let (mut hit, mut tried) = (20, 0);
            for oi in 0..19 {
                let e = order[oi];
                if !have[e] || order[..oi].iter().any(|&g| have[g] && es[g] == es[e]) {
                    continue;
                }
                let l = h(lane, e);
                let c = [
                    l,
                    h(l, ln.hist & 0xfff),
                    h(l, h(pr, ln.hist & 0xf)),
                    h(l, h(rel, ln.hist & 0xff)),
                    h(l, h(tried, pr)),
                    h(h(e, tried), h(ln.hist & 0xff, p)),
                ];
                let ok = self.flag.code(ac, es[e] == x, &c, tried.min(18));
                tried += 1;
                if ok {
                    hit = e;
                    x = es[e];
                    break;
                }
            }
            if hit > 19 {
                let at = index(&ln.cache, x);
                let l = h(lane, 55);
                let c = [
                    l,
                    h(l, ln.hist & 0xfff),
                    h(l, pr),
                    h(l, rel),
                    h(l, ln.cache.len()),
                    h(p, ln.hist & 0xff),
                ];
                if self.flag.code(ac, at < ln.cache.len(), &c, 19) {
                    hit = 19;
                    let mut pre = 1;
                    for b in (0..6).rev() {
                        let c = [h(l, pre), h(pre, 0), h(h(l, pre), ln.hist & 0xf)];
                        pre = pre * 2 + self.idx.code(ac, at >> b & 1 != 0, &c, 5 - b) as usize;
                    }
                    x = ln.cache[pre - 64];
                }
            }
            if hit > 19 {
                x = Self::value(&mut self.res, ac, x as u64, lane, p, j, pr, &es, &have, ln);
            }
            for e in 0..19 {
                if have[e] {
                    let df = x.wrapping_sub(es[e]).unsigned_abs();
                    ln.err[e] += (nbits(df) as f32 - ln.err[e]) * 0.1;
                    ln.score[e] += ((es[e] == x) as u8 as f32 - ln.score[e]) * 0.05;
                }
            }
            ln.hist = (ln.hist & !0xfff) | ((ln.hist << 4 | hit.min(15)) & 0xfff);
            ln.res = hit;
            if k != 0 {
                ln.delta = x.wrapping_sub(prev);
                self.pair_delta.insert(kp, ln.delta);
            }
            match last {
                None => {
                    self.last_node.insert(kn, [x, x]);
                }
                Some(last) if last[0] != x => {
                    self.last_node.insert(kn, [x, last[0]]);
                }
                _ => {}
            }
            self.last_mid.insert(km, x);
            if k != 0 {
                self.seq1.insert(k1, x);
                if k > 1 {
                    self.seq2.insert(k2, x);
                }
            }
            if j != 0 {
                self.assoc.insert(ka, x);
                if k != 0 {
                    self.assoc2.insert(ka2, x);
                }
            }
            if known {
                self.nsucc.insert(kns, x);
            }
            ln.prev2 = prev;
            let at = index(&ln.cache, x);
            if at < ln.cache.len() {
                ln.cache.remove(at);
            }
            ln.cache.insert(0, x);
            ln.cache.truncate(64);
            if ac.enc {
                assert_eq!(v[j], x, "numeric prediction mismatch");
            }
            v[j] = x;
        }
        t.prev.copy_from_slice(v);
        t.pnd = nd;
        t.rows += 1;
    }
    fn value<const ENC: bool>(
        res: &mut CM<7>,
        ac: &mut Coder<ENC>,
        a: u64,
        lane: u64,
        p: usize,
        j: usize,
        pr: usize,
        es: &[i64; 19],
        have: &[bool; 19],
        ln: &mut Lane,
    ) -> i64 {
        let mut best = None;
        for e in 0..19 {
            if have[e] && best.is_none_or(|b| ln.err[e] < ln.err[b]) {
                best = Some(e);
            }
        }
        let predictions = [
            best.map_or(0, |b| es[b] as u64),
            if have[0] { es[0] as u64 } else { 0 },
            if have[1] { es[1] as u64 } else { 0 },
        ];
        let l = h(lane, 77);
        let pnb = ln.hist >> 12;
        let mut bit = |ac: &mut Coder<ENC>,
                       b: bool,
                       sel: usize,
                       shift: usize,
                       isnb: bool,
                       prefix: u64,
                       value: u64| {
            let lp = h(l, prefix);
            let mut c = [lp, h(lp, pnb), h(lp, pr), h(h(p, j), prefix), 0, 0, 0];
            for g in 0..3 {
                if isnb {
                    c[4 + g] = h(lp, h(g, q(predictions[g], 2)));
                    continue;
                }
                let df = (value as i64)
                    .wrapping_sub((predictions[g] >> shift) as i64)
                    .clamp(-3, 3);
                let nx = if shift >= 3 {
                    (predictions[g] >> (shift - 3)) & 7
                } else {
                    (predictions[g] << (3 - shift)) & 7
                };
                c[4 + g] = h(h(l, g), h(df + 3, h(nx, shift.min(20))));
            }
            res.code(ac, b, &c, sel)
        };
        let nbt = nbits(a);
        let (mut nb, mut prefix) = (0usize, 1u64);
        for b in (0..6).rev() {
            let y = bit(ac, nbt >> b & 1 != 0, 1 + 5 - b, 0, true, prefix, 0) as usize;
            prefix = prefix * 2 + y as u64;
            nb = nb * 2 + y;
        }
        prefix = 256 + nb as u64;
        let mut value = (nb > 0) as u64;
        for (k, b) in (0..nb.saturating_sub(1)).rev().enumerate() {
            if k >= 12 {
                prefix = 100000 + b as u64;
            }
            let y = bit(
                ac,
                a >> b & 1 != 0,
                8 + k.min(15) + 16 * (nb / 8).min(4),
                b + 1,
                false,
                prefix,
                value,
            ) as u64;
            value = value * 2 + y;
            if k < 12 {
                prefix = prefix * 2 + y;
            }
        }
        let x = value as i64;
        let r = best.map_or(x, |b| x.wrapping_sub(es[b]));
        ln.hist = (ln.hist & 0xfff) | (nbits(r.unsigned_abs()).min(63) << 12);
        x
    }
}

pub(crate) struct Rows {
    pub(crate) node: Vec<usize>,
    pub(crate) sel: Vec<usize>,
    pub(crate) head: Vec<usize>,
    pub(crate) ticks: Vec<i64>,
    pub(crate) nums: Vec<i64>,
}
fn put<T: PartialEq + std::fmt::Debug>(enc: bool, slot: &mut T, value: T) {
    if enc {
        assert_eq!(*slot, value, "prediction mismatch");
    }
    *slot = value;
}
fn run<const ENC: bool, const EXTERNAL_CLOCK: bool>(
    ac: &mut Coder<ENC>,
    rows: &mut Rows,
    d: &Dict,
    nheads: usize,
    report: bool,
) {
    let n = rows.node.len();
    let tb = 22.min(nbits(n as u64) + 5);
    let mut node = Node::new(d, tb);
    let mut sel = Sel::new(d, tb);
    let mut head = Head::new(nheads, tb);
    let mut clock = (!EXTERNAL_CLOCK).then(|| Clock::new(tb));
    let mut num = Num::new(d, tb);
    let mut at = 0;
    for i in 0..n {
        let pn = if i != 0 { rows.node[i - 1] } else { 0 };
        let ps = if i != 0 { rows.sel[i - 1] } else { 0 };
        let prev = if i != 0 { rows.ticks[i - 1] } else { 0 };
        let nd = node.code(ac, d, rows.node[i], ps, ps, prev);
        put(ac.enc, &mut rows.node[i], nd);
        let s = sel.code(ac, d, rows.sel[i], nd, if i != 0 { pn } else { nd });
        put(ac.enc, &mut rows.sel[i], s);
        assert!(s < d.lanes.len(), "selector out of range");
        let h = head.code(
            ac,
            rows.head[i],
            s,
            nd,
            if i != 0 { rows.head[i - 1] } else { 0 },
        );
        put(ac.enc, &mut rows.head[i], h);
        if let Some(clock) = &mut clock {
            let tick = clock.code(ac, d, rows.ticks[i], prev, i == 0, nd, pn, s, ps);
            put(ac.enc, &mut rows.ticks[i], tick);
        }
        let p = d.lanes[s];
        if !ac.enc {
            rows.nums.resize(at + p, 0);
        }
        num.code(ac, d, s, nd, &mut rows.nums[at..at + p]);
        at += p;
    }
    assert_eq!(at, rows.nums.len(), "numeric lane length mismatch");
    let parts = [
        ("node.rank", node.rank.cost + node.rankraw),
        ("node.within", node.within.cost),
        ("node.escape", node.esc.cost + node.rawc),
        ("selector.flags", sel.flag.cost),
        ("selector.cache", sel.idx.cost),
        ("selector.raw", sel.rawc),
        ("head", head.hd.cost + head.rawc),
        ("clock.cm", clock.as_ref().map_or(0.0, |c| c.cm.cost)),
        ("clock.raw", clock.as_ref().map_or(0.0, |c| c.rawc)),
        ("nums.flags", num.flag.cost),
        ("nums.cache", num.idx.cost),
        ("nums.value", num.res.cost),
    ];
    if !report {
        return;
    }
    print!("{{\"rows\":{n},\"model_mb\":{{");
    let mut total = 0.0;
    for (name, cost) in parts {
        print!("\"{name}\":{:.6},", cost / 8e6);
        total += cost;
    }
    print!("\"total\":{:.6}}},", total / 8e6);
}

fn load_ids(dir: &Path, name: &str) -> Vec<usize> {
    let data = fs::read(dir.join(name)).expect("cannot read ID column");
    assert_eq!(data.len() % 4, 0, "partial ID");
    data.chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect()
}
fn load_nums(dir: &Path, name: &str) -> Vec<i64> {
    let data = fs::read(dir.join(name)).expect("cannot read numeric column");
    assert_eq!(data.len() % 8, 0, "partial number");
    data.chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn dict(dir: &Path) -> Dict {
    let d = Dict {
        card: load_ids(dir, "nodecard.u32"),
        mid: load_ids(dir, "nodemid.u32"),
        rack: load_ids(dir, "noderack.u32"),
        pos: load_ids(dir, "nodepos.u32"),
        lanes: load_ids(dir, "lanes.u32"),
    };
    assert!(
        !d.card.is_empty() && !d.lanes.is_empty(),
        "empty dictionary"
    );
    assert!(
        d.mid.len() == d.card.len() && d.rack.len() == d.card.len() && d.pos.len() == d.card.len(),
        "dictionary lengths differ"
    );
    d
}
pub(crate) fn encode_rows(
    rows: &mut Rows,
    d: &Dict,
    external_clock: bool,
    report: bool,
) -> Vec<u8> {
    let n = rows.node.len();
    assert!(
        n > 0 && rows.sel.len() == n && rows.head.len() == n && rows.ticks.len() == n,
        "row arrays differ in length"
    );
    assert!(
        rows.node.iter().all(|&x| x < d.card.len()),
        "node out of range"
    );
    let nheads = *rows.head.iter().max().unwrap() + 1;
    let mut ac = Coder::<true>::new(Vec::new());
    if external_clock {
        run::<true, true>(&mut ac, rows, d, nheads, report);
    } else {
        run::<true, false>(&mut ac, rows, d, nheads, report);
    }
    ac.flush();
    let mut out = Vec::with_capacity(16 + ac.buf.len());
    out.extend_from_slice(&((n as u64) | ((external_clock as u64) << 63)).to_le_bytes());
    out.extend_from_slice(&(nheads as u64).to_le_bytes());
    out.extend_from_slice(&ac.buf);
    out
}

pub(crate) fn decode_rows(
    input: &[u8],
    d: &Dict,
    ticks: Option<Vec<i64>>,
    report: bool,
) -> Result<Rows, Box<dyn std::error::Error>> {
    if input.len() < 16 {
        return Err("truncated stream".into());
    }
    let n = u64::from_le_bytes(input[..8].try_into()?);
    let external_clock = n >> 63 != 0;
    if external_clock != ticks.is_some() {
        return Err("external clock streams require unpack-external-clock and ticks.i64".into());
    }
    let n = (n & !(1 << 63)) as usize;
    let nheads = u64::from_le_bytes(input[8..16].try_into()?) as usize;
    let mut rows = Rows {
        node: vec![0; n],
        sel: vec![0; n],
        head: vec![0; n],
        ticks: ticks.unwrap_or_else(|| vec![0; n]),
        nums: Vec::new(),
    };
    if rows.ticks.len() != n {
        return Err("external clock row count mismatch".into());
    }
    let mut ac = Coder::<false>::new(input[16..].to_vec());
    if external_clock {
        run::<false, true>(&mut ac, &mut rows, d, nheads, report);
    } else {
        run::<false, false>(&mut ac, &mut rows, d, nheads, report);
    }
    Ok(rows)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        return Err(
            "usage: bgl_cm_rust pack ARR_DIR OUT.cm | bgl_cm_rust unpack IN.cm ARR_DIR".into(),
        );
    }
    if args[1] == "pack" || args[1] == "pack-external-clock" {
        let external_clock = args[1] == "pack-external-clock";
        let dir = Path::new(&args[2]);
        let d = dict(dir);
        let mut rows = Rows {
            node: load_ids(dir, "node.u32"),
            sel: load_ids(dir, "sel.u32"),
            head: load_ids(dir, "head.u32"),
            ticks: load_nums(dir, "ticks.i64"),
            nums: load_nums(dir, "nums.i64"),
        };
        let out = encode_rows(&mut rows, &d, external_clock, true);
        fs::write(&args[3], &out)?;
        println!("\"stream_bytes\":{}}}", out.len());
    } else if args[1] == "unpack" || args[1] == "unpack-external-clock" {
        let dir = Path::new(&args[3]);
        let d = dict(dir);
        let input = fs::read(&args[2])?;
        let ticks = if args[1] == "unpack-external-clock" {
            Some(load_nums(dir, "ticks.i64"))
        } else {
            None
        };
        let rows = decode_rows(&input, &d, ticks, true)?;
        for (name, v) in [
            ("node.u32", &rows.node),
            ("sel.u32", &rows.sel),
            ("head.u32", &rows.head),
        ] {
            let bytes: Vec<_> = v.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect();
            fs::write(dir.join(name), bytes)?;
        }
        for (name, v) in [("ticks.i64", &rows.ticks), ("nums.i64", &rows.nums)] {
            let bytes: Vec<_> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            fs::write(dir.join(name), bytes)?;
        }
        println!("\"stream_bytes\":{}}}", input.len());
    } else {
        return Err("mode must be pack or unpack".into());
    }
    Ok(())
}
