//! Runtime-pluggable writer-side codec packs (`CodecPack` & `PackRegistry`).
//!
//! Every [`CodecPack`] runs exclusively on the encoder (writer) side: it inspects a column,
//! solves domain-specific parameters (epoch offsets, LCG seeds, MT19937 states, decimal scales,
//! structural templates), and emits a self-describing [`PayloadGraph`] that
//! [`CoreDecoder`](crate::payload::graph::CoreDecoder) executes without knowing anything about
//! the pack that produced it.

use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

use crate::codecs::{ColSummary, int_bounds, time_bounds};
use crate::payload::graph::{
    FastMap, FastSet, GraphBuilder, GraphNode, IntStream, PayloadGraph, fast_map_with_capacity,
    mtf_split, pack_leaf, with_encode_opts,
};
use crate::payload::ir::{Builder, Len, Op, Program, V};
use crate::payload::recur::RecurrenceSpec;
use crate::wire::EntropyBackend;

// ============================================================================
// 1. Writer-Side Pre-Filtering Profile & CodecPack Registry
// ============================================================================

/// Single-pass O(1) structural summary sampled from up to 16 rows of a column.
///
/// Computed once per column in `encode_best_graph` (<100 ns) so every [`CodecPack::precheck`]
/// rejects non-matching columns in ~1 ns without touching the full column slice.
#[derive(Clone, Copy, Debug)]
pub struct ColumnProfile {
    pub n_rows: usize,
    pub min_len: usize,
    pub max_len: usize,
    pub all_ascii: bool,
    pub any_digit_start: bool,
    pub any_decimal_dot: bool,
    pub any_iso_ts: bool,
    pub any_clf_ts: bool,
    pub any_syslog_ts: bool,
    pub any_uuid_dash: bool,
    pub any_ipv4_dot: bool,
    pub any_hex_even_ge16: bool,
    pub any_base32_26: bool,
    pub any_traceparent_55: bool,
    pub any_at: bool,
    pub any_delim: bool,
    pub any_mult4_ge16: bool,
}

impl ColumnProfile {
    pub fn sample(rows: &[&[u8]]) -> Self {
        let n_rows = rows.len();
        if n_rows == 0 {
            return Self {
                n_rows: 0,
                min_len: 0,
                max_len: 0,
                all_ascii: true,
                any_digit_start: false,
                any_decimal_dot: false,
                any_iso_ts: false,
                any_clf_ts: false,
                any_syslog_ts: false,
                any_uuid_dash: false,
                any_ipv4_dot: false,
                any_hex_even_ge16: false,
                any_base32_26: false,
                any_traceparent_55: false,
                any_at: false,
                any_delim: false,
                any_mult4_ge16: false,
            };
        }
        let mut p = Self {
            n_rows,
            min_len: usize::MAX,
            max_len: 0,
            all_ascii: true,
            any_digit_start: false,
            any_decimal_dot: false,
            any_iso_ts: false,
            any_clf_ts: false,
            any_syslog_ts: false,
            any_uuid_dash: false,
            any_ipv4_dot: false,
            any_hex_even_ge16: false,
            any_base32_26: false,
            any_traceparent_55: false,
            any_at: false,
            any_delim: false,
            any_mult4_ge16: false,
        };
        let k = n_rows.min(16);
        let step = if k <= 1 { 1 } else { (n_rows - 1) / (k - 1) }.max(1);
        for s in 0..k {
            let idx = (s * step).min(n_rows - 1);
            let r = rows[idx];
            let len = r.len();
            p.min_len = p.min_len.min(len);
            p.max_len = p.max_len.max(len);
            if r.is_empty() {
                continue;
            }
            let b0 = r[0];
            if b0.is_ascii_digit() || (len >= 2 && b0 == b'-' && r[1].is_ascii_digit()) {
                p.any_digit_start = true;
            }
            if len <= 32 && p.any_digit_start && r.contains(&b'.') {
                p.any_decimal_dot = true;
            }
            if (19..=36).contains(&len)
                && r[4] == b'-'
                && r[7] == b'-'
                && matches!(r[10], b'T' | b't' | b' ')
                && r[13] == b':'
                && r[16] == b':'
            {
                p.any_iso_ts = true;
            }
            if len == 26 && r[2] == b'/' && r[6] == b'/' && r[11] == b':' && r[20] == b' ' {
                p.any_clf_ts = true;
            }
            if len == 15
                && r[0].is_ascii_uppercase()
                && r[3] == b' '
                && r[6] == b' '
                && r[9] == b':'
                && r[12] == b':'
            {
                p.any_syslog_ts = true;
            }
            if len == 36 && r[8] == b'-' && r[13] == b'-' && r[18] == b'-' && r[23] == b'-' {
                p.any_uuid_dash = true;
            }
            if (7..=21).contains(&len) && b0.is_ascii_digit() && r.contains(&b'.') {
                p.any_ipv4_dot = true;
            }
            if matches!(len, 16 | 32 | 40 | 64) && r.iter().all(|b| b.is_ascii_hexdigit()) {
                p.any_hex_even_ge16 = true;
            }
            if len == 26
                && b0 <= b'7'
                && r.iter()
                    .all(|b| b.is_ascii_digit() || b.is_ascii_uppercase())
            {
                p.any_base32_26 = true;
            }
            if len == 55 && r[2] == b'-' && r[35] == b'-' && r[52] == b'-' {
                p.any_traceparent_55 = true;
            }
            if len >= 16 && len.is_multiple_of(4) {
                p.any_mult4_ge16 = true;
            }
            for &b in r {
                if b >= 0x80 {
                    p.all_ascii = false;
                } else if b == b'@' {
                    p.any_at = true;
                } else if matches!(b, b'/' | b':' | b'.') {
                    p.any_delim = true;
                }
            }
        }
        p
    }
}

/// Pluggable writer-side codec pack.
pub trait CodecPack: Send + Sync {
    fn name(&self) -> &'static str;

    /// Fast O(1) structural pre-filter checked before touching the full column slice.
    fn precheck(&self, _profile: &ColumnProfile) -> bool {
        true
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph>;
}

/// Runtime registry of writer-side [`CodecPack`]s.
///
/// Cloning a `PackRegistry` is cheap (`Arc`-backed), and callers may register domain-specific
/// packs at runtime via [`PackRegistry::register`] without modifying [`CoreDecoder`](crate::payload::graph::CoreDecoder).
#[derive(Clone)]
pub struct PackRegistry {
    packs: Vec<Arc<dyn CodecPack>>,
}

impl fmt::Debug for PackRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&'static str> = self.packs.iter().map(|p| p.name()).collect();
        f.debug_struct("PackRegistry")
            .field("packs", &names)
            .finish()
    }
}

impl Default for PackRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register(ExactIntPack);
        reg.register(DecimalScaledPack);
        reg.register(JavaLcgPack);
        reg.register(PythonMtPack);
        reg.register(TimestampIsoPack);
        reg.register(TimestampClfPack);
        reg.register(UuidPack);
        reg.register(UlidPack);
        reg.register(Ipv4Pack);
        reg.register(HexFixedPack);
        reg.register(TraceparentPack);
        reg.register(StructuredSplitPack);
        reg.register(NumericTemplatePack);
        reg.register(Base64Pack);
        reg.register(F64BinaryPack);
        reg.register(BinaryPlanesPack);
        reg
    }
}

impl PackRegistry {
    /// Creates an empty pack registry (falls back to generic dictionary / raw-slice encoding).
    pub fn new() -> Self {
        Self { packs: Vec::new() }
    }

    /// Registers a custom [`CodecPack`] at runtime.
    pub fn register<P: CodecPack + 'static>(&mut self, pack: P) {
        self.packs.push(Arc::new(pack));
    }

    pub fn packs(&self) -> &[Arc<dyn CodecPack>] {
        &self.packs
    }

    /// Evaluates matching packs and generic dictionary/raw fallback on `rows`, returning the
    /// smallest self-describing [`PayloadGraph`].
    pub fn encode_best_graph(&self, rows: &[&[u8]]) -> PayloadGraph {
        self.encode_best_graph_inner(rows, 0).0
    }

    /// Encodes `rows` into serialized [`PayloadGraph`] wire bytes.
    pub fn encode_best(&self, rows: &[&[u8]]) -> Vec<u8> {
        self.encode_best_graph(rows).encode()
    }

    fn encode_best_graph_inner(&self, rows: &[&[u8]], depth: usize) -> (PayloadGraph, bool) {
        if rows.is_empty() {
            return (PayloadGraph::empty(), false);
        }
        let n = rows.len();
        let first = rows[0];
        if rows.iter().all(|&r| r == first) {
            let mut gb = GraphBuilder::new(n);
            let root = gb.add_const_cells(first, n);
            return (gb.finish(root), false);
        }

        // JSON `"..."` / `null` wrapper peeling -> `GraphNode::QuoteOrNull` (decodes into lazy `Col::Quoted`!).
        if depth == 0 && n >= 4 {
            let probe = n.min(32);
            let mut quoted_or_null = 0usize;
            let mut has_quoted = false;
            for &r in &rows[..probe] {
                if r == b"null" {
                    quoted_or_null += 1;
                } else if is_clean_json_quoted(r) {
                    quoted_or_null += 1;
                    has_quoted = true;
                }
            }
            if has_quoted && quoted_or_null == probe {
                let mut all_ok = true;
                let mut any_null = false;
                for &r in rows {
                    if r == b"null" {
                        any_null = true;
                    } else if !is_clean_json_quoted(r) {
                        all_ok = false;
                        break;
                    }
                }
                if all_ok {
                    if !any_null {
                        let inner_rows: Vec<&[u8]> =
                            rows.iter().map(|r| &r[1..r.len() - 1]).collect();
                        let (inner_g, raw_like) = self.encode_best_graph_inner(&inner_rows, 1);
                        let mut gb = GraphBuilder::new(n);
                        let inner_root = gb.merge_graph(&inner_g);
                        let q_root = gb.add_quote_or_null(inner_root, None);
                        return (gb.finish(q_root), raw_like);
                    }
                    let mut non_null: Vec<&[u8]> = Vec::with_capacity(n);
                    let mut row_map: Vec<i64> = Vec::with_capacity(n);
                    for &r in rows {
                        if r == b"null" {
                            row_map.push(-1);
                        } else {
                            let id = non_null.len() as i64;
                            non_null.push(&r[1..r.len() - 1]);
                            row_map.push(id);
                        }
                    }
                    if !non_null.is_empty() {
                        let (inner_g, _) = self.encode_best_graph_inner(&non_null, 1);
                        let mut gb = GraphBuilder::new(n);
                        let inner_root = gb.merge_graph(&inner_g);
                        let map_node = gb.add_ints(&row_map);
                        let q_root = gb.add_quote_or_null(inner_root, Some(map_node));
                        return (gb.finish(q_root), false);
                    }
                }
            }
        }

        // Generic baseline: dictionary or raw-slice `PayloadGraph`.
        let mut base_gb = GraphBuilder::new(n);
        let base_root = base_gb.add_dict_cells(rows);
        let is_raw_fallback = matches!(
            base_gb.nodes[base_root as usize],
            GraphNode::SliceBuf { .. }
        );
        let mut best_graph = base_gb.finish(base_root);
        let mut best_len = best_graph.cost();
        let mut won_by_pack = false;

        // Ultra-low-cardinality categorical columns (`<= 8` unique values across `>= 64` rows,
        // already <1 byte/row via `DictCells`) cannot be meaningfully beaten by structural packs
        // unless cells are fixed-width binary (`<= 16` bytes) where an MT walk might apply.
        let is_tiny_dict = n >= 64
            && !is_raw_fallback
            && best_len <= n + 48
            && matches!(
                &best_graph.nodes[best_graph.root as usize],
                GraphNode::DictCells { dict_ends, .. } if dict_ends.len() <= 8
            );
        let profile = ColumnProfile::sample(rows);
        if is_tiny_dict && !(profile.min_len == profile.max_len && profile.min_len <= 16) {
            return (best_graph, false);
        }

        for pack in &self.packs {
            if !pack.precheck(&profile) {
                continue;
            }
            if let Some(cand) = pack.try_encode(rows) {
                let clen = cand.cost();
                if clen < best_len {
                    best_len = clen;
                    best_graph = cand;
                    won_by_pack = true;
                }
            }
        }
        (best_graph, is_raw_fallback && !won_by_pack)
    }
}

#[inline]
pub(crate) fn is_clean_json_quoted(r: &[u8]) -> bool {
    r.len() >= 2
        && r[0] == b'"'
        && r[r.len() - 1] == b'"'
        && !r[1..r.len() - 1]
            .iter()
            .any(|&b| b == b'"' || b == b'\\' || b < 0x20)
}

#[inline]
pub(crate) fn unquote_json_opt(s: &[u8]) -> (&[u8], bool) {
    if is_clean_json_quoted(s) {
        (&s[1..s.len() - 1], true)
    } else {
        (s, false)
    }
}

// ============================================================================
// 2. Shared Multi-Column MT19937 Seed State & Self-Describing Recurrence Emitter
// ============================================================================

/// The bits of `-0.0`, the identity of float addition.
const NEG_ZERO: i64 = i64::MIN;

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MAX_WORDS: usize = 1 << 22;
const MT_SEEDS: u32 = 256;
const MT_REACH: usize = 1024;
const MT_RESCAN: usize = 1 << 21;

fn days_in_month(y: u32, m: u32) -> u32 {
    match m {
        4 | 6 | 9 | 11 => 30,
        2 if y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400)) => 29,
        2 => 28,
        _ => 31,
    }
}

#[inline]
fn mt_temper(mut y: u32) -> u32 {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c_5680;
    y ^= (y << 15) & 0xefc6_0000;
    y ^ (y >> 18)
}

#[inline]
fn mt_untemper(mut y: u32) -> u32 {
    y ^= y >> 18;
    y ^= (y << 15) & 0xefc6_0000;
    let mut t = y;
    for _ in 0..4 {
        t = y ^ ((t << 7) & 0x9d2c_5680);
    }
    t ^ (t >> 11) ^ (t >> 22)
}

#[inline]
pub fn invert_mt_twist_624(state: &[u32; MT_N]) -> [u32; MT_N] {
    let mut old = [0u32; MT_N];
    for i in (0..MT_N).rev() {
        let next_idx = if i == MT_N - 1 { 0 } else { i + 1 };
        let far_val = if i + MT_M < MT_N {
            old[i + MT_M]
        } else {
            state[i + MT_M - MT_N]
        };
        let z = state[i] ^ far_val;
        let bit0 = (z >> 31) & 1;
        let y_shifted = z ^ if bit0 != 0 { 0x9908_b0df } else { 0 };
        let y = (y_shifted << 1) | bit0;
        if i == MT_N - 1 {
            old[MT_N - 1] = (old[MT_N - 1] & 0x7fff_ffff) | (y & 0x8000_0000);
        } else {
            old[i] = (old[i] & 0x7fff_ffff) | (y & 0x8000_0000);
            old[next_idx] = (old[next_idx] & 0x8000_0000) | (y & 0x7fff_ffff);
        }
    }
    let far0 = old[MT_M - 1];
    let z0 = state[MT_N - 1] ^ far0;
    let b0 = (z0 >> 31) & 1;
    let ys0 = z0 ^ if b0 != 0 { 0x9908_b0df } else { 0 };
    let y0 = (ys0 << 1) | b0;
    old[0] = (old[0] & 0x8000_0000) | (y0 & 0x7fff_ffff);
    old
}

struct MtStream {
    state: [u32; MT_N],
    words: Vec<u32>,
    ends: Vec<usize>,
    /// `(b_new_1, c_new_1)` boundary words from `init_by_array([seed])` when seeded.
    seeded_bounds: Option<(u32, u32)>,
}

impl MtStream {
    fn seeded(seed: u32) -> Self {
        let mut s = [0u32; MT_N];
        s[0] = 19_650_218;
        for i in 1..MT_N {
            s[i] = (s[i - 1] ^ (s[i - 1] >> 30))
                .wrapping_mul(1_812_433_253)
                .wrapping_add(i as u32);
        }
        let mut idx = 1usize;
        for _ in 0..MT_N {
            s[idx] = (s[idx] ^ (s[idx - 1] ^ (s[idx - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(seed);
            idx += 1;
            if idx == MT_N {
                s[0] = s[MT_N - 1];
                idx = 1;
            }
        }
        let b_new_1 = s[1];
        for _ in 1..MT_N {
            s[idx] = (s[idx] ^ (s[idx - 1] ^ (s[idx - 1] >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(idx as u32);
            idx += 1;
            if idx == MT_N {
                s[0] = s[MT_N - 1];
                idx = 1;
            }
        }
        let c_new_1 = s[1];
        s[0] = 1 << 31;
        Self {
            state: s,
            words: Vec::new(),
            ends: Vec::new(),
            seeded_bounds: Some((b_new_1, c_new_1)),
        }
    }

    fn after(outputs: &[u32]) -> Self {
        let mut state = [0u32; MT_N];
        for (s, &o) in state.iter_mut().zip(outputs) {
            *s = mt_untemper(o);
        }
        Self {
            state,
            words: outputs[..MT_N].to_vec(),
            ends: Vec::new(),
            seeded_bounds: None,
        }
    }

    fn grow(&mut self, len: usize) {
        let twist = |hi: u32, lo: u32, far: u32| {
            let y = (hi & (1 << 31)) | (lo & !(1 << 31));
            far ^ (y >> 1) ^ (0x9908_b0dfu32.wrapping_mul(y & 1))
        };
        while self.words.len() < len {
            let s = &mut self.state;
            for i in 0..MT_N - MT_M {
                s[i] = twist(s[i], s[i + 1], s[i + MT_M]);
            }
            for i in MT_N - MT_M..MT_N - 1 {
                s[i] = twist(s[i], s[i + 1], s[i + MT_M - MT_N]);
            }
            s[MT_N - 1] = twist(s[MT_N - 1], s[0], s[MT_M - 1]);
            self.words.extend(s.iter().map(|&x| mt_temper(x)));
        }
    }
}

/// Shared generator-seed discovery cache across blocks of a stream.
pub struct Seeds {
    found: Vec<(u32, MtStream)>,
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
    static SHARED_SEEDS: RefCell<Option<Seeds>> = const { RefCell::new(None) };
}

/// Runs `f` with `seeds` shared across column pack evaluations inside a stream.
pub(crate) fn with_seeds<R>(seeds: &mut Seeds, f: impl FnOnce() -> R) -> R {
    let outer = SHARED_SEEDS.replace(Some(std::mem::take(seeds)));
    let r = f();
    *seeds = SHARED_SEEDS.replace(outer).unwrap_or_default();
    r
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MtLayout {
    Le,
    Be,
    Uuid7,
    Walk { a: f64, b: f64 },
}

#[inline]
fn mt_walk_step(a: f64, b: f64, w: &[u32], prev: &[u8]) -> [u8; 8] {
    let r = u64::from(w[0] >> 5) << 26 | u64::from(w[1] >> 6);
    let step = a + (b - a) * (r as f64 / (1u64 << 53) as f64);
    let prev = f64::from_le_bytes(prev.try_into().unwrap());
    let prev = if prev.is_nan() { 0.0 } else { prev };
    (prev + step).to_le_bytes()
}

impl MtLayout {
    pub(crate) fn detect_walk(recs: &[u8]) -> Option<Self> {
        let rows = recs.chunks_exact(8);
        let v: Vec<f64> = rows
            .map(|r| f64::from_le_bytes(r.try_into().unwrap()))
            .collect();
        let mut steps: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).collect();
        let n = steps.len();
        if n < 64 {
            return None;
        }
        let mut nth = |i: usize| *steps.select_nth_unstable_by(i, f64::total_cmp).1;
        let (p1, p50, p99) = (nth(n / 100), nth(n / 2), nth(n - 1 - n / 100));
        if !p1.is_finite() || !p50.is_finite() || !p99.is_finite() {
            return None;
        }
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

    fn words(self, width: usize) -> Option<usize> {
        match self {
            Self::Uuid7 => (width == 10).then_some(3),
            Self::Walk { .. } => (width == 8).then_some(2),
            _ => (width.is_multiple_of(4) && (4..=64).contains(&width)).then_some(width / 4),
        }
    }

    fn word(self, rec: &[u8], i: usize) -> u32 {
        let at = |o: usize| rec[o..o + 4].try_into().unwrap();
        match self {
            Self::Be => u32::from_be_bytes(at(rec.len() - 4 * (i + 1))),
            _ => u32::from_le_bytes(at(4 * i)),
        }
    }

    fn probe(self, rec: &[u8]) -> Option<(usize, u32)> {
        match self {
            Self::Le | Self::Be => Some((0, self.word(rec, 0))),
            Self::Uuid7 => Some((1, u32::from_be_bytes(rec[6..10].try_into().unwrap()))),
            Self::Walk { .. } => None,
        }
    }

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
                let b = 2u64 << 62 | u64::from(w[2] >> 2) << 32 | u64::from(w[1]);
                out[..2].copy_from_slice(&(0x7000 | a as u16).to_be_bytes());
                out[2..].copy_from_slice(&b.to_be_bytes());
            }
            Self::Walk { a, b } => out.copy_from_slice(&mt_walk_step(a, b, w, prev)),
        }
    }

    fn fits(self, w: &[u32], prev: &[u8], rec: &[u8]) -> bool {
        if let Self::Walk { a, b } = self {
            return rec.try_into().ok() == Some(mt_walk_step(a, b, w, prev));
        }
        self.probe(rec).is_none_or(|(j, x)| w[j] == x) && {
            let mut buf = [0u8; 64];
            self.render(w, prev, &mut buf[..rec.len()]);
            buf[..rec.len()] == *rec
        }
    }
}

fn mt_before(recs: &[u8], row: usize, width: usize) -> &[u8] {
    match row.checked_sub(1) {
        Some(r) => &recs[r * width..][..width],
        None => &[0u8; 64][..width],
    }
}

fn mt_locate(
    stream: &mut MtStream,
    recs: &[u8],
    width: usize,
    layout: MtLayout,
) -> Option<(Vec<i64>, Vec<usize>)> {
    let k = layout.words(width)?;
    let n = recs.len() / width;
    let (mut at, mut missing) = (Vec::with_capacity(n), Vec::new());
    let mut next = 0usize;
    let mut rescan = MT_RESCAN;
    let ends = stream.ends.clone();
    for (row, rec) in recs.chunks_exact(width).enumerate() {
        if row == 32 && missing.len() > 16 {
            return None;
        }
        if row == 0 && matches!(layout, MtLayout::Walk { .. }) {
            missing.push(0);
            continue;
        }
        let prev = mt_before(recs, row, width);
        let mut find = |from: usize, to: usize| {
            (from..to.min(MT_MAX_WORDS - k)).find(|&p| {
                stream.grow(p + k);
                layout.fits(&stream.words[p..p + k], prev, rec)
            })
        };
        let mut reach = MT_REACH;
        let mut hit = None;
        if at.is_empty() {
            let mut ends_rev = ends.iter().rev();
            hit = ends_rev.find_map(|e| find(e.saturating_sub(MT_REACH), e + MT_REACH));
            reach = (MT_REACH << missing.len().min(11)).min(rescan);
            rescan -= reach;
        }
        hit = hit.or_else(|| find(next, next + reach));
        if hit.is_none() && (1..=rescan).contains(&next) {
            rescan -= next;
            hit = find(0, next);
        }
        match hit {
            Some(p) => {
                at.push(p as i64);
                next = next.max(p + k);
            }
            None => missing.push(row),
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

fn mt_search_seed(recs: &[u8], width: usize, layouts: &[(MtLayout, usize)]) -> Option<u32> {
    (0..MT_SEEDS).find(|&seed| {
        let mut stream = MtStream::seeded(seed);
        stream.grow(2 * MT_N);
        let rows = recs.chunks_exact(width).take(4).enumerate();
        rows.flat_map(|row| layouts.iter().map(move |l| (row, l)))
            .any(|((row, rec), &(layout, k))| {
                let prev = mt_before(recs, row, width);
                stream.words.windows(k).any(|w| layout.fits(w, prev, rec))
            })
    })
}

/// Self-describing MT19937 `RecurrenceSpec` (`Width::W32`).
pub fn recur_mt19937(with_temper: bool) -> RecurrenceSpec {
    let b = Builder::new_u32(3);
    let old = b.input(0);
    let next = b.input(1);
    let far = b.input(2);
    let y = b.rr(
        Op::Or,
        b.ri(Op::AndI, old, 0x8000_0000u32 as i32 as i64),
        b.ri(Op::AndI, next, 0x7fff_ffff),
    );
    let mag = b.ri(Op::MulI, b.ri(Op::AndI, y, 1), 0x9908_b0dfu32 as i32 as i64);
    let st = b.rr(Op::Xor, b.rr(Op::Xor, far, b.ri(Op::ShrI, y, 1)), mag);
    let state_prog = b.finish(&[(st, 4)]);

    let out_prog = if with_temper {
        let tb = Builder::new_u32(1);
        let s0 = tb.input(0);
        let s1 = tb.rr(Op::Xor, s0, tb.ri(Op::ShrI, s0, 11));
        let s2 = tb.rr(
            Op::Xor,
            s1,
            tb.ri(
                Op::AndI,
                tb.ri(Op::ShlI, s1, 7),
                0x9d2c_5680u32 as i32 as i64,
            ),
        );
        let s3 = tb.rr(
            Op::Xor,
            s2,
            tb.ri(
                Op::AndI,
                tb.ri(Op::ShlI, s2, 15),
                0xefc6_0000u32 as i32 as i64,
            ),
        );
        let s4 = tb.rr(Op::Xor, s3, tb.ri(Op::ShrI, s3, 18));
        Some(tb.finish(&[(s4, 4)]))
    } else {
        None
    };

    RecurrenceSpec {
        lags: vec![624, 623, 227],
        state_prog,
        out_prog,
    }
}

/// Emits a self-describing `RecurBuf` chain computing Python `random.Random(seed)`'s 624-word
/// `init_by_array` state and twisting `n_words` tempered 32-bit outputs (~190 bytes total on wire!).
fn emit_seeded_mt_words_node(
    gb: &mut GraphBuilder,
    seed: u32,
    b_new_1: u32,
    c_new_1: u32,
    n_words: u32,
) -> u16 {
    let add_iota = |gb: &mut GraphBuilder, start: i64, count: usize| -> u16 {
        let ones = gb.push(GraphNode::IntStream(IntStream::Const {
            val: 1,
            len: count as u32,
        }));
        let b = Builder::new(1);
        let x = b.input(0);
        let idx = b.scan_add(x, start - 1);
        gb.add_prog_i64(b.finish(&[(idx, 8)]), vec![ones], vec![])
    };

    // Phase A: `s[i] = ((s[i-1] ^ (s[i-1] >> 30)) * 1812433253) + i` for `i = 1..624` (623 words).
    let spec_a = {
        let b = Builder::new_u32(2);
        let prev = b.input(0);
        let idx = b.input(1);
        let mixed = b.ri(
            Op::MulI,
            b.rr(Op::Xor, prev, b.ri(Op::ShrI, prev, 30)),
            1_812_433_253i64,
        );
        let cur = b.rr(Op::Add, mixed, idx);
        RecurrenceSpec {
            lags: vec![1],
            state_prog: b.finish(&[(cur, 4)]),
            out_prog: None,
        }
    };
    let iota_a = add_iota(gb, 1, 623);
    let node_a = gb.add_recur_u32(spec_a, &[19_650_218], vec![iota_a], 623, false);

    // Phase B: First 623 steps of Mix 1 (`i = 1..624`).
    let spec_b = {
        let b = Builder::new_u32(2);
        let prev = b.input(0);
        let ai = b.input(1);
        let mixed = b.ri(
            Op::MulI,
            b.rr(Op::Xor, prev, b.ri(Op::ShrI, prev, 30)),
            1_664_525i64,
        );
        let cur = b.ri(Op::AddI, b.rr(Op::Xor, ai, mixed), seed as i32 as i64);
        RecurrenceSpec {
            lags: vec![1],
            state_prog: b.finish(&[(cur, 4)]),
            out_prog: None,
        }
    };
    let col_a = gb.add_byte_chunks(node_a, 4);
    let node_b = gb.add_recur_u32(spec_b, &[19_650_218], vec![col_a], 623, false);

    // Phase C: First 622 steps of Mix 2 (`i = 2..624`), starting from `b_new_1`.
    let spec_c = {
        let b = Builder::new_u32(3);
        let prev = b.input(0);
        let bi = b.input(1);
        let idx = b.input(2);
        let mixed = b.ri(
            Op::MulI,
            b.rr(Op::Xor, prev, b.ri(Op::ShrI, prev, 30)),
            1_566_083_941i64,
        );
        let cur = b.rr(Op::Sub, b.rr(Op::Xor, bi, mixed), idx);
        RecurrenceSpec {
            lags: vec![1],
            state_prog: b.finish(&[(cur, 4)]),
            out_prog: None,
        }
    };
    let b_tail = gb.add_slice_buf(node_b, 4, &[622 * 4]);
    let col_b = gb.add_byte_chunks(b_tail, 4);
    let iota_c = add_iota(gb, 2, 622);
    let node_c = gb.add_recur_u32(spec_c, &[b_new_1], vec![col_b, iota_c], 622, false);

    // Assemble 624-word `init_state` = `[0x8000_0000, c_new_1]` ++ `node_c` (1 row of 2,496 bytes).
    let mut head8 = [0u8; 8];
    head8[..4].copy_from_slice(&0x8000_0000u32.to_le_bytes());
    head8[4..].copy_from_slice(&c_new_1.to_le_bytes());
    let head_cell = gb.add_const_cells(&head8, 1);
    let tail_cell = gb.add_slice_buf(node_c, 0, &[622 * 4]);
    let state_node = gb.add_concat(vec![head_cell, tail_cell]);

    // Twist forward `n_words` tempered outputs!
    gb.add_recur_node(recur_mt19937(true), state_node, vec![], n_words, false)
}

enum MtAnchor {
    Seeded {
        seed: u32,
        b_new_1: u32,
        c_new_1: u32,
    },
    ConsecutiveHead {
        pre_seed: Box<[u32; MT_N]>,
    },
}

/// Attempts to match `recs` (`n` fixed-width binary records of `width` bytes) against either a
/// self-seeded consecutive MT19937 stream or the shared `Seeds` (`random.Random(0..256)`) context,
/// building a `Cells`-producing node inside `gb` if matched and smaller than `recs.len() / 2`.
pub(crate) fn try_build_mt_cells(
    gb: &mut GraphBuilder,
    recs: &[u8],
    width: usize,
    layouts: &[MtLayout],
) -> Option<u16> {
    let n = recs.len().checked_div(width)?;
    if n < 8 || recs.len() != n * width {
        return None;
    }
    let valid_layouts: Vec<(MtLayout, usize)> = layouts
        .iter()
        .filter_map(|&l| Some((l, l.words(width)?)))
        .collect();
    if valid_layouts.is_empty() {
        return None;
    }

    let mut matched: Option<(MtAnchor, MtLayout, usize, Vec<i64>, Vec<usize>)> = None;

    // 1. Consecutive self-seeded draws (`n * k > 624 + 8`).
    for &(layout, k) in &valid_layouts {
        if !matches!(layout, MtLayout::Le | MtLayout::Be) || n * k <= MT_N + 8 {
            continue;
        }
        let mut flat = recs
            .chunks_exact(width)
            .flat_map(|rec| (0..k).map(move |i| layout.word(rec, i)));
        let head: Vec<u32> = flat.by_ref().take(MT_N).collect();
        let mut stream = MtStream::after(&head);
        stream.grow(MT_N + 1);
        if Some(stream.words[MT_N]) == flat.next()
            && let Some((at, missing)) = mt_locate(&mut stream, recs, width, layout)
        {
            let mut untempered = [0u32; MT_N];
            for i in 0..MT_N {
                untempered[i] = mt_untemper(head[i]);
            }
            let pre_seed = invert_mt_twist_624(&untempered);
            matched = Some((
                MtAnchor::ConsecutiveHead {
                    pre_seed: Box::new(pre_seed),
                },
                layout,
                k,
                at,
                missing,
            ));
            break;
        }
    }

    // 2. Shared `Seeds` (`random.Random(0..256)`) discovery when inside `with_seeds`.
    if matched.is_none() {
        matched = SHARED_SEEDS.with_borrow_mut(|seeds| {
            let seeds = seeds.as_mut()?;
            let try_stream = |seed: u32, stream: &mut MtStream| {
                let (b_new_1, c_new_1) = stream.seeded_bounds?;
                for &(layout, k) in &valid_layouts {
                    if let Some((at, missing)) = mt_locate(stream, recs, width, layout) {
                        return Some((
                            MtAnchor::Seeded {
                                seed,
                                b_new_1,
                                c_new_1,
                            },
                            layout,
                            k,
                            at,
                            missing,
                        ));
                    }
                }
                None
            };
            if let Some(res) = seeds
                .found
                .iter_mut()
                .find_map(|(seed, s)| try_stream(*seed, s))
            {
                return Some(res);
            }
            if seeds.searches == 0 {
                return None;
            }
            let found = mt_search_seed(recs, width, &valid_layouts).and_then(|seed| {
                let mut stream = MtStream::seeded(seed);
                let res = try_stream(seed, &mut stream)?;
                seeds.found.push((seed, stream));
                Some(res)
            });
            seeds.searches -= u8::from(found.is_none());
            found
        });
    }

    let (anchor, layout, k, at, missing) = matched?;
    let far = at.iter().copied().max().unwrap_or(0) as usize;
    let total_words = (far + k) as u32;

    let mt_buf = match anchor {
        MtAnchor::Seeded {
            seed,
            b_new_1,
            c_new_1,
        } => emit_seeded_mt_words_node(gb, seed, b_new_1, c_new_1, total_words),
        MtAnchor::ConsecutiveHead { pre_seed } => {
            gb.add_recur_u32(recur_mt19937(true), &*pre_seed, vec![], total_words, false)
        }
    };

    // Fast path: 100% back-to-back `Le` with zero missing rows (`at == [0, k, 2*k, ...]`)!
    if layout == MtLayout::Le
        && missing.is_empty()
        && at.len() == n
        && at.iter().enumerate().all(|(i, &p)| p == (i * k) as i64)
    {
        return Some(gb.add_slice_buf(mt_buf, 0, &vec![width as i64; n]));
    }

    // Build full row-aligned `pos` array (`0` at `missing` rows) and delta-encode it.
    let mut is_missing = vec![false; n];
    for &m in &missing {
        is_missing[m] = true;
    }
    let mut full_pos = Vec::with_capacity(n);
    let mut at_it = at.iter().copied();
    let mut last_p = 0i64;
    for &miss in &is_missing {
        if miss {
            full_pos.push(last_p);
        } else {
            let p = at_it.next()?;
            full_pos.push(p);
            last_p = p;
        }
    }
    let mut pos_deltas = Vec::with_capacity(n);
    for i in 0..n {
        pos_deltas.push(if i == 0 {
            full_pos[0]
        } else {
            full_pos[i] - full_pos[i - 1]
        });
    }
    let pos_node = gb.add_ints(&pos_deltas);

    let base_cells = match layout {
        MtLayout::Le => {
            let b = Builder::new(1);
            let buf_id = b.add_buf();
            let pd = b.input(0);
            let pos = b.scan_add(pd, 0);
            let byte_off = b.ri(Op::ShlI, pos, 2);
            let mut sinks = Vec::with_capacity(width.div_ceil(8));
            let full8 = width / 8;
            for c in 0..full8 {
                let q = b.gather(buf_id, 3, byte_off, (c * 8) as u64);
                sinks.push((q, 8));
            }
            if width % 8 == 4 {
                let rem = b.gather(buf_id, 2, byte_off, (full8 * 8) as u64);
                sinks.push((rem, 4));
            }
            gb.add_prog_cells(b.finish(&sinks), vec![pos_node], vec![mt_buf])
        }
        MtLayout::Be => {
            let b = Builder::new(1);
            let buf_id = b.add_buf();
            let pd = b.input(0);
            let pos = b.scan_add(pd, 0);
            let byte_off = b.ri(Op::ShlI, pos, 2);
            let mut sinks = Vec::with_capacity(width.div_ceil(8));
            let mut rem_words = k;
            while rem_words >= 2 {
                let q = b.gather(buf_id, 3, byte_off, ((rem_words - 2) * 4) as u64);
                sinks.push((b.r(Op::Bswap, q), 8));
                rem_words -= 2;
            }
            if rem_words == 1 {
                let w0 = b.gather(buf_id, 2, byte_off, 0);
                let be = b.ri(Op::ShrI, b.r(Op::Bswap, w0), 32);
                sinks.push((be, 4));
            }
            gb.add_prog_cells(b.finish(&sinks), vec![pos_node], vec![mt_buf])
        }
        MtLayout::Uuid7 => {
            let b = Builder::new(1);
            let buf_id = b.add_buf();
            let pd = b.input(0);
            let pos = b.scan_add(pd, 0);
            let byte_off = b.ri(Op::ShlI, pos, 2);
            let w0 = b.gather(buf_id, 2, byte_off, 0);
            let w1 = b.gather(buf_id, 2, byte_off, 4);
            let w2 = b.gather(buf_id, 2, byte_off, 8);
            let hi16 = b.ri(
                Op::ShrI,
                b.r(Op::Bswap, b.ri(Op::OrI, b.ri(Op::ShrI, w0, 20), 0x7000)),
                48,
            );
            let lo64_val = b.ri(
                Op::OrI,
                b.rr(Op::Or, b.ri(Op::ShlI, b.ri(Op::ShrI, w2, 2), 32), w1),
                (2u64 << 62) as i64,
            );
            let lo64 = b.r(Op::Bswap, lo64_val);
            gb.add_prog_cells(
                b.finish(&[(hi16, 2), (lo64, 8)]),
                vec![pos_node],
                vec![mt_buf],
            )
        }
        MtLayout::Walk { a, b: wb } => {
            // Find the anchor row where the walk starts, and verify exact step accumulation.
            let mut step_flags = vec![0i64; n];
            // -0.0 is the one step that leaves every value, -0.0 included, as it is.
            let mut override_steps = vec![NEG_ZERO; n];
            let mut any_override = false;
            let first_anchor = (0..n).find(|&i| {
                !f64::from_le_bytes(recs[i * 8..i * 8 + 8].try_into().unwrap()).is_nan()
                    && (i + 1 < n && !is_missing[i + 1])
            })?;
            let init_f64 = f64::from_le_bytes(
                recs[first_anchor * 8..first_anchor * 8 + 8]
                    .try_into()
                    .unwrap(),
            );
            let mut acc = init_f64;
            let mut patch_rows: Vec<usize> = (0..first_anchor).collect();

            for i in (first_anchor + 1)..n {
                let target_bytes = &recs[i * 8..i * 8 + 8];
                let target_bits = u64::from_le_bytes(target_bytes.try_into().unwrap());
                let target_f64 = f64::from_bits(target_bits);
                if !is_missing[i] && !acc.is_nan() {
                    step_flags[i] = 1;
                    acc = target_f64;
                } else {
                    patch_rows.push(i);
                    if target_f64.is_nan() {
                        let diff = 0.0 - acc;
                        if (acc + diff).to_bits() == 0 {
                            override_steps[i] = diff.to_bits() as i64;
                            any_override = true;
                            acc = 0.0;
                        } else {
                            return None;
                        }
                    } else {
                        let diff = target_f64 - acc;
                        if (acc + diff).to_bits() == target_bits {
                            override_steps[i] = diff.to_bits() as i64;
                            any_override = true;
                            acc = target_f64;
                        } else {
                            return None;
                        }
                    }
                }
            }

            let flag_node = gb.add_ints(&step_flags);
            let n_in = if any_override { 3 } else { 2 };
            let b = Builder::new(n_in);
            let buf_id = b.add_buf();
            let pd = b.input(0);
            let fl = b.input(1);
            let pos = b.scan_add(pd, 0);
            let byte_off = b.ri(Op::ShlI, pos, 2);
            let w0 = b.gather(buf_id, 2, byte_off, 0);
            let w1 = b.gather(buf_id, 2, byte_off, 4);
            let r = b.rr(
                Op::Or,
                b.ri(Op::ShlI, b.ri(Op::ShrI, w0, 5), 26),
                b.ri(Op::ShrI, w1, 6),
            );
            let unit = b.rf(
                Op::F64MulI,
                b.r(Op::U64ToF64, r),
                1.0 / ((1u64 << 53) as f64),
            );
            let mt_step = b.rf(Op::F64AddI, b.rf(Op::F64MulI, unit, wb - a), a);
            let fallback_step = if any_override {
                b.input(2)
            } else {
                b.konst(pos, NEG_ZERO)
            };
            let step = b.select(fl, mt_step, fallback_step);
            let out = b.scan_f64_add(step, init_f64);
            let inputs = if any_override {
                let ov_node = gb.add_ints(&override_steps);
                vec![pos_node, flag_node, ov_node]
            } else {
                vec![pos_node, flag_node]
            };
            let walk_cells = gb.add_prog_cells(b.finish(&[(out, 8)]), inputs, vec![mt_buf]);
            let patch_vals: Vec<&[u8]> = patch_rows
                .iter()
                .map(|&r| &recs[r * 8..r * 8 + 8])
                .collect();
            return Some(gb.add_patch(walk_cells, &patch_rows, &patch_vals));
        }
    };

    let miss_vals: Vec<&[u8]> = missing
        .iter()
        .map(|&r| &recs[r * width..(r + 1) * width])
        .collect();
    Some(gb.add_patch(base_cells, &missing, &miss_vals))
}

// ============================================================================
// 3. Shared SSA Arithmetic & ASCII Formatting Helpers
// ============================================================================

/// Unsigned `x / d`, exact only while `x * d < 2^64`; wider values take [`div_mod_wide`].
#[inline]
pub(crate) fn div_const(b: &Builder, x: V, d: u64) -> V {
    assert!((2..=(1 << 30)).contains(&d));
    if d.is_power_of_two() {
        return b.ri(Op::ShrI, x, d.trailing_zeros() as i64);
    }
    let m = (1u128 << 64).div_ceil(d as u128) as u64;
    b.ri(Op::MulHiI, x, m as i64)
}

#[inline]
pub(crate) fn div_mod_const(b: &Builder, x: V, d: u64) -> (V, V) {
    let q = div_const(b, x, d);
    let r = b.rr(Op::Add, x, b.ri(Op::MulI, q, -(d as i64)));
    (q, r)
}

/// Unsigned `(x / d, x % d)` over the whole 64-bit range.
#[inline]
pub(crate) fn div_mod_wide(b: &Builder, x: V, d: u64) -> (V, V) {
    let q = b.div_wide(x, d);
    let r = b.rr(Op::Add, x, b.ri(Op::MulI, q, -(d as i64)));
    (q, r)
}

#[inline]
pub(crate) fn two_digits(b: &Builder, val: V) -> V {
    let q = b.ri(Op::ShrI, b.ri(Op::MulI, val, 103), 10);
    let r = b.rr(Op::Add, val, b.ri(Op::MulI, q, -10));
    let raw = b.rr(Op::Or, q, b.ri(Op::ShlI, r, 8));
    b.ri(Op::AddI, raw, 0x3030)
}

#[inline]
pub(crate) fn eight_digits_raw_and_ascii(b: &Builder, val: V) -> (V, V) {
    let (d7654, d3210) = div_mod_const(b, val, 10_000);
    let (d76, d54) = div_mod_const(b, d7654, 100);
    let (d32, d10) = div_mod_const(b, d3210, 100);

    let pair_raw = |v: V| -> V {
        let q = b.ri(Op::ShrI, b.ri(Op::MulI, v, 103), 10);
        let r = b.rr(Op::Add, v, b.ri(Op::MulI, q, -10));
        b.rr(Op::Or, q, b.ri(Op::ShlI, r, 8))
    };

    let p0 = pair_raw(d76);
    let p1 = pair_raw(d54);
    let p2 = pair_raw(d32);
    let p3 = pair_raw(d10);

    let lo32 = b.rr(Op::Or, p0, b.ri(Op::ShlI, p1, 16));
    let hi32 = b.rr(Op::Or, p2, b.ri(Op::ShlI, p3, 16));
    let raw = b.rr(Op::Or, lo32, b.ri(Op::ShlI, hi32, 32));
    let ascii = b.ri(Op::AddI, raw, 0x3030_3030_3030_3030u64 as i64);
    (raw, ascii)
}

/// Formats variable-length signed `i64` values as canonical decimal text.
pub fn prog_fmt_int(is_deltas: bool, seed: i64) -> Program {
    let b = Builder::new(1);
    let inp = b.input(0);
    let val = if is_deltas {
        b.scan_add(inp, seed)
    } else {
        inp
    };

    let sign_mask = b.ri(Op::SarI, val, 63);
    let sign_len = b.ri(Op::ShrI, val, 63);
    let sign_ch = b.ri(Op::OrI, b.ri(Op::AndI, inp, 0), b'-' as i64);
    let abs = b.rr(Op::Sub, b.rr(Op::Xor, val, sign_mask), sign_mask);

    let inv_1e8 = 12_379_400_392_853_802_749u64 as i64;
    let hi1 = b.ri(Op::ShrI, b.ri(Op::MulHiI, abs, inv_1e8), 26);
    let lo8 = b.rr(Op::Add, abs, b.ri(Op::MulI, hi1, -100_000_000));

    let hi2 = b.ri(Op::ShrI, b.ri(Op::MulHiI, hi1, inv_1e8), 26);
    let mid8 = b.rr(Op::Add, hi1, b.ri(Op::MulI, hi2, -100_000_000));

    let (raw_hi, asc_hi) = eight_digits_raw_and_ascii(&b, hi2);
    let (raw_mid, asc_mid) = eight_digits_raw_and_ascii(&b, mid8);
    let (raw_lo, asc_lo) = eight_digits_raw_and_ascii(&b, lo8);

    let z_hi = b.ri(Op::ShrI, b.r(Op::Ctz, raw_hi), 3);
    let z_mid = b.ri(Op::ShrI, b.r(Op::Ctz, raw_mid), 3);
    let z_lo = b.ri(Op::ShrI, b.r(Op::Ctz, raw_lo), 3);

    let eight = b.ri(Op::AddI, b.ri(Op::AndI, inp, 0), 8);
    let one = b.ri(Op::AddI, b.ri(Op::AndI, inp, 0), 1);
    let zero = b.ri(Op::AndI, inp, 0);

    let len_hi = b.rr(Op::Sub, eight, z_hi);
    let shift_hi = b.ri(Op::ShlI, z_hi, 3);
    let bytes_hi = b.rr(Op::Shr, asc_hi, shift_hi);

    let mid_trimmed_len = b.rr(Op::Sub, eight, z_mid);
    let len_mid_no_hi = b.select(mid8, mid_trimmed_len, zero);
    let len_mid = b.select(hi2, eight, len_mid_no_hi);
    let shift_mid = b.select(hi2, zero, b.ri(Op::ShlI, z_mid, 3));
    let bytes_mid = b.rr(Op::Shr, asc_mid, shift_mid);

    let has_upper = b.rr(Op::Or, hi2, mid8);
    let lo_trimmed_len = b.select(lo8, b.rr(Op::Sub, eight, z_lo), one);
    let len_lo = b.select(has_upper, eight, lo_trimmed_len);
    let z_lo_clamped = b.select(lo8, z_lo, b.ri(Op::AddI, zero, 7));
    let shift_lo = b.select(has_upper, zero, b.ri(Op::ShlI, z_lo_clamped, 3));
    let bytes_lo = b.rr(Op::Shr, asc_lo, shift_lo);

    b.finish_sinks(&[
        (sign_ch, Len::Lane(sign_len.0)),
        (bytes_hi, Len::Lane(len_hi.0)),
        (bytes_mid, Len::Lane(len_mid.0)),
        (bytes_lo, Len::Lane(len_lo.0)),
    ])
}

/// Formats fixed-width zero-padded unsigned integers (`width` in `2..=16`).
fn prog_fmt_zero_padded_uint(width: usize, seed: i64) -> Program {
    let b = Builder::new(1);
    let d = b.input(0);
    let val = b.scan_add(d, seed);
    if width <= 8 {
        let scale = 10i64.pow((8 - width) as u32);
        let scaled = if scale == 1 {
            val
        } else {
            b.ri(Op::MulI, val, scale)
        };
        let (_, asc) = eight_digits_raw_and_ascii(&b, scaled);
        b.finish(&[(asc, width as u8)])
    } else {
        let lo_w = width - 8;
        let lo_div = 10u64.pow(lo_w as u32);
        let (hi8, lo_rem) = div_mod_wide(&b, val, lo_div);
        let (_, asc_hi) = eight_digits_raw_and_ascii(&b, hi8);
        let lo_scaled = b.ri(Op::MulI, lo_rem, 10i64.pow((8 - lo_w) as u32));
        let (_, asc_lo) = eight_digits_raw_and_ascii(&b, lo_scaled);
        b.finish(&[(asc_hi, 8), (asc_lo, lo_w as u8)])
    }
}

/// Formats `W64` values as `width` (`16` or `8`) hex characters (`lower` or `UPPER`).
fn emit_hex_64_sinks(b: &Builder, val_be: V, n_chars: usize, upper: bool) -> (V, V) {
    let hex_lut = if upper {
        b.add_lut16(*b"0123456789ABCDEF")
    } else {
        b.add_lut16(*b"0123456789abcdef")
    };
    let even_nib = b.nibble_lut(b.ri(Op::ShrI, val_be, 4), hex_lut);
    let odd_nib = b.nibble_lut(val_be, hex_lut);

    let unpack_lo = |a: V, c: V| -> V {
        let m1 = 0x0000_ffff_0000_ffffu64 as i64;
        let m2 = 0x00ff_00ff_00ff_00ffu64 as i64;
        let a0 = b.ri(Op::AndI, a, 0xffff_ffff);
        let a1 = b.ri(Op::AndI, b.rr(Op::Or, a0, b.ri(Op::ShlI, a0, 16)), m1);
        let a2 = b.ri(Op::AndI, b.rr(Op::Or, a1, b.ri(Op::ShlI, a1, 8)), m2);
        let c0 = b.ri(Op::AndI, c, 0xffff_ffff);
        let c1 = b.ri(Op::AndI, b.rr(Op::Or, c0, b.ri(Op::ShlI, c0, 16)), m1);
        let c2 = b.ri(Op::AndI, b.rr(Op::Or, c1, b.ri(Op::ShlI, c1, 8)), m2);
        b.rr(Op::Or, a2, b.ri(Op::ShlI, c2, 8))
    };

    let lo_hex = unpack_lo(even_nib, odd_nib);
    if n_chars <= 8 {
        return (lo_hex, lo_hex);
    }
    let hi_hex = unpack_lo(b.ri(Op::ShrI, even_nib, 32), b.ri(Op::ShrI, odd_nib, 32));
    (lo_hex, hi_hex)
}

// ============================================================================
// 4. Built-in Codec Packs (1..9)
// ============================================================================

/// Pack 1: Exact canonical signed integers and uniform fixed-width zero-padded unsigned integers.
pub struct ExactIntPack;

impl ExactIntPack {
    #[inline]
    pub fn parse_canonical_i64(s: &[u8]) -> Option<i64> {
        if s.is_empty() || s.len() > 20 {
            return None;
        }
        let (neg, digits) = if s[0] == b'-' {
            (true, &s[1..])
        } else {
            (false, s)
        };
        if digits.is_empty() {
            return None;
        }
        if digits[0] == b'0' && digits.len() > 1 {
            return None;
        }
        if neg && digits == b"0" {
            return None;
        }
        let mut acc: u64 = 0;
        for &b in digits {
            if !b.is_ascii_digit() {
                return None;
            }
            acc = acc.checked_mul(10)?.checked_add((b - b'0') as u64)?;
        }
        let val = if neg {
            if acc == (i64::MAX as u64) + 1 {
                i64::MIN
            } else if acc <= i64::MAX as u64 {
                -(acc as i64)
            } else {
                return None;
            }
        } else if acc <= i64::MAX as u64 {
            acc as i64
        } else {
            return None;
        };
        Some(val)
    }

    #[inline]
    fn parse_zero_padded_u64(s: &[u8], width: usize) -> Option<i64> {
        if s.len() != width || !(2..=16).contains(&width) {
            return None;
        }
        let mut acc = 0i64;
        for &b in s {
            if !b.is_ascii_digit() {
                return None;
            }
            acc = acc * 10 + (b - b'0') as i64;
        }
        Some(acc)
    }
}

impl CodecPack for ExactIntPack {
    fn name(&self) -> &'static str {
        "pack/exact-int"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.all_ascii && p.any_digit_start && p.max_len <= 20
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        // Check uniform zero-padded unsigned integers first when leading zeros appear.
        let w0 = rows[0].len();
        if (2..=16).contains(&w0)
            && rows
                .iter()
                .take(16)
                .any(|r| r.len() == w0 && r.starts_with(b"0"))
        {
            let mut vals = Vec::with_capacity(rows.len());
            let mut exc_idx = Vec::new();
            let mut exc_vals = Vec::new();
            let mut last = 0i64;
            let mut pad_ok = true;
            for (i, &r) in rows.iter().enumerate() {
                if let Some(v) = Self::parse_zero_padded_u64(r, w0) {
                    vals.push(v);
                    last = v;
                } else {
                    exc_idx.push(i);
                    exc_vals.push(r);
                    vals.push(last);
                    if exc_idx.len() * 8 > rows.len() {
                        pad_ok = false;
                        break;
                    }
                }
            }
            if pad_ok && exc_idx.len() < rows.len() {
                let mut deltas = Vec::with_capacity(vals.len());
                deltas.push(0);
                for w in vals.windows(2) {
                    deltas.push(w[1].wrapping_sub(w[0]));
                }
                let mut gb = GraphBuilder::new(rows.len());
                let dn = gb.add_ints(&deltas);
                let pn =
                    gb.add_prog_cells(prog_fmt_zero_padded_uint(w0, vals[0]), vec![dn], vec![]);
                let root = gb.add_patch(pn, &exc_idx, &exc_vals);
                return Some(gb.finish(root));
            }
        }

        let mut vals = Vec::with_capacity(rows.len());
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut last = 0i64;

        for (i, &r) in rows.iter().enumerate() {
            if let Some(v) = Self::parse_canonical_i64(r) {
                vals.push(v);
                last = v;
            } else {
                exc_idx.push(i);
                exc_vals.push(r);
                vals.push(last);
                if exc_idx.len() * 8 > rows.len() {
                    return None;
                }
            }
        }
        if exc_idx.len() == rows.len() {
            return None;
        }

        let mut deltas = Vec::with_capacity(vals.len());
        deltas.push(0);
        for w in vals.windows(2) {
            deltas.push(w[1].wrapping_sub(w[0]));
        }

        let mut gb_dir = GraphBuilder::new(rows.len());
        // Ids that come back (an actor, a repository) are recent ones again.
        let dir_in = gb_dir.add_mtf_ints(&vals, 4096);
        let dir_cells = gb_dir.add_prog_cells(prog_fmt_int(false, 0), vec![dir_in], vec![]);
        let dir_root = gb_dir.add_patch(dir_cells, &exc_idx, &exc_vals);
        let g_dir = gb_dir.finish(dir_root);

        let mut gb_del = GraphBuilder::new(rows.len());
        let del_in = gb_del.add_ints(&deltas);
        let del_cells = gb_del.add_prog_cells(prog_fmt_int(true, vals[0]), vec![del_in], vec![]);
        let del_root = gb_del.add_patch(del_cells, &exc_idx, &exc_vals);
        let g_del = gb_del.finish(del_root);

        if g_del.cost() <= g_dir.cost() {
            Some(g_del)
        } else {
            Some(g_dir)
        }
    }
}

/// Pack 2: Scaled decimal numbers (`12.34`, `-0.005`, `100.0`).
pub struct DecimalScaledPack;

impl DecimalScaledPack {
    fn parse_scaled(s: &[u8]) -> Option<(i64, u8)> {
        if s.len() < 3 || s.len() > 20 {
            return None;
        }
        let (neg, rest) = if s[0] == b'-' {
            (true, &s[1..])
        } else {
            (false, s)
        };
        let dot = rest.iter().position(|&b| b == b'.')?;
        let int_p = &rest[..dot];
        let frac_p = &rest[dot + 1..];
        if int_p.is_empty() || frac_p.is_empty() || frac_p.len() > 8 {
            return None;
        }
        if int_p.len() > 1 && int_p[0] == b'0' {
            return None;
        }
        if neg && int_p == b"0" && frac_p.iter().all(|&b| b == b'0') {
            return None;
        }
        let mut acc: u64 = 0;
        for &b in int_p.iter().chain(frac_p.iter()) {
            if !b.is_ascii_digit() {
                return None;
            }
            acc = acc.checked_mul(10)?.checked_add((b - b'0') as u64)?;
        }
        if acc >= 100_000_000_000_000_000 {
            return None;
        }
        let signed = if neg { -(acc as i64) } else { acc as i64 };
        Some((signed, frac_p.len() as u8))
    }

    fn prog_scaled_decimal(scale: u8, is_deltas: bool, seed: i64) -> Program {
        let b = Builder::new(1);
        let inp = b.input(0);
        let val = if is_deltas {
            b.scan_add(inp, seed)
        } else {
            inp
        };
        let sign_mask = b.ri(Op::SarI, val, 63);
        let sign_len = b.ri(Op::ShrI, val, 63);
        let sign_ch = b.ri(Op::OrI, b.ri(Op::AndI, inp, 0), b'-' as i64);
        let abs = b.rr(Op::Sub, b.rr(Op::Xor, val, sign_mask), sign_mask);

        let pow10 = 10u64.pow(scale as u32);
        let (int_val, frac_val) = div_mod_wide(&b, abs, pow10);

        let inv_1e8 = 12_379_400_392_853_802_749u64 as i64;
        let hi_int = b.ri(Op::ShrI, b.ri(Op::MulHiI, int_val, inv_1e8), 26);
        let lo_int = b.rr(Op::Add, int_val, b.ri(Op::MulI, hi_int, -100_000_000));

        let (raw_hi, asc_hi) = eight_digits_raw_and_ascii(&b, hi_int);
        let (raw_lo, asc_lo) = eight_digits_raw_and_ascii(&b, lo_int);
        let (_, asc_frac_full) = eight_digits_raw_and_ascii(&b, frac_val);

        let zero = b.ri(Op::AndI, inp, 0);
        let one = b.ri(Op::AddI, zero, 1);
        let eight = b.ri(Op::AddI, zero, 8);

        let z_hi = b.ri(Op::ShrI, b.r(Op::Ctz, raw_hi), 3);
        let len_hi = b.rr(Op::Sub, eight, z_hi);
        let bytes_hi = b.rr(Op::Shr, asc_hi, b.ri(Op::ShlI, z_hi, 3));

        let z_lo = b.ri(Op::ShrI, b.r(Op::Ctz, raw_lo), 3);
        let lo_trim_len = b.select(lo_int, b.rr(Op::Sub, eight, z_lo), one);
        let len_lo = b.select(hi_int, eight, lo_trim_len);
        let z_lo_c = b.select(lo_int, z_lo, b.ri(Op::AddI, zero, 7));
        let shift_lo = b.select(hi_int, zero, b.ri(Op::ShlI, z_lo_c, 3));
        let bytes_lo = b.rr(Op::Shr, asc_lo, shift_lo);

        let dot_ch = b.ri(Op::AddI, zero, b'.' as i64);
        let frac_bytes = b.ri(Op::ShrI, asc_frac_full, ((8 - scale) * 8) as i64);

        b.finish_sinks(&[
            (sign_ch, Len::Lane(sign_len.0)),
            (bytes_hi, Len::Lane(len_hi.0)),
            (bytes_lo, Len::Lane(len_lo.0)),
            (dot_ch, Len::Fixed(1)),
            (frac_bytes, Len::Fixed(scale)),
        ])
    }
}

impl CodecPack for DecimalScaledPack {
    fn name(&self) -> &'static str {
        "pack/decimal-scaled"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.all_ascii && p.any_decimal_dot && p.min_len >= 3 && p.max_len <= 24
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        let mut parsed: Vec<Option<(i64, u8)>> = Vec::with_capacity(rows.len());
        let mut scale_counts = [0usize; 9];
        let mut valid = 0usize;
        for &r in rows {
            let p = Self::parse_scaled(r);
            if let Some((_, sc)) = p {
                scale_counts[sc as usize] += 1;
                valid += 1;
            }
            parsed.push(p);
        }
        if valid * 8 < rows.len() * 7 {
            return None;
        }
        let best_scale = (1..=8u8).max_by_key(|&s| scale_counts[s as usize])?;
        if scale_counts[best_scale as usize] == 0 {
            return None;
        }

        let mut vals = Vec::with_capacity(rows.len());
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut last = 0i64;
        for (i, p) in parsed.into_iter().enumerate() {
            match p {
                Some((v, sc)) if sc == best_scale => {
                    vals.push(v);
                    last = v;
                }
                _ => {
                    exc_idx.push(i);
                    exc_vals.push(rows[i]);
                    vals.push(last);
                    if exc_idx.len() * 4 > rows.len() {
                        return None;
                    }
                }
            }
        }
        let mut deltas = Vec::with_capacity(vals.len());
        deltas.push(0);
        for w in vals.windows(2) {
            deltas.push(w[1].wrapping_sub(w[0]));
        }

        let mut gb = GraphBuilder::new(rows.len());
        let del_in = gb.add_ints(&deltas);
        let prog = Self::prog_scaled_decimal(best_scale, true, vals[0]);
        let cells = gb.add_prog_cells(prog, vec![del_in], vec![]);
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

/// Pack 3: Java `java.util.Random::nextLong()` LCG solver (`s_{k+1} = (s_k * 0x5deece66d + 11) & ((1<<48)-1)`).
pub struct JavaLcgPack;

const LCG_MASK: u64 = (1 << 48) - 1;
const LCG_MULT: u64 = 0x5deece66d;
const LCG_ADD: u64 = 11;

const LCG_JUMP: [[(u64, u64); 48]; 2] = {
    let mut inv = LCG_MULT;
    let mut i = 0;
    while i < 4 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(LCG_MULT.wrapping_mul(inv)));
        i += 1;
    }
    let steps = [
        (LCG_MULT, LCG_ADD),
        (
            inv & LCG_MASK,
            LCG_ADD.wrapping_neg().wrapping_mul(inv) & LCG_MASK,
        ),
    ];
    let mut table = [[(0, 0); 48]; 2];
    let mut dir = 0;
    while dir < 2 {
        let (mut a, mut c) = steps[dir];
        let mut k = 0;
        while k < 48 {
            table[dir][k] = (a, c);
            c = c.wrapping_mul(a.wrapping_add(1)) & LCG_MASK;
            a = a.wrapping_mul(a) & LCG_MASK;
            k += 1;
        }
        dir += 1;
    }
    table
};

impl JavaLcgPack {
    /// The value `nextLong()` returns from `state`, the state its high half is taken from.
    fn output(state: u64) -> i64 {
        let next = state.wrapping_mul(LCG_MULT).wrapping_add(LCG_ADD) & LCG_MASK;
        // The low half is sign extended before the addition.
        ((state >> 16) << 32).wrapping_add((next >> 16) as i32 as u64) as i64
    }

    /// The state whose output is `value`. One 64-bit value in 65,536 has one, and never two.
    fn state_of(value: i64) -> Option<u64> {
        let low = value as u32 as u64;
        let high = ((value as u64 >> 32) as u32).wrapping_add((low >> 31) as u32) as u64;
        // The state is `high << 16 | x`. Its successor's top 32 bits are `low` when `MULT * x`
        // lies in `[t, t + 2^16)` modulo 2^48, and `MULT * x < 2^51` leaves seven such ranges
        // with at most one multiple of `MULT` in each.
        let t = (low << 16).wrapping_sub((high << 16).wrapping_mul(LCG_MULT).wrapping_add(LCG_ADD))
            & LCG_MASK;
        for k in -1..=5i64 {
            let base = t as i64 + (k << 48);
            let x = (base.max(0) as u64).div_ceil(LCG_MULT);
            if x >> 16 != 0 {
                break;
            }
            if ((x * LCG_MULT) as i64) < base + (1 << 16) {
                let state = high << 16 | x;
                return (Self::output(state) == value).then_some(state);
            }
        }
        None
    }

    /// Steps from `from` to `to`, in `-2^47..2^47`.
    fn distance(from: u64, to: u64) -> i64 {
        // 2^i steps flip bit i of a state and keep the bits below it, so the count comes out
        // lowest bit first.
        let (mut state, mut steps) = (from, 0u64);
        for (i, &(a, c)) in LCG_JUMP[0].iter().enumerate() {
            if state == to {
                break;
            }
            if (state ^ to) >> i & 1 != 0 {
                state = state.wrapping_mul(a).wrapping_add(c) & LCG_MASK;
                steps |= 1 << i;
            }
        }
        (steps << 16) as i64 >> 16
    }

    /// Jumps the decoder makes for `steps`: one per set bit, walking the shorter way round.
    fn jumps(steps: i64) -> u32 {
        let forward = steps as u64 & LCG_MASK;
        let left = if forward >> 47 == 0 {
            forward
        } else {
            (1 << 48) - forward
        };
        left.count_ones()
    }

    /// Values whose states are `anchor` moved by the running sum of the input, making at most
    /// `rounds` jumps a row.
    fn prog_lcg_values(anchor: u64, rounds: u32) -> (Program, Vec<u8>) {
        let mut table_bytes = Vec::with_capacity(2 * 48 * 16);
        for dir in &LCG_JUMP {
            for &(a, c) in dir {
                table_bytes.extend_from_slice(&a.to_le_bytes());
                table_bytes.extend_from_slice(&c.to_le_bytes());
            }
        }
        let b = Builder::new(1);
        let buf = b.add_buf();
        let steps = b.scan_add(b.input(0), 0);
        let forward = b.ri(Op::AndI, steps, LCG_MASK as i64);
        let is_back = b.ri(Op::ShrI, forward, 47);
        let back_left = b.rr(
            Op::Sub,
            b.ri(Op::AddI, b.ri(Op::AndI, forward, 0), 1i64 << 48),
            forward,
        );
        let mut left = b.select(is_back, back_left, forward);
        let dir_base = b.ri(Op::MulI, is_back, 48 * 16);
        let mut state = b.ri(Op::AddI, b.ri(Op::AndI, forward, 0), anchor as i64);
        let one = b.ri(Op::AddI, b.ri(Op::AndI, forward, 0), 1);

        for _ in 0..rounds {
            let has_bit = left;
            let bit = b.ri(Op::AndI, b.r(Op::Ctz, left), 63);
            let idx = b.rr(Op::Add, dir_base, b.ri(Op::ShlI, bit, 4));
            let a = b.gather(buf, 3, idx, 0);
            let c = b.gather(buf, 3, idx, 8);
            let stepped = b.ri(
                Op::AndI,
                b.rr(Op::Add, b.rr(Op::Mul, state, a), c),
                LCG_MASK as i64,
            );
            state = b.select(has_bit, stepped, state);
            let cleared = b.rr(Op::And, left, b.rr(Op::Sub, left, one));
            left = b.select(has_bit, cleared, left);
        }

        let next = b.ri(
            Op::AndI,
            b.ri(
                Op::AddI,
                b.ri(Op::MulI, state, LCG_MULT as i64),
                LCG_ADD as i64,
            ),
            LCG_MASK as i64,
        );
        let hi = b.ri(Op::ShlI, b.ri(Op::ShrI, state, 16), 32);
        let lo_sext = b.ri(Op::SarI, b.ri(Op::ShlI, b.ri(Op::ShrI, next, 16), 32), 32);
        let out = b.rr(Op::Add, hi, lo_sext);
        (b.finish(&[(out, 8)]), table_bytes)
    }
}

impl CodecPack for JavaLcgPack {
    fn name(&self) -> &'static str {
        "pack/java-lcg"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 8 && p.all_ascii && p.any_digit_start && p.min_len >= 12 && p.max_len <= 20
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 8 {
            return None;
        }
        // Every instance walks one 2^48 cycle, so each output is its state's distance from the
        // first: small for ids drawn one after another, 48 bits for an unrelated instance.
        let mut anchor = None;
        let mut steps = Vec::with_capacity(rows.len());
        let (mut exc_idx, mut exc_vals) = (Vec::new(), Vec::new());
        for (i, &r) in rows.iter().enumerate() {
            // Other data almost never passes, so 16 values settle it.
            if i == 16 && exc_idx.len() > 8 {
                return None;
            }
            match ExactIntPack::parse_canonical_i64(r).and_then(Self::state_of) {
                Some(state) => {
                    steps.push(Self::distance(*anchor.get_or_insert(state), state));
                }
                None => {
                    exc_idx.push(i);
                    exc_vals.push(r);
                    steps.push(steps.last().copied().unwrap_or(0));
                }
            }
        }
        if exc_idx.len() * 8 > rows.len() {
            return None;
        }
        let anchor = anchor?;
        let rounds = steps.iter().map(|&s| Self::jumps(s)).max().unwrap_or(0);
        // The ids of `steps`, given as row-to-row differences.
        let ids = |gb: &mut GraphBuilder, steps: &[i64]| {
            let mut prev = 0i64;
            let deltas: Vec<i64> = steps
                .iter()
                .map(|&s| s - std::mem::replace(&mut prev, s))
                .collect();
            let (prog, jump_table) = Self::prog_lcg_values(anchor, rounds);
            let deltas_node = gb.add_ints(&deltas);
            let jumps_node = gb.add_raw_bytes(&jump_table);
            let vals_node = gb.add_prog_i64(prog, vec![deltas_node], vec![jumps_node]);
            gb.add_prog_cells(prog_fmt_int(false, 0), vec![vals_node], vec![])
        };

        let mut gb = GraphBuilder::new(rows.len());
        let cells = ids(&mut gb, &steps);
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        let mut best = gb.finish(root);

        // An id logged several times is a recent one again: only first sightings are
        // generated, and the rows index them.
        let (codes, firsts) = mtf_split(&steps, 4096);
        if firsts.len() * 2 < steps.len() {
            let mut gb = GraphBuilder::new(rows.len());
            let dict = ids(&mut gb, &firsts);
            let codes_node = gb.add_ints(&codes);
            let firsts_node = gb.add_row_index(firsts.len());
            let ords = gb.push(GraphNode::MtfInts {
                codes: codes_node,
                misses: firsts_node,
                window: 4096,
            });
            let cells = gb.add_index_cells(dict, ords);
            let root = gb.add_patch(cells, &exc_idx, &exc_vals);
            let g = gb.finish(root);
            if g.cost() < best.cost() {
                best = g;
            }
        }
        Some(best)
    }
}

/// Pack 4: MT19937 decimal `u32` streams, 32-char hex draws, and fixed-width binary draws.
pub struct PythonMtPack;

impl CodecPack for PythonMtPack {
    fn name(&self) -> &'static str {
        "pack/python-mt19937"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        if p.n_rows < 8 {
            return false;
        }
        let fixed_bin =
            p.min_len == p.max_len && (4..=64).contains(&p.min_len) && p.min_len.is_multiple_of(4);
        let dec_u32 = p.n_rows >= 625 && p.all_ascii && p.any_digit_start && p.max_len <= 10;
        let hex32 = p.n_rows >= 160 && p.min_len == 32 && p.max_len == 32 && p.any_hex_even_ge16;
        fixed_bin || dec_u32 || hex32
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        let n = rows.len();
        if n == 0 {
            return None;
        }
        // 1. Decimal `u32` draws (`>= 625` consecutive words).
        if n >= 625 && rows[0].len() <= 10 && rows[0].iter().all(|b| b.is_ascii_digit()) {
            let mut words = Vec::with_capacity(n);
            let mut ok = true;
            for &r in rows {
                if let Some(v) = ExactIntPack::parse_canonical_i64(r)
                    && (0..=(u32::MAX as i64)).contains(&v)
                {
                    words.push(v as u32);
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                let mut head624 = [0u32; MT_N];
                for i in 0..MT_N {
                    head624[i] = mt_untemper(words[i]);
                }
                let mut s = MtStream::after(&words[..MT_N]);
                s.grow(n);
                if s.words[..n] == words[..] {
                    let pre_seed = invert_mt_twist_624(&head624);
                    let mut gb = GraphBuilder::new(n);
                    let rbuf =
                        gb.add_recur_u32(recur_mt19937(true), &pre_seed, vec![], n as u32, false);
                    let col = gb.add_byte_chunks(rbuf, 4);
                    let cells = gb.add_prog_cells(prog_fmt_int(false, 0), vec![col], vec![]);
                    return Some(gb.finish(cells));
                }
            }
        }

        // 2. Fixed-width binary draws (`4..=64` bytes multiple of 4, `Le` or `Be`).
        let w = rows[0].len();
        if (4..=64).contains(&w) && w.is_multiple_of(4) && rows.iter().all(|r| r.len() == w) {
            let mut flat = Vec::with_capacity(n * w);
            for &r in rows {
                flat.extend_from_slice(r);
            }
            let mut gb = GraphBuilder::new(n);
            if let Some(root) = try_build_mt_cells(&mut gb, &flat, w, &[MtLayout::Le, MtLayout::Be])
            {
                let g = gb.finish(root);
                if g.cost() * 2 < flat.len() {
                    return Some(g);
                }
            }
        }
        None
    }
}

/// Pack 5: Generalized ISO-8601 timestamps (`YYYY-MM-DD[T|t| ]HH:MM:SS[(.|,)f{1..9}][Z|z|±HH:MM|±HHMM]`).
pub struct TimestampIsoPack;

#[derive(Clone, PartialEq, Eq)]
struct IsoLayout {
    sep: u8,
    frac_sep: u8,
    frac_digits: u8,
    tz_len: u8,
    tz_bytes: [u8; 6],
}

impl TimestampIsoPack {
    fn parse_two(s: &[u8]) -> Option<u32> {
        let d0 = s[0].wrapping_sub(b'0');
        let d1 = s[1].wrapping_sub(b'0');
        if d0 <= 9 && d1 <= 9 {
            Some((d0 as u32) * 10 + (d1 as u32))
        } else {
            None
        }
    }

    fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
        let y = y - i32::from(m <= 2);
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = (y - era * 400) as u32;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        (era as i64) * 146_097 + (doe as i64) - 719_468
    }

    fn parse_iso(s: &[u8]) -> Option<(IsoLayout, i64, i64)> {
        if !(19..=36).contains(&s.len()) {
            return None;
        }
        if s[4] != b'-' || s[7] != b'-' || s[13] != b':' || s[16] != b':' {
            return None;
        }
        let sep = s[10];
        if !matches!(sep, b'T' | b't' | b' ') {
            return None;
        }
        let y = Self::parse_two(&s[0..2])? * 100 + Self::parse_two(&s[2..4])?;
        let m = Self::parse_two(&s[5..7])?;
        let d = Self::parse_two(&s[8..10])?;
        let hh = Self::parse_two(&s[11..13])?;
        let mm = Self::parse_two(&s[14..16])?;
        let ss = Self::parse_two(&s[17..19])?;
        if !(1970..=9999).contains(&y)
            || !(1..=12).contains(&m)
            || !(1..=31).contains(&d)
            || hh > 23
            || mm > 59
            || ss > 59
        {
            return None;
        }
        let days = Self::days_from_civil(y as i32, m, d);
        if days < 0 {
            return None;
        }
        // Verify roundtrip of civil date (rejects invalid dates like Feb 30!).
        let z = days as u32 + 719_468;
        let era = z / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d_rt = doy - (153 * mp + 2) / 5 + 1;
        let m_rt = if mp < 10 { mp + 3 } else { mp - 9 };
        let y_rt = yoe + era * 400 + u32::from(m_rt <= 2);
        if (y_rt, m_rt, d_rt) != (y, m, d) {
            return None;
        }

        let secs = days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64);
        let mut pos = 19usize;
        let mut frac_sep = 0u8;
        let mut frac_digits = 0u8;
        let mut subsec = 0i64;
        if pos < s.len() && matches!(s[pos], b'.' | b',') {
            frac_sep = s[pos];
            pos += 1;
            let fstart = pos;
            while pos < s.len() && s[pos].is_ascii_digit() {
                subsec = subsec * 10 + (s[pos] - b'0') as i64;
                pos += 1;
            }
            let fd = pos - fstart;
            if !(1..=9).contains(&fd) {
                return None;
            }
            frac_digits = fd as u8;
        }
        let tz = &s[pos..];
        let valid_tz = match tz.len() {
            0 => true,
            1 => matches!(tz[0], b'Z' | b'z'),
            5 => {
                matches!(tz[0], b'+' | b'-')
                    && Self::parse_two(&tz[1..3]).is_some()
                    && Self::parse_two(&tz[3..5]).is_some()
            }
            6 => {
                matches!(tz[0], b'+' | b'-')
                    && tz[3] == b':'
                    && Self::parse_two(&tz[1..3]).is_some()
                    && Self::parse_two(&tz[4..6]).is_some()
            }
            _ => false,
        };
        if !valid_tz {
            return None;
        }
        let mut tz_bytes = [0u8; 6];
        tz_bytes[..tz.len()].copy_from_slice(tz);
        Some((
            IsoLayout {
                sep,
                frac_sep,
                frac_digits,
                tz_len: tz.len() as u8,
                tz_bytes,
            },
            secs,
            subsec,
        ))
    }

    fn prog_iso(layout: &IsoLayout, seed_sec: i64, seed_sub: i64) -> Program {
        let has_frac = layout.frac_digits > 0;
        let b = Builder::new(if has_frac { 2 } else { 1 });
        let sec_d = b.input(0);
        let t = b.scan_add(sec_d, seed_sec);
        let zero = b.ri(Op::AndI, sec_d, 0);

        let (days, sod) = div_mod_const(&b, t, 86_400);
        let z = b.ri(Op::AddI, days, 719_468);
        let (era, doe) = div_mod_const(&b, z, 146_097);

        let doe_1460 = div_const(&b, doe, 1460);
        let doe_36524 = div_const(&b, doe, 36_524);
        let doe_146096 = div_const(&b, doe, 146_096);
        let num = b.rr(
            Op::Sub,
            b.rr(Op::Add, b.rr(Op::Sub, doe, doe_1460), doe_36524),
            doe_146096,
        );
        let yoe = div_const(&b, num, 365);

        let yoe_365 = b.ri(Op::MulI, yoe, 365);
        let yoe_4 = b.ri(Op::ShrI, yoe, 2);
        let yoe_100 = div_const(&b, yoe, 100);
        let leap_days = b.rr(Op::Sub, b.rr(Op::Add, yoe_365, yoe_4), yoe_100);
        let doy = b.rr(Op::Sub, doe, leap_days);

        let mp = div_const(&b, b.ri(Op::AddI, b.ri(Op::MulI, doy, 5), 2), 153);
        let mp_days = div_const(&b, b.ri(Op::AddI, b.ri(Op::MulI, mp, 153), 2), 5);
        let d = b.ri(Op::AddI, b.rr(Op::Sub, doy, mp_days), 1);

        let mp_lt_10 = b.ri(Op::ShrI, b.ri(Op::AddI, mp, -10), 63);
        let m = b.select(mp_lt_10, b.ri(Op::AddI, mp, 3), b.ri(Op::AddI, mp, -9));
        let m_le_2 = b.ri(Op::ShrI, b.ri(Op::AddI, m, -3), 63);
        let y = b.rr(
            Op::Add,
            b.rr(Op::Add, yoe, b.ri(Op::MulI, era, 400)),
            m_le_2,
        );

        let (y_hi, y_lo) = div_mod_const(&b, y, 100);
        let (hh, rem_s) = div_mod_const(&b, sod, 3600);
        let (mm, ss) = div_mod_const(&b, rem_s, 60);

        let yy_hi = two_digits(&b, y_hi);
        let yy_lo = two_digits(&b, y_lo);
        let mo = two_digits(&b, m);
        let da = two_digits(&b, d);
        let hr = two_digits(&b, hh);
        let mi = two_digits(&b, mm);
        let sc = two_digits(&b, ss);

        let s0_digits = b.rr(
            Op::Or,
            b.rr(Op::Or, yy_hi, b.ri(Op::ShlI, yy_lo, 16)),
            b.ri(Op::ShlI, mo, 40),
        );
        let s0 = b.ri(
            Op::OrI,
            s0_digits,
            ((b'-' as i64) << 32) | ((b'-' as i64) << 56),
        );

        let s1_digits = b.rr(
            Op::Or,
            b.rr(Op::Or, da, b.ri(Op::ShlI, hr, 24)),
            b.ri(Op::ShlI, mi, 48),
        );
        let s1 = b.ri(
            Op::OrI,
            s1_digits,
            ((layout.sep as i64) << 16) | ((b':' as i64) << 40),
        );

        let s2 = b.ri(Op::OrI, b.ri(Op::ShlI, sc, 8), b':' as i64);
        let mut sinks = vec![(s0, 8u8), (s1, 8u8), (s2, 3u8)];

        if has_frac {
            let sub_d = b.input(1);
            let sub = b.scan_add(sub_d, seed_sub);
            let fsep_v = b.ri(Op::OrI, zero, layout.frac_sep as i64);
            sinks.push((fsep_v, 1));
            let fd = layout.frac_digits as usize;
            if fd <= 8 {
                let scaled = if fd == 8 {
                    sub
                } else {
                    b.ri(Op::MulI, sub, 10i64.pow((8 - fd) as u32))
                };
                let (_, asc) = eight_digits_raw_and_ascii(&b, scaled);
                sinks.push((asc, fd as u8));
            } else {
                let (hi8, lo1) = div_mod_const(&b, sub, 10);
                let (_, asc8) = eight_digits_raw_and_ascii(&b, hi8);
                let asc1 = b.ri(Op::AddI, lo1, b'0' as i64);
                sinks.push((asc8, 8));
                sinks.push((asc1, 1));
            }
        }
        if layout.tz_len > 0 {
            let mut tz_u64 = [0u8; 8];
            tz_u64[..layout.tz_len as usize]
                .copy_from_slice(&layout.tz_bytes[..layout.tz_len as usize]);
            let tz_v = b.ri(Op::OrI, zero, u64::from_le_bytes(tz_u64) as i64);
            sinks.push((tz_v, layout.tz_len));
        }

        b.finish(&sinks)
    }
}

impl CodecPack for TimestampIsoPack {
    fn name(&self) -> &'static str {
        "pack/timestamp-iso8601"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 4 && p.all_ascii && p.any_iso_ts
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        let target_layout = rows
            .iter()
            .take(16)
            .find_map(|r| Self::parse_iso(r))
            .map(|x| x.0)?;
        let mut secs = Vec::with_capacity(rows.len());
        let mut subs = Vec::with_capacity(rows.len());
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut last_s = 0i64;
        let mut last_sub = 0i64;

        for (i, &r) in rows.iter().enumerate() {
            if let Some((lay, sc, sb)) = Self::parse_iso(r)
                && lay == target_layout
            {
                secs.push(sc);
                subs.push(sb);
                last_s = sc;
                last_sub = sb;
            } else {
                exc_idx.push(i);
                exc_vals.push(r);
                secs.push(last_s);
                subs.push(last_sub);
                if exc_idx.len() * 8 > rows.len() {
                    return None;
                }
            }
        }
        if exc_idx.len() == rows.len() {
            return None;
        }
        let mut sec_d = Vec::with_capacity(secs.len());
        sec_d.push(0);
        for w in secs.windows(2) {
            sec_d.push(w[1] - w[0]);
        }
        let mut gb = GraphBuilder::new(rows.len());
        let in0 = gb.add_ints(&sec_d);
        let inputs = if target_layout.frac_digits > 0 {
            let mut sub_d = Vec::with_capacity(subs.len());
            sub_d.push(0);
            for w in subs.windows(2) {
                sub_d.push(w[1] - w[0]);
            }
            let in1 = gb.add_ints(&sub_d);
            vec![in0, in1]
        } else {
            vec![in0]
        };
        let prog = Self::prog_iso(&target_layout, secs[0], subs[0]);
        let cells = gb.add_prog_cells(prog, inputs, vec![]);
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

/// Pack 6: Apache CLF (`26` bytes: `DD/Mon/YYYY:HH:MM:SS ±HHMM`) and RFC3164 Syslog (`15` bytes: `Mmm [ D]D HH:MM:SS`).
pub struct TimestampClfPack;

const MONTHS_3: [&[u8; 3]; 12] = [
    b"Jan", b"Feb", b"Mar", b"Apr", b"May", b"Jun", b"Jul", b"Aug", b"Sep", b"Oct", b"Nov", b"Dec",
];

impl TimestampClfPack {
    fn parse_clf_26(s: &[u8]) -> Option<(i64, [u8; 5])> {
        if s.len() != 26
            || s[2] != b'/'
            || s[6] != b'/'
            || s[11] != b':'
            || s[14] != b':'
            || s[17] != b':'
            || s[20] != b' '
        {
            return None;
        }
        let d = TimestampIsoPack::parse_two(&s[0..2])?;
        let mon_b: &[u8; 3] = s[3..6].try_into().ok()?;
        let m = (MONTHS_3.iter().position(|&x| x == mon_b)? + 1) as u32;
        let y =
            TimestampIsoPack::parse_two(&s[7..9])? * 100 + TimestampIsoPack::parse_two(&s[9..11])?;
        let hh = TimestampIsoPack::parse_two(&s[12..14])?;
        let mm = TimestampIsoPack::parse_two(&s[15..17])?;
        let ss = TimestampIsoPack::parse_two(&s[18..20])?;
        let tz: [u8; 5] = s[21..26].try_into().ok()?;
        if !matches!(tz[0], b'+' | b'-') || !tz[1..].iter().all(|b| b.is_ascii_digit()) {
            return None;
        }
        if !(1970..=9999).contains(&y) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 59 {
            return None;
        }
        if d > days_in_month(y, m) {
            return None;
        }
        let days = TimestampIsoPack::days_from_civil(y as i32, m, d);
        if days < 0 {
            return None;
        }
        Some((
            days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64),
            tz,
        ))
    }

    fn parse_syslog_15(s: &[u8]) -> Option<i64> {
        if s.len() != 15 || s[3] != b' ' || s[6] != b' ' || s[9] != b':' || s[12] != b':' {
            return None;
        }
        let mon_b: &[u8; 3] = s[0..3].try_into().ok()?;
        let m = (MONTHS_3.iter().position(|&x| x == mon_b)? + 1) as u32;
        let d = if s[4] == b' ' {
            let d1 = s[5].wrapping_sub(b'0');
            if !(1..=9).contains(&d1) {
                return None;
            }
            d1 as u32
        } else if s[4] == b'0' {
            // Days under ten are written space-padded.
            return None;
        } else {
            TimestampIsoPack::parse_two(&s[4..6])?
        };
        let hh = TimestampIsoPack::parse_two(&s[7..9])?;
        let mm = TimestampIsoPack::parse_two(&s[10..12])?;
        let ss = TimestampIsoPack::parse_two(&s[13..15])?;
        if !(1..=days_in_month(2024, m)).contains(&d) || hh > 23 || mm > 59 || ss > 59 {
            return None;
        }
        // Reference leap year 2024 so `Feb 29` roundtrips cleanly.
        let days = TimestampIsoPack::days_from_civil(2024, m, d);
        if days < 0 {
            return None;
        }
        Some(days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64))
    }

    fn prog_clf_26(seed: i64, tz: [u8; 5]) -> (Program, Vec<u8>) {
        let mut month_tbl = Vec::with_capacity(48);
        for m in MONTHS_3 {
            month_tbl.extend_from_slice(m);
            month_tbl.push(b'/');
        }
        let b = Builder::new(1);
        let mbuf = b.add_buf();
        let inp = b.input(0);
        let t = b.scan_add(inp, seed);

        let (days, sod) = div_mod_const(&b, t, 86_400);
        let z = b.ri(Op::AddI, days, 719_468);
        let (era, doe) = div_mod_const(&b, z, 146_097);
        let num = b.rr(
            Op::Sub,
            b.rr(
                Op::Add,
                b.rr(Op::Sub, doe, div_const(&b, doe, 1460)),
                div_const(&b, doe, 36_524),
            ),
            div_const(&b, doe, 146_096),
        );
        let yoe = div_const(&b, num, 365);
        let leap_days = b.rr(
            Op::Sub,
            b.rr(Op::Add, b.ri(Op::MulI, yoe, 365), b.ri(Op::ShrI, yoe, 2)),
            div_const(&b, yoe, 100),
        );
        let doy = b.rr(Op::Sub, doe, leap_days);
        let mp = div_const(&b, b.ri(Op::AddI, b.ri(Op::MulI, doy, 5), 2), 153);
        let d = b.ri(
            Op::AddI,
            b.rr(
                Op::Sub,
                doy,
                div_const(&b, b.ri(Op::AddI, b.ri(Op::MulI, mp, 153), 2), 5),
            ),
            1,
        );
        let mp_lt_10 = b.ri(Op::ShrI, b.ri(Op::AddI, mp, -10), 63);
        let m = b.select(mp_lt_10, b.ri(Op::AddI, mp, 3), b.ri(Op::AddI, mp, -9));
        let m_le_2 = b.ri(Op::ShrI, b.ri(Op::AddI, m, -3), 63);
        let y = b.rr(
            Op::Add,
            b.rr(Op::Add, yoe, b.ri(Op::MulI, era, 400)),
            m_le_2,
        );

        let (y_hi, y_lo) = div_mod_const(&b, y, 100);
        let (hh, rem_s) = div_mod_const(&b, sod, 3600);
        let (mm, ss) = div_mod_const(&b, rem_s, 60);

        let da = two_digits(&b, d);
        let mon_slash = b.gather(mbuf, 2, b.ri(Op::ShlI, b.ri(Op::AddI, m, -1), 2), 0);
        let yy_hi = two_digits(&b, y_hi);
        let yy_lo = two_digits(&b, y_lo);
        let hr = two_digits(&b, hh);
        let mi = two_digits(&b, mm);
        let sc = two_digits(&b, ss);

        let s0 = b.ri(
            Op::OrI,
            b.rr(
                Op::Or,
                b.rr(Op::Or, da, b.ri(Op::ShlI, mon_slash, 24)),
                b.ri(Op::ShlI, b.ri(Op::AndI, yy_hi, 0xff), 56),
            ),
            (b'/' as i64) << 16,
        );
        let s1_digits = b.rr(
            Op::Or,
            b.rr(
                Op::Or,
                b.rr(Op::Or, b.ri(Op::ShrI, yy_hi, 8), b.ri(Op::ShlI, yy_lo, 8)),
                b.ri(Op::ShlI, hr, 32),
            ),
            b.ri(Op::ShlI, b.ri(Op::AndI, mi, 0xff), 56),
        );
        let s1 = b.ri(
            Op::OrI,
            s1_digits,
            ((b':' as i64) << 24) | ((b':' as i64) << 48),
        );
        let tz_lo4 = u32::from_le_bytes(tz[0..4].try_into().unwrap()) as i64;
        let s2 = b.ri(
            Op::OrI,
            b.rr(Op::Or, b.ri(Op::ShrI, mi, 8), b.ri(Op::ShlI, sc, 16)),
            ((b':' as i64) << 8) | ((b' ' as i64) << 32) | (tz_lo4 << 40),
        );
        let s3 = b.ri(
            Op::OrI,
            b.ri(Op::ShrI, yy_lo, 8),
            ((tz[3] as i64) & 0xff) | ((tz[4] as i64) << 8),
        );
        let s3_clean = b.ri(
            Op::OrI,
            b.ri(Op::AndI, s3, 0),
            (tz[3] as i64) | ((tz[4] as i64) << 8),
        );
        (
            b.finish(&[(s0, 8), (s1, 8), (s2, 8), (s3_clean, 2)]),
            month_tbl,
        )
    }

    fn prog_syslog_15(seed: i64) -> (Program, Vec<u8>) {
        let mut month_tbl = Vec::with_capacity(48);
        for m in MONTHS_3 {
            month_tbl.extend_from_slice(m);
            month_tbl.push(b' ');
        }
        let b = Builder::new(1);
        let mbuf = b.add_buf();
        let inp = b.input(0);
        let t = b.scan_add(inp, seed);
        let (days, sod) = div_mod_const(&b, t, 86_400);
        let z = b.ri(Op::AddI, days, 719_468);
        let (_, doe) = div_mod_const(&b, z, 146_097);
        let num = b.rr(
            Op::Sub,
            b.rr(
                Op::Add,
                b.rr(Op::Sub, doe, div_const(&b, doe, 1460)),
                div_const(&b, doe, 36_524),
            ),
            div_const(&b, doe, 146_096),
        );
        let yoe = div_const(&b, num, 365);
        let leap_days = b.rr(
            Op::Sub,
            b.rr(Op::Add, b.ri(Op::MulI, yoe, 365), b.ri(Op::ShrI, yoe, 2)),
            div_const(&b, yoe, 100),
        );
        let doy = b.rr(Op::Sub, doe, leap_days);
        let mp = div_const(&b, b.ri(Op::AddI, b.ri(Op::MulI, doy, 5), 2), 153);
        let d = b.ri(
            Op::AddI,
            b.rr(
                Op::Sub,
                doy,
                div_const(&b, b.ri(Op::AddI, b.ri(Op::MulI, mp, 153), 2), 5),
            ),
            1,
        );
        let mp_lt_10 = b.ri(Op::ShrI, b.ri(Op::AddI, mp, -10), 63);
        let m = b.select(mp_lt_10, b.ri(Op::AddI, mp, 3), b.ri(Op::AddI, mp, -9));

        let mon_sp = b.gather(mbuf, 2, b.ri(Op::ShlI, b.ri(Op::AddI, m, -1), 2), 0);
        let d_lt_10 = b.ri(Op::ShrI, b.ri(Op::AddI, d, -10), 63);
        let da_sp = b.ri(
            Op::OrI,
            b.ri(Op::ShlI, b.ri(Op::AddI, d, b'0' as i64), 8),
            b' ' as i64,
        );
        let da_2 = two_digits(&b, d);
        let da = b.select(d_lt_10, da_sp, da_2);

        let (hh, rem_s) = div_mod_const(&b, sod, 3600);
        let (mm, ss) = div_mod_const(&b, rem_s, 60);
        let hr = two_digits(&b, hh);
        let mi = two_digits(&b, mm);
        let sc = two_digits(&b, ss);

        // 15 bytes = 8 (`Mmm DD H`) + 7 (`H:MM:SS`)
        let s0 = b.ri(
            Op::OrI,
            b.rr(
                Op::Or,
                b.rr(Op::Or, mon_sp, b.ri(Op::ShlI, da, 32)),
                b.ri(Op::ShlI, b.ri(Op::AndI, hr, 0xff), 56),
            ),
            (b' ' as i64) << 48,
        );
        let s1 = b.ri(
            Op::OrI,
            b.rr(
                Op::Or,
                b.rr(Op::Or, b.ri(Op::ShrI, hr, 8), b.ri(Op::ShlI, mi, 16)),
                b.ri(Op::ShlI, sc, 40),
            ),
            ((b':' as i64) << 8) | ((b':' as i64) << 32),
        );
        (b.finish(&[(s0, 8), (s1, 7)]), month_tbl)
    }
}

impl CodecPack for TimestampClfPack {
    fn name(&self) -> &'static str {
        "pack/timestamp-clf"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 4 && p.all_ascii && (p.any_clf_ts || p.any_syslog_ts)
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        // 1. 15-byte RFC3164 Syslog (`Mmm [ D]D HH:MM:SS`).
        if rows
            .iter()
            .take(8)
            .any(|r| Self::parse_syslog_15(r).is_some())
        {
            let mut secs = Vec::with_capacity(rows.len());
            let mut exc_idx = Vec::new();
            let mut exc_vals = Vec::new();
            let mut last = 0i64;
            let mut ok = true;
            for (i, &r) in rows.iter().enumerate() {
                if let Some(sc) = Self::parse_syslog_15(r) {
                    secs.push(sc);
                    last = sc;
                } else {
                    exc_idx.push(i);
                    exc_vals.push(r);
                    secs.push(last);
                    if exc_idx.len() * 8 > rows.len() {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && exc_idx.len() < rows.len() {
                let mut deltas = Vec::with_capacity(secs.len());
                deltas.push(0);
                for w in secs.windows(2) {
                    deltas.push(w[1] - w[0]);
                }
                let (prog, tbl) = Self::prog_syslog_15(secs[0]);
                let mut gb = GraphBuilder::new(rows.len());
                let in_node = gb.add_ints(&deltas);
                let mbuf = gb.add_raw_bytes(&tbl);
                let cells = gb.add_prog_cells(prog, vec![in_node], vec![mbuf]);
                let root = gb.add_patch(cells, &exc_idx, &exc_vals);
                return Some(gb.finish(root));
            }
        }

        // 2. 26-byte Apache CLF (`DD/Mon/YYYY:HH:MM:SS ±HHMM`).
        let mut secs = Vec::with_capacity(rows.len());
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut target_tz: Option<[u8; 5]> = None;
        let mut last = 0i64;

        for (i, &r) in rows.iter().enumerate() {
            if let Some((sc, tz)) = Self::parse_clf_26(r) {
                let tz_ref = *target_tz.get_or_insert(tz);
                if tz == tz_ref {
                    secs.push(sc);
                    last = sc;
                    continue;
                }
            }
            exc_idx.push(i);
            exc_vals.push(r);
            secs.push(last);
            if exc_idx.len() * 8 > rows.len() {
                return None;
            }
        }
        let tz = target_tz?;
        let mut deltas = Vec::with_capacity(secs.len());
        deltas.push(0);
        for w in secs.windows(2) {
            deltas.push(w[1] - w[0]);
        }
        let (prog, tbl) = Self::prog_clf_26(secs[0], tz);
        let mut gb = GraphBuilder::new(rows.len());
        let in_node = gb.add_ints(&deltas);
        let mbuf = gb.add_raw_bytes(&tbl);
        let cells = gb.add_prog_cells(prog, vec![in_node], vec![mbuf]);
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

/// Pack 7: UUID (`v1..v8` 36-byte canonical dashed hex, with v7 timestamp delta & MT19937 tail support).
pub struct UuidPack;

#[inline]
pub(crate) fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

impl UuidPack {
    fn parse_uuid(s: &[u8]) -> Option<([u8; 16], bool)> {
        if s.len() != 36 || s[8] != b'-' || s[13] != b'-' || s[18] != b'-' || s[23] != b'-' {
            return None;
        }
        let mut has_lower = false;
        let mut has_upper = false;
        let mut out = [0u8; 16];
        const PAIRS: [usize; 16] = [0, 2, 4, 6, 9, 11, 14, 16, 19, 21, 24, 26, 28, 30, 32, 34];
        for (byte, &pos) in out.iter_mut().zip(&PAIRS) {
            let c0 = s[pos];
            let c1 = s[pos + 1];
            if c0.is_ascii_lowercase() || c1.is_ascii_lowercase() {
                has_lower = true;
            }
            if c0.is_ascii_uppercase() || c1.is_ascii_uppercase() {
                has_upper = true;
            }
            if has_lower && has_upper {
                return None;
            }
            *byte = (hex_val(c0)? << 4) | hex_val(c1)?;
        }
        Some((out, has_upper))
    }

    fn prog_uuid_36(upper: bool, v7_ts_seed: Option<i64>) -> Program {
        let b = Builder::new(if v7_ts_seed.is_some() { 3 } else { 2 });
        let hi64_be = if let Some(seed) = v7_ts_seed {
            let ts_d = b.input(0);
            let tail2_be = b.input(1);
            let ts48 = b.scan_add(ts_d, seed);
            let ts_swapped = b.ri(Op::ShrI, b.r(Op::Bswap, ts48), 16);
            b.rr(Op::Or, ts_swapped, b.ri(Op::ShlI, tail2_be, 48))
        } else {
            b.input(0)
        };
        let lo64_be = b.input(if v7_ts_seed.is_some() { 2 } else { 1 });

        let (h_0_7, h_8_15) = emit_hex_64_sinks(&b, hi64_be, 16, upper);
        let (h_16_23, h_24_31) = emit_hex_64_sinks(&b, lo64_be, 16, upper);
        let dash = b.ri(Op::OrI, b.ri(Op::AndI, lo64_be, 0), b'-' as i64);

        let g2 = b.ri(Op::AndI, h_8_15, 0xffff_ffff);
        let g3 = b.ri(Op::ShrI, h_8_15, 32);
        let g4 = b.ri(Op::AndI, h_16_23, 0xffff_ffff);
        let g5_hi4 = b.ri(Op::ShrI, h_16_23, 32);

        b.finish(&[
            (h_0_7, 8),
            (dash, 1),
            (g2, 4),
            (dash, 1),
            (g3, 4),
            (dash, 1),
            (g4, 4),
            (dash, 1),
            (g5_hi4, 4),
            (h_24_31, 8),
        ])
    }
}

impl CodecPack for UuidPack {
    fn name(&self) -> &'static str {
        "pack/uuid"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 4 && p.any_uuid_dash
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        let mut raw16: Vec<[u8; 16]> = Vec::with_capacity(rows.len());
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut target_upper: Option<bool> = None;
        let mut v7_hits = 0usize;
        let mut last = [0u8; 16];

        for (i, &r) in rows.iter().enumerate() {
            if let Some((bytes, up)) = Self::parse_uuid(r) {
                let u_ref = *target_upper.get_or_insert(up);
                if up == u_ref {
                    if (bytes[6] >> 4) == 7 {
                        v7_hits += 1;
                    }
                    raw16.push(bytes);
                    last = bytes;
                    continue;
                }
            }
            exc_idx.push(i);
            exc_vals.push(r);
            raw16.push(last);
            if exc_idx.len() * 8 > rows.len() {
                return None;
            }
        }
        let upper = target_upper?;
        let n = rows.len();
        let mut gb = GraphBuilder::new(n);

        let cells = if v7_hits * 2 >= n {
            let mut ts_vals = Vec::with_capacity(n);
            let mut tail10 = Vec::with_capacity(n * 10);
            for b16 in &raw16 {
                let mut t8 = [0u8; 8];
                t8[2..8].copy_from_slice(&b16[0..6]);
                ts_vals.push(u64::from_be_bytes(t8) as i64);
                tail10.extend_from_slice(&b16[6..16]);
            }
            let mut ts_d = Vec::with_capacity(n);
            ts_d.push(0);
            for w in ts_vals.windows(2) {
                ts_d.push(w[1] - w[0]);
            }
            let ts_node = gb.add_ints(&ts_d);

            // Try `Seeds` MT19937 `Uuid7` layout on the 10-byte `(rand_a, rand_b)` tail first!
            if let Some(mt_tail10) = try_build_mt_cells(&mut gb, &tail10, 10, &[MtLayout::Uuid7]) {
                // Split `mt_tail10` (`10` bytes/row) into `tail2` (`2`B) and `lo8` (`8`B).
                let b_split = Builder::new(1);
                let tbuf = b_split.add_buf();
                let idx = b_split.input(0);
                let off = b_split.ri(Op::MulI, idx, 10);
                let t2 = b_split.gather(tbuf, 1, off, 0);
                let l8 = b_split.gather(tbuf, 3, off, 2);
                let row_idx = gb.add_row_index(n);
                let t2_col = gb.add_prog_i64(
                    {
                        let bs = Builder::new(1);
                        let tb = bs.add_buf();
                        let i0 = bs.input(0);
                        let v = bs.gather(tb, 1, bs.ri(Op::MulI, i0, 10), 0);
                        bs.finish(&[(v, 8)])
                    },
                    vec![row_idx],
                    vec![mt_tail10],
                );
                let l8_col = gb.add_prog_i64(
                    {
                        let bs = Builder::new(1);
                        let tb = bs.add_buf();
                        let i0 = bs.input(0);
                        let v = bs.gather(tb, 3, bs.ri(Op::MulI, i0, 10), 2);
                        bs.finish(&[(v, 8)])
                    },
                    vec![row_idx],
                    vec![mt_tail10],
                );
                let _ = (b_split, t2, l8);
                gb.add_prog_cells(
                    Self::prog_uuid_36(upper, Some(ts_vals[0])),
                    vec![ts_node, t2_col, l8_col],
                    vec![],
                )
            } else {
                let mut tail2_bytes = Vec::with_capacity(n * 2);
                let mut lo8_bytes = Vec::with_capacity(n * 8);
                for b16 in &raw16 {
                    tail2_bytes.extend_from_slice(&b16[6..8]);
                    lo8_bytes.extend_from_slice(&b16[8..16]);
                }
                let t2_buf = gb.add_bytes(&tail2_bytes);
                let t2_col = gb.add_byte_chunks(t2_buf, 2);
                let lo_buf = gb.add_mtf_fixed(&lo8_bytes, 8);
                let lo_col = gb.add_byte_chunks(lo_buf, 8);
                gb.add_prog_cells(
                    Self::prog_uuid_36(upper, Some(ts_vals[0])),
                    vec![ts_node, t2_col, lo_col],
                    vec![],
                )
            }
        } else {
            let mut flat16 = Vec::with_capacity(n * 16);
            for b16 in &raw16 {
                flat16.extend_from_slice(b16);
            }
            let buf16 = try_build_mt_cells(&mut gb, &flat16, 16, &[MtLayout::Le, MtLayout::Be])
                .unwrap_or_else(|| gb.add_mtf_fixed(&flat16, 16));
            let row_idx = gb.add_row_index(n);
            let hi_col = gb.add_prog_i64(
                {
                    let bs = Builder::new(1);
                    let tb = bs.add_buf();
                    let i0 = bs.input(0);
                    let v = bs.gather(tb, 3, bs.ri(Op::ShlI, i0, 4), 0);
                    bs.finish(&[(v, 8)])
                },
                vec![row_idx],
                vec![buf16],
            );
            let lo_col = gb.add_prog_i64(
                {
                    let bs = Builder::new(1);
                    let tb = bs.add_buf();
                    let i0 = bs.input(0);
                    let v = bs.gather(tb, 3, bs.ri(Op::ShlI, i0, 4), 8);
                    bs.finish(&[(v, 8)])
                },
                vec![row_idx],
                vec![buf16],
            );
            gb.add_prog_cells(
                Self::prog_uuid_36(upper, None),
                vec![hi_col, lo_col],
                vec![],
            )
        };

        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

/// Pack 8: Crockford Base32 ULIDs (26 chars = 48-bit ms timestamp + 80-bit entropy).
pub struct UlidPack;

const CROCKFORD_32: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

impl UlidPack {
    fn inv_table() -> [u8; 256] {
        let mut t = [255u8; 256];
        for (i, &c) in CROCKFORD_32.iter().enumerate() {
            t[c as usize] = i as u8;
        }
        t
    }

    fn parse_ulid(s: &[u8], inv: &[u8; 256]) -> Option<(i64, u64, u16)> {
        if s.len() != 26 || s[0] > b'7' {
            return None;
        }
        let mut ts: u64 = 0;
        for &c in &s[0..10] {
            let v = inv[c as usize];
            if v == 255 {
                return None;
            }
            ts = (ts << 5) | (v as u64);
        }
        let mut ent: u128 = 0;
        for &c in &s[10..26] {
            let v = inv[c as usize];
            if v == 255 {
                return None;
            }
            ent = (ent << 5) | (v as u128);
        }
        let ent_hi64 = (ent >> 16) as u64;
        let ent_lo16 = (ent & 0xffff) as u16;
        Some((ts as i64, ent_hi64, ent_lo16))
    }

    fn prog_ulid(ts_seed: i64) -> Program {
        let b = Builder::new(3);
        let lut = b.add_lut32(CROCKFORD_32);
        let ts_d = b.input(0);
        let ent_hi = b.input(1);
        let ent_lo = b.input(2);
        let ts = b.scan_add(ts_d, ts_seed);

        let pack_8_chars = |vals_5bit: [V; 8]| -> V {
            let mut acc = vals_5bit[0];
            for (k, &v) in vals_5bit.iter().enumerate().skip(1) {
                acc = b.rr(Op::Or, acc, b.ri(Op::ShlI, v, (k * 8) as i64));
            }
            b.byte_lut32(acc, lut)
        };

        let c0_7 = pack_8_chars([
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 45), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 40), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 35), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 30), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 25), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 20), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 15), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 10), 31),
        ]);
        let c8_15 = pack_8_chars([
            b.ri(Op::AndI, b.ri(Op::ShrI, ts, 5), 31),
            b.ri(Op::AndI, ts, 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 59), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 54), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 49), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 44), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 39), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 34), 31),
        ]);
        let mid_boundary = b.rr(
            Op::Or,
            b.ri(Op::ShlI, b.ri(Op::AndI, ent_hi, 0x0f), 1),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_lo, 15), 1),
        );
        let c16_23 = pack_8_chars([
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 29), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 24), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 19), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 14), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 9), 31),
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_hi, 4), 31),
            mid_boundary,
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_lo, 10), 31),
        ]);
        let c24_25_raw = b.rr(
            Op::Or,
            b.ri(Op::AndI, b.ri(Op::ShrI, ent_lo, 5), 31),
            b.ri(Op::ShlI, b.ri(Op::AndI, ent_lo, 31), 8),
        );
        let c24_25 = b.byte_lut32(c24_25_raw, lut);
        b.finish(&[(c0_7, 8), (c8_15, 8), (c16_23, 8), (c24_25, 2)])
    }
}

impl CodecPack for UlidPack {
    fn name(&self) -> &'static str {
        "pack/ulid"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 4 && p.any_base32_26
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        let inv = Self::inv_table();
        let mut ts_vals = Vec::with_capacity(rows.len());
        let mut ent_hi_bytes = Vec::with_capacity(rows.len() * 8);
        let mut ent_lo_bytes = Vec::with_capacity(rows.len() * 2);
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut last = (0i64, 0u64, 0u16);

        for (i, &r) in rows.iter().enumerate() {
            if let Some((t, eh, el)) = Self::parse_ulid(r, &inv) {
                ts_vals.push(t);
                ent_hi_bytes.extend_from_slice(&eh.to_le_bytes());
                ent_lo_bytes.extend_from_slice(&el.to_le_bytes());
                last = (t, eh, el);
            } else {
                exc_idx.push(i);
                exc_vals.push(r);
                ts_vals.push(last.0);
                ent_hi_bytes.extend_from_slice(&last.1.to_le_bytes());
                ent_lo_bytes.extend_from_slice(&last.2.to_le_bytes());
                if exc_idx.len() * 8 > rows.len() {
                    return None;
                }
            }
        }
        let mut ts_d = Vec::with_capacity(rows.len());
        ts_d.push(0);
        for w in ts_vals.windows(2) {
            ts_d.push(w[1] - w[0]);
        }
        let mut gb = GraphBuilder::new(rows.len());
        let ts_node = gb.add_ints(&ts_d);
        let eh_buf = gb.add_bytes(&ent_hi_bytes);
        let eh_col = gb.add_byte_chunks(eh_buf, 8);
        let el_buf = gb.add_bytes(&ent_lo_bytes);
        let el_col = gb.add_byte_chunks(el_buf, 2);
        let cells = gb.add_prog_cells(
            Self::prog_ulid(ts_vals[0]),
            vec![ts_node, eh_col, el_col],
            vec![],
        );
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

/// Pack 9: IPv4 addresses (`a.b.c.d`) + optional `:port` or `/cidr`.
pub struct Ipv4Pack;

impl Ipv4Pack {
    fn parse_octet(s: &[u8]) -> Option<u8> {
        if s.is_empty() || s.len() > 3 || (s.len() > 1 && s[0] == b'0') {
            return None;
        }
        let mut acc: u16 = 0;
        for &b in s {
            if !b.is_ascii_digit() {
                return None;
            }
            acc = acc * 10 + (b - b'0') as u16;
        }
        u8::try_from(acc).ok()
    }

    fn parse_ipv4(s: &[u8]) -> Option<([u8; 4], i64)> {
        if s.len() < 7 || s.len() > 21 {
            return None;
        }
        let (ip_part, suffix_code) = if let Some(p) = s.iter().position(|&b| b == b':' || b == b'/')
        {
            let delim = s[p];
            let num_s = &s[p + 1..];
            let val = ExactIntPack::parse_canonical_i64(num_s)?;
            if !(0..=65535).contains(&val) {
                return None;
            }
            let code = if delim == b':' { val + 1 } else { 65537 + val };
            (&s[..p], code)
        } else {
            (s, 0)
        };
        let mut parts = ip_part.split(|&b| b == b'.');
        let o0 = Self::parse_octet(parts.next()?)?;
        let o1 = Self::parse_octet(parts.next()?)?;
        let o2 = Self::parse_octet(parts.next()?)?;
        let o3 = Self::parse_octet(parts.next()?)?;
        if parts.next().is_some() {
            return None;
        }
        Some(([o0, o1, o2, o3], suffix_code))
    }

    fn emit_u16_decimal(b: &Builder, val: V, zero: V) -> (V, V) {
        let (_, asc8) = eight_digits_raw_and_ascii(b, val);
        let raw8 = b.ri(Op::AddI, asc8, -(0x3030_3030_3030_3030u64 as i64));
        let z_bytes = b.ri(Op::ShrI, b.r(Op::Ctz, raw8), 3);
        let eight = b.ri(Op::AddI, zero, 8);
        let one = b.ri(Op::AddI, zero, 1);
        let len_nz = b.rr(Op::Sub, eight, z_bytes);
        let len = b.select(val, len_nz, one);
        let z_clamped = b.select(val, z_bytes, b.ri(Op::AddI, zero, 7));
        let shifted = b.rr(Op::Shr, asc8, b.ri(Op::ShlI, z_clamped, 3));
        (shifted, len)
    }

    fn prog_ipv4(has_suffix: bool) -> Program {
        let b = Builder::new(if has_suffix { 2 } else { 1 });
        let ip = b.input(0);
        let zero = b.ri(Op::AndI, ip, 0);
        let dot = b.ri(Op::AddI, zero, b'.' as i64);

        let o0 = b.ri(Op::AndI, ip, 0xff);
        let o1 = b.ri(Op::AndI, b.ri(Op::ShrI, ip, 8), 0xff);
        let o2 = b.ri(Op::AndI, b.ri(Op::ShrI, ip, 16), 0xff);
        let o3 = b.ri(Op::AndI, b.ri(Op::ShrI, ip, 24), 0xff);

        let (b0, l0) = Self::emit_u16_decimal(&b, o0, zero);
        let (b1, l1) = Self::emit_u16_decimal(&b, o1, zero);
        let (b2, l2) = Self::emit_u16_decimal(&b, o2, zero);
        let (b3, l3) = Self::emit_u16_decimal(&b, o3, zero);

        if !has_suffix {
            return b.finish_sinks(&[
                (b0, Len::Lane(l0.0)),
                (dot, Len::Fixed(1)),
                (b1, Len::Lane(l1.0)),
                (dot, Len::Fixed(1)),
                (b2, Len::Lane(l2.0)),
                (dot, Len::Fixed(1)),
                (b3, Len::Lane(l3.0)),
            ]);
        }

        let suf = b.input(1);
        let one = b.ri(Op::AddI, zero, 1);
        let has_s = b.select(suf, one, zero);
        let is_slash = b.ri(Op::ShrI, b.ri(Op::AddI, suf, -65537), 63);
        let is_colon = is_slash;
        let delim = b.select(
            is_colon,
            b.ri(Op::AddI, zero, b':' as i64),
            b.ri(Op::AddI, zero, b'/' as i64),
        );
        let num = b.select(
            is_colon,
            b.ri(Op::AddI, suf, -1),
            b.ri(Op::AddI, suf, -65537),
        );
        let (bs, ls_raw) = Self::emit_u16_decimal(&b, b.ri(Op::AndI, num, 0xffff), zero);
        let ls = b.select(suf, ls_raw, zero);

        b.finish_sinks(&[
            (b0, Len::Lane(l0.0)),
            (dot, Len::Fixed(1)),
            (b1, Len::Lane(l1.0)),
            (dot, Len::Fixed(1)),
            (b2, Len::Lane(l2.0)),
            (dot, Len::Fixed(1)),
            (b3, Len::Lane(l3.0)),
            (delim, Len::Lane(has_s.0)),
            (bs, Len::Lane(ls.0)),
        ])
    }
}

impl CodecPack for Ipv4Pack {
    fn name(&self) -> &'static str {
        "pack/ipv4"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 4 && p.all_ascii && p.any_ipv4_dot && p.min_len >= 7 && p.max_len <= 21
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        let mut ip_bytes = Vec::with_capacity(rows.len() * 4);
        let mut suffixes = Vec::with_capacity(rows.len());
        let mut any_suffix = false;
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut last = ([0u8; 4], 0i64);

        for (i, &r) in rows.iter().enumerate() {
            if let Some((ip, suf)) = Self::parse_ipv4(r) {
                ip_bytes.extend_from_slice(&ip);
                suffixes.push(suf);
                if suf != 0 {
                    any_suffix = true;
                }
                last = (ip, suf);
            } else {
                exc_idx.push(i);
                exc_vals.push(r);
                ip_bytes.extend_from_slice(&last.0);
                suffixes.push(last.1);
                if exc_idx.len() * 8 > rows.len() {
                    return None;
                }
            }
        }
        if exc_idx.len() == rows.len() {
            return None;
        }
        let mut gb = GraphBuilder::new(rows.len());
        let ip_buf = gb.add_mtf_fixed(&ip_bytes, 4);
        let ip_col = gb.add_byte_chunks(ip_buf, 4);
        let inputs = if any_suffix {
            let suf_col = gb.add_ints(&suffixes);
            vec![ip_col, suf_col]
        } else {
            vec![ip_col]
        };
        let cells = gb.add_prog_cells(Self::prog_ipv4(any_suffix), inputs, vec![]);
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

// ============================================================================
// 5. Built-in Codec Packs (10..16)
// ============================================================================

/// Pack 10: Fixed-width hexadecimal IDs (`16`, `32`, `40`, or `64` chars), with MT19937 stream support.
pub struct HexFixedPack;

impl HexFixedPack {
    fn parse_hex_row(s: &[u8], expected_chars: usize) -> Option<(Vec<u8>, bool)> {
        if s.len() != expected_chars || !expected_chars.is_multiple_of(2) {
            return None;
        }
        let mut has_lo = false;
        let mut has_up = false;
        let mut out = Vec::with_capacity(expected_chars / 2);
        for pair in s.chunks_exact(2) {
            let c0 = pair[0];
            let c1 = pair[1];
            if c0.is_ascii_lowercase() || c1.is_ascii_lowercase() {
                has_lo = true;
            }
            if c0.is_ascii_uppercase() || c1.is_ascii_uppercase() {
                has_up = true;
            }
            if has_lo && has_up {
                return None;
            }
            out.push((hex_val(c0)? << 4) | hex_val(c1)?);
        }
        Some((out, has_up))
    }

    fn prog_hex_from_buf(raw_bytes: usize, upper: bool) -> Program {
        let b = Builder::new(1);
        let buf = b.add_buf();
        let row_idx = b.input(0);
        let base_off = b.ri(Op::MulI, row_idx, raw_bytes as i64);

        let full_8 = raw_bytes / 8;
        let rem_4 = (raw_bytes % 8) / 4;
        let mut sinks = Vec::new();
        for k in 0..full_8 {
            let chunk = b.gather(buf, 3, base_off, (k * 8) as u64);
            let (lo, hi) = emit_hex_64_sinks(&b, chunk, 16, upper);
            sinks.push((lo, 8u8));
            sinks.push((hi, 8u8));
        }
        if rem_4 == 1 {
            let chunk = b.gather(buf, 2, base_off, (full_8 * 8) as u64);
            let (lo, _) = emit_hex_64_sinks(&b, chunk, 8, upper);
            sinks.push((lo, 8u8));
        }
        b.finish(&sinks)
    }
}

impl CodecPack for HexFixedPack {
    fn name(&self) -> &'static str {
        "pack/hex-fixed"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 4 && p.any_hex_even_ge16
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        let mut counts = [0usize; 65];
        for &r in rows.iter().take(64) {
            if matches!(r.len(), 16 | 32 | 40 | 64) {
                counts[r.len()] += 1;
            }
        }
        let target_chars = [16usize, 32, 40, 64]
            .into_iter()
            .max_by_key(|&w| counts[w])?;
        if counts[target_chars] == 0 {
            return None;
        }
        let raw_w = target_chars / 2;

        let mut flat = Vec::with_capacity(rows.len() * raw_w);
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut target_up: Option<bool> = None;
        let mut last = vec![0u8; raw_w];

        for (i, &r) in rows.iter().enumerate() {
            if let Some((bytes, up)) = Self::parse_hex_row(r, target_chars) {
                let u_ref = *target_up.get_or_insert(up);
                if up == u_ref {
                    flat.extend_from_slice(&bytes);
                    last = bytes;
                    continue;
                }
            }
            exc_idx.push(i);
            exc_vals.push(r);
            flat.extend_from_slice(&last);
            if exc_idx.len() * 8 > rows.len() {
                return None;
            }
        }
        let upper = target_up?;
        let mut gb = GraphBuilder::new(rows.len());
        let raw_buf = if raw_w.is_multiple_of(4) {
            try_build_mt_cells(&mut gb, &flat, raw_w, &[MtLayout::Le, MtLayout::Be])
                .unwrap_or_else(|| gb.add_mtf_fixed(&flat, raw_w as u8))
        } else {
            gb.add_mtf_fixed(&flat, raw_w as u8)
        };
        let row_idx = gb.add_row_index(rows.len());
        let cells = gb.add_prog_cells(
            Self::prog_hex_from_buf(raw_w, upper),
            vec![row_idx],
            vec![raw_buf],
        );
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

/// Pack 11: W3C `traceparent` (`00-<32hex>-<16hex>-<2hex>`, 55 bytes), with MT19937 stream support.
pub struct TraceparentPack;

impl TraceparentPack {
    fn parse_tp(s: &[u8]) -> Option<([u8; 16], [u8; 8], u8)> {
        if s.len() != 55 || &s[0..3] != b"00-" || s[35] != b'-' || s[52] != b'-' {
            return None;
        }
        let (tr, up1) = HexFixedPack::parse_hex_row(&s[3..35], 32)?;
        let (sp, up2) = HexFixedPack::parse_hex_row(&s[36..52], 16)?;
        let (fl, up3) = HexFixedPack::parse_hex_row(&s[53..55], 2)?;
        if up1 || up2 || up3 {
            return None;
        }
        Some((tr.try_into().unwrap(), sp.try_into().unwrap(), fl[0]))
    }

    fn prog_traceparent() -> Program {
        let b = Builder::new(2);
        let tr_buf = b.add_buf();
        let sp_buf = b.add_buf();
        let row_idx = b.input(0);
        let flags = b.input(1);

        let tr_off = b.ri(Op::ShlI, row_idx, 4);
        let sp_off = b.ri(Op::ShlI, row_idx, 3);
        let tr0 = b.gather(tr_buf, 3, tr_off, 0);
        let tr1 = b.gather(tr_buf, 3, tr_off, 8);
        let sp = b.gather(sp_buf, 3, sp_off, 0);

        let (t0, t1) = emit_hex_64_sinks(&b, tr0, 16, false);
        let (t2, t3) = emit_hex_64_sinks(&b, tr1, 16, false);
        let (s0, s1) = emit_hex_64_sinks(&b, sp, 16, false);
        let (fl_hex, _) = emit_hex_64_sinks(&b, flags, 8, false);

        let zero = b.ri(Op::AndI, flags, 0);
        let pfx = b.ri(Op::OrI, zero, u32::from_le_bytes(*b"00-\0") as i64);
        let dash = b.ri(Op::OrI, zero, b'-' as i64);

        b.finish(&[
            (pfx, 3),
            (t0, 8),
            (t1, 8),
            (t2, 8),
            (t3, 8),
            (dash, 1),
            (s0, 8),
            (s1, 8),
            (dash, 1),
            (fl_hex, 2),
        ])
    }
}

impl CodecPack for TraceparentPack {
    fn name(&self) -> &'static str {
        "pack/traceparent"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 4 && p.any_traceparent_55
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        let mut traces = Vec::with_capacity(rows.len() * 16);
        let mut spans = Vec::with_capacity(rows.len() * 8);
        let mut flags = Vec::with_capacity(rows.len());
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut last = ([0u8; 16], [0u8; 8], 1u8);

        for (i, &r) in rows.iter().enumerate() {
            if let Some((tr, sp, fl)) = Self::parse_tp(r) {
                traces.extend_from_slice(&tr);
                spans.extend_from_slice(&sp);
                flags.push(fl as i64);
                last = (tr, sp, fl);
            } else {
                exc_idx.push(i);
                exc_vals.push(r);
                traces.extend_from_slice(&last.0);
                spans.extend_from_slice(&last.1);
                flags.push(last.2 as i64);
                if exc_idx.len() * 8 > rows.len() {
                    return None;
                }
            }
        }
        let mut gb = GraphBuilder::new(rows.len());
        let tr_buf = try_build_mt_cells(&mut gb, &traces, 16, &[MtLayout::Le, MtLayout::Be])
            .unwrap_or_else(|| gb.add_mtf_fixed(&traces, 16));
        let sp_buf = try_build_mt_cells(&mut gb, &spans, 8, &[MtLayout::Le, MtLayout::Be])
            .unwrap_or_else(|| gb.add_mtf_fixed(&spans, 8));
        let row_idx = gb.add_row_index(rows.len());
        let fl_col = gb.add_ints(&flags);
        let cells = gb.add_prog_cells(
            Self::prog_traceparent(),
            vec![row_idx, fl_col],
            vec![tr_buf, sp_buf],
        );
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

/// Pack 12: Structured string splitter for Email (`local@domain`) and Hierarchical paths/FQDNs.
pub struct StructuredSplitPack;

impl StructuredSplitPack {
    fn try_split_on(rows: &[&[u8]], delim: u8, last_occ: bool) -> Option<PayloadGraph> {
        let mut prefixes = Vec::with_capacity(rows.len());
        let mut suffixes = Vec::with_capacity(rows.len());
        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();
        let mut hits = 0usize;

        for (i, &r) in rows.iter().enumerate() {
            let pos = if last_occ {
                r.iter().rposition(|&b| b == delim)
            } else {
                r.iter().position(|&b| b == delim)
            };
            if let Some(p) = pos
                && p >= 1
                && p + 1 < r.len()
            {
                prefixes.push(&r[..p + 1]);
                suffixes.push(&r[p + 1..]);
                hits += 1;
            } else {
                prefixes.push(&b""[..]);
                suffixes.push(&b""[..]);
                exc_idx.push(i);
                exc_vals.push(r);
                if exc_idx.len() * 4 > rows.len() {
                    return None;
                }
            }
        }
        if hits * 4 < rows.len() * 3 {
            return None;
        }
        let p_uniq: FastSet<&[u8]> = prefixes.iter().copied().collect();
        let s_uniq: FastSet<&[u8]> = suffixes.iter().copied().collect();
        if p_uniq.len() * 2 > rows.len() && s_uniq.len() * 2 > rows.len() {
            return None;
        }

        let mut gb = GraphBuilder::new(rows.len());
        let p_node = gb.add_dict_cells(&prefixes);
        let s_node = if let Some(sub_g) = ExactIntPack.try_encode(&suffixes) {
            gb.merge_graph(&sub_g)
        } else {
            gb.add_dict_cells(&suffixes)
        };
        let cat = gb.add_concat(vec![p_node, s_node]);
        let root = gb.add_patch(cat, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

impl CodecPack for StructuredSplitPack {
    fn name(&self) -> &'static str {
        "pack/structured-split"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 8 && (p.any_at || p.any_delim) && p.max_len <= 512
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 8 {
            return None;
        }
        let mut best: Option<(usize, PayloadGraph)> = None;
        for (delim, last_occ) in [(b'@', false), (b'/', true), (b':', true), (b'.', false)] {
            let probe = rows.len().min(16);
            let hits = rows[..probe].iter().filter(|r| r.contains(&delim)).count();
            if hits * 2 < probe {
                continue;
            }
            if let Some(g) = Self::try_split_on(rows, delim, last_occ) {
                let len = g.cost();
                if best.as_ref().is_none_or(|(bl, _)| len < *bl) {
                    best = Some((len, g));
                }
            }
        }
        best.map(|(_, g)| g)
    }
}

/// Pack 13: Structural & Numeric String Template Pack (`blk_-1608999687919862906`, CloudTrail ARNs, etc.).
pub struct NumericTemplatePack;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Int(i64),
    Hex(usize),
}

impl NumericTemplatePack {
    fn extract_skeleton(
        s: &[u8],
        skel: &mut Vec<u8>,
        slots: &mut Vec<(usize, usize, TokenKind)>,
    ) -> bool {
        skel.clear();
        slots.clear();
        if s.is_empty() || s.len() > 256 {
            return false;
        }
        let mut i = 0usize;
        while i < s.len() {
            if slots.len() < 6 {
                if s[i] == b'-' && i + 1 < s.len() && (b'1'..=b'9').contains(&s[i + 1]) {
                    let prev_ok = i == 0 || !s[i - 1].is_ascii_alphanumeric();
                    if prev_ok {
                        let start = i;
                        let mut j = i + 1;
                        while j < s.len() && s[j].is_ascii_digit() {
                            j += 1;
                        }
                        if (2..=20).contains(&(j - start))
                            && let Some(val) = ExactIntPack::parse_canonical_i64(&s[start..j])
                        {
                            skel.push(0);
                            slots.push((start, j, TokenKind::Int(val)));
                            i = j;
                            continue;
                        }
                    }
                }
                if s[i].is_ascii_hexdigit() {
                    let start = i;
                    let mut j = i;
                    let mut all_dec = true;
                    let mut has_lo = false;
                    let mut has_up = false;
                    while j < s.len() && s[j].is_ascii_hexdigit() {
                        if !s[j].is_ascii_digit() {
                            all_dec = false;
                        }
                        if s[j].is_ascii_lowercase() {
                            has_lo = true;
                        }
                        if s[j].is_ascii_uppercase() {
                            has_up = true;
                        }
                        j += 1;
                    }
                    let tok_len = j - start;
                    let left_boundary = start == 0 || !s[start - 1].is_ascii_alphanumeric();
                    let right_boundary = j == s.len() || !s[j].is_ascii_alphanumeric();
                    if left_boundary && right_boundary {
                        if !(all_dec || has_lo && has_up) && matches!(tok_len, 16 | 32 | 40 | 64) {
                            skel.push(1);
                            skel.push(tok_len as u8);
                            slots.push((start, j, TokenKind::Hex(tok_len)));
                            i = j;
                            continue;
                        }
                        if all_dec {
                            let canonical = tok_len == 1 || s[start] != b'0';
                            if canonical
                                && tok_len <= 19
                                && let Some(val) = ExactIntPack::parse_canonical_i64(&s[start..j])
                            {
                                skel.push(0);
                                slots.push((start, j, TokenKind::Int(val)));
                                i = j;
                                continue;
                            }
                        }
                    }
                }
            }
            // Bytes 0 and 1 mark slots in the skeleton.
            if s[i] < 2 {
                return false;
            }
            skel.push(s[i]);
            i += 1;
        }
        !slots.is_empty() && slots.len() <= s.len() / 2
    }
}

impl CodecPack for NumericTemplatePack {
    fn name(&self) -> &'static str {
        "pack/numeric-template"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 8 && p.all_ascii && p.min_len >= 4 && p.max_len <= 256
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 8 {
            return None;
        }
        let probe = rows.len().min(16);
        let mut skel_counts: FastMap<Vec<u8>, usize> = fast_map_with_capacity(probe);
        let mut tmp_skel = Vec::with_capacity(64);
        let mut tmp_slots = Vec::with_capacity(6);
        for &r in &rows[..probe] {
            if Self::extract_skeleton(r, &mut tmp_skel, &mut tmp_slots) {
                *skel_counts.entry(tmp_skel.clone()).or_insert(0) += 1;
            }
        }
        let (dom_skel, &dom_hits) = skel_counts.iter().max_by_key(|(_, c)| *c)?;
        if dom_hits * 4 < probe * 3 {
            return None;
        }
        let dom_skel = dom_skel.clone();

        let mut first_row_idx = None;
        let mut first_slots = Vec::new();
        for (i, &r) in rows.iter().take(probe).enumerate() {
            if Self::extract_skeleton(r, &mut tmp_skel, &mut tmp_slots) && tmp_skel == dom_skel {
                first_row_idx = Some(i);
                first_slots = tmp_slots.clone();
                break;
            }
        }
        let first_r = rows[first_row_idx?];
        let n_slots = first_slots.len();
        let mut literals: Vec<Vec<u8>> = Vec::with_capacity(n_slots + 1);
        let mut prev_end = 0usize;
        for &(st, en, _) in &first_slots {
            literals.push(first_r[prev_end..st].to_vec());
            prev_end = en;
        }
        literals.push(first_r[prev_end..].to_vec());
        if literals.iter().all(|l| l.is_empty()) {
            return None;
        }

        let mut slot_strs: Vec<Vec<&[u8]>> = vec![Vec::with_capacity(rows.len()); n_slots];
        let mut slot_ints: Vec<Vec<i64>> = vec![Vec::with_capacity(rows.len()); n_slots];
        let mut slot_is_int: Vec<bool> = first_slots
            .iter()
            .map(|s| matches!(s.2, TokenKind::Int(_)))
            .collect();
        let mut last_strs: Vec<&[u8]> = first_slots
            .iter()
            .map(|&(st, en, _)| &first_r[st..en])
            .collect();
        let mut last_ints: Vec<i64> = first_slots
            .iter()
            .map(|s| match s.2 {
                TokenKind::Int(v) => v,
                TokenKind::Hex(_) => 0,
            })
            .collect();

        let mut exc_idx = Vec::new();
        let mut exc_vals = Vec::new();

        for (i, &r) in rows.iter().enumerate() {
            if Self::extract_skeleton(r, &mut tmp_skel, &mut tmp_slots)
                && tmp_skel == dom_skel
                && tmp_slots.len() == n_slots
            {
                for k in 0..n_slots {
                    let (st, en, tk) = tmp_slots[k];
                    let sl = &r[st..en];
                    slot_strs[k].push(sl);
                    last_strs[k] = sl;
                    match tk {
                        TokenKind::Int(v) => {
                            slot_ints[k].push(v);
                            last_ints[k] = v;
                        }
                        TokenKind::Hex(_) => {
                            slot_is_int[k] = false;
                            slot_ints[k].push(0);
                        }
                    }
                }
            } else {
                exc_idx.push(i);
                exc_vals.push(r);
                for k in 0..n_slots {
                    slot_strs[k].push(last_strs[k]);
                    slot_ints[k].push(last_ints[k]);
                }
                if exc_idx.len() * 5 > rows.len() {
                    return None;
                }
            }
        }
        if exc_idx.len() == rows.len() {
            return None;
        }

        let mut gb = GraphBuilder::new(rows.len());
        let mut parts: Vec<u16> = Vec::with_capacity(2 * n_slots + 1);

        for k in 0..n_slots {
            if !literals[k].is_empty() {
                parts.push(gb.add_const_cells(&literals[k], rows.len()));
            }
            let srows = &slot_strs[k];
            let s_first = srows[0];
            if srows.iter().all(|&x| x == s_first) {
                parts.push(gb.add_const_cells(s_first, rows.len()));
                continue;
            }

            let mut best_slot_g: Option<(usize, PayloadGraph)> = {
                let mut sgb = GraphBuilder::new(rows.len());
                let r = sgb.add_dict_cells(srows);
                let g = sgb.finish(r);
                Some((g.cost(), g))
            };
            if slot_is_int[k] {
                let vals = &slot_ints[k];
                let mut deltas = Vec::with_capacity(vals.len());
                deltas.push(0);
                for w in vals.windows(2) {
                    deltas.push(w[1].wrapping_sub(w[0]));
                }
                let mut sgb_del = GraphBuilder::new(rows.len());
                let dn = sgb_del.add_ints(&deltas);
                let dc = sgb_del.add_prog_cells(prog_fmt_int(true, vals[0]), vec![dn], vec![]);
                let g_del = sgb_del.finish(dc);
                let len_del = g_del.cost();
                if best_slot_g.as_ref().is_none_or(|(bl, _)| len_del < *bl) {
                    best_slot_g = Some((len_del, g_del));
                }

                let mut sgb_mtf = GraphBuilder::new(rows.len());
                let mn = sgb_mtf.add_mtf_ints(vals, 4096);
                let mc = sgb_mtf.add_prog_cells(prog_fmt_int(false, 0), vec![mn], vec![]);
                let g_mtf = sgb_mtf.finish(mc);
                let len_mtf = g_mtf.cost();
                if best_slot_g.as_ref().is_none_or(|(bl, _)| len_mtf < *bl) {
                    best_slot_g = Some((len_mtf, g_mtf));
                }

                if vals.len() >= 8
                    && vals.iter().take(4).any(|v| v.unsigned_abs() > (1 << 40))
                    && let Some(g_lcg) = JavaLcgPack.try_encode(srows)
                {
                    let len_lcg = g_lcg.cost();
                    if best_slot_g.as_ref().is_none_or(|(bl, _)| len_lcg < *bl) {
                        best_slot_g = Some((len_lcg, g_lcg));
                    }
                }
            } else if let Some(g_hex) = HexFixedPack.try_encode(srows) {
                let len_hex = g_hex.cost();
                if best_slot_g.as_ref().is_none_or(|(bl, _)| len_hex < *bl) {
                    best_slot_g = Some((len_hex, g_hex));
                }
            }
            let (_, slot_g) = best_slot_g?;
            parts.push(gb.merge_graph(&slot_g));
        }
        if !literals[n_slots].is_empty() {
            parts.push(gb.add_const_cells(&literals[n_slots], rows.len()));
        }
        let cat = if parts.len() == 1 {
            parts[0]
        } else {
            gb.add_concat(parts)
        };
        let root = gb.add_patch(cat, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

/// Pack 14: Base64 (standard & URL-safe) binary unpacking, deduplication, and MT19937 stream detection.
pub struct Base64Pack;

const B64_STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn prog_base64_quads(url_safe: bool, has_padding: bool) -> Program {
    let alpha = if url_safe { B64_URL } else { B64_STD };
    let mut lo32 = [0u8; 32];
    let mut hi32 = [0u8; 32];
    lo32.copy_from_slice(&alpha[..32]);
    hi32.copy_from_slice(&alpha[32..64]);

    let b = Builder::new(if has_padding { 2 } else { 1 });
    let lut_lo = b.add_lut32(&lo32);
    let lut_hi = b.add_lut32(&hi32);

    let t = b.input(0);
    let b0 = b.ri(Op::AndI, t, 0xff);
    let b1 = b.ri(Op::AndI, b.ri(Op::ShrI, t, 8), 0xff);
    let b2 = b.ri(Op::AndI, b.ri(Op::ShrI, t, 16), 0xff);

    let s0 = b.ri(Op::ShrI, b0, 2);
    let s1 = b.rr(
        Op::Or,
        b.ri(Op::ShlI, b.ri(Op::AndI, b0, 0x03), 4),
        b.ri(Op::ShrI, b1, 4),
    );
    let s2 = b.rr(
        Op::Or,
        b.ri(Op::ShlI, b.ri(Op::AndI, b1, 0x0f), 2),
        b.ri(Op::ShrI, b2, 6),
    );
    let s3 = b.ri(Op::AndI, b2, 0x3f);

    let sextets = b.rr(
        Op::Or,
        b.rr(Op::Or, s0, b.ri(Op::ShlI, s1, 8)),
        b.rr(Op::Or, b.ri(Op::ShlI, s2, 16), b.ri(Op::ShlI, s3, 24)),
    );
    let c_lo = b.byte_lut32(sextets, lut_lo);
    let c_hi = b.byte_lut32(sextets, lut_hi);

    let bit5 = b.ri(Op::AndI, b.ri(Op::ShrI, sextets, 5), 0x0101_0101);
    let mask = b.ri(Op::MulI, bit5, 0xff);
    let chars = b.rr(
        Op::Or,
        b.rr(Op::And, c_hi, mask),
        b.rr(Op::And, c_lo, b.ri(Op::XorI, mask, 0xffff_ffffu32 as i64)),
    );

    if !has_padding {
        return b.finish(&[(chars, 4)]);
    }
    let pad = b.input(1);
    let pad1 = b.ri(
        Op::OrI,
        b.ri(Op::AndI, chars, 0x00ff_ffff),
        (b'=' as i64) << 24,
    );
    let pad2 = b.ri(Op::OrI, b.ri(Op::AndI, chars, 0x0000_ffff), 0x3d3d_0000);
    let is_pad2 = b.ri(Op::ShrI, pad, 1);
    let padded = b.select(is_pad2, pad2, pad1);
    let out = b.select(pad, padded, chars);
    b.finish(&[(out, 4)])
}

impl Base64Pack {
    fn build_inv_table(url_safe: bool) -> [u8; 256] {
        let mut inv = [255u8; 256];
        let alpha = if url_safe { B64_URL } else { B64_STD };
        for (i, &c) in alpha.iter().enumerate() {
            inv[c as usize] = i as u8;
        }
        inv
    }

    fn decode_cell(s: &[u8], inv: &[u8; 256], triplets: &mut Vec<u8>, pads: &mut Vec<i64>) -> bool {
        if s.len() < 16 || !s.len().is_multiple_of(4) {
            return false;
        }
        let n_quads = s.len() / 4;
        let orig_t = triplets.len();
        let orig_p = pads.len();
        for (qi, q) in s.chunks_exact(4).enumerate() {
            let is_last = qi + 1 == n_quads;
            let s0 = inv[q[0] as usize];
            let s1 = inv[q[1] as usize];
            if s0 == 255 || s1 == 255 {
                triplets.truncate(orig_t);
                pads.truncate(orig_p);
                return false;
            }
            if is_last && q[2] == b'=' && q[3] == b'=' {
                if (s1 & 0x0f) != 0 {
                    triplets.truncate(orig_t);
                    pads.truncate(orig_p);
                    return false;
                }
                triplets.extend_from_slice(&[(s0 << 2) | (s1 >> 4), 0, 0]);
                pads.push(2);
            } else if is_last && q[3] == b'=' {
                let s2 = inv[q[2] as usize];
                if s2 == 255 || (s2 & 0x03) != 0 {
                    triplets.truncate(orig_t);
                    pads.truncate(orig_p);
                    return false;
                }
                triplets.extend_from_slice(&[
                    (s0 << 2) | (s1 >> 4),
                    ((s1 & 0x0f) << 4) | (s2 >> 2),
                    0,
                ]);
                pads.push(1);
            } else {
                let s2 = inv[q[2] as usize];
                let s3 = inv[q[3] as usize];
                if s2 == 255 || s3 == 255 {
                    triplets.truncate(orig_t);
                    pads.truncate(orig_p);
                    return false;
                }
                triplets.extend_from_slice(&[
                    (s0 << 2) | (s1 >> 4),
                    ((s1 & 0x0f) << 4) | (s2 >> 2),
                    ((s2 & 0x03) << 6) | s3,
                ]);
                pads.push(0);
            }
        }
        true
    }
}

impl CodecPack for Base64Pack {
    fn name(&self) -> &'static str {
        "pack/base64"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 4 && p.any_mult4_ge16
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.len() < 4 {
            return None;
        }
        let probe_ok = rows
            .iter()
            .take(8)
            .filter(|r| r.len() >= 16 && r.len().is_multiple_of(4))
            .count();
        if probe_ok * 2 < rows.len().min(8) {
            return None;
        }
        let url_safe = rows
            .iter()
            .take(16)
            .any(|r| r.contains(&b'-') || r.contains(&b'_'));
        let inv = Self::build_inv_table(url_safe);
        {
            let mut tmp_t = Vec::new();
            let mut tmp_p = Vec::new();
            let ok0 = Self::decode_cell(rows[0], &inv, &mut tmp_t, &mut tmp_p);
            let ok1 = rows.len() > 1 && Self::decode_cell(rows[1], &inv, &mut tmp_t, &mut tmp_p);
            if !ok0 && !ok1 {
                return None;
            }
        }

        let mut uniq_cells: Vec<&[u8]> = Vec::new();
        let mut remap: FastMap<&[u8], i64> = fast_map_with_capacity(rows.len().min(256));
        let mut ords: Vec<i64> = Vec::with_capacity(rows.len());
        let use_dedup = {
            let mut aborted = false;
            for &r in rows {
                if let Some(&id) = remap.get(r) {
                    ords.push(id);
                } else {
                    if uniq_cells.len() * 2 >= rows.len() && rows.len() >= 32 {
                        aborted = true;
                        break;
                    }
                    let id = uniq_cells.len() as i64;
                    remap.insert(r, id);
                    uniq_cells.push(r);
                    ords.push(id);
                }
            }
            !aborted && uniq_cells.len() < rows.len()
        };

        let target_rows: &[&[u8]] = if use_dedup { &uniq_cells } else { rows };
        let mut triplets: Vec<u8> = Vec::new();
        let mut pads: Vec<i64> = Vec::new();
        let mut cell_lens: Vec<i64> = Vec::with_capacity(target_rows.len());
        let mut exc_idx: Vec<usize> = Vec::new();
        let mut exc_vals: Vec<&[u8]> = Vec::new();

        for (i, &r) in target_rows.iter().enumerate() {
            if Self::decode_cell(r, &inv, &mut triplets, &mut pads) {
                cell_lens.push(r.len() as i64);
            } else {
                exc_idx.push(i);
                exc_vals.push(r);
                cell_lens.push(0);
                if exc_idx.len() * 8 > target_rows.len() {
                    return None;
                }
            }
        }
        if pads.is_empty() {
            return None;
        }
        let any_pad = pads.iter().any(|&p| p != 0);
        let prog = prog_base64_quads(url_safe, any_pad);

        let mut gb = GraphBuilder::new(rows.len());
        // Check if decoded binary `triplets` come from a uniform-width MT19937 stream (e.g. 48B lines!).
        let uniform_raw_w = if exc_idx.is_empty()
            && !any_pad
            && !target_rows.is_empty()
            && triplets.len() == target_rows.len() * (triplets.len() / target_rows.len())
        {
            let rw = triplets.len() / target_rows.len();
            ((4..=64).contains(&rw) && rw.is_multiple_of(4)).then_some(rw)
        } else {
            None
        };
        let t_buf = uniform_raw_w
            .and_then(|rw| {
                try_build_mt_cells(&mut gb, &triplets, rw, &[MtLayout::Le, MtLayout::Be])
            })
            .unwrap_or_else(|| gb.add_bytes(&triplets));
        let t_chunks = gb.add_byte_chunks(t_buf, 3);
        let inputs = if any_pad {
            let p_node = gb.add_ints(&pads);
            vec![t_chunks, p_node]
        } else {
            vec![t_chunks]
        };
        let quads_cells = gb.add_prog_cells(prog, inputs, vec![]);
        let sliced = gb.add_slice_buf(quads_cells, 0, &cell_lens);
        let target_root = gb.add_patch(sliced, &exc_idx, &exc_vals);
        let root = if use_dedup {
            let ord_node = gb.add_mtf_ints(&ords, 4096);
            gb.add_index_cells(target_root, ord_node)
        } else {
            target_root
        };
        Some(gb.finish(root))
    }
}

/// Pack 15: Binary IEEE-754 `f64` cells (8B): plane transpose, ordered deltas, `ScanF64Add` walk, and MT19937 walk/draws.
pub struct F64BinaryPack;

impl F64BinaryPack {
    #[inline]
    fn to_ordered(bits: u64) -> u64 {
        if (bits as i64) < 0 {
            !bits
        } else {
            bits ^ (1 << 63)
        }
    }
}

impl CodecPack for F64BinaryPack {
    fn name(&self) -> &'static str {
        "pack/f64-binary"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.min_len == 8 && p.max_len == 8
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        if rows.is_empty() || rows.iter().any(|r| r.len() != 8) {
            return None;
        }
        let n = rows.len();
        let mut best: Option<(usize, PayloadGraph)> = None;

        // Candidate 0: MT19937 `Walk { a, b }` or `Le`/`Be` 8-byte draws!
        if n >= 8 {
            let mut flat = Vec::with_capacity(n * 8);
            for &r in rows {
                flat.extend_from_slice(r);
            }
            let mut layouts = Vec::with_capacity(3);
            if let Some(wlay) = MtLayout::detect_walk(&flat) {
                layouts.push(wlay);
            }
            layouts.push(MtLayout::Le);
            layouts.push(MtLayout::Be);
            let mut gb_mt = GraphBuilder::new(n);
            if let Some(mt_root) = try_build_mt_cells(&mut gb_mt, &flat, 8, &layouts) {
                let g_mt = gb_mt.finish(mt_root);
                let mt_len = g_mt.cost();
                if mt_len * 2 < flat.len() {
                    best = Some((mt_len, g_mt));
                }
            }
        }

        let bits: Vec<u64> = rows
            .iter()
            .map(|r| u64::from_le_bytes((*r).try_into().unwrap()))
            .collect();

        let mut steps_bits = Vec::with_capacity(n);
        let mut walk_exact = !f64::from_bits(bits[0]).is_nan();
        steps_bits.push(NEG_ZERO);
        if walk_exact {
            let mut acc = f64::from_bits(bits[0]);
            for &b in &bits[1..] {
                let target = f64::from_bits(b);
                let diff = target - acc;
                if (acc + diff).to_bits() != b {
                    walk_exact = false;
                    break;
                }
                acc = target;
                steps_bits.push(diff.to_bits() as i64);
            }
        }

        let ord: Vec<u64> = bits.iter().map(|&b| Self::to_ordered(b)).collect();
        let mut ord_deltas = Vec::with_capacity(n);
        ord_deltas.push(0i64);
        for w in ord.windows(2) {
            ord_deltas.push(w[1].wrapping_sub(w[0]) as i64);
        }

        if n >= 16 && best.as_ref().is_none_or(|(bl, _)| *bl > 256) {
            let mut raw_planes = vec![0u8; n * 8];
            for (i, &r) in rows.iter().enumerate() {
                for k in 0..8 {
                    raw_planes[k * n + i] = r[k];
                }
            }
            let mut gb_r = GraphBuilder::new(n);
            let r_buf = gb_r.add_bytes(&raw_planes);
            let unshuf_r = gb_r.add_byte_unshuffle(r_buf, 8);
            let r_root = gb_r.add_slice_buf(unshuf_r, 0, &vec![8i64; n]);
            let g_r = gb_r.finish(r_root);
            let r_len = g_r.cost();
            if best.as_ref().is_none_or(|(bl, _)| r_len < *bl) {
                best = Some((r_len, g_r));
            }
        }

        if best.as_ref().is_none_or(|(bl, _)| *bl > 256) {
            let ob = Builder::new(1);
            let od = ob.input(0);
            let o = ob.scan_add(od, ord[0] as i64);
            let pos_sign = ob.ri(Op::ShrI, o, 63);
            let pos_bits = ob.ri(Op::XorI, o, (1u64 << 63) as i64);
            let neg_bits = ob.ri(Op::XorI, o, -1);
            let raw_bits = ob.select(pos_sign, pos_bits, neg_bits);
            let ord_prog = ob.finish(&[(raw_bits, 8)]);

            let mut gb_ord = GraphBuilder::new(n);
            let dn = gb_ord.add_ints(&ord_deltas);
            let rn = gb_ord.add_prog_cells(ord_prog, vec![dn], vec![]);
            let g_ord = gb_ord.finish(rn);
            let o_len = g_ord.cost();
            if best.as_ref().is_none_or(|(bl, _)| o_len < *bl) {
                best = Some((o_len, g_ord));
            }
        }

        if n >= 16 && best.as_ref().is_none_or(|(bl, _)| *bl > 256) {
            let mut planes = vec![0u8; n * 8];
            for (i, &d) in ord_deltas.iter().enumerate() {
                let zz = ((d << 1) ^ (d >> 63)) as u64;
                let le = zz.to_le_bytes();
                for k in 0..8 {
                    planes[k * n + i] = le[k];
                }
            }
            let pb = Builder::new(1);
            let zz = pb.input(0);
            let half = pb.ri(Op::ShrI, zz, 1);
            let mask = pb.ri(Op::SarI, pb.ri(Op::ShlI, zz, 63), 63);
            let delta = pb.rr(Op::Xor, half, mask);
            let o = pb.scan_add(delta, ord[0] as i64);
            let pos_sign = pb.ri(Op::ShrI, o, 63);
            let pos_bits = pb.ri(Op::XorI, o, (1u64 << 63) as i64);
            let neg_bits = pb.ri(Op::XorI, o, -1);
            let raw_bits = pb.select(pos_sign, pos_bits, neg_bits);
            let plane_prog = pb.finish(&[(raw_bits, 8)]);

            let mut gb_p = GraphBuilder::new(n);
            let p_buf = gb_p.add_bytes(&planes);
            let unshuf = gb_p.add_byte_unshuffle(p_buf, 8);
            let zz_col = gb_p.add_byte_chunks(unshuf, 8);
            let rp = gb_p.add_prog_cells(plane_prog, vec![zz_col], vec![]);
            let g_p = gb_p.finish(rp);
            let p_len = g_p.cost();
            if best.as_ref().is_none_or(|(bl, _)| p_len < *bl) {
                best = Some((p_len, g_p));
            }
        }

        if walk_exact && best.as_ref().is_none_or(|(bl, _)| *bl > 256) {
            let wb = Builder::new(1);
            let st = wb.input(0);
            let out = wb.scan_f64_add(st, f64::from_bits(bits[0]));
            let walk_prog = wb.finish(&[(out, 8)]);
            let mut gb_w = GraphBuilder::new(n);
            let sn = gb_w.add_ints(&steps_bits);
            let rw = gb_w.add_prog_cells(walk_prog, vec![sn], vec![]);
            let g_w = gb_w.finish(rw);
            let w_len = g_w.cost();
            if best.as_ref().is_none_or(|(bl, _)| w_len < *bl) {
                best = Some((w_len, g_w));
            }
        }
        best.map(|(_, g)| g)
    }
}

/// Pack 16: Fixed-width binary struct / `f32` byte-plane transpose (`2` or `4` bytes per cell).
pub struct BinaryPlanesPack;

impl CodecPack for BinaryPlanesPack {
    fn name(&self) -> &'static str {
        "pack/binary-planes"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 16 && p.min_len == p.max_len && matches!(p.min_len, 2 | 4)
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        let n = rows.len();
        if n < 16 {
            return None;
        }
        let w = rows[0].len();
        if !matches!(w, 2 | 4) || rows.iter().any(|r| r.len() != w) {
            return None;
        }
        let mut planes = vec![0u8; n * w];
        for (i, &r) in rows.iter().enumerate() {
            for k in 0..w {
                planes[k * n + i] = r[k];
            }
        }
        let mut gb = GraphBuilder::new(n);
        let p_buf = gb.add_bytes(&planes);
        let unshuf = gb.add_byte_unshuffle(p_buf, w as u8);
        let root = gb.add_slice_buf(unshuf, 0, &vec![w as i64; n]);
        let mut best_g = gb.finish(root);

        if w == 4 {
            let ord32: Vec<u32> = rows
                .iter()
                .map(|r| {
                    let b = u32::from_le_bytes((*r).try_into().unwrap());
                    if (b as i32) < 0 { !b } else { b ^ 0x8000_0000 }
                })
                .collect();
            let mut f32_planes = vec![0u8; n * 4];
            for i in 0..n {
                let d = if i == 0 {
                    0i32
                } else {
                    ord32[i].wrapping_sub(ord32[i - 1]) as i32
                };
                let zz = ((d << 1) ^ (d >> 31)) as u32;
                let le = zz.to_le_bytes();
                for k in 0..4 {
                    f32_planes[k * n + i] = le[k];
                }
            }
            let fb = Builder::new(1);
            let zz = fb.input(0);
            let half = fb.ri(Op::ShrI, zz, 1);
            let mask = fb.ri(Op::SarI, fb.ri(Op::ShlI, zz, 63), 63);
            let delta = fb.rr(Op::Xor, half, mask);
            let o_full = fb.scan_add(delta, ord32[0] as i64);
            let o32 = fb.ri(Op::AndI, o_full, 0xffff_ffff);
            let pos_sign = fb.ri(Op::ShrI, o32, 31);
            let pos_bits = fb.ri(Op::XorI, o32, 0x8000_0000);
            let neg_bits = fb.ri(Op::XorI, o32, 0xffff_ffff);
            let raw32 = fb.select(pos_sign, pos_bits, neg_bits);
            let f32_prog = fb.finish(&[(raw32, 4)]);

            let mut gb_f = GraphBuilder::new(n);
            let f_buf = gb_f.add_bytes(&f32_planes);
            let unshuf_f = gb_f.add_byte_unshuffle(f_buf, 4);
            let zz_col = gb_f.add_byte_chunks(unshuf_f, 4);
            let f_root = gb_f.add_prog_cells(f32_prog, vec![zz_col], vec![]);
            let g_f = gb_f.finish(f_root);
            if g_f.cost() < best_g.cost() {
                best_g = g_f;
            }
        }
        Some(best_g)
    }
}

// ============================================================================
// 6. Multi-Column Dense Block Coordinator
// ============================================================================

/// Result of encoding all columns of a dense block via [`PackRegistry::encode_dense_block`].
pub struct EncodedDenseBlock {
    pub shared_dict_blobs: Vec<Vec<u8>>,
    pub prose_pool_leaf: Option<Vec<u8>>,
    pub col_blobs: Vec<Vec<u8>>,
    pub summaries: Vec<ColSummary>,
}

fn replace_subslice(hay: &[u8], needle: &[u8], rep: &[u8]) -> Option<Vec<u8>> {
    let pos = memchr::memmem::find(hay, needle)?;
    let mut out = Vec::with_capacity(hay.len() - needle.len() + rep.len());
    out.extend_from_slice(&hay[..pos]);
    out.extend_from_slice(rep);
    out.extend_from_slice(&hay[pos + needle.len()..]);
    Some(out)
}

impl PackRegistry {
    /// Encodes all dense occurrence columns of a structural block, coordinating single-column
    /// [`CodecPack`]s, horizontal peer derivations (`ExtPeer`), cross-column dictionary sharing
    /// (`ExtDict`), and long-text pooling (`ExtPool`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_dense_block(
        &self,
        occ_streams: &[Vec<&[u8]>],
        dict_groups: &[&str],
        horiz_cands: &[Vec<usize>],
        preencoded_nested: &mut [Option<(ColSummary, Vec<u8>)>],
        standalone: &[bool],
        backend: EntropyBackend,
        level: i32,
    ) -> EncodedDenseBlock {
        with_encode_opts(backend, level, || {
            let n_cols = occ_streams.len();
            let mut col_blobs = Vec::with_capacity(n_cols);
            // What each blob counts as when sizes are compared: its length less shared code.
            let mut col_costs = Vec::with_capacity(n_cols);
            let mut summaries = Vec::with_capacity(n_cols);
            let mut used_ext = vec![false; n_cols];
            let mut raw_backed = vec![false; n_cols];
            let mut fixed_id_widths = vec![None::<usize>; n_cols];

            for cid in 0..n_cols {
                let vals = &occ_streams[cid];
                if vals.len() >= 8 {
                    let (u0, _) = unquote_json_opt(vals[0]);
                    if matches!(u0.len(), 16 | 32 | 36 | 40 | 64)
                        && vals
                            .iter()
                            .take(16)
                            .all(|v| unquote_json_opt(v).0.len() == u0.len())
                    {
                        fixed_id_widths[cid] = Some(u0.len());
                    }
                }
            }

            // Phase 1: Single-column `PayloadGraph` + horizontal `ExtPeer` derivations.
            for cid in 0..n_cols {
                if let Some((sum, blob)) = preencoded_nested.get_mut(cid).and_then(Option::take) {
                    summaries.push(sum);
                    col_costs.push(blob.len());
                    col_blobs.push(blob);
                    used_ext[cid] = true;
                    continue;
                }
                let vals = &occ_streams[cid];
                if vals.is_empty() {
                    summaries.push(ColSummary::of(&[]));
                    col_blobs.push(PayloadGraph::empty().encode());
                    col_costs.push(1);
                    continue;
                }
                let occ_len = vals.len();
                let mut sum = ColSummary::of(vals);
                if let Some((lo, hi)) = int_bounds(vals) {
                    (sum.numeric_kind, sum.min_val, sum.max_val) = (1, lo, hi);
                } else if let Some((lo, hi)) = time_bounds(vals) {
                    (sum.numeric_kind, sum.min_val, sum.max_val) = (2, lo, hi);
                }

                let (best_single_g, is_raw_fb) = self.encode_best_graph_inner(vals, 0);
                let (mut best_blob, mut best_cost) = best_single_g.priced();
                let must_standalone = standalone.get(cid).copied().unwrap_or(false);

                if !must_standalone && occ_len >= 8 {
                    let cands = horiz_cands.get(cid).map_or(&[][..], |v| v.as_slice());

                    // 1a. Equal-length `ExactCopy` & `SubstringInterp`.
                    let mut interp_srcs = Vec::<usize>::new();
                    for &src_i in cands {
                        if src_i >= cid
                            || occ_streams[src_i].len() != occ_len
                            || preencoded_nested.get(src_i).is_some_and(|x| x.is_some())
                        {
                            continue;
                        }
                        let src_vals = &occ_streams[src_i];
                        let probe = occ_len.min(64);
                        let mut eq_probe = 0usize;
                        let mut sub_probe = 0usize;
                        for r in 0..probe {
                            let tv = vals[r];
                            let sv = src_vals[r];
                            if tv == sv {
                                eq_probe += 1;
                            } else {
                                let (sv_u, _) = unquote_json_opt(sv);
                                if sv_u.len() >= 3
                                    && tv.len() > sv_u.len()
                                    && !sv_u.contains(&0x1f)
                                    && memchr::memmem::find(tv, sv_u).is_some()
                                {
                                    sub_probe += 1;
                                }
                            }
                        }
                        if eq_probe * 4 >= probe {
                            let mut exc_idx = Vec::new();
                            let mut exc_vals = Vec::new();
                            for r in 0..occ_len {
                                if vals[r] != src_vals[r] {
                                    exc_idx.push(r);
                                    exc_vals.push(vals[r]);
                                }
                            }
                            if exc_idx.len() * 2 <= occ_len {
                                let mut gb = GraphBuilder::new(occ_len);
                                let peer = gb.add_ext_peer(src_i as u32);
                                let root = gb.add_patch(peer, &exc_idx, &exc_vals);
                                let (cand, cand_cost) = gb.finish(root).priced();
                                if cand_cost + 6 < best_cost {
                                    best_blob = cand;
                                    best_cost = cand_cost;
                                    used_ext[cid] = true;
                                }
                            }
                        }
                        if interp_srcs.len() < 3 && sub_probe * 4 >= probe {
                            interp_srcs.push(src_i);
                        }
                    }

                    if !interp_srcs.is_empty() {
                        let mut templated: Vec<Vec<u8>> = Vec::with_capacity(occ_len);
                        let mut hits = 0usize;
                        for r in 0..occ_len {
                            let target = vals[r];
                            if target.contains(&0x1f) {
                                templated.push(target.to_vec());
                                continue;
                            }
                            let mut cur = target.to_vec();
                            let mut replaced = false;
                            for (slot_k, &src_i) in interp_srcs.iter().enumerate() {
                                let sv = occ_streams[src_i][r];
                                let (sv_u, was_q) = unquote_json_opt(sv);
                                if sv.len() >= 3
                                    && cur.len() > sv.len()
                                    && !sv.contains(&0x1f)
                                    && let Some(next) =
                                        replace_subslice(&cur, sv, &[0x1f, (slot_k * 2) as u8])
                                {
                                    cur = next;
                                    replaced = true;
                                } else if was_q
                                    && sv_u.len() >= 3
                                    && cur.len() > sv_u.len()
                                    && !sv_u.contains(&0x1f)
                                    && let Some(next) = replace_subslice(
                                        &cur,
                                        sv_u,
                                        &[0x1f, (slot_k * 2 + 1) as u8],
                                    )
                                {
                                    cur = next;
                                    replaced = true;
                                }
                            }
                            if replaced {
                                hits += 1;
                            }
                            templated.push(cur);
                        }
                        if hits * 4 >= occ_len {
                            let t_refs: Vec<&[u8]> =
                                templated.iter().map(|x| x.as_slice()).collect();
                            let t_graph = self.encode_best_graph(&t_refs);
                            let mut gb = GraphBuilder::new(occ_len);
                            let t_node = gb.merge_graph(&t_graph);
                            let mut slot_nodes = Vec::with_capacity(interp_srcs.len());
                            for &src_i in &interp_srcs {
                                slot_nodes.push(gb.add_ext_peer(src_i as u32));
                            }
                            let root = gb.push(GraphNode::TemplateCells {
                                template: t_node,
                                slots: slot_nodes,
                            });
                            let (cand, cand_cost) = gb.finish(root).priced();
                            if cand_cost + 6 < best_cost {
                                best_blob = cand;
                                best_cost = cand_cost;
                                used_ext[cid] = true;
                            }
                        }
                    }

                    // 1b. Equal-length `AffineInt` (`c = a ± b + residual`, bare or JSON-quoted).
                    let (u0, q0) = unquote_json_opt(vals[0]);
                    if ExactIntPack::parse_canonical_i64(u0).is_some() {
                        let mut int_srcs = Vec::new();
                        for &src_i in cands {
                            if src_i < cid
                                && occ_streams[src_i].len() == occ_len
                                && occ_streams[src_i].iter().take(16).all(|b| {
                                    ExactIntPack::parse_canonical_i64(unquote_json_opt(b).0)
                                        .is_some()
                                })
                            {
                                int_srcs.push(src_i);
                                if int_srcs.len() >= 4 {
                                    break;
                                }
                            }
                        }
                        for ia in 0..int_srcs.len() {
                            for ib in 0..int_srcs.len() {
                                if ia == ib {
                                    continue;
                                }
                                let sa = int_srcs[ia];
                                let sb = int_srcs[ib];
                                for sub in [false, true] {
                                    let mut res = Vec::with_capacity(occ_len);
                                    let mut exc_idx = Vec::new();
                                    let mut exc_vals = Vec::new();
                                    let mut ok = true;
                                    for r in 0..occ_len {
                                        let (ut, qt) = unquote_json_opt(vals[r]);
                                        let (ua, _) = unquote_json_opt(occ_streams[sa][r]);
                                        let (ub, _) = unquote_json_opt(occ_streams[sb][r]);
                                        if qt == q0
                                            && let (Some(vt), Some(va), Some(vb)) = (
                                                ExactIntPack::parse_canonical_i64(ut),
                                                ExactIntPack::parse_canonical_i64(ua),
                                                ExactIntPack::parse_canonical_i64(ub),
                                            )
                                        {
                                            let pred = if sub {
                                                va.wrapping_sub(vb)
                                            } else {
                                                va.wrapping_add(vb)
                                            };
                                            res.push(vt.wrapping_sub(pred));
                                        } else {
                                            res.push(0);
                                            exc_idx.push(r);
                                            exc_vals.push(vals[r]);
                                            if exc_idx.len() * 8 > occ_len {
                                                ok = false;
                                                break;
                                            }
                                        }
                                    }
                                    if !ok {
                                        continue;
                                    }
                                    let mut gb = GraphBuilder::new(occ_len);
                                    let pa = gb.add_ext_peer(sa as u32);
                                    let pb = gb.add_ext_peer(sb as u32);
                                    let ia_n = gb.add_parse_ascii_int(pa);
                                    let ib_n = gb.add_parse_ascii_int(pb);
                                    let res_n = gb.add_ints(&res);
                                    let b = Builder::new(3);
                                    let va = b.input(0);
                                    let vb = b.input(1);
                                    let vr = b.input(2);
                                    let comb = if sub {
                                        b.rr(Op::Sub, va, vb)
                                    } else {
                                        b.rr(Op::Add, va, vb)
                                    };
                                    let out_v = b.rr(Op::Add, comb, vr);
                                    let sum_n = gb.add_prog_i64(
                                        b.finish(&[(out_v, 8)]),
                                        vec![ia_n, ib_n, res_n],
                                        vec![],
                                    );
                                    let fmt_n = gb.add_prog_cells(
                                        prog_fmt_int(false, 0),
                                        vec![sum_n],
                                        vec![],
                                    );
                                    let base_n = if q0 {
                                        gb.add_quote_or_null(fmt_n, None)
                                    } else {
                                        fmt_n
                                    };
                                    let root = gb.add_patch(base_n, &exc_idx, &exc_vals);
                                    let (cand, cand_cost) = gb.finish(root).priced();
                                    if cand_cost + 6 < best_cost {
                                        best_blob = cand;
                                        best_cost = cand_cost;
                                        used_ext[cid] = true;
                                    }
                                }
                            }
                        }
                    }

                    // 1c. Any-length Cross-Column ID Index (`payload.head` -> `commits[].sha`, etc.)!
                    if let Some(fw) = fixed_id_widths[cid] {
                        let mut tried = 0usize;
                        for src_i in (0..cid).rev() {
                            if fixed_id_widths[src_i] == Some(fw) && !occ_streams[src_i].is_empty()
                            {
                                let src_vals = &occ_streams[src_i];
                                let mut lut: FastMap<&[u8], i64> =
                                    fast_map_with_capacity(src_vals.len().min(4096));
                                for (idx, &sv) in src_vals.iter().enumerate() {
                                    lut.entry(sv).or_insert(idx as i64);
                                }
                                let probe = occ_len.min(64);
                                let probe_hits = vals[..probe]
                                    .iter()
                                    .filter(|&&v| lut.contains_key(v))
                                    .count();
                                if probe_hits * 5 >= probe {
                                    let mut idxs = Vec::with_capacity(occ_len);
                                    let mut exc_idx = Vec::new();
                                    let mut exc_vals = Vec::new();
                                    let mut last_idx = 0i64;
                                    for (r, &v) in vals.iter().enumerate() {
                                        if let Some(&pos) = lut.get(v) {
                                            idxs.push(pos);
                                            last_idx = pos;
                                        } else {
                                            idxs.push(last_idx);
                                            exc_idx.push(r);
                                            exc_vals.push(v);
                                        }
                                    }
                                    if exc_idx.len() < occ_len {
                                        let mut gb = GraphBuilder::new(occ_len);
                                        let peer = gb.add_ext_peer(src_i as u32);
                                        let idx_node = gb.add_mtf_ints(&idxs, 4096);
                                        let indexed = gb.add_index_cells(peer, idx_node);
                                        let root = gb.add_patch(indexed, &exc_idx, &exc_vals);
                                        let (cand, cand_cost) = gb.finish(root).priced();
                                        if cand_cost + 12 < best_cost {
                                            best_blob = cand;
                                            best_cost = cand_cost;
                                            used_ext[cid] = true;
                                        }
                                    }
                                }
                                tried += 1;
                                if tried >= 3 {
                                    break;
                                }
                            }
                        }
                    }
                }

                if !used_ext[cid]
                    && !must_standalone
                    && (is_raw_fb || dict_groups.get(cid).copied().unwrap_or("") == "text")
                {
                    raw_backed[cid] = true;
                }
                summaries.push(sum);
                col_costs.push(best_cost);
                col_blobs.push(best_blob);
            }

            // Phase 2: Block-level `SharedDictPool` (`ExtDict` + `IndexCells`) across categorical columns.
            struct DictCand<'a> {
                cid: usize,
                all_quoted: bool,
                vocab: FastMap<&'a [u8], u32>,
            }
            let mut dict_cands = Vec::<DictCand<'_>>::new();
            for cid in 0..n_cols {
                let vals = &occ_streams[cid];
                if vals.is_empty()
                    || used_ext[cid]
                    || standalone.get(cid).copied().unwrap_or(false)
                    || summaries[cid].numeric_kind != 0
                    || dict_groups.get(cid).copied().unwrap_or("") == "text"
                {
                    continue;
                }
                let all_q = vals.iter().all(|v| is_clean_json_quoted(v));
                let mut vocab: FastMap<&[u8], u32> = fast_map_with_capacity(vals.len().min(512));
                let mut uniq_bytes = 0usize;
                let mut ok = true;
                for &v in vals {
                    let inner = if all_q { &v[1..v.len() - 1] } else { v };
                    if let Some(c) = vocab.get_mut(inner) {
                        *c += 1;
                    } else {
                        if vocab.len() >= 4096 || uniq_bytes + inner.len() > (256 << 10) {
                            ok = false;
                            break;
                        }
                        uniq_bytes += inner.len();
                        vocab.insert(inner, 1);
                    }
                }
                if ok && vocab.len() >= 2 && vocab.len() * 4 <= vals.len() * 3 {
                    dict_cands.push(DictCand {
                        cid,
                        all_quoted: all_q,
                        vocab,
                    });
                }
            }

            let mut shared_dict_blobs = Vec::<Vec<u8>>::new();
            let mut assigned = vec![false; dict_cands.len()];
            for i in 0..dict_cands.len() {
                if assigned[i] {
                    continue;
                }
                let mut cluster = vec![i];
                let gi = dict_groups.get(dict_cands[i].cid).copied().unwrap_or("");
                for j in (i + 1)..dict_cands.len() {
                    if assigned[j] || dict_cands[j].all_quoted != dict_cands[i].all_quoted {
                        continue;
                    }
                    let gj = dict_groups.get(dict_cands[j].cid).copied().unwrap_or("");
                    let same_group = !gi.is_empty() && gi == gj;
                    let inter = dict_cands[i]
                        .vocab
                        .keys()
                        .filter(|k| dict_cands[j].vocab.contains_key(*k))
                        .count();
                    let min_v = dict_cands[i].vocab.len().min(dict_cands[j].vocab.len());
                    if (same_group && inter >= 1) || (min_v >= 4 && inter * 10 >= min_v) {
                        cluster.push(j);
                    }
                }
                if cluster.len() < 2 {
                    continue;
                }
                let mut merged: FastMap<&[u8], u32> = fast_map_with_capacity(512);
                for &ci in &cluster {
                    for (&s, &cnt) in &dict_cands[ci].vocab {
                        *merged.entry(s).or_insert(0) += cnt;
                    }
                }
                if merged.len() > 8192 {
                    continue;
                }
                let mut entries: Vec<(&[u8], u32)> = merged.into_iter().collect();
                entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                let entry_slices: Vec<&[u8]> = entries.iter().map(|e| e.0).collect();
                let dict_blob = self.encode_best(&entry_slices);

                let mut lut: FastMap<&[u8], i64> = fast_map_with_capacity(entry_slices.len());
                for (id, &s) in entry_slices.iter().enumerate() {
                    lut.insert(s, id as i64);
                }

                let dict_id = shared_dict_blobs.len() as u16;
                let mut cur_sum = 0usize;
                let mut new_sum = dict_blob.len() + 3;
                let mut member_blobs = Vec::with_capacity(cluster.len());
                for &ci in &cluster {
                    let cid = dict_cands[ci].cid;
                    let all_q = dict_cands[ci].all_quoted;
                    cur_sum += col_costs[cid];
                    let vals = &occ_streams[cid];
                    let ords: Vec<i64> = vals
                        .iter()
                        .map(|v| {
                            let inner = if all_q { &v[1..v.len() - 1] } else { *v };
                            lut[inner]
                        })
                        .collect();
                    let mut gb = GraphBuilder::new(vals.len());
                    let d_node = gb.add_ext_dict(dict_id);
                    let o_node = gb.add_mtf_ints(&ords, 4096);
                    let idx_node = gb.add_index_cells(d_node, o_node);
                    let root = if all_q {
                        gb.add_quote_or_null(idx_node, None)
                    } else {
                        idx_node
                    };
                    let mb = gb.finish(root).encode();
                    new_sum += mb.len();
                    member_blobs.push((ci, cid, mb));
                }
                if new_sum + 16 < cur_sum {
                    shared_dict_blobs.push(dict_blob);
                    for (ci, cid, mb) in member_blobs {
                        assigned[ci] = true;
                        col_costs[cid] = mb.len();
                        col_blobs[cid] = mb;
                        used_ext[cid] = true;
                        raw_backed[cid] = false;
                    }
                }
            }

            // Phase 3: Block-level `ProseSlice` (`ExtPool` + `SliceBuf`) across long-text columns.
            let mut prose_cols = Vec::<usize>::new();
            let mut total_prose_bytes = 0usize;
            for cid in 0..n_cols {
                if !raw_backed[cid]
                    || used_ext[cid]
                    || standalone.get(cid).copied().unwrap_or(false)
                {
                    continue;
                }
                let vals = &occ_streams[cid];
                if vals.is_empty() {
                    continue;
                }
                let col_bytes: usize = vals.iter().map(|b| b.len()).sum();
                let mean_len = col_bytes / vals.len().max(1);
                if (mean_len >= 40 || dict_groups.get(cid).copied().unwrap_or("") == "text")
                    && col_bytes >= 256
                    && total_prose_bytes + col_bytes <= (16 << 20)
                {
                    prose_cols.push(cid);
                    total_prose_bytes += col_bytes;
                }
            }

            let mut prose_pool_leaf = None;
            if prose_cols.len() >= 2 {
                let mut cur_sum = 0usize;
                let mut pool_raw = Vec::with_capacity(total_prose_bytes);
                let mut member_specs = Vec::with_capacity(prose_cols.len());

                for &cid in &prose_cols {
                    cur_sum += col_costs[cid];
                    let vals = &occ_streams[cid];
                    let base = pool_raw.len() as u32;
                    let mut lens = Vec::with_capacity(vals.len());
                    for &v in vals {
                        pool_raw.extend_from_slice(v);
                        lens.push(v.len() as i64);
                    }
                    let mut gb = GraphBuilder::new(vals.len());
                    let p_node = gb.add_ext_pool();
                    let s_node = gb.add_slice_buf(p_node, base, &lens);
                    member_specs.push((cid, gb.finish(s_node).encode()));
                }
                let p_leaf = pack_leaf(&pool_raw);
                let new_sum =
                    p_leaf.len() + 3 + member_specs.iter().map(|(_, b)| b.len()).sum::<usize>();
                if new_sum + 16 < cur_sum {
                    prose_pool_leaf = Some(p_leaf);
                    for (cid, blob) in member_specs {
                        col_costs[cid] = blob.len();
                        col_blobs[cid] = blob;
                    }
                }
            }

            EncodedDenseBlock {
                shared_dict_blobs,
                prose_pool_leaf,
                col_blobs,
                summaries,
            }
        })
    }
}

#[cfg(test)]
#[path = "packs_tests.rs"]
mod tests;
