use std::panic::{AssertUnwindSafe, catch_unwind};

use super::*;
use crate::cells::Cells;
use crate::payload::graph::CoreDecoder;
use crate::wire::DecodeBudget;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn lossy(b: &[u8]) -> String {
    if b.iter().all(|&c| (0x20..0x7f).contains(&c)) {
        String::from_utf8_lossy(b).into_owned()
    } else {
        format!("{b:02x?}")
    }
}

fn show_rows(rows: &[&[u8]]) -> String {
    let mut s: Vec<String> = rows
        .iter()
        .take(24)
        .map(|r| format!("{:?}", lossy(r)))
        .collect();
    if rows.len() > 24 {
        s.push(format!("... {} rows", rows.len()));
    }
    s.join(", ")
}

fn compare(cells: &Cells, rows: &[&[u8]]) -> Result<(), String> {
    if cells.len() != rows.len() {
        return Err(format!("{} cells for {} rows", cells.len(), rows.len()));
    }
    for (i, &want) in rows.iter().enumerate() {
        let got = cells.get(i).unwrap_or_default();
        if got != want {
            return Err(format!(
                "row {i}: got {:?} want {:?}",
                lossy(got),
                lossy(want)
            ));
        }
    }
    Ok(())
}

fn decode_and_compare(wire: &[u8], rows: &[&[u8]]) -> Result<(), String> {
    let cells = CoreDecoder::decode_bytes(wire, &DecodeBudget::default())
        .map_err(|e| format!("decode error: {e:?}"))?;
    compare(&cells, rows)
}

/// `Ok(false)` when the pack declines, `Err` when its graph does not decode back to `rows`.
fn check(pack: &dyn CodecPack, rows: &[&[u8]]) -> Result<bool, String> {
    let g = catch_unwind(AssertUnwindSafe(|| pack.try_encode(rows)))
        .map_err(|_| "encoder panicked".to_string())?;
    let Some(g) = g else { return Ok(false) };
    let wire = g.encode();
    catch_unwind(AssertUnwindSafe(|| decode_and_compare(&wire, rows)))
        .map_err(|_| "decoder panicked".to_string())??;
    Ok(true)
}

/// `pack`'s graph for `rows`, which the bare decoder must turn back into `rows`.
fn roundtrip(pack: &dyn CodecPack, rows: &[&[u8]]) -> usize {
    let wire = pack
        .try_encode(rows)
        .unwrap_or_else(|| panic!("{} refused", pack.name()))
        .encode();
    if let Err(e) = decode_and_compare(&wire, rows) {
        panic!("{}: {e}", pack.name());
    }
    wire.len()
}

/// [`roundtrip`] for a column the pack must rebuild from an MT19937 stream.
fn roundtrip_generated(pack: &dyn CodecPack, rows: &[&[u8]]) -> usize {
    let g = pack
        .try_encode(rows)
        .unwrap_or_else(|| panic!("{} refused", pack.name()));
    let found = g
        .nodes
        .iter()
        .any(|n| matches!(n, GraphNode::RecurBuf { .. }));
    assert!(found, "{} missed the stream", pack.name());
    roundtrip(pack, rows)
}

fn roundtrip_strs(pack: &dyn CodecPack, rows: &[&str]) -> usize {
    let rows: Vec<&[u8]> = rows.iter().map(|s| s.as_bytes()).collect();
    roundtrip(pack, &rows)
}

fn refuses(pack: &dyn CodecPack, rows: &[&[u8]]) {
    assert!(
        pack.try_encode(rows).is_none(),
        "{} accepted {}",
        pack.name(),
        show_rows(rows)
    );
}

fn refuses_strs(pack: &dyn CodecPack, rows: &[&str]) {
    let rows: Vec<&[u8]> = rows.iter().map(|s| s.as_bytes()).collect();
    refuses(pack, &rows);
}

fn texts(vals: impl IntoIterator<Item = i64>) -> Vec<Vec<u8>> {
    vals.into_iter()
        .map(|v| v.to_string().into_bytes())
        .collect()
}

fn owned(rows: impl IntoIterator<Item = impl Into<String>>) -> Vec<Vec<u8>> {
    rows.into_iter().map(|s| s.into().into_bytes()).collect()
}

fn refs(rows: &[Vec<u8>]) -> Vec<&[u8]> {
    rows.iter().map(Vec::as_slice).collect()
}

const SHOW: usize = 6;

/// xorshift64*: deterministic fuzz input.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Inclusive.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo.wrapping_add((self.next() % (hi.wrapping_sub(lo) as u64).wrapping_add(1)) as i64)
    }
    fn one_in(&mut self, n: u64) -> bool {
        self.below(n) == 0
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// Runs `cases` generated columns through `pack`; every accepted one must round-trip and at
/// least `min_hits` must be accepted, so the fuzz cannot pass by refusing everything.
fn fuzz(
    pack: &dyn CodecPack,
    seed: u64,
    cases: usize,
    min_hits: usize,
    mut gen_rows: impl FnMut(&mut Rng) -> Vec<Vec<u8>>,
) {
    let mut rng = Rng::new(seed);
    let (mut hits, mut fails) = (0usize, Vec::new());
    for case in 0..cases {
        let rows = gen_rows(&mut rng);
        let r = refs(&rows);
        match check(pack, &r) {
            Ok(hit) => hits += usize::from(hit),
            Err(e) => fails.push(format!("case {case}: {e}\n    rows: {}", show_rows(&r))),
        }
    }
    assert!(
        fails.is_empty(),
        "{}: {}/{cases} cases failed:\n{}",
        pack.name(),
        fails.len(),
        fails[..fails.len().min(SHOW)].join("\n")
    );
    assert!(
        hits >= min_hits,
        "{} encoded only {hits}/{cases}",
        pack.name()
    );
}

const JUNK: &[&str] = &[
    "", "-", "abc", "null", " 1", "1 ", "+5", "0x1f", "1e3", "-0", "007", "NaN", "--1", "1.2.3",
    "\u{663}",
];

/// Replaces about one row in `rate` with junk or one of `extra`.
fn sprinkle(r: &mut Rng, rows: &mut [Vec<u8>], rate: u64, extra: &[String]) {
    for row in rows.iter_mut() {
        if r.one_in(rate) {
            *row = if !extra.is_empty() && r.one_in(2) {
                extra[r.below(extra.len() as u64) as usize]
                    .clone()
                    .into_bytes()
            } else {
                r.pick(JUNK).as_bytes().to_vec()
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Domain formatters
// ---------------------------------------------------------------------------

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn secs_of(y: i64, m: u32, d: u32, hh: i64, mm: i64, ss: i64) -> i64 {
    days_from_civil(y, m, d) * 86_400 + hh * 3600 + mm * 60 + ss
}

/// `(y, m, d, hh, mm, ss)` of a Unix time.
fn split_secs(t: i64) -> (i64, u32, u32, i64, i64, i64) {
    let (y, m, d) = civil_from_days(t.div_euclid(86_400));
    let sod = t.rem_euclid(86_400);
    (y, m, d, sod / 3600, sod / 60 % 60, sod % 60)
}

const MAX_ISO_SECS: i64 = 253_402_300_799; // 9999-12-31T23:59:59
const MON: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

fn iso(t: i64, sep: char, frac: Option<(char, usize, u64)>, tz: &str) -> String {
    let (y, m, d, hh, mm, ss) = split_secs(t);
    let mut s = format!("{y:04}-{m:02}-{d:02}{sep}{hh:02}:{mm:02}:{ss:02}");
    if let Some((fsep, digits, sub)) = frac {
        s += &format!("{fsep}{sub:0digits$}");
    }
    s + tz
}

fn clf(t: i64, tz: &str) -> String {
    let (y, m, d, hh, mm, ss) = split_secs(t);
    format!(
        "{d:02}/{}/{y:04}:{hh:02}:{mm:02}:{ss:02} {tz}",
        MON[m as usize - 1]
    )
}

fn syslog(t: i64) -> String {
    let (_, m, d, hh, mm, ss) = split_secs(t);
    format!("{} {d:>2} {hh:02}:{mm:02}:{ss:02}", MON[m as usize - 1])
}

fn hex(b: &[u8], upper: bool) -> String {
    b.iter()
        .map(|x| {
            if upper {
                format!("{x:02X}")
            } else {
                format!("{x:02x}")
            }
        })
        .collect()
}

fn uuid(b: &[u8; 16], upper: bool) -> String {
    let h = hex(b, upper);
    format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

fn ulid(ts: u64, ent: u128) -> String {
    let v = (u128::from(ts) << 80) | (ent & ((1 << 80) - 1));
    (0..26)
        .map(|k| CROCKFORD_32[((v >> (125 - 5 * k)) & 31) as usize] as char)
        .collect()
}

fn b64(data: &[u8], url: bool, pad: bool) -> String {
    let alpha = if url { B64_URL } else { B64_STD };
    let mut s = String::new();
    for c in data.chunks(3) {
        let v = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for k in 0..=c.len() {
            s.push(alpha[(v >> (18 - 6 * k) & 63) as usize] as char);
        }
        if pad {
            s.extend(std::iter::repeat_n('=', 3 - c.len()));
        }
    }
    s
}

fn dec(v: i64, scale: u32) -> String {
    let p = 10u64.pow(scale);
    let a = v.unsigned_abs();
    let sign = if v < 0 { "-" } else { "" };
    format!("{sign}{}.{:0w$}", a / p, a % p, w = scale as usize)
}

/// Python `random.Random(seed)` tempered outputs.
fn mt_words(seed: u32, n: usize) -> Vec<u32> {
    let mut s = MtStream::seeded(seed);
    s.grow(n);
    s.words[..n].to_vec()
}

fn le_rows(words: &[u32], per_row: usize) -> Vec<Vec<u8>> {
    words
        .chunks_exact(per_row)
        .map(|w| w.iter().flat_map(|x| x.to_le_bytes()).collect())
        .collect()
}

fn be_rows(words: &[u32], per_row: usize) -> Vec<Vec<u8>> {
    words
        .chunks_exact(per_row)
        .map(|w| w.iter().rev().flat_map(|x| x.to_be_bytes()).collect())
        .collect()
}

fn edge_i64(r: &mut Rng) -> i64 {
    let p = 10i64.pow(r.below(19) as u32);
    match r.below(6) {
        0 => r.pick(&[0, 1, -1, i64::MAX, i64::MIN, i64::MIN + 1, i64::MAX - 1]),
        1 => p - 1,
        2 => -p,
        3 => p,
        4 => 1 - p,
        _ => r.next() as i64,
    }
}

// ---------------------------------------------------------------------------
// Column generators (shared by the pack fuzzers and the registry matrix)
// ---------------------------------------------------------------------------

fn gen_ints(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 9, 16, 17, 40]);
    let mut rows: Vec<Vec<u8>> = match r.below(5) {
        0 => {
            let mut v = r.range(-1_000_000, 1_000_000);
            let step = r.pick(&[1i64, 3, 100, 1 << 20]);
            texts((0..n).map(|_| {
                v = v.wrapping_add(r.range(-1, step));
                v
            }))
        }
        1 => texts((0..n).map(|_| edge_i64(r))),
        2 => {
            let w = r.range(2, 16) as usize;
            let max = 10u64.pow(w as u32);
            (0..n)
                .map(|i| {
                    let v = match (i, r.below(3)) {
                        (1, _) => r.below(10),
                        (_, 0) => r.below(max),
                        (_, 1) => max - 1 - r.below(3),
                        _ => r.below(100),
                    };
                    format!("{v:0w$}").into_bytes()
                })
                .collect()
        }
        3 => texts((0..n).map(|_| r.range(-20, 20))),
        _ => texts((0..n).map(|_| r.next() as i64 >> r.below(64))),
    };
    sprinkle(r, &mut rows, 12, &[]);
    rows
}

fn gen_decimals(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 9, 16, 40]);
    let scale = r.range(1, 8) as u32;
    let mag = r.pick(&[100i64, 100_000, 10_000_000_000, 99_999_999_999_999_999]);
    let mut v = r.range(-mag, mag);
    let mut rows = owned((0..n).map(|_| {
        if r.one_in(2) {
            v = r.range(-mag, mag);
        } else {
            v = (v + r.range(-50, 50)).clamp(-mag, mag);
        }
        dec(v, scale)
    }));
    let other = if scale == 1 { "1.25" } else { "1.5" };
    let extra: Vec<String> = [
        other, "-0.0", "-0.00", "1.", ".5", "01.5", "1e5", "12", "-.5",
    ]
    .map(String::from)
    .to_vec();
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn lcg_column(r: &mut Rng, n: usize) -> Vec<Vec<u8>> {
    let mut states = [r.below(1 << 48), r.below(1 << 48)];
    let mut vals: Vec<i64> = Vec::with_capacity(n);
    while vals.len() < n {
        let k = usize::from(r.one_in(5));
        let s = &mut states[k];
        let v = JavaLcgPack::output(*s);
        for _ in 0..2 {
            *s = s.wrapping_mul(LCG_MULT).wrapping_add(LCG_ADD) & LCG_MASK;
        }
        vals.push(v);
        if r.one_in(4) && !vals.is_empty() {
            let back = r.below(vals.len() as u64) as usize;
            vals.push(vals[back]);
        }
    }
    vals.truncate(n);
    texts(vals)
}

fn gen_lcg(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[8usize, 9, 16, 17, 64]);
    let mut rows = lcg_column(r, n);
    let extra = vec![(r.next() as i64).to_string()];
    sprinkle(r, &mut rows, 20, &extra);
    rows
}

fn gen_iso(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 9, 17, 40]);
    let sep = r.pick(&['T', 't', ' ']);
    let digits = r.pick(&[0usize, 0, 1, 3, 6, 8, 9]);
    let fsep = r.pick(&['.', ',']);
    let tz = r.pick(&["", "Z", "z", "+05:30", "-08:00", "+0000", "-0130"]);
    let mut t = match r.below(3) {
        0 => r.range(0, MAX_ISO_SECS),
        1 => r.pick(&[
            0,
            secs_of(2000, 2, 28, 23, 59, 58),
            secs_of(2024, 2, 28, 23, 59, 59),
            secs_of(2023, 12, 31, 23, 59, 58),
            secs_of(1999, 12, 31, 23, 59, 59),
            secs_of(2100, 2, 28, 23, 59, 59),
            MAX_ISO_SECS - 3,
        ]),
        _ => r.range(1_600_000_000, 1_800_000_000),
    };
    let jump = r.pick(&[2i64, 86_400, 40_000_000]);
    let mut rows = owned((0..n).map(|_| {
        let s = iso(
            t,
            sep,
            (digits > 0).then(|| (fsep, digits, r.below(10u64.pow(digits as u32)))),
            tz,
        );
        t = (t + r.range(-jump / 8, jump)).clamp(0, MAX_ISO_SECS);
        s
    }));
    let other_tz = if tz == "Z" { "+00:00" } else { "Z" };
    let extra = vec![
        iso(t, sep, None, other_tz),
        iso(t, if sep == 'T' { ' ' } else { 'T' }, None, tz),
        "2023-02-29T00:00:00".into(),
        "2024-13-01T00:00:00".into(),
        "2024-04-31T00:00:00".into(),
        "1969-12-31T23:59:59".into(),
        "2024-01-01T24:00:00".into(),
        "2024-01-01T00:00:00.".into(),
        "2024-01-01T00:00:00.1234567890".into(),
    ];
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn gen_clf(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 9, 17, 40]);
    let tz = r.pick(&["+0000", "-0700", "+0530", "-1200"]);
    let mut t = r.range(0, MAX_ISO_SECS - 100_000_000);
    let jump = r.pick(&[2i64, 86_400, 40_000_000]);
    let mut rows = owned((0..n).map(|_| {
        let s = clf(t, tz);
        t += r.range(-jump / 8, jump);
        t = t.max(0);
        s
    }));
    let (_, m, _, _, _, _) = split_secs(t);
    let extra = vec![
        clf(t, if tz == "+0000" { "-0000" } else { "+0000" }),
        format!(
            "{}/{}/2023:00:00:00 {tz}",
            r.range(29, 31),
            MON[m as usize - 1]
        ),
        "31/Dec/1969:23:59:59 +0000".into(),
        "01/jan/2024:00:00:00 +0000".into(),
        "01/Jan/2024:24:00:00 +0000".into(),
        "01/Jan/2024:00:00:00 +00:0".into(),
    ];
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn gen_syslog(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 9, 17, 40]);
    let mut t = r.range(secs_of(2023, 1, 1, 0, 0, 0), secs_of(2025, 12, 31, 0, 0, 0));
    let jump = r.pick(&[2i64, 3600, 86_400 * 9]);
    let mut rows = owned((0..n).map(|_| {
        let s = syslog(t);
        t += r.range(-jump / 8, jump);
        s
    }));
    let mon = MON[r.below(12) as usize];
    let mut extra: Vec<String> = [
        "Jan  0 00:00:00",
        "jan  1 00:00:00",
        "Jan 1 00:00:00",
        "Jan 32 00:00:00",
        "Jan  1 24:00:00",
    ]
    .map(String::from)
    .to_vec();
    extra.push(format!("{mon} {:02} 00:00:00", r.range(1, 9)));
    extra.push(format!("{mon} {} 00:00:00", r.range(29, 31)));
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn uuid_bytes(r: &mut Rng, v7_ts: Option<u64>, digits_only: bool) -> [u8; 16] {
    let mut b: [u8; 16] = r.bytes(16).try_into().unwrap();
    if digits_only {
        for x in &mut b {
            *x = (((*x >> 4) % 10) << 4) | ((*x & 15) % 10);
        }
        return b;
    }
    match v7_ts {
        Some(ts) => {
            b[..6].copy_from_slice(&ts.to_be_bytes()[2..]);
            b[6] = 0x70 | (b[6] & 15);
        }
        None => b[6] = 0x40 | (b[6] & 15),
    }
    b[8] = 0x80 | (b[8] & 0x3f);
    b
}

fn gen_uuid(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 16, 33]);
    let upper = r.one_in(3);
    let mode = r.below(3);
    let mut ts = r.below(1 << 47);
    let mut rows = owned((0..n).map(|i| {
        ts += r.below(1000);
        let v7 = mode == 1 || (mode == 2 && i % 2 == 0);
        let digits = r.one_in(10);
        let b = uuid_bytes(r, v7.then_some(ts), digits);
        uuid(&b, upper)
    }));
    let b = uuid_bytes(r, None, false);
    let mut mixed = uuid(&b, false).into_bytes();
    mixed[0] = b'A';
    mixed[35] = b'b';
    let extra = vec![
        uuid(&b, !upper),
        String::from_utf8(mixed).unwrap(),
        format!("{{{}}}", &uuid(&b, upper)[1..35]),
        hex(&b, upper),
        uuid(&b, upper).replacen('-', "_", 1),
    ];
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn gen_ulid(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 16, 33]);
    let mut ts = match r.below(3) {
        0 => 0,
        1 => (1 << 48) - 1000,
        _ => r.below(1 << 48),
    };
    let mut ent = u128::from(r.next()) << 64 | u128::from(r.next());
    let mono = r.one_in(2);
    let mut rows = owned((0..n).map(|_| {
        ts = (ts + r.below(3)).min((1 << 48) - 1);
        ent = if mono {
            ent + 1
        } else {
            u128::from(r.next()) << 64 | u128::from(r.next())
        };
        ulid(ts, ent)
    }));
    let u = ulid(ts, ent);
    let extra = vec![
        u.to_lowercase(),
        format!("8{}", &u[1..]),
        format!("{}U", &u[..25]),
        u[..25].to_string(),
        "00000000000000000000000000".into(),
        "7ZZZZZZZZZZZZZZZZZZZZZZZZZ".into(),
    ];
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn gen_ipv4(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 9, 17, 40]);
    let mode = r.below(4);
    let octet = |r: &mut Rng| -> u32 {
        if r.one_in(2) {
            r.pick(&[0, 1, 9, 10, 99, 100, 199, 200, 255])
        } else {
            r.below(256) as u32
        }
    };
    let mut rows = owned((0..n).map(|_| {
        let ip = format!("{}.{}.{}.{}", octet(r), octet(r), octet(r), octet(r));
        let m = if mode == 3 { r.below(3) } else { mode };
        match m {
            1 => {
                let any = r.below(65536);
                format!("{ip}:{}", r.pick(&[0, 1, 9, 10, 80, 443, 8080, 65535, any]))
            }
            2 => format!("{ip}/{}", r.below(33)),
            _ => ip,
        }
    }));
    let extra: Vec<String> = [
        "256.1.1.1",
        "1.2.3",
        "01.2.3.4",
        "1.2.3.4:",
        "1.2.3.4:65536",
        "1.2.3.4:080",
        "1.2.3.4.5",
        "1.2.3.4:-1",
        "1.2.3.4/24:80",
        "1.2.3.4 ",
    ]
    .map(String::from)
    .to_vec();
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn gen_hex(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 16, 33]);
    let w = r.pick(&[16usize, 32, 40, 64]);
    let upper = r.one_in(3);
    let mut prev = r.bytes(w / 2);
    let mut rows = owned((0..n).map(|_| {
        if !r.one_in(4) {
            prev = r.bytes(w / 2);
        }
        if r.one_in(12) {
            prev.iter()
                .map(|b| (b % 10 + b'0') as char)
                .chain(prev.iter().map(|b| (b / 26 % 10 + b'0') as char))
                .collect()
        } else {
            hex(&prev, upper)
        }
    }));
    let other = hex(&prev, !upper);
    let mut mixed = hex(&prev, false).into_bytes();
    mixed[0] = b'A';
    mixed[1] = b'b';
    let extra = vec![
        other,
        String::from_utf8(mixed).unwrap(),
        format!("g{}", &hex(&prev, upper)[1..]),
        hex(&prev[1..], upper),
    ];
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn traceparent(tr: &[u8], sp: &[u8], fl: u8) -> String {
    format!("00-{}-{}-{fl:02x}", hex(tr, false), hex(sp, false))
}

fn gen_tp(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 16, 33]);
    let mut tr = r.bytes(16);
    let mut rows = owned((0..n).map(|_| {
        if r.one_in(3) {
            tr = r.bytes(16);
        }
        let any = r.next() as u8;
        let fl = r.pick(&[0u8, 1, 0xff, any]);
        traceparent(&tr, &r.bytes(8), fl)
    }));
    let t = traceparent(&tr, &r.bytes(8), 1);
    let extra = vec![
        t.to_uppercase(),
        format!("01{}", &t[2..]),
        t[..54].to_string(),
        t.replacen('-', "_", 1),
    ];
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

fn gen_split(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[8usize, 9, 16, 17, 40]);
    let mode = r.below(7);
    let mut rows = owned((0..n).map(|i| {
        match mode {
            0 => format!(
                "user{}@{}",
                r.below(50),
                r.pick(&["example.com", "mail.org", "x.io"])
            ),
            1 => format!(
                "/var/log/{}/{}.log",
                r.pick(&["nginx", "app", "db"]),
                r.below(9)
            ),
            2 => {
                let any = r.below(70000);
                format!(
                    "{}:{}",
                    r.pick(&["host-a", "host-b", "10.0.0.1"]),
                    r.pick(&[80, 443, 8080, any])
                )
            }
            3 => format!(
                "{}.{}",
                r.pick(&["www", "api", "cdn", "mx1"]),
                r.pick(&["example.com", "test.org"])
            ),
            4 => format!("key:{}", i64::from(i as i32) * 3 - 7),
            5 => format!(
                "{}/{}",
                r.pick(&["https://a.b", "s3://bucket"]),
                r.below(100)
            ),
            _ => (0..r.range(1, 12))
                .map(|_| r.pick(&['a', 'b', '1', '@', '/', ':', '.', '0']))
                .collect(),
        }
    }));
    let extra: Vec<String> = ["nodelim", "@x", "x@", "/", ":", ".", "a@b@c", "key:007"]
        .map(String::from)
        .to_vec();
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

/// One random shape of literals and numeric/hex tokens, instantiated per row.
fn gen_template_shape(r: &mut Rng) -> Vec<Vec<u8>> {
    const LITS: &[&str] = &[
        "blk_", " ", "-", "=", "/", ":", "x", "id", ".", ",", "ms", "_", "g", "a", "--", "0",
    ];
    let n = r.pick(&[8usize, 9, 16, 17, 40]);
    // Piece kinds: 0 literal, 1 int, 2 negative int, 3 zero-led int, 4 hex, 5 hex either case.
    let shape: Vec<(u64, &str, usize)> = (0..r.range(1, 9))
        .map(|_| (r.below(6), r.pick(LITS), r.pick(&[8usize, 16, 32, 40, 64])))
        .collect();
    let upper = r.one_in(2);
    owned((0..n).map(|_| {
        let mut s = String::new();
        for &(kind, lit, w) in &shape {
            match kind {
                0 => s += lit,
                1 => s += &(r.next() >> r.below(64)).to_string(),
                2 => s += &format!("-{}", r.next() >> r.below(64)),
                3 => s += &format!("{:06}", r.below(1000)),
                4 => s += &hex(&r.bytes(w / 2), upper),
                _ => s += &hex(&r.bytes(w / 2), r.one_in(2)),
            }
        }
        s
    }))
}

fn gen_template(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[8usize, 9, 16, 17, 40]);
    let mode = r.below(9);
    if mode >= 7 {
        return gen_template_shape(r);
    }
    let upper = r.one_in(3);
    let mut k = r.range(-1000, 1000);
    let lcg = lcg_column(r, n);
    let mut rows = owned((0..n).map(|i| {
        k += r.range(0, 3);
        match mode {
            0 => format!("blk_{}", r.next() as i64 >> r.below(64)),
            1 => format!(
                "user {k} from {}.{}.{}.{} port {}",
                r.below(256),
                r.below(256),
                r.below(256),
                r.below(256),
                r.pick(&[0, 80, 65535])
            ),
            2 => format!(
                "trace={} span={}",
                hex(&r.bytes(16), upper),
                hex(&r.bytes(8), upper)
            ),
            3 => format!("req-{k}-{}", hex(&r.bytes(20), false)),
            4 => format!(
                "arn:aws:iam::{}:role/svc{}",
                r.pick(&[123_456_789_012u64, 999_999_999_999]),
                r.below(4)
            ),
            5 => format!("took {} ms at -{}", r.below(10_000), r.range(1, 99)),
            _ => format!("id={}", String::from_utf8_lossy(&lcg[i])),
        }
    }));
    let extra: Vec<String> = [
        "hello world",
        "user x from y",
        "blk_",
        "blk_-0",
        "blk_007",
        "trace=ABCDEFabcdef0123",
    ]
    .map(String::from)
    .to_vec();
    sprinkle(r, &mut rows, 20, &extra);
    rows
}

fn gen_b64(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[4usize, 5, 8, 16, 33, 40]);
    let url = r.one_in(2);
    let pad = !url || r.one_in(2);
    let len = r.pick(&[12usize, 13, 14, 16, 18, 32, 33, 34, 48]);
    let pool: Vec<Vec<u8>> = (0..3).map(|_| r.bytes(len)).collect();
    let dedup = r.one_in(3);
    let mut rows = owned((0..n).map(|_| {
        let data = if dedup {
            pool[r.below(3) as usize].clone()
        } else {
            r.bytes(len)
        };
        b64(&data, url, pad)
    }));
    let extra: Vec<String> = [
        "short==",
        "@@@@AAAAAAAAAAAAAAAA",
        "AAAAAAAAAAAAAB==",
        "AAAAAAAAAAAAAAB=",
        "AAAAAAAAAAAAAAAAA",
        "AAAAAAAA=AAAAAAA",
        "AAAAAAAAAAAA====",
        "AAAAAAAAAAAAA===",
    ]
    .map(String::from)
    .to_vec();
    sprinkle(r, &mut rows, 16, &extra);
    rows
}

const F64_SPECIALS: &[u64] = &[
    0x0000_0000_0000_0000, // 0.0
    0x8000_0000_0000_0000, // -0.0
    0x7ff0_0000_0000_0000, // inf
    0xfff0_0000_0000_0000, // -inf
    0x7ff8_0000_0000_0000, // NaN
    0xfff8_0000_0000_0001, // -NaN with payload
    0x7ff0_0000_0000_0001, // signalling NaN
    0x0000_0000_0000_0001, // min subnormal
    0x7fef_ffff_ffff_ffff, // MAX
    0xffef_ffff_ffff_ffff, // MIN
];

fn gen_f64(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[1usize, 2, 7, 8, 15, 16, 17, 64, 100]);
    let mode = r.below(4);
    let x0 = r.pick(&[0.0f64, 1.5, -1e300, 123.456, 1e-310]);
    let step = r.pick(&[0.5f64, 0.1, -0.25, 1e10]);
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let b = match mode {
            0 => r.next(),
            1 => (x0 + (i as f64 * 0.1).sin() * 100.0).to_bits(),
            2 => (x0 + step * i as f64).to_bits(),
            _ => (r.range(-1000, 1000) as f64 / 8.0).to_bits(),
        };
        let b = if r.one_in(10) {
            r.pick(F64_SPECIALS)
        } else {
            b
        };
        rows.push(b.to_le_bytes().to_vec());
    }
    rows
}

fn gen_planes(r: &mut Rng) -> Vec<Vec<u8>> {
    let n = r.pick(&[16usize, 17, 31, 64]);
    let w = r.pick(&[2usize, 4]);
    let mode = r.below(3);
    (0..n)
        .map(|i| match (w, mode) {
            (4, 0) => (100.0f32 + (i as f32 * 0.3).sin()).to_le_bytes().to_vec(),
            (4, 1) => {
                let s = [
                    0u32,
                    0x8000_0000,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7fc0_0001,
                    1,
                    0x7f7f_ffff,
                ];
                let v = if r.one_in(3) {
                    r.pick(&s)
                } else {
                    r.next() as u32
                };
                v.to_le_bytes().to_vec()
            }
            (2, 0) => ((i as u16).wrapping_mul(3) ^ 0x8000).to_le_bytes().to_vec(),
            _ => r.bytes(w),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ExactIntPack
// ---------------------------------------------------------------------------

#[test]
fn exact_ints_cover_the_whole_i64_range() {
    let mut vals = vec![0, 1, -1, 9, 10, 99_999_999, 100_000_000, i64::MAX, i64::MIN];
    for p in 1..=18 {
        let x = 10i64.pow(p);
        vals.extend([x - 1, x, x + 1, -x, 1 - x, 7 * (x / 10) + 3]);
    }
    vals.extend((0..64).map(|i| i64::MAX - i * 0x0123_4567_89ab_cdef));
    let rows = texts(vals);
    roundtrip(&ExactIntPack, &refs(&rows));
}

#[test]
fn exact_ints_patch_non_canonical_rows() {
    // Seven clean rows allow at most one exception per eight rows.
    let mut rows = texts((0..16).map(|i| 1000 + i * 7));
    rows[0] = b"-0".to_vec();
    rows[9] = b"+5".to_vec();
    roundtrip(&ExactIntPack, &refs(&rows));
    for junk in [
        "",
        "-",
        "007",
        " 1",
        "1e3",
        "99999999999999999999",
        "-9223372036854775809",
    ] {
        let mut rows = texts(0..8);
        rows[3] = junk.as_bytes().to_vec();
        roundtrip(&ExactIntPack, &refs(&rows));
    }
}

#[test]
fn exact_ints_refuse_short_or_mostly_junk_columns() {
    refuses_strs(&ExactIntPack, &["1", "2", "3"]);
    refuses_strs(&ExactIntPack, &[]);
    refuses_strs(&ExactIntPack, &["1", "x", "y", "4"]);
    refuses_strs(&ExactIntPack, &["a", "b", "c", "d"]);
}

#[test]
fn exact_ints_keep_zero_padding_at_every_width() {
    for w in 2..=16usize {
        let max = 10u64.pow(w as u32) - 1;
        let vals = [0, 1, 9, max / 2, max - 1, max, 7, 10u64.pow(w as u32 - 1)];
        let rows = owned(vals.iter().map(|v| format!("{v:0w$}")));
        roundtrip(&ExactIntPack, &refs(&rows));
    }
}

#[test]
fn exact_ints_zero_padded_width_sixteen_near_a_block_boundary() {
    // Smallest value the padded formatter splits wrongly at width 16.
    roundtrip_strs(
        &ExactIntPack,
        &[
            "0000203999999999",
            "0000000000000001",
            "0000000000000002",
            "0000000000000003",
        ],
    );
}

#[test]
fn exact_ints_zero_padded_patch_other_widths_and_signs() {
    let mut rows = owned((1..=16).map(|i| format!("{i:04}")));
    rows[2] = b"-003".to_vec();
    rows[15] = b"12345".to_vec();
    roundtrip(&ExactIntPack, &refs(&rows));
    // Too many misfits for the padded form fall back to canonical ints.
    roundtrip_strs(
        &ExactIntPack,
        &["05", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    );
}

#[test]
fn exact_ints_fuzz() {
    fuzz(&ExactIntPack, 1, 400, 250, gen_ints);
}

// ---------------------------------------------------------------------------
// DecimalScaledPack
// ---------------------------------------------------------------------------

#[test]
fn decimals_roundtrip_at_every_scale() {
    for scale in 1..=8 {
        let vals = [
            0i64,
            1,
            -1,
            5,
            -5,
            10i64.pow(scale) - 1,
            -(10i64.pow(scale)),
            12_345,
            999_999,
            -7,
        ];
        let rows = owned(vals.iter().map(|&v| dec(v, scale)));
        roundtrip(&DecimalScaledPack, &refs(&rows));
    }
}

#[test]
fn decimals_patch_other_scales_and_non_canonical_forms() {
    roundtrip_strs(
        &DecimalScaledPack,
        &[
            "1.25", "2.50", "3.75", "-0.05", "0.00", "1.5", "10.00", "11.00", "-0.0", "12.00",
            "13.00", "14.00", "15.00", "16.00", "17.00", "18.00",
        ],
    );
    for junk in ["1.", ".5", "01.5", "-0.00", "1e5", "", "1.2.3"] {
        let mut rows = owned((0..8).map(|i| dec(i * 3 + 100, 2)));
        rows[5] = junk.as_bytes().to_vec();
        roundtrip(&DecimalScaledPack, &refs(&rows));
    }
}

#[test]
fn decimals_refuse_integers_and_sparse_columns() {
    refuses_strs(&DecimalScaledPack, &["1.5", "2.5", "3.5"]);
    refuses_strs(&DecimalScaledPack, &["1", "2", "3", "4"]);
    refuses_strs(&DecimalScaledPack, &["1.5", "2.5", "x", "4.5"]);
}

#[test]
fn decimals_with_seventeen_significant_digits() {
    for scale in 1..=8 {
        let big = 99_999_999_999_999_999i64;
        let vals = [
            big,
            -big,
            big - 1,
            big / 2,
            10i64.pow(16),
            -(10i64.pow(16)) - 1,
            1,
            0,
        ];
        let rows = owned(vals.iter().map(|&v| dec(v, scale)));
        roundtrip(&DecimalScaledPack, &refs(&rows));
    }
}

#[test]
fn decimals_scale_eight_just_past_two_thousand() {
    // Smallest magnitude whose integer/fraction split goes wrong at scale 8.
    roundtrip_strs(
        &DecimalScaledPack,
        &["2039.99999999", "0.00000001", "0.00000002", "0.00000003"],
    );
}

#[test]
fn decimals_fuzz() {
    fuzz(&DecimalScaledPack, 2, 400, 250, gen_decimals);
}

// ---------------------------------------------------------------------------
// JavaLcgPack
// ---------------------------------------------------------------------------

#[test]
fn java_lcg_ids_are_steps_on_one_cycle() {
    // Two instances far apart on the cycle, each id logged again a few rows later.
    let (mut a, mut b) = (0x1234_5678_9abcu64, 0x0fed_cba9_8765u64);
    let next = |s: &mut u64| {
        let v = JavaLcgPack::output(*s);
        *s = s.wrapping_mul(LCG_MULT).wrapping_add(LCG_ADD) & LCG_MASK;
        // `nextLong()` draws twice.
        *s = s.wrapping_mul(LCG_MULT).wrapping_add(LCG_ADD) & LCG_MASK;
        v
    };
    let mut vals = Vec::new();
    for i in 0..600 {
        vals.push(if i % 7 == 3 {
            next(&mut b)
        } else {
            next(&mut a)
        });
        if i % 3 == 0 {
            vals.push(vals[vals.len() - 1 - i % 5]);
        }
    }
    assert_eq!(JavaLcgPack::state_of(vals[0]), Some(0x1234_5678_9abc));
    let third = JavaLcgPack::state_of(vals[2]).unwrap();
    assert_eq!(JavaLcgPack::distance(0x1234_5678_9abc, third), 2);
    let mut rows = texts(vals);
    rows[40] = b"not-a-number".to_vec();
    rows[41] = b"12345678901234".to_vec();
    let raw: usize = rows.iter().map(Vec::len).sum();
    let size = roundtrip(&JavaLcgPack, &refs(&rows));
    assert!(size * 3 < raw, "lcg size={size} of {raw}");
}

/// `s` after `k` steps of the generator, backwards when negative.
fn lcg_jump(s: u64, k: i64) -> u64 {
    let (mut m, mut a) = (LCG_MULT, LCG_ADD);
    let (mut am, mut ac) = (1u64, 0u64);
    let mut n = k as u64 & LCG_MASK;
    while n > 0 {
        if n & 1 == 1 {
            am = am.wrapping_mul(m);
            ac = ac.wrapping_mul(m).wrapping_add(a);
        }
        a = a.wrapping_mul(m.wrapping_add(1));
        m = m.wrapping_mul(m);
        n >>= 1;
    }
    s.wrapping_mul(am).wrapping_add(ac) & LCG_MASK
}

#[test]
fn java_lcg_handles_state_extremes_and_a_patched_first_row() {
    for s0 in [0u64, 1, LCG_MASK, 1 << 47, (1 << 47) - 1] {
        let mut s = s0;
        let mut vals = Vec::new();
        for _ in 0..9 {
            vals.push(JavaLcgPack::output(s));
            s = lcg_jump(s, 2);
        }
        let mut rows = texts(vals);
        roundtrip(&JavaLcgPack, &refs(&rows));
        rows[0] = b"x".to_vec();
        roundtrip(&JavaLcgPack, &refs(&rows));
    }
}

#[test]
fn java_lcg_jumps_both_ways_round_the_cycle() {
    let mut r = Rng::new(9);
    for _ in 0..64 {
        let (s, k) = (r.below(1 << 48), r.next() as i64 >> 16);
        let t = lcg_jump(s, k);
        assert_eq!(JavaLcgPack::distance(s, t), k, "s={s} k={k}");
        assert_eq!(JavaLcgPack::state_of(JavaLcgPack::output(t)), Some(t));
    }
}

#[test]
fn java_lcg_refuses_unrelated_ints() {
    let mut r = Rng::new(3);
    let rows = texts((0..32).map(|_| r.next() as i64));
    refuses(&JavaLcgPack, &refs(&rows));
    refuses(&JavaLcgPack, &refs(&texts(0..7)));
}

#[test]
fn java_lcg_fuzz() {
    fuzz(&JavaLcgPack, 3, 200, 150, gen_lcg);
}

// ---------------------------------------------------------------------------
// PythonMtPack (and the shared MT19937 matcher)
// ---------------------------------------------------------------------------

#[test]
fn python_mt_decimal_words_regrow_from_their_head() {
    let words = mt_words(42, 700);
    let rows = texts(words.iter().map(|&w| i64::from(w)));
    roundtrip_generated(&PythonMtPack, &refs(&rows));
    // One word off the stream sends the column elsewhere.
    let mut rows = rows;
    rows[650] = b"1".to_vec();
    refuses(&PythonMtPack, &refs(&rows));
}

#[test]
fn python_mt_fixed_binary_draws_in_both_byte_orders() {
    let words = mt_words(7, 1600);
    for per_row in [1usize, 2, 4, 16] {
        let n = 1600 / per_row;
        let w = &words[..n * per_row];
        for mut rows in [le_rows(w, per_row), be_rows(w, per_row)] {
            roundtrip_generated(&PythonMtPack, &refs(&rows));
            // Rows off the stream are patched.
            rows[n / 2] = vec![0; 4 * per_row];
            rows[n - 1] = vec![0xff; 4 * per_row];
            roundtrip_generated(&PythonMtPack, &refs(&rows));
        }
    }
}

#[test]
fn python_mt_finds_seeded_streams_mid_way() {
    let words = mt_words(9, 2000);
    // Big enough to clear the pack's half-size gate, short enough to need the seed search.
    let rows = le_rows(&words[100..100 + 100 * 4], 4);
    let mut seeds = Seeds::default();
    with_seeds(&mut seeds, || {
        roundtrip_generated(&PythonMtPack, &refs(&rows));
        let rows = be_rows(&words[600..600 + 150 * 2], 2);
        roundtrip_generated(&PythonMtPack, &refs(&rows));
    });
}

#[test]
fn python_mt_refuses_random_bytes_and_empty_columns() {
    let mut r = Rng::new(5);
    let rows: Vec<Vec<u8>> = (0..64).map(|_| r.bytes(8)).collect();
    refuses(&PythonMtPack, &refs(&rows));
    refuses(&PythonMtPack, &refs(&texts(0..700)));
}

#[test]
fn python_mt_fuzz() {
    let mut r = Rng::new(6);
    for case in 0..12 {
        let seed = r.below(1 << 32) as u32;
        let skip = r.below(500) as usize;
        let per_row = r.pick(&[1usize, 2, 3, 4, 8]);
        // The 624-word head costs ~2.5 KB, so short streams fail the half-size gate.
        let n = (1400 + r.below(600) as usize) / per_row;
        let words = mt_words(seed, skip + n * per_row);
        let mut rows = if r.one_in(2) {
            le_rows(&words[skip..], per_row)
        } else {
            be_rows(&words[skip..], per_row)
        };
        // Rows inside the 624-word head would hide the stream.
        let head = MT_N / per_row + 1;
        for _ in 0..r.below(4) {
            let i = head + r.below((rows.len() - head) as u64) as usize;
            rows[i] = r.bytes(4 * per_row);
        }
        let got = check(&PythonMtPack, &refs(&rows));
        assert_eq!(
            got,
            Ok(true),
            "case {case} seed={seed} skip={skip} per_row={per_row}"
        );
    }
}

// ---------------------------------------------------------------------------
// TimestampIsoPack
// ---------------------------------------------------------------------------

#[test]
fn iso_timestamps_cross_calendar_boundaries() {
    roundtrip_strs(
        &TimestampIsoPack,
        &[
            "1970-01-01T00:00:00Z",
            "1999-12-31T23:59:59Z",
            "2000-01-01T00:00:00Z",
            "2000-02-29T12:00:00Z",
            "2023-02-28T23:59:59Z",
            "2023-03-01T00:00:00Z",
            "2024-02-29T00:00:00Z",
            "2024-12-31T23:59:59Z",
            "2100-02-28T00:00:00Z",
            "2100-03-01T00:00:00Z",
            "9999-12-31T23:59:59Z",
            "2024-04-30T23:59:59Z",
        ],
    );
}

#[test]
fn iso_timestamps_every_layout() {
    let t0 = secs_of(2024, 2, 28, 23, 59, 57);
    for sep in ['T', 't', ' '] {
        for tz in ["", "Z", "z", "+05:30", "-08:00", "+0000", "-0130"] {
            for digits in 0..=9usize {
                for fsep in ['.', ','] {
                    let rows = owned((0..6).map(|i| {
                        let sub = (i as u64 * 987_654_321) % 10u64.pow(digits as u32);
                        iso(t0 + i, sep, (digits > 0).then_some((fsep, digits, sub)), tz)
                    }));
                    roundtrip(&TimestampIsoPack, &refs(&rows));
                }
            }
        }
    }
}

#[test]
fn iso_timestamps_patch_other_layouts_and_impossible_dates() {
    // The first parsed row fixes the layout, so odd rows start at row 1.
    let base = owned((0..16).map(|i| {
        iso(
            1_700_000_000 + i * 61,
            'T',
            Some(('.', 3, i as u64 * 7)),
            "Z",
        )
    }));
    for odd in [
        "2023-02-29T00:00:00.000Z",
        "2024-04-31T00:00:00.000Z",
        "2024-01-01T24:00:00.000Z",
        "1969-12-31T23:59:59.000Z",
        "2024-01-01T00:00:00.00Z",
        "2024-01-01T00:00:00.000+00:00",
        "2024-01-01 00:00:00.000Z",
        "",
    ] {
        for at in [1, 7, 15] {
            let mut rows = base.clone();
            rows[at] = odd.as_bytes().to_vec();
            roundtrip(&TimestampIsoPack, &refs(&rows));
        }
    }
}

#[test]
fn iso_timestamps_refuse_short_or_foreign_columns() {
    refuses_strs(&TimestampIsoPack, &["2024-01-01T00:00:00Z"; 3]);
    refuses_strs(
        &TimestampIsoPack,
        &["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    );
    refuses_strs(&TimestampIsoPack, &["2023-02-29T00:00:00", "x", "y", "z"]);
}

#[test]
fn iso_timestamps_fuzz() {
    fuzz(&TimestampIsoPack, 4, 400, 250, gen_iso);
}

// ---------------------------------------------------------------------------
// TimestampClfPack
// ---------------------------------------------------------------------------

#[test]
fn clf_timestamps_cross_calendar_boundaries() {
    roundtrip_strs(
        &TimestampClfPack,
        &[
            "01/Jan/1970:00:00:00 -0700",
            "31/Dec/1999:23:59:59 -0700",
            "01/Jan/2000:00:00:00 -0700",
            "29/Feb/2000:12:00:00 -0700",
            "28/Feb/2023:23:59:59 -0700",
            "01/Mar/2023:00:00:00 -0700",
            "29/Feb/2024:00:00:00 -0700",
            "30/Apr/2024:23:59:59 -0700",
            "31/Dec/9999:23:59:59 -0700",
        ],
    );
}

#[test]
fn clf_timestamps_patch_other_zones_and_malformed_rows() {
    // The first parsed row fixes the zone, so odd rows start at row 1.
    let base = owned((0..16).map(|i| clf(1_700_000_000 + i * 3, "+0530")));
    for odd in [
        "17/Nov/2023:00:00:00 +0000",
        "17/nov/2023:00:00:00 +0530",
        "17/Nov/2023:24:00:00 +0530",
        "31/Dec/1969:23:59:59 +0530",
        "17/Nov/2023:00:00:00 +05:3",
        "",
    ] {
        for at in [1, 7, 15] {
            let mut rows = base.clone();
            rows[at] = odd.as_bytes().to_vec();
            roundtrip(&TimestampClfPack, &refs(&rows));
        }
    }
}

#[test]
fn clf_day_past_month_end_is_patched() {
    roundtrip_strs(
        &TimestampClfPack,
        &[
            "31/Feb/2024:00:00:00 +0000",
            "01/Mar/2024:00:00:00 +0000",
            "02/Mar/2024:00:00:00 +0000",
            "03/Mar/2024:00:00:00 +0000",
            "04/Mar/2024:00:00:00 +0000",
            "05/Mar/2024:00:00:00 +0000",
            "06/Mar/2024:00:00:00 +0000",
            "07/Mar/2024:00:00:00 +0000",
        ],
    );
}

#[test]
fn syslog_timestamps_space_pad_single_digit_days() {
    roundtrip_strs(
        &TimestampClfPack,
        &[
            "Jan  1 00:00:00",
            "Jan  9 23:59:59",
            "Jan 10 00:00:00",
            "Feb 29 12:00:00",
            "Mar  1 00:00:00",
            "Dec 31 23:59:59",
            "Jan  1 00:00:01",
            "Apr 30 01:02:03",
        ],
    );
}

#[test]
fn syslog_timestamps_patch_malformed_rows() {
    let base = owned((0..16).map(|i| syslog(1_700_000_000 + i * 86_400 * 3)));
    for odd in [
        "Jan  0 00:00:00",
        "jan  1 00:00:00",
        "Jan 1 00:00:00",
        "Jan 32 00:00:00",
        "Jan  1 24:00:00",
        // A zero-padded day and days past the month's end must not be rewritten.
        "Jan 05 00:00:00",
        "Apr 31 00:00:00",
        "Feb 30 00:00:00",
        "",
    ] {
        for at in [1, 15] {
            let mut rows = base.clone();
            rows[at] = odd.as_bytes().to_vec();
            roundtrip(&TimestampClfPack, &refs(&rows));
        }
    }
}

#[test]
fn clf_timestamps_refuse_foreign_columns() {
    refuses_strs(&TimestampClfPack, &["Jan  1 00:00:00"; 3]);
    refuses_strs(&TimestampClfPack, &["a", "b", "c", "d"]);
}

#[test]
fn clf_timestamps_fuzz() {
    fuzz(&TimestampClfPack, 5, 300, 200, gen_clf);
}

#[test]
fn syslog_timestamps_fuzz() {
    fuzz(&TimestampClfPack, 6, 300, 200, gen_syslog);
}

// ---------------------------------------------------------------------------
// UuidPack
// ---------------------------------------------------------------------------

#[test]
fn uuids_v4_both_cases_and_digit_only() {
    let mut r = Rng::new(10);
    for upper in [false, true] {
        // Digit-only rows count as lower case, so an upper-case column patches them.
        let mut rows = owned((0..40).map(|i| uuid(&uuid_bytes(&mut r, None, i % 10 == 3), upper)));
        rows.push(b"00000000-0000-0000-0000-000000000000".to_vec());
        rows.push(
            if upper {
                "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"
            } else {
                "ffffffff-ffff-ffff-ffff-ffffffffffff"
            }
            .into(),
        );
        roundtrip(&UuidPack, &refs(&rows));
    }
}

#[test]
fn uuids_v7_split_timestamp_and_tail() {
    let mut r = Rng::new(11);
    for start in [0u64, 1_700_000_000_000, (1 << 48) - 100] {
        let mut ts = start;
        let mut rows = owned((0..16).map(|i| {
            ts = (ts + r.below(3)).min((1 << 48) - 1);
            // A quarter v4 rows still ride the v7 split.
            uuid(
                &uuid_bytes(&mut r, (i % 4 != 0).then_some(ts), false),
                false,
            )
        }));
        roundtrip(&UuidPack, &refs(&rows));
        rows[3] = b"not-a-uuid".to_vec();
        roundtrip(&UuidPack, &refs(&rows));
    }
}

#[test]
fn uuids_patch_mixed_case_and_malformed_rows() {
    // The first parsed row fixes the case, so odd rows start at row 1.
    let mut r = Rng::new(12);
    let base = owned((0..16).map(|_| uuid(&uuid_bytes(&mut r, None, false), false)));
    let b = uuid_bytes(&mut r, None, false);
    for odd in [
        uuid(&b, true),
        format!("A{}", &uuid(&b, false)[1..35]) + "b",
        hex(&b, false),
        uuid(&b, false).replacen('-', "_", 1),
        format!("{}g", &uuid(&b, false)[..35]),
        String::new(),
    ] {
        for at in [1, 15] {
            let mut rows = base.clone();
            rows[at] = odd.clone().into_bytes();
            roundtrip(&UuidPack, &refs(&rows));
        }
    }
}

#[test]
fn uuids_drawn_from_mt_streams() {
    let words = mt_words(21, 1600);
    let rows = owned(
        le_rows(&words[..400 * 4], 4)
            .iter()
            .map(|b| uuid(b[..].try_into().unwrap(), false)),
    );
    roundtrip_generated(&UuidPack, &refs(&rows));
}

#[test]
fn uuid7_tails_drawn_from_seeded_python_random() {
    let words = mt_words(3, 600);
    let mut ts = 1_700_000_000_000u64;
    let mut rows = owned(words.chunks_exact(3).take(40).map(|w| {
        let mut tail = [0u8; 10];
        MtLayout::Uuid7.render(w, &[], &mut tail);
        ts += 1;
        let mut b = [0u8; 16];
        b[..6].copy_from_slice(&ts.to_be_bytes()[2..]);
        b[6..].copy_from_slice(&tail);
        uuid(&b, false)
    }));
    rows[17] = uuid(&uuid_bytes(&mut Rng::new(1), Some(ts), false), false).into_bytes();
    let mut seeds = Seeds::default();
    with_seeds(&mut seeds, || roundtrip_generated(&UuidPack, &refs(&rows)));
}

#[test]
fn uuids_refuse_short_or_foreign_columns() {
    refuses_strs(&UuidPack, &["00000000-0000-0000-0000-000000000000"; 3]);
    refuses_strs(&UuidPack, &["a", "b", "c", "d"]);
}

#[test]
fn uuids_fuzz() {
    fuzz(&UuidPack, 7, 300, 200, gen_uuid);
}

// ---------------------------------------------------------------------------
// UlidPack
// ---------------------------------------------------------------------------

#[test]
fn ulids_cover_timestamp_and_entropy_extremes() {
    roundtrip_strs(
        &UlidPack,
        &[
            "00000000000000000000000000",
            "7ZZZZZZZZZZZZZZZZZZZZZZZZZ",
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "01ARZ3NDEKTSV4RRFFQ69G5FAW",
            "7ZZZZZZZZZ0000000000000000",
            "0000000000ZZZZZZZZZZZZZZZZ",
            "01BX5ZZKBKACTAV9WEVGEMMVRZ",
        ],
    );
}

#[test]
fn ulids_patch_non_crockford_rows() {
    let mut r = Rng::new(13);
    let base = owned((0..16).map(|i| ulid(1_700_000_000_000 + i, u128::from(r.next()) << 16)));
    for odd in [
        "01arz3ndektsv4rrffq69g5fav",
        "81ARZ3NDEKTSV4RRFFQ69G5FAV",
        "01ARZ3NDEKTSV4RRFFQ69G5FAU",
        "01ARZ3NDEKTSV4RRFFQ69G5FAI",
        "01ARZ3NDEKTSV4RRFFQ69G5FA",
        "",
    ] {
        for at in [0, 15] {
            let mut rows = base.clone();
            rows[at] = odd.as_bytes().to_vec();
            roundtrip(&UlidPack, &refs(&rows));
        }
    }
}

#[test]
fn ulids_refuse_mostly_foreign_columns() {
    refuses_strs(&UlidPack, &["01ARZ3NDEKTSV4RRFFQ69G5FAV"; 3]);
    refuses_strs(&UlidPack, &["a", "b", "c", "01ARZ3NDEKTSV4RRFFQ69G5FAV"]);
}

#[test]
fn ulids_fuzz() {
    fuzz(&UlidPack, 8, 300, 200, gen_ulid);
}

// ---------------------------------------------------------------------------
// Ipv4Pack
// ---------------------------------------------------------------------------

#[test]
fn ipv4_bare_addresses_cover_octet_widths() {
    roundtrip_strs(
        &Ipv4Pack,
        &[
            "0.0.0.0",
            "255.255.255.255",
            "1.22.133.4",
            "10.0.0.1",
            "99.100.199.200",
            "9.9.9.9",
        ],
    );
}

#[test]
fn ipv4_ports_and_prefixes_mix_with_bare_addresses() {
    roundtrip_strs(
        &Ipv4Pack,
        &[
            "10.0.0.1:0",
            "10.0.0.1:65535",
            "10.0.0.1/0",
            "10.0.0.1/32",
            "255.255.255.255:65535",
            "0.0.0.0/0",
            "10.0.0.1",
            "1.2.3.4:80",
            "1.2.3.4/65535",
        ],
    );
}

#[test]
fn ipv4_patch_malformed_rows() {
    let base = owned((0..16).map(|i| format!("192.168.{}.{}:{}", i / 4, i * 13 % 256, 8000 + i)));
    for odd in [
        "256.1.1.1",
        "1.2.3",
        "01.2.3.4",
        "1.2.3.4:",
        "1.2.3.4:65536",
        "1.2.3.4:080",
        "1.2.3.4.5",
        "1.2.3.4/24:80",
        "",
    ] {
        for at in [0, 15] {
            let mut rows = base.clone();
            rows[at] = odd.as_bytes().to_vec();
            roundtrip(&Ipv4Pack, &refs(&rows));
        }
    }
}

#[test]
fn ipv4_refuses_foreign_columns() {
    refuses_strs(&Ipv4Pack, &["1.2.3.4"; 3]);
    refuses_strs(&Ipv4Pack, &["a", "b", "c", "d"]);
}

#[test]
fn ipv4_fuzz() {
    fuzz(&Ipv4Pack, 9, 400, 250, gen_ipv4);
}

// ---------------------------------------------------------------------------
// HexFixedPack
// ---------------------------------------------------------------------------

#[test]
fn hex_ids_every_width_and_case() {
    let mut r = Rng::new(14);
    for w in [16usize, 32, 40, 64] {
        for upper in [false, true] {
            let mut rows = owned((0..20).map(|_| hex(&r.bytes(w / 2), upper)));
            rows.push("0".repeat(w).into_bytes());
            rows.push("9".repeat(w).into_bytes());
            rows.push(if upper { "F" } else { "f" }.repeat(w).into_bytes());
            roundtrip(&HexFixedPack, &refs(&rows));
        }
    }
}

#[test]
fn hex_ids_patch_other_widths_and_cases() {
    // The first parsed row fixes the case, so odd rows start at row 1.
    let mut r = Rng::new(15);
    let base = owned((0..16).map(|_| hex(&r.bytes(20), false)));
    let b = r.bytes(20);
    for odd in [
        hex(&b, true),
        format!("A{}", &hex(&b, false)[1..]),
        hex(&b[..8], false),
        format!("g{}", &hex(&b, false)[1..]),
        String::new(),
    ] {
        for at in [1, 15] {
            let mut rows = base.clone();
            rows[at] = odd.clone().into_bytes();
            roundtrip(&HexFixedPack, &refs(&rows));
        }
    }
}

#[test]
fn hex_ids_drawn_from_mt_streams() {
    let words = mt_words(31, 4000);
    for per_row in [2usize, 4, 5, 8] {
        let n = 1000 / per_row;
        for upper in [false, true] {
            for mut bins in [
                le_rows(&words[..n * per_row], per_row),
                be_rows(&words[..n * per_row], per_row),
            ] {
                bins[n / 3] = vec![0; per_row * 4];
                let rows = owned(bins.iter().map(|b| hex(b, upper)));
                roundtrip_generated(&HexFixedPack, &refs(&rows));
            }
        }
    }
}

#[test]
fn hex_ids_refuse_foreign_columns() {
    refuses_strs(&HexFixedPack, &["0123456789abcdef"; 3]);
    refuses_strs(&HexFixedPack, &["a", "b", "c", "d"]);
    refuses_strs(&HexFixedPack, &["0123456789abcdeg"; 4]);
}

#[test]
fn hex_ids_fuzz() {
    fuzz(&HexFixedPack, 10, 300, 200, gen_hex);
}

// ---------------------------------------------------------------------------
// TraceparentPack
// ---------------------------------------------------------------------------

#[test]
fn traceparents_cover_flag_values() {
    let mut r = Rng::new(16);
    let rows = owned(
        [0u8, 1, 0xff, 0x0a, 0xa0, 0x10]
            .iter()
            .map(|&f| traceparent(&r.bytes(16), &r.bytes(8), f)),
    );
    roundtrip(&TraceparentPack, &refs(&rows));
}

#[test]
fn traceparents_patch_malformed_rows() {
    let mut r = Rng::new(17);
    let base = owned((0..16).map(|_| traceparent(&r.bytes(16), &r.bytes(8), 1)));
    let t = traceparent(&r.bytes(16), &r.bytes(8), 1);
    for odd in [
        t.to_uppercase(),
        format!("01{}", &t[2..]),
        t[..54].to_string(),
        t.replacen('-', "_", 2),
        String::new(),
    ] {
        for at in [0, 15] {
            let mut rows = base.clone();
            rows[at] = odd.clone().into_bytes();
            roundtrip(&TraceparentPack, &refs(&rows));
        }
    }
}

#[test]
fn traceparents_drawn_from_mt_streams() {
    let tw = mt_words(41, 400 * 4);
    let sw = mt_words(42, 400 * 2);
    let rows = owned(
        le_rows(&tw, 4)
            .iter()
            .zip(be_rows(&sw, 2))
            .map(|(t, s)| traceparent(t, &s, 1)),
    );
    roundtrip_generated(&TraceparentPack, &refs(&rows));
}

#[test]
fn traceparents_refuse_foreign_columns() {
    refuses_strs(&TraceparentPack, &["a", "b", "c", "d"]);
}

#[test]
fn traceparents_fuzz() {
    fuzz(&TraceparentPack, 11, 300, 200, gen_tp);
}

// ---------------------------------------------------------------------------
// StructuredSplitPack
// ---------------------------------------------------------------------------

#[test]
fn split_emails_paths_ports_and_domains() {
    let emails = owned((0..12).map(|i| format!("user{}@{}", i % 3, ["a.com", "b.org"][i % 2])));
    roundtrip(&StructuredSplitPack, &refs(&emails));
    let paths = owned((0..12).map(|i| format!("/var/log/app/{}.log", i % 4)));
    roundtrip(&StructuredSplitPack, &refs(&paths));
    let ports = owned((0..12).map(|i| format!("host-{}:{}", i % 2, 8000 + i)));
    roundtrip(&StructuredSplitPack, &refs(&ports));
    let fqdns = owned((0..12).map(|i| format!("{}.example.com", ["www", "api", "cdn"][i % 3])));
    roundtrip(&StructuredSplitPack, &refs(&fqdns));
    let ints = owned((0..12).map(|i| format!("key:{}", i as i64 * 1000 - 5000)));
    roundtrip(&StructuredSplitPack, &refs(&ints));
}

#[test]
fn split_patches_rows_without_a_usable_delimiter() {
    let base = owned((0..16).map(|i| format!("user{}@example.com", i % 3)));
    for odd in ["nodelim", "@x", "x@", "", "a@b@c"] {
        for at in [0, 15] {
            let mut rows = base.clone();
            rows[at] = odd.as_bytes().to_vec();
            roundtrip(&StructuredSplitPack, &refs(&rows));
        }
    }
}

#[test]
fn split_refuses_unique_halves_and_short_columns() {
    refuses_strs(&StructuredSplitPack, &["a@b"; 7]);
    let uniq = owned((0..16).map(|i| format!("u{i}@d{i}")));
    refuses(&StructuredSplitPack, &refs(&uniq));
}

#[test]
fn split_fuzz() {
    fuzz(&StructuredSplitPack, 12, 300, 150, gen_split);
}

// ---------------------------------------------------------------------------
// NumericTemplatePack
// ---------------------------------------------------------------------------

#[test]
fn templates_signed_block_ids() {
    let mut r = Rng::new(18);
    let rows = owned((0..16).map(|_| format!("blk_{}", r.next() as i64)));
    roundtrip(&NumericTemplatePack, &refs(&rows));
    let rows = owned((0..16).map(|i| format!("blk_{}", i * 7 - 50)));
    roundtrip(&NumericTemplatePack, &refs(&rows));
}

#[test]
fn templates_six_int_slots_and_trailing_literals() {
    let rows = owned((0..16).map(|i| {
        format!(
            "user {i} from 10.0.{}.{} port {} ok",
            i / 4,
            i * 9,
            8000 + i
        )
    }));
    roundtrip(&NumericTemplatePack, &refs(&rows));
}

#[test]
fn templates_hex_slots_both_cases() {
    let mut r = Rng::new(19);
    for upper in [false, true] {
        let rows = owned((0..16).map(|_| {
            format!(
                "trace={} span={}",
                hex(&r.bytes(16), upper),
                hex(&r.bytes(8), upper)
            )
        }));
        roundtrip(&NumericTemplatePack, &refs(&rows));
    }
    // A hex slot whose case flips on some rows.
    let rows = owned((0..16).map(|i| format!("id {}", hex(&r.bytes(20), i % 5 == 0))));
    roundtrip(&NumericTemplatePack, &refs(&rows));
}

#[test]
fn templates_lcg_slot() {
    let mut r = Rng::new(20);
    let ids = lcg_column(&mut r, 24);
    let rows = owned(
        ids.iter()
            .map(|v| format!("id={}", String::from_utf8_lossy(v))),
    );
    roundtrip(&NumericTemplatePack, &refs(&rows));
}

#[test]
fn templates_patch_rows_of_another_shape() {
    let base = owned((0..20).map(|i| format!("req-{i}-done")));
    for odd in [
        "hello",
        "req-x-done",
        "req-007-done",
        "req--1-done",
        "req-1-2-done",
        "",
    ] {
        for at in [0, 19] {
            let mut rows = base.clone();
            rows[at] = odd.as_bytes().to_vec();
            roundtrip(&NumericTemplatePack, &refs(&rows));
        }
    }
}

#[test]
fn templates_literal_control_bytes_do_not_alias_slots() {
    // Skeletons mark slots with bytes 0 and 1, so rows holding either as text stay out.
    let nul = [
        "x 5\0", "x 6\0", "x 7\0", "x 8\0", "x 9\0", "x 1\0", "x 2\0", "x \x003",
    ];
    refuses_strs(&NumericTemplatePack, &nul);
    let mut rows = owned((0..16).map(|i| format!("x {i} y")));
    rows[5] = b"x \x00 y".to_vec();
    rows[9] = b"x \x01\x10 y".to_vec();
    roundtrip(&NumericTemplatePack, &refs(&rows));
}

#[test]
fn templates_refuse_columns_without_numbers() {
    refuses_strs(
        &NumericTemplatePack,
        &[
            "alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta",
        ],
    );
    refuses_strs(&NumericTemplatePack, &["a 1", "a 2", "a 3"]);
}

#[test]
fn templates_fuzz() {
    fuzz(&NumericTemplatePack, 13, 300, 150, gen_template);
}

// ---------------------------------------------------------------------------
// Base64Pack
// ---------------------------------------------------------------------------

#[test]
fn base64_all_padding_lengths_both_alphabets() {
    let mut r = Rng::new(22);
    for url in [false, true] {
        for len in [12usize, 13, 14, 15, 16, 17, 47, 48] {
            let mut rows = owned((0..8).map(|_| b64(&r.bytes(len), url, true)));
            rows.push(b64(&[0xff; 48][..len], url, true).into_bytes());
            rows.push(b64(&[0xfb; 48][..len], url, true).into_bytes());
            roundtrip(&Base64Pack, &refs(&rows));
        }
    }
}

#[test]
fn base64_variable_lengths_and_repeats() {
    let mut r = Rng::new(23);
    let pool: Vec<String> = (0..3).map(|i| b64(&r.bytes(12 + i), false, true)).collect();
    let rows = owned((0..40).map(|i| pool[i % 3].clone()));
    roundtrip(&Base64Pack, &refs(&rows));
    let rows = owned((0..40).map(|i| b64(&r.bytes(12 + i % 7), false, true)));
    roundtrip(&Base64Pack, &refs(&rows));
}

#[test]
fn base64_patch_invalid_rows() {
    let mut r = Rng::new(24);
    let base = owned((0..16).map(|_| b64(&r.bytes(16), false, true)));
    for odd in [
        "short==",
        "@@@@AAAAAAAAAAAAAAAA",
        "AAAAAAAAAAAAAB==",
        "AAAAAAAAAAAAAAB=",
        "AAAAAAAAAAAAAAAAA",
        "AAAAAAAA=AAAAAAA",
        "AAAAAAAAAAAA====",
        "AAAAAAAAAAAAA===",
        "",
    ] {
        for at in [0, 15] {
            let mut rows = base.clone();
            rows[at] = odd.as_bytes().to_vec();
            roundtrip(&Base64Pack, &refs(&rows));
        }
    }
}

#[test]
fn base64_of_mt_stream_rows() {
    let words = mt_words(51, 2000);
    let rows = owned(
        le_rows(&words[..120 * 12], 12)
            .iter()
            .map(|b| b64(b, false, false)),
    );
    roundtrip_generated(&Base64Pack, &refs(&rows));
}

#[test]
fn base64_refuses_short_cells() {
    refuses_strs(&Base64Pack, &["QUJD", "REVG", "R0hJ", "SktM"]);
    refuses_strs(&Base64Pack, &["AAAAAAAAAAAAAAAA"; 3]);
}

#[test]
fn base64_fuzz() {
    fuzz(&Base64Pack, 14, 300, 150, gen_b64);
}

// ---------------------------------------------------------------------------
// F64BinaryPack
// ---------------------------------------------------------------------------

fn f64_rows(vals: impl IntoIterator<Item = f64>) -> Vec<Vec<u8>> {
    vals.into_iter().map(|v| v.to_le_bytes().to_vec()).collect()
}

#[test]
fn f64_special_bit_patterns_survive_every_candidate() {
    for n in [1usize, 2, 7, 8, 15, 16, 17, 64] {
        let rows: Vec<Vec<u8>> = (0..n)
            .map(|i| F64_SPECIALS[i % F64_SPECIALS.len()].to_le_bytes().to_vec())
            .collect();
        roundtrip(&F64BinaryPack, &refs(&rows));
    }
}

fn f64_walk(x0: f64, n: usize) -> Vec<Vec<u8>> {
    let mut r = Rng::new(25);
    let mut x = x0;
    f64_rows((0..n).map(|i| {
        if i > 0 {
            x += r.pick(&[0.1, 0.2, 0.3]);
        }
        x
    }))
}

#[test]
fn f64_exact_walk_roundtrips() {
    roundtrip(&F64BinaryPack, &refs(&f64_walk(3.0, 300)));
}

#[test]
fn f64_exact_walk_from_negative_zero() {
    roundtrip(&F64BinaryPack, &refs(&f64_walk(-0.0, 300)));
}

#[test]
fn f64_smooth_and_stepped_series() {
    roundtrip(
        &F64BinaryPack,
        &refs(&f64_rows((0..200).map(|i| 0.5 * i as f64))),
    );
    roundtrip(
        &F64BinaryPack,
        &refs(&f64_rows((0..200).map(|i| (i as f64 * 0.01).sin()))),
    );
    roundtrip(
        &F64BinaryPack,
        &refs(&f64_rows((0..20).map(|i| -1e300 + i as f64 * 1e290))),
    );
}

/// `x0` then 399 steps of Python `random.Random(7).uniform(-1, 1)`.
fn python_uniform_walk(x0: f64) -> Vec<Vec<u8>> {
    let words = mt_words(7, 800);
    let mut x = x0;
    let mut vals = vec![x];
    for w in words.chunks_exact(2) {
        let r = u64::from(w[0] >> 5) << 26 | u64::from(w[1] >> 6);
        x += -1.0 + 2.0 * (r as f64 / (1u64 << 53) as f64);
        vals.push(x);
    }
    f64_rows(vals)
}

#[test]
fn f64_mt_walk_from_negative_zero() {
    let rows = python_uniform_walk(-0.0);
    with_seeds(&mut Seeds::default(), || {
        roundtrip_generated(&F64BinaryPack, &refs(&rows))
    });
}

#[test]
fn f64_walk_drawn_from_seeded_python_uniform() {
    let mut rows = python_uniform_walk(100.0);
    let mut seeds = Seeds::default();
    with_seeds(&mut seeds, || {
        roundtrip_generated(&F64BinaryPack, &refs(&rows));
        // A repeated row and a NaN reset are patched in; two NaN steps stay inside detect_walk's 1% tail.
        rows[50] = rows[49].clone();
        rows[120] = f64::NAN.to_le_bytes().to_vec();
        roundtrip_generated(&F64BinaryPack, &refs(&rows));
    });
}

#[test]
fn f64_refuses_other_widths() {
    refuses(&F64BinaryPack, &[&b"12345678"[..], &b"1234567"[..]]);
    refuses(&F64BinaryPack, &[]);
}

#[test]
fn f64_fuzz() {
    fuzz(&F64BinaryPack, 15, 300, 300, gen_f64);
}

// ---------------------------------------------------------------------------
// BinaryPlanesPack
// ---------------------------------------------------------------------------

#[test]
fn planes_two_and_four_byte_cells() {
    let rows: Vec<Vec<u8>> = (0..16u16)
        .map(|i| (i * 300).to_le_bytes().to_vec())
        .collect();
    roundtrip(&BinaryPlanesPack, &refs(&rows));
    let rows: Vec<Vec<u8>> = (0..40)
        .map(|i| (1.0f32 + i as f32 * 0.25).to_le_bytes().to_vec())
        .collect();
    roundtrip(&BinaryPlanesPack, &refs(&rows));
    let specials = [
        0u32,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        1,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    let rows: Vec<Vec<u8>> = (0..32)
        .map(|i| specials[i % 8].to_le_bytes().to_vec())
        .collect();
    roundtrip(&BinaryPlanesPack, &refs(&rows));
}

#[test]
fn planes_refuse_other_shapes() {
    let rows: Vec<Vec<u8>> = (0..15u16).map(|i| i.to_le_bytes().to_vec()).collect();
    refuses(&BinaryPlanesPack, &refs(&rows));
    let rows: Vec<Vec<u8>> = (0..16u32).map(|i| i.to_le_bytes()[..3].to_vec()).collect();
    refuses(&BinaryPlanesPack, &refs(&rows));
    let mut rows: Vec<Vec<u8>> = (0..16u32).map(|i| i.to_le_bytes().to_vec()).collect();
    rows[9].pop();
    refuses(&BinaryPlanesPack, &refs(&rows));
}

#[test]
fn planes_fuzz() {
    fuzz(&BinaryPlanesPack, 16, 300, 300, gen_planes);
}

// ---------------------------------------------------------------------------
// Every pack
// ---------------------------------------------------------------------------

#[test]
fn every_pack_patches_a_junk_first_row() {
    let mut r = Rng::new(26);
    let cols: Vec<(&dyn CodecPack, Vec<Vec<u8>>)> = vec![
        (&ExactIntPack, texts((0..16).map(|i| 5000 + i))),
        (&DecimalScaledPack, owned((0..16).map(|i| dec(i * 37, 2)))),
        (&JavaLcgPack, lcg_column(&mut r, 16)),
        (
            &TimestampIsoPack,
            owned((0..16).map(|i| iso(1_700_000_000 + i, 'T', None, "Z"))),
        ),
        (
            &TimestampClfPack,
            owned((0..16).map(|i| clf(1_700_000_000 + i, "+0000"))),
        ),
        (
            &TimestampClfPack,
            owned((0..16).map(|i| syslog(1_700_000_000 + i))),
        ),
        (
            &UuidPack,
            owned((0..16).map(|_| uuid(&uuid_bytes(&mut r, None, false), false))),
        ),
        (
            &UuidPack,
            owned((0..16).map(|i| uuid(&uuid_bytes(&mut r, Some(1 << 40 | i), false), false))),
        ),
        (
            &UlidPack,
            owned((0..16).map(|i| ulid(1_700_000_000_000 + i, u128::from(r.next())))),
        ),
        (
            &Ipv4Pack,
            owned((0..16).map(|i| format!("10.0.0.{i}:{}", 80 + i))),
        ),
        (
            &HexFixedPack,
            owned((0..16).map(|_| hex(&r.bytes(8), true))),
        ),
        (
            &TraceparentPack,
            owned((0..16).map(|_| traceparent(&r.bytes(16), &r.bytes(8), 1))),
        ),
        (
            &StructuredSplitPack,
            owned((0..16).map(|i| format!("u{}@x.com", i % 3))),
        ),
        (
            &NumericTemplatePack,
            owned((0..16).map(|i| format!("blk_-{}", 1000 + i))),
        ),
        (
            &Base64Pack,
            owned((0..16).map(|_| b64(&r.bytes(16), false, true))),
        ),
    ];
    for (pack, mut rows) in cols {
        rows[0] = b"?".to_vec();
        roundtrip(pack, &refs(&rows));
    }
}

#[test]
fn every_pack_handles_long_columns() {
    // Past the 4096-entry MTF window and 16-bit row counts.
    let mut r = Rng::new(27);
    let n = 70_000;
    let ints = texts((0..n).map(|i| (i as i64 * 7919) % 100_003 - 50_000));
    roundtrip(&ExactIntPack, &refs(&ints));
    let mut lcg = lcg_column(&mut r, 6000);
    for i in (5000..6000).step_by(3) {
        lcg[i] = lcg[i - 4500].clone();
    }
    roundtrip(&JavaLcgPack, &refs(&lcg));
    let ips = owned((0..n).map(|i| format!("10.{}.{}.{}", i >> 16, (i >> 8) & 255, i & 255)));
    roundtrip(&Ipv4Pack, &refs(&ips));
    let isos = owned((0..n / 4).map(|i| {
        iso(
            1_700_000_000 + i as i64 * 13,
            'T',
            Some(('.', 3, i as u64 % 1000)),
            "Z",
        )
    }));
    roundtrip(&TimestampIsoPack, &refs(&isos));
    let pool: Vec<String> = (0..5000).map(|_| b64(&r.bytes(18), false, true)).collect();
    let b64s = owned((0..n / 4).map(|i| pool[(i * 7) % 5000].clone()));
    roundtrip(&Base64Pack, &refs(&b64s));
}

#[test]
fn every_pack_declines_an_empty_column() {
    for pack in PackRegistry::default().packs() {
        let got = catch_unwind(AssertUnwindSafe(|| pack.try_encode(&[]).is_none()));
        assert_eq!(got.ok(), Some(true), "{}", pack.name());
    }
}

// ---------------------------------------------------------------------------
// PackRegistry
// ---------------------------------------------------------------------------

fn check_registry(rows: &[&[u8]]) -> Result<(), String> {
    let wire = catch_unwind(AssertUnwindSafe(|| {
        PackRegistry::default().encode_best(rows)
    }))
    .map_err(|_| "encoder panicked".to_string())?;
    catch_unwind(AssertUnwindSafe(|| decode_and_compare(&wire, rows)))
        .map_err(|_| "decoder panicked".to_string())?
}

fn quoted(rows: &[Vec<u8>]) -> Vec<Vec<u8>> {
    rows.iter()
        .map(|r| [b"\"", &r[..], b"\""].concat())
        .collect()
}

/// About one row in `rate` becomes `null`, the rest are quoted.
fn quoted_or_null(r: &mut Rng, rows: &[Vec<u8>], rate: u64) -> Vec<Vec<u8>> {
    quoted(rows)
        .into_iter()
        .map(|q| if r.one_in(rate) { b"null".to_vec() } else { q })
        .collect()
}

const TEXT_GENS: &[(&str, fn(&mut Rng) -> Vec<Vec<u8>>)] = &[
    ("ints", gen_ints),
    ("decimals", gen_decimals),
    ("lcg", gen_lcg),
    ("iso", gen_iso),
    ("clf", gen_clf),
    ("syslog", gen_syslog),
    ("uuid", gen_uuid),
    ("ulid", gen_ulid),
    ("ipv4", gen_ipv4),
    ("hex", gen_hex),
    ("traceparent", gen_tp),
    ("split", gen_split),
    ("template", gen_template),
    ("base64", gen_b64),
];

#[test]
fn registry_roundtrips_bare_quoted_and_nullable_columns() {
    let mut r = Rng::new(100);
    let mut fails = Vec::new();
    for &(name, gen_rows) in TEXT_GENS {
        for case in 0..30 {
            let rows = gen_rows(&mut r);
            let variants = [
                ("bare", rows.clone()),
                ("quoted", quoted(&rows)),
                ("nullable", quoted_or_null(&mut r, &rows, 5)),
            ];
            for (shape, v) in variants {
                let v = refs(&v);
                if let Err(e) = check_registry(&v) {
                    fails.push(format!(
                        "{name}/{shape} case {case}: {e}\n    rows: {}",
                        show_rows(&v)
                    ));
                }
            }
        }
    }
    for gen_rows in [gen_f64, gen_planes] {
        for case in 0..30 {
            let rows = gen_rows(&mut r);
            if let Err(e) = check_registry(&refs(&rows)) {
                fails.push(format!(
                    "binary case {case}: {e}\n    rows: {}",
                    show_rows(&refs(&rows))
                ));
            }
        }
    }
    assert!(
        fails.is_empty(),
        "{} failures:\n{}",
        fails.len(),
        fails[..fails.len().min(6)].join("\n")
    );
}

#[test]
fn registry_roundtrips_degenerate_columns() {
    let mut cols: Vec<Vec<&str>> = vec![
        vec![],
        vec![""],
        vec!["", ""],
        vec!["", "", ""],
        vec![""; 9],
        vec!["abc"; 10],
        vec!["\"x\""; 5],
        vec!["null"; 6],
        vec!["\"\""; 6],
        vec!["1"],
        vec!["1", "2"],
        vec!["1", "2", "3"],
        vec!["\"1\"", "\"2\"", "\"3\""],
        vec!["null", "\"a\"", "null"],
        vec!["null", "null", "null", "\"a\""],
        vec!["null", "null", "null", "null", "\"\""],
        vec!["\"\"", "null", "\"a\"", "\"\"", "null", "\"b\""],
        vec!["", "1", "", "2", "", "3", "", "4"],
        vec!["\"a\\\"b\"", "\"c\"", "\"d\"", "\"e\""],
        vec!["\"a\"", "\"b\"", "c", "\"d\""],
        vec!["\"", "\"\"", "\"a\"", "\"b\""],
        vec![
            "\"2024-01-01T00:00:00Z\"",
            "null",
            "\"2024-01-01T00:00:01Z\"",
            "\"2024-01-01T00:00:02Z\"",
            "null",
        ],
        vec!["\"1.5\"", "\"2.5\"", "\"-0.5\"", "null", "\"3.5\""],
        vec!["\"10.0.0.1\"", "\"10.0.0.2:80\"", "null", "\"10.0.0.3/8\""],
        vec!["0", "-0", "00", "000", "0000"],
        vec!["null", "1", "2", "3", "4"],
        ["red", "green", "blue"].repeat(40),
        ["\x01\x02", "\x03\x04", "\x01\x02"].repeat(30),
    ];
    // Quoted probe rows followed by a bare row past the 32-row probe.
    let mut long: Vec<&str> = vec!["\"7\""; 40];
    long[1] = "\"8\"";
    long[39] = "9";
    cols.push(long);
    let mut fails = Vec::new();
    for col in &cols {
        let rows: Vec<&[u8]> = col.iter().map(|s| s.as_bytes()).collect();
        if let Err(e) = check_registry(&rows) {
            fails.push(format!("{col:?}: {e}"));
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

// ---------------------------------------------------------------------------
// Near-miss mutations
// ---------------------------------------------------------------------------

const MUTATION_BYTES: &[u8] = b"0123456789-:./ +=_abcdefABCDEFZTzt@\0\x01\xff";

/// Valid columns with a few bytes flipped, inserted or dropped, so parsers see near misses.
fn mutate(r: &mut Rng, mut rows: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    for _ in 0..=r.below(3) {
        let i = r.below(rows.len() as u64) as usize;
        let row = &mut rows[i];
        let at = r.below(row.len() as u64 + 1) as usize;
        let b = r.pick(MUTATION_BYTES);
        match r.below(3) {
            0 if at < row.len() => row[at] = b,
            1 => row.insert(at, b),
            _ if at < row.len() => {
                row.remove(at);
            }
            _ => row.push(b),
        }
    }
    rows
}

#[test]
fn every_pack_survives_near_miss_rows() {
    let packs: &[(&dyn CodecPack, fn(&mut Rng) -> Vec<Vec<u8>>)] = &[
        (&ExactIntPack, gen_ints),
        (&DecimalScaledPack, gen_decimals),
        (&JavaLcgPack, gen_lcg),
        (&TimestampIsoPack, gen_iso),
        (&TimestampClfPack, gen_clf),
        (&TimestampClfPack, gen_syslog),
        (&UuidPack, gen_uuid),
        (&UlidPack, gen_ulid),
        (&Ipv4Pack, gen_ipv4),
        (&HexFixedPack, gen_hex),
        (&TraceparentPack, gen_tp),
        (&StructuredSplitPack, gen_split),
        (&NumericTemplatePack, gen_template),
        (&Base64Pack, gen_b64),
        (&F64BinaryPack, gen_f64),
        (&BinaryPlanesPack, gen_planes),
    ];
    let mut r = Rng::new(200);
    let mut fails = Vec::new();
    for &(pack, gen_rows) in packs {
        for case in 0..150 {
            let rows = gen_rows(&mut r);
            let rows = mutate(&mut r, rows);
            let v = refs(&rows);
            if let Err(e) = check(pack, &v) {
                fails.push(format!(
                    "{} case {case}: {e}\n    rows: {}",
                    pack.name(),
                    show_rows(&v)
                ));
            }
        }
    }
    assert!(
        fails.is_empty(),
        "{} failures:\n{}",
        fails.len(),
        fails[..fails.len().min(SHOW)].join("\n")
    );
}

/// `vals` through move-to-front codes and back, as cells of their decimal text.
fn mtf_roundtrip(codes: &[i64], misses: &[i64], window: u16) -> crate::wire::Result<Vec<i64>> {
    let mut gb = GraphBuilder::new(codes.len());
    let codes_node = gb.add_ints(codes);
    let misses_node = gb.add_ints(misses);
    let ints = gb.push(GraphNode::MtfInts {
        codes: codes_node,
        misses: misses_node,
        window,
    });
    let root = gb.add_prog_cells(prog_fmt_int(false, 0), vec![ints], vec![]);
    let cells = CoreDecoder::decode_bytes(&gb.finish(root).encode(), &DecodeBudget::default())?;
    Ok(cells
        .iter()
        .map(|c| std::str::from_utf8(c).unwrap().parse().unwrap())
        .collect())
}

#[test]
fn recency_codes_survive_eviction_and_deep_ranks() {
    // 300 ids over a window of 64: most repeats find their id evicted, the rest sit deep.
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let vals: Vec<i64> = (0..5000)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if i % 3 == 0 {
                (x % 8) as i64
            } else {
                (x % 300) as i64 - 150
            }
        })
        .collect();
    for window in [1, 2, 64, 4096] {
        let (codes, misses) = mtf_split(&vals, window);
        assert!(codes.iter().all(|&c| (0..=window as i64).contains(&c)));
        assert_eq!(mtf_roundtrip(&codes, &misses, window as u16).unwrap(), vals);
    }
}

#[test]
fn recency_codes_out_of_range_are_refused() {
    assert_eq!(
        mtf_roundtrip(&[0, 1, 0, 2, 2], &[7, 9], 4).unwrap(),
        [7, 7, 9, 7, 9]
    );
    // A rank past the window's contents, a negative one, and misses short or left over.
    for (codes, misses) in [
        (&[0, 2][..], &[7][..]),
        (&[0, -1], &[7]),
        (&[0, i64::MIN], &[7]),
        (&[0, 0], &[7]),
        (&[0, 1], &[7, 9]),
    ] {
        assert!(
            mtf_roundtrip(codes, misses, 4).is_err(),
            "{codes:?} {misses:?}"
        );
    }
}
