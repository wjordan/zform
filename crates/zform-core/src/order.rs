//! Row order: a stream whose runs of line records are stored sorted on fields of the writer's
//! choosing, each run preceded by an order frame that puts its records back.
//!
//! The frames of the stream hold the stored order and know nothing of it. An order frame names
//! the run by its byte range in what the frames after it decode to, and codes the original order as
//! ranks: taking the original records in turn, each one's rank among the records not yet taken.
//! The ranks are counted in stored order or, when the frame names a restore field, with the
//! stored records sorted on it; a field that nearly ascends in the original, as an event id
//! does, leaves most ranks zero.

use crate::codecs::{decode_u32_seq, encode_u32_seq};
use crate::wire::{Cursor, DecodeBudget, EntropyBackend, Error, Result, put_bytes, put_uvarint};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};

/// The runs the reader has sorted and the encoder has yet to write order frames for: each run's
/// offset in the stream and its order frame's payload after `skip`.
pub(crate) type OrderSink = Arc<Mutex<Vec<(u64, Vec<u8>)>>>;

/// An order frame's payload for a run that begins `skip` bytes into what the frames after it
/// decode to.
pub(crate) fn payload(skip: u64, run: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(run.len() + 10);
    put_uvarint(&mut out, skip);
    out.extend_from_slice(run);
    out
}

/// Rows sampled to pick a restore field.
const RESTORE_SAMPLE: usize = 4096;

fn skip_ws(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

/// The index past the string whose opening quote is at `i`.
fn skip_string(s: &[u8], mut i: usize) -> Option<usize> {
    i += 1;
    loop {
        i += memchr::memchr2(b'"', b'\\', s.get(i..)?)?;
        if s[i] == b'"' {
            return Some(i + 1);
        }
        i += 2;
    }
}

/// The index past the JSON value at `i`.
fn skip_value(s: &[u8], mut i: usize) -> Option<usize> {
    match *s.get(i)? {
        b'"' => skip_string(s, i),
        b'{' | b'[' => {
            let mut depth = 0usize;
            loop {
                match *s.get(i)? {
                    b'"' => {
                        i = skip_string(s, i)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
        _ => {
            while i < s.len() && !matches!(s[i], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
            {
                i += 1;
            }
            Some(i)
        }
    }
}

/// Calls `f` with each member of the object at `i`, its key as written between the quotes and
/// the index of its value, until `f` returns true.
fn members<'a>(s: &'a [u8], i: usize, mut f: impl FnMut(&'a [u8], usize) -> bool) -> Option<()> {
    let mut i = skip_ws(s, i);
    if *s.get(i)? != b'{' {
        return None;
    }
    i = skip_ws(s, i + 1);
    while *s.get(i)? == b'"' {
        let end = skip_string(s, i)?;
        let key = &s[i + 1..end - 1];
        i = skip_ws(s, end);
        if *s.get(i)? != b':' {
            return None;
        }
        i = skip_ws(s, i + 1);
        if f(key, i) {
            return Some(());
        }
        i = skip_ws(s, skip_value(s, i)?);
        if *s.get(i)? != b',' {
            break;
        }
        i = skip_ws(s, i + 1);
    }
    Some(())
}

/// The scalar at `i` as an order key: a string's bytes between its quotes, a number or literal
/// as written, and nothing for an object or array.
fn scalar(s: &[u8], i: usize) -> &[u8] {
    match s.get(i) {
        Some(b'"') => skip_string(s, i).map_or(&[], |e| &s[i + 1..e - 1]),
        Some(b'{' | b'[') | None => &[],
        Some(_) => skip_value(s, i).map_or(&[], |e| &s[i..e]),
    }
}

/// The key `rec` sorts by for `path`: the scalar at that path of a JSON object, or nothing when
/// the record has none.
pub fn field<'a>(rec: &'a [u8], path: &[Vec<u8>]) -> &'a [u8] {
    let mut at = 0;
    for (d, seg) in path.iter().enumerate() {
        let mut hit = None;
        members(rec, at, |k, v| {
            hit = (k == &seg[..]).then_some(v);
            hit.is_some()
        });
        match hit {
            Some(v) if d + 1 == path.len() => return scalar(rec, v),
            Some(v) => at = v,
            None => break,
        }
    }
    &[]
}

/// A dotted path, with or without a leading `$.`, as its segments.
pub fn parse_path(p: &str) -> Vec<Vec<u8>> {
    let p = p.strip_prefix('$').unwrap_or(p);
    p.split('.')
        .filter(|s| !s.is_empty())
        .map(|s| s.as_bytes().to_vec())
        .collect()
}

/// Keys order as unsigned integers when both are digits, which sort first, and as bytes otherwise.
fn key_cmp(a: &[u8], b: &[u8]) -> Ordering {
    let digits = |k: &[u8]| !k.is_empty() && k.iter().all(u8::is_ascii_digit);
    match (digits(a), digits(b)) {
        (true, true) => a.len().cmp(&b.len()).then_with(|| a.cmp(b)),
        (x, y) => y.cmp(&x).then_with(|| a.cmp(b)),
    }
}

/// The positions `0..n` not yet taken, as a Fenwick tree of counts.
struct Remaining(Vec<u32>);

impl Remaining {
    fn new(n: usize) -> Self {
        Self((0..=n as u32).map(|i| i & i.wrapping_neg()).collect())
    }

    /// How many positions before `p` remain.
    fn rank(&self, mut p: usize) -> u32 {
        let mut r = 0;
        while p > 0 {
            r += self.0[p];
            p &= p - 1;
        }
        r
    }

    /// The position with `k` remaining ones before it, when there is one.
    fn select(&self, mut k: u32) -> Option<usize> {
        let n = self.0.len() - 1;
        let mut p = 0;
        let mut step = n.checked_next_power_of_two()?;
        while step > 0 {
            if p + step <= n && self.0[p + step] <= k {
                p += step;
                k -= self.0[p];
            }
            step >>= 1;
        }
        (p < n).then_some(p)
    }

    fn take(&mut self, p: usize) {
        let mut i = p + 1;
        while i < self.0.len() {
            self.0[i] -= 1;
            i += i & i.wrapping_neg();
        }
    }
}

/// `run` cut into its lines, each with its newline.
fn lines(run: &[u8]) -> Vec<&[u8]> {
    let mut v = Vec::new();
    let mut s = 0;
    for e in memchr::memchr_iter(b'\n', run) {
        v.push(&run[s..=e]);
        s = e + 1;
    }
    v
}

/// The stored rows sorted on `path`, rows of equal keys in stored order.
fn sorted_on(stored: &[&[u8]], path: &[Vec<u8>]) -> Vec<u32> {
    let keys: Vec<&[u8]> = stored.iter().map(|r| field(r, path)).collect();
    let mut sigma: Vec<u32> = (0..stored.len() as u32).collect();
    sigma.sort_by(|&a, &b| key_cmp(keys[a as usize], keys[b as usize]));
    sigma
}

/// The top-level field of `recs` that most nearly ascends through them, when under a quarter
/// of them fail to rise on it.
fn restore_field(recs: &[&[u8]]) -> Option<Vec<u8>> {
    let sample = &recs[..recs.len().min(RESTORE_SAMPLE)];
    let mut names = Vec::new();
    members(sample[0], 0, |k, v| {
        if !scalar(sample[0], v).is_empty() {
            names.push(k.to_vec());
        }
        false
    });
    names
        .into_iter()
        .map(|name| {
            let path = [name];
            let falls = sample
                .windows(2)
                .filter(|w| key_cmp(field(w[0], &path), field(w[1], &path)) != Ordering::Less)
                .count();
            let [name] = path;
            (falls, name)
        })
        .min()
        .filter(|(falls, _)| falls * 4 < sample.len())
        .map(|(_, name)| name)
}

/// `run`'s lines sorted on `paths`, with the order frame payload that restores them, less its
/// `skip`; `None` when the sort leaves them as they are.
fn order_run(
    run: &[u8],
    paths: &[Vec<Vec<u8>>],
    backend: EntropyBackend,
    level: i32,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let recs = lines(run);
    let n = recs.len();
    if n < 2 || n > u32::MAX as usize {
        return None;
    }
    let keys: Vec<Vec<&[u8]>> = paths
        .iter()
        .map(|p| recs.iter().map(|r| field(r, p)).collect())
        .collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        keys.iter()
            .map(|k| key_cmp(k[a as usize], k[b as usize]))
            .find(|o| o.is_ne())
            .unwrap_or(Ordering::Equal)
    });
    if order.iter().enumerate().all(|(j, &i)| j == i as usize) {
        return None;
    }
    let stored: Vec<&[u8]> = order.iter().map(|&i| recs[i as usize]).collect();
    let restore = restore_field(&recs);
    // Each stored row's place in the order the ranks count in.
    let mut place: Vec<u32> = (0..n as u32).collect();
    if let Some(name) = &restore {
        for (p, &j) in sorted_on(&stored, std::slice::from_ref(name))
            .iter()
            .enumerate()
        {
            place[j as usize] = p as u32;
        }
    }
    let mut at = vec![0u32; n];
    for (j, &i) in order.iter().enumerate() {
        at[i as usize] = place[j];
    }
    let mut left = Remaining::new(n);
    let ranks: Vec<u32> = at
        .iter()
        .map(|&p| {
            let r = left.rank(p as usize);
            left.take(p as usize);
            r
        })
        .collect();

    let mut wire = Vec::new();
    put_uvarint(&mut wire, run.len() as u64);
    put_bytes(&mut wire, restore.as_deref().unwrap_or(&[]));
    encode_u32_seq(&ranks, backend, level, &mut wire);
    let mut out = Vec::with_capacity(run.len());
    for r in stored {
        out.extend_from_slice(r);
    }
    Some((out, wire))
}

/// Reads a stream of line records with each run of up to `cap` bytes sorted on `paths`, and
/// hands `sink` the order frame of every run it changed.
pub struct OrderReader<R> {
    inner: R,
    paths: Vec<Vec<Vec<u8>>>,
    cap: usize,
    backend: EntropyBackend,
    level: i32,
    sink: Option<OrderSink>,
    /// The current run as stored, read up to `at`.
    out: Vec<u8>,
    at: usize,
    /// Input past the current run.
    carry: Vec<u8>,
    offset: u64,
    eof: bool,
}

impl<R: Read> OrderReader<R> {
    pub(crate) fn new(
        inner: R,
        order_by: &[String],
        cap: usize,
        backend: EntropyBackend,
        level: i32,
        sink: Option<OrderSink>,
    ) -> Self {
        Self {
            inner,
            paths: order_by.iter().map(|p| parse_path(p)).collect(),
            // A reader's default budget refuses a longer run.
            cap: cap.clamp(1, DecodeBudget::default().max_block_decoded_bytes),
            backend,
            level,
            sink,
            out: Vec::new(),
            at: 0,
            carry: Vec::new(),
            offset: 0,
            eof: false,
        }
    }

    /// Makes the next run the current one: whole lines of JSON objects, or input passed as it is.
    fn fill(&mut self) -> io::Result<()> {
        let mut buf = vec![0u8; 1 << 20];
        while !self.eof && self.carry.len() < self.cap {
            match self.inner.read(&mut buf)? {
                0 => self.eof = true,
                n => self.carry.extend_from_slice(&buf[..n]),
            }
        }
        // A run is the whole lines within `cap`. With none, a longer line or the unterminated
        // end, what was read passes as it is.
        let within = &self.carry[..self.carry.len().min(self.cap)];
        let end = memchr::memrchr(b'\n', within).map(|p| p + 1);
        let cut = end.unwrap_or(self.carry.len());
        let run = &self.carry[..cut];
        let json = run.iter().find(|b| !b.is_ascii_whitespace()) == Some(&b'{');
        let ordered = (json && end.is_some())
            .then(|| order_run(run, &self.paths, self.backend, self.level))
            .flatten();
        self.at = 0;
        match ordered {
            Some((stored, wire)) => {
                self.out = stored;
                if let Some(s) = &self.sink {
                    s.lock().unwrap().push((self.offset, wire));
                }
                self.carry = self.carry.split_off(cut);
            }
            None => {
                self.out = self.carry.split_off(cut);
                std::mem::swap(&mut self.out, &mut self.carry);
            }
        }
        self.offset += cut as u64;
        Ok(())
    }
}

impl<R: Read> Read for OrderReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.at == self.out.len() {
            self.fill()?;
        }
        let n = buf.len().min(self.out.len() - self.at);
        buf[..n].copy_from_slice(&self.out[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

/// A run to restore: its byte range in the decoded stream, the field its ranks count on, and
/// the ranks.
struct Run {
    start: u64,
    len: usize,
    restore: Vec<u8>,
    ranks: Vec<u32>,
}

/// Writes a decoded stream with the runs its order frames name put back in their original
/// order. Bytes outside a run pass through.
pub(crate) struct Restore<W> {
    inner: W,
    runs: VecDeque<Run>,
    /// Bytes of the decoded stream taken so far.
    pos: u64,
    /// The front run's bytes so far.
    buf: Vec<u8>,
}

impl<W: Write> Restore<W> {
    pub(crate) fn new(inner: W) -> Self {
        Self {
            inner,
            runs: VecDeque::new(),
            pos: 0,
            buf: Vec::new(),
        }
    }

    /// Takes an order frame's payload. Its run lies past every run and byte taken so far.
    pub(crate) fn order(&mut self, payload: &[u8], budget: &DecodeBudget) -> Result<()> {
        let mut cur = Cursor::new(payload);
        let start = self.pos.saturating_add(cur.uvar()?);
        let len = cur.uvar()? as usize;
        let restore = cur.bytes()?.to_vec();
        let ranks = decode_u32_seq(&mut cur, budget)?;
        let after = self
            .runs
            .back()
            .map_or(self.pos, |r| r.start + r.len as u64);
        if start < after || len == 0 || len > budget.max_block_decoded_bytes || ranks.len() > len {
            return Err(Error::Format("order frame names a bad run".into()));
        }
        self.runs.push_back(Run {
            start,
            len,
            restore,
            ranks,
        });
        Ok(())
    }

    /// Writes the front run, complete in `buf`, in its original order.
    fn restore(&mut self) -> Result<()> {
        let run = self.runs.pop_front().expect("a run is in progress");
        let buf = std::mem::take(&mut self.buf);
        let stored = lines(&buf);
        if stored.len() != run.ranks.len() || !buf.ends_with(b"\n") {
            return Err(Error::Format("order frame does not fit its run".into()));
        }
        let sigma = (!run.restore.is_empty()).then(|| sorted_on(&stored, &[run.restore]));
        let mut left = Remaining::new(stored.len());
        let mut out = Vec::with_capacity(buf.len().min(4 << 20));
        for &r in &run.ranks {
            let p = left
                .select(r)
                .ok_or_else(|| Error::Format("order rank out of range".into()))?;
            left.take(p);
            out.extend_from_slice(stored[sigma.as_ref().map_or(p, |s| s[p] as usize)]);
            if out.len() >= 4 << 20 {
                self.inner.write_all(&out)?;
                out.clear();
            }
        }
        self.inner.write_all(&out)?;
        Ok(())
    }

    /// Ends the stream: every run named must have arrived whole.
    pub(crate) fn finish(&mut self) -> Result<()> {
        match self.runs.is_empty() {
            true => Ok(()),
            false => Err(Error::Format("stream ends inside an ordered run".into())),
        }
    }

    fn take(&mut self, mut data: &[u8]) -> Result<()> {
        while !data.is_empty() {
            let Some(run) = self.runs.front() else {
                self.pos += data.len() as u64;
                return Ok(self.inner.write_all(data)?);
            };
            let n = if self.pos < run.start {
                let n = data.len().min((run.start - self.pos) as usize);
                self.inner.write_all(&data[..n])?;
                n
            } else {
                let n = data.len().min(run.len - self.buf.len());
                self.buf
                    .try_reserve(n)
                    .map_err(|_| Error::BudgetExceeded("ordered run"))?;
                self.buf.extend_from_slice(&data[..n]);
                if self.buf.len() == run.len {
                    self.restore()?;
                }
                n
            };
            self.pos += n as u64;
            data = &data[n..];
        }
        Ok(())
    }
}

impl<W: Write> Write for Restore<W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.take(data).map_err(io::Error::other)?;
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(p: &str) -> Vec<Vec<u8>> {
        parse_path(p)
    }

    #[test]
    fn field_reads_scalars_at_a_path() {
        let r = br#"{"id":"12","a":{"s":"x\"}","b":[1,{"id":3}],"id":7},"n":-1.5,"t":true}"#;
        assert_eq!(field(r, &path("id")), b"12");
        assert_eq!(field(r, &path("a.id")), b"7");
        assert_eq!(field(r, &path("$.a.s")), br#"x\"}"#);
        assert_eq!(field(r, &path("n")), b"-1.5");
        assert_eq!(field(r, &path("t")), b"true");
        assert_eq!(field(r, &path("a")), b"");
        assert_eq!(field(r, &path("a.b")), b"");
        assert_eq!(field(r, &path("missing.id")), b"");
        assert_eq!(field(b"not json", &path("id")), b"");
        assert_eq!(field(br#"{"id":"unterminated"#, &path("id")), b"");
    }

    #[test]
    fn remaining_selects_what_rank_counts() {
        let mut left = Remaining::new(11);
        for p in [3usize, 0, 10, 7] {
            left.take(p);
        }
        let live: Vec<usize> = (0..11).filter(|p| ![3, 0, 10, 7].contains(p)).collect();
        for (k, &p) in live.iter().enumerate() {
            assert_eq!(left.rank(p), k as u32);
            assert_eq!(left.select(k as u32), Some(p));
        }
        assert_eq!(left.select(live.len() as u32), None);
    }

    /// Sorts `input` in runs of `cap` bytes and restores it, returning the stored stream.
    fn round_trip(input: &[u8], by: &[&str], cap: usize) -> Vec<u8> {
        let sink = OrderSink::default();
        let by: Vec<String> = by.iter().map(|s| s.to_string()).collect();
        let mut rdr = OrderReader::new(
            input,
            &by,
            cap,
            EntropyBackend::Zstd,
            3,
            Some(Arc::clone(&sink)),
        );
        let mut stored = Vec::new();
        rdr.read_to_end(&mut stored).unwrap();
        assert_eq!(stored.len(), input.len());
        let mut back = Restore::new(Vec::new());
        for (start, run) in sink.lock().unwrap().iter() {
            assert!(Cursor::new(run).uvar().unwrap() <= cap as u64);
            back.order(&payload(*start, run), &DecodeBudget::default())
                .unwrap();
        }
        // The frames deliver the stored stream in pieces of any size.
        for piece in stored.chunks(37) {
            back.write_all(piece).unwrap();
        }
        back.finish().unwrap();
        assert_eq!(back.inner, input);
        stored
    }

    fn events(n: u64) -> Vec<u8> {
        let mut s = String::new();
        for i in 0..n {
            // Ids rise but for a swapped pair every seventh row, and repeat every 50th.
            let swapped = match i % 7 {
                3 => i + 1,
                4 => i - 1,
                _ => i,
            };
            let id = 1000 + swapped - u64::from(i % 50 == 49);
            let (t, repo) = (["push", "fork", "watch"][(i * 7 % 3) as usize], i * 13 % 10);
            s += &format!("{{\"id\":\"{id}\",\"type\":\"{t}\",\"repo\":{{\"id\":{repo}}}}}\n");
        }
        s.into_bytes()
    }

    #[test]
    fn runs_sort_and_come_back() {
        let input = events(500);
        let stored = round_trip(&input, &["type", "repo.id"], 1 << 30);
        let keys: Vec<(&[u8], &[u8])> = lines(&stored)
            .iter()
            .map(|r| (field(r, &path("type")), field(r, &path("repo.id"))))
            .collect();
        assert!(keys.windows(2).all(|w| w[0] <= w[1]));
        assert_ne!(stored, input);
        // Several runs, with the last line of the input unterminated.
        let mut open = input.clone();
        open.pop();
        for cap in [100, 1000, 4096] {
            round_trip(&input, &["type", "repo.id"], cap);
            round_trip(&open, &["repo.id"], cap);
        }
        // A line longer than a run passes between the runs around it.
        let long = format!("{{\"id\":\"1\",\"type\":\"{}\"}}\n", "x".repeat(3000));
        let cut = lines(&input)[..100].concat().len();
        let mixed = [&input[..cut], long.as_bytes(), &input[cut..]].concat();
        round_trip(&mixed, &["type"], 1000);
    }

    #[test]
    fn a_packed_stream_unpacks_in_its_original_order() {
        use crate::container::{PackOptions, pack_stream, unpack_stream, unpack_stream_threads};
        let input = events(8000);
        let pack = |order_by: &[&str], threads| {
            let opts = PackOptions {
                order_by: order_by.iter().map(|s| s.to_string()).collect(),
                // Runs end inside chunks, so a frame holds the end of one and the start of the next.
                order_run_bytes: 150_000,
                chunk_bytes: 64 << 10,
                threads,
                ..PackOptions::default()
            };
            let mut packed = Vec::new();
            pack_stream(&input[..], &mut packed, &opts).unwrap();
            packed
        };
        let packed = pack(&["type", "repo.id"], 1);
        assert_eq!(packed, pack(&["type", "repo.id"], 4));
        assert_ne!(packed, pack(&[], 1));
        // Twice over, as concatenated streams unpack.
        let mut back = Vec::new();
        unpack_stream(&packed.repeat(2)[..], &mut back, &DecodeBudget::default()).unwrap();
        assert_eq!(back, input.repeat(2));
        let mut par = Vec::new();
        unpack_stream_threads(&packed.repeat(2)[..], &mut par, &DecodeBudget::default(), 3)
            .unwrap();
        assert!(par == back);
    }

    #[test]
    fn a_nearly_sorted_field_makes_most_ranks_zero() {
        let input = events(2000);
        let recs = lines(&input);
        assert_eq!(restore_field(&recs).as_deref(), Some(&b"id"[..]));
        let (_, wire) = order_run(&input, &[path("type")], EntropyBackend::Raw, 3).unwrap();
        let mut cur = Cursor::new(&wire);
        cur.uvar().unwrap();
        assert_eq!(cur.bytes().unwrap(), b"id");
        let ranks = decode_u32_seq(&mut cur, &DecodeBudget::default()).unwrap();
        assert!(ranks.iter().filter(|&&r| r == 0).count() > 1500);
    }

    #[test]
    fn other_input_passes_through() {
        // Not JSON objects, already in order, and a single line: no order frame.
        for input in [
            &b"b line\na line\n"[..],
            b"{\"k\":1}\n{\"k\":2}\n",
            b"{\"k\":2}",
        ] {
            let sink = OrderSink::default();
            let by = ["k".to_string()];
            let mut rdr = OrderReader::new(
                input,
                &by,
                1 << 20,
                EntropyBackend::Zstd,
                3,
                Some(sink.clone()),
            );
            let mut stored = Vec::new();
            rdr.read_to_end(&mut stored).unwrap();
            assert_eq!(stored, input);
            assert!(sink.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn bad_order_frames_are_refused() {
        let input = events(100);
        let (stored, wire) = order_run(&input, &[path("type")], EntropyBackend::Zstd, 3).unwrap();
        let (wire, budget) = (payload(0, &wire), DecodeBudget::default());
        // A run that ends early.
        let mut back = Restore::new(Vec::new());
        back.order(&wire, &budget).unwrap();
        back.write_all(&stored[..stored.len() - 1]).unwrap();
        assert!(back.finish().is_err());
        // A second frame over the same bytes.
        let mut back = Restore::new(Vec::new());
        back.order(&wire, &budget).unwrap();
        assert!(back.order(&wire, &budget).is_err());
        // Ranks for a different run.
        let (_, other) = order_run(&events(90), &[path("type")], EntropyBackend::Zstd, 3).unwrap();
        let mut back = Restore::new(Vec::new());
        back.order(&payload(0, &other), &budget).unwrap();
        assert!(back.write_all(&stored).is_err());
    }
}
