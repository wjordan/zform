//! Configurable, byte-exact record framing for text, length-prefixed, and fixed-stride binary streams.

use crate::wire::{Cursor, Error, Result, put_uvarint};
use memchr::{memchr, memchr2};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FramingSpec {
    /// Automatically chooses between text (`Lf` / `Crlf`) and periodic fixed-stride binary records.
    #[default]
    Auto,
    /// Line-feed (`\n`) delimited records (preserves missing final `\n` via `open_end`).
    Lf,
    /// Carriage-return + line-feed (`\r\n`) when uniform; per-chunk falls back cleanly to `Lf` if mixed.
    Crlf,
    /// NUL (`\0`) delimited records (`find -print0`, C string streams).
    Nul,
    /// Little-endian `u32` length-prefixed records (fallbacks to passthrough if invalid).
    LengthU32Le,
    /// Fixed-width binary records (`stride` bytes) after an optional file/chunk `header_bytes` prefix.
    FixedStride { stride: u32, header_bytes: u32 },
    /// One JSON document; the elements of its main array are the records. Falls back to `Lf` when
    /// the input is not such a document. `Auto` also detects this.
    Json,
}

/// Resolved concrete framing stored in each `ZFM1` frame header so decoding is 100% deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedFraming {
    Lf {
        open_end: bool,
    },
    Crlf {
        open_end: bool,
    },
    Nul {
        open_end: bool,
    },
    LengthU32Le,
    FixedStride {
        stride: u32,
        header_bytes: u32,
        tail_bytes: u32,
    },
    /// Records are joined by a per-frame separator between a frame prefix and suffix.
    JsonArray,
}

impl ResolvedFraming {
    pub fn encode(&self, out: &mut Vec<u8>) {
        match *self {
            Self::Lf { open_end } => {
                out.push(0);
                out.push(u8::from(open_end));
            }
            Self::Crlf { open_end } => {
                out.push(1);
                out.push(u8::from(open_end));
            }
            Self::Nul { open_end } => {
                out.push(2);
                out.push(u8::from(open_end));
            }
            Self::LengthU32Le => {
                out.push(3);
            }
            Self::FixedStride {
                stride,
                header_bytes,
                tail_bytes,
            } => {
                out.push(4);
                put_uvarint(out, stride as u64);
                put_uvarint(out, header_bytes as u64);
                put_uvarint(out, tail_bytes as u64);
            }
            Self::JsonArray => out.push(5),
        }
    }

    pub fn decode(cur: &mut Cursor<'_>) -> Result<Self> {
        match cur.u8()? {
            0 => Ok(Self::Lf {
                open_end: cur.u8()? != 0,
            }),
            1 => Ok(Self::Crlf {
                open_end: cur.u8()? != 0,
            }),
            2 => Ok(Self::Nul {
                open_end: cur.u8()? != 0,
            }),
            3 => Ok(Self::LengthU32Le),
            4 => {
                let stride = cur.uvar()? as u32;
                let header_bytes = cur.uvar()? as u32;
                let tail_bytes = cur.uvar()? as u32;
                if stride == 0 || stride > (16 << 20) {
                    return Err(Error::Format(format!("invalid fixed stride {stride}")));
                }
                Ok(Self::FixedStride {
                    stride,
                    header_bytes,
                    tail_bytes,
                })
            }
            5 => Ok(Self::JsonArray),
            k => Err(Error::Format(format!("unknown framing tag {k}"))),
        }
    }
}

/// Framed view over an input byte slice without copying record payloads.
pub struct FramedChunk<'a> {
    pub framing: ResolvedFraming,
    pub prefix: &'a [u8],
    pub suffix: &'a [u8],
    /// `(start, end)` byte offsets inside `raw` for each logical record.
    pub spans: Vec<(u32, u32)>,
    pub raw: &'a [u8],
}

impl<'a> FramedChunk<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    #[inline]
    pub fn record(&self, idx: usize) -> &'a [u8] {
        let (s, e) = self.spans[idx];
        &self.raw[s as usize..e as usize]
    }

    /// Bytes between consecutive `JsonArray` records; empty for framings that end each record with a delimiter.
    pub fn sep(&self) -> &'a [u8] {
        match (self.framing, self.spans.as_slice()) {
            (ResolvedFraming::JsonArray, [a, b, ..]) => &self.raw[a.1 as usize..b.0 as usize],
            _ => &[],
        }
    }
}

/// Splits `raw` (`<= u32::MAX` bytes) into zero-copy record spans according to `spec`.
pub fn split_chunk<'a>(raw: &'a [u8], spec: FramingSpec) -> FramedChunk<'a> {
    if raw.is_empty() {
        return FramedChunk {
            framing: ResolvedFraming::Lf { open_end: false },
            prefix: &[],
            suffix: &[],
            spans: Vec::new(),
            raw,
        };
    }

    if matches!(spec, FramingSpec::Auto | FramingSpec::Json)
        && let Some(js) = split_json_array(raw, false)
    {
        return js.chunk(raw, raw.len());
    }
    let chosen = match spec {
        FramingSpec::Auto => auto_detect_framing(raw),
        other => other,
    };

    match chosen {
        FramingSpec::Crlf => {
            if let Some(fc) = try_split_crlf(raw) {
                return fc;
            }
            split_delim(raw, b'\n', false)
        }
        FramingSpec::Lf | FramingSpec::Auto | FramingSpec::Json => split_delim(raw, b'\n', false),
        FramingSpec::Nul => split_delim(raw, 0, true),
        FramingSpec::LengthU32Le => {
            if let Some(fc) = try_split_u32_le(raw) {
                return fc;
            }
            split_delim(raw, b'\n', false)
        }
        FramingSpec::FixedStride {
            stride,
            header_bytes,
        } => {
            let s = stride.max(1) as usize;
            let h = (header_bytes as usize).min(raw.len());
            let body = raw.len() - h;
            let n = body / s;
            let tail = body % s;
            let mut spans = Vec::with_capacity(n);
            let mut off = h;
            for _ in 0..n {
                spans.push((off as u32, (off + s) as u32));
                off += s;
            }
            FramedChunk {
                framing: ResolvedFraming::FixedStride {
                    stride: s as u32,
                    header_bytes: h as u32,
                    tail_bytes: tail as u32,
                },
                prefix: &raw[..h],
                suffix: &raw[off..],
                spans,
                raw,
            }
        }
    }
}

fn split_delim(raw: &[u8], delim: u8, is_nul: bool) -> FramedChunk<'_> {
    let mut spans = Vec::with_capacity((raw.len() / 64).max(16));
    let mut start = 0usize;
    while start < raw.len() {
        match memchr(delim, &raw[start..]) {
            Some(rel) => {
                let end = start + rel;
                spans.push((start as u32, end as u32));
                start = end + 1;
            }
            None => {
                spans.push((start as u32, raw.len() as u32));
                let framing = if is_nul {
                    ResolvedFraming::Nul { open_end: true }
                } else {
                    ResolvedFraming::Lf { open_end: true }
                };
                return FramedChunk {
                    framing,
                    prefix: &[],
                    suffix: &[],
                    spans,
                    raw,
                };
            }
        }
    }
    let framing = if is_nul {
        ResolvedFraming::Nul { open_end: false }
    } else {
        ResolvedFraming::Lf { open_end: false }
    };
    FramedChunk {
        framing,
        prefix: &[],
        suffix: &[],
        spans,
        raw,
    }
}

fn try_split_crlf(raw: &[u8]) -> Option<FramedChunk<'_>> {
    let mut spans = Vec::with_capacity((raw.len() / 64).max(16));
    let mut start = 0usize;
    while start < raw.len() {
        match memchr(b'\n', &raw[start..]) {
            Some(rel) => {
                let lf = start + rel;
                if lf == start || raw[lf - 1] != b'\r' {
                    return None; // Mixed LF without CR; use exact Lf mode so `\r` stays in payload
                }
                // Ensure record interior contains no bare `\r`? Actually `\r` inside payload is fine
                // because `unpack` appends `\r\n` after each record!
                spans.push((start as u32, (lf - 1) as u32));
                start = lf + 1;
            }
            None => {
                spans.push((start as u32, raw.len() as u32));
                return Some(FramedChunk {
                    framing: ResolvedFraming::Crlf { open_end: true },
                    prefix: &[],
                    suffix: &[],
                    spans,
                    raw,
                });
            }
        }
    }
    Some(FramedChunk {
        framing: ResolvedFraming::Crlf { open_end: false },
        prefix: &[],
        suffix: &[],
        spans,
        raw,
    })
}

fn try_split_u32_le(raw: &[u8]) -> Option<FramedChunk<'_>> {
    let mut spans = Vec::new();
    let mut pos = 0usize;
    while pos < raw.len() {
        if pos + 4 > raw.len() {
            return None;
        }
        let n = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + n > raw.len() {
            return None;
        }
        spans.push((pos as u32, (pos + n) as u32));
        pos += n;
    }
    Some(FramedChunk {
        framing: ResolvedFraming::LengthU32Le,
        prefix: &[],
        suffix: &[],
        spans,
        raw,
    })
}

/// Auto-detects CRLF text, LF text, or strongly periodic binary record strides (e.g., `sao`'s 28B header + 28B records).
fn auto_detect_framing(raw: &[u8]) -> FramingSpec {
    let probe = &raw[..raw.len().min(65536)];
    let mut non_text = 0usize;
    let mut lf_count = 0usize;
    let mut crlf_count = 0usize;
    for i in 0..probe.len() {
        let b = probe[i];
        if b == b'\n' {
            lf_count += 1;
            if i > 0 && probe[i - 1] == b'\r' {
                crlf_count += 1;
            }
        } else if (b < 0x20 && b != b'\t' && b != b'\r') || b >= 0x80 {
            non_text += 1;
        }
    }
    // Check UTF-8 validity on prefix to avoid misclassifying UTF-8 text as binary
    let utf8_like = std::str::from_utf8(&probe[..probe.len().saturating_sub(4)]).is_ok();
    if !utf8_like
        && non_text * 10 > probe.len()
        && probe.len() >= 1024
        && let Some((stride, hdr)) = detect_binary_stride(raw)
    {
        return FramingSpec::FixedStride {
            stride,
            header_bytes: hdr,
        };
    }
    if lf_count > 0 && crlf_count == lf_count && try_split_crlf(raw).is_some() {
        return FramingSpec::Crlf;
    }
    FramingSpec::Lf
}

/// Autocorrelation-based byte-column match detector for fixed-width binary structs (4..=256 bytes).
pub fn detect_binary_stride(raw: &[u8]) -> Option<(u32, u32)> {
    let win = &raw[..raw.len().min(32768)];
    if win.len() < 1024 {
        return None;
    }
    // Baseline byte coincidence rate at lag 1..7 prime-ish offset
    let base_start = 256;
    let span = win.len() - base_start - 256;
    let mut base_eq = 0usize;
    for i in 0..span {
        if win[base_start + i] == win[base_start + i + 251] {
            base_eq += 1;
        }
    }
    let mut best_stride = 0usize;
    let mut best_eq = 0usize;
    for s in 4..=128 {
        let mut eq = 0usize;
        for i in 0..span {
            if win[base_start + i] == win[base_start + i + s] {
                eq += 1;
            }
        }
        // Prefer fundamental period over harmonics unless harmonic is meaningfully stronger
        if eq > best_eq + span / 64
            && (best_stride == 0 || s % best_stride != 0 || eq > best_eq + span / 20)
        {
            best_eq = eq;
            best_stride = s;
        }
    }
    // Require clear structural periodicity (> baseline + 12% of positions matching adjacent record!)
    if best_stride == 0 || best_eq < base_eq + span / 8 {
        return None;
    }
    // Check if raw.len() % best_stride == 0 (covering header = 0 or header = best_stride, as in sao!)
    let hdr = if raw.len().is_multiple_of(best_stride) && raw.len() >= best_stride * 4 {
        // Test if first `best_stride` bytes differ strongly from record 1..4 (header detection)
        let mut h_match = 0usize;
        let mut r_match = 0usize;
        for b in 0..best_stride {
            if raw[b] == raw[best_stride + b] {
                h_match += 1;
            }
            if raw[best_stride + b] == raw[2 * best_stride + b] {
                r_match += 1;
            }
        }
        if r_match > h_match + best_stride / 4 {
            best_stride as u32
        } else {
            0
        }
    } else {
        (raw.len() % best_stride) as u32
    };
    Some((best_stride as u32, hdr))
}

/// Emits a decoded record along with its exact delimiter into `out`.
#[inline]
pub fn emit_framed_record(
    out: &mut Vec<u8>,
    rec: &[u8],
    framing: ResolvedFraming,
    is_last_in_frame: bool,
) {
    match framing {
        ResolvedFraming::Lf { open_end } => {
            out.extend_from_slice(rec);
            if !(is_last_in_frame && open_end) {
                out.push(b'\n');
            }
        }
        ResolvedFraming::Crlf { open_end } => {
            out.extend_from_slice(rec);
            if !(is_last_in_frame && open_end) {
                out.extend_from_slice(b"\r\n");
            }
        }
        ResolvedFraming::Nul { open_end } => {
            out.extend_from_slice(rec);
            if !(is_last_in_frame && open_end) {
                out.push(0);
            }
        }
        ResolvedFraming::LengthU32Le => {
            out.extend_from_slice(&(rec.len() as u32).to_le_bytes());
            out.extend_from_slice(rec);
        }
        ResolvedFraming::FixedStride { .. } => {
            out.extend_from_slice(rec);
        }
        // The caller writes the frame's separator between records.
        ResolvedFraming::JsonArray => out.extend_from_slice(rec),
    }
}

// --- JSON document framing ---

/// Fewest array elements worth framing as records; fewer cannot train a model.
const JSON_MIN_ELEMENTS: usize = 4;

/// Element split of a JSON array. Records are contiguous and one separator apart, so
/// `prefix + records joined by sep + suffix` reproduces the input whatever it contains.
pub(crate) struct JsonArraySplit {
    pub prefix_len: usize,
    pub spans: Vec<(u32, u32)>,
    /// The array's closing `]` was reached.
    pub closed: bool,
}

impl JsonArraySplit {
    /// Offset at which to cut a frame that stops before the array ends: past the last record's `,`.
    pub(crate) fn cut(&self) -> usize {
        self.spans.last().map_or(0, |s| s.1 as usize + 1)
    }

    /// Frames `raw[..end]`; bytes after the last record become the frame suffix.
    pub(crate) fn chunk(self, raw: &[u8], end: usize) -> FramedChunk<'_> {
        let body_end = self.spans.last().map_or(self.prefix_len, |s| s.1 as usize);
        FramedChunk {
            framing: ResolvedFraming::JsonArray,
            prefix: &raw[..self.prefix_len],
            suffix: &raw[body_end..end],
            spans: self.spans,
            raw: &raw[..end],
        }
    }

    /// Moves whitespace out of the records: the first record's into the prefix, and what every later
    /// record shares into the separator. Pretty-printed elements then read as bare objects.
    fn new(raw: &[u8], mut spans: Vec<(u32, u32)>, closed: bool) -> Self {
        let lead = |&(s, e): &(u32, u32)| {
            let rec = &raw[s as usize..e as usize];
            &rec[..rec.iter().take_while(|b| b.is_ascii_whitespace()).count()]
        };
        let mut shared: Option<&[u8]> = None;
        for span in spans.iter().skip(1) {
            let ws = lead(span);
            let prev = shared.unwrap_or(ws);
            shared = Some(&prev[..prev.iter().zip(ws).take_while(|(a, b)| a == b).count()]);
        }
        let first = spans.first().map_or(0, |s| lead(s).len() as u32);
        let shared = shared.map_or(0, |ws| ws.len() as u32);
        for (i, span) in spans.iter_mut().enumerate() {
            span.0 += if i == 0 { first } else { shared };
        }
        Self {
            prefix_len: spans.first().map_or(0, |s| s.0 as usize),
            spans,
            closed,
        }
    }
}

/// Index just past the string whose opening quote is at `raw[i]`.
fn skip_json_string(raw: &[u8], i: usize) -> Option<usize> {
    let mut p = i + 1;
    loop {
        p += memchr2(b'"', b'\\', raw.get(p..)?)?;
        if raw[p] == b'"' {
            return Some(p + 1);
        }
        p += 2;
    }
}

/// Splits array elements from `start`, which is just past `[` or an element's `,`. Returns the spans
/// between commas and the index of the closing `]` if reached; an element cut off by the end of
/// `raw` is left out.
fn scan_json_elements(raw: &[u8], start: usize) -> (Vec<(u32, u32)>, Option<usize>) {
    let mut spans = Vec::new();
    let mut depth = 0usize;
    let mut elem = start;
    let mut i = start;
    while i < raw.len() {
        match raw[i] {
            b'"' => match skip_json_string(raw, i) {
                Some(end) => {
                    i = end;
                    continue;
                }
                None => break,
            },
            b'{' | b'[' => depth += 1,
            b'}' | b']' if depth > 0 => depth -= 1,
            b']' => {
                // Whitespace before `]` belongs to the suffix, so the last record looks like the others.
                let mut end = i;
                while end > elem && raw[end - 1].is_ascii_whitespace() {
                    end -= 1;
                }
                spans.push((elem as u32, end as u32));
                return (spans, Some(i));
            }
            b'}' => break,
            b',' if depth == 0 => {
                spans.push((elem as u32, i as u32));
                elem = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    (spans, None)
}

/// True when the document that owns the array closed at `close` ends with `raw`, or runs past it.
/// A root value followed by more content is a value stream such as NDJSON, not a document.
fn json_document_ends(raw: &[u8], close: usize, mut depth: usize) -> bool {
    let mut nest = 0usize;
    let mut i = close + 1;
    while depth > 0 {
        let Some(&b) = raw.get(i) else { return true };
        match b {
            b'"' => match skip_json_string(raw, i) {
                Some(end) => {
                    i = end;
                    continue;
                }
                None => return true,
            },
            b'{' | b'[' => nest += 1,
            b'}' | b']' if nest > 0 => nest -= 1,
            b'}' | b']' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    raw[i..].iter().all(u8::is_ascii_whitespace)
}

/// Splits the main array of a JSON document into element records. With `continued`, `raw` starts
/// inside the array at an element boundary, as it does after a frame that ended on a `,`.
pub(crate) fn split_json_array(raw: &[u8], continued: bool) -> Option<JsonArraySplit> {
    if continued {
        let (spans, close) = scan_json_elements(raw, 0);
        return (!spans.is_empty()).then(|| JsonArraySplit::new(raw, spans, close.is_some()));
    }
    let mut i = raw.iter().position(|b| !b.is_ascii_whitespace())?;
    if raw[i] != b'{' && raw[i] != b'[' {
        return None;
    }
    // Descend through objects only: the main array is the first array of objects found that way.
    let mut depth = 0usize;
    while i < raw.len() {
        match raw[i] {
            b'"' => {
                i = skip_json_string(raw, i)?;
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return None;
                }
            }
            b'[' => {
                let (spans, close) = scan_json_elements(raw, i + 1);
                let is_object = |&(s, e): &(u32, u32)| {
                    raw[s as usize..e as usize]
                        .iter()
                        .find(|b| !b.is_ascii_whitespace())
                        == Some(&b'{')
                };
                if spans.len() >= JSON_MIN_ELEMENTS
                    && spans[..JSON_MIN_ELEMENTS].iter().all(is_object)
                {
                    if let Some(close) = close
                        && !json_document_ends(raw, close, depth)
                    {
                        return None;
                    }
                    return Some(JsonArraySplit::new(raw, spans, close.is_some()));
                }
                // Skip an array that is not the main one, unless it is the root or runs off the end.
                i = close.filter(|_| depth > 0)?;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Re-splits a decoded passthrough frame into the records its framing describes.
pub(crate) fn split_passthrough(raw: &[u8], framing: ResolvedFraming) -> FramedChunk<'_> {
    if framing == ResolvedFraming::JsonArray
        && let Some(js) = split_json_array(raw, false).or_else(|| split_json_array(raw, true))
    {
        return js.chunk(raw, raw.len());
    }
    split_chunk(
        raw,
        match framing {
            ResolvedFraming::Lf { .. } | ResolvedFraming::JsonArray => FramingSpec::Lf,
            ResolvedFraming::Crlf { .. } => FramingSpec::Crlf,
            ResolvedFraming::Nul { .. } => FramingSpec::Nul,
            ResolvedFraming::LengthU32Le => FramingSpec::LengthU32Le,
            ResolvedFraming::FixedStride {
                stride,
                header_bytes,
                ..
            } => FramingSpec::FixedStride {
                stride,
                header_bytes,
            },
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rejoin(fc: &FramedChunk<'_>) -> Vec<u8> {
        let mut out = fc.prefix.to_vec();
        for i in 0..fc.len() {
            if i > 0 {
                out.extend_from_slice(fc.sep());
            }
            emit_framed_record(&mut out, fc.record(i), fc.framing, i + 1 == fc.len());
        }
        out.extend_from_slice(fc.suffix);
        out
    }

    fn json_records(raw: &[u8]) -> Option<Vec<&[u8]>> {
        let fc = split_chunk(raw, FramingSpec::Auto);
        assert_eq!(rejoin(&fc), raw, "split must be exact");
        (fc.framing == ResolvedFraming::JsonArray)
            .then(|| (0..fc.len()).map(|i| fc.record(i)).collect())
    }

    #[test]
    fn json_document_elements_become_records() {
        let wrapped = br#"{"meta":{"tags":["a","b"],"n":4},"Records":[{"a":"x,]}"},{"a":[1,2]},{"a":{"b":"\""}},{"a":4}],"next":null}"#;
        let recs = json_records(wrapped).unwrap();
        assert_eq!(
            recs,
            [
                &br#"{"a":"x,]}"}"#[..],
                br#"{"a":[1,2]}"#,
                br#"{"a":{"b":"\""}}"#,
                br#"{"a":4}"#
            ]
        );

        let pretty = b"[\n  {\n    \"a\": 1\n  },\n  {\n    \"a\": 2\n  },\n  {\n    \"a\": 3\n  },\n  {\n    \"a\": 4\n  }\n]\n";
        let recs = json_records(pretty).unwrap();
        assert_eq!(recs.len(), 4);
        assert!(
            recs.iter()
                .all(|r| r.starts_with(b"{\n    \"a\": ") && r.ends_with(b"\n  }"))
        );

        // Uneven indentation: only the whitespace every element shares leaves the records.
        let recs = json_records(b"[ {\"a\":1},\n {\"a\":2},\n   {\"a\":3},\n {\"a\":4} ]").unwrap();
        assert_eq!(
            recs,
            [
                &b"{\"a\":1}"[..],
                b"{\"a\":2}",
                b"  {\"a\":3}",
                b"{\"a\":4}"
            ]
        );

        // An array cut off mid-element keeps its complete elements; the tail rides in the suffix.
        let recs = json_records(br#"[{"a":1},{"a":2},{"a":3},{"a":4},{"a":"#).unwrap();
        assert_eq!(recs.len(), 4);
    }

    #[test]
    fn value_streams_and_non_documents_stay_line_framed() {
        let ndjson = br#"{"c":[{"a":1},{"a":2},{"a":3},{"a":4}]}
{"c":[]}
"#;
        for raw in [
            &ndjson[..],
            b"[2026-11-10 12:00:00] INFO started\n[2026-11-10 12:00:01] INFO ready\n",
            br#"{"a":{"b":1},"c":{"b":2}}"#,
            br#"[1,2,3,4,5]"#,
            br#"[[{"a":1}],[{"a":1}],[{"a":1}],[{"a":1}]]"#,
            br#"{"Records":[{"a":1},{"a":2},{"a":3}]}"#,
            br#"{"Records":[{"a":1},{"a":2},{"a":3},{"a":4}]} trailing"#,
            b"{\"unterminated",
            b"",
        ] {
            assert!(
                json_records(raw).is_none(),
                "{}",
                String::from_utf8_lossy(raw)
            );
        }
    }

    #[test]
    fn continued_frames_split_exactly_on_any_bytes() {
        let doc = br#"{"k":[{"a":1},{"a":2},{"a":3},{"a":4},{"a":5},{"a":6}]}"#;
        for cut in 0..doc.len() {
            // First frame over a truncated document, then the rest as a continuation.
            let Some(js) = split_json_array(&doc[..cut], false) else {
                continue;
            };
            let end = if js.closed { cut } else { js.cut() };
            let first = js.chunk(doc, end);
            let mut out = rejoin(&first);
            match split_json_array(&doc[end..], true) {
                Some(js) => out.extend(rejoin(&js.chunk(&doc[end..], doc.len() - end))),
                None => out.extend_from_slice(&doc[end..]),
            }
            assert_eq!(out, doc, "cut {cut}");
        }
        for junk in [&b"}}]],,\"\\"[..], b",,,", b"]", b"\"", b"{[{[,"] {
            if let Some(js) = split_json_array(junk, true) {
                assert_eq!(rejoin(&js.chunk(junk, junk.len())), junk);
            }
        }
    }
}
