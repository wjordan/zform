//! Byte-exact, structure-aware lossless compression of record streams, with queries that read
//! only the parts of the compressed file they need.
//!
//! [`pack`] and [`unpack`] convert whole streams. [`Archive`] queries a packed file in place.
//! [`file`] and [`block`] are the layers under it, for a host that stores the bytes itself.

#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct ReadmeExamples;

pub use zform_core::block;
pub use zform_core::cells;
pub mod column;
pub mod numeric;
pub mod bytes;
pub use zform_core::file;
pub use zform_core::file::{RowSet, Stats};
pub use zform_core::json_tree::JsonDoc;
pub use zform_core::packs;
pub use zform_core::payload;
pub use zform_core::{
    CodecPack, ColumnProfile, CoreDecoder, DecodeBudget, EntropyBackend, Error, FrameFit,
    FramingSpec, InspectReport, ModelPlan, PackOptions, PackRegistry, PayloadGraph, Predicate,
    Profile, ResolvedFraming, Result, StreamEncoder, TrainOptions, eval_raw_record, inspect_stream,
    logical, pack_stream, raw_field_values,
};

use std::io::{Read, Seek, SeekFrom, Write};
use zform_core::file::{ByteRange, File, RangeCache, Region, Session, Step, drive};
use zform_core::framing::split_chunk;
use zform_core::index::{INDEX_END_MAGIC, INDEX_TAIL_LEN};
use zform_core::plan::train_from_records;

/// Compresses `reader` into `ZFM1` stream written to `writer`.
pub fn pack<R: Read, W: Write>(reader: R, writer: W, opts: &PackOptions) -> Result<()> {
    pack_stream(reader, writer, opts)
}

/// Decompresses a `ZFM1` stream with default [`DecodeBudget`] (verifying CRC32 + XXH64 checksums).
pub fn unpack<R: Read, W: Write>(reader: R, writer: W) -> Result<()> {
    zform_core::unpack_stream(reader, writer, &DecodeBudget::default())
}

/// Decompresses a `ZFM1` stream enforcing an explicit [`DecodeBudget`].
pub fn unpack_with_budget<R: Read, W: Write>(
    reader: R,
    writer: W,
    budget: &DecodeBudget,
) -> Result<()> {
    zform_core::unpack_stream(reader, writer, budget)
}

/// Trains a new [`Profile`] from an in-memory slice or streaming reader (with optional `prior` profile extension).
pub fn train<R: Read>(
    mut reader: R,
    opts: &TrainOptions,
    prior: Option<&Profile>,
) -> Result<Profile> {
    let mut buf = Vec::new();
    (&mut reader)
        .take(opts.max_sample_bytes as u64)
        .read_to_end(&mut buf)?;
    let plan = zform_core::train_from_bytes(&buf, opts, prior.map(|p| &p.plan));
    Ok(Profile::from_plan(plan))
}

/// Trains a [`Profile`] by sampling strided 64 KiB windows evenly across an entire seekable file/archive,
/// avoiding head-of-file schema bias without reading multi-GB files into RAM!
pub fn train_seekable<R: Read + Seek>(
    reader: &mut R,
    opts: &TrainOptions,
    prior: Option<&Profile>,
) -> Result<Profile> {
    let total_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    if total_len <= opts.max_sample_bytes as u64 {
        return train(reader, opts, prior);
    }
    // A JSON document has no line records to sample by seeking, so train on its head.
    let mut head = Vec::new();
    reader.by_ref().take(1 << 20).read_to_end(&mut head)?;
    reader.seek(SeekFrom::Start(0))?;
    if split_chunk(&head, opts.framing).framing == ResolvedFraming::JsonArray {
        return train(reader, opts, prior);
    }
    const WIN_BYTES: usize = 64 << 10;
    let n_wins = (opts.max_sample_bytes / WIN_BYTES).clamp(4, 256);
    let max_off = total_len.saturating_sub(WIN_BYTES as u64);
    let mut collected = Vec::<Vec<u8>>::new();
    let mut win_buf = vec![0u8; WIN_BYTES];

    for w in 0..n_wins {
        let off = (w as u64 * max_off) / (n_wins.saturating_sub(1).max(1) as u64);
        reader.seek(SeekFrom::Start(off))?;
        let n = reader.read(&mut win_buf)?;
        let slice = &win_buf[..n];
        // Drop partial first and last line when seeking into mid-file (`off > 0`)
        let start = if off == 0 {
            0
        } else {
            match slice.iter().position(|&b| b == b'\n') {
                Some(p) => p + 1,
                None => continue,
            }
        };
        let end = match slice[start..].iter().rposition(|&b| b == b'\n') {
            Some(p) => start + p,
            None => continue,
        };
        for line in slice[start..end].split(|&b| b == b'\n') {
            let clean = line.strip_suffix(b"\r").unwrap_or(line);
            if !clean.is_empty() {
                collected.push(clean.to_vec());
                if collected.len() >= opts.max_sample_rows {
                    break;
                }
            }
        }
        if collected.len() >= opts.max_sample_rows {
            break;
        }
    }
    if collected.is_empty() {
        reader.seek(SeekFrom::Start(0))?;
        return train(reader, opts, prior);
    }
    let refs: Vec<&[u8]> = collected.iter().map(|v| v.as_slice()).collect();
    let resolved = split_chunk(&win_buf, opts.framing).framing;
    let plan = train_from_records(&refs, resolved, opts, prior.map(|p| &p.plan));
    Ok(Profile::from_plan(plan))
}

/// A packed stream that ends with its index region, read through `Read + Seek`.
pub struct Archive<R: Read + Seek> {
    inner: R,
    cache: RangeCache,
    file: File,
    /// Bytes read from the data region so far.
    pub data_bytes_read: u64,
}

impl<R: Read + Seek> Archive<R> {
    pub fn open(mut inner: R, budget: DecodeBudget) -> Result<Self> {
        let no_index = || Error::Format("stream has no index region".into());
        let len = inner.seek(SeekFrom::End(0))?;
        let tail_len = INDEX_TAIL_LEN as u64;
        let mut tail = [0u8; INDEX_TAIL_LEN];
        inner.seek(SeekFrom::Start(
            len.checked_sub(tail_len).ok_or_else(no_index)?,
        ))?;
        inner.read_exact(&mut tail)?;
        let region_len = u64::from_le_bytes(tail[12..20].try_into().unwrap());
        if &tail[20..24] != INDEX_END_MAGIC || region_len > len || region_len < tail_len {
            return Err(no_index());
        }
        // The index is a small share of the file; hold all of it.
        let index_off = len - region_len;
        let mut index = budget.alloc_vec(region_len as usize, "index region")?;
        inner.seek(SeekFrom::Start(index_off))?;
        inner.read_exact(&mut index)?;
        let mut cache = RangeCache::default();
        cache.insert(
            ByteRange {
                region: Region::Index,
                off: 0,
                len: region_len,
            },
            index,
        );
        let Step::Ready(file) = File::open(&cache, region_len, budget)? else {
            return Err(no_index());
        };
        if file.data_len() != index_off {
            return Err(Error::Format(
                "index region does not follow its data".into(),
            ));
        }
        Ok(Self {
            inner,
            cache,
            file,
            data_bytes_read: 0,
        })
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    /// Runs `op` block by block, reading what it reports missing and dropping each block's bytes
    /// once it is done with them.
    fn each_block(
        &mut self,
        mut op: impl FnMut(&mut Session<'_>, &RangeCache, &RowSet) -> Result<Step<()>>,
    ) -> Result<Stats> {
        let Self {
            inner,
            cache,
            file,
            data_bytes_read,
            ..
        } = self;
        let mut session = file.session();
        for (first, n) in file.blocks() {
            let within = RowSet::range(first, first + n);
            let fetch = |ranges: &[ByteRange]| {
                let mut parts = Vec::with_capacity(ranges.len());
                for r in ranges {
                    let mut buf = vec![0u8; r.len as usize];
                    inner.seek(SeekFrom::Start(r.off))?;
                    inner.read_exact(&mut buf)?;
                    *data_bytes_read += r.len;
                    parts.push(buf);
                }
                Ok(parts)
            };
            drive(cache, fetch, |c| op(&mut session, c, &within))?;
            cache.clear_region(Region::Data);
        }
        Ok(session.stats)
    }

    /// Calls `f(row, record)` for each record that satisfies `pred`, in order.
    pub fn grep(
        &mut self,
        pred: &Predicate,
        mut f: impl FnMut(u64, &[u8]) -> Result<()>,
    ) -> Result<Stats> {
        self.each_block(|session, src, within| {
            let rows = match session.matches(src, pred, Some(within))? {
                Step::Ready(rows) => rows,
                Step::Missing(m) => return Ok(Step::Missing(m)),
            };
            session.records(src, &rows, &mut f)
        })
    }

    /// Calls `f(row, path, value)` for each of `paths` on each record that satisfies `pred`, as
    /// `Session::fields` does.
    pub fn select(
        &mut self,
        paths: &[&str],
        pred: &Predicate,
        mut f: impl FnMut(u64, usize, Option<&[u8]>) -> Result<()>,
    ) -> Result<Stats> {
        self.each_block(|session, src, within| {
            let rows = match session.matches(src, pred, Some(within))? {
                Step::Ready(rows) => rows,
                Step::Missing(m) => return Ok(Step::Missing(m)),
            };
            session.fields(src, paths, &rows, &mut f)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor as IoCursor;
    use zform_core::file::{ByteRange, MemSource, RangeCache, Region, Step, Writer, drive};

    fn assert_roundtrip(raw: &[u8], opts: &PackOptions) -> Vec<u8> {
        let mut comp = Vec::new();
        pack(raw, &mut comp, opts).expect("pack");
        let mut restored = Vec::new();
        unpack(&comp[..], &mut restored).expect("unpack");
        assert_eq!(raw, &restored[..], "byte-exact roundtrip failed");
        comp
    }

    fn lines(raw: &[u8]) -> Vec<&[u8]> {
        raw.strip_suffix(b"\n")
            .unwrap_or(raw)
            .split(|&b| b == b'\n')
            .collect()
    }

    /// Rows of `recs` that satisfy `pred`, by the raw rule.
    fn brute(recs: &[&[u8]], pred: &Predicate) -> Vec<u64> {
        (0..recs.len())
            .filter(|&i| eval_raw_record(recs[i], pred))
            .map(|i| i as u64)
            .collect()
    }

    /// Checks that a packed stream answers `pred` as its raw records do, and returns the rows.
    fn assert_grep(comp: &[u8], recs: &[&[u8]], pred: &Predicate) -> (Vec<u64>, Stats) {
        let mut arch = Archive::open(IoCursor::new(comp), DecodeBudget::default()).unwrap();
        assert_eq!(arch.file().rows(), recs.len() as u64);
        let mut rows = Vec::new();
        let stats = arch
            .grep(pred, |row, rec| {
                assert_eq!(rec, recs[row as usize], "row {row} under {pred:?}");
                rows.push(row);
                Ok(())
            })
            .unwrap();
        assert_eq!(rows, brute(recs, pred), "{pred:?}");
        (rows, stats)
    }

    fn eq(field: &str, value: &str) -> Predicate {
        Predicate::FieldEq {
            field: field.into(),
            value: value.into(),
        }
    }

    fn int_range(field: &str, min: Option<i64>, max: Option<i64>) -> Predicate {
        Predicate::FieldIntRange {
            field: field.into(),
            min,
            max,
        }
    }

    fn contains(needle: &str) -> Predicate {
        Predicate::Contains(needle.into())
    }

    fn word(needle: &str) -> Predicate {
        Predicate::Word(needle.into())
    }

    #[test]
    fn a_word_has_no_word_byte_beside_it() {
        let holds = |rec: &str, w: &str| eval_raw_record(rec.as_bytes(), &word(w));
        assert!(
            holds("retry 3 after", "retry")
                && holds("a retry", "retry")
                && holds("x=\"retry\"", "retry")
        );
        assert!(
            !holds("retry3 after", "retry")
                && !holds("_retry", "retry")
                && !holds("Aretry", "retry")
        );
        assert!(
            holds("retrying, retry.", "retry"),
            "a later place can stand where an earlier one does not"
        );
        assert!(holds("a-b c", "a-b") && holds("job 12", "job 12") && !holds("job 123", "job 12"));
        assert!(holds("caf\u{e9}s", "s") && holds("anything", ""));
    }

    #[test]
    fn hybrid_multi_arm_observability_stream_roundtrip_and_query() {
        // Simultaneous JSON + Docker escaped JSON-in-JSON + K8s CRI + Syslog + logfmt + HDFS blk_ templates + variable words + stack trace!
        let mut raw = Vec::new();
        for i in 0..300 {
            // 1. Nested JSON with URL stem derivation & AffineInt (`end_ms = start_ms + dur_ms`)
            let start = 1_731_260_000_000i64 + i * 25;
            let dur = 10 + (i % 9);
            let end = start + dur;
            let repo = format!("acme/svc-{}", i % 6);
            let line1 = format!(
                "{{\"ts\":\"2026-11-10T11:24:{:02}.{:03}Z\",\"repo\":\"{repo}\",\"url\":\"https://api.github.com/repos/{repo}/commits\",\"traceparent\":\"00-4bf92f3577b34da6a3ce929d0e0e{:04x}-00f067aa0ba9{:04x}-01\",\"start_ms\":{start},\"dur_ms\":{dur},\"end_ms\":{end},\"latency\":{}.{:03}}}\n",
                (i / 10) % 60,
                (i * 37) % 1000,
                i % 12,
                i,
                i % 15,
                (i * 7) % 1000
            );
            raw.extend_from_slice(line1.as_bytes());

            // 2. Docker JSON-in-JSON envelope!
            let line2 = format!(
                "{{\"log\":\"{{\\\"level\\\":\\\"info\\\",\\\"service\\\":\\\"payments\\\",\\\"req_id\\\":\\\"018f5a3e-7c10-7b2a-9f01-{:012x}\\\",\\\"user\\\":\\\"u{}@corp.example.com\\\"}}\\n\",\"stream\":\"stdout\",\"time\":\"2026-11-10T11:24:{:02}.100000000Z\"}}\n",
                i % 20,
                i % 15,
                (i / 10) % 60
            );
            raw.extend_from_slice(line2.as_bytes());

            // 3. Kubernetes CRI envelope wrapping logfmt!
            let line3 = format!(
                "2026-11-10T11:24:{:02}.250000000Z stdout F ts=2026-11-10T11:24:{:02}Z level={} pod=checkout-7d9f8b-x{:02} addr=10.24.1.{}:8080 msg=\"processed order\"\n",
                (i / 10) % 60,
                (i / 10) % 60,
                if i % 10 == 0 { "warn" } else { "info" },
                i % 8,
                (i % 40) + 1
            );
            raw.extend_from_slice(line3.as_bytes());

            // 4. Plain-text HDFS-style `blk_` + variable-word log + stack trace banner!
            let blk = -1608999687919862900i64 + (i % 16);
            let method = ["GET", "POST", "PUT"][(i as usize) % 3];
            let line4 = format!(
                "2026-11-10 11:24:{:02},500 INFO dfs.DataNode: Receiving block blk_{blk} src: /10.0.0.{}:50010 dest: /10.0.0.{}:50010 op={method}\n",
                (i / 10) % 60,
                (i % 8) + 1,
                (i % 8) + 2
            );
            raw.extend_from_slice(line4.as_bytes());
        }
        raw.extend_from_slice(b"java.io.IOException: unexpected reset by peer\n\tat com.acme.Net.read(Net.java:142)\n");

        let opts = PackOptions::default();
        let comp = assert_roundtrip(&raw, &opts);
        // Should compress dramatically (> 12x even on small ~280 KB hybrid sample!)
        assert!(
            comp.len() * 10 < raw.len(),
            "expected >10x ratio on hybrid observability corpus: raw={} comp={}",
            raw.len(),
            comp.len()
        );

        // Every arm answers predicates as its raw records do.
        let recs = lines(&raw);
        let pred = Predicate::And(vec![
            eq("repo", "acme/svc-2"),
            int_range("dur_ms", Some(15), None),
        ]);
        let (rows, _) = assert_grep(&comp, &recs, &pred);
        assert!(!rows.is_empty());
        for pred in [
            contains("svc-2/commits"),
            contains("blk_-16089996879198628"),
            contains("\\\"level\\\":\\\"info"),
            contains("stdout F ts="),
            contains("IOException"),
            eq("level", "warn"),
            eq("stream", "stdout"),
            eq("msg", "processed order"),
            Predicate::FieldExists("pod".into()),
            Predicate::FieldContains {
                field: "url".into(),
                needle: b"svc-4".to_vec(),
            },
            Predicate::Or(vec![eq("repo", "acme/svc-5"), contains("op=PUT")]),
            Predicate::Not(Box::new(Predicate::FieldExists("ts".into()))),
        ] {
            assert_grep(&comp, &recs, &pred);
        }

        // Selected fields come back as the records write them.
        let mut arch = Archive::open(IoCursor::new(&comp), DecodeBudget::default()).unwrap();
        let mut got = Vec::new();
        arch.select(&["repo", "end_ms"], &pred, |row, path, v| {
            got.push((row, path, v.map(<[u8]>::to_vec)));
            Ok(())
        })
        .unwrap();
        assert_eq!(got.len(), rows.len() * 2);
        assert_eq!(got[0], (rows[0], 0, Some(b"\"acme/svc-2\"".to_vec())));
        for (row, path, v) in got {
            let mut want = Vec::new();
            raw_field_values(recs[row as usize], ["repo", "end_ms"][path], &mut |v| {
                want.push(v.to_vec())
            });
            assert_eq!(v.as_slice(), want.as_slice());
        }
    }

    #[test]
    fn blocks_embed_in_a_foreign_container() {
        use crate::block::{BlockOptions, decode_block, encode_block, train};
        let rows: Vec<String> = (0..2000)
            .map(|i| {
                format!(
                    r#"{{"ts":"2026-11-10T12:00:{:02}Z","svc":"{}","status":{},"dur_ms":{}}}"#,
                    i % 60,
                    ["auth", "cart", "search"][i % 3],
                    if i % 50 == 0 { 503 } else { 200 },
                    i % 97
                )
            })
            .collect();
        let recs: Vec<&[u8]> = rows.iter().map(|r| r.as_bytes()).collect();
        let budget = DecodeBudget::default();
        let opts = BlockOptions::default();

        // The container stores the model once, as profile bytes, and each block with its row count.
        let trained = train(&recs[..500], &TrainOptions::default(), None);
        let model = Profile::from_bytes(&Profile::from_plan(trained).bytes, &budget)
            .unwrap()
            .plan;
        let (a, fit) = encode_block(&recs[..1200], &model, &opts);
        let (b, _) = encode_block(&recs[1200..], &model, &opts);
        assert_eq!(
            (fit.rows, fit.residual_rows, fit.residual_rate()),
            (1200, 0, 0.0)
        );
        let raw: usize = rows.iter().map(String::len).sum();
        assert!(
            (a.len() + b.len()) * 8 < raw,
            "{} + {} of {raw}",
            a.len(),
            b.len()
        );
        for (bytes, part) in [(&a, &recs[..1200]), (&b, &recs[1200..])] {
            let got = decode_block(&model, bytes, part.len(), &budget).unwrap();
            assert!(got.iter().map(Vec::as_slice).eq(part.iter().copied()));
        }

        // The model carries the programs its columns decode with, so a small block does not.
        let mut bare = model.clone();
        bare.library = Default::default();
        assert!(!model.library.is_empty());
        let few = &recs[40..104];
        let (shared, _) = encode_block(few, &model, &opts);
        let (alone, _) = encode_block(few, &bare, &opts);
        assert!(
            shared.len() * 5 < alone.len() * 4,
            "{} sharing, {} alone",
            shared.len(),
            alone.len()
        );
        for (bytes, model) in [(&shared, &model), (&alone, &bare)] {
            let got = decode_block(model, bytes, few.len(), &budget).unwrap();
            assert!(got.iter().map(Vec::as_slice).eq(few.iter().copied()));
        }
        // Read against the wrong library, a block is refused or differs; it is never trusted.
        assert!(decode_block(&bare, &shared, few.len(), &budget).is_err());

        // A model that fits nothing still encodes exactly, and says so.
        let noise: Vec<String> = (0..64)
            .map(|i| format!("\u{1}{i:x}\u{2} ~{}", i * 7919))
            .collect();
        let noise: Vec<&[u8]> = noise.iter().map(|r| r.as_bytes()).collect();
        let (c, fit) = encode_block(&noise, &model, &opts);
        assert_eq!((fit.residual_rows, fit.residual_rate()), (64, 1.0));
        let got = decode_block(&model, &c, 64, &budget).unwrap();
        assert!(got.iter().map(Vec::as_slice).eq(noise.iter().copied()));
    }

    #[test]
    fn null_objects_and_empty_arrays_do_not_multiply_shapes() {
        use crate::block::{BlockOptions, decode_block, encode_block, train};
        let rows: Vec<String> = (0..1500)
            .map(|i| {
                let user = if i % 3 == 0 {
                    "null".into()
                } else {
                    format!(r#"{{"id":{},"name":"u{}"}}"#, i % 7, i % 7)
                };
                let team = if i % 5 < 2 {
                    "null".into()
                } else {
                    format!(r#"{{"slug":"t{}"}}"#, i % 4)
                };
                let tags =
                    ["[]", r#"["a"]"#, r#"["a","b{}"]"#][i % 3].replace("{}", &(i % 9).to_string());
                let items = if i % 4 == 0 {
                    "[]".into()
                } else {
                    format!(r#"[{{"id":{}}},{{"id":{}}}]"#, i % 11, i % 13)
                };
                format!(r#"{{"n":{i},"user":{user},"team":{team},"tags":{tags},"items":{items}}}"#)
            })
            .collect();
        let recs: Vec<&[u8]> = rows.iter().map(|r| r.as_bytes()).collect();
        let budget = DecodeBudget::default();
        let model = train(&recs[..600], &TrainOptions::default(), None);
        let root = &model.json_tree.as_ref().unwrap().paths[0];
        assert_eq!(
            root.shapes.len(),
            1,
            "one shape, whichever of user, team, tags and items are empty"
        );
        let (bytes, fit) = encode_block(&recs, &model, &BlockOptions::default());
        assert_eq!(fit.residual_rows, 0);
        let got = decode_block(&model, &bytes, recs.len(), &budget).unwrap();
        assert!(got.iter().map(Vec::as_slice).eq(recs.iter().copied()));

        let opts = PackOptions {
            profile: Some(Profile::from_plan(model)),
            block_bytes: 16 << 10,
            ..Default::default()
        };
        let (mut w, mut comp) = (Writer::new(Vec::new(), opts), Vec::new());
        for r in &recs {
            w.push(r).unwrap();
        }
        let (data, index) = w.finish().unwrap();
        comp.extend_from_slice(&data);
        comp.extend_from_slice(&index);
        // A test of a scalar is answered from columns; one of a whole object rebuilds its rows.
        for (pred, from_columns) in [
            (eq("user.name", "u3"), true),
            (eq("team.slug", "t2"), true),
            (eq("tags", "b5"), true),
            (eq("items.id", "10"), true),
            (Predicate::FieldExists("team.slug".into()), true),
            (eq("user", "null"), false),
            (
                Predicate::Not(Box::new(Predicate::FieldExists("items".into()))),
                false,
            ),
        ] {
            let (rows, stats) = assert_grep(&comp, &recs, &pred);
            assert_eq!(rows, brute(&recs, &pred), "{pred:?}");
            assert!(!rows.is_empty(), "{pred:?}");
            assert!(
                !from_columns || stats.rebuilt == 0,
                "{pred:?} read columns only: {stats:?}"
            );
        }
    }

    #[test]
    fn json_document_packs_like_its_ndjson_and_queries_by_element() {
        let rows: Vec<String> = (0..600)
            .map(|i| {
                format!(
                    "{{\"ts\":\"2026-11-10T11:24:{:02}.{:03}Z\",\"svc\":\"svc-{}\",\"status\":{},\"msg\":\"a, b] and }} c\"}}",
                    (i / 10) % 60,
                    (i * 37) % 1000,
                    i % 5,
                    if i % 50 == 7 { 503 } else { 200 }
                )
            })
            .collect();
        let ndjson = rows.join("\n") + "\n";
        let minified = format!(
            "{{\"meta\":{{\"n\":600}},\"Records\":[{}]}}\n",
            rows.join(",")
        );
        let pretty = format!(
            "[\n  {}\n]\n",
            rows.join(",\n  ").replace("\",\"", "\",\n    \"")
        );

        let nd_len = assert_roundtrip(ndjson.as_bytes(), &PackOptions::default()).len();
        // Small chunks force the array across many frames.
        let chunked = PackOptions {
            chunk_bytes: 64 << 10,
            ..Default::default()
        };
        for doc in [&minified, &pretty] {
            let comp = assert_roundtrip(doc.as_bytes(), &PackOptions::default());
            assert!(
                comp.len() < nd_len + nd_len / 10,
                "doc {} vs ndjson {nd_len}",
                comp.len()
            );
            for comp in [comp, assert_roundtrip(doc.as_bytes(), &chunked)] {
                let mut arch =
                    Archive::open(IoCursor::new(&comp), DecodeBudget::default()).unwrap();
                assert_eq!(arch.file().rows(), 600);
                let mut hits = Vec::new();
                arch.grep(&int_range("status", Some(500), None), |row, rec| {
                    assert!(rec.starts_with(b"{") && rec.ends_with(b"}"));
                    hits.push(row);
                    Ok(())
                })
                .unwrap();
                assert_eq!(hits, (7..600).step_by(50).collect::<Vec<u64>>());
            }
        }
    }

    #[test]
    fn binary_stride_multi_frame_and_random_passthrough() {
        // 1. Synthetic `sao`-style binary file: 28-byte header + 2,000 * 28-byte star records (`f64 ra, f64 dec, 2s sp, i16 mag, f32 pm1, f32 pm2`)
        let mut sao = Vec::with_capacity(28 + 2000 * 28);
        sao.extend_from_slice(b"SAO_STAR_CATALOG_HDR_28BYTES");
        assert_eq!(sao.len(), 28);
        for i in 0..2000u32 {
            let ra: f64 = 1.23456789 + (i as f64) * 0.0001;
            let dec: f64 = -0.54321098 + ((i % 50) as f64) * 0.0002;
            let sp = [b'A', b'0' + (i % 9) as u8];
            let mag: i16 = (450 + (i % 80)) as i16;
            let pm1: f32 = 0.0125 * ((i % 10) as f32);
            let pm2: f32 = -0.005 * ((i % 7) as f32);
            sao.extend_from_slice(&ra.to_le_bytes());
            sao.extend_from_slice(&dec.to_le_bytes());
            sao.extend_from_slice(&sp);
            sao.extend_from_slice(&mag.to_le_bytes());
            sao.extend_from_slice(&pm1.to_le_bytes());
            sao.extend_from_slice(&pm2.to_le_bytes());
        }
        let comp_sao = assert_roundtrip(&sao, &PackOptions::default());
        assert!(
            comp_sao.len() * 3 < sao.len(),
            "sao-like binary should auto-detect 28B stride and beat 3x"
        );

        // 2. Many frames under one model: any row is read without the frames before it.
        let multi_opts = PackOptions {
            chunk_bytes: 64 << 10,
            sync_interval_frames: 3,
            ..Default::default()
        };
        let mut big_lines = Vec::new();
        for i in 0..4000 {
            big_lines.extend_from_slice(
                format!(
                    "ts=2026-11-10T11:24:{:02}Z seq={i:06} user=alice@acme.io status=200\n",
                    i % 60
                )
                .as_bytes(),
            );
        }
        let comp_multi = assert_roundtrip(&big_lines, &multi_opts);
        let recs = lines(&big_lines);
        let mut arch = Archive::open(IoCursor::new(&comp_multi), DecodeBudget::default()).unwrap();
        assert!(arch.file().blocks().count() >= 3);
        let (rows, _) = assert_grep(&comp_multi, &recs, &eq("seq", "003999"));
        assert_eq!(rows, [3999]);
        arch.grep(&contains("seq=003999"), |row, _| {
            let _: () = assert_eq!(row, 3999);
            Ok(())
        })
        .unwrap();
        assert!(
            arch.data_bytes_read * 2 < comp_multi.len() as u64,
            "read {} of {}",
            arch.data_bytes_read,
            comp_multi.len()
        );

        // 3. Incompressible input is stored verbatim: little overhead, and still searchable.
        let mut rng_bytes = vec![0u8; 32768];
        let mut s = 0x1234_5678_9abc_def0u64;
        for b in &mut rng_bytes {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s & 0xff) as u8;
        }
        let rng_opts = PackOptions {
            index: false,
            ..Default::default()
        };
        let comp_rng = assert_roundtrip(&rng_bytes, &rng_opts);
        assert!(
            comp_rng.len() <= rng_bytes.len() + 64,
            "passthrough overhead too high: {} vs {}",
            comp_rng.len(),
            rng_bytes.len()
        );
        let comp_rng = assert_roundtrip(&rng_bytes, &PackOptions::default());
        let recs = lines(&rng_bytes);
        let needle = Predicate::Contains(recs[recs.len() / 2][..3].to_vec());
        let (rows, stats) = assert_grep(&comp_rng, &recs, &needle);
        assert!(rows.contains(&(recs.len() as u64 / 2)));
        assert_eq!(stats.blocks_verbatim, 1);
    }

    /// Records of several shapes; `i / 700` picks the service, so most blocks hold one.
    #[test]
    fn blocks_of_a_stream_share_their_programs() {
        let mut raw = Vec::new();
        for i in 0..6000u64 {
            raw.extend_from_slice(
                format!(
                    "2026-11-10T11:{:02}:{:02}Z req={} bytes={} took={}.{:03}\n",
                    i / 60 % 60,
                    i % 60,
                    100_000 + i * 7,
                    i * 37 % 5000,
                    i % 9,
                    i * 13 % 1000
                )
                .as_bytes(),
            );
        }
        // 94 blocks in one frame, then the same rows over several frames that reuse the model.
        let small = PackOptions {
            block_rows: 64,
            ..Default::default()
        };
        let framed = PackOptions {
            chunk_bytes: 64 << 10,
            ..small.clone()
        };
        for opts in [&small, &framed] {
            let comp = assert_roundtrip(&raw, opts);
            let mut enc = StreamEncoder::new(opts.clone());
            enc.encode_frame(&raw, &mut Vec::new()).unwrap();
            let mut library = Vec::new();
            enc.active_plan().unwrap().library.encode(&mut library);
            // Were the programs in every block, they alone would outweigh the whole file.
            assert!(library.len() > 500, "library of {} bytes", library.len());
            assert!(
                comp.len() < library.len() * 94 / 2,
                "{} bytes, library {}",
                comp.len(),
                library.len()
            );
        }
    }

    fn mixed_record(i: usize) -> String {
        let svc = ["auth", "cart", "search", "billing", "gateway", "ledger"][(i / 700) % 6];
        let ts = format!(
            "2026-11-10T12:{:02}:{:02}.{:03}Z",
            (i / 60) % 60,
            i % 60,
            (i * 37) % 1000
        );
        match i % 23 {
            0 => format!(
                "ts={ts} level=warn svc={svc} msg=\"slow \\\"query\\\"\" dur_ms={}",
                i % 400
            ),
            1 => format!("{ts} WARN {svc}: retry {} after timeout", i % 5),
            2 => format!("  {{\"ts\": \"{ts}\", \"svc\": \"{svc}\", \"status\": 204}}"),
            3 => format!(
                "{{\"svc\":\"{svc}\",\"note\":\"tab\\there \\u00e9 \\\"q\\\"\",\"tags\":[]}}"
            ),
            _ => format!(
                "{{\"ts\":\"{ts}\",\"svc\":\"{svc}\",\"status\":{},\"dur_ms\":{},\"user\":{{\"id\":{},\"name\":\"user{}\"}},\"tags\":[\"{}\",\"t{}\"],\"items\":[{{\"id\":{},\"qty\":{}}},{{\"id\":{}}}],\"msg\":\"{} order {} for clickhouse{}\"}}",
                if i % 50 == 7 { 503 } else { 200 },
                i % 400,
                i % 97,
                i % 97,
                ["red", "green", "blue"][i % 3],
                i % 4,
                i % 11,
                i % 3,
                (i % 11) + 100,
                ["placed", "shipped", "refunded"][i % 3],
                i,
                if i % 300 == 11 { "" } else { " ok" },
            ),
        }
    }

    fn write_file(recs: &[&[u8]], block_rows: usize) -> (Vec<u8>, Vec<u8>) {
        let opts = PackOptions {
            block_rows,
            ..Default::default()
        };
        let mut w = Writer::new(Vec::new(), opts);
        for r in recs {
            w.push(r).unwrap();
        }
        assert_eq!(w.rows(), recs.len() as u64);
        w.finish().unwrap()
    }

    #[test]
    fn file_queries_answer_as_raw_records_do() {
        let rows: Vec<String> = (0..4200).map(mixed_record).collect();
        let recs: Vec<&[u8]> = rows.iter().map(|r| r.as_bytes()).collect();
        let (data, index) = write_file(&recs, 500);
        let mut restored = Vec::new();
        unpack(&data[..], &mut restored).expect("the data region is a stream");
        let src = MemSource {
            data: &data,
            index: &index,
        };
        let budget = DecodeBudget::default();
        let Step::Ready(file) = file::File::open(&src, index.len() as u64, budget).unwrap() else {
            panic!("index in memory")
        };
        assert_eq!((file.rows(), file.blocks().count()), (4200, 9));
        assert!(
            file.paths().iter().any(|p| p == "user.name"),
            "{:?}",
            file.paths()
        );

        let field = |f: &str| f.to_string();
        let time = |lo: &str, hi: &str| Predicate::FieldTimeRange {
            field: field("ts"),
            min_ns: Some(zform_core::lex::parse_timestamp(lo.as_bytes()).unwrap().0),
            max_ns: Some(zform_core::lex::parse_timestamp(hi.as_bytes()).unwrap().0),
        };
        let preds = [
            eq("svc", "ledger"),
            eq("svc", "nowhere"),
            eq("status", "503"),
            eq("user.name", "user42"),
            eq("tags", "blue"),
            eq("items.id", "104"),
            eq("items[].qty", "2"),
            eq("note", "tab\there \u{e9} \"q\""),
            eq("msg", "slow \"query\""),
            Predicate::FieldIn {
                field: field("svc"),
                values: vec![b"cart".to_vec(), b"gateway".to_vec()],
            },
            Predicate::FieldContains {
                field: field("msg"),
                needle: b"refunded order 2".to_vec(),
            },
            int_range("dur_ms", Some(390), None),
            int_range("user.id", None, Some(1)),
            int_range("status", Some(600), None),
            time("2026-11-10T12:10:00Z", "2026-11-10T12:10:30Z"),
            Predicate::FieldExists(field("items.qty")),
            Predicate::FieldExists(field("user")),
            Predicate::FieldExists(field("nope")),
            contains("clickhouse\"}"),
            contains("refunded"),
            contains("\",\"svc\":\"cart"),
            contains("retry 3"),
            contains("\"id\":104}"),
            contains("zzzzzzzz"),
            contains(""),
            word("order"),
            word("clickhouse"),
            word("retry"),
            word("user4"),
            word("20"),
            word("id"),
            word("ok"),
            word("slow"),
            word("green"),
            word("order 2"),
            word("ledger: retry"),
            word("WARN"),
            word("t"),
            word("zzz"),
            word(""),
            Predicate::And(vec![
                eq("svc", "search"),
                contains("shipped"),
                int_range("dur_ms", Some(200), None),
            ]),
            Predicate::Or(vec![eq("status", "503"), eq("level", "warn")]),
            Predicate::Not(Box::new(eq("svc", "auth"))),
            Predicate::And(vec![
                Predicate::Not(Box::new(contains("clickhouse"))),
                Predicate::FieldExists(field("ts")),
            ]),
        ];
        let mut session = file.session();
        for pred in &preds {
            let Step::Ready(got) = session.matches(&src, pred, None).unwrap() else {
                panic!("{pred:?}")
            };
            assert_eq!(
                got.iter().collect::<Vec<_>>(),
                brute(&recs, pred),
                "{pred:?}"
            );
            let within = RowSet::range(450, 1900);
            let Step::Ready(part) = session.matches(&src, pred, Some(&within)).unwrap() else {
                panic!()
            };
            assert_eq!(part, got.intersect(&within), "{pred:?} within rows");
        }
        // Every block here holds rows outside the JSON tree, which are rebuilt to test a field.
        assert!(
            session.stats.blocks_pruned > 0 && session.stats.rebuilt > 0,
            "{:?}",
            session.stats
        );

        // Records and fields of scattered rows.
        let rows_wanted = RowSet::from_sorted([0, 1, 2, 3, 499, 500, 1234, 2300, 2301, 4199]);
        let mut got = Vec::new();
        let Step::Ready(()) = session
            .records(&src, &rows_wanted, |row, rec| {
                got.push(row);
                let _: () = assert_eq!(rec, recs[row as usize]);
                Ok(())
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(got, rows_wanted.iter().collect::<Vec<_>>());

        let paths = [
            "svc",
            "user.name",
            "tags",
            "items.id",
            "msg",
            "dur_ms",
            "nope",
        ];
        let all = RowSet::range(0, 4200);
        let mut got: Vec<Vec<Vec<Vec<u8>>>> = vec![vec![Vec::new(); paths.len()]; recs.len()];
        let mut calls = 0usize;
        let Step::Ready(()) = session
            .fields(&src, &paths, &all, |row, p, v| {
                calls += 1;
                got[row as usize][p].extend(v.map(<[u8]>::to_vec));
                Ok(())
            })
            .unwrap()
        else {
            panic!()
        };
        let mut want_calls = 0usize;
        for (r, rec) in recs.iter().enumerate() {
            for (p, path) in paths.iter().enumerate() {
                let mut want = Vec::new();
                raw_field_values(rec, path, &mut |v| want.push(v.to_vec()));
                want_calls += want.len().max(1);
                assert_eq!(got[r][p], want, "row {r} path {path}: {}", rows[r]);
            }
        }
        assert_eq!(calls, want_calls, "a path with no value reports once");

        // Distinct values with their rows, block by block.
        let mut by_value = std::collections::BTreeMap::<Vec<u8>, Vec<u64>>::new();
        let Step::Ready(()) = session
            .distinct(&src, "svc", &all, |v, rows| {
                let _: () = by_value
                    .entry(logical(v).into_owned())
                    .or_default()
                    .extend(rows);
                Ok(())
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(by_value.len(), 6);
        for (v, mut rows) in by_value {
            rows.sort_unstable();
            let pred = Predicate::FieldExists(field("svc"));
            let want: Vec<u64> = brute(&recs, &pred)
                .into_iter()
                .filter(|&r| {
                    let mut hit = false;
                    raw_field_values(recs[r as usize], "svc", &mut |x| hit |= logical(x) == v);
                    hit
                })
                .collect();
            assert_eq!(rows, want);
        }
    }

    #[test]
    fn a_field_every_row_holds_once_is_read_without_the_row_structure() {
        // `user` is in every record once; `team` has the same values but some records lack it,
        // some hold it twice in an array, and `opt` is null in some.
        let rows: Vec<String> = (0..3000usize)
            .map(|i| {
                let name = match i % 7 {
                    0 => format!("o\\\"brien {}", i % 5),
                    _ => format!("user{}", i % 40),
                };
                let team = match i % 13 {
                    0 => String::new(),
                    1 => format!(",\"team\":[\"{name}\",\"user{}\"]", i % 5),
                    _ => format!(",\"team\":\"{name}\""),
                };
                let opt = if i % 4 == 0 { "null".to_string() } else { format!("\"v{}\"", i % 3) };
                format!("{{\"n\":{i},\"user\":\"{name}\",\"opt\":{opt}{team},\"msg\":\"step {} of job {i}\"}}", i % 9)
            })
            .collect();
        let recs: Vec<&[u8]> = rows.iter().map(|r| r.as_bytes()).collect();
        let (data, index) = write_file(&recs, 500);
        let budget = DecodeBudget::default();
        // Data bytes a fresh session reads to answer `pred`.
        let read = |pred: &Predicate| {
            let fetched = std::cell::Cell::new(0u64);
            let fetch = |ranges: &[ByteRange]| {
                let part = |r: &ByteRange| {
                    let buf = if r.region == Region::Data {
                        &data
                    } else {
                        &index
                    };
                    fetched.set(fetched.get() + if r.region == Region::Data { r.len } else { 0 });
                    buf[r.off as usize..r.end() as usize].to_vec()
                };
                Ok(ranges.iter().map(part).collect())
            };
            let mut cache = RangeCache::default();
            let file = drive(&mut cache, fetch, |c| {
                file::File::open(c, index.len() as u64, budget)
            })
            .unwrap();
            let mut session = file.session();
            let hits = drive(&mut cache, fetch, |c| session.matches(c, pred, None)).unwrap();
            assert_eq!(
                hits.iter().collect::<Vec<_>>(),
                brute(&recs, pred),
                "{pred:?}"
            );
            (fetched.get(), hits.len(), session.stats.rebuilt)
        };
        let field = |f: &str| f.to_string();
        for f in ["user", "team", "opt", "n", "msg"] {
            for pred in [
                eq(f, "user3"),
                eq(f, "o\"brien 2"),
                eq(f, "null"),
                eq(f, "v1"),
                eq(f, "2999"),
                Predicate::FieldIn {
                    field: field(f),
                    values: vec![b"user39".to_vec(), b"v2".to_vec(), b"7".to_vec()],
                },
                Predicate::FieldContains {
                    field: field(f),
                    needle: b"brien".to_vec(),
                },
                int_range(f, Some(2990), None),
                Predicate::FieldExists(field(f)),
                Predicate::Not(Box::new(eq(f, "user3"))),
            ] {
                assert_eq!(read(&pred).2, 0, "{pred:?}");
            }
        }
        for w in ["job", "step 3", "job 2999"] {
            assert_eq!(read(&word(w)).2, 0, "{w}");
        }
        // These may be in the text between values, where records are rebuilt to test them.
        for w in ["user3", "brien", "brien 2", "7", "v1", "o", "null", "team"] {
            read(&word(w));
        }
        let (by_row, hits, _) = read(&eq("user", "user3"));
        let (by_walk, more, _) = read(&eq("team", "user3"));
        assert!(hits > 0 && more > hits);
        assert!(
            by_row < by_walk,
            "{by_row} bytes for the field held once, {by_walk} for the other"
        );

        let src = MemSource {
            data: &data,
            index: &index,
        };
        let Step::Ready(file) = file::File::open(&src, index.len() as u64, budget).unwrap() else {
            panic!()
        };
        let mut session = file.session();
        let paths = ["user", "team", "opt", "n"];
        let mut got: Vec<Vec<Vec<Vec<u8>>>> = vec![vec![Vec::new(); paths.len()]; recs.len()];
        let some = RowSet::from_sorted((0..3000).filter(|r| r % 3 != 1));
        let Step::Ready(()) = session
            .fields(&src, &paths, &some, |row, p, v| {
                let _: () = got[row as usize][p].extend(v.map(<[u8]>::to_vec));
                Ok(())
            })
            .unwrap()
        else {
            panic!()
        };
        for (r, rec) in recs.iter().enumerate() {
            for (p, path) in paths.iter().enumerate() {
                let mut want = Vec::new();
                if r % 3 != 1 {
                    raw_field_values(rec, path, &mut |v| want.push(v.to_vec()));
                }
                assert_eq!(got[r][p], want, "row {r} path {path}");
            }
        }
        assert_eq!(session.stats.rebuilt, 0);
    }

    #[test]
    fn file_queries_fetch_only_what_they_need() {
        // One shape of record, so every block answers from its columns.
        let rows: Vec<String> = (0..6000)
            .map(|i| mixed_record((i / 19) * 23 + 4 + i % 19))
            .collect();
        let recs: Vec<&[u8]> = rows.iter().map(|r| r.as_bytes()).collect();
        let (data, index) = write_file(&recs, 1000);
        let budget = DecodeBudget::default();

        // Data bytes, index bytes and round trips.
        let fetched = std::cell::Cell::new((0u64, 0u64, 0usize));
        let fetch = |ranges: &[ByteRange]| {
            let (mut d, mut i, trips) = fetched.get();
            let parts = ranges.iter().map(|r| {
                let buf = match r.region {
                    Region::Data => {
                        d += r.len;
                        &data
                    }
                    Region::Index => {
                        i += r.len;
                        &index
                    }
                };
                buf[r.off as usize..r.end() as usize].to_vec()
            });
            let parts = parts.collect();
            fetched.set((d, i, trips + 1));
            Ok(parts)
        };
        let mut cache = RangeCache::default();
        let file = drive(&mut cache, fetch, |c| {
            file::File::open(c, index.len() as u64, budget)
        })
        .unwrap();
        assert_eq!(
            fetched.get().2,
            1,
            "the root comes with the first read of the index"
        );
        let mut session = file.session();

        let pred = eq("svc", "ledger");
        let hits = drive(&mut cache, fetch, |c| session.matches(c, &pred, None)).unwrap();
        assert_eq!(hits.iter().collect::<Vec<_>>(), brute(&recs, &pred));
        assert!(!hits.is_empty() && hits.len() < 2000);
        let (d, _, trips) = fetched.get();
        assert!(
            d * 10 < data.len() as u64,
            "equality read {d} of {} data bytes",
            data.len()
        );
        assert_eq!(
            trips, 2,
            "one round trip for the index and one for the columns"
        );
        assert!(
            session.stats.blocks_pruned >= 3 && session.stats.blocks_whole == 0,
            "{:?}",
            session.stats
        );

        let mut total = 0u64;
        drive(&mut cache, fetch, |c| {
            total = 0;
            session.fields(c, &["dur_ms"], &hits, |_, _, v| {
                total += std::str::from_utf8(v.unwrap())
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
                Ok(())
            })
        })
        .unwrap();
        assert!(total > 0);
        let (d, _, _) = fetched.get();
        assert!(
            d * 10 < data.len() as u64,
            "projection read {d} of {} data bytes",
            data.len()
        );
        assert_eq!(
            (session.stats.blocks_whole, session.stats.rebuilt),
            (0, 0),
            "{:?}",
            session.stats
        );

        // A substring that sits in one column reads that column alone. Here it is the one column
        // that does not compress away, so it is most of the data.
        let pred = contains("refunded order 12");
        let hits = drive(&mut cache, fetch, |c| session.matches(c, &pred, None)).unwrap();
        assert_eq!(hits.iter().collect::<Vec<_>>(), brute(&recs, &pred));
        assert!(!hits.is_empty());
        let (d, _, _) = fetched.get();
        assert!(
            d * 4 < data.len() as u64 * 3,
            "substring read {d} of {} data bytes",
            data.len()
        );
        assert_eq!(session.stats.blocks_whole, 0, "{:?}", session.stats);

        // Records come from whole blocks, and only the blocks that hold them.
        let one = RowSet::from_sorted([4321]);
        let before = fetched.get().0;
        drive(&mut cache, fetch, |c| {
            session.records(c, &one, |_, rec| {
                let _: () = assert_eq!(rec, recs[4321]);
                Ok(())
            })
        })
        .unwrap();
        assert!(fetched.get().0 - before < data.len() as u64 / 4);

        // Without filters the answer is the same and no filter page is read.
        let index_before = fetched.get().1;
        let mut unfiltered = file.session().without_filters();
        let pred = eq("user.name", "user42");
        let same = drive(&mut cache, fetch, |c| unfiltered.matches(c, &pred, None)).unwrap();
        assert_eq!(same.iter().collect::<Vec<_>>(), brute(&recs, &pred));
        assert_eq!(
            fetched.get().1,
            index_before,
            "the directories were at hand"
        );
    }
}
