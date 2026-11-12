# `zform` — Lossless Structure-Aware Compression

`zform` is a lossless compression CLI and Rust library for structured logs and event streams: JSON and NDJSON, OpenTelemetry metrics/logs/traces, `logfmt`, Kubernetes CRI and syslog lines, plain-text application logs, CSV/TSV, and fixed-stride binary records.

Byte-stream compressors (`gzip`, `zstd`, `xz`) match repeated byte sequences and never see record structure. Columnar formats (`Parquet`, `Arrow`) exploit field types and support selective reads, but require an upfront schema, discard the original bytes, and treat columns independently.

`zform` keeps the drop-in contract of a stream compressor—arbitrary bytes in, identical bytes out, no schema—while learning a structural model of the input: record shapes and templates, typed columns, and cross-field relationships (such as `end = start + dur` or URLs built from other fields). It stores that model plus only the residual it cannot predict, in indexed columnar blocks:

- **Higher density:** **1.4–4.6x smaller** than `zstd -3` and **1.2–3.0x smaller** than `zstd -19` on structured corpora ([benchmarks](#benchmarks)), built-in query index included.
- **Byte-exact fidelity on messy inputs:** Preserves exact formatting (`12.300`, `007`, `-0`, key order, whitespace). Mixed formats can share a file, unrecognized lines fall back to verbatim storage, and incompressible data passes straight through to `zstd`.
- **Selective reads, as in a columnar format:** `zform select` and `zform grep` filter and project `.zfm` archives without unpacking them, pruning blocks by index and decoding only the columns a query touches.

The tradeoff is CPU: on a single core, `zform` compresses logs at **21–44 MB/s** (`zstd -3`: 550–620 MB/s; `zstd -19`: 4–6 MB/s) and decompresses them at **360–510 MB/s** (`zstd`: 1,000–1,600 MB/s).

## Benchmarks

Compressed size and ratio (`raw size / compressed size`):

| Corpus | Raw | `zstd -3` | `zstd -19` | `xz -6` | `zform` |
|---|---:|---:|---:|---:|---:|
| HDFS DataNode logs | 8.00 MiB | 0.72 MiB (11.1x) | 0.47 MiB (17.0x) | 0.49 MiB (16.2x) | **0.16 MiB (49.8x)** |
| GitHub Archive NDJSON | 8.00 MiB | 0.89 MiB (8.9x) | 0.69 MiB (11.6x) | 0.69 MiB (11.6x) | **0.60 MiB (13.4x)** |
| Silesia `sao` star catalog (binary) | 6.92 MiB | 5.29 MiB (1.31x) | 4.77 MiB (1.45x) | 4.21 MiB (1.64x) | **3.57 MiB (1.94x)** |

`zform` outputs include the trailing query index, which accounts for 7% of the single-block, 766-column GitHub Archive file (`0.56 MiB` / `14.4x` with `--no-index`) and ~3% on multi-block files. Larger-scale archive and cold-query benchmarks are in [`DESIGN.md`](DESIGN.md#evidence).

Single-core throughput and peak memory on the two log corpora:

| Codec | Compress MB/s | Decompress MB/s | Compress RSS MiB | Decompress RSS MiB |
|---|---:|---:|---:|---:|
| `zstd -3` | 550–620 | 1,300–1,600 | 6.6–6.7 | 5.3–5.7 |
| `zstd -19` | 3.6–6.1 | 1,000–1,100 | 92 | 11 |
| `xz -6` | 8.7–9.2 | 360–440 | 79–90 | 10 |
| `zform` | 21–44 | 360–510 | 31–170 | 15–16 |

### Where it fits

- **Good fits:** Log archives, audit and event streams, JSON/NDJSON exports, and telemetry—especially heterogeneous streams (e.g., JSON, `logfmt`, and stack traces in one file) and archives that benefit from direct filtering without full decompression or warehouse ingestion.
- **Poor fits:**
  - **Opaque or high-entropy data** (media, existing archives, encrypted payloads), which bypass structural modeling and compress at plain `zstd` ratios plus a few bytes of framing.
  - **Latency-critical hot paths** where raw `lz4` or `zstd -1` speed matters more than size or queryability.

## Quickstart

### CLI

```bash
cargo build --release --workspace   # binary: ./target/release/zform
```

```bash
# Compress and restore files (or stream via stdin/stdout)
zform pack events.ndjson -o events.zfm
zform unpack events.zfm -o events.ndjson
zform pack < app.log > app.zfm

# Pack a JSON document (elements of its main array become records)
zform pack cloudtrail.json -o cloudtrail.zfm

# Inspect stream metadata, ratio, and top columns by compressed size
zform inspect -v events.zfm

# Query compressed files directly (--stats reports blocks pruned and bytes read)
zform select -f id,type,actor.login,repo.name --where type=PushEvent events.zfm
zform grep --where level=ERROR --range dur_us=10000.. events.zfm --stats

# Train a reusable profile once so small chunks compress well without warm-up
zform train sample.ndjson -o github.zfp
zform pack --profile github.zfp live-chunk.ndjson -o live-chunk.zfm
```

### Rust library

```rust
use std::io::Cursor;
use zform::{Archive, DecodeBudget, PackOptions, Predicate, pack, unpack};

fn main() -> zform::Result<()> {
    let input = br#"{"ts":"2026-11-10T12:00:00Z","service":"auth","status":200,"dur_ms":4}
{"ts":"2026-11-10T12:00:01Z","service":"auth","status":503,"dur_ms":142}
"#;

    let mut compressed = Vec::new();
    pack(&input[..], &mut compressed, &PackOptions::default())?;

    let mut restored = Vec::new();
    unpack(&compressed[..], &mut restored)?;
    assert_eq!(&input[..], &restored[..]);

    // Project three fields for rows with status >= 500, decoding only needed columns
    let pred = Predicate::FieldIntRange {
        field: "status".into(),
        min: Some(500),
        max: None,
    };
    let fields = ["ts", "service", "dur_ms"];
    let mut archive = Archive::open(Cursor::new(&compressed), DecodeBudget::default())?;
    archive.select(&fields, &pred, |row, field, value| {
        println!("row {row} {}: {:?}", fields[field], value.map(String::from_utf8_lossy));
        Ok(())
    })?;
    Ok(())
}
```

## How it works

```text
             ┌───────┐   ┌───────┐   ┌────────┐   ┌──────────┐
raw bytes ──►│ frame │──►│ parse │──►│ encode │──►│ compress │──► .zfm
             └───────┘   └───────┘   └────────┘   └──────────┘
              records     fields      typed        zstd or LZ4
                                      columns      + block index
```

1. **Frame.** Detects record boundaries (line or NUL delimiters, length prefixes, fixed binary strides, or elements of a JSON array) and splits the stream into records. Incompressible chunks skip parsing and pass directly to `zstd`.
2. **Parse.** Learns a structural model from a sample of records, then splits each record into fields. Records that do not match the model go to a verbatim residual column. Subsequent frames reuse the model and retrain automatically if the stream drifts.
3. **Encode.** Transposes fields into typed columns. Per block, each column is trial-encoded across candidate codecs and keeps the smallest output. Fields derivable from others (`end_us = start_us + dur_us`, or a URL embedding a repository name) are stored as rules at near-zero per-record cost.
4. **Compress & index.** Compresses column buffers with `zstd` (default) or `LZ4` into independent blocks (up to 65,536 records or 8 MiB). A trailing index records part offsets, per-block column min/max bounds, character bitmasks, and a substring filter so queries fetch and decode only the byte ranges they need.

For example, these three records:

```text
{"ts":"2026-11-10T12:00:00.004Z","svc":"auth","status":200,"start_us":1000,"dur_us":412,"end_us":1412}
{"ts":"2026-11-10T12:00:00.019Z","svc":"auth","status":200,"start_us":1015,"dur_us":388,"end_us":1403}
{"ts":"2026-11-10T12:00:00.031Z","svc":"cart","status":503,"start_us":1027,"dur_us":97,"end_us":1124}
```

encode into:

```text
shape      const     {"ts":"_","svc":"_","status":_,"start_us":_,"dur_us":_,"end_us":_}
ts         delta     2026-11-10T12:00:00.004Z, +15ms, +12ms
svc        dict      [auth, cart] -> 0, 0, 1
status     int       200, 200, 503
start_us   delta     1000, +15, +12
dur_us     int       412, 388, 97
end_us     derived   start_us + dur_us
```

Codecs are selected per block; `zform inspect -v` shows the compressed byte breakdown by column.

## Model and container

Models are plain data interpreted by a single fixed decoder (specified in [`FORMAT.md`](FORMAT.md)) and built from two kinds of primitives:

- **Structural families** describe record layout: nested JSON trees, flat `logfmt` pairs, CSV/TSV rows, fixed-width binary structs, Kubernetes CRI or syslog envelopes around inner payloads, and literal text templates with variable slots. Templates serve as a general fallback for line-oriented text, splitting custom log formats into columns without dedicated parsers.
- **Value codecs** encode column contents: timestamps, integers, decimals, UUIDs/ULIDs, network addresses, per-column and shared dictionaries, pooled free text, hex IDs, cross-column derivations (copies, affine sums, interpolated strings), and pseudo-random values stored as positions in their generator's output (`java.util.Random`, Python's Mersenne Twister). A column's payload spells out the steps that rebuild it, so the decoder has no code per value type; steps the blocks of a model share are stored once with the model.

Regardless of how records are modeled, decoding preserves exact byte representation: key order, whitespace, `null` versus `"null"`, trailing zeros (`12.300`), leading zeros and signs (`007`, `-0`, `+42`), and timestamp separators and zones.

The `.zfm` container is designed for both streaming pipelines and indexed archives:

- **Self-describing & profile-ready:** Models travel inside the stream so every file decodes standalone, or can be saved as `.zfp` profiles (`zform train`, extended with `train --extend`, and applied via `pack --profile`) to avoid cold-start overhead on small chunks.
- **Stream concatenation:** `cat a.zfm b.zfm > ab.zfm` produces a valid stream that unpacks to both inputs in order, just like `gzip` and `zstd` (`grep` and `select` require a single-run index).
- **Sparse columns:** Columns store values only for records where a field occurs, so wide, sparse JSON pays nothing per row for absent fields.
- **Self-contained blocks:** Blocks share no state across boundaries, allowing readers to fetch and decode individual blocks and column parts on demand.
- **Integrity:** Frame headers carry CRC32 checksums and frame payloads carry XXH64 content checksums.
- **Embeddable layout:** Storage engines and databases can embed `zform` blocks or files inside their own objects alongside external columns and indexes ([`DESIGN.md`](DESIGN.md)).

## Status

Experimental. The wire format is not frozen and changes without notice; keep original copies of data you cannot repack. Known limitations are tracked in [`DESIGN.md`](DESIGN.md#status).

## Repository layout

- [`crates/zform-core`](crates/zform-core) — wire format, framing, structural models, column codecs, trainer, and query engine.
- [`crates/zform`](crates/zform) — public Rust API: `pack`, `unpack`, `train`, `Archive` for querying files, and lower-level `file` and `block` modules for host embedding.
- [`crates/zform-cli`](crates/zform-cli) — the `zform` command-line tool.
- [`FORMAT.md`](FORMAT.md) — binary specification for `ZFM1` streams, `ZFIX` index regions, and `ZFP1` profiles.
- [`DESIGN.md`](DESIGN.md) — queryable file architecture, host embedding interface, known limits, and benchmark evidence.
- [`bench/`](bench) — corpus generators and benchmark runners.

## Related work

- **Format-aware compression:** OpenZL (Meta, 2024–2025) routes typed streams through codec graphs. Zstandard (Collet & Kucherawy, RFC 8878) is `zform`'s default entropy backend.
- **Log compression:** CLP (Rodrigues et al., *OSDI 2021*) and LogGrep (Wei et al., *EuroSys 2023*) split logs into templates and variables for compressed search; Drain (He et al., *ICWS 2017*) parses log templates online.
- **Semi-structured columns:** JSON Tiles (Durner et al., *SIGMOD 2021*; Zeng et al., *VLDB 2023*) extracts typed sub-columns from heterogeneous JSON documents.
- **Lightweight codecs:** FSST (Boncz, Neumann & Leis, *VLDB 2020*) for strings; Gorilla (Pelkonen et al., *VLDB 2015*), BtrBlocks (Kuschewski et al., *SIGMOD 2023*), and FastLanes (Afroozeh & Boncz, *VLDB 2023*) for delta and frame-of-reference encodings; Blosc byte-shuffle (Alted, 2010).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
