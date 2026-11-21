# `zform` — Lossless Structure-Aware Compression

`zform` is a lossless compressor for logs, telemetry and event streams, available as a CLI and a Rust library. It learns the structure of its input, compresses it and lets you search the archive directly.

<!-- bench:comparison -->
<!-- /bench:comparison -->

```bash
zform pack app.log -o app.zfm      # any bytes in; no schema, no format flag
zform unpack app.zfm -o app.log    # identical bytes out
zform grep "timed out" app.zfm     # search the compressed file
```

## Why it compresses better

Machine-written data is full of structure: the same line layouts and keys over and over, fields that count up, tick forward or follow from other fields. Byte-stream compressors (`gzip`, `zstd`, `xz`) match repeated byte sequences and never see that structure. Columnar formats (`Parquet`, `Arrow`) do, but they need a schema up front and discard the original bytes.

`zform` keeps a stream compressor's contract and adds a columnar format's model. It learns how records split into fields, what each field's values look like, and how fields relate (`end = start + dur`, a URL built from other fields), then stores the model and only what the model cannot predict:

- **Byte-exact output.** Formatting survives (`12.300`, `007`, `-0`, key order, whitespace). Mixed formats can share a file, and lines the model does not recognise are stored verbatim.
- **Queries on the compressed file.** `zform grep` and `zform select` filter and project records without unpacking, skipping blocks by index and decoding only the columns a query touches.

`zform` suits archives of logs, events and telemetry that you want to filter in place. For opaque data such as media or encrypted payloads, it falls back to byte-stream compression.

## Usage

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

# Try context mixing on selected integer streams for smaller files (slower decoding)
zform pack --context-mixing app.log -o app.zfm

# Try HDFS field predictions through the same payload engine
zform pack --hdfs-predictors --context-mixing HDFS.log -o hdfs.zfm

# Store an event log sorted on its entity fields; unpack restores the original order
zform pack --best --order-by type,repo.id events.ndjson -o events.zfm

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
raw bytes ──►│ frame │──►│ model │──►│ encode │──►│ compress │──► .zfm
             └───────┘   └───────┘   └────────┘   └──────────┘
              records     fields      columns      zstd or LZ4
                                                   + block index
```

1. **Frame.** Splits the stream into records: lines, length-prefixed or fixed-width binary records, or the elements of a JSON array. Incompressible chunks skip modelling and go straight to `zstd`.
2. **Model.** Learns record layouts from a sample, drawn from across the file when packing one, then splits each record into fields. A record or JSON object in a layout the model lacks is stored whole. Later input reuses the model, and the writer retrains when the stream drifts.
3. **Encode.** Gathers each field into a column and tries candidate encodings per block, keeping the smallest. A field derivable from others (`end_us = start_us + dur_us`, a URL embedding a repository name) is stored as the rule.
4. **Compress & index.** Compresses the columns with `zstd` or `LZ4` into independent blocks. A trailing index holds byte ranges, per-column bounds and a substring filter, so a query fetches and decodes only what it needs.

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

`zform inspect -v` shows the compressed bytes each column takes.

## Architecture

The writer holds all knowledge of formats. A file carries its own decoding instructions, and the reader runs a small fixed set of generic operations. A new log syntax or value type can use the existing decoder operations.

The instructions describe two layers:

- **Record layout.** A record follows a *shape*: literal text with slots filled from columns. The writer's front ends for `logfmt`, CSV/TSV, CRI and syslog envelopes, fixed-width binary structs and learned free-text templates all produce shapes, with separators, keys and quotes as literal text. Nested JSON has a tree model of its own.
- **Column values.** A column carries a small graph of operations over byte buffers and integer streams: dictionaries, deltas, references to other columns (a copy, or a lookup by a shared key such as a block id), and short programs that do arithmetic and format bytes. Writer-side codec packs recognise timestamps, decimals, UUIDs, network addresses, hex ids and pseudo-random sequences (`java.util.Random`, Python's Mersenne Twister) and express each in those operations.

The reader therefore has no timestamp, `logfmt` or generator code. Programs have no branches, loops or writes to memory, so a reader validates one in a single pass and bounds its cost before running it. Code that blocks share is stored once with the model. An application can register its own `CodecPack` for a column or `RelationPack` for relationships between columns without changing the format.

The `.zfm` container serves streaming pipelines and indexed archives:

- **Standalone or profiled:** Models travel in the stream, so every file decodes alone. `zform train` saves a model as a `.zfp` profile so small chunks compress well from the first record.
- **Concatenation:** `cat a.zfm b.zfm` is a valid stream that unpacks to both inputs in order.
- **Independent blocks:** No block refers to another, so a reader fetches and decodes single blocks and columns. Columns hold values only where a field occurs, so sparse JSON pays nothing for absent fields.
- **Integrity:** Frame headers carry CRC32 checksums and payloads XXH64.
- **Embeddable:** A storage engine can keep `zform` blocks or files inside its own objects, next to its own columns and indexes ([`DESIGN.md`](DESIGN.md)).

The wire format is experimental and changes without notice; keep original copies of data you cannot repack.

## Benchmarks

The comparison uses the full Loghub HDFS log, the complete GitHub Archive hour `2020-06-17-16`, and the whole Silesia `sao` star catalog. `zform` sizes include the query index.

<!-- bench:presets -->
<!-- /bench:presets -->

`--best` builds a shared dictionary from recurring values. `--order-by type,repo.id` groups events in stored order, which `grep` and `select` use. Unpack buffers one run, up to 512 MiB by default, to restore input order. The HDFS predictor preset models block IDs, write pipelines and per-node clocks.

The [benchmark workflow](https://github.com/wjordan/zform/actions/workflows/bench.yml) reports per-corpus throughput and peak memory. [Benchmark commands](bench/README.md) cover the same inputs, queries and CloudTrail.


## Repository layout

- [`crates/zform-core`](crates/zform-core) — framing, modelling, writer-side codec packs, the payload decoder and the query engine.
- [`crates/zform`](crates/zform) — public Rust API: `pack`, `unpack`, `train`, `Archive` for queries, and the `file` and `block` modules for embedding.
- [`crates/zform-cli`](crates/zform-cli) — the `zform` command-line tool.
- [`DESIGN.md`](DESIGN.md) — layers, queryable files and the host embedding interface.
- [`FORMAT.md`](FORMAT.md) — binary layout of streams, column payloads, index regions and profiles.
- [`PROGRAMS.md`](PROGRAMS.md) — the instruction set column payloads compute with.
- [`bench/`](bench) — corpus generators and benchmark runners.

## Related work

- **Format-aware compression:** OpenZL (Meta, 2024–2025) routes typed streams through codec graphs. Zstandard (Collet & Kucherawy, RFC 8878) is `zform`'s default entropy backend.
- **Log compression:** CLP (Rodrigues et al., *OSDI 2021*) and LogGrep (Wei et al., *EuroSys 2023*) split logs into templates and variables for compressed search; Drain (He et al., *ICWS 2017*) parses log templates online.
- **Semi-structured columns:** JSON Tiles (Durner et al., *SIGMOD 2021*; Zeng et al., *VLDB 2023*) extracts typed sub-columns from heterogeneous JSON documents.
- **Lightweight codecs:** FSST (Boncz, Neumann & Leis, *VLDB 2020*) for strings; Gorilla (Pelkonen et al., *VLDB 2015*), BtrBlocks (Kuschewski et al., *SIGMOD 2023*), and FastLanes (Afroozeh & Boncz, *VLDB 2023*) for delta and frame-of-reference encodings; Blosc byte-shuffle (Alted, 2010).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
