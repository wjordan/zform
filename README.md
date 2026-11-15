# `zform` — Lossless Structure-Aware Compression

`zform` is a lossless compressor for logs, telemetry and event streams, available as a CLI and a Rust library. It learns the structure of its input and writes files **1.3–3.1x smaller than `zstd -19`** that can be searched without unpacking them.

| Corpus | Raw | `zstd -3` | `zstd -19` | `xz -6` | `zform` |
|---|---:|---:|---:|---:|---:|
| HDFS DataNode logs | 8.00 MiB | 0.72 MiB (11.1x) | 0.47 MiB (17.0x) | 0.49 MiB (16.2x) | **0.15 MiB (52.9x)** |
| GitHub Archive NDJSON | 8.00 MiB | 0.89 MiB (8.9x) | 0.69 MiB (11.6x) | 0.69 MiB (11.6x) | **0.55 MiB (14.5x)** |
| Silesia `sao` star catalog (binary) | 6.92 MiB | 5.29 MiB (1.31x) | 4.77 MiB (1.45x) | 4.21 MiB (1.64x) | **3.57 MiB (1.94x)** |

```bash
zform pack app.log -o app.zfm      # any bytes in; no schema, no format flag
zform unpack app.zfm -o app.log    # identical bytes out
zform grep "timed out" app.zfm     # search the compressed file
```

## Why it compresses better

Machine-written data is full of structure: the same line layouts and keys over and over, fields that count up, tick forward or follow from other fields. Byte-stream compressors (`gzip`, `zstd`, `xz`) match repeated byte sequences and never see that structure. Columnar formats (`Parquet`, `Arrow`) do, but they need a schema up front and discard the original bytes.

`zform` keeps a stream compressor's contract and adds a columnar format's model. It learns how records split into fields, what each field's values look like, and how fields relate (`end = start + dur`, a URL built from other fields), then stores the model and only what the model cannot predict:

- **Smaller files.** 1.5–4.8x smaller than `zstd -3` and 1.3–3.1x smaller than `zstd -19` on the corpora above, query index included.
- **Byte-exact output.** Formatting survives (`12.300`, `007`, `-0`, key order, whitespace). Mixed formats can share a file, and lines the model does not recognise are stored verbatim.
- **Queries on the compressed file.** `zform grep` and `zform select` filter and project records without unpacking, skipping blocks by index and decoding only the columns a query touches.

The price is CPU. On one core `zform` compresses logs at 25–92 MB/s, between `zstd -19` (4–6 MB/s) and `zstd -3` (580–680 MB/s), and decompresses at 360–430 MB/s, a quarter to a third of `zstd`'s speed. It suits archives of logs, events and telemetry, above all ones worth filtering in place. It does not suit hot paths where `lz4` or `zstd -1` speed matters more than size, and it gains nothing on opaque data (media, archives, encrypted payloads), which passes through at plain `zstd` ratios.

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

The writer holds all knowledge of formats. A file carries its own decoding instructions, and the reader runs a small fixed set of generic operations. Supporting a new log syntax or value type means new writer code; existing readers decode the result.

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

The size table at the top uses 8 MiB slices of each log corpus and the whole of `sao`. `zform` sizes include its query index: 9% of the single-block, 830-column GitHub Archive file (`0.50 MiB` / `16.0x` with `--no-index`) and about 6% of multi-block files.

Single-core throughput and peak memory on the two log corpora:

| Codec | Compress MB/s | Decompress MB/s | Compress RSS MiB | Decompress RSS MiB |
|---|---:|---:|---:|---:|
| `zstd -3` | 580–680 | 1,500 | 6.6–6.8 | 5.7–6.0 |
| `zstd -19` | 3.6–6.3 | 1,100–1,200 | 92 | 11 |
| `xz -6` | 8.7–9.4 | 380–440 | 79–91 | 10 |
| `zform` | 25–92 | 360–430 | 38–250 | 26–30 |

Larger inputs give every compressor more context. On a full hour of GitHub Archive (499 MB), `zform pack --best` writes 21.4 MB (23.3x), against 37.8 MB for `zstd -19` and 26.6 MB for `zstd --ultra -22 --long=31`; the default setting writes 30.4 MB. `--best` adds a file dictionary of values that recur across blocks, so blocks stay small and independently readable. `--order-by type,repo.id` stores the events sorted by type and repository, which brings `--best` to 18.4 MB (27.1x) and the default to 23.7 MB; unpacking restores the original order and holds a run of up to 512 MiB in memory, and `grep` and `select` answer in the stored order.

On the whole Loghub HDFS log (1.58 GB), `zform pack` writes 35.0 MB (45x) in 12.2 s on one core and unpacks it in 2.6 s. `gzip -6` takes 13.1 s for 155.7 MB; `zstd -19` and `xz -9` write 98.3 MB and 96.3 MB. Adding `-l 6 --hdfs-predictors --context-mixing`, which models how HDFS hands out block ids, builds write pipelines, numbers threads and deletes blocks, brings the file to 17.6 MB (90x) in 18.4 s, unpacked in 5.3 s. `pack` encodes frames on every core by default and writes the same bytes: 2.3 s and 5.1 s on 24 cores.

[`bench/README.md`](bench/README.md) has the commands for these tables, cold-query measurements and a CloudTrail comparison.

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
