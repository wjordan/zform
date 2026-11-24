# zform

A structure-aware lossless compressor and queryable archive format for logs, event streams, telemetry, and structured binary files, available as a CLI and a Rust library.

### Archive Size

<!-- bench:comparison -->
| Corpus | Raw | `zstd -3` | `zstd -19` | `xz -6` | `zform` |
|---|---:|---:|---:|---:|---:|
| HDFS DataNode logs | 1,504.88 MiB | 145.88 MiB (10.3x) | 93.86 MiB (16x) | 95.08 MiB (15.8x) | **16.88 MiB (89.1x)** |
| GitHub Archive NDJSON | 476.32 MiB | 50.54 MiB (9.42x) | 36.03 MiB (13.2x) | 36.26 MiB (13.1x) | **17.46 MiB (27.3x)** |
| Silesia sao | 6.92 MiB | 5.30 MiB (1.31x) | 4.77 MiB (1.45x) | 4.21 MiB (1.64x) | **3.38 MiB (2.05x)** |
<!-- /bench:comparison -->

### Throughput (Pack / Unpack MB/s, Single-Threaded)

| Corpus | `gzip -6` | `zstd -3` | `zstd -19` | `xz -6` | `zform` (default) | `zform` (best) |
|---|---:|---:|---:|---:|---:|---:|
| HDFS DataNode logs | 118 / 435 | **802** / 2,207 | 3.2 / **2,794** | 6.8 / 448 | 148 / 1,113 | 97 / 397 |
| GitHub Archive NDJSON | 106 / 394 | **699** / 2,334 | 6.6 / **2,335** | 9.8 / 380 | 211 / 1,370 | 17 / 883 |
| Silesia sao | 23 / 114 | **114** / **471** | 6.0 / 466 | 4.5 / 44 | 34 / 466 | 1.8 / 467 |

<details>
<summary>Benchmark details</summary>

All runs are single-threaded on an AMD Ryzen 7 7700, include CLI startup and file I/O, verify exact SHA-256 roundtrips, and include `zform`'s trailing query index. [Full CI report](https://github.com/wjordan/zform/actions/runs/45268934613) · [Benchmark commands](bench/README.md)

<!-- bench:presets -->
| Corpus | Pack flags | Archive |
|---|---|---:|
| HDFS DataNode logs | `--best` | 31.70 MiB (47.5x) |
| HDFS DataNode logs | `-l 6 --hdfs-predictors --context-mixing` | 16.88 MiB (89.1x) |
| GitHub Archive NDJSON | `--best` | 20.05 MiB (23.8x) |
| GitHub Archive NDJSON | `--order-by type,repo.id` | 22.48 MiB (21.2x) |
| GitHub Archive NDJSON | `--best --order-by type,repo.id` | 17.46 MiB (27.3x) |
| Silesia sao | `--best` | 3.38 MiB (2.05x) |
<!-- /bench:presets -->

Default `zform` (no flags) produces **33.41 MiB (45x)** on HDFS, **28.63 MiB (16.6x)** on GitHub Archive, and **3.57 MiB (1.94x)** on Silesia `sao`. On GitHub Archive, `zform select` projects four fields across 63,340 `PushEvent` records in **48 ms**, reading **2.11 MB** of compressed data.

</details>

```bash
zform pack app.log -o app.zfm      # any bytes in; no schema, no format flag
zform unpack app.zfm -o app.log    # identical bytes out
zform grep "timed out" app.zfm     # query the compressed archive in place
```

## How `zform` Is Different

Stream compressors (`gzip`, `zstd`, `xz`) match flat byte windows and stay blind to data structure. Columnar formats (`Parquet`, `Arrow`) and log stores (`CLP`) require schemas or fixed codecs and discard original formatting.

`zform` keeps a stream compressor's lossless, schema-free contract while compiling each input's **generative model** into small decoding programs stored inside the archive:

```text
          ┌───────────────────┐   ┌────────────────────────────┐   ┌─────────────────┐
raw bytes▶│ Smart Writer      │──▶│ .zfm Archive               │──▶│ Format-Blind VM │
          │ discovery + packs │   │ programs, residuals, index │   │ unpack / query  │
          └───────────────────┘   └────────────────────────────┘   └─────────────────┘
```

- **Zero-schema structure discovery.** The writer infers recursive log templates, nested sparse JSON trees, and fixed-width binary structs automatically while preserving every byte verbatim, falling back to `zstd` on incompressible data.
- **Cross-field and stateful prediction.** Rather than encoding columns in isolation, `zform` models peer derivations (such as URLs and affine arithmetic), per-entity running sums, file-wide entity tables, reversible entity sorting, and optional nibble context mixing.
- **Programs instead of fixed codecs.** Each column stores a self-describing decoding graph over a small, branchless VM (25 graph nodes, 35 lane ops, bounded recurrences). The reader contains no format, log, or PRNG logic, validating each graph in one pass under a bounded execution budget.
- **Extensible writer plug-ins.** Because decoding logic travels as data, custom encoders can target bespoke domain structure without changing the reader or wire format. That makes it practical to write, or have an LLM tune and optimize, specialized encoders for specific workloads; the 89.1x HDFS predictor is one example.
- **Queryable in place.** Independent blocks and a trailing index let `zform grep` and `zform select` prune blocks and decode only the columns a query touches, without unpacking the file.

<details>
<summary>Corpus breakdown</summary>

- **HDFS DataNode logs (`1,504.88 MiB -> 16.88 MiB`, 89.1x):** Learns log templates, cracks the 48-bit `java.util.Random` LCG generating block IDs into a bounded recurrence, stores each block's three pipeline replicas once in a file table, tracks per-datanode clocks and thread counters with ordered per-key running sums, and applies bounded nibble context mixing to integer residuals.
- **GitHub Archive NDJSON (`476.32 MiB -> 17.46 MiB`, 27.3x):** Learns the sparse heterogeneous JSON tree automatically, derives redundant URLs and identifiers from peer columns, clusters recurring repositories with `--order-by type,repo.id` (restoring exact input order on unpack), and builds a shared file dictionary.
- **Silesia `sao` star catalog (`6.92 MiB -> 3.38 MiB`, 2.05x):** Detects the 28-byte binary header and 28-byte struct stride automatically, splits the star fields (`f64` coordinates, spectral type, `i16` magnitude, `f32` proper motion), and compiles lane programs for IEEE-754 ordered-float deltas, prefix scans, and byte-plane transposes.

</details>

## Usage

```bash
cargo build --release --workspace   # binary: ./target/release/zform
```

```bash
zform pack events.ndjson -o events.zfm
zform unpack events.zfm -o events.ndjson
zform select -f id,type,actor.login,repo.name --where type=PushEvent events.zfm
zform inspect -v events.zfm
```

<details>
<summary>More CLI examples</summary>

```bash
# Stream via stdin/stdout, or pack a JSON document (elements of its main array become records)
zform pack < app.log > app.zfm
zform pack cloudtrail.json -o cloudtrail.zfm

# High-density options: shared dictionary, reversible entity sorting, context mixing, custom packs
zform pack --best --order-by type,repo.id events.ndjson -o events.zfm
zform pack --context-mixing app.log -o app.zfm
zform pack -l 6 --hdfs-predictors --context-mixing HDFS.log -o hdfs.zfm

# Filtered search with index pruning stats
zform grep --where level=ERROR --range dur_us=10000.. events.zfm --stats

# Train a reusable profile so small chunks compress well from record one
zform train sample.ndjson -o github.zfp
zform pack --profile github.zfp live-chunk.ndjson -o live-chunk.zfm
```

</details>

<details>
<summary>Rust library</summary>

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

</details>

## Documentation

- [`DESIGN.md`](DESIGN.md): architecture, self-describing payloads, query engine, and host embedding.
- [`FORMAT.md`](FORMAT.md): binary wire specification for streams, column payloads, indexes, and profiles.
- [`PROGRAMS.md`](PROGRAMS.md): the 35-opcode lane VM instruction set and bounded recurrences.
- [`bench/README.md`](bench/README.md): benchmark runners, corpora, and ablations.

## Related Work

- **Format-aware compression:** OpenZL (Meta, 2024–2025) routes typed streams through codec graphs. Zstandard (Collet & Kucherawy, RFC 8878) is `zform`'s default entropy backend.
- **Log compression:** CLP (Rodrigues et al., *OSDI 2021*) and LogGrep (Wei et al., *EuroSys 2023*) split logs into templates and variables for compressed search; Drain (He et al., *ICWS 2017*) parses log templates online.
- **Semi-structured columns:** JSON Tiles (Durner et al., *SIGMOD 2021*; Zeng et al., *VLDB 2023*) extracts typed sub-columns from heterogeneous JSON documents.
- **Lightweight codecs:** FSST (Boncz, Neumann & Leis, *VLDB 2020*) for strings; Gorilla (Pelkonen et al., *VLDB 2015*), BtrBlocks (Kuschewski et al., *SIGMOD 2023*), and FastLanes (Afroozeh & Boncz, *VLDB 2023*) for delta and frame-of-reference encodings; Blosc byte-shuffle (Alted, 2010).

<details>
<summary>What's the Weissman score?</summary>

Following the [IEEE Spectrum formulation](https://spectrum.ieee.org/a-madefortv-compression-metric-moves-to-the-real-world), `W = α · (r / r̄) × (log T̄ / log T)` with `α = 1`, `T` in seconds, and `gzip -6` (`r̄ = 10.14x`, `T̄ = 13.40 s`) as baseline on the full 1,577.98 MB HDFS corpus:

- `zform` (`r = 89.14x`, `T = 16.30 s`): **`W = (89.14 / 10.14) × (log 13.40 / log 16.30) = 8.17`**

This exceeds the [award-winning score](https://en.wikipedia.org/wiki/Optimal_Tip-to-Tip_Efficiency) of 5.2, though it's a single result on a highly compressible log corpus- structure-aware encoding for 3D video is left to future work.

</details>

*The wire format is experimental and changes without notice; keep original copies of data you cannot repack. Licensed under the [Apache License, Version 2.0](LICENSE).*
