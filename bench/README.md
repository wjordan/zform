# `zform` Benchmarks

`run.py` builds `zform` in release mode and compares it with `gzip`, `zstd` and `xz` on ratio, throughput and peak RSS, checking a SHA-256 roundtrip for every codec. It runs on whichever of these local corpora it is given: Loghub `HDFS.log` and a GitHub Archive hour, each cut to `--slice-mb` (default 8) at a line, and the Silesia corpus's `sao`, taken whole:

```bash
python3 bench/run.py \
  --hdfs /path/to/HDFS.log \
  --gharchive /path/to/2020-06-17-16.json.gz \
  --sao /path/to/silesia/sao
```

Every codec runs on one thread. Timings are the best of `--repeats` runs (default 3) and include process startup and file I/O. Peak RSS comes from GNU `time` and prints as `n/a` without it. `--markdown` appends the tables used in the top-level README.

## Queries

`crates/zform/examples/qbench.rs` packs records into a data and an index file and runs single queries against them cold, reporting the bytes of each region read, round trips and CPU:

```bash
cargo build --release -p zform --example qbench
qbench=target/release/examples/qbench
$qbench pack day profile.zfp 6 8 < day.ndjson          # level 6, 8 MiB blocks
$qbench grep day clickhouse                            # count a substring
$qbench fetch day repo.name flutter/flutter            # records where a field equals a value
$qbench among day repo.name flutter/flutter            # the same, told which blocks hold them
$qbench newest day clickhouse created_at 20 1592438400 # newest 20 with a substring
$qbench top day type PullRequestEvent payload.pull_request.base.repo.language 20
$qbench filters day                                    # index sizes and filter fill
```

## CloudTrail

`cloudtrail.py` compares gzip -6, zstd -3/-19, xz -6, and the three zform
presets on local CloudTrail JSON, `.json.gz`, or single-JSON `.zip` files.
It checks SHA-256 roundtrips for every codec and writes sizes, timings, input
hashes, tool versions, and binary hash to `bench/results.json` (ignored).
Build the release binary before running it.

Public sources:

- [ArielCyber Cloud Log Sampling](https://github.com/ArielCyber/cloud_log_sampling):
  real CloudTrail events from a controlled, load-driven AWS application, with
  benign and attack activity. The files below contain 16,495 benign events and
  24,087 mixed events. The separate 4,216-event training split is omitted.
- [Summit Route flAWS](https://summitroute.com/blog/2020/10/09/public_dataset_of_cloudtrail_logs_from_flaws_cloud/):
  1,939,207 anonymized events from a small account, mostly attack activity.
  The range download below extracts only the first 100,000-event gzip member
  from the published tar archive; its hash verifies the archive layout.

```bash
mkdir -p bench/corpus/cloudtrail
curl -fL https://raw.githubusercontent.com/ArielCyber/cloud_log_sampling/18d3f248f4def592f4e86152c3fcb54506fbcbf6/TrainOn/base_a.zip -o bench/corpus/cloudtrail/base_a.zip
curl -fL https://raw.githubusercontent.com/ArielCyber/cloud_log_sampling/18d3f248f4def592f4e86152c3fcb54506fbcbf6/TestOn/labeled_cloudtrail_a.zip -o bench/corpus/cloudtrail/labeled_cloudtrail_a.zip
curl -fL --range 1024-19211796 https://summitroute.com/downloads/flaws_cloudtrail_logs.tar -o bench/corpus/cloudtrail/flaws_cloudtrail00.json.gz
echo '62dadeb453cefd0ac3a863003fa340b09375380daac3009855907234fa42845e  bench/corpus/cloudtrail/flaws_cloudtrail00.json.gz' | sha256sum -c -
cargo build --release --workspace
python3 bench/cloudtrail.py bench/corpus/cloudtrail/base_a.zip bench/corpus/cloudtrail/labeled_cloudtrail_a.zip bench/corpus/cloudtrail/flaws_cloudtrail00.json.gz
```

The default selects up to 8 MiB of events per source using seed 42, retains
source order, removes research labels (`should_sample`, `attack_name`), and
serializes compact NDJSON. All codecs receive identical bytes. Byte-exactness
applies to this NDJSON, not the source JSON formatting. Use `--layout native`
to measure the complete original JSON bytes, including formatting and labels,
or `--layout both` to measure both representations.
Each command has a 30-second timeout, configurable with `--timeout`.

Timings are single runs with process startup and file I/O included; use them
as a smoke benchmark. These anonymized, single-environment corpora do not
establish a compression ratio for arbitrary production accounts. Parquet
needs a separate comparison with schema and fidelity requirements specified.

To compare another record compressor, pass `--compare-bin /path/to/compressor`.
The adapter expects `pack --level N` and `unpack`, and measures levels 3, 6,
and 9 with its default training and chunk sizes. It raises `--max-record-bytes`
to 128 MiB on both commands to accommodate single-line JSON batches. The report
includes its binary hash and Git revision. Use `--only-compare` to skip codecs
already measured, and `--output` to retain the previous report. Match inputs by
`source`, `variant`, and `raw_sha256` when combining reports.
