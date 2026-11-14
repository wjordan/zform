# `zform` Benchmarks

`run.py` builds `zform` in release mode and compares it with `gzip`, `zstd` and `xz` on ratio, throughput and peak RSS, checking a SHA-256 roundtrip for every codec. It runs on whichever of these local corpora it is given: Loghub `HDFS.log` and a GitHub Archive hour, each cut to `--slice-mb` (default 8) at a line, and the Silesia corpus's `sao`, taken whole:

```bash
python3 bench/run.py \
  --hdfs /path/to/HDFS.log \
  --gharchive /path/to/2020-06-17-16.json.gz \
  --sao /path/to/silesia/sao
```

Every codec runs on one thread. Timings are the best of `--repeats` runs (default 3) and include process startup and file I/O. Peak RSS comes from GNU `time` and prints as `n/a` without it. `--markdown` appends the tables used in the top-level README.

## context mixing

`context_mixing.py` compares the normal record compressor with and without the optional payload
density pass. It checks deterministic encoding and a SHA-256 roundtrip for both variants:

```bash
cargo build --release -p zform-cli
python3 bench/context_mixing.py --repeats 3 --output results.json HDFS.log HPC.log Zookeeper.log
python3 bench/context_mixing.py --best --repeats 3 --output best.json HPC.log
```

The [recorded results](context-mixing-results.json) use 8 MiB samples, including the query index,
on a Ryzen 9 7900X. Encode times are medians of two process runs; decode times are single runs.
These are ordinary block-based `.zfm` files, separate from the whole-file BGL experiment.

| Corpus | Default bytes | With mixing | Smaller | Encode time factor | Decode time factor |
|---|---:|---:|---:|---:|---:|
| HDFS | 156,157 | 146,714 | 6.05% | 1.18× | 2.63× |
| HPC | 213,732 | 202,602 | 5.21% | 1.13× | 2.26× |
| ZooKeeper | 89,471 | 84,173 | 5.92% | 1.15× | 2.24× |
| HealthApp | 202,490 | 188,864 | 6.73% | 1.18× | 2.91× |
| CloudTrail | 637,596 | 628,883 | 1.37% | 1.08× | 1.38× |
| GitHub Archive | 578,430 | 576,167 | 0.39% | 1.02× | 1.17× |

Three BGL windows save 0.05–2.78%. With `--best`, measured additional savings are 2.48% on HPC,
0.66% on ZooKeeper, and 0.90% on CloudTrail. Those high-density comparisons use one run each.
The option remains off by default because selected context-coded streams decode more slowly.

## hdfs predictors

`hdfs_predictors.py` compares the optional relation pack with and without context mixing.
It checks deterministic encoding, SHA-256 roundtrips, and `grep` results. An optional earlier
binary verifies reader compatibility and that disabling the feature leaves archive bytes unchanged.

```bash
python3 bench/hdfs_predictors.py --repeats 3 --output results.json HDFS.log
cargo run --release -p zform-core --example hdfs_predict_probe -- HDFS.log
```

The [recorded results](hdfs-predictors-results.json) include three 8 MiB HDFS windows and BGL,
HPC, and ZooKeeper controls. Times are medians of three process runs on a Ryzen 9 7900X.
File sizes include the index. The head is the existing reference sample. Middle and tail
start at newline boundaries near offsets 128 MiB and 248 MiB in the local 256 MiB HDFS prefix.

| HDFS window | Default | HDFS predictors | Context mixing | Both | Extra saving over mixing |
|---|---:|---:|---:|---:|---:|
| Head | 156,157 | 150,092 | 146,714 | 141,242 | 3.73% |
| Middle | 160,013 | 154,407 | 150,797 | 145,381 | 3.59% |
| Tail | 301,574 | 301,574 | 289,181 | 289,181 | 0% |

With mixing enabled, the predictor added 5–7% encode time on the first two windows; decode
time was unchanged within measurement noise. Tail and non-HDFS control archives were byte-identical
with the predictor enabled or disabled. The partition field shrank from 5,673 to 198 bytes
in the head sample. The middle sample also selected the per-block size table.

On the entire 256 MiB prefix, the combined options reduced 5,770,436 bytes to 5,663,072 bytes
relative to context mixing alone: another 1.86%. This full-prefix validation used one run per
variant and checked exact roundtrips, existing-reader compatibility, and queries across blocks.

`hdfs_predict_probe` screens grouping, recent-block caches, and keyed deltas. Cache and delta
numbers omit the proposed stateful node and context wiring, so they are optimistic size estimates,
not complete codecs. Grouping models execute and verify their existing-payload graphs. None of
these block-reuse models gave a consistent improvement over the current codec across the three
windows; no new mutable-state decoder operation was added.

## hdfs block traces

HDFS.log is sorted by locale collation over the whole line: alphanumerics only, case-folded.
All 11,175,629 lines obey it, and the 2,118 ties are identical lines, so sorting decoded lines
by that key rebuilds the file and a coder may emit lines in any order. The BGL, HPC, HealthApp
and ZooKeeper samples are not sorted.

`hdfs_trace.py` uses that freedom and reports ideal code lengths from causal adaptive models.
It is an estimator, not a codec. It groups lines by block and orders blocks by their
`java.util.Random` index; allocations are consecutive `nextLong` calls. It codes each block's
trace against its own history and an inferred logging node per line. Datanode deletions form a
separate per-node stream.

```bash
python3 bench/hdfs_trace.py HDFS.log            # whole file
WIN=60000 python3 bench/hdfs_trace.py HDFS.log  # reset per-block and per-node state every 60k rows
RESID=1 python3 bench/hdfs_trace.py HDFS.log    # also size the residual streams under plain LZMA2
SL=1 python3 bench/hdfs_trace.py HDFS.log       # keyed previous-row predictors only
```

| Input | `--hdfs-predictors --context-mixing` | Trace model, whole input | 60k-row windows |
|---|---:|---:|---:|
| First 256 MiB (LF line ends) | 5,755,995 | 3,495,157 | 3,939,185 |
| HDFS.log | 32,709,019 | 15,192,577 | 21,053,965 |

Estimates omit the template text (about 5 KB). Most of the gain comes from relations the
column codecs cannot see:

- Thread ids work as per-node clocks. A PacketResponder thread is usually the receiving
  thread plus one; `Received block` reuses the PacketResponder thread. Other lines fall between
  the thread ids already decoded for the same node and logger class.
- Write-pipeline source ports from a local client follow the host's ephemeral port counter plus a
  fixed offset for each source and destination. Datanode-to-datanode ports still cost about 15 bits
  each and make up most of the 2.6 MB of ports; the kernel's port hash probably rekeys too
  often to learn those offsets.
- Each datanode deletes the smallest pending block ids, in signed order, first. That cuts deletion
  costs from about 2.2 MB to 1.2 MB. Windowing loses most of this: most deletions refer to an
  invalidation in an earlier window, and coding those blocks directly costs 4.1 MB.
- Second and third replicas cost about 7.7 bits each. They show no `/16` rack structure.

Context modelling is not what carries this. Writing each predictor's output as a residual stream,
one per model and template, and compressing each with raw LZMA2 nearly matches the adaptive
models on the 256 MiB prefix:

| Predictors | Adaptive, whole | Adaptive, windows | LZMA2 residuals, whole | LZMA2 residuals, windows |
|---|---:|---:|---:|---:|
| All | 3,495,157 | 3,939,185 | 3,596,817 | 4,159,820 |
| Keyed previous-row references only (`SL=1`) | 3,907,209 | 4,043,811 | 3,996,595 | 4,345,081 |

The `SL=1` predictors avoid per-node thread bracketing and port counters. They need only the
value of an earlier row with the same key (block, node, or source and destination), which
`ExtAlign`, `MatchIndex` and `IndexCells` already express. They still need the row reordering and
a coded node column. The 0.4 MB gap to the full predictors is all thread ids; keyed
port references did slightly better than the counter model.

## field partition search

The manual `partition_probe` tests whether fields shared across templates should remain in
one column. `adaptive-fast` screens for different numeric scales, prices strided samples of
4,096 occurrences, and merges groups with overlapping values when that reduces estimated cost.
It uses ordinary payload graphs and changes only the structural profile.

```bash
cargo build --release -p zform-core --example partition_probe
python3 bench/model_search.py --output results.json HealthApp.log HPC.log
```

[Recorded results](model-search-results.json) compare the current context-mixing encoder,
an unchanged fixed-profile control, and the prototype. The complete 23,529,930-byte HealthApp
file shrank from 520,781 to 420,947 bytes: 19.17% beyond context mixing. The fixed-profile
control produced 520,150 bytes, so the gain comes from partitioning, not profile reuse.
Three-run median encode time rose from 0.906 to 1.127 seconds; decode time was 0.225 versus
0.217 seconds. Every archive passed SHA-256 roundtrip and query checks.

The planner had combined counters, durations, and other unrelated values in `$int._`,
`$int._#1`, and `$int._#2`. Separating their template contexts while preserving useful
getter/setter sharing saved about 100 KB. Blindly splitting every numeric column made
HealthApp, HDFS, HPC, ZooKeeper, and BGL samples larger.

This remains a whole-input research prototype. It reads the input before selecting a fixed
profile, and its candidate estimates omit existing peer derivations and use approximate metadata
costs. Some controls regress. Integration needs a bounded training search and a comparison of
complete candidate block costs, including dependencies, before accepting a new partition.
The CLI's default planner has not changed.

`clock_probe` separately tries the existing combined microsecond clock pack in the ordinary
BGL planner. It saved 1.33–6.78% on the three BGL windows. `health_clock_probe` tests a fixed
header and combined clock; those changes were less effective than selective field partitioning.
`column_probe` prints physical columns and their template contexts for diagnosis.

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
