# `zform` benchmarks

`run.py` compares `zform` with `gzip`, `zstd` and `xz` on complete inputs. It checks each roundtrip with SHA-256 and prints the input sizes and hashes. The README comparison uses the full Loghub `HDFS.log`, the GitHub Archive hour `2020-06-17-16`, and Silesia's `sao`.

```bash
python3 bench/run.py \
  --hdfs /path/to/HDFS.log \
  --gharchive /path/to/2020-06-17-16.json.gz \
  --sao /path/to/silesia/sao \
  --repeats 1 --markdown
```

Every codec runs on one thread. Timings include process startup and file I/O. The runner reports the fastest time and highest RSS across repeats. It measures RSS with GNU `time` and prints `n/a` when it is unavailable. `--slice-mb` limits the log inputs for quick runs.

## automated results

The [full-corpus benchmark workflow](https://github.com/wjordan/zform/actions/workflows/bench.yml) runs on changes to the compressor, weekly, and on demand. Each run publishes a summary and a report artifact with per-corpus timings, memory, archive sizes, input hashes and tool versions.

The first successful run records the README size baseline. Subsequent runs update the comparison and preset tables when an archive size moves at least 1% from the published baseline.

```bash
bash bench/fetch_corpora.sh bench/corpus
```

## context mixing

`context_mixing.py` compares the ordinary and high-density presets with and without context mixing. It checks deterministic encoding and SHA-256 roundtrips for both variants.

```bash
cargo build --release -p zform-cli
python3 bench/context_mixing.py --timeout 120 --output results.json /path/to/HDFS.log
python3 bench/context_mixing.py --best --timeout 120 --output best.json /path/to/events.ndjson
```

The [earlier context-mixing measurements](context-mixing-results.json) cover sample windows and use an earlier writer. Context mixing is opt-in because it can slow down decoding.

## hdfs predictors

`hdfs_predictors.py` measures all four combinations of HDFS predictors and context mixing. It checks deterministic encoding, SHA-256 roundtrips and `grep` results. `--baseline-bin` also checks reader compatibility and archive equality with a binary from a wire-compatible revision.

```bash
python3 bench/hdfs_predictors.py --timeout 120 --output results.json /path/to/HDFS.log
cargo run --release -p zform-core --example hdfs_predict_probe -- /path/to/HDFS.log
```

The [earlier HDFS predictor measurements](hdfs-predictors-results.json) cover sample windows and a 256 MiB prefix. The current whole-file result is in the [main README](../README.md#benchmarks).

`hdfs_predict_probe` screens grouping, recent-block caches, and keyed deltas. Its cache and delta estimates omit the proposed stateful node and context wiring. Grouping models execute and verify their existing-payload graphs.

## RankedSum ablation

`ranked_sum_ablation.py` builds a temporary copy of revision `fc5e945` and compares
deletion-clock representations in complete indexed archives. The runner leaves the working
tree unchanged; its historical source is pinned so the original experiments stay reproducible. Every mode
keeps the same writer-side datanode inference for a given record order. `off` disables
deletion prediction while retaining the other HDFS predictors.

```bash
python3 bench/ranked_sum_ablation.py build --work /tmp/ranked-ablation
python3 bench/ranked_sum_ablation.py run --work /tmp/ranked-ablation /path/to/HDFS.log
```

The build runs the payload tests and HDFS fixtures in every mode, including missing table
keys, invalid times, sparse rows, negative values and stable sorting ties. Each test process
has a one-second timeout. The manual benchmark checks exact SHA-256 roundtrips and `grep`
results, and repeated encodes must produce identical bytes. `--modes`, `--no-mixing`,
`--level`, `--repeats`, and `--timeout` restrict a run.

On the whole 1,577,982,906-byte HDFS file, at level 6 with HDFS predictors, context mixing,
one encoding thread, and the query index included:

| Mode | Decoder construction for deletion clocks | Archive bytes | Change from RankedSum |
|---|---|---:|---:|
| `ranked` | Original `RankedSum` | 17,599,245 | — |
| `off` | No deletion prediction | 18,235,313 | +3.61% |
| `keyed-row` | Explicit node IDs, `KeyedSum`, stored row order | 18,390,759 | +4.50% |
| `keyed-sorted` | Explicit node IDs, `KeyedSum`, explicit inverse permutation | 18,441,470 | +4.79% |
| `scan-grouped` | Group by inferred node, `ScanAdd`, explicit inverse permutation | 18,417,705 | +4.65% |
| `keyed-argsort` | Explicit node IDs, `KeyedSum`, derived permutation | 18,366,984 | +4.36% |
| `replica-row` | Replica-table lookup, `KeyedSum`, stored row order | 18,343,134 | +4.23% |
| `replica-argsort` | Replica-table lookup, `KeyedSum`, derived permutation | 17,725,378 | +0.72% |

`replica-argsort` keeps 80.2% of the savings over `off`. Its time-column payload grows
from 2,831,332 to 2,957,269 bytes. Both predictors are selected in 62 column payloads.
The other replacements lose more to key or permutation storage, or to worse clock residuals,
than they save. These are results after ordinary candidate selection; a rejected relation
falls back to another codec.

Three repeated whole-file runs give median encode/decode times of 18.058/5.524 seconds
for `ranked` and 18.126/5.563 seconds for `replica-argsort`: differences below 1%, within
the range of run-to-run decode variation. Encoded bytes were identical across repeats.
The other whole-file variants were screened once; their timings are not used to rank speed.

The argsort modes introduce one experimental graph node, `SortIndices(values)`, returning
row indices ordered by signed integer value, with ties in original row order. The replacement
uses existing operations to look up the selected replica, assign its value ID, gather keys
and deltas in block-ID order, apply `KeyedSum`, and gather the results back. Sorting the first
permutation's indices produces its inverse. Candidate ranking and selection without replacement
remain entirely in the writer.

Only the temporary reader understands `SortIndices` (experimental tag 30). All other modes
use existing operations. This is an ablation, not a production format change or a decision
to retain that tag.

Three 32 MiB windows cover the head, middle and tail of the full input. The head selects no
deletion predictor. On the middle and tail, `replica-argsort` costs 481 and 1,004 more bytes
than `ranked`; `off` is actually 927 bytes smaller than `ranked` on the tail. Each window
builds its own file table, so it does not reproduce the whole file's available replica history.

Recorded measurements and validation are in [ranked-sum-results.json](ranked-sum-results.json).

### Further simplification

The follow-up compares simpler state initialization, smaller slot streams, a fused ordered
sum, and a change to stored row order. Sizes use the same whole input and flags as above:

| Mode | Archive bytes | Extra bytes over RankedSum |
|---|---:|---:|
| `replica-argsort` (previous result) | 17,725,378 | 126,133 |
| `replica-split` | 17,704,190 | 104,945 |
| `replica-ordered` | 17,722,305 | 123,060 |
| `replica-ordered-split` | 17,701,119 | 101,874 |
| `replica-ordered-zero` | 17,703,621 | 104,376 |
| `replica-physical` | 21,472,105 | 3,872,860 |

`replica-split` codes replica slots per template through the existing `Interleave` operation.
It removes repeated information already available from record variants; it adds no decoder rule.

`replica-ordered` replaces the sort/gather/sum/inverse/gather composition with one experimental
`OrderedSum(keys, deltas, order)` node. It accumulates in ascending order, breaks ties by row
position, and returns results in original row positions. Key 0 passes its delta through.
`replica-ordered-split` combines that operation with template-separated slot streams.

`replica-ordered-zero` starts every key at zero, removing the dependency on the last sum emitted
for another key. The writer stores the first value of each key as an absolute delta. It retains
83.6% of the deletion predictor's savings and costs only 2,502 bytes more than the shared-seed
version, or 0.014% of that archive. This is the simplest state model tested that keeps most of
the savings. The prototype carries a seed-mode byte solely to compare the two definitions;
a production choice of zero-seeded sums would not need that option.

The temporary reader also computes `SortIndices(SortIndices(x))` by a linear permutation
inversion. The first node already guarantees a permutation, so the second sort is unnecessary.
This changes neither the graph nor its bytes. An ordered sum needs neither explicit permutation
node and sorts only rows with a nonzero key.

`replica-physical` sorts deletion rows by block ID within each block before encoding and uses
the existing keyed sum. It needs no new decoder operation, but the two block-ID columns grow
by about 3.71 MB relative to `replica-split`. Reordering one column's useful neighbours harms
the others. This mode also changes the writer's samples and stored layout, so it is a broader
experiment than the fixed-order graph comparisons.

On the middle and tail windows, `replica-ordered-zero` writes 562,017 and 385,048 bytes,
respectively: 229 and 197 bytes larger than `RankedSum`. No result establishes a lower bound
on compression or complexity; the tradeoff measured here favors a plain ordered per-key sum
and writer-side slot coding over additional domain rules in the reader.

The added modes are available through the same runner. The temporary reader assigns tag 31
to `OrderedSum`; that experimental tag is not part of the production format. Follow-up measurements are in
[ranked-sum-simpler-results.json](ranked-sum-simpler-results.json).

### Production KeyedSum extension

The writer now uses existing kind 25 with optional ordering and zero-seed options. It emits
explicit replica slots, table lookups, `CellIds`, and `KeyedSum`; neither experimental node
was added. `RankedSum` remains a compatibility decoder. The final graph reuses its alignment
input and removes the now-unused deletion-selector table.

| Indexed archive | Bytes |
|---|---:|
| Original `RankedSum` | 17,599,245 |
| Deletion prediction disabled | 18,235,313 |
| Ordered, zero-seeded `KeyedSum` | 17,703,055 |

The simpler production form costs 103,810 bytes (0.590%) over `RankedSum`, retaining 83.7% of
its deletion-clock savings. Two full encodes produced identical bytes, with median pack/unpack
times of 18.167/5.476 seconds. Head/middle/tail 32 MiB windows measured 382,990 / 561,990 /
385,042 bytes. Every archive passed SHA256 roundtrip and indexed `grep` checks.

The new reader also decoded the old full-file `RankedSum` archive exactly. Old readers reject
new sum options through a graph capability flag. All 21 non-HDFS regression inputs retained
identical archives to `fc5e945`, with HDFS predictors both disabled and enabled.
[Production results](keyed-sum-results.json) include hashes and timing samples; the
[node audit](../DESIGN.md#payload-node-scope) records the remaining node boundaries.

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
WIN=60000 KEEP=del python3 bench/hdfs_trace.py HDFS.log  # carry state across windows: blk,node,del,ent
WIN=60000 WINB=1 python3 bench/hdfs_trace.py HDFS.log  # windows of whole block lifecycles
```

| Input | `--hdfs-predictors --context-mixing`, file order | The same, rows by block | Rows by block, replica table | Per-node clocks and counters | Trace model, whole input | 60k-row windows |
|---|---:|---:|---:|---:|---:|---:|
| First 256 MiB (LF line ends) | 5,755,995 | 5,305,515 | 4,868,361 | 3,925,739 | 3,495,157 | 3,939,185 |
| HDFS.log | 32,709,019 | 25,283,398 | 23,470,438 | 17,599,245 | 15,192,577 | 21,053,965 |

"Rows by block" is what `--hdfs-predictors` now writes when the input is collation-sorted
(`FLAG_SORTED_ROWS`). Each block stores rows grouped by block id: write-pipeline rows by
thread, then the block's other rows. Groups follow `java.util.Random` order. That cut `$blk_id`
on the prefix from 1,128 KB to 130 KB, below the trace model's 166 KB. The other rows' times
and threads lost about 0.7 MB of that back.

"Replica table" adds the file table `--hdfs-predictors` builds for a file of over a block: each
block id with its first three datanodes, from the `addStoredBlock` lines, in generator order.
Every datanode column keyed by block id (stored, received, served, invalidated) codes its value
as which replica it is. On HDFS.log the table costs 1.72 MB and those five columns fall from
4.9 MB to 1.3 MB. Served-to clients are a replica only half the time and stay as they are.

"Per-node clocks and counters" is the current writer. Write times step on a start clock and a
finish clock, thread ids on a counter per datanode, with the table's replicas and the write rows
in pipeline order; deletion times step on a clock per datanode, taken in block-id order. The
table lets it carry per-node state the windowed trace model resets, so it lands under that
estimate.

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

The `SL=1` predictors avoid per-node thread bracketing and port counters, but they still need
the hidden logging node. Without it, keyed deltas do not pay. On the reordered prefix, threads
keyed by the Receiving line's node cost 1.28 MB, against 1.22 MB for the current coding. Ports
keyed by source IP cost 625 KB, against 602 KB raw: the line prints the destination IP as the
source. The 0.4 MB gap to the full predictors is all thread ids.

Across 60k-row windows, the state worth carrying is the pending deletes. Carrying them gives
18,432,367 B; carrying everything else gives nothing (21,070,123). The pending set at a
window boundary averages 23.5k entries (max 280k), so snapshotting it per block costs more than
it saves. Windows of whole block lifecycles, blocks in generator order, give 16,034,397 B at
60k rows and 15,461,397 at 240k. Because the file is collation-sorted, such blocks are sorted
runs that a reader merges, but a point in time then falls in a median of 30 of them. zform
gains nothing from either: 32 to 512 MiB blocks give 25.0 to 28.9 MB, lifecycle blocks
25,410,777 B.

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

For `qbench pack`, use `wall_s` to compare ingest times. Its `cpu_s` measures only the calling thread, while the file writer can encode blocks on worker threads.

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
python3 bench/cloudtrail.py --layout native bench/corpus/cloudtrail/base_a.zip bench/corpus/cloudtrail/labeled_cloudtrail_a.zip bench/corpus/cloudtrail/flaws_cloudtrail00.json.gz
```

`--layout native` measures the complete JSON bytes, including formatting and labels.
`--layout ndjson` samples events with seed 42 up to `--sample-mib`, preserves their order,
and removes the research labels `should_sample` and `attack_name`. All codecs receive the
same bytes and must restore them exactly. `--layout both` measures both representations.
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
