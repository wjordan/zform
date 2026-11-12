# `zform` design: queryable files and embedding

The wire layout is in [`FORMAT.md`](FORMAT.md). [Status](#status) lists what is built and what
is not, and [Evidence](#evidence) the measurements the decisions rest on.

## Scope

`zform` stores records, which are byte strings, and returns them byte-exact. A file answers three
questions without being unpacked: which rows match a predicate, what those records are, and what
named fields of those records hold.

Its peers are compressed log stores (CLP, LogGrep) and seekable zstd. It has no schema, returns
bytes rather than typed vectors, and has no aggregation kernels, so against Parquet or Vortex the
comparison that applies is file size and the bytes and round trips of a selective read.

## Layers

| Layer | Owns | API |
|---|---|---|
| Model | how records split into columns and how columns relate | `train`, `Profile` |
| Block | a run of records as column parts | `zform::block` |
| File | a stream of blocks and the index over them | `zform::file` |
| Host | where bytes live, and all data that is not record bytes | the host's code |

A host is the `zform` CLI reading a `.zfm` file, or a database that keeps a zform file inside its
own object next to its own columns and indexes.

## File layout

A file is two byte regions. Offsets inside each are relative to its start, so a host can place
the regions anywhere and pass their bases to the reader.

**The data region is a stream of self-contained blocks.** It decodes front to back with nothing
else, and two data regions concatenated are one valid stream.

- A block needs only its model and row count to decode. Nothing is shared between blocks: no
  dictionary, no text, no reference to another block's values.
- A block is stored as parts: the row structure, then one part per column, plus the tables
  several columns share (a dictionary, pooled free text). A part decompresses on its own once
  the parts of the same block it depends on are present.
- Reading any row of a block decompresses the whole of each part the query needs.

**Block size is the one knob between file size and read cost**, set by the writer in rows and
bytes. A larger block gives every column more context and a selective read more to fetch and
decode. The stream default is 65,536 rows or 8 MiB.

**The index region is derived data.** Scanning the data region rebuilds it.

- The **root** holds every model and each block's row count, byte range, model and page
  lengths. It grows with the number of blocks, never with blocks times columns, and it sits
  last so that a reader holding the tail of the file has it.
- **Directory pages**, one per block, hold the block's part ranges and, for each column with a
  value in some row, its min/max, the letters its cells contain and whether every row has one.
  They sit together before the root, and every query reads the ones for the blocks it may
  touch.
- **Filter pages**, one per block, hold a bit filter over the letter runs of its records, case
  kept: each run of 3 to 7 letters, and each 8 letters of a longer run. A needle prunes a block
  when one of its own runs is missing. A query fetches a block's filter only when the filter is
  smaller than the parts it might save.

Rows are numbered from 0 across the file. The row number is the only identifier a host shares
with `zform`. A host with its own row ranges (one per stream, say) maps them onto blocks by row
number. The writer accepts extra cuts, but small blocks are expensive and a host should let
short runs share a block.

## Reading

The reader does no IO. It is given a source that returns a byte range or reports it absent. An
operation that lacks bytes returns every range it found missing, the host fetches them, and the
operation runs again. Caching, request coalescing and concurrency belong to the host.

A cold selective query costs three round trips: the root; directory and filter pages; parts.
Fetching whole blocks needs no pages, since the root has each block's byte range.

Operations take and return row sets:

| Operation | Result |
|---|---|
| `paths()` | The field paths the file's models define. |
| `matches(pred, within)` | Rows of `within` that satisfy `pred`. No record is built, so a count is the set's size. |
| `records(rows)` | Those records, byte-exact, in row order. |
| `fields(paths, rows)` | Those fields of those rows. |
| `distinct(path, within)` | Per block, each value of a column with the rows holding it; a dictionary-coded column yields each value once. |

A host combines row sets from its own sources (time ranges, a word index, deletions) with these
by passing them as `within`. A test `zform` has no predicate for runs on the host: over
`distinct` for one field, such as a regular expression or a score per value, or over `records`
for the whole line.

A predicate is a tree of and/or/not over substring and whole-word tests on the record and
equality, set membership, substring, integer range, time range and presence tests on a field.
Substring and word tests prune blocks by their filter and columns by their letters, and read
only the columns left. A needle with no run of 8 letters and no run of 3 to 7 letters bounded
by non-letters inside it is not prunable by filter and scans; digits and punctuation carry no
tokens. An equality test on a field prunes the same way, so a value made of common words reads
its column in most blocks; a host whose own index chose the rows opens the session
`without_filters`. A field every row of a block holds once is tested as a column, with no walk
of the rows' structure.

A field value is one of: the field's bytes exactly as they appear in the record, quotes and
escapes included; absent; or unparsed, when the record was stored verbatim and the host has to
look at the record itself. A path that occurs several times in a record, as array elements do,
yields its values in order.

One query keeps its decoded state in a session. Nothing is kept between queries, and the only
global state is a per-thread zstd context. `DecodeBudget` bounds decoding of untrusted bytes.

## Writing

The writer takes records in their final order. `cut()` ends a block early, and `finish()`
returns the index region. Data region bytes are emitted in order as blocks complete and never
revisited.

- **Memory** is a multiple of the block size and does not grow with the file.
- **Models** come from a `Profile` or from training on the first records, and are stored in the
  data region. Every block reports a `Fit`: the rows and bytes its model could not place. When
  to retrain is the host's policy.
- **Determinism.** The same records, options, model and library version give the same bytes.
- **Threads.** None. A writer runs on the calling thread.

## What the host owns

- Storing, fetching and caching the two regions.
- Data per row that is not record bytes (timestamps, stream identity, labels), in its own
  columns, joined by row number.
- Indexes over data it stores, indexes that span files, and any index whose rules are a product
  decision, such as a word index with its own tokenizer. It builds these from the records and
  row numbers it already holds before it calls the writer.
- Training policy, where profiles are kept, and the block size.

`zform` indexes only what it stores: the columns inside a record. The encoder has those values
in hand, and their identifiers mean nothing outside the file.

## Status

Built: the block as parts, the index region, `zform::file` (writer, row sets, `matches`,
`records`, `fields`, `distinct`) and the CLI's `select` and `grep` on it. The data region is
still framed as `ZFM1`.

Known limits:

- A block holding rows outside the JSON tree (templates, verbatim rows) does not prune field
  predicates and is read whole for them.
- A predicate on a field whose value is a whole object or array rebuilds the rows it tests.
- A substring or word that also occurs in the constant text of a record shape rebuilds the
  rows of that shape.
- A profile trained before the shape rule for `null` objects and empty arrays still decodes
  what it wrote, but fits new data worse than a retrained one (10% larger on GitHub events).
- JSON records with leading whitespace are stored verbatim, so every query on them rebuilds
  rows.
- The trainer has no sample cap; it is the writer's memory peak when no profile is given.
- Concatenated files unpack, but their index regions are not merged, so the result cannot be
  queried.

## Evidence

All at zstd level 6 on one thread.

**Block size.** Files written by the current encoder:

| Block | GitHub Archive hour, 499 MB | HDFS logs, 256 MiB | Writer peak, with a profile |
|---|---:|---:|---:|
| 8 MiB | 31.8 MB | 6.49 MB | 32–37 MiB |
| 16 MiB | 28.6 MB | 6.49 MB | |
| 32 MiB | 26.6 MB | 6.56 MB | 75–112 MiB |
| 64 MiB | 25.3 MB | 6.27 MB | 137–215 MiB |

A whole block decodes at 0.6–1.0 GB/s on the GitHub hour and about 250 MB/s on HDFS, which
bounds a selective read at 8–32 ms for an 8 MiB block and 65–250 ms for a 64 MiB one. Without a
profile the trainer's sample adds about 150 MiB to the writer.

**Selective reads.** A GitHub Archive day (2,500,841 records, 9.56 GB) grouped by event type,
in 1,141 blocks of 8 MiB: 508.4 MB of data and 15.5 MB of index (14.3 MB of it filters), packed
with a profile in 116 CPU seconds at a 56 MiB peak. Cold, from `examples/qbench.rs`:

| Query | Data read | Index read | Round trips | CPU |
|---|---:|---:|---:|---:|
| Count a 10-letter substring with 324 matches | 82.1 MB | 15.5 MB | 3 | 0.31 s |
| Fetch the 962 records of one repository | 205.9 MB | 7.1 MB | 5 | 0.93 s |
| The same, given the 320 blocks that hold them | 182.6 MB | 1.9 MB | 5 | 0.76 s |
| The 20 newest records with that substring | 9.5 MB | 3.0 MB | 7 | 0.09 s |
| Top values of a depth-5 path over 229,000 records | 1.7 MB | 1.2 MB | 4 | 0.09 s |

The filter passes 208 blocks for that substring. Simulated over the same day, it passes 2% of
the blocks that lack it; with one bit per token 11%, and at 64 Kbit 20%. The repository's name
is two common words, so no filter prunes it: the fetch reads its name column in 1,097 blocks.

Larger blocks on the same day, files written by an earlier encoder: 16 MiB is 6% smaller and
reads 15% more for the 962 records, 32 MiB 11% and 32%, 64 MiB 16% and 55%.

**Sharing between blocks was measured and rejected.** Free text is 82.5 MB of the GitHub hour
and most of the gap between block sizes, so three ways of widening its scope at 8 MiB blocks
were computed inside the encoder:

| Text layout at 8 MiB blocks | Text size | A block's text also needs |
|---|---:|---|
| Pooled per block (today) | 17.3 MB | nothing |
| Each value stored once per file, later blocks refer back | 14.7 MB | parts of 26 earlier blocks on average, 21x its own bytes |
| The same, within runs of 8 blocks | 16.4 MB | 2.9x its own bytes |
| Text of 6 to 12 blocks pooled into one segment | 14.2–15.0 MB | a 1.5–2.8 MB segment |
| Pooled per 64 MiB block, for reference | 13.5 MB | nothing |

None reaches what a 64 MiB block gets with no mechanism at all, and each makes a block depend
on bytes outside it. Raising the block size moves along the same trade with one option.

A file-wide table of repeated values of 256 bytes or more, in pages of 64 KiB to 1 MiB, was
simulated on two hours of that day at 8 MiB blocks. Chosen in a first pass over the input it
saves 6–7% of the file, and reading a whole block then needs 3–7x its own text in table pages;
16 MiB blocks save 6% with no table. Chosen on a value's second appearance it saves under 2%.

**Small blocks.** Against 65,536-row blocks, an HDFS file is 1.15x the size at 4,096 rows
per block, 1.8x at 512 and 4.2x at 64.

**The index is paged by block.** The GitHub model has about 780 columns, and a block's
directory lists only those with a value in some row: 1.2 MB for the day's 1,141 blocks.
