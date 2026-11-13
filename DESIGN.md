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

- A block needs its model and row count to decode and, in a file written with one, the file's
  dictionary. No block refers to another block's bytes.
- A block is stored as parts: the row structure, then one part per column, plus the tables
  several columns share (a dictionary, pooled free text). A part decompresses on its own once
  the parts of the same block it depends on are present.
- Reading any row of a block decompresses the whole of each part the query needs.

**Block size trades file size for read cost**, set by the writer in rows and bytes. A larger
block gives every column more context and a selective read more to fetch and decode. The stream
default is 65,536 rows or 8 MiB.

**A dictionary gives small blocks the context of the file.** It is the column values that two
or more blocks hold, each once, stored ahead of the blocks, and every part is compressed
against it. Columns, queries and decode speed are as without one. The price is fixed per file
instead of per block: whoever reads any block reads the whole dictionary first, in the round
trip that fetches the block's parts, and holds it expanded. The writer is handed the dictionary;
`DictionaryBuilder` chooses one within a byte budget in a pass over the records before they
are written.

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

One query keeps its decoded state in a session. Between queries a `File` keeps only its
dictionary, which a host that opens the file again hands back (`set_dictionary`), and the only
global state is a per-thread zstd context. `DecodeBudget` bounds decoding of untrusted bytes.

## Writing

The writer takes records in their final order. `cut()` ends a block early, and `finish()`
returns the index region. Data region bytes are emitted in order as blocks complete and never
revisited.

- **Memory** is a multiple of the block size and does not grow with the file. A dictionary
  adds its content and match tables, about four times its size; choosing one holds a block, a
  filter of up to 64 MiB and four budgets of candidate values.
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
- Training policy, where profiles are kept, the block size, the dictionary's budget, and
  keeping the dictionaries of files it opens often.

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
- Choosing a dictionary needs a profile and a second pass over the records, so the CLI, which
  streams, writes none. It is measured on one corpus, and not on files past the 16 million
  distinct values its filter holds.
- Concatenated files unpack, but their index regions are not merged, so the result cannot be
  queried.

## Evidence

All at zstd level 6 on one thread.

**Block size.** Files written by the current encoder:

| Block | GitHub Archive hour, 499 MB | HDFS logs, 256 MiB | Writer peak, with a profile |
|---|---:|---:|---:|
| 8 MiB | 32.8 MB | 6.69 MB | 32–37 MiB |
| 16 MiB | 29.4 MB | 6.63 MB | |
| 32 MiB | 27.2 MB | 6.61 MB | 75–112 MiB |
| 64 MiB | 25.9 MB | 6.35 MB | 137–215 MiB |

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

**Dictionary.** Two hours of that day (250,493 records, 999 MB) with a profile trained on a
16 MiB sample. 26.0 MB of values recur across the 8 MiB blocks, so the 32 MiB budget holds
them all:

| Blocks | Dictionary budget | Level | File | Dictionary in it | Encode | First read |
|---|---:|---:|---:|---:|---:|---:|
| 8 MiB | none | 6 | 64.9 MB | | 10.6 s | 1.2 ms |
| 8 MiB | 4 MiB | 6 | 57.9 MB | 1.0 MB | 12.7 s | 3.7 ms |
| 8 MiB | 8 MiB | 6 | 55.1 MB | 2.0 MB | 16.1 s | 5.8 ms |
| 8 MiB | 16 MiB | 6 | 52.8 MB | 3.2 MB | 18.9 s | 8.6 ms |
| 8 MiB | 32 MiB | 6 | 51.1 MB | 6.1 MB | 20.6 s | 15–24 ms |
| 8 MiB | 32 MiB | 9 | 49.9 MB | 6.1 MB | 23.8 s | 15–24 ms |
| 8 MiB | none | 9 | 64.4 MB | | 13.3 s | 1.2 ms |
| 128 MiB | none | 6 | 48.2 MB | | 9.8 s | |

First read is opening the file and reading one record of a small block; with the dictionary
already held it is 1.3 ms. With the full dictionary the data region is 48.8 MB against 48.0 MB
in 128 MiB blocks; the rest of the difference is the index, 2.3 MB against 0.2 MB. Choosing
the dictionary takes another 1.5 s, and the writer's peak rises by 94 MiB. Reading every
record back takes 0.89 to 0.94 s in each layout.

Values under 16 bytes are left out: from 8 bytes the file is 0.3 MB larger. Sorted content
makes it 0.2 MB smaller than most-shared-first. The match tables hold a position per
dictionary byte; at zstd's default size the blocks are 61 MB instead of 45 MB. The content is
stored at level 16: level 19 saves 0.2 MB of it and takes 2.7 s longer.

Layouts in which a block refers to other blocks' bytes were measured first and rejected. Free
text stored once per file with back references, pooled across 6 to 12 blocks, or held in a
paged table of long values saved 2–7% of a file, and reading one block then needed 3 to 21
times its own text from elsewhere. Sorted dictionaries shared by a frame's blocks reached
56.0 MB on the two hours and made selective reads 1.5–2.9x slower.

**Small blocks.** Against 65,536-row blocks, an HDFS file is 1.19x the size at 4,096 rows
per block, 2.0x at 512 and 4.9x at 64. A block names the decode programs in its model's
library and stores only the constants that differ; with the programs in every block the
figures were 1.5x, 2.9x and 7.3x.

**The index is paged by block.** The GitHub model has about 780 columns, and a block's
directory lists only those with a value in some row: 1.2 MB for the day's 1,141 blocks.
