# `zform` design: queryable files and embedding

The wire layout, including column graphs, is in [`FORMAT.md`](FORMAT.md); the instruction set
graphs compute with is in [`PROGRAMS.md`](PROGRAMS.md). Benchmark commands are in [`bench/README.md`](bench/README.md).

## scope

`zform` stores records, which are byte strings, and returns them byte-exact. A file answers three
questions without being unpacked: which rows match a predicate, what those records are, and what
named fields of those records hold.

Its peers are compressed log stores (CLP, LogGrep) and seekable zstd. It has no schema, returns
bytes rather than typed vectors, and has no aggregation kernels, so against Parquet or Vortex the
comparison that applies is file size and the bytes and round trips of a selective read.

## layers

| Layer | Owns | API |
|---|---|---|
| Model | how records split into columns and how columns relate | `train`, `Profile` |
| Payload | how column values are reconstructed from generic operations | `PayloadGraph`, `CoreDecoder` |
| Block | a run of records as column parts | `zform::block` |
| File | a stream of blocks and the index over them | `zform::file` |
| Host | where bytes live, and all data that is not record bytes | the host's code |

A host is the `zform` CLI reading a `.zfm` file, or a database that keeps a zform file inside its
own object next to its own columns and indexes.

## self-describing payloads

The structural model routes records into columns and describes how to assemble them again.
Each column's payload carries the graph that reconstructs its values. Graph nodes produce
byte buffers, integer lanes, or byte-string cells; the root supplies the column's output.
The payload's row count counts field occurrences, which can differ from the block's record count.

`CodecPack` and `PackRegistry` belong to the writer. Packs recognize value patterns and emit
candidate graphs; the writer compares their encoded costs, including code sharing. A pack
for a timestamp can emit integer residuals, arithmetic, and a program that formats bytes.
`CoreDecoder` executes those primitives without knowing the timestamp format or the pack.
A new pack using existing nodes and instructions needs no reader changes. Adding a node or
instruction changes the decoder's contract; the format is experimental and is not frozen.

Each model carries a library of shared programs, recurrences, and byte tables. Column graphs
can name library entries and override a program's constants for the block, so code does not
repeat with every column or block. Inline code handles programs absent from the library.
The library travels with the model in streams and profiles; it needs no external registry.

A column header declares its dependencies on the block's prose pool, one shared dictionary,
and peer columns. The index records those dependencies so a selective read fetches the parts
needed to evaluate the graph. Shared code in the model and shared data in the block have
separate lifetimes. Neither creates a dependency on another block's column data.

`CoreDecoder::decode_col` preserves constant, dictionary, pooled, and quoted columns as lazy
views when the graph root permits it. This avoids copying repeated cell bytes and lets a
predicate run once per dictionary value. `decode_cells` materializes byte strings;
`decode_ints` returns native integer lanes for numeric roots. The `zform::column` and
`zform::numeric` APIs use the same payload format outside record files.
[`FORMAT.md` §4](FORMAT.md#4-model-and-column-payloads) specifies the wire representation.

## file layout

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
trip that fetches the block's parts, and holds it expanded. The writer is handed the dictionary.
`DictionaryBuilder` needs a profile and chooses values within a byte budget in a pass before
writing. The CLI streams its input and writes no file dictionary.

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

Concatenated streams unpack in order. Their index regions are separate, so `grep` and
`select` require a file with a single-run index.

## reading

The reader does no IO. It is given a source that returns a byte range or reports it absent. An
operation that lacks bytes returns every range it found missing, the host fetches them, and the
operation runs again. Caching, request coalescing and concurrency belong to the host.

A cold selective query fetches the root, then directory and filter pages, then parts.
Cached or coalesced ranges can reduce round trips; column dependencies can require more.
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

A block containing rows outside the JSON tree, such as templates or verbatim records, is read
whole for field predicates. Predicates on whole objects or arrays rebuild the tested rows.
A substring or word in a shape's constant text also requires rebuilding rows of that shape.
JSON records with leading whitespace are stored verbatim and are rebuilt for queries.

One query keeps its decoded column state in a session. A `File` shares its models, query-plan
cache, and loaded dictionary across sessions. A host opening the file again can supply the
dictionary through `set_dictionary`. Per-thread caches retain compiled payload programs and
zstd contexts. `DecodeBudget` bounds decoding of untrusted bytes.

## writing

The writer takes records in their final order. `cut()` ends a block early, and `finish()`
returns the index region. Data region bytes are emitted in order as blocks complete and never
revisited.

- **Memory** is a multiple of the block size and does not grow with the file. A dictionary
  adds its content and match tables, about four times its size; choosing one holds a block, a
  filter of up to 64 MiB and four budgets of candidate values. Training adds sample and
  model-building memory. `TrainOptions` controls sampling, with defaults of 16,384 rows and 16 MiB.
- **Models** come from a `Profile` or from training on the first records, and are stored in the
  data region. Every block reports a `Fit`: the rows and bytes its model could not place. When
  to retrain is the host's policy.
- **Determinism.** The same records, options, model and library version give the same bytes.
- **Threads.** None. A writer runs on the calling thread.

## what the host owns

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
