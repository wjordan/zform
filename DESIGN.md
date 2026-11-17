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

Nothing depends on the format yet, so it changes in place with no compatibility path.

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

A block of 16,384 records or more is searched in two steps. The writer runs the whole search
on 4,096 records from the block's middle, then codes each column with the candidate that won
there, and with the runner-up where that came within a quarter. The candidate also repeats the
sample's choices between integer codings (as they are, by difference, by recency) without
building the alternatives. A choice more values could turn is made again: recency that lost
only to its misses, a stream too short to judge. A column gets the whole search when the
sample held under 512 of its cells, or when its winner costs a cell a quarter more on the
block than on the sample; keyed lookups need 1,024 cells, so a smaller sample does not rule
them out. On HDFS this removes 30% of the encoding work for 0.001% more bytes.

Each model carries a library of shared programs, recurrences, and byte tables. Column graphs
can name library entries and override a program's constants for the block, so code does not
repeat with every column or block. Inline code handles programs absent from the library.
The library travels with the model in streams and profiles; it needs no external registry.

A column header declares its dependencies on the block's prose pool, one shared dictionary,
and peer columns. The index records those dependencies so a selective read fetches the parts
needed to evaluate the graph. Shared code in the model and shared data in the block have
separate lifetimes. Neither creates a dependency on another block's column data; a file's tables,
below, are the one input from outside the block.

`CoreDecoder::decode_col` preserves constant, dictionary, pooled, and quoted columns as lazy
views when the graph root permits it. This avoids copying repeated cell bytes and lets a
predicate run once per dictionary value. `decode_cells` materializes byte strings;
`decode_ints` returns native integer lanes for numeric roots. The `zform::column` and
`zform::numeric` APIs use the same payload format outside record files.
[`FORMAT.md` §4](FORMAT.md#4-model-and-column-payloads) specifies the wire representation.

### context mixing

`zform pack --context-mixing input.log -o output.zfm` enables an optional density pass.
Library writers use `PackRegistry::default().with_context_mixing()` in their pack options.
Default and fast encoding keep their existing choices.

The planner first chooses the column graphs, shared dictionaries, and text pools. The density
pass then tries context mixing on integer transports in those selected graphs. Trying it inside
every pack candidate spent most of its time on graphs the planner discarded. Row-structure
streams also participate. Programs, peer dependencies, dictionary views, and block indexes do
not change; a selective read still fetches only its declared parts.

This generalizes the experiment's prediction of dictionary IDs, flags, and small residuals.
Writers emit kind `10`, which codes a value a nibble at a time: two 16-symbol models, one on
the nibble prefix and one hashed from the preceding pair, are mixed and range-coded. That is
about two coding steps per value instead of kind `9`'s one per bit. It costs a third of kind
`9`'s instructions in both directions, and its streams come out slightly smaller. The transport
has no log-format names, hardware topology, field parser, or state shared between payloads.
Each stream uses at most 1.5 MiB of model tables and resets at its boundary.

Candidates need at least 512 values and an offset width of 1–16 bits. A prefix probe rejects
clear losses; a complete candidate must save at least 32 bytes and approximately 2% of its
integer stream, including framing. Constants, wide values, and streams already below 128 bytes
skip the trial. Timestamps, numeric reconstruction, and byte formatting remain ordinary payload
programs. The latency-oriented `add_ints_fast` path does not run this search.

Context mixing trades decode speed for density, so it is opt-in even with `--best`.
Readers implement two bounded integer transports: kind `10`, and kind `9` for older files.
Lane programs gain no mutable hash tables or arbitrary loops. Older readers cannot decode a
payload that selects either kind.
Cross-field contexts could be a later extension with explicit graph dependencies, but the
current transport uses only its own preceding values.

### predictive relationships

`zform pack --hdfs-predictors --context-mixing HDFS.log -o hdfs.zfm` enables HDFS relationships
before the density pass. Library writers use `PackRegistry::with_hdfs_predictors()` or register
their own `RelationPack`. A relation pack sees a column, its peers, and the record layout;
it returns a self-describing payload graph. Single-column models still use `CodecPack`.

`HdfsRelations` recognizes task/partition paths and size fields from template literals.
It predicts a partition number from the task number, preserving zero padding and the filename
hyphen. `ExtPeer`, `ParseAsciiInt`, a formatting program, and `PatchCells` reconstruct the bytes.
For sizes, it tries a table holding the first attribute observed for each block ID. `ExtAlign`
and `MatchIndex` locate the block's first occurrence; `IndexCells` reads its table entry.
Changes become exact patches. Missing keys or ambiguous multi-key records decline the candidate.

Counters a log keeps per entity are coded as keyed running sums: `KeyedSum` adds each delta to
the state its key names and yields the sum. A key starts at zero or, for the original seed
policy, at the last sum yielded; key 0 passes the delta through. An optional integer input
orders the scan, with ties in input order and results kept in their original positions.
Keys come from the record's template (`ExtVariant` through a table) or from a value (`CellIds`, each value's first-occurrence rank); `Interleave` gives each
template its own delta stream. HDFS times keep one clock for rows that start a block's write,
one for rows that finish it and one for reads; other rows code their time by recency. Thread
ids keep a counter per datanode: the writer infers which replica logged a write row from the
pipeline (a `Receiving` row names its upstream, `PacketResponder N` its place) and codes it as
an index into the replica table, which lists each block's replicas in pipeline order; with
write rows stored by template and pipeline place, those codes and the IP columns' repeat block
after block.

A datanode deletes its blocks in id order and its deletion rows do not name it. The writer
infers the replica from its clock and recent use, then stores its explicit position in the
block's replica table. Ordinary table lookups and `CellIds` produce keys; ordered, zero-seeded
`KeyedSum` reconstructs the clocks. The reader does no candidate ranking. `RankedSum` remains
only for decoding older archives, in `payload/legacy.rs`.

A keyed relation that loses a sample by under a half is tried again on the block, where its
keys' first steps weigh less. On a 256 MB HDFS slice the write clocks and counters cut threads
25% and times 20%, the file 12%. The deletion-clock ablation and production validation are
recorded in [the benchmark notes](bench/README.md#rankedsum-ablation).

The registry accepts only graphs with earlier peer dependencies and at least 32 bytes of savings
over the current choice, including graph and shared-code costs. Standalone columns skip relation
packs; context mixing then prices the selected graph's integer transports. The reader knows no
HDFS. State and dependencies stay within the block, and the index records the peers a selective
read needs.

The HDFS pack remains opt-in. Task/partition reuse saved more than the per-block size model on
the measured samples. Emitter- and operation-keyed recent-block caches did not reliably improve
the existing block-ID encoding, so they remain benchmark experiments rather than decoder features.

## payload node scope

The graph has 25 active node kinds plus legacy `RankedSum`; the lane program still has 35
operations. Ordered `KeyedSum` adds options to an existing node, not another opcode or a
second sorting/permutation language. The full node sweep found these boundaries worth keeping:

| Nodes | Why the distinction exists |
|---|---|
| `Bytes` (inline or `LibraryBytes`), `IntStream` | Entropy leaves, native integer transports, and shared model bytes have different storage and lifetime rules. |
| `ByteChunks`, `ParseAsciiInt` | Typed adapters from packed binary words or text cells into integers; neither embeds a corpus vocabulary. |
| `ProgramI64`, `ProgramCells` | The same lane language with native integer or byte-cell outputs. |
| `RecurBuf` | A bounded recurrence expressed as code, used for multiple generator families; no generator-specific reader opcode. |
| `MtfFixed`, `MtfInts` | The same recency idea for 1–64-byte items versus native integers, with different caches and miss storage. A tagged union would retain both implementations. |
| `SliceBuf`, `PagedSlice`, `IndexCells`, `ConstCells` | Consecutive slices, independently paged slices, indexed cell views, and repeated literals. These preserve lazy output and selective reads; lowering them to materializing programs would lose those properties. |
| `TemplateCells`, `PatchCells` | Variable-length assembly and sparse exact exceptions, respectively. Positional indexing alone does neither. |
| `QuoteOrNull` | An intentional JSON-oriented convenience, preserving lazy quoted/dictionary views. This is the clearest remaining format specialization; replacing it with templates and indexing needs a query/memory ablation first. |
| `ExtPool`, `ExtDict`, `ExtPeer` | External buffer, block dictionary cells, and peer/table cells. Their declarations drive dependency loading. Combining tags would still require the same source distinctions. |
| `ExtAlign`, `ExtVariant` | Record alignment and variant metadata supplied by the host; no HDFS template or field names. |
| `MatchIndex`, `CellIds` | Keyed occurrence lookup versus value interning. `IndexCells` is a positional lookup, so it does not replace either. |
| `Interleave`, `KeyedSum` | Stable multiplexing of integer streams and per-key accumulation, composable with explicit keys and ordering values. |
| `RankedSum` | Compatibility only. Candidate ranking, elimination, ordering and clock state made it the over-specialized node; new graphs express the relationship using the simpler nodes above. |

Integer transports are a separate layer. The older bitwise `ContextMix` remains a reader
compatibility form; writers choose `NibbleMix`. The other transports encode constant,
run-length, bit-packed, sparse, Rice, varint, or byte-plane distributions without corpus
semantics. Similar names alone are not evidence that two representations are redundant.

## file layout

A file is two byte regions. Offsets inside each are relative to its start, so a host can place
the regions anywhere and pass their bases to the reader.

**The data region is a stream of self-contained blocks.** It decodes front to back with nothing
else, and two data regions concatenated are one valid stream.

- A block needs its model and row count to decode and, in a file written with one, the file's
  dictionary and tables. No block refers to another block's bytes.
- A block is stored as parts: the row structure, then one part per column, plus the tables
  several columns share (a dictionary, pooled free text). A part decompresses on its own once
  the parts of the same block it depends on are present.
- Reading any row of a block decompresses the whole of each part the query needs, but for a
  large text pool, which is cut into pages of whole rows ordered by content. A record or field
  read takes the pool's first page, which the others are compressed against, and its own, found
  in a per-row page map; a search takes the pages whose token filters hold the needle.

**Block size trades file size for read cost**, set by the writer in rows and bytes. A larger
block gives every column more context and a selective read more to fetch and decode. The stream
default is 65,536 rows or 8 MiB.

**A dictionary gives small blocks the context of the file.** It is the column values that two
or more blocks hold, each once, stored ahead of the blocks; every byte leaf of a part is
compressed against it. Columns, queries and decode speed are as without one. The price is fixed
per file: whoever reads any block reads the whole dictionary first, in the round trip that
fetches the block's parts, and holds it expanded. The writer is handed the dictionary.
`DictionaryBuilder` needs a profile and, in a pass before writing, keeps the values in the most
blocks within a byte budget, at most 16 MiB at levels 1 to 4, which is all zstd searches there.
It skips numbers and holds a hex id as the bytes it spells, as their columns code them. The CLI
builds one for a file of over a block under `--best` (1/32 of the file, 8 to 32 MiB) or
`--dict-bytes`.

**Tables hold what blocks refer to across the file.** An entity named in many blocks, such as
an HDFS block and its replicas, is stored once in a table column carried by the dictionary
frame, and a block column codes a value as a lookup into it by key. Back-references to earlier
blocks would do the same work but fan one query out to every block referred to (up to 48 for
HDFS deletes); a table keeps each block self-contained apart from the table. A table column is
decoded the first time a block reads it, so queries that do not touch those columns pay only
for the table's bytes. A relation pack builds its table in a pass before writing; the CLI makes
one for a file of over a block whose packs keep one.

**Row order is the stream writer's to choose, a run at a time.** Sorting an event log on its
entity fields puts the records that repeat each other in the same blocks, which a column needs
before it can code one from another; a table reaches across blocks only for the relations a
pack models. `pack_stream` given `order_by` fields sorts each run of JSON lines (512 MiB by
default) on them and stores the lines in that order. Blocks, the index and queries see only the
stored order, so row numbers and query results follow it. An order frame ahead of each run
restores the original on unpack, which holds one run in memory; it codes the original order as
ranks counted over the stored lines sorted on a field that nearly ascends in the input, such as
an event id, and so costs about a bit a line. One hour of GitHub Archive sorted on
`type,repo.id` packs 14% smaller with `--best` and 22% smaller without a dictionary, and the
gain grows with the run: 6%, 8% and 10% for runs of 32, 64 and 128 MiB.

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

The writer takes records in their final order; only `pack_stream` reorders them, before any
block is cut. `cut()` ends a block early, and `finish()`
returns the index region. Data region bytes are emitted in order as blocks complete and never
revisited.

- **Memory** is a multiple of the block size and does not grow with the file. A dictionary
  adds its content and match tables, 7 to 13 times its size; choosing one holds a block, a
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
