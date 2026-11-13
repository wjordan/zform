# `zform` wire format (`ZFM1`, `ZFIX`, `ZFP1`)

`zform` has three binary formats:

1. **`ZFM1`**: a stream of frames, the data region of a `.zfm` file. Streams concatenate.
2. **`ZFIX` / `ZFIE`**: the index region over a data region, appended to it or stored apart.
3. **`ZFP1`**: a trained model saved as a profile (`.zfp`).

The container holds structural models and self-describing column payloads. [§4](#4-model-and-column-payloads) describes how those payloads reconstruct values, with a worked example, graph operands, integer encodings, lane instructions, recurrences, and shared code. The serialized structural model is defined by the implementation in [`grammar.rs`](crates/zform-core/src/grammar.rs) and [`json_tree.rs`](crates/zform-core/src/json_tree.rs). The format is experimental and changes without notice.

Fixed-width integers are little-endian (`LE`). `uvarint` is 7-bit LEB128 (up to 10 bytes for a `u64`); `ivarint` is the `uvarint` of the ZigZag mapping `(x << 1) ^ (x >> 63)`.

---

## 1. entropy leaf

Every compressed buffer in a frame, an index region or a profile is a leaf that names its own backend:

```text
+---------------+-------------------+--------------------+-------------------------+
| backend: u8   | raw_len: uvarint  | comp_len: uvarint  | payload: [u8; comp_len] |
+---------------+-------------------+--------------------+-------------------------+
```
- `backend = 0` (`Raw`): the bytes themselves; `comp_len` equals `raw_len`.
- `backend = 1` (`Zstd`): a Zstandard frame expanding to `raw_len`.
- `backend = 2` (`Lz4`): an LZ4 block expanding to `raw_len`.
- `backend = 3` (`ZstdDict`): a Zstandard frame expanding to `raw_len` with the stream's dictionary (§2) as raw content. Only the leaves of a block's parts may be one.

The encoder writes `Raw` for a buffer under 32 bytes and whenever compression saves fewer than 3 bytes. A Zstandard frame's window spans `raw_len` up to 2^27 bytes, so a match reaches the start of the buffer and, in a `ZstdDict` leaf, the dictionary.

---

## 2. `ZFM1` frame

A `.zfm` file is one or more `ZFM1` frames, optionally followed by a `ZFIX` index region. As with `gzip` and `zstd`, two `.zfm` files concatenated are a valid stream.

```text
+-------------------------------------------------------------------+
| Frame Header                                                      |
|   magic:        [u8; 4] = "ZFM1"                                  |
|   flags:        u16 LE                                            |
|   backend:      u8 (0=Raw, 1=Zstd, 2=Lz4)                         |
|   raw_len:      uvarint (exact uncompressed byte length)          |
|   record_count: uvarint (number of framed records in frame)       |
|   framing:      ResolvedFraming (1B tag + framing metadata)       |
|   payload_len:  u64 LE (byte length of Frame Payload)             |
|   header_crc32: u32 LE (IEEE CRC32 over preceding header bytes)   |
+-------------------------------------------------------------------+
| Frame Payload (payload_len bytes)                                 |
|   If FLAG_PASSTHROUGH:                                            |
|     raw_chunk_leaf: EntropyLeaf                                   |
|   Else:                                                           |
|     [model_plan_leaf: EntropyLeaf]  (omitted if REUSE_PREV_MODEL) |
|     [prefix_leaf, suffix_leaf] (if FixedStride or JsonArray)      |
|     [sep_leaf] (if JsonArray)                                     |
|     block_count: uvarint                                          |
|     block_dir:   [(n_rows: uvarint, block_len: uvarint); N]       |
|     blocks:      [BlockPayload; N]                                |
+-------------------------------------------------------------------+
| [content_xxh64: u64 LE]  (present if FLAG_HAS_CHECKSUM)           |
+-------------------------------------------------------------------+
```

`content_xxh64` is the XXH64 of the frame's uncompressed bytes with seed `0x5A464D31`.

### frame flags (`u16`)
| Bit | Name | Meaning |
|---:|---|---|
| `0x0001` | `FLAG_HAS_CHECKSUM` | The trailing `content_xxh64` is present. |
| `0x0002` | `FLAG_PASSTHROUGH` | The chunk is stored as one entropy leaf, with no model. |
| `0x0004` | `FLAG_REUSE_PREV_MODEL` | The frame uses the `ModelPlan` of the preceding structured frame of the stream. |
| `0x0008` | `FLAG_SYNC_POINT` | The frame decodes without any frame before it: it carries its `ModelPlan` or is passthrough. |
| `0x0010` | `FLAG_DICTIONARY` | The frame is the stream's dictionary and holds no records. |

A dictionary frame comes first in its stream. Its payload is one entropy leaf holding the dictionary's content, `raw_len` is the content's length, and `content_xxh64`, which is required, is the dictionary's id. The `ZstdDict` leaves of the frames after it, up to the next index region, decode against that content. Content that begins with Zstandard's dictionary magic `37 A4 30 EC` is not valid; a writer puts a zero byte before it.

### record framing (`ResolvedFraming`)
- `0` (`Lf { open_end: u8 }`): `\n`-delimited lines (`open_end = 1` if the last byte is not `\n`).
- `1` (`Crlf { open_end: u8 }`): `\r\n`-delimited lines.
- `2` (`Nul { open_end: u8 }`): `\0`-delimited records.
- `3` (`LengthU32Le`): records with a 4-byte LE length prefix. Input that does not parse this way is framed as `Lf`.
- `4` (`FixedStride { stride: uvarint, prefix_len: uvarint, suffix_len: uvarint }`): fixed-size binary records.
- `5` (`JsonArray`): one JSON document whose records are the elements of its main array, the first array of objects reached from the root through object values only. The frame is `prefix`, the records joined by `sep`, then `suffix`. `sep` is `,` plus the whitespace that follows every `,` in the frame, so the elements of minified and evenly indented documents are bare objects. A frame that stops before the array ends has `,` as its `suffix`; the frame that reaches the closing `]` carries it and everything after it there.

---

## 3. block

Rows of a block stay in arrival order. A column holds a value only for the rows or array elements where its path or template slot occurs, so an absent field costs nothing per row.

```text
+-------------------------------------------------------------------+
| variant_ids:        u32_seq (the model variant of each row)       |
| [json_struct_len]:  uvarint (present iff the model has a JSON tree)|
| [n_struct_streams]: uvarint                                       |
| [json_struct]:      [u32_seq; n_struct_streams]                   |
| n_shared_dicts:     uvarint                                       |
| shared_dicts:       [(len: uvarint, PayloadGraph); n_shared_dicts]|
| has_prose_pool:     u8 (0 or 1)                                   |
| [prose_pool]:       len: uvarint, EntropyLeaf                     |
| col_lengths:        [uvarint; n_cols]                             |
| col_blobs:          [ColumnPayload; n_cols]                      |
+-------------------------------------------------------------------+
```

`u32_seq` is an `IntStream` (§4) whose decoded values must fit `u32`. Each `ColumnPayload` starts with its occurrence count: this is the graph's `rows` field, not a separate prefix. An empty column is the single byte `0`. Nonempty columns hold a `PayloadGraph` or the [nested-column wrapper](#nested-columns).

Every part decodes once its dependencies are present: a graph declares the shared dictionary, prose pool, and source columns it reads in its header. Code references resolve against the model's library.

### JSON structure streams
Each object path has a shape stream (one shape id per occurrence, `0` for a value stored as a cell), each array path a count stream, and each path with derivation candidates a selector stream. At a path that has a table of its own, a non-object value such as `null` is a slot of the parent's shape whose value the child stores as a cell, and an empty `[]` is an array slot with count `0`, so neither adds a shape to the parent.

### shared tables
- **Shared dictionaries** (`shared_dicts`): columns whose string values overlap (GitHub `actor.login`, `user.login`, `owner.login`; source and destination addresses) share a table of cells per block. Each table is itself a `PayloadGraph`. A consuming graph uses `ExtDict` and `IndexCells` with a stream of ordinals.
- **Prose pool** (`prose_pool`): free-text columns whose cells average 40 bytes or more (`body`, commit messages) share one compressed buffer. Pooling requires at least two qualifying columns and must make the block smaller. A consuming graph uses `ExtPool` and `SliceBuf` with its base offset and a stream of cell lengths. Columns that hold raw JSON subtrees stay out of it, so a field read inside one does not fetch the pool.

---

## 4. model and column payloads

Payload reference: [overview](#column-payloads), [example](#worked-example), [graph layout](#graph-layout), [nodes](#node-operands), [integer streams](#integer-streams), [programs](#lane-programs), [recurrences](PROGRAMS.md#recurrences), [shared code](#program-library).

### `ModelPlan` variants

A model routes each record to one variant; the block's `variant_ids` name it per record. A reader knows three kinds, by tag:

- `0 = Shape`: `n: uvarint`, then `n + 1` literals (`blob` each), then `n` slot column ids (`uvarint` each). The record is `lit[0] cell lit[1] cell ... lit[n]`, each cell being the next unread cell of its slot's column. A column may fill several slots of a shape and slots of many shapes.
- `1 = JsonTree`: a tree of JSON paths. Objects have shape tables with inline constant fields, arrays have count streams (`$.payload.commits[]`), scalars are typed cells, a string holding JSON may carry a nested plan one level deep, and a value may be derived from other paths by `Copy`, `Affix` or `MultiConcat` candidates. The trainer keeps up to 1,024 paths and a decoder accepts 2,048.
- `2 = ShapeCell`: `col: uvarint`. The next cell of `col` holds a shape in the encoding above and the record follows it. This carries layouts that vary by row: key order, optional fields, per-field quoting.

Variant `0` of every model is a one-slot shape with empty literals over `$residual`: the record verbatim. Other tags are rejected. A slot id outside the column table is rejected when the model is decoded; a `ShapeCell` shape is checked when its cell is read and may not name its own column.

Record syntax is the writer's business. Its front ends (a template miner, Kubernetes CRI and syslog prefixes, logfmt, CSV, fixed-stride binary) choose columns and emit shapes: separators, keys and quotes are literals, and a field's name is its column's name. A new syntax needs a front end and no reader change. After the variants comes `hints: blob`, which a reader skips: per variant, the front end that made it and its parameters, so a profile splits new records the same way.

### column payloads

A `PayloadGraph` describes how to reconstruct one column. Writer-side `CodecPack`s recognize value patterns and emit graphs; `PackRegistry` compares their encoded costs. The reader executes the selected graph with `CoreDecoder`. No pack name or domain-specific codec id is stored in the payload.

The graph has three kinds of intermediate values:

| Value | Contents | Uses |
|---|---|---|
| Buffer | A byte sequence | Literal data, dictionary bytes, lookup tables, recurrence output |
| Integers | A sequence of 64-bit lanes | Ordinals, lengths, residuals, arithmetic inputs, native numeric output |
| Cells | A sequence of byte strings | Exact field spellings, including quotes, signs, and padding |

Nodes form a directed acyclic graph, serialized in dependency order. A node refers only to earlier nodes by zero-based index. The root supplies the result: cells for a record column, or integers for a native numeric column. Intermediate sequences may have different lengths, such as dictionary entries and exception values. The root's length must equal `rows`, which counts field occurrences rather than records. A field in an array can occur several times per record; an absent field contributes no cell.

The structural model maps those occurrences back to records and assembles record bytes. It also supplies any shared code library. A standalone graph with inline code and no external dependencies decodes without a structural model.

A new writer-side pack can use existing primitives without changing readers. A new node kind, instruction, or entropy backend requires reader support.

### worked example

One possible graph for the cells `auth`, `auth`, `cart` is:

```text
rows = 3, flags = 0

node 0: IntStream [0, 0, 1]              -> integers (row ordinals)
node 1: Bytes "authcart"                 -> buffer
node 2: DictCells(codes = 0,
                  dict_ends = [4, 8],
                  dict_bytes = 1)       -> cells ["auth", "auth", "cart"]
root = 2
```

`dict_ends` is an inline `IntStream` of cumulative byte offsets, so the dictionary is `auth`, then `cart`. `CoreDecoder` needs only the node semantics to reconstruct the column. The writer may select a different graph if it is smaller.

A complete payload for three copies of `ok` is nine bytes:

```text
03 00 01 0f 03 02 6f 6b 00
|  |  |  |  |  |  ----- |
|  |  |  |  |  |   "ok" root = 0
|  |  |  |  |  byte length = 2
|  |  |  |  count = 3
|  |  |  ConstCells tag = 15
|  |  node count = 1
|  flags = 0
rows = 3
```

The public API builds and decodes the same format without a record container:

```rust
use zform::{CoreDecoder, DecodeBudget, PackRegistry};

fn main() -> zform::Result<()> {
    let values: &[&[u8]] = &[b"auth", b"auth", b"cart"];
    let payload = PackRegistry::default().encode_best(values);
    let cells = CoreDecoder::decode_bytes(&payload, &DecodeBudget::default())?;
    assert_eq!(cells.iter().collect::<Vec<_>>(), values);

    let repeated = [0x03, 0x00, 0x01, 0x0f, 0x03, 0x02, b'o', b'k', 0x00];
    let cells = CoreDecoder::decode_bytes(&repeated, &DecodeBudget::default())?;
    assert_eq!(cells.iter().collect::<Vec<_>>(), vec![&b"ok"[..]; 3]);
    Ok(())
}
```

### graph layout

```text
rows: uvarint | flags: u8 | [dict_id: uvarint] | [n_src: uvarint | src: [uvarint; n_src]] | n_nodes: uvarint | nodes | root: uvarint
```

A graph with no rows is exactly one byte, `0`, with no flags or nodes. Nonempty general graphs have 1–4,096 nodes. Their dependency flags combine as follows:

| Flag | Following header fields | Dependency |
|---:|---|---|
| `0x01` | None | The block's prose pool |
| `0x02` | `dict_id: uvarint` | One shared dictionary of cells in the block |
| `0x04` | `n_src: uvarint`, then `src: [uvarint; n_src]` | 1–64 peer column ids in the block |

`ExtPeer.slot` indexes the header's `src` list; it is not a column id. `ExtDict.dict_id` must match the header's dictionary id. These declarations let a reader fetch dependencies without evaluating nodes. The model's library supplies code references separately. A missing external input returns `Error::NotLoaded`; an undeclared input is a format error.

All node references must point backward. `root` must name a node. Unknown flags, unknown node kinds, and trailing graph bytes are rejected. Graphs have no separate magic, version, or checksum; the enclosing column length bounds their bytes and frame checksums protect the record stream.

Two standalone numeric graph forms omit the node directory. Their flags are exact
values, not bits to combine with external dependencies:

- `flags = 0x08`: one `IntStream`, which is the root.
- `flags = 0x10`: one `IntStream`, followed by a `Program` through the end of
  the payload. The root is a `ProgramI64` with that stream as its sole input and no
  buffers. Its only output is an eight-byte lane. The stream length must equal `rows`.

These forms have inline code and no external dependencies or library references. General numeric graphs can also end at `IntStream` or `ProgramI64` nodes. `CoreDecoder::decode_ints_bytes` returns their native lanes; `decode_bytes` expects byte-string cells.

### node operands

Each node starts with `tag: u8`. The low seven bits select its kind. Bit `0x80` means `include_seed` for `RecurBuf` and `has_map` for `QuoteOrNull`; writers leave it clear for other kinds.

Unless typed otherwise, operands are `uvarint`. `ref` names an earlier node; `refs` is a count followed by that many references. `blob` is a byte length followed by those bytes. A `leaf_blob` is a `blob` containing one complete entropy leaf (§1). `IntStream` and code references have their own layouts below.

| Kind | Node | Operands, in wire order | Result |
|---:|---|---|---|
| `0` | `Bytes` | `leaf_blob` | Buffer expanded from the leaf |
| `1` | `IntStream` | `stream: IntStream` | Integers |
| `2` | `ByteChunks` | `buf: ref`, `chunk_bytes: u8` | Integers from 1–8 byte LE chunks; short chunks are zero-extended |
| `3` | `MtfFixed` | `codes: ref`, `fallback_buf: ref`, `width: u8` | Buffer of fixed-width items, width 1–64 |
| `4` | `DictCells` | `codes: ref`, `dict_ends: IntStream`, `dict_bytes: ref` | Cells selected by zero-based dictionary ordinals |
| `5` | `RecurBuf` | `spec: code_ref`, `seed: ref`, `cols: refs`, `out_words` | Buffer of recurrence output words, optionally prefixed with seed bytes |
| `6` | `ProgramI64` | `prog: code_ref`, `inputs: refs`, `bufs: refs` | Integer lanes from one eight-byte output of a 64-bit program |
| `7` | `ProgramCells` | `prog: code_ref`, `inputs: refs`, `bufs: refs` | One cell per input row, concatenating the program's outputs |
| `8` | `PatchCells` | `base: ref`, `indices: ref`, `values: ref` | Base cells with exception cells substituted |
| `9` | `ConcatCells` | `parts: refs` | Row-wise concatenation of 1–32 cell sequences |
| `10` | `ByteUnshuffle` | `buf: ref`, `width: u8` | Buffer interleaving byte planes into items of width 1–64 |
| `11` | `MtfInts` | `codes: ref`, `misses: ref`, `window` | Integer move-to-front decoding, window 1–4,096 |
| `12` | `SliceBuf` | `buf: ref`, `base`, `lens: ref` | Consecutive cells of the given lengths, starting at byte offset `base` |
| `13` | `IndexCells` | `source: ref`, `indices: ref` | Cells selected by zero-based indices |
| `14` | `TemplateCells` | `template: ref`, `slots: refs` | Templates with same-row slot values inserted; 1–64 slots |
| `15` | `ConstCells` | `count`, `value: blob` | `count` copies of the literal cell |
| `16` | `QuoteOrNull` | `inner: ref`, `[map: ref]` | Double-quoted inner cells, or literal `null` for negative map entries |
| `17` | `ParseAsciiInt` | `cells: ref` | Integers parsed from optionally quoted ASCII, with optional sign; invalid cells produce zero |
| `18` | `ExtPool` | None | The declared external prose buffer |
| `19` | `ExtDict` | `dict_id` | The declared external dictionary cells |
| `20` | `ExtPeer` | `slot` | Cells from that slot in the declared peer list |
| `21` | `LibraryBytes` | `id` | Buffer from a byte-table library entry |

`MtfFixed` keeps the last 64 items, most recent first. Code `0` consumes the next `width` fallback bytes and inserts that item. Code `k > 0` selects rank `k - 1` and moves it to the front. `MtfInts` uses the same code convention with integer misses and its explicit window.

`DictCells.dict_ends` contains cumulative byte offsets, ending at the buffer's length. `PatchCells.indices` holds the first absolute occurrence index followed by nonnegative differences from the previous patch index. `ConcatCells` concatenates bytes within each row, not whole columns end to end.

`TemplateCells` copies literal bytes until `0x1f`. The next byte is either `0xff`, meaning a literal `0x1f`, or `(slot << 1) | strip`. A slot inserts that row's cell; `strip = 1` removes its first and last byte when it has at least two bytes. All template and slot sequences have the same row count.

`QuoteOrNull` adds quotes without escaping the inner bytes. With a map, a nonnegative value selects an inner cell and a negative value emits `null`. Without a map, inner cell `i` supplies output cell `i`. The writer applies it only when this preserves the original bytes.

### integer streams

An `IntStream` starts with `kind: u8`, then `length: uvarint`, followed by the operands below. Its result is `length` signed 64-bit lanes. This transport is also used for block structure streams and embedded dictionary ends.

| Kind | Encoding | Following operands |
|---:|---|---|
| `0` | Constant | `value: ivarint` |
| `1` | Run length | `leaf_blob` |
| `2` | Frame of reference, bit-packed | `min: ivarint`, `bits: u8`, `leaf_blob` |
| `3` | ZigZag varints | `leaf_blob` |
| `4` | Frame of reference, byte planes | `min: ivarint`, `width: u8`, `leaf_blob` |
| `5` | Frame of reference, byte head and exceptions | `min: ivarint`, `head: leaf_blob`, `tail: IntStream` |
| `6` | ZigZag byte head and exceptions | `head: leaf_blob`, `tail: IntStream` |
| `7` | Shifted constant | `shift: u8`, `coefficient: ivarint` |
| `8` | Centered Rice | `bytes: blob` |

- **Constant:** repeat `value`. A shifted constant repeats `coefficient << shift`; the shift is 1–63 and must preserve the signed coefficient without overflow.
- **Run length:** the expanded leaf holds `n_runs: uvarint`, then `(value: ivarint, count: uvarint)` pairs. Counts sum to `length`.
- **Bit-packed:** the expanded leaf holds unsigned offsets from `min`, each in `bits` bits, least-significant bit first. `bits` is 0–64. Add `min` with 64-bit wrapping arithmetic to reconstruct each lane.
- **ZigZag varints:** the expanded leaf holds exactly `length` `ivarint` values.
- **Byte planes:** `width` is 1, 2, 4, or 8. The expanded leaf holds `length * width` bytes: all low bytes of unsigned offsets from `min`, then all second bytes, and so on. Interleave and add `min` with 64-bit wrapping arithmetic.
- **Byte head and exceptions:** the expanded head holds one byte per lane. Bytes 0–254 represent offsets from `min` (kind `5`) or ZigZag codes (kind `6`). Byte 255 consumes the next tail value and adds 255 to obtain the offset or ZigZag code. Each exception consumes exactly one tail value. Kind `5` tails must be nonnegative; kind `6` tails carry unsigned bit patterns in signed lanes. Tail nesting permits at most four edges from the outer stream.
- **Centered Rice:** `bytes` contains `center: ivarint`, `k: u8`, then a least-significant-bit-first bitstream. ZigZag-encode each signed difference from `center`. Write `code >> k` zero bits, one bit, then the low `k` bits. `k` is 0–63. Decoders reject arithmetic overflow, truncation, trailing bytes, and nonzero padding.

Delta reconstruction, common integer factors, floating-point scaling, and formatting are programs around these transports. They have no separate `IntStream` mode or domain tag.

### lane programs

`ProgramI64` and `ProgramCells` run a `Program`: a straight-line sequence of instructions over 32- or 64-bit lanes, evaluated once per row. `RecurBuf` runs a recurrence, whose next state word depends on earlier ones. [`PROGRAMS.md`](PROGRAMS.md) specifies both: the encoding, the instruction set, the compact form, and recurrences.

### program library

A `ModelPlan`, nested ones included, ends with the programs and tables its blocks share:

```text
n_entries: uvarint | entries: [(kind: u8, body_len: uvarint, body); n_entries]
```

`kind` is `0` for a `Program`, `1` for a recurrence specification, and `2` for raw table bytes. A library holds at most 4,096 entries. Blocks name an entry by its zero-based index. The blocks of a nested column use the nested plan's library.

### code references

Nodes `5`, `6` and `7` hold their recurrence or program as a code reference `t: uvarint`:

- `t` even: the code follows inline, in `t >> 1` bytes.
- `t` odd: entry `t >> 1` of the library supplies the code. Node `5` requires a recurrence entry; nodes `6` and `7` require a program entry. Program overrides follow: `n: uvarint`, then `n` pairs `(skip: uvarint, value: ivarint)`.

A program's values are the immediates of its instructions in order, then the widths of its fixed-width outputs in order. The first override replaces value `skip`. Each later override advances by `skip + 1` values from the previous override. The resulting constants receive the same validation as inline code. A program shared across blocks is stored once; each block stores only the constants it changes, such as a starting state or field width.

Node `21` is `id: uvarint` and stands for a `Bytes` node holding library table `id`.

A writer builds a sync frame's library from its blocks' programs and writes it with the model. Frames reusing that model also reuse the library. Code stays inline when the library lacks it or a reference and overrides would be larger.

### nested columns

A column whose model supplies a nested plan uses a container wrapper instead of a graph:

```text
rows: uvarint | 0x80: u8 | is_json_tagged: u8
row_modes: u32_seq | child_block: blob | [fallback_graph: blob]
```

`row_modes` has one entry per occurrence. Mode `1` consumes a child record. Mode `2` consumes a child record and restores a trailing newline for JSON-tagged values. Mode `3` consumes a fallback cell. The fallback graph is present only when mode `3` occurs. JSON-tagged values receive an internal `1` byte for child records or `0` for fallbacks. The structural model uses these tags when restoring the enclosing JSON string. Envelope inner records use untagged values. The child block decodes with the nested model and its library. `0x80` is not a `PayloadGraph` flag accepted by `CoreDecoder`.

### decoding and implementation map

`PayloadGraph::decode` parses standalone graphs against an empty library; `decode_linked` resolves references against a supplied library. `CoreDecoder` takes the parsed graph, declared external inputs, and a `DecodeBudget`. Node input types, lengths, indices, program arity, output widths, and allocation limits are checked while parsing or executing. `decode_col` preserves lazy constant, dictionary, pool, and quoted views where possible; `decode_cells` materializes cells and `decode_ints` returns integer lanes.

| Contract | Implementation |
|---|---|
| Graph wire layout, nodes, integer streams, evaluation | [`payload/graph.rs`](crates/zform-core/src/payload/graph.rs) |
| Program encoding and instruction validation | [`payload/ir.rs`](crates/zform-core/src/payload/ir.rs) |
| Program execution | [`payload/engine.rs`](crates/zform-core/src/payload/engine.rs) |
| Recurrence encoding and execution | [`payload/recur.rs`](crates/zform-core/src/payload/recur.rs) |
| Centered Rice bitstream | [`payload/rice.rs`](crates/zform-core/src/payload/rice.rs) |
| Library entries and constant overrides | [`payload/library.rs`](crates/zform-core/src/payload/library.rs) |
| Writer-side pattern recognition and graph selection | [`packs.rs`](crates/zform-core/src/packs.rs) |
| Block dependency loading | [`codecs.rs`](crates/zform-core/src/codecs.rs) |
| Nested-column wrapper | [`container.rs`](crates/zform-core/src/container.rs) |

---

## 5. `ZFIX` / `ZFIE` index region and `ZFP1` profiles

### `ZFIX` index region
Derived from the data region and rebuilt by scanning it. `pack` appends it at EOF; the file writer returns it apart.
```text
"ZFIX" (4B) | body_len: u64 LE
| filter pages | directory pages | root: EntropyLeaf
| root_crc32: u32 LE | root_len: u64 LE | region_len: u64 LE | "ZFIE" (4B)
```
`body_len` counts the bytes after it, `region_len` the whole region, and `root_crc32` is the IEEE CRC32 of the root leaf as stored. Each block has one filter page and one directory page, in block order, the filter pages starting at offset 12; a page may be empty, and a verbatim block has no directory page. A streaming decoder skips the region. A reader takes the last 24 bytes to find the root, and the directories and root are the contiguous tail every query reads.

All fields below are `uvarint` unless typed.

- **Root**: `data_len`; `n_models`, then each model as a length and its `ModelPlan` bytes, each stored once; `n_blocks`, then per block:
  - `kind: u8`. `0` is a structured block, followed by its model id and `kinds`, a mask of what its rows are: `0x01` residual, `0x02` JSON, `0x04` template, `0x08` envelope, `0x10` logfmt, `0x20` CSV, `0x40` binary. `1` is a verbatim block, followed by its `ResolvedFraming`.
  - the gap between the previous block's end and this block's offset in the data region, its length, its row count, its directory page length and its filter page length. Page offsets follow from the lengths.

  After the blocks comes the length of the dictionary's leaf in the data region, `0` when there is none, else followed by the leaf's offset and `id: u64 LE`.
- **Directory page**: an `EntropyLeaf` holding, with offsets relative to the block's first byte: `struct_len`; the offset of the first structure stream; `n_streams` and each stream's length; `n_dicts` and each shared dictionary's `(offset, length)`; `pool_len`, then the pool's offset if `pool_len > 0`; `n_cols` and the first column's offset; then the columns in order. A column is `flags: u8`, `len`, and then by flag:
  - `0x01`, the column has values: `letters`, the ASCII letters its cells hold, case folded, bit `i` for `'a' + i` and bit 26 for a backslash.
  - `0x06`, a two-bit kind, `1` integer or `2` timestamp in nanoseconds: `min, max: ivarint`.
  - `0x08`: the column reads the prose pool.
  - `0x10`: the id of the shared dictionary it reads.
  - `0x20`: `n_src` and the ids of the source columns it reads.
  - `0x40`: cell `i` is row `i`'s, every row having one.

  `flags = 0x80` followed by a count stands for a run of that many empty one-byte columns.
- **Filter page**: a power-of-two number of bytes, at least 8, read as bits: bit `p` is bit `p & 7` of byte `p >> 3`. The tokens of a block's records come from the runs of ASCII letters in them, case kept. A run of 3 to 7 letters is one token, `mix(h)` with `h` the FNV-1a 64 hash of its letters. A run of 8 or more gives no such token and one token for every 8 consecutive letters in it, `mix(w ^ 0x6772616d)` with `w` those 8 letters as a `u64 LE`. `mix` is the MurmurHash3 64-bit finalizer. A token sets the bits at its low and its high 32 bits, each modulo the filter's size in bits. The writer starts at 2^17 bits and, while under a quarter of the bits are set, halves the filter by OR-ing its upper half into its lower.

### `ZFP1` profile (`.zfp`)
```text
"ZFP1" (4B) | version: u16 LE (= 1) | plan_leaf: EntropyLeaf(ModelPlan) | xxh64: u64 LE
```
`xxh64` is the XXH64 of the uncompressed `ModelPlan` bytes with seed `0x5A465031`.
