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
| `0x0020` | `FLAG_SORTED_ROWS` | Each block stores its rows in any order. The reader restores a block by sorting its records on their collation key: the ASCII letters and digits, letters lowercased, compared bytewise. The writer sets it only when sorting each block gives its records back, so records with equal keys are equal; the frame's records are its blocks' records in block order. |
| `0x0040` | `FLAG_ORDER` | The frame is an order frame and holds no records: its payload restores the original order of a run of the records after it. |

A dictionary frame comes first in its stream. Its payload is one entropy leaf holding the dictionary's content, then optionally the file's tables; `raw_len` is the content's length, and `content_xxh64`, which is required, is the dictionary's id: the XXH64 of the content followed by the table section. The table section is `n_cols: uvarint` (1–64), then each column as `len: uvarint` and a payload graph (§4) that reads no block part and names as peers only earlier table columns. The content may be empty. The `ZstdDict` leaves of the frames after it, up to the next index region, decode against that content. Content that begins with Zstandard's dictionary magic `37 A4 30 EC` is not valid; a writer puts a zero byte before it.

An order frame's payload is not an entropy leaf, and `content_xxh64`, which is required, is the payload's XXH64 with seed `0x5A464D6F`. The frames of a stream decode to a byte stream; a run is `len` bytes of it, whole `\n`-ended lines stored in an order of the writer's choosing, and it begins `skip` bytes past what the frames before the order frame decode to. Runs do not overlap, and each order frame precedes the frames that hold its run. The payload is `skip: uvarint`, `len: uvarint`, `restore: bytes`, then the run's ranks as an integer stream (§4), one per line. With `restore` empty the stored lines are taken in stored order; otherwise they are stably sorted on the key each has at the top-level field `restore`: a string's bytes between its quotes, a number or literal as written, and the empty key when the line is not a JSON object or lacks the field. Keys that are both ASCII digits compare by length, then bytes, and sort before the rest, which compare bytewise. The original line `i` is the line with `rank[i]` lines before it among those not yet taken.

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
| prose_pool_kind:    u8 (0 none, 1 one leaf, 2 pages)              |
| [prose_pool]:       1: len: uvarint, EntropyLeaf                  |
|                     2: n: uvarint, [len: uvarint; n], [leaf; n]   |
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
- **Pool pages**: a pool over the writer's page size (default 384 KiB raw) may be cut into 2 to 255 pages of whole rows, so a reader decompresses only the pages its rows use. The pool's bytes are the pages' raw bytes back to back. Page 0 is a `Zstd` or `Raw` leaf; every later page is a `Raw` leaf or a `Zstd` frame decoded with page 0's raw bytes as raw-content dictionary (`ZSTD_decompress_usingDict`), so it needs page 0 and itself; page 0 must not begin with the dictionary magic `37 A4 30 EC`. A consuming graph uses `ExtPool` and `PagedSlice`. Within a page, each column's cells lie back to back in cell order.

---

## 4. model and column payloads

Payload reference: [overview](#column-payloads), [example](#worked-example), [graph layout](#graph-layout), [nodes](#node-operands), [integer streams](#integer-streams), [programs](#lane-programs), [recurrences](PROGRAMS.md#recurrences), [shared code](#program-library).

### `ModelPlan` variants

A model routes each record to one variant; the block's `variant_ids` name it per record. A reader knows three kinds, by tag:

- `0 = Shape`: `n: uvarint`, then `n + 1` literals (`blob` each), then `n` slot column ids (`uvarint` each). The record is `lit[0] cell lit[1] cell ... lit[n]`, each cell being the next unread cell of its slot's column. A column may fill several slots of a shape and slots of many shapes.
- `1 = JsonTree`: a tree of JSON paths. Objects have shape tables with inline constant fields, arrays have count streams (`$.payload.commits[]`), scalars are typed cells, a string holding JSON may carry a nested plan one level deep, and a value may be derived from earlier paths of the row by a candidate the selector picks: `Copy`, `Affix`, or `MultiConcat` (tag 2) of `lits[0] ref[0] lits[1] … lits[n]`. A ref naming the path itself is a free part, the path's next cell, at most one per candidate; tag 3 is `"` + Base64 of that concatenation + `"`, with no free part and only printable ASCII (`0x20`–`0x7e`) in the concatenation, so a search can rule a needle out of every such value at once. The trainer keeps up to 1,024 paths and a decoder accepts 2,048.
- `2 = ShapeCell`: `col: uvarint`, `n: uvarint`, then `n` column ids (`uvarint` each), the columns its shapes may name. The next cell of `col` holds exactly one shape in the encoding above and the record follows it. This carries layouts that vary by row: key order, optional fields, per-field quoting.

Variant `0` of every model is a one-slot shape with empty literals over `$residual`: the record verbatim. Other tags are rejected. A slot or declared column id outside the column table, or a `ShapeCell` that declares its own `col`, is rejected when the model is decoded; a `ShapeCell` shape is checked when its cell is read, and a slot naming a column its variant does not declare rejects the block.

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

Writer-side `RelationPack`s use the same graph representation. The optional HDFS pack emits
peer references, integer parsing and formatting, keyed lookups, and exact patches. No HDFS
identifier or additional node kind appears on the wire; only enabling context mixing can add
the integer transport kinds `9` and `10` described below.

### worked example

One possible graph for the cells `auth`, `auth`, `cart` is:

```text
rows = 3, flags = 0

node 0: IntStream [0, 0, 1]              -> integers (row ordinals)
node 1: Bytes "authcart"                 -> buffer
node 2: IntStream [4, 4]                 -> integers (entry lengths)
node 3: SliceBuf(buf = 1, base = 0, lens = 2)       -> cells ["auth", "cart"]
node 4: IndexCells(source = 3, indices = 0)         -> cells ["auth", "auth", "cart"]
root = 4
```

`CoreDecoder` needs only the node semantics to reconstruct the column. The writer may select a different graph if it is smaller.

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

A `src` entry with bit 31 set names column `src & 0x7fffffff` of the file's tables, which it must have. `ExtPeer.slot` indexes the header's `src` list; it is not a column id. `ExtDict.dict_id` must match the header's dictionary id. These declarations let a reader fetch dependencies without evaluating nodes. The model's library supplies code references separately. A missing external input returns `Error::NotLoaded`; an undeclared input is a format error.

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
| `5` | `RecurBuf` | `spec: code_ref`, `seed: ref`, `cols: refs`, `out_words` | Buffer of recurrence output words, optionally prefixed with seed bytes |
| `6` | `ProgramI64` | `prog: code_ref`, `inputs: refs`, `bufs: refs` | Integer lanes from one eight-byte output of a 64-bit program |
| `7` | `ProgramCells` | `prog: code_ref`, `inputs: refs`, `bufs: refs` | One cell per input row, concatenating the program's outputs |
| `8` | `PatchCells` | `base: ref`, `indices: ref`, `values: ref` | Base cells with exception cells substituted |
| `11` | `MtfInts` | `codes: ref`, `misses: ref` | Integer move-to-front decoding over the last 4,096 distinct values |
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
| `22` | `ExtAlign` | `key`, `at` | Integers: for each cell of `at`, the value id of the first `key` cell in its record |
| `23` | `MatchIndex` | `needles: ref`, `hay: ref`, `codes: ref` | Integers: positions in `hay` matched by key |
| `24` | `PagedSlice` | `buf: ref`, `lens: ref`, `pages: ref`, `bases: ref` | Cell `i` of `lens[i]` bytes in page `pages[i]`; page `p`'s cells follow one another from offset `bases[p]` of `buf`; the pages' runs must not overlap |
| `25` | `KeyedSum` | `keys: ref`, `deltas: ref` | Integers: `state[keys[i]] += deltas[i]`, wrapping; a new key's state starts at the last sum yielded (0 at first); key `0` yields `deltas[i]`; keys `0..=rows` |
| `26` | `ExtVariant` | None | Integers: for each cell of this column, its record's variant id |
| `27` | `CellIds` | `cells: ref` | Integers: each cell's value id, the rank of its value's first occurrence |
| `28` | `Interleave` | `sel: ref`, `streams: refs` | Integers: row `i` is the next unread value of `streams[sel[i]]`; every stream is read to its end |
| `29` | `RankedSum` | `at: ref`, `picks: ref`, `deltas: ref`, `order: ref`, `cands: ref`, `width: u8` | Integers: a row with pick `0` yields `deltas[i]`. The others are taken in order of the integer in cell `at[i]` of `order` (parsed as by `ParseAsciiInt`), ties in row order. Rows of one order share the candidates of its first row: the distinct nonempty cells among `cands[width * at[i]..][..width]`, those that were picked in the last order they were candidates of first, each group in cell order. Pick `p` takes the `p`-th candidate that no earlier row of the order took and yields its sum plus `deltas[i]`, wrapping; a candidate's sum starts at the last sum yielded (0 at first) |

`MtfFixed` keeps the last 64 items, most recent first. Code `0` consumes the next `width` fallback bytes and inserts that item. Code `k > 0` selects rank `k - 1` and moves it to the front. `MtfInts` uses the same code convention with integer misses.

`ExtAlign.key` is a peer slot; `at` is `0` for this column or `s + 1` for peer slot `s`. The block supplies it from `variant_ids` and the shapes' slots, so every record must be of a `Shape` variant and every record holding an `at` cell must hold a `key` cell. A value id is the rank of the value's first occurrence among the `key` column's distinct values, so equal values get equal ids. `key` must be a block column. An `at` naming a table column aligns this column, and its ids are joint with that table column's: a value it holds has its id there, and any other value takes the next id past the table's, in order of first occurrence in the block. For needle `i` with code `c > 0`, `MatchIndex` yields the position of the `c`-th `hay` entry equal to the needle; code `0` yields `0`, which the writer patches. `hay` entries must be nonnegative; the reader sizes tables by the largest. With `IndexCells` over a peer and `ExtAlign` for both columns, this codes a value as the one another column holds in a record with the same key.

`PatchCells.indices` holds the first absolute occurrence index followed by nonnegative differences from the previous patch index.

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
| `9` | Context mixture | `min: ivarint`, `bits: u8`, `table_bits: u8`, `bytes: blob` |
| `10` | Nibble context mixture | `min: ivarint`, `bits: u8`, `table_bits: u8`, `escape: u8`, `bytes: blob` |

- **Constant:** repeat `value`. A shifted constant repeats `coefficient << shift`; the shift is 1–63 and must preserve the signed coefficient without overflow.
- **Run length:** the expanded leaf holds `n_runs: uvarint`, then `(value: ivarint, count: uvarint)` pairs. Counts sum to `length`.
- **Bit-packed:** the expanded leaf holds unsigned offsets from `min`, each in `bits` bits, least-significant bit first. `bits` is 0–64. Add `min` with 64-bit wrapping arithmetic to reconstruct each lane.
- **ZigZag varints:** the expanded leaf holds exactly `length` `ivarint` values.
- **Byte planes:** `width` is 1, 2, 4, or 8. The expanded leaf holds `length * width` bytes: all low bytes of unsigned offsets from `min`, then all second bytes, and so on. Interleave and add `min` with 64-bit wrapping arithmetic.
- **Byte head and exceptions:** the expanded head holds one byte per lane. Bytes 0–254 represent offsets from `min` (kind `5`) or ZigZag codes (kind `6`). Byte 255 consumes the next tail value and adds 255 to obtain the offset or ZigZag code. Each exception consumes exactly one tail value. Kind `5` tails must be nonnegative; kind `6` tails carry unsigned bit patterns in signed lanes. A tail is never itself kind `5` or `6`.
- **Centered Rice:** `bytes` contains `center: ivarint`, `k: u8`, then a least-significant-bit-first bitstream. ZigZag-encode each signed difference from `center`. Write `code >> k` zero bits, one bit, then the low `k` bits. `k` is 0–63. Decoders reject arithmetic overflow, truncation, trailing bytes, and nonzero padding.
- **Context mixture:** kind `9` fixes the integer-only model and range coder in
  [`payload/context_mix.rs`](crates/zform-core/src/payload/context_mix.rs).
  `length` is nonzero, `bits` is 1–16, and `table_bits` is 8–14. Each value is an unsigned
  `bits`-wide offset from `min`, coded most-significant bit first. Add `min` with wrapping
  arithmetic to reconstruct signed lanes. `bytes` is the arithmetic stream itself, without
  another entropy-leaf wrapper. It ends with four range-flush bytes.

Kind `9` starts three tables of `2^table_bits` counters at probability 32768 and count zero.
The contexts are the partial bit prefix (starting at 1), the preceding value plus prefix,
and the preceding pair plus prefix. Previous values start at zero. Table indexing uses the
fixed hashes and bit layouts in the reference implementation. A counter's probability uses
16 fractional bits; its update rate is `floor(65536 / (count + 2))`, with count capped at 63.
Probability updates use signed arithmetic shifts and clamp to 1–65535.

Each bit depth has three mixture weights, initially `[21845, 21845, 21846]`, summing to 65536.
Predict with the weighted probability sum shifted right 16, clamped to 1–65535. After the bit,
update weights from their likelihoods and blend 31/32 of that posterior with 1/32 uniform weight.
Assign rounding remainder to the third weight. Then update the model counters. No state
carries into another integer stream. Decoder limits cover parameters, compressed bytes, output
allocation, and model memory; truncated or trailing arithmetic bytes are errors. Predictions
are part of this wire kind: a different model requires a different kind, not an encoder tweak.

Kind `10` has the same parameters and bounds, and its reference implementation is
[`payload/nibble_mix.rs`](crates/zform-core/src/payload/nibble_mix.rs). It codes each offset
most-significant nibble first with a 15-bit multi-symbol range coder. When `escape` is 1 (only for
`bits` above 4), a first symbol codes offsets 0–14 directly, and symbol 15 is followed by the
offset minus 15. Two models are mixed: one indexed by the nibble prefix, and one hashed from the
preceding pair. Each has a 16-symbol cumulative-frequency node per context, and the two-weight
mixture adapts per nibble depth. The stream drops the flush's three trailing zero bytes. Decoders
reject truncation, trailing bytes, and offsets wider than `bits`.

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

`row_modes` has one entry per occurrence. Mode `1` consumes a child record. Mode `2` consumes a child record and restores a trailing newline for JSON-tagged values. Mode `3` consumes a fallback cell. The fallback graph is present only when mode `3` occurs. JSON-tagged values receive an internal `1` byte for child records or `0` for fallbacks. The structural model uses these tags when restoring the enclosing JSON string. Other inner records use untagged values. The child block decodes with the nested model and its library. `0x80` is not a `PayloadGraph` flag accepted by `CoreDecoder`.

### decoding and implementation map

`PayloadGraph::decode` parses standalone graphs against an empty library; `decode_linked` resolves references against a supplied library. `CoreDecoder` takes the parsed graph, declared external inputs, and a `DecodeBudget`. Node input types, lengths, indices, program arity, output widths, and allocation limits are checked while parsing or executing. `decode_col` preserves lazy constant, dictionary, pool, and quoted views where possible; `decode_cells` materializes cells and `decode_ints` returns integer lanes.

| Contract | Implementation |
|---|---|
| Graph wire layout, nodes, integer streams, evaluation | [`payload/graph.rs`](crates/zform-core/src/payload/graph.rs) |
| Program encoding and instruction validation | [`payload/ir.rs`](crates/zform-core/src/payload/ir.rs) |
| Program execution | [`payload/engine.rs`](crates/zform-core/src/payload/engine.rs) |
| Recurrence encoding and execution | [`payload/recur.rs`](crates/zform-core/src/payload/recur.rs) |
| Centered Rice bitstream | [`payload/rice.rs`](crates/zform-core/src/payload/rice.rs) |
| Context-mixture bitstream | [`payload/context_mix.rs`](crates/zform-core/src/payload/context_mix.rs) |
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
| page filter pages | filter pages | directory pages | root: EntropyLeaf
| root_crc32: u32 LE | root_len: u64 LE | region_len: u64 LE | "ZFIE" (4B)
```
`body_len` counts the bytes after it, `region_len` the whole region, and `root_crc32` is the IEEE CRC32 of the root leaf as stored. Each block has one filter page and one directory page, and a block with a paged pool one page filter page, each kind in block order from offset 12; a page may be empty, and a verbatim block has no directory page. A streaming decoder skips the region. A reader takes the last 24 bytes to find the root, and the directories and root are the contiguous tail every query reads.

All fields below are `uvarint` unless typed.

- **Root**: `data_len`; `n_models`, then each model as a length and its `ModelPlan` bytes, each stored once; `n_blocks`, then per block:
  - `kind: u8`. `0` is a structured block, followed by its model id and `kinds`, a mask of what its rows are: `0x01` variant 0, `0x02` `JsonTree`, `0x04` another `Shape`, `0x08` `ShapeCell`. `1` is a verbatim block, followed by its `ResolvedFraming`. `2` is a structured block as `0` with a page filter page. `3` is a block of a `FLAG_SORTED_ROWS` frame, followed by its model id; it has no directory or page filter page and is read whole.
  - the gap between the previous block's end and this block's offset in the data region, its length, its row count, its directory page length, its filter page length and, for kind `2`, its page filter page length. Page offsets follow from the lengths.

  After the blocks comes the length of the dictionary's leaf in the data region, `0` when there is none, else followed by the leaf's offset and `id: u64 LE`.
- **Directory page**: an `EntropyLeaf` holding, with offsets relative to the block's first byte: `struct_len`; the offset of the first structure stream; `n_streams` and each stream's length; `n_dicts` and each shared dictionary's `(offset, length)`; `pool_len`, then if `pool_len > 0` the pool's offset and `n_pages`, `0` for one leaf, else each page's compressed and raw length, which must sum to `pool_len` and stay within the decode budget; `n_cols` and the first column's offset; then the columns in order. A column is `flags: u8`, `len`, and then by flag:
  - `0x01`, the column has values: `letters`, the ASCII letters its cells hold, case folded, bit `i` for `'a' + i` and bit 26 for a backslash.
  - `0x06`, a two-bit kind, `1` integer or `2` timestamp in nanoseconds: `min, max: ivarint`.
  - `0x08`: the column reads the prose pool.
  - `0x10`: the id of the shared dictionary it reads.
  - `0x20`: `n_src` and the ids of the source columns it reads.
  - `0x40`: cell `i` is row `i`'s, every row having one.

  `flags = 0x80` followed by a count stands for a run of that many empty one-byte columns.
- **Filter page**: a power-of-two number of bytes, at least 8, read as bits: bit `p` is bit `p & 7` of byte `p >> 3`. The tokens of a block's records come from the runs of ASCII letters in them, case kept. A run of 3 to 7 letters is one token, `mix(h)` with `h` the FNV-1a 64 hash of its letters. A run of 8 or more gives no such token and one token for every 8 consecutive letters in it, `mix(w ^ 0x6772616d)` with `w` those 8 letters as a `u64 LE`. `mix` is the MurmurHash3 64-bit finalizer. A token sets the bits at its low and its high 32 bits, each modulo the filter's size in bits. The writer starts at 2^17 bits and, while under a quarter of the bits are set, halves the filter by OR-ing its upper half into its lower.
- **Page filter page**: for a pool of `n` pages, `n` bytes giving each page filter's size as `log2` of its bytes, at least 3; then the row page map, row `r`'s value in `w` bits from bit `r * w`, `w` the bit length of `n`, `page + 1` for the page holding the row's pool cells or `0` for none, in `ceil(rows * w / 8)` bytes; then the page filters back to back. A page filter is a filter page of the tokens of that page's cells; the writer halves it while at most 70% of the half's bits would be set. A reader may skip a page whose filter lacks a token that a needle held within one pooled cell must have.

### `ZFP1` profile (`.zfp`)
```text
"ZFP1" (4B) | version: u16 LE (= 1) | plan_leaf: EntropyLeaf(ModelPlan) | xxh64: u64 LE
```
`xxh64` is the XXH64 of the uncompressed `ModelPlan` bytes with seed `0x5A465031`.
