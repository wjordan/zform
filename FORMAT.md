# `zform` wire format (`ZFM1`, `ZFIX`, `ZFP1`)

`zform` has three binary formats:

1. **`ZFM1`**: a stream of frames, the data region of a `.zfm` file. Streams concatenate.
2. **`ZFIX` / `ZFIE`**: the index region over a data region, appended to it or stored apart.
3. **`ZFP1`**: a trained model saved as a profile (`.zfp`).

This document fixes the container: leaves, frames, the parts of a block, the index region and the profile file. It lists the model's record families and the column codecs with their tags. The serialized model and the payload of each codec are so far defined only by the reference implementation in `crates/zform-core`. The format is not frozen: until 1.0 a release may change it.

Fixed-width integers are little-endian (`LE`). `uvarint` is 7-bit LEB128 (up to 10 bytes for a `u64`); `ivarint` is the `uvarint` of the ZigZag mapping `(x << 1) ^ (x >> 63)`.

---

## 1. Entropy leaf

Every compressed buffer in a frame, an index region or a profile is a leaf that names its own backend:

```text
+---------------+-------------------+--------------------+-------------------------+
| backend: u8   | raw_len: uvarint  | comp_len: uvarint  | payload: [u8; comp_len] |
+---------------+-------------------+--------------------+-------------------------+
```
- `backend = 0` (`Raw`): the bytes themselves; `comp_len` equals `raw_len`.
- `backend = 1` (`Zstd`): a Zstandard frame expanding to `raw_len`.
- `backend = 2` (`Lz4`): an LZ4 block expanding to `raw_len`.

The encoder writes `Raw` for a buffer under 32 bytes and whenever compression saves fewer than 3 bytes.

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

### Frame flags (`u16`)
| Bit | Name | Meaning |
|---:|---|---|
| `0x0001` | `FLAG_HAS_CHECKSUM` | The trailing `content_xxh64` is present. |
| `0x0002` | `FLAG_PASSTHROUGH` | The chunk is stored as one entropy leaf, with no model. |
| `0x0004` | `FLAG_REUSE_PREV_MODEL` | The frame uses the `ModelPlan` of the preceding structured frame of the stream. |
| `0x0008` | `FLAG_SYNC_POINT` | The frame decodes without any frame before it: it carries its `ModelPlan` or is passthrough. |

### Record framing (`ResolvedFraming`)
- `0` (`Lf { open_end: u8 }`): `\n`-delimited lines (`open_end = 1` if the last byte is not `\n`).
- `1` (`Crlf { open_end: u8 }`): `\r\n`-delimited lines.
- `2` (`Nul { open_end: u8 }`): `\0`-delimited records.
- `3` (`LengthU32Le`): records with a 4-byte LE length prefix. Input that does not parse this way is framed as `Lf`.
- `4` (`FixedStride { stride: uvarint, prefix_len: uvarint, suffix_len: uvarint }`): fixed-size binary records.
- `5` (`JsonArray`): one JSON document whose records are the elements of its main array, the first array of objects reached from the root through object values only. The frame is `prefix`, the records joined by `sep`, then `suffix`. `sep` is `,` plus the whitespace that follows every `,` in the frame, so the elements of minified and evenly indented documents are bare objects. A frame that stops before the array ends has `,` as its `suffix`; the frame that reaches the closing `]` carries it and everything after it there.

---

## 3. Block

Rows of a block stay in arrival order. A column holds a value only for the rows or array elements where its path or template slot occurs, so an absent field costs nothing per row.

```text
+-------------------------------------------------------------------+
| variant_ids:        u32_seq (the model variant of each row)       |
| [json_struct_len]:  uvarint (present iff the model has a JSON tree)|
| [n_struct_streams]: uvarint                                       |
| [json_struct]:      [u32_seq; n_struct_streams]                   |
| n_shared_dicts:     uvarint                                       |
| shared_dicts:       [(len: uvarint, DictTable); n_shared_dicts]   |
| has_prose_pool:     u8 (0 or 1)                                   |
| [prose_pool]:       len: uvarint, EntropyLeaf                     |
| col_lengths:        [uvarint; n_cols]                             |
| col_blobs:          [(occ_count: uvarint, DenseColPayload); N]    |
+-------------------------------------------------------------------+
```

Every part decodes on its own once the parts it depends on are present: a column names the shared dictionary, the prose pool and the source columns it reads in its header. A column no row has a value in is the single byte `0`.

### JSON structure streams
Each object path has a shape stream (one shape id per occurrence, `0` for a value stored as a cell), each array path a count stream, and each path with derivation candidates a selector stream. At a path that has a table of its own, a non-object value such as `null` is a slot of the parent's shape whose value the child stores as a cell, and an empty `[]` is an array slot with count `0`, so neither adds a shape to the parent.

### Shared tables
- **Shared dictionaries** (`shared_dicts`): columns whose string values overlap (GitHub `actor.login`, `user.login`, `owner.login`; source and destination addresses) share one `DictTable` per block and each store a stream of ordinals into it. The table is the smallest of a frequency-sorted, a first-seen and a sorted front-coded FSST encoding.
- **Prose pool** (`prose_pool`): free-text columns whose cells average 40 bytes or more (`body`, commit messages) are concatenated into one buffer compressed as a whole, when at least two columns qualify and the pool makes the block smaller. Each such column becomes a `ProseSlice`: its offset and length in the pool and a `u32_seq` of cell lengths. Columns that hold raw JSON subtrees stay out of it, so a field read inside one does not fetch the pool.

---

## 4. Model and column codecs

### `ModelPlan` variants
A model routes each record to one variant, by tag:
- `0 = Residual`: the record verbatim, in the `$residual` column.
- `1 = JsonTree`: a tree of JSON paths. Objects have shape tables with inline constant fields, arrays have count streams (`$.payload.commits[]`), scalars are typed cells, a string holding JSON may carry a nested plan one level deep, and a value may be derived from other paths by `Copy`, `Affix` or `MultiConcat` candidates. The trainer keeps up to 1,024 paths and a decoder accepts 2,048.
- `3 = Envelope`: a Kubernetes CRI (`<ts> stdout|stderr F|P <inner>`) or syslog (`[<pri>]<ts> <host> <tag>: <inner>`) prefix around an inner `ModelPlan`.
- `4 = Logfmt`: `key=value` pairs in any key order, through `$logfmt.recipe` and `$logfmt.overflow`.
- `5 = Template`: literal text with typed slots (`IPv4`, `:port`, `blk_<i64>`, timestamps, UUIDs); slots of the same kind and context share a column (`$ts`, `$traceparent`, `$ip.<ctx>`, `$blk_id`, `lvl`).
- `6 = Csv`: `,` or `\t` separated fields and a per-row quote bitmask.
- `7 = BinaryStride`: field slices of fixed-stride binary records.

Tag `2` is unused and rejected.

### Column codecs
Each column has a horizontal mode (`0 = Standalone`, `1 = ExactCopy`, `2 = SubstringInterp` over up to 3 source columns, `3 = AffineInt` `c = a ± b + residual`, `4 = NestedPlan`, `5 = CrossHexId`, a hex id found in another column) and a leaf codec:
- `0 = Const`, `1 = Raw`, `2 = InlineDict` (plain, front-coded, FSST front-coded or integer-valued, with direct or move-to-front ordinals), `3 = SharedDictRef` (ordinals into a shared dictionary),
- `4 = IntExact` (constant, bit-packed frame of reference, byte planes, GCD-normalized delta, delta of delta, or move-to-front; keeps `-0`, `+42`, `007`),
- `5 = DecimalExact` (scaled integer mantissa with exact decimal places and sign),
- `6 = Timestamp` (ISO-8601/RFC 3339, Apache CLF, syslog RFC 3164: epoch deltas and a format descriptor),
- `7 = Uuid` (16 bytes, the UUIDv7 timestamp split off as deltas),
- `8 = Ulid` (48-bit ms timestamp deltas and an 80-bit payload),
- `9 = Traceparent` (16-byte `trace_id`, 8-byte `span_id`, flags),
- `10 = HexMtf`, `11 = Ipv4Port` (4-byte address and a `u32_seq` port), `12 = EmailSplit`, `13 = HierarchicalSplit` (`/` in URLs, `-` in pod names), `14 = FixedBinary` (byte shuffle; IEEE-754 `f64`/`f32` bit deltas),
- `15 = QuotedOrNull` (`submode: u8`: `0` every value is a quoted string; `1` each is a quoted string or `null`, by a `u32_seq` of flags; the strings' inner bytes go to another codec),
- `16 = HexId` (nibble-packed hex ids with back-references within the column, for commit and tree SHAs),
- `17 = ProseSlice` (a slice of the block's prose pool).

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
