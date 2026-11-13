# `zform` lane programs

The instruction set that column payload graphs use to compute values. [`FORMAT.md` §4](FORMAT.md#4-model-and-column-payloads) covers the graphs that hold programs, the library that shares them, and per-block constant overrides.

A program is data: an encoder writes it and any reader runs it. It is a straight-line expression evaluated per row, with no branches, loops, calls or writes to memory, so it validates in one pass and its cost is its length times the row count. A recurrence is the one form of state carried across rows. `eval_reference` in [`payload/ir.rs`](crates/zform-core/src/payload/ir.rs) defines the semantics; [`payload/engine.rs`](crates/zform-core/src/payload/engine.rs) and [`payload/recur.rs`](crates/zform-core/src/payload/recur.rs) execute them.

## programs

A `Program` is a straight-line sequence of instructions over 32- or 64-bit lanes. It runs once per row. Input values are numbered `0 .. n_in`; instruction `i` defines value `n_in + i`. Operands name earlier values. Integer arithmetic wraps to lane width; floating operations use IEEE-754 `f64` bit patterns in 64-bit lanes.

General programs encode:

```text
header: u8 = width_bit | (n_bufs << 1)    # width_bit: 0 = 32, 1 = 64
n_in: uvarint
n_luts: uvarint | luts: [[u8; 16]; n_luts]
n_ops: uvarint | instructions
n_sinks: uvarint | sinks
```

`n_bufs` is at most 63. Inputs and 16-byte lookup tables are each limited to 64; instructions to 16,384; outputs (`sinks`) to 1–64. The high header bit is reserved except for the compact form below.

An instruction is `opcode: u8`, then one `uvarint` backward distance per operand: `current_value_id - operand_id`. `Select` has operands `a, b, c`; binary operations have `a, b`; `Const` has none; other operations have `a`. `NibbleLut`, `ByteLut32`, and `Gather` then carry `aux: u8`. An immediate follows where required:

| Immediate | Encoding in a general program |
|---|---|
| `ShlI`, `ShrI`, `SarI` | One byte, less than the lane width |
| `F64AddI`, `F64MulI`, `F64DivI`, `ScanF64Add` | Eight LE bytes of IEEE-754 bits |
| Other immediate operations | `ivarint` |

An output is a backward distance from `end = n_in + n_ops`, then `width: u8`. Width 1–4 (32-bit programs) or 1–8 (64-bit programs) emits that many low bytes in LE order. Width `0` is followed by a backward distance naming the lane that supplies each row's byte count. Interpret that count as unsigned and clamp it to the lane's byte width. `ProgramCells` concatenates outputs in order to form each cell. `ProgramI64` requires exactly one fixed eight-byte output in a 64-bit program.

## instruction set

`I` means an immediate operand. All floating operations, integer-to-`f64` conversions, and `MulHiI` require 64-bit lanes.

| Opcode | Operation | Semantics |
|---:|---|---|
| `0` | `Add` | `a + b` |
| `1` | `Sub` | `a - b` |
| `2` | `Mul` | `a * b` |
| `3` | `And` | `a & b` |
| `4` | `Or` | `a \| b` |
| `5` | `Xor` | `a ^ b` |
| `6` | `Shl` | Logical left shift by lane `b` |
| `7` | `Shr` | Logical right shift by lane `b` |
| `8` | `F64Add` | `f64(a) + f64(b)` |
| `9` | `F64Sub` | `f64(a) - f64(b)` |
| `10` | `F64Mul` | `f64(a) * f64(b)` |
| `11` | `Select` | `c != 0 ? a : b` |
| `12` | `AddI` | `a + I` |
| `13` | `MulI` | `a * I` |
| `14` | `AndI` | `a & I` |
| `15` | `OrI` | `a \| I` |
| `16` | `XorI` | `a ^ I` |
| `17` | `ShlI` | Logical left shift by `I` |
| `18` | `ShrI` | Logical right shift by `I` |
| `19` | `SarI` | Arithmetic right shift by `I` |
| `20` | `MulHiI` | High 64 bits of unsigned `a * I` |
| `21` | `F64AddI` | `f64(a) + f64(I)` |
| `22` | `F64MulI` | `f64(a) * f64(I)` |
| `23` | `Ctz` | Count trailing zero bits, or lane width for zero |
| `24` | `Bswap` | Reverse bytes within a lane |
| `25` | `U64ToF64` | Convert unsigned integer to `f64` |
| `26` | `NibbleLut` | Map each byte `b` through `luts[aux][b & 15]` |
| `27` | `ByteLut32` | Map each byte through the 32-byte table at `luts[aux .. aux + 2]`, indexed by `b & 31` |
| `28` | `Gather` | LE read of `1 << (aux >> 6)` bytes at byte offset `a + I` in buffer `aux & 63`; out-of-range reads yield zero |
| `29` | `ScanAdd` | Prefix sum with initial state `I` before row 0 |
| `30` | `ScanXor` | Prefix XOR with initial state `I` before row 0 |
| `31` | `ScanF64Add` | Sequential `f64` prefix sum with initial bit pattern `I` |
| `32` | `I64ToF64` | Convert signed integer to `f64` |
| `33` | `F64DivI` | `f64(a) / f64(I)` |
| `34` | `Const` | `I` in every row |

Variable shift counts use the low five bits in 32-bit programs or the low six bits in 64-bit programs. `Gather`'s immediate offset is nonnegative; the input lane is interpreted as unsigned. Reads address only the declared read-only buffers. Scans carry state across rows within this program execution, starting from their immediate seed. They do not carry state across blocks. `ScanF64Add` reads a NaN running sum as `0.0` before adding the next row. Floating operations preserve their specified order; division and reciprocal multiplication can produce different bits.

## compact programs

A program header of `0x81` encodes a linear 64-bit lane program: `n_ops: uvarint`,
then each opcode and its immediate, if any. There is one input and no tables or
buffers; each instruction takes the previous lane and the last lane is the sole
eight-byte output. Only unary opcodes that need no table or buffer are allowed.
Shift immediates are one byte in 0–63, integer immediates are `ivarint`, and floating
immediates are `uvarint` of their byte-swapped IEEE-754 bits. The compact form uses the same instruction set. It can appear wherever a `Program` is stored, including an inline or library code body.

## recurrences

`RecurBuf` describes `n_state` sequences of state words that advance together. Each next word depends on earlier words of any of them and on optional integer columns:

```text
state[s][i] = prog.sink[s](taps..., cols[0][j], ...)     # s in 0 .. n_state
output[j]   = prog.sink[n_state](...)                    # or state[0][i] when prog has n_state sinks
```

A tap `(lane, lag)` reads `state[lane][i - lag]`. Here `j` counts newly generated steps, starting at zero, and `i` starts after the seed. The recurrence specification encodes:

```text
n_state: uvarint
n_taps: uvarint | taps: [(lane: uvarint, lag: uvarint); n_taps]
prog: blob(Program)
```

There are 1–8 state lanes and 1–16 taps, each lag 1 to 2^20. The seed buffer holds `max(lag)` steps, oldest first, each `n_state` LE words of the program's width. `prog` consumes the taps in listed order, followed by the external integer inputs; each external input has at least `out_words` values. Its sinks are full-width: one per state lane, then optionally the output, which may use any value of the program. It may not contain scans or gathers. One state lane with several lags describes generators such as the Mersenne Twister; several lanes at lag 1 describe those that update a multi-word state jointly.

`out_words` counts newly generated outputs, one per step. When the node's tag has bit `0x80`, the buffer starts with the original seed bytes, followed by those outputs. Generator parameters and transforms are carried in the program; there is no Java or Python generator id in the reader.
