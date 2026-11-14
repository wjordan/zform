"""Manual whole-file BGL compression experiment; not a supported zform format.

Requires NumPy, libzstd, the release bgl example, and the zform reader.
Build the context helper with: c++ -O3 -std=c++17 bench/bgl_context.cpp -lzstd -o /tmp/bgl_context
Run from the repository root:
  python3 bench/bgl_joint.py pack BGL.log experiment.bgj --clock-bin /tmp/bgl_context
  python3 bench/bgl_joint.py unpack experiment.bgj restored.log --clock-bin /tmp/bgl_context

All models and reconstruction data travel in the archive. Parameters may be
stably grouped by source node or represented as residuals of another parameter.
This prototype buffers the full corpus and provides no message query index.
"""

import collections
import ctypes
import datetime
import json
import pathlib
import pickle
import subprocess
import time
import zipfile

import numpy as np

P = None
CLOCK = None
BGL = None
ZFORM = None
z = ctypes.CDLL("libzstd.so.1")
for name, args in [
    ("ZSTD_compressBound", [ctypes.c_size_t]),
    (
        "ZSTD_compress",
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ],
    ),
    (
        "ZSTD_decompress",
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t],
    ),
]:
    fn = getattr(z, name)
    fn.argtypes = args
    fn.restype = ctypes.c_size_t
z.ZSTD_isError.argtypes = [ctypes.c_size_t]
z.ZSTD_isError.restype = ctypes.c_uint


def comp(b):
    b = bytes(b)
    out = ctypes.create_string_buffer(z.ZSTD_compressBound(len(b)))
    n = z.ZSTD_compress(out, len(out), b, len(b), 9)
    if z.ZSTD_isError(n):
        raise ValueError("zstd compression failed")
    return len(b).to_bytes(8, "little") + out.raw[:n]


def decomp(b):
    size = int.from_bytes(b[:8], "little")
    out = ctypes.create_string_buffer(size)
    n = z.ZSTD_decompress(out, size, b[8:], len(b) - 8)
    if z.ZSTD_isError(n) or n != size:
        raise ValueError("invalid zstd stream length")
    return out.raw


def lane(a):
    a = np.asarray(a, dtype="<i8")
    best = None
    for mode in ["raw", "delta", "xor"]:
        v = (
            a.copy()
            if mode == "raw"
            else np.diff(a, prepend=0)
            if mode == "delta"
            else a ^ np.roll(a, 1)
        )
        if mode == "xor":
            v[0] = a[0]
        lo = int(v.min())
        width = ((int(v.max()) - lo).bit_length() + 7) // 8
        b = (v - lo).astype("<u8").view("u1").reshape(-1, 8)[:, :width].T.tobytes()
        info = [mode, lo, width, len(a)]
        cost = len(comp(b)) + len(json.dumps(info))
        if best is None or cost < best[0]:
            best = cost, info, b
    return best[1:]


def readlane(info, data, pos):
    mode, lo, width, count = info
    size = width * count
    if width:
        raw = np.zeros((count, 8), dtype="u1")
        raw[:, :width] = (
            np.frombuffer(data[pos : pos + size], dtype="u1").reshape(width, count).T
        )
        a = raw.reshape(-1).view("<i8").copy() + lo
    else:
        a = np.full(count, lo, dtype="<i8")
    if mode == "delta":
        a = np.cumsum(a)
    if mode == "xor":
        a = np.bitwise_xor.accumulate(a)
    return a, pos + size


def pack():
    started = time.monotonic()
    shapes, ids, values = pickle.load(open(P / "template/data.pkl", "rb"))
    ids = np.asarray(ids)
    counts = np.bincount(ids)
    nodes = np.fromfile(P / "columns/node_ids", dtype="<u4")
    selected = np.flatnonzero(counts >= 16).tolist()
    mapping = {t: i + 1 for i, t in enumerate(selected)}
    selector = np.array([mapping.get(int(i), 0) for i in ids], dtype="<i8")
    si, sb = lane(selector)
    order = np.argsort(ids, kind="stable")
    sorted_nodes = nodes[order]
    starts = np.r_[0, np.cumsum(counts)]
    meta = []
    nums = bytearray()
    hits = collections.Counter()
    for tid in selected:
        ns = sorted_nodes[starts[tid] : starts[tid + 1]]
        group = np.argsort(ns, kind="stable")
        descs = []
        previous = []
        for j, raw in enumerate(values[tid]):
            a = np.asarray(raw, dtype="<i8")
            info, b = lane(a)
            best = (len(comp(b)) + len(json.dumps(info)), ["plain", -1, False, info], b)

            def consider(v, kind, ref):
                nonlocal best
                for grouped, w in [(False, v), (True, v[group])]:
                    inf, pay = lane(w)
                    desc = [kind, ref, grouped, inf]
                    cost = len(comp(pay)) + len(json.dumps(desc))
                    if cost < best[0]:
                        best = cost, desc, pay

            consider(a, "plain", -1)
            if j:
                probe = a[:512]
                rank = []
                for k, other in enumerate(previous):
                    _, rb = lane(probe - other[:512])
                    rank.append((len(comp(rb)), k))
                for _, k in sorted(rank)[:2]:
                    consider(a - previous[k], "sub", k)
            if len(a) >= 32:
                last = {}
                res = []
                for k, v in zip(ns.tolist(), a.tolist()):
                    res.append(v - last.get(k, 0))
                    last[k] = v
                consider(np.array(res, dtype="<i8"), "node_delta", -1)
            hits[best[1][0] + ("/group" if best[1][2] else "")] += 1
            descs.append(best[1])
            nums.extend(best[2])
            previous.append(a)
        meta.append([[[lit.decode("latin1"), fmt] for lit, fmt in shapes[tid]], descs])
    raw = bytearray()
    for i, line in zip(ids, open(P / "columns/message", "rb")):
        if i not in mapping:
            raw.extend(line)
    manifest = {
        "version": 2,
        "rows": len(ids),
        "selector": si,
        "templates": meta,
        **json.loads((P / "source.json").read_text()),
    }
    streams = {
        "manifest": json.dumps(manifest, separators=(",", ":")).encode(),
        "selector": sb,
        "numeric": nums,
        "fallback": raw,
        "offsets": (P / "offsets").read_bytes(),
        "present": (P / "present").read_bytes(),
    }
    with zipfile.ZipFile(P / "joint.bgj", "w", compression=zipfile.ZIP_STORED) as out:
        for name, buf in streams.items():
            c = comp(buf)
            out.writestr(name, c)
            print(name, len(c), flush=True)
        for name, path in [
            ("header", P / "header200.zfm"),
            ("clock", P / "clock-context.bin"),
        ]:
            out.write(path, name)
            print(name, path.stat().st_size, flush=True)
    print(
        "choices",
        hits,
        "archive",
        (P / "joint.bgj").stat().st_size,
        "assembly_seconds",
        time.monotonic() - started,
        flush=True,
    )


def unpack():
    started = time.monotonic()
    with zipfile.ZipFile(P / "joint.bgj") as archive:
        manifest = json.loads(decomp(archive.read("manifest")))
        selector, _ = readlane(
            manifest["selector"], decomp(archive.read("selector")), 0
        )
        numeric = decomp(archive.read("numeric"))
        fallback = iter(decomp(archive.read("fallback")).split(b"\n"))
        offsets = np.frombuffer(decomp(archive.read("offsets")), dtype="<i4")
        present = decomp(archive.read("present"))
        (P / "decode-header.zfm").write_bytes(archive.read("header"))
        (P / "decode-clock.bin").write_bytes(archive.read("clock"))
    subprocess.run(
        [
            str(ZFORM),
            "unpack",
            str(P / "decode-header.zfm"),
            "-o",
            str(P / "decode-header.log"),
        ],
        check=True,
        timeout=20,
    )
    nodes = []
    nodeids = {}
    for line in open(P / "decode-header.log", "rb"):
        name = line.split(b" ", 4)[3]
        nodes.append(nodeids.setdefault(name, len(nodeids)))
    nodes = np.array(nodes, dtype="<u4")
    nodes.tofile(P / "decoded-nodes")
    subprocess.run(
        [
            str(CLOCK),
            "unpack",
            str(P / "decode-clock.bin"),
            str(P / "decoded-nodes"),
            str(P / "decoded-ticks"),
        ],
        check=True,
        timeout=20,
    )
    ticks = np.fromfile(P / "decoded-ticks", dtype="<u8")
    counts = np.bincount(selector, minlength=len(manifest["templates"]) + 1)
    order = np.argsort(selector, kind="stable")
    sorted_nodes = nodes[order]
    starts = np.r_[0, np.cumsum(counts)]
    decoded = []
    pos = 0
    for tid, (shape, descs) in enumerate(manifest["templates"], 1):
        ns = sorted_nodes[starts[tid] : starts[tid + 1]]
        group = np.argsort(ns, kind="stable")
        lanes = []
        for kind, ref, grouped, info in descs:
            a, pos = readlane(info, numeric, pos)
            if grouped:
                back = np.empty_like(a)
                back[group] = a
                a = back
            if kind == "sub":
                a += lanes[ref]
            if kind == "node_delta":
                last = {}
                for i, k in enumerate(ns):
                    a[i] += last.get(int(k), 0)
                    last[int(k)] = int(a[i])
            lanes.append(a)
        decoded.append(([(lit.encode("latin1"), fmt) for lit, fmt in shape], lanes))
    assert pos == len(numeric)
    cursors = [0] * len(decoded)
    seconds = {}
    epochs = {}
    with open(P / "restored.log", "wb") as out:
        for row, header in enumerate(open(P / "decode-header.log", "rb")):
            tick = int(ticks[row])
            wall = tick // 1000000
            if wall not in seconds:
                seconds[wall] = (
                    (datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=wall))
                    .strftime("%Y-%m-%d-%H.%M.%S")
                    .encode()
                )
            clock = seconds[wall] + f".{tick % 1000000:06d}".encode()
            sec = wall - int(offsets[row])
            epoch = epochs.setdefault(sec, str(sec).encode())
            f = header[:-1].split(b" ")
            f[1] = epoch
            f[2] = clock[:10].replace(b"-", b".")
            f[4] = clock
            tid = int(selector[row])
            if tid == 0:
                body = next(fallback)
            else:
                shape, lanes = decoded[tid - 1]
                cursor = cursors[tid - 1]
                cursors[tid - 1] += 1
                parts = []
                col = 0
                for lit, fmt in shape:
                    parts.append(lit)
                    if fmt:
                        kind, width = fmt
                        value = int(lanes[col][cursor])
                        col += 1
                        parts.append(format(value, "0" + str(width) + kind).encode())
                body = b"".join(parts)
            if present[row]:
                f.append(body)
            out.write(
                b" ".join(f)
                + (b"\n" if manifest["newline"] or row + 1 < manifest["rows"] else b"")
            )
    print("decode_seconds", time.monotonic() - started, flush=True)


def prepare(source):
    import hashlib
    import re
    from array import array

    (P / "columns").mkdir()
    (P / "template").mkdir()
    pattern = re.compile(rb"0[xX][0-9a-fA-F]+|[0-9]+")
    templates = {}
    shapes = []
    values = []
    message_cache = {}
    ids = array("I")
    node_ids = {}
    nodes = array("I")
    ticks = array("Q")
    offsets = array("i")
    present = bytearray()
    seconds = {}
    digest = hashlib.sha256()
    raw_bytes = 0
    newline = True
    with (
        open(source, "rb") as source_file,
        open(P / "header.log", "wb") as header,
        open(P / "columns/message", "wb") as messages,
    ):
        for row, line in enumerate(source_file):
            digest.update(line)
            raw_bytes += len(line)
            newline = line.endswith(b"\n")
            record = line[:-1] if newline else line
            fields = record.split(b" ", 9)
            if len(fields) not in (9, 10):
                raise ValueError(f"row {row}: expected a BGL header")
            clock = fields[4]
            if len(clock) != 26 or clock[19:20] != b"." or not clock[20:].isdigit():
                raise ValueError("expected six fractional timestamp digits")
            prefix = clock[:19]
            if prefix not in seconds:
                date = datetime.datetime.strptime(prefix.decode(), "%Y-%m-%d-%H.%M.%S")
                if date.strftime("%Y-%m-%d-%H.%M.%S").encode() != prefix:
                    raise ValueError("noncanonical clock")
                seconds[prefix] = int(
                    (date - datetime.datetime(1970, 1, 1)).total_seconds()
                )
            wall = seconds[prefix]
            tick = wall * 1000000 + int(clock[20:])
            epoch = int(fields[1])
            if fields[1] != str(epoch).encode() or fields[2] != clock[:10].replace(
                b"-", b"."
            ):
                raise ValueError("epoch/date cannot be derived exactly")
            if ticks and tick < ticks[-1]:
                raise ValueError("context clock requires ordered timestamps")
            ticks.append(tick)
            offsets.append(wall - epoch)
            present.append(len(fields) == 10)
            nodes.append(node_ids.setdefault(fields[3], len(node_ids)))
            header.write(
                b" ".join(
                    [
                        fields[0],
                        b"0",
                        b"0000.00.00",
                        fields[3],
                        b"1970-01-01-00.00.00.000000",
                        *fields[5:9],
                    ]
                )
                + b"\n"
            )
            body = fields[9] if len(fields) == 10 else b""
            messages.write(body + b"\n")
            cached = message_cache.get(body)
            if cached is None:
                parts = []
                numbers = []
                end = 0
                for match in pattern.finditer(body):
                    token = match.group()
                    literal = body[end : match.start()]
                    end = match.end()
                    if token.startswith((b"0x", b"0X")):
                        digits = token[2:]
                        literal += token[:2]
                        if digits.lower() != digits and digits.upper() != digits:
                            parts.append((literal + digits, None))
                            continue
                        fmt = ("X" if digits.upper() == digits else "x", len(digits))
                        value = int(digits, 16)
                    else:
                        fmt = (
                            "d",
                            len(token)
                            if token.startswith(b"0") and len(token) > 1
                            else 0,
                        )
                        value = int(token)
                    if value > 2**63 - 1:
                        parts.append(
                            (
                                literal
                                + (
                                    digits
                                    if token.startswith((b"0x", b"0X"))
                                    else token
                                ),
                                None,
                            )
                        )
                        continue
                    parts.append((literal, fmt))
                    numbers.append(value)
                parts.append((body[end:], None))
                shape = tuple(parts)
                tid = templates.get(shape)
                if tid is None:
                    tid = len(shapes)
                    templates[shape] = tid
                    shapes.append(shape)
                    values.append([array("q") for _ in numbers])
                cached = (tid, numbers)
                message_cache[body] = cached
            tid, numbers = cached
            ids.append(tid)
            for lane_values, value in zip(values[tid], numbers):
                lane_values.append(value)
    if not ids:
        raise ValueError("empty input")
    with open(P / "template/data.pkl", "wb") as output:
        pickle.dump((shapes, ids, values), output)
    np.asarray(nodes, dtype="<u4").tofile(P / "columns/node_ids")
    np.asarray(ticks, dtype="<u8").tofile(P / "ticks.bin")
    np.asarray(offsets, dtype="<i4").tofile(P / "offsets")
    (P / "present").write_bytes(present)
    (P / "source.json").write_text(
        json.dumps(
            {
                "raw_sha256": digest.hexdigest(),
                "raw_bytes": raw_bytes,
                "newline": newline,
            }
        )
    )
    subprocess.run(
        [
            str(BGL),
            str(P / "header.log"),
            str(P / "header200.zfm"),
            "--level",
            "9",
            "--block-bytes",
            "33554432",
            "--block-rows",
            "200000",
            "--raw-message",
        ],
        check=True,
        timeout=45,
    )
    subprocess.run(
        [
            str(CLOCK),
            "pack",
            str(P / "ticks.bin"),
            str(P / "columns/node_ids"),
            str(P / "clock-context.bin"),
        ],
        check=True,
        timeout=20,
    )


if __name__ == "__main__":
    import argparse
    import hashlib
    import shutil
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pack", "unpack"])
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--bgl-bin",
        type=pathlib.Path,
        default=pathlib.Path("target/release/examples/bgl"),
    )
    parser.add_argument(
        "--zform-bin", type=pathlib.Path, default=pathlib.Path("target/release/zform")
    )
    parser.add_argument("--clock-bin", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    CLOCK = args.clock_bin.resolve()
    BGL = args.bgl_bin.resolve()
    ZFORM = args.zform_bin.resolve()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bgl-joint-") as work:
        P = pathlib.Path(work)
        if args.mode == "pack":
            prepare(args.input)
            pack()
            shutil.copyfile(P / "joint.bgj", args.output)
        else:
            shutil.copyfile(args.input, P / "joint.bgj")
            unpack()
            with zipfile.ZipFile(P / "joint.bgj") as archive:
                manifest = json.loads(decomp(archive.read("manifest")))
            with open(P / "restored.log", "rb") as restored:
                digest = hashlib.file_digest(restored, "sha256").hexdigest()
            if digest != manifest["raw_sha256"]:
                raise ValueError("restored SHA-256 mismatch")
            shutil.copyfile(P / "restored.log", args.output)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "bytes": args.output.stat().st_size,
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
