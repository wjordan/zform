"""Manual whole-file BGL compression experiment; not a supported zform format.

Requires NumPy and the context-mixing codec, built with:
  c++ -O2 -std=c++17 bench/bgl_cm.cpp -o /tmp/bgl_cm
Run from the repository root:
  python3 bench/bgl_cm.py pack BGL.log experiment.bcm --bin /tmp/bgl_cm
  python3 bench/bgl_cm.py unpack experiment.bcm restored.log --bin /tmp/bgl_cm

The codec carries node, template, head tuple, clock and template numbers in one arithmetic-coded
stream. The archive adds xz-compressed dictionaries (node names, head tuples, the shapes of
templates with at least 16 rows), the verbatim messages of all other rows, and the epoch offsets.
Date and epoch fields must follow from the clock. Unpack verifies the sha256 of the result.

For the Rust codec, build bench/bgl_cm.rs with rustc --edition=2024 -C opt-level=3.
Optional --clock-bin target/release/examples/bgl_payload_probe stores timestamps as a
self-describing payload graph (BGLCM2). Build it with:
  cargo build --release -p zform-core --example bgl_payload_probe
Pass both binaries when unpacking BGLCM2. The default BGLCM1 format is unchanged.

The fully native frontend also handles parsing, hashing and archive I/O in Rust:
  cargo build --release --manifest-path bench/bgl-native/Cargo.toml
  bench/bgl-native/target/release/bgl-native pack BGL.log experiment.bcm --hybrid
  bench/bgl-native/target/release/bgl-native unpack experiment.bcm restored.log
It calls both codecs in memory and requires neither Python nor helper binaries.
"""

import argparse
import datetime
import hashlib
import json
import lzma
import pathlib
import re
import struct
import subprocess
import tempfile
import time

import numpy as np

MAGIC = b"BGLCM1\n"
HYBRID_MAGIC = b"BGLCM2\n"
MIN_ROWS = 16
STREAMS = ["cm", "nodes", "heads", "shapes", "fallback"]
# 0x numbers, bare hex words of 8+ digits that hold a letter or a leading zero, decimals
NUM = re.compile(rb"0[xX][0-9a-fA-F]+|(?<![0-9A-Za-z_])(?=[0-9a-fA-F]*[a-fA-F]|0)[0-9a-fA-F]{8,}(?![0-9A-Za-z_])|[0-9]+")


def tokenize(body):
    """Split a message into a printf-style shape and its numbers; None shape means no message."""
    if body is None:
        return None, []
    out, nums, end = [], [], 0
    for m in NUM.finditer(body):
        tok = m.group()
        out.append(body[end : m.start()].replace(b"%", b"%%"))
        end = m.end()
        if tok[:2] in (b"0x", b"0X"):
            dg = tok[2:]
            if (dg.lower() != dg and dg.upper() != dg) or len(dg) > 15:
                out.append(tok)  # mixed case or too long: keep literal
                continue
            out.append(tok[:2] + b"%%0%d%s" % (len(dg), b"X" if dg.lower() != dg else b"x"))
            nums.append(int(dg, 16))
        elif len(tok) >= 8 and (tok[:1] == b"0" or not tok.isdigit()):
            if tok.lower() != tok and tok.upper() != tok:
                out.append(tok)
                continue
            kind = b"X" if tok.lower() != tok else b"x"
            for at in range(0, len(tok), 14):  # chunks that fit an int64
                ch = tok[at : at + 14]
                out.append(b"%%0%d%s" % (len(ch), kind))
                nums.append(int(ch, 16))
        elif len(tok) > 18:
            out.append(tok)
        else:
            out.append(b"%%0%dd" % len(tok) if tok[:1] == b"0" and len(tok) > 1 else b"%d")
            nums.append(int(tok))
    out.append(body[end:].replace(b"%", b"%%"))
    return b"".join(out), nums


def lane_count(shape):
    return 0 if shape is None else shape.count(b"%") - 2 * shape.count(b"%%")


def node_dicts(names):
    """Ids of each node's name prefixes through 3, 2 and 1 dash-separated parts, and of the rest."""
    cols = [{}, {}, {}, {}]
    out = [[], [], [], []]
    for name in names:
        p = name.split(b"-")
        for k, key in enumerate((p[:3], p[:2], p[:1], p[3:])):
            out[k].append(cols[k].setdefault(b"-".join(key), len(cols[k])))
    return [np.asarray(a, dtype="<u4") for a in out]


def write_dicts(tmp, names, lanes):
    for name, a in zip(("nodecard", "nodemid", "noderack", "nodepos"), node_dicts(names)):
        a.tofile(tmp / f"{name}.u32")
    np.asarray(lanes, dtype="<u4").tofile(tmp / "lanes.u32")


def second_of(pre):
    """Unix second of a 'YYYY-MM-DD-HH.MM.SS' clock prefix; raises unless it formats back exactly."""
    delta = datetime.datetime.fromisoformat(pre.decode().replace(".", ":")) - datetime.datetime(1970, 1, 1)
    wall = delta.days * 86400 + delta.seconds
    if clock_prefix(wall) != pre:
        raise ValueError(f"non-canonical clock {pre!r}")
    return wall


def clock_prefix(wall):
    return time.strftime("%Y-%m-%d-%H.%M.%S", time.gmtime(wall)).encode()


def pack(src, out, binary, clock_binary=None):
    t0 = time.time()
    digest, newline = hashlib.sha256(), True
    node_ids, head_ids, shape_ids, cache, seconds = {}, {}, {}, {}, {}
    nodes, heads, sids, ticks, offs, nums, counts = [], [], [], [], [], [], []
    for no, line in enumerate(open(src, "rb")):
        digest.update(line)
        newline = line.endswith(b"\n")
        f = (line[:-1] if newline else line).split(b" ", 9)
        if len(f) < 9:
            raise ValueError(f"line {no + 1}: fewer than 9 fields")
        clock = f[4]
        pre, us = clock[:19], clock[20:]
        sec = seconds.get(pre)
        if sec is None:
            wall = second_of(pre)
            sec = (wall, pre[:10].replace(b"-", b"."), None, 0)
        if len(clock) != 26 or clock[19:20] != b"." or not us.isdigit() or f[2] != sec[1]:
            raise ValueError(f"line {no + 1}: clock or date field not canonical")
        # Most rows in a wall-clock second repeat the same epoch spelling and offset.
        if f[1] != sec[2]:
            epoch = int(f[1])
            if b"%d" % epoch != f[1]:
                raise ValueError(f"line {no + 1}: epoch field not canonical")
            sec = seconds[pre] = (sec[0], sec[1], f[1], sec[0] - epoch)
        ticks.append(sec[0] * 1000000 + int(us))
        offs.append(sec[3])
        nodes.append(node_ids.setdefault(f[3], len(node_ids)))
        h = (f[0], f[6], f[7], f[8]) + ((f[5],) if f[3] != f[5] else ())
        heads.append(head_ids.setdefault(h, len(head_ids)))
        body = f[9] if len(f) == 10 else None
        c = cache.get(body)
        if c is None:
            shape, vals = tokenize(body)
            c = cache[body] = (shape_ids.setdefault(shape, len(shape_ids)), vals)
        sids.append(c[0])
        nums.extend(c[1])
        counts.append(len(c[1]))
    if not sids:
        raise ValueError("empty input")
    del cache
    shapes = list(shape_ids)
    ticks = np.asarray(ticks, dtype="<i8")
    if (np.diff(ticks) < 0).any():
        raise ValueError("clock goes backwards")
    sids = np.asarray(sids)
    counts = np.asarray(counts)
    nums = np.asarray(nums, dtype="<i8")
    chosen = np.flatnonzero(np.bincount(sids, minlength=len(shapes)) >= MIN_ROWS)
    sel_of = np.zeros(len(shapes), dtype="<u4")
    sel_of[chosen] = np.arange(1, len(chosen) + 1)
    sel = sel_of[sids]
    starts = np.r_[0, np.cumsum(counts)]
    fallback = []
    for r in np.flatnonzero(sel == 0).tolist():
        shape = shapes[sids[r]]
        vals = tuple(nums[starts[r] : starts[r + 1]].tolist())
        fallback.append(b"-" if shape is None else b"+" + shape % vals)
    lanes = [0] + [lane_count(shapes[s]) for s in chosen]
    names = list(node_ids)
    offs = np.asarray(offs)
    cut = np.flatnonzero(np.diff(offs)) + 1
    runs = [[int(offs[a]), int(b - a)] for a, b in zip(np.r_[0, cut], np.r_[cut, len(offs)])]
    t_parse = time.time() - t0
    with tempfile.TemporaryDirectory(prefix="bgl_cm.") as d:
        tmp = pathlib.Path(d)
        write_dicts(tmp, names, lanes)
        np.asarray(nodes, dtype="<u4").tofile(tmp / "node.u32")
        sel.astype("<u4").tofile(tmp / "sel.u32")
        np.asarray(heads, dtype="<u4").tofile(tmp / "head.u32")
        ticks.tofile(tmp / "ticks.i64")
        nums[np.repeat(sel > 0, counts)].tofile(tmp / "nums.i64")
        if clock_binary:
            subprocess.run([clock_binary, "pack", tmp / "ticks.i64", tmp / "clock.graph"], check=True)
            clock_graph = (tmp / "clock.graph").read_bytes()
        mode = "pack-external-clock" if clock_binary else "pack"
        res = subprocess.run([binary, mode, tmp, tmp / "out.cm"], check=True, capture_output=True, text=True)
        cm = (tmp / "out.cm").read_bytes()
    xz = lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)
    streams = {
        "cm": cm,
        "nodes": xz(b"\n".join(names)),
        "heads": xz(b"\n".join(b" ".join(h) for h in head_ids)),
        "shapes": xz(b"\n".join(b"-" if shapes[s] is None else b"+" + shapes[s] for s in chosen)),
        "fallback": xz(b"\n".join(fallback)),
    }
    if clock_binary:
        streams["clock"] = clock_graph
    magic = HYBRID_MAGIC if clock_binary else MAGIC
    header = json.dumps({"rows": len(sel), "newline": newline, "sha256": digest.hexdigest(), "offsets": runs}).encode()
    with open(out, "wb") as o:
        o.write(magic + struct.pack("<I", len(header)) + header)
        for stream in streams.values():
            o.write(struct.pack("<I", len(stream)) + stream)
    sizes = {"header": len(magic) + 4 + len(header)} | {k: 4 + len(v) for k, v in streams.items()}
    print(json.dumps({"total_bytes": sum(sizes.values()), "streams": sizes, "codec": json.loads(res.stdout),
                      "selected_templates": len(chosen), "fallback_rows": len(fallback),
                      "parse_seconds": round(t_parse, 2), "seconds": round(time.time() - t0, 2)}, indent=1))


def unpack(src, out, binary, clock_binary=None):
    t0 = time.time()
    data = pathlib.Path(src).read_bytes()
    hybrid = data.startswith(HYBRID_MAGIC)
    if not hybrid and not data.startswith(MAGIC):
        raise ValueError("not a bgl_cm archive")
    if hybrid and not clock_binary:
        raise ValueError("BGLCM2 requires --clock-bin to decode its timestamp payload")
    at = len(MAGIC)
    (n,) = struct.unpack_from("<I", data, at)
    header = json.loads(data[at + 4 : at + 4 + n])
    at += 4 + n
    streams = {}
    for k in STREAMS + (["clock"] if hybrid else []):
        (n,) = struct.unpack_from("<I", data, at)
        streams[k] = data[at + 4 : at + 4 + n]
        at += 4 + n
    def split(k):
        b = lzma.decompress(streams[k])
        return b.split(b"\n") if b else []

    names, heads = split("nodes"), [h.split(b" ") for h in split("heads")]
    shapes = [None if s == b"-" else s[1:] for s in split("shapes")]
    fallback = [None if b == b"-" else b[1:] for b in split("fallback")]
    lanes = [0] + [lane_count(s) for s in shapes]
    with tempfile.TemporaryDirectory(prefix="bgl_cm.") as d:
        tmp = pathlib.Path(d)
        write_dicts(tmp, names, lanes)
        (tmp / "in.cm").write_bytes(streams["cm"])
        if hybrid:
            (tmp / "clock.graph").write_bytes(streams["clock"])
            subprocess.run([clock_binary, "unpack", tmp / "clock.graph", tmp / "ticks.i64"], check=True)
        mode = "unpack-external-clock" if hybrid else "unpack"
        res = subprocess.run([binary, mode, tmp / "in.cm", tmp], check=True, capture_output=True, text=True)
        node = np.fromfile(tmp / "node.u32", dtype="<u4").tolist()
        sel = np.fromfile(tmp / "sel.u32", dtype="<u4").tolist()
        head = np.fromfile(tmp / "head.u32", dtype="<u4").tolist()
        ticks = np.fromfile(tmp / "ticks.i64", dtype="<i8")
        nums = np.fromfile(tmp / "nums.i64", dtype="<i8").tolist()
    t_codec = time.time() - t0
    walls, us = (ticks // 1000000).tolist(), (ticks % 1000000).tolist()
    offs = np.repeat(*zip(*header["offsets"])).tolist() if header["offsets"] else []
    h = hashlib.sha256()
    buf, fb, pos, last = [], iter(fallback), 0, None
    with open(out, "wb") as o:
        for i in range(header["rows"]):
            wall = walls[i]
            if wall != last:
                pre = clock_prefix(wall)
                date, last = pre[:10].replace(b"-", b"."), wall
            name, hd, s = names[node[i]], heads[head[i]], sel[i]
            f = [hd[0], b"%d" % (wall - offs[i]), date, name, pre + b".%06d" % us[i], hd[4] if len(hd) == 5 else name, hd[1], hd[2], hd[3]]
            if s:
                shape, k = shapes[s - 1], lanes[s]
                body = None if shape is None else shape % tuple(nums[pos : pos + k])
                pos += k
            else:
                body = next(fb)
            if body is not None:
                f.append(body)
            buf.append(b" ".join(f))
            if len(buf) == 65536 and i + 1 < header["rows"]:
                chunk = b"\n".join(buf) + b"\n"
                h.update(chunk)
                o.write(chunk)
                buf = []
        chunk = b"\n".join(buf) + (b"\n" if header["newline"] else b"")
        h.update(chunk)
        o.write(chunk)
    if h.hexdigest() != header["sha256"]:
        raise SystemExit(f"sha256 mismatch: restored {h.hexdigest()}, archive {header['sha256']}")
    print(json.dumps({"rows": header["rows"], "sha256": "ok", "codec_seconds": round(t_codec, 2),
                      "seconds": round(time.time() - t0, 2), "codec": json.loads(res.stdout)}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["pack", "unpack"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bin", required=True, help="path to the built bgl_cm codec")
    ap.add_argument("--clock-bin", help="payload graph helper; requires the Rust codec")
    a = ap.parse_args()
    (pack if a.mode == "pack" else unpack)(a.src, a.out, a.bin, a.clock_bin)


if __name__ == "__main__":
    main()
