#!/usr/bin/env python3
"""Marginal cost of fields: blank one field at a time, repack, and subtract.

`zform inspect -v` attributes bytes to the column that stores them; this measures what a
field costs the whole file, including what other columns spend because of it.

  bench/ablate.py LOG --field 4:digits --field 9:const --field 9:prefix=generating\\ core. \\
      -- target/release/zform pack -l 9 {in} -o {out}

A field is `INDEX:MODE` over the line split on single spaces (the last field keeps the rest).
Modes: `const` replaces the field with its first value, `digits` zeroes every digit,
`prefix=P` keeps only P in rows whose field starts with P. `{in}` and `{out}` in the
command are replaced by the paths. Runs the variants in parallel.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def rewrite(src, dst, index, mode, maxsplit):
    first = None
    digits = re.compile(rb"[0-9]")
    prefix = mode[len("prefix=") :].encode() if mode.startswith("prefix=") else None
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for line in fin:
            end = b"\n" if line.endswith(b"\n") else b""
            parts = line[: len(line) - len(end)].split(b" ", maxsplit)
            if index < len(parts):
                v = parts[index]
                if mode == "const":
                    first = v if first is None else first
                    parts[index] = first
                elif mode == "digits":
                    parts[index] = digits.sub(b"0", v)
                elif prefix is not None and v.startswith(prefix):
                    parts[index] = prefix
            fout.write(b" ".join(parts) + end)


def pack(cmd, src, out):
    if os.path.exists(out):
        os.remove(out)
    argv = [a.replace("{in}", src).replace("{out}", out) for a in cmd]
    subprocess.run(argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--field", action="append", required=True, help="INDEX:MODE")
    ap.add_argument("--maxsplit", type=int, default=9, help="split count; the last field keeps the rest")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tmp", default=None, help="scratch dir (needs ~input size per variant)")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a, cmd = ap.parse_args(argv[:cut]), argv[cut + 1 :]
    if not cmd or "{in}" not in " ".join(cmd) or "{out}" not in " ".join(cmd):
        sys.exit("give a pack command with {in} and {out} after --")
    with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
        def run(job):
            n, spec = job
            if spec == "base":
                return spec, pack(cmd, a.input, os.path.join(tmp, "base.zf"))
            idx, mode = spec.split(":", 1)
            src = os.path.join(tmp, f"v{n}.log")
            rewrite(a.input, src, int(idx), mode, a.maxsplit)
            size = pack(cmd, src, src + ".zf")
            os.remove(src)
            return spec, size

        with ThreadPoolExecutor(a.jobs) as ex:
            results = dict(ex.map(run, enumerate(["base"] + a.field)))
    base = results.pop("base")
    print(f"{'base':24s} {base:>12,d}")
    for spec, size in results.items():
        print(f"{spec:24s} {size:>12,d}  field costs {base - size:>11,d} B  ({(base - size) / base:6.1%})")


if __name__ == "__main__":
    main()
