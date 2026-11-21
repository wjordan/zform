#!/usr/bin/env python3
"""Measure complete corpora and check SHA-256 roundtrips for every codec."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TIME_BIN = shutil.which("time") if sys.platform == "linux" else None


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def corpus_input(source, scratch, limit):
    if limit is None and source.suffix != ".gz":
        return source
    open_source = gzip.open if source.suffix == ".gz" else open
    with open_source(source, "rb") as src, scratch.open("wb") as dst:
        if limit is None:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        else:
            data = src.read(limit + 1)
            if len(data) > limit:
                cut = data.rfind(b"\n", 0, limit) + 1
                data = data[:cut or limit]
            dst.write(data)
    return scratch


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def timed(cmd, out_path, repeats, timeout):
    best = float("inf")
    peak = None
    rss_path = out_path.with_suffix(".rss")
    # GNU time avoids counting the Python interpreter's pre-exec high-water mark.
    if TIME_BIN is not None:
        cmd = [TIME_BIN, "-f", "%M", "-o", rss_path, *cmd]
    for _ in range(repeats):
        with out_path.open("wb") as out:
            t0 = time.perf_counter()
            with subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, start_new_session=TIME_BIN is not None) as proc:
                try:
                    code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if TIME_BIN is None:
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                    raise
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
            best = min(best, time.perf_counter() - t0)
        if TIME_BIN is not None:
            rss = int(rss_path.read_text().strip())
            peak = rss if peak is None else max(peak, rss)
    return best, peak


def run_pair(enc, dec, inp, comp, restored, repeats, timeout):
    enc_s, enc_rss = timed([*enc, inp], comp, repeats, timeout)
    dec_s, dec_rss = timed([*dec, comp], restored, repeats, timeout)
    return dict(size=comp.stat().st_size, enc_s=enc_s, dec_s=dec_s, enc_rss=enc_rss, dec_rss=dec_rss)


def fmt_rss(kib):
    return "n/a" if kib is None else f"{kib / 1024:.1f}"


COMPARISON = ("zstd -3", "zstd -19", "xz -6", "zform")
PRESETS = ("zform --best", "zform HDFS predictors", "zform ordered", "zform --best ordered")


def size_cell(raw_len, size):
    return f"{size / 2**20:,.2f} MiB ({raw_len / size:.3g}x)"


def size_table(report, codecs):
    rows = [
        "| Corpus | Raw | " + " | ".join(f"`{codec}`" for codec in codecs) + " |",
        "|---|---:|" + "---:|" * len(codecs),
    ]
    for dataset in report:
        raw_len = dataset["raw_bytes"]
        cells = " | ".join(size_cell(raw_len, dataset["codecs"][c]["size"]) for c in codecs)
        rows.append(f"| {dataset['name']} | {raw_len / 2**20:,.2f} MiB | {cells} |")
    return "\n".join(rows)


def preset_table(report):
    flags = {
        "zform --best": "--best",
        "zform HDFS predictors": "-l 6 --hdfs-predictors --context-mixing",
        "zform ordered": "--order-by type,repo.id",
        "zform --best ordered": "--best --order-by type,repo.id",
    }
    rows = ["| Corpus | Pack flags | Archive |", "|---|---|---:|"]
    for dataset in report:
        for c in PRESETS:
            if c in dataset["codecs"]:
                size = size_cell(dataset["raw_bytes"], dataset["codecs"][c]["size"])
                rows.append(f"| {dataset['name']} | `{flags[c]}` | {size} |")
    return "\n".join(rows)


def print_markdown(report):
    codecs = [c for c in COMPARISON if all(c in d["codecs"] for d in report)]
    print("\n" + size_table(report, codecs))
    print("\n" + preset_table(report))

    print("\n| Corpus | Codec | Compress MB/s | Decompress MB/s | Compress RSS MiB | Decompress RSS MiB |")
    print("|---|---|---:|---:|---:|---:|")
    for dataset in report:
        raw_mb = dataset["raw_bytes"] / 1e6
        for codec, m in dataset["codecs"].items():
            enc = raw_mb / m["enc_s"]
            dec = raw_mb / m["dec_s"]
            print(f"| {dataset['name']} | `{codec}` | {enc:,.1f} | {dec:,.1f} | "
                  f"{fmt_rss(m['enc_rss'])} | {fmt_rss(m['dec_rss'])} |")


def main():
    ap = argparse.ArgumentParser(description="Benchmark full corpora against gzip, zstd and xz.")
    ap.add_argument("--hdfs", type=Path, help="Path to Loghub HDFS.log")
    ap.add_argument("--gharchive", type=Path, help="Path to a GH Archive hour, .json or .json.gz")
    ap.add_argument("--sao", type=Path, help="Path to the Silesia corpus's sao, taken whole")
    ap.add_argument("--slice-mb", type=positive_int, help="Cap log inputs at this many MiB for a quick run")
    ap.add_argument("--repeats", type=positive_int, default=3, help="Runs per codec and direction (default: 3)")
    ap.add_argument("--timeout", type=positive_int, default=600, help="Seconds per codec subprocess (default: 600)")
    ap.add_argument("--markdown", action="store_true", help="Also print the README tables")
    ap.add_argument("--json", type=Path, help="Write the measurements as JSON")
    args = ap.parse_args()

    datasets = [
        (name, path.resolve(), is_log)
        for name, path, is_log in [
            ("HDFS DataNode logs", args.hdfs, True),
            ("GitHub Archive NDJSON", args.gharchive, True),
            ("Silesia sao", args.sao, False),
        ]
        if path is not None
    ]
    if not datasets:
        ap.error("pass --hdfs, --gharchive or --sao")
    for _, source, _ in datasets:
        if not source.is_file():
            ap.error(f"corpus not found: {source}")

    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(["cargo", "build", "--release", "--locked", "--workspace"], cwd=repo_root, check=True)
    zform_bin = str(repo_root / "target" / "release" / "zform")
    limit = None if args.slice_mb is None else args.slice_mb * 2**20

    # Every codec is held to one thread; recent zstd CLIs default to several.
    baselines = []
    if shutil.which("gzip"):
        baselines.append(("gzip -6", ["gzip", "-6", "-c"], ["gzip", "-d", "-c"]))
    if shutil.which("zstd"):
        for level in ("3", "19"):
            baselines.append((f"zstd -{level}", ["zstd", "--single-thread", f"-{level}", "-q", "-c"], ["zstd", "-d", "-q", "-c"]))
    if shutil.which("xz"):
        baselines.append(("xz -6", ["xz", "-T1", "-6", "-c"], ["xz", "-T1", "-d", "-c"]))
    if TIME_BIN is None:
        print("GNU time not found; peak RSS is reported as n/a", file=sys.stderr)

    report = []
    print(f"{'Dataset':<22} | {'Codec':<24} | {'Raw MiB':>8} | {'Comp MiB':>8} | {'Ratio':>8} | {'Enc MB/s':>8} | {'Dec MB/s':>8} | {'Enc RSS':>7} | {'Dec RSS':>7}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        for name, source, is_log in datasets:
            inp = corpus_input(source, scratch / "input", limit if is_log else None)
            raw_len = inp.stat().st_size
            if raw_len == 0:
                ap.error(f"empty corpus: {source}")
            sha = digest(inp)
            print(f"{name} | {raw_len:,} bytes | SHA-256 {sha}", flush=True)

            presets = [("zform", []), ("zform --lz4", ["--lz4"]), ("zform --best", ["--best"])]
            if name.startswith("HDFS"):
                presets.append(("zform HDFS predictors", ["-l", "6", "--hdfs-predictors", "--context-mixing"]))
            if name.startswith("GitHub"):
                order = ["--order-by", "type,repo.id"]
                presets.extend([("zform ordered", order), ("zform --best ordered", ["--best", *order])])
            codecs = baselines + [
                (label, [zform_bin, "pack", "--threads", "1", *extra], [zform_bin, "unpack", "--threads", "1"])
                for label, extra in presets
            ]
            # xz and the other decoders read an explicit input path and write to stdout.
            comp = scratch / "compressed"
            restored = scratch / "restored"
            results = {}
            for label, enc, dec in codecs:
                r = run_pair(enc, dec, inp, comp, restored, args.repeats, args.timeout)
                if digest(restored) != sha:
                    raise RuntimeError(f"roundtrip mismatch on {name} ({label})")
                results[label] = r
                mb = raw_len / 1e6
                print(f"{name:<22} | {label:<24} | {raw_len/2**20:>8.2f} | {r['size']/2**20:>8.2f} | {raw_len/r['size']:>7.2f}x | {mb/r['enc_s']:>8.1f} | {mb/r['dec_s']:>8.1f} | {fmt_rss(r['enc_rss']):>7} | {fmt_rss(r['dec_rss']):>7}", flush=True)
                if label == "zform" and name.startswith("GitHub"):
                    t0 = time.perf_counter()
                    query = [zform_bin, "select", comp, "-f", "id,type,actor.login,repo.name", "--where", "type=PushEvent", "--stats"]
                    p = subprocess.run(query, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, text=True, check=True, timeout=args.timeout)
                    print(f"  -> {(time.perf_counter() - t0) * 1000:.1f} ms | {p.stderr.strip()}", flush=True)
            report.append(dict(name=name, raw_bytes=raw_len, sha256=sha, codecs=results))

    if args.markdown:
        print_markdown(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
