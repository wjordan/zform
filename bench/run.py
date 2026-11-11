#!/usr/bin/env python3
"""Reproducible compression & compressed-domain pushdown benchmark for zform."""

import argparse
import base64
import gzip
import hashlib
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TIME_BIN = shutil.which("time") if sys.platform == "linux" else None


def slice_lines(raw_bytes: bytes, max_bytes: int) -> bytes:
    if len(raw_bytes) <= max_bytes:
        return raw_bytes
    cut = raw_bytes.rfind(b"\n", 0, max_bytes)
    return raw_bytes[: cut + 1] if cut > 0 else raw_bytes[:max_bytes]


def make_otel_hybrid(target_bytes: int = 8 * 1024 * 1024) -> bytes:
    rng = random.Random(42)
    services = ["checkout-api", "payment-worker", "inventory-svc", "auth-gateway", "edge-proxy", "fraud-scorer"]
    pods = [f"{s}-{rng.choice(['7d9f8b', '5c4a1e', '9b2d0f'])}-{rng.choice(['xk2p9', 'm4n7q', 'v8r3t'])}" for s in services for _ in range(3)]
    routes = ["/api/v2/checkout", "/api/v2/payments/authorize", "/api/v2/inventory/reserve", "/api/v1/tokens/verify", "/healthz"]
    traces = [rng.randbytes(16).hex() for _ in range(256)]
    users = [f"usr_{100000 + i}@corp-{i % 17}.example.com" for i in range(400)]

    out = bytearray()
    ts_ms = 1731250000_000
    seq = 0
    while len(out) < target_bytes:
        ts_ms += rng.randint(0, 12)
        sec, ms = divmod(ts_ms, 1000)
        ns_rem = rng.randint(0, 999999)
        tm = time.gmtime(sec)
        iso = f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}T{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{ms:03d}{ns_rem:06d}Z"
        trace_id = traces[rng.randrange(len(traces))]
        span_id = rng.randbytes(8).hex()
        tp = f"00-{trace_id}-{span_id}-01"
        pod = pods[seq % len(pods)]
        svc = pod.rsplit("-", 2)[0]
        start_us = ts_ms * 1000 + rng.randint(0, 999)
        dur_us = rng.choice([450, 820, 1250, 3400, 18900, 142000]) + rng.randint(0, 50)
        end_us = start_us + dur_us
        uuid7_int = (
            ((ts_ms & ((1 << 48) - 1)) << 80)
            | (0x7 << 76)
            | (rng.getrandbits(12) << 64)
            | (0b10 << 62)
            | rng.getrandbits(62)
        )
        u_hex = f"{uuid7_int:032x}"
        uuid7 = f"{u_hex[:8]}-{u_hex[8:12]}-{u_hex[12:16]}-{u_hex[16:20]}-{u_hex[20:]}"

        mode = seq % 5
        if mode in (0, 1):
            route = routes[seq % len(routes)]
            status = 200 if rng.random() < 0.94 else rng.choice([400, 429, 502, 503])
            inner_msg = f'{{"event":"rpc.complete","service":"{svc}","route":"{route}","status":{status},"dur_us":{dur_us}}}'
            inner_esc = inner_msg.replace("\\", "\\\\").replace('"', '\\"')
            line = (
                f'{{"ts":"{iso}","level":"{"INFO" if status < 500 else "ERROR"}","service":"{svc}","pod":"{pod}",'
                f'"req_id":"{uuid7}","traceparent":"{tp}","url":"https://{svc}.internal{route}/trace/{trace_id}",'
                f'"start_us":{start_us},"dur_us":{dur_us},"end_us":{end_us},"body":"{inner_esc}"}}\n'
            )
        elif mode == 2:
            user = users[rng.randrange(len(users))]
            ip = f"10.42.{(seq * 7) % 32}.{(seq * 13) % 250}:{30000 + (seq % 128)}"
            if seq % 2 == 0:
                lf = f'ts={iso} level=info svc={svc} pod={pod} peer={ip} user={user} traceparent={tp} latency_ms={dur_us / 1000:.3f} msg="request handled"'
            else:
                lf = f'level=info svc={svc} ts={iso} traceparent={tp} peer={ip} pod={pod} latency_ms={dur_us / 1000:.3f} user={user} msg="request handled"'
            line = f"{iso} stdout F {lf}\n"
        elif mode == 3:
            blk = -9000000000000000000 + (seq // 3) * 10007
            ip1 = f"10.250.{seq % 16}.{(seq * 3) % 250}:50010"
            ip2 = f"10.250.{(seq + 5) % 16}.{(seq * 11) % 250}:50010"
            line = f"{iso} INFO {svc} [{pod}] Receiving block blk_{blk} src: /{ip1} dest: /{ip2} trace={tp} bytes={4096 * (1 + (seq % 16))}\n"
        else:
            extra = f',"retry_after_ms":{rng.randint(100, 5000)},"upstream":"10.99.1.{seq % 8}:8443"' if seq % 15 == 0 else ""
            line = f'{{"ts":"{iso}","service":"{svc}","pod":"{pod}","traceparent":"{tp}","start_us":{start_us},"dur_us":{dur_us},"end_us":{end_us}{extra}}}\n'

        out.extend(line.encode())
        seq += 1
    return bytes(out)


def make_sao_binary(n_records: int = 200_000) -> bytes:
    out = bytearray(b"SAO_STAR_CATALOG_HDR_28BYTES")
    rng = random.Random(7)
    ra, dec = 0.01, -1.2
    specs = [b"G2", b"K0", b"A0", b"F5", b"M1", b"B8", b"G8", b"K5"]
    for i in range(n_records):
        ra += rng.uniform(0.0, 0.00003)
        dec += rng.uniform(-0.0001, 0.00012)
        sp = specs[i % len(specs)]
        mag = int(400 + (i % 650))
        out.extend(struct.pack("<dd2sHff", ra, dec, sp, mag, round(rng.gauss(0.0, 1e-6), 8), round(rng.gauss(0.0, 1e-6), 8)))
    return bytes(out)


def make_random_b64(n_bytes: int = 3 * 1024 * 1024) -> bytes:
    rng = random.Random(99)
    raw = rng.randbytes(n_bytes)
    return b"\n".join(base64.b64encode(raw[i : i + 96]) for i in range(0, len(raw), 96)) + b"\n"


def timed(cmd, out_path, repeats):
    """Best wall time over `repeats` runs of `cmd`, stdout redirected to `out_path`."""
    best = float("inf")
    for _ in range(repeats):
        with open(out_path, "wb") as out:
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL)
            best = min(best, time.perf_counter() - t0)
    return best


def peak_rss_kib(cmd, out_path):
    # GNU time rather than wait4() from here: a child's ru_maxrss is floored by the
    # pre-exec RSS it inherits from this interpreter, which is larger than zstd's own.
    if TIME_BIN is None:
        return None
    rss_path = out_path + ".rss"
    with open(out_path, "wb") as out:
        subprocess.run([TIME_BIN, "-f", "%M", "-o", rss_path, *cmd], check=True, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL)
    with open(rss_path) as f:
        return int(f.read().split()[-1])


def run_pair(enc_cmd, dec_cmd, comp_path, dec_path, repeats):
    enc_s = timed(enc_cmd, comp_path, repeats)
    enc_rss = peak_rss_kib(enc_cmd, comp_path)
    dec_s = timed(dec_cmd, dec_path, repeats)
    dec_rss = peak_rss_kib(dec_cmd, dec_path)
    return dict(size=os.path.getsize(comp_path), enc_s=enc_s, dec_s=dec_s, enc_rss=enc_rss, dec_rss=dec_rss)


def fmt_rss(kib):
    return "n/a" if kib is None else f"{kib / 1024:.1f}"


def print_markdown(report):
    """README tables: the default preset against the baselines that differ from it materially."""
    codecs = ["zstd -3", "zstd -19", "xz -6", "zform -3"]
    rows = [(name, raw_len, dict(results)) for name, raw_len, results in report]
    print("\n| Corpus | Raw | " + " | ".join(f"`{c}`" for c in codecs) + " |")
    print("|---|---:|" + "---:|" * len(codecs))
    for name, raw_len, res in rows:
        cells = " | ".join(
            f"{res[c]['size'] / 2**20:.2f} MiB ({raw_len / res[c]['size']:.3g}x)" for c in codecs
        )
        print(f"| {name} | {raw_len / 2**20:.2f} MiB | {cells} |")

    def span(vals, digits):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "n/a"
        lo, hi = (f"{float(f'{v:.{digits}g}'):,g}" for v in (min(vals), max(vals)))
        return lo if lo == hi else f"{lo}–{hi}"

    # Throughput and memory as a range over the text corpora; the binary and control rows are not logs.
    text = [(raw_len, res) for name, raw_len, res in rows if not name.startswith(("sao-", "random-"))]
    print("\n| Codec | Compress MB/s | Decompress MB/s | Compress RSS MiB | Decompress RSS MiB |")
    print("|---|---:|---:|---:|---:|")
    for c in codecs:
        mbps = lambda k: span([raw_len / 1e6 / res[c][k] for raw_len, res in text], 2)
        rss = lambda k: span([res[c][k] and res[c][k] / 1024 for _, res in text], 2)
        print(f"| `{c}` | {mbps('enc_s')} | {mbps('dec_s')} | {rss('enc_rss')} | {rss('dec_rss')} |")


def main():
    ap = argparse.ArgumentParser(description="Benchmark zform against zstd and gzip.")
    ap.add_argument("--hdfs", help="Optional path to raw HDFS.log")
    ap.add_argument("--gharchive", help="Optional path to GH Archive .json or .json.gz")
    ap.add_argument("--slice-mb", type=int, default=8, help="Max MiB slice for external files (default: 8)")
    ap.add_argument("--external-only", action="store_true", help="Only run external --hdfs / --gharchive datasets")
    ap.add_argument("--repeats", type=int, default=3, help="Timed runs per codec and direction; the best is reported (default: 3)")
    ap.add_argument("--markdown", action="store_true", help="Also print the README tables")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(["cargo", "build", "--release", "--workspace"], cwd=repo_root, check=True)
    zform_bin = str(repo_root / "target" / "release" / "zform")
    max_b = args.slice_mb * 1024 * 1024

    datasets = []
    if args.hdfs and os.path.exists(args.hdfs):
        with open(args.hdfs, "rb") as f:
            datasets.append((f"hdfs-{args.slice_mb}m (real)", slice_lines(f.read(max_b + 1048576), max_b)))
    if args.gharchive and os.path.exists(args.gharchive):
        if args.gharchive.endswith(".gz"):
            buf = bytearray()
            with gzip.open(args.gharchive, "rb") as gz:
                while len(buf) < max_b + 1048576:
                    chunk = gz.read(262144)
                    if not chunk:
                        break
                    buf.extend(chunk)
            datasets.append((f"gharchive-{args.slice_mb}m (real)", slice_lines(bytes(buf), max_b)))
        else:
            with open(args.gharchive, "rb") as f:
                datasets.append((f"gharchive-{args.slice_mb}m (real)", slice_lines(f.read(max_b + 1048576), max_b)))

    if not args.external_only:
        datasets.append(("otel-hybrid-8m", make_otel_hybrid(8 * 1024 * 1024)))
        datasets.append(("sao-binary-28b", make_sao_binary(200_000)))
        datasets.append(("random-b64 (neg)", make_random_b64(3 * 1024 * 1024)))

    # (label, file extension, encode argv, decode argv); the input path is appended and output goes to stdout.
    # Every codec is held to one thread; recent zstd CLIs default to several.
    codecs = []
    if shutil.which("gzip"):
        codecs.append(("gzip -6", "gz", ["gzip", "-6", "-c"], ["gzip", "-d", "-c"]))
    if shutil.which("zstd"):
        for lvl in ("3", "19"):
            codecs.append((f"zstd -{lvl}", "zst", ["zstd", "--single-thread", f"-{lvl}", "-q", "-c"], ["zstd", "-d", "-q", "-c"]))
    if shutil.which("xz"):
        codecs.append(("xz -6", "xz", ["xz", "-T1", "-6", "-c"], ["xz", "-d", "-c"]))
    for label, extra in [("zform --lz4", ["--lz4"]), ("zform -3", ["-l", "3"]), ("zform --best", ["--best"])]:
        codecs.append((label, "zfm", [zform_bin, "pack", *extra], [zform_bin, "unpack"]))
    if TIME_BIN is None:
        print("note: GNU time not found; peak RSS is reported as n/a", file=sys.stderr)

    report = []
    print(f"{'Dataset':<22} | {'Codec':<14} | {'Raw MiB':>7} | {'Comp KiB':>9} | {'Ratio':>7} | {'Enc MB/s':>8} | {'Dec MB/s':>8} | {'Enc RSS':>7} | {'Dec RSS':>7}   (RSS in MiB)")
    print("-" * 116)

    with tempfile.TemporaryDirectory() as td:
        for name, raw in datasets:
            inp = os.path.join(td, "in.bin")
            with open(inp, "wb") as f:
                f.write(raw)
            sha = hashlib.sha256(raw).hexdigest()
            raw_mb = len(raw) / (1024 * 1024)
            results = []

            zf3_path = None
            for i, (label, ext, enc, dec) in enumerate(codecs):
                comp, restored = os.path.join(td, f"c{i}.{ext}"), os.path.join(td, f"d{i}.bin")
                r = run_pair(enc + [inp], dec + [comp], comp, restored, args.repeats)
                with open(restored, "rb") as f:
                    assert hashlib.sha256(f.read()).hexdigest() == sha, f"roundtrip mismatch on {name} ({label})"
                results.append((label, r))
                if label == "zform -3":
                    zf3_path = comp
                print(f"{name:<22} | {label:<14} | {raw_mb:>7.2f} | {r['size']/1024:>9.1f} | {len(raw)/r['size']:>6.2f}x | {len(raw)/1e6/r['enc_s']:>8.1f} | {len(raw)/1e6/r['dec_s']:>8.1f} | {fmt_rss(r['enc_rss']):>7} | {fmt_rss(r['dec_rss']):>7}", flush=True)
            report.append((name, len(raw), results))

            query = None
            if "gharchive" in name:
                query = ["-f", "id,type,actor.login,repo.name", "--where", "type=PushEvent"]
            elif "otel-hybrid" in name:
                query = ["-f", "ts,service,dur_us", "--where", "level=ERROR"]
            if zf3_path and query:
                t0 = time.perf_counter()
                p = subprocess.run([zform_bin, "select", zf3_path, *query, "--stats"], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)
                print(f"  -> {(time.perf_counter() - t0) * 1000:.1f} ms | {p.stderr.strip()}", flush=True)
            print("-" * 116, flush=True)

    if args.markdown and report:
        print_markdown(report)


if __name__ == "__main__":
    main()
