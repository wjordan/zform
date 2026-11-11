#!/usr/bin/env python3
"""Compare byte-exact codecs on local public CloudTrail JSON, gzip, or ZIP files."""

import argparse
import gzip
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path


def read_source(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".json")]
            if len(names) != 1:
                raise ValueError(f"{path}: expected one JSON member, got {names}")
            return archive.read(names[0])
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes())
    return path.read_bytes()


def sample_records(records, max_bytes):
    # Sample across the file rather than biasing toward its first time window.
    indices = list(range(len(records)))
    random.Random(42).shuffle(indices)
    selected = []
    size = 0
    for index in indices:
        event = {k: v for k, v in records[index].items() if k not in ("should_sample", "attack_name")}
        line = (json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        if size + len(line) > max_bytes:
            break
        selected.append((index, event, line))
        size += len(line)
    if not selected:
        raise ValueError("sample limit is smaller than one event, or source has no events")
    selected.sort(key=lambda item: item[0])
    return b"".join(line for _, _, line in selected), [event for _, event, _ in selected]


def run_codec(encode, decode, compressed, restored, raw, timeout):
    started = time.perf_counter()
    with compressed.open("wb") as output:
        subprocess.run(encode, stdout=output, stderr=subprocess.PIPE, check=True, timeout=timeout)
    encode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with restored.open("wb") as output:
        subprocess.run(decode, stdout=output, stderr=subprocess.PIPE, check=True, timeout=timeout)
    decode_seconds = time.perf_counter() - started
    if hashlib.sha256(restored.read_bytes()).digest() != hashlib.sha256(raw).digest():
        raise ValueError(f"roundtrip mismatch: {encode}")
    size = compressed.stat().st_size
    return dict(bytes=size, ratio=len(raw) / size,
                encode_seconds=encode_seconds, decode_seconds=decode_seconds,
                encode_MB_s=len(raw) / 1e6 / encode_seconds,
                decode_MB_s=len(raw) / 1e6 / decode_seconds, sha256_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--sample-mib", type=int, default=8)
    parser.add_argument("--layout", choices=("ndjson", "native", "both"), default="ndjson",
                        help="Sampled NDJSON, complete original JSON bytes, or both")
    parser.add_argument("--timeout", type=float, default=30, help="Seconds per codec subprocess")
    parser.add_argument("--output", type=Path, default=Path("bench/results.json"))
    parser.add_argument("--compare-bin", type=Path, help="Additional record compressor with pack/unpack commands")
    parser.add_argument("--only-compare", action="store_true", help="Run only the additional compressor")
    args = parser.parse_args()
    if args.sample_mib <= 0 or args.timeout <= 0:
        parser.error("sample size and timeout must be positive")
    if args.only_compare and not args.compare_bin:
        parser.error("--only-compare requires --compare-bin")
    root = Path(__file__).resolve().parent.parent
    binary = root / "target/release/zform"
    if not binary.is_file():
        parser.error("build first: cargo build --release --workspace")
    codecs = [
        ("gzip -6", ["gzip", "-n", "-6", "-c"], ["gzip", "-d", "-c"]),
        ("zstd -3", ["zstd", "--single-thread", "-3", "-q", "-c"], ["zstd", "-d", "-q", "-c"]),
        ("zstd -19", ["zstd", "--single-thread", "-19", "-q", "-c"], ["zstd", "-d", "-q", "-c"]),
        ("xz -6", ["xz", "-T1", "-6", "-c"], ["xz", "-d", "-c"]),
        ("zform --lz4", [str(binary), "pack", "--lz4"], [str(binary), "unpack"]),
        ("zform -3", [str(binary), "pack", "-l", "3"], [str(binary), "unpack"]),
        ("zform --best", [str(binary), "pack", "--best"], [str(binary), "unpack"]),
    ]
    if args.only_compare:
        codecs = []
    if args.compare_bin:
        args.compare_bin = args.compare_bin.resolve()
        for level in (3, 6, 9):
            limit = ["--max-record-bytes", str(128 * 1024 * 1024)]
            codecs.append((f"{args.compare_bin.name} -{level}",
                           [str(args.compare_bin), "pack", "--level", str(level), *limit],
                           [str(args.compare_bin), "unpack", *limit]))
    for _, encode, _ in codecs:
        if not shutil.which(encode[0]):
            parser.error(f"missing executable: {encode[0]}")
    report = dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        tool_versions={tool: subprocess.check_output([tool, "--version"], text=True).splitlines()[0]
                       for tool in ("gzip", "zstd", "xz")},
        methodology="Single timed run, one thread, includes process startup and file I/O; "
                     "seed-42 record sample in source order; NDJSON removes research labels. "
                     "Each codec roundtrip must preserve its exact input bytes.",
        datasets=[],
    )
    if args.compare_bin:
        report["comparison"] = dict(binary=str(args.compare_bin),
                                  binary_sha256=hashlib.sha256(args.compare_bin.read_bytes()).hexdigest(),
                                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                   cwd=args.compare_bin.parent, text=True).strip(),
                                  max_record_bytes=128 * 1024 * 1024,
                                  training_and_chunk_options="defaults")
    for source in args.sources:
        original = read_source(source)
        records = json.loads(original)["Records"]
        variants = []
        if args.layout in ("ndjson", "both"):
            sampled, events = sample_records(records, args.sample_mib * 1024 * 1024)
            variants.append(("ndjson", sampled, events))
        if args.layout in ("native", "both"):
            variants.append(("native", original, records))
        for variant, raw, events in variants:
            dataset = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           source_events=len(records), variant=variant, events=len(events),
                           raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
                           event_sources=dict(Counter(event.get("eventSource") for event in events)),
                           unique_event_names=len({event.get("eventName") for event in events}), codecs={})
            report["datasets"].append(dataset)
            print(f"\n{source.name} / {variant}: {len(events):,}/{len(records):,} events, "
                  f"{len(raw) / 2**20:.3f} MiB", flush=True)
            print(f"{'Codec':<16} {'KiB':>10} {'Ratio':>8} {'Enc MB/s':>10} {'Dec MB/s':>10}", flush=True)
            with tempfile.TemporaryDirectory(prefix="cloudtrail-bench-") as temp:
                temp = Path(temp)
                input_path, compressed, restored = [temp / name for name in ("in", "packed", "restored")]
                input_path.write_bytes(raw)
                for name, encode, decode in codecs:
                    result = run_codec(encode + [str(input_path)], decode + [str(compressed)],
                                       compressed, restored, raw, args.timeout)
                    dataset["codecs"][name] = result
                    print(f"{name:<16} {result['bytes'] / 1024:>10.1f} {result['ratio']:>7.2f}x "
                          f"{result['encode_MB_s']:>10.1f} {result['decode_MB_s']:>10.1f}", flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
