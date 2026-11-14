"""Manual A/B benchmark for the optional context-mixing payload pass.

Build first: cargo build --release -p zform-cli
Run: python3 bench/context_mixing.py --repeats 3 --output results.json FILE...
Pass --best to compare the high-density preset with and without context mixing.
Every variant is decoded and checked against the input SHA256.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import tempfile
import time


def digest(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--bin", default="target/release/zform", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--timeout", default=20, type=float)
    parser.add_argument("--best", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    report = {
        "method": "Sequential warm-cache process timings, including training, archive/index writing and I/O; medians over encode repeats, one decode per variant. All inputs kept in their original order and spelling. Size includes the query index. Decode SHA256 checked and repeated encodes checked for deterministic bytes.",
        "platform": platform.platform(),
        "binary_sha256": digest(args.bin),
        "preset": "best" if args.best else "default",
        "datasets": [],
    }
    with tempfile.TemporaryDirectory(prefix="zform-context-bench-") as tmp:
        tmp = pathlib.Path(tmp)
        for src in args.inputs:
            expected = digest(src)
            dataset = {"source": str(src), "raw_bytes": src.stat().st_size, "raw_sha256": expected, "variants": []}
            for mixing in (False, True):
                compressed, restored = tmp / "out.zfm", tmp / "restored"
                flags = (["--best"] if args.best else []) + (["--context-mixing"] if mixing else [])
                command = [str(args.bin), "pack", *flags, str(src), "-o", str(compressed)]
                times, encoded_hash = [], None
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    subprocess.run(command, check=True, capture_output=True, timeout=args.timeout)
                    times.append(time.perf_counter() - start)
                    checksum = digest(compressed)
                    assert encoded_hash in (None, checksum), "nondeterministic archive"
                    encoded_hash = checksum
                start = time.perf_counter()
                subprocess.run([str(args.bin), "unpack", str(compressed), "-o", str(restored)], check=True, capture_output=True, timeout=args.timeout)
                decode = time.perf_counter() - start
                assert digest(restored) == expected, "roundtrip mismatch"
                dataset["variants"].append({"context_mixing": mixing, "bytes": compressed.stat().st_size,
                    "encode_seconds": statistics.median(times), "encode_samples": times, "decode_seconds": decode,
                    "compressed_sha256": encoded_hash, "roundtrip": True})
                restored.unlink()
            base, mixed = dataset["variants"]
            dataset["size_reduction_percent"] = 100 * (1 - mixed["bytes"] / base["bytes"])
            dataset["encode_time_factor"] = mixed["encode_seconds"] / base["encode_seconds"]
            dataset["decode_time_factor"] = mixed["decode_seconds"] / base["decode_seconds"]
            report["datasets"].append(dataset)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{src.name}: {base['bytes']} -> {mixed['bytes']} bytes, {dataset['size_reduction_percent']:.2f}% smaller; encode {dataset['encode_time_factor']:.2f}x, decode {dataset['decode_time_factor']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
