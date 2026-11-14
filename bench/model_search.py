"""Manual benchmark of selective field partitioning against context mixing.

Build: cargo build --release -p zform-core --example partition_probe
Run: python3 bench/model_search.py --output results.json FILE...
Compares normal packing, an unchanged whole-input profile, and the sampled partition search.
The prototype reads the whole input to choose a fixed profile; it is not a streaming planner.
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
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--bin", type=pathlib.Path, default="target/release/zform")
    parser.add_argument("--probe", type=pathlib.Path, default="target/release/examples/partition_probe")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    report = {"method": "Sequential single-thread warm-cache process runs; median times include input I/O, profile search, encoding and index. All variants use context mixing. Verify deterministic bytes, exact SHA256 roundtrips, and grep against raw records.",
              "platform": platform.platform(), "binary_sha256": digest(args.bin),
              "probe_sha256": digest(args.probe), "datasets": []}

    def run(command, grep=False):
        start = time.perf_counter()
        result = subprocess.run(list(map(str, command)), capture_output=True, timeout=args.timeout)
        if result.returncode not in ([0, 1] if grep else [0]):
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return time.perf_counter() - start, result

    with tempfile.TemporaryDirectory(prefix="zform-model-bench-") as temp:
        temp = pathlib.Path(temp)
        for source in args.inputs:
            dataset = {"source": str(source), "raw_bytes": source.stat().st_size,
                       "raw_sha256": digest(source), "variants": []}
            expected_query = b"".join(v for v in source.read_bytes().splitlines(keepends=True) if b"onExtend" in v).replace(b"\r\n", b"\n")
            for mode in ("baseline", "none", "adaptive-fast"):
                out, restored = temp / "out.zfm", temp / "restored"
                command = ([args.bin, "pack", "--context-mixing", source, "-o", out]
                           if mode == "baseline" else [args.probe, source, out, mode])
                encodes, decodes, checksum = [], [], None
                for _ in range(args.repeats):
                    elapsed, encoded = run(command)
                    encodes.append(elapsed)
                    new_hash = digest(out)
                    assert checksum in (None, new_hash), "nondeterministic archive"
                    checksum = new_hash
                    elapsed, _ = run([args.bin, "unpack", out, "-o", restored])
                    decodes.append(elapsed)
                    assert digest(restored) == dataset["raw_sha256"], "roundtrip mismatch"
                _, got = run([args.bin, "grep", "onExtend", out], grep=True)
                assert got.stdout.replace(b"\r\n", b"\n") == expected_query, "grep mismatch"
                result = {"mode": mode, "bytes": out.stat().st_size, "encode_seconds": statistics.median(encodes),
                          "decode_seconds": statistics.median(decodes), "encode_samples": encodes, "decode_samples": decodes,
                          "compressed_sha256": checksum, "roundtrip": True, "grep": True,
                          "choices": encoded.stderr.decode(errors="replace").splitlines() if mode == "adaptive-fast" else []}
                dataset["variants"].append(result)
                print(f"{source.name} {mode}: {result['bytes']} bytes, encode {result['encode_seconds']:.3f}s, decode {result['decode_seconds']:.3f}s", flush=True)
            report["datasets"].append(dataset)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
