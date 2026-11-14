"""Manual A/B benchmark for writer-side HDFS relations, with and without context mixing.

Build: cargo build --release -p zform-cli
Run: python3 bench/hdfs_predictors.py --output results.json FILE...
Optional --baseline-bin checks unchanged default output and older-reader compatibility.
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
    parser.add_argument("--bin", type=pathlib.Path, default="target/release/zform")
    parser.add_argument("--baseline-bin", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--best", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")

    def run(binary, *arguments):
        start = time.perf_counter()
        proc = subprocess.run([str(binary), *map(str, arguments)], capture_output=True, timeout=args.timeout)
        # grep uses 1 for no matches.
        if proc.returncode not in ([0, 1] if arguments[0] == "grep" else [0]):
            raise RuntimeError(proc.stderr.decode(errors="replace"))
        return time.perf_counter() - start, proc.stdout

    report = {
        "method": "Sequential warm-cache process timings, including training, index writing and I/O. Median encode/decode timings. SHA256 roundtrip, deterministic repeated encode, and grep output checked for every variant.",
        "platform": platform.platform(), "binary_sha256": digest(args.bin),
        "preset": "best" if args.best else "default", "datasets": [],
    }
    if args.baseline_bin:
        report["baseline_binary_sha256"] = digest(args.baseline_bin)
    with tempfile.TemporaryDirectory(prefix="zform-hdfs-bench-") as temp:
        temp = pathlib.Path(temp)
        for source in args.inputs:
            dataset = {"source": str(source), "raw_bytes": source.stat().st_size,
                       "raw_sha256": digest(source), "variants": []}
            expected_query = b"".join(line for line in source.read_bytes().splitlines(keepends=True) if b"part-" in line)
            for mixing in (False, True):
                baseline_hash = None
                for hdfs in (False, True):
                    flags = (["--best"] if args.best else []) + (["--context-mixing"] if mixing else []) + (["--hdfs-predictors"] if hdfs else [])
                    archive, restored = temp / "out.zfm", temp / "restored"
                    encodes, decodes, encoded_hash = [], [], None
                    for _ in range(args.repeats):
                        elapsed, _ = run(args.bin, "pack", *flags, source, "-o", archive)
                        encodes.append(elapsed)
                        checksum = digest(archive)
                        assert encoded_hash in (None, checksum), "nondeterministic archive"
                        encoded_hash = checksum
                        elapsed, _ = run(args.bin, "unpack", archive, "-o", restored)
                        decodes.append(elapsed)
                        assert digest(restored) == dataset["raw_sha256"], "roundtrip mismatch"
                    _, query = run(args.bin, "grep", "part-", archive)
                    # CLI emits LF-delimited matches even when source framing is CRLF.
                    assert query.replace(b"\r\n", b"\n") == expected_query.replace(b"\r\n", b"\n"), "grep mismatch"
                    result = {"hdfs_predictors": hdfs, "context_mixing": mixing,
                              "bytes": archive.stat().st_size, "compressed_sha256": encoded_hash,
                              "encode_seconds": statistics.median(encodes), "encode_samples": encodes,
                              "decode_seconds": statistics.median(decodes), "decode_samples": decodes,
                              "roundtrip": True, "grep": True}
                    if not hdfs:
                        baseline_hash = encoded_hash
                    result["same_as_disabled"] = baseline_hash == encoded_hash
                    if args.baseline_bin:
                        run(args.baseline_bin, "unpack", archive, "-o", restored)
                        assert digest(restored) == dataset["raw_sha256"], "older-reader roundtrip mismatch"
                        result["older_reader_roundtrip"] = True
                        if not hdfs:
                            run(args.baseline_bin, "pack", *flags, source, "-o", temp / "before.zfm")
                            assert digest(temp / "before.zfm") == encoded_hash, "disabled feature changed output"
                            result["unchanged_default"] = True
                    dataset["variants"].append(result)
                    print(f"{source.name} cm={mixing} hdfs={hdfs}: {result['bytes']} bytes, encode {result['encode_seconds'] * 1000:.2f} ms, decode {result['decode_seconds'] * 1000:.2f} ms", flush=True)
            report["datasets"].append(dataset)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
