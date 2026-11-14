"""Fast codec roundtrips: timeout 1s python3 bench/bgl_cm_test.py --bin /tmp/bgl_cm.

Optionally pass --reference-bin to check byte compatibility with an earlier build.
Pass --wrapper to also check timestamp parsing and byte-exact text reconstruction.
Pass --clock-bin to check hybrid archives and signed timestamp deltas.
Build binaries separately; direct codec calls have a 250 ms timeout.
"""

import argparse
import contextlib
import importlib.util
import io
import pathlib
import struct
import subprocess
import tempfile
import time


def run(binary, *args):
    subprocess.run(
        [binary, *map(str, args)], check=True, capture_output=True, timeout=0.25
    )


def check(binary, reference, midplanes, numeric):
    with tempfile.TemporaryDirectory(prefix="bgl-cm-test-") as work:
        root = pathlib.Path(work)
        src, dst = root / "src", root / "dst"
        src.mkdir()
        dst.mkdir()

        def column(name, values, kind, dictionary=False):
            data = struct.pack("<" + kind * len(values), *values)
            (src / name).write_bytes(data)
            if dictionary:
                (dst / name).write_bytes(data)

        nodes = midplanes * 4
        column("nodecard.u32", [n // 2 for n in range(nodes)], "I", True)
        column("nodemid.u32", [n // 4 for n in range(nodes)], "I", True)
        column("noderack.u32", [n // 8 for n in range(nodes)], "I", True)
        column("nodepos.u32", [n % 2 for n in range(nodes)], "I", True)
        lanes = [i % 3 if numeric else 0 for i in range(70)]
        column("lanes.u32", lanes, "I", True)
        selectors = [(i * 17 + i // 13) % len(lanes) for i in range(192)]
        column("sel.u32", selectors, "I")
        column("node.u32", [i if i < nodes else (i * 7 + i // 11) % nodes for i in range(192)], "I")
        column("head.u32", [i % 11 for i in range(192)], "I")
        ticks, nums, tick = [], [], 10**15
        for i, selector in enumerate(selectors):
            tick += [0, 1, 1024, 1 << 40][i % 4]
            ticks.append(tick)
            for j in range(lanes[selector]):
                nums.append([0, i * 31 + j, 10**17 + j, i % 7][i % 4])
        column("ticks.i64", ticks, "q")
        column("nums.i64", nums, "q")
        packed = root / "out.cm"
        run(binary, "pack", src, packed)
        run(binary, "unpack", packed, dst)
        for name in ("node.u32", "sel.u32", "head.u32", "ticks.i64", "nums.i64"):
            assert (src / name).read_bytes() == (dst / name).read_bytes(), name
        if reference:
            old = root / "reference.cm"
            run(reference, "pack", src, old)
            assert old.read_bytes() == packed.read_bytes(), "compressed bytes changed"


def check_clock_payload(binary):
    with tempfile.TemporaryDirectory(prefix="bgl-clock-test-") as work:
        root = pathlib.Path(work)
        src, packed, restored = root / "in", root / "graph", root / "out"
        data = struct.pack("<8q", -(1 << 63), (1 << 63) - 1, 0, 1, -1, 1024, -1024, 0)
        src.write_bytes(data)
        run(binary, "pack", src, packed)
        run(binary, "unpack", packed, restored)
        assert restored.read_bytes() == data, "signed/wrapping clock deltas changed"


def check_wrapper(binary, clock_binary=None):
    spec = importlib.util.spec_from_file_location("bgl_cm", pathlib.Path(__file__).with_name("bgl_cm.py"))
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    assert codec.second_of(b"2000-02-29-00.00.00") == 951782400
    assert codec.second_of(b"1969-12-31-23.59.59") == -1
    for pre in [b"2001-02-29-00.00.00", b"2006-01-01-24.00.00", b"2006-01-01-00:00:00"]:
        try:
            codec.second_of(pre)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted noncanonical clock {pre!r}")

    with tempfile.TemporaryDirectory(prefix="bgl-cm-wrapper-test-") as work:
        root = pathlib.Path(work)
        src, packed, restored = root / "in.log", root / "out.bcm", root / "out.log"
        lines = []
        for i in range(24):
            epoch = [1136073600, 1136073601, 1136073600][i % 3]
            lines.append(
                b"- %d 2006.01.01 R00-M0-N00-J00 2006-01-01-00.00.00.%06d "
                b"R00-M0-N00-J00 RAS KERNEL INFO value %04d" % (epoch, i, i)
            )
        # Distinguish an absent message, an empty message, and an unterminated final row.
        header = lines[-1].rsplit(b" value ", 1)[0]
        data = b"\n".join(lines + [header, header + b" ", header + b" tail 007"])
        src.write_bytes(data)
        with contextlib.redirect_stdout(io.StringIO()):
            codec.pack(src, packed, binary, clock_binary)
            codec.unpack(packed, restored, binary, clock_binary)
        assert restored.read_bytes() == data
        if clock_binary:
            try:
                codec.unpack(packed, restored, binary)
            except ValueError as error:
                assert "requires --clock-bin" in str(error)
            else:
                raise AssertionError("accepted hybrid archive without clock decoder")
        # A different spelling within an already-cached second must still be validated.
        src.write_bytes(lines[0] + b"\n" + lines[1].replace(b"1136073601", b"01136073601"))
        try:
            codec.pack(src, packed, binary)
        except ValueError as error:
            assert "epoch field not canonical" in str(error)
        else:
            raise AssertionError("accepted noncanonical epoch after a cache hit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--reference-bin")
    parser.add_argument("--wrapper", action="store_true")
    parser.add_argument("--clock-bin")
    args = parser.parse_args()
    started = time.monotonic()
    for midplanes, numeric in [(1, False), (3, True), (16, True)]:
        check(args.bin, args.reference_bin, midplanes, numeric)
    if args.wrapper:
        check_wrapper(args.bin)
    if args.clock_bin:
        check_clock_payload(args.clock_bin)
        check_wrapper(args.bin, args.clock_bin)
    print(f"Codec checks passed in {time.monotonic() - started:.3f}s")
