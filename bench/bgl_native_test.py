"""Fast differential frontend checks; build separately, then run with timeout 1s.

python3 bench/bgl_native_test.py --bin bench/bgl-native/target/release/bgl-native \
    --reference-bin /tmp/bgl_cm_rust --clock-bin target/release/examples/bgl_payload_probe
"""
import argparse
import contextlib
import io
import pathlib
import random
import subprocess
import tempfile
import time

import bgl_cm as reference


def run(binary, *args, success=True):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, timeout=0.3)
    assert (result.returncode == 0) == success, result.stderr.decode(errors="replace")
    return result


def row(i, body):
    node = b"R%02d-M%d-N%02d-J%02d" % (i % 3, i % 2, i % 4, i % 5)
    copy = node if i % 3 else b"different-node"
    header = b"- %d 2000.02.29 %s 2000-02-29-00.00.00.%06d %s RAS KERNEL INFO" % (
        951782400 - i % 3, node, i, copy
    )
    return header if body is None else header + b" " + body


def check(binary, reference_bin, clock_bin):
    bodies = [None, b"", b"% 100%% %d %08x", b"0x0 0X00AB 0xAbCd 0x1234567890123456",
              b"abcdef12 ABCDEF12 aBcDeF12 00000000123456789", b"x00000000_ abcdef12G z01234567",
              b"012345678901234567890123456789abcdef 1234567890123456789", b"\xff\x00 literal 123",
              b"-12 +009 12345678 0x abcdef1234_ 0Xfffffffffffffff", b"none! \r"]
    rng = random.Random(42)
    alphabet = b"abCDxX_ 0123456789%:-!"
    bodies += [bytes(rng.choices(alphabet, k=rng.randrange(1, 55))) for _ in range(30)]
    data = b"\n".join(row(i, body) for i, body in enumerate(bodies * 17 + [b"rare tail 001"]))
    with tempfile.TemporaryDirectory(prefix="bgl-native-test-") as tmp:
        root = pathlib.Path(tmp)
        src, actual, expected, restored = [root / p for p in ("in.log", "native.bcm", "python.bcm", "out.log")]
        src.write_bytes(data)
        for hybrid in (False, True):
            helper = clock_bin if hybrid else None
            flags = ["--hybrid"] if hybrid else []
            with contextlib.redirect_stdout(io.StringIO()):
                reference.pack(src, expected, reference_bin, helper)
            run(binary, "pack", src, actual, *flags)
            assert actual.read_bytes() == expected.read_bytes(), f"archive mismatch, hybrid={hybrid}"
            run(binary, "unpack", expected, restored)
            assert restored.read_bytes() == data
            with contextlib.redirect_stdout(io.StringIO()):
                reference.unpack(actual, restored, reference_bin, helper)
            assert restored.read_bytes() == data
        valid = row(0, b"valid") + b"\n"
        bad = [b"", b"too few fields", row(1, b"bad").replace(b"2000-02-29", b"2001-02-29"),
               row(1, b"bad").replace(b"951782399", b"0951782399"),
               row(1, b"bad").replace(b"951782399", b""),
               row(1, b"bad").replace(b"00.00.00.000001", b"24.00.00.000001"),
               row(1, b"bad").replace(b"2000.02.29", b"2000.02.28"),
               row(1, b"bad").replace(b".000001", b".00000x"),
               row(1, b"bad") + b"\n" + row(0, b"backwards")]
        for invalid in bad:
            src.write_bytes(valid + invalid if invalid else b"")
            run(binary, "pack", src, actual, success=False)
        src.write_bytes(row(0, b"bad").replace(b"951782400", b""))
        run(binary, "pack", src, actual, success=False)
        # Negative Unix seconds, changing offsets, and a trailing newline.
        data = b"- -1 1969.12.31 R0 1969-12-31-23.59.59.999999 R0 RAS KERNEL INFO 0x0001\n"
        src.write_bytes(data)
        run(binary, "pack", src, actual, "--hybrid")
        run(binary, "unpack", actual, restored)
        assert restored.read_bytes() == data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", required=True)
    p.add_argument("--reference-bin", required=True)
    p.add_argument("--clock-bin", required=True)
    args = p.parse_args()
    start = time.monotonic()
    check(args.bin, args.reference_bin, args.clock_bin)
    print(f"Native frontend differential checks passed in {time.monotonic() - start:.3f}s")
