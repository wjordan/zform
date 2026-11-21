#!/usr/bin/env python3
"""Publish README size tables when a measured archive changes by at least 1%."""

import argparse
import json
from pathlib import Path

from run import COMPARISON, preset_table, size_table

EXPECTED = {
    "HDFS DataNode logs": (*COMPARISON, "zform --best", "zform HDFS predictors"),
    "GitHub Archive NDJSON": (*COMPARISON, "zform --best", "zform ordered", "zform --best ordered"),
    "Silesia sao": (*COMPARISON, "zform --best"),
}

CORPUS_HASHES = {
    "HDFS DataNode logs": "0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52",
    "GitHub Archive NDJSON": "51796a11afd3645a5c8d60ea939ec94ad550653750003d04c3fad886589f2cf7",
    "Silesia sao": "c2d0ea2cc59d4c21b7fe43a71499342a00cbe530a1d5548770e91ecd6214adcc",
}


def snapshot(results):
    if [d["name"] for d in results] != list(EXPECTED):
        raise ValueError("expected the full HDFS, GitHub Archive and sao comparison")
    for dataset in results:
        if dataset["sha256"] != CORPUS_HASHES[dataset["name"]]:
            raise ValueError(f"unexpected benchmark input for {dataset['name']}")
    return [
        {
            "name": d["name"],
            "raw_bytes": d["raw_bytes"],
            "sha256": d["sha256"],
            "codecs": {c: {"size": d["codecs"][c]["size"]} for c in EXPECTED[d["name"]]},
        }
        for d in results
    ]


def changes(before, after):
    deltas = []
    for old, new in zip(before, after, strict=True):
        for key in ("name", "raw_bytes", "sha256"):
            if old[key] != new[key]:
                raise ValueError(f"benchmark input changed for {new['name']}")
        for codec, result in new["codecs"].items():
            size = old["codecs"][codec]["size"]
            diff = result["size"] - size
            if abs(diff) * 100 >= size:
                deltas.append(f"- {new['name']}, `{codec}` archive {diff / size:+.1%}")
    return deltas


def replace_table(markdown, name, table):
    start = f"<!-- bench:{name} -->\n"
    end = f"<!-- /bench:{name} -->"
    head, _, tail = markdown.partition(start)
    _, marker, rest = tail.partition(end)
    if not marker:
        raise ValueError(f"missing README table markers for {name}")
    return head + start + table + "\n" + marker + rest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    published = root / "bench" / "readme-sizes.json"
    readme = root / "README.md"

    new = snapshot(json.loads(args.results.read_text()))
    if published.exists():
        old = json.loads(published.read_text())
        deltas = changes(old, new)
        if not deltas:
            print("README archive sizes are within 1% of the published results.")
            return
        print("Archive sizes that changed by at least 1%\n\n" + "\n".join(deltas))
    else:
        print("Recording the first CI archive sizes.")

    markdown = replace_table(readme.read_text(), "comparison", size_table(new, COMPARISON))
    markdown = replace_table(markdown, "presets", preset_table(new))
    readme.write_text(markdown)
    published.write_text(json.dumps(new, indent=2) + "\n")


if __name__ == "__main__":
    main()
