"""Manual archive ablation of RankedSum; production sources and format stay unchanged.

Build an instrumented copy of the pinned pre-extension revision:
  python3 bench/ranked_sum_ablation.py build --work /tmp/ranked-ablation
Compare full indexed archives, with exact roundtrips and grep checks:
  python3 bench/ranked_sum_ablation.py run --work /tmp/ranked-ablation HDFS.log

The copy adds benchmark-only modes selected by ZFORM_RANKED_ABLATION. All
replacements use the same inferred datanodes for a given record order. The
physical-order variant also changes stored rows. SortIndices and OrderedSum
are experimental nodes in the copy; archives selecting them need its reader.
"""

import argparse
import hashlib
import io
import json
import os
import pathlib
import tarfile
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODES = ("ranked", "off", "keyed-row", "keyed-sorted", "scan-grouped", "keyed-argsort", "replica-row", "replica-argsort",
         "replica-split", "replica-ordered", "replica-ordered-split", "replica-physical", "replica-ordered-zero")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"source changed; expected exactly one occurrence of {old!r}")
    return source.replace(old, new, 1)


def build(args):
    dst = args.work / "src"
    if dst.exists():
        raise ValueError(f"source copy already exists: {dst}; use a new work directory")
    dst.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", args.revision], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(
        ["git", "archive", revision, "Cargo.toml", "Cargo.lock", "README.md", "crates"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(dst, filter="data")
    path = dst / "crates/zform-core/src/packs/hdfs.rs"
    source = path.read_text()
    base_hdfs_sha256 = digest(path)
    source = replace_once(source,
        '    let picks = key.and_then(|key| deletions(ctx, key, &marked, &mut deltas));',
        '''    let picks = if deletion_mode() == DeletionMode::Off {
        None
    } else {
        key.and_then(|key| deletions(ctx, key, &marked, &mut deltas))
    };
    if let Some(choice) = &picks {
        prepare_deletions(choice, &mut deltas, deletion_mode());
    }''')
    start = source.index('    if let (Some(picks), Some(key)) = (picks, key) {')
    end = source.index('    let b = Builder::new(1);', start)
    ranked = source[start:end]
    ranked = replace_once(ranked, '(picks, key) {', '(picks, key) {\n        let picks = picks.picks;')
    source = source[:start] + '''    if matches!(deletion_mode(), DeletionMode::KeyedArgsort | DeletionMode::ReplicaArgsort
        | DeletionMode::ReplicaSplit | DeletionMode::ReplicaOrdered | DeletionMode::ReplicaOrderedSplit | DeletionMode::ReplicaOrderedZero) {
        if let (Some(choice), Some(key)) = (&picks, key) {
            let split = matches!(deletion_mode(), DeletionMode::ReplicaSplit | DeletionMode::ReplicaOrderedSplit | DeletionMode::ReplicaOrderedZero)
                .then_some((sel, &sels[..], recency.len()));
            let ordered = matches!(deletion_mode(), DeletionMode::ReplicaOrdered | DeletionMode::ReplicaOrderedSplit | DeletionMode::ReplicaOrderedZero);
            t = argsort_deletion_graph(&mut gb, t, choice, key, deletion_mode() != DeletionMode::KeyedArgsort, split, ordered);
        }
    } else if matches!(deletion_mode(), DeletionMode::ReplicaRow | DeletionMode::ReplicaPhysical) {
        if let (Some(choice), Some(key)) = (&picks, key) {
            let keys = replica_keys(&mut gb, choice, key, None);
            t = gb.push(GraphNode::KeyedSum { keys, deltas: t });
        }
    } else if deletion_mode() != DeletionMode::Ranked {
        if let Some(choice) = &picks {
            t = simple_deletion_graph(&mut gb, t, choice, deletion_mode());
        }
    } else {
''' + ranked + '    }\n' + source[end:]
    source = replace_once(source, '        Some(order)\n', '''        if deletion_mode() == DeletionMode::ReplicaPhysical {
            order_deletions(recs, &mut order);
        }
        Some(order)
''')
    source = replace_once(source, ') -> Option<Vec<i64>> {\n    let tables = ctx.tables.filter(|t| t.len() == 2)?;',
        ') -> Option<DeletionChoice> {\n    let tables = ctx.tables.filter(|t| t.len() == 2)?;')
    source = replace_once(source,
        '    let (mut picks, mut steps) = (vec![0i64; marked.len()], deltas.to_vec());',
        '''    let targets = deltas.to_vec();
    let mut chosen_keys = vec![0i64; marked.len()];
    let mut chosen_slots = vec![-1i64; marked.len()];
    let mut chosen_order = Vec::new();
    let (mut picks, mut steps) = (vec![0i64; marked.len()], deltas.to_vec());''')
    source = replace_once(source, '            picks[i] = at as i64 + 1;',
        '''            picks[i] = at as i64 + 1;
            chosen_keys[i] = k as i64 + 1;
            chosen_slots[i] = (from..from + REPLICAS)
                .position(|c| ips.get(c).and_then(|v| state.key_id(v)) == Some(k))? as i64;
            chosen_order.push(i);''')
    source = replace_once(source, '    Some(picks)\n}',
        '    Some(DeletionChoice { picks, keys: chosen_keys, order: chosen_order, targets, slots: chosen_slots })\n}')
    source = replace_once(source,
        '        assert!(graph.nodes.iter().any(|n| matches!(n, GraphNode::RankedSum { .. })));',
        '''        assert_eq!(graph.nodes.iter().any(|n| matches!(n, GraphNode::RankedSum { .. })),
            deletion_mode() == DeletionMode::Ranked);''')
    start = source.index('    fn deletions_step_on_their_datanodes_clocks() {')
    at = source.index('        let cells = |v: &[Vec<u8>]| {', start)
    source = source[:at] + '''        // Missing table keys and malformed times must remain ordinary rows or patches.
        for (variant, block, time) in [
            (2, b"999999".to_vec(), hms(6100)),
            (1, b"999998".to_vec(), hms(6200)),
            (1, keys[0].clone(), b"250000".to_vec()),
        ] {
            vids.push(variant);
            cols[1].push(block);
            cols[2].push(b"081109".to_vec());
            cols[3].push(time);
        }
''' + source[at:]
    source += '\n' + (ROOT / 'bench/ranked_sum_ablation.rs').read_text()
    path.write_text(source)
    add_sort_indices(dst)
    add_ordered_sum(dst)
    metadata = {
        "commit": revision,
        "overlay_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "base_hdfs_sha256": base_hdfs_sha256,
        "benchmark_python_sha256": digest(pathlib.Path(__file__)),
        "benchmark_rust_sha256": digest(ROOT / "bench/ranked_sum_ablation.rs"),
    }
    (args.work / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run(["cargo", "build", "--release", "-p", "zform-cli", "--manifest-path",
                    str(dst / "Cargo.toml")], check=True, timeout=60)
    compiled = subprocess.check_output(["cargo", "test", "--release", "-p", "zform-core", "--lib",
        "--no-run", "--message-format=json", "--manifest-path", str(dst / "Cargo.toml")], timeout=40)
    tests = [json.loads(line) for line in compiled.splitlines()]
    binary = next(t["executable"] for t in tests if t.get("executable") and t.get("profile", {}).get("test"))
    for mode in MODES:
        subprocess.run([binary, "packs::hdfs::", "--test-threads=8"], check=True, capture_output=True,
                       env=dict(os.environ, ZFORM_RANKED_ABLATION=mode), timeout=1)
    subprocess.run([binary, "payload::tests::", "--test-threads=8"], check=True, capture_output=True, timeout=1)
    print(f"Focused payload tests and HDFS fixtures passed in all {len(MODES)} modes.")


def add_sort_indices(dst):
    """One experimental primitive: stable argsort of an integer stream, no domain state."""
    path = dst / "crates/zform-core/src/payload/graph.rs"
    source = path.read_text()
    source = replace_once(source, "impl<'a> Ranked<'a> {", """impl<'a> Ranked<'a> {
    pub(crate) fn key_id(&self, value: &[u8]) -> Option<u32> {
        self.ids.get(value).copied()
    }
""")
    source = replace_once(source, 'pub enum GraphNode {',
        'pub enum GraphNode {\n    SortIndices { values: u16 },')
    source = replace_once(source, '            Self::CellIds { cells } => f(*cells),',
        '            Self::SortIndices { values } => f(*values),\n            Self::CellIds { cells } => f(*cells),')
    source = replace_once(source, '            Self::CellIds { cells } => Self::CellIds { cells: f(*cells) },',
        '            Self::SortIndices { values } => Self::SortIndices { values: f(*values) },\n            Self::CellIds { cells } => Self::CellIds { cells: f(*cells) },')
    source = replace_once(source, '        GraphNode::ExtVariant => out.push(26),',
        '''        GraphNode::SortIndices { values } => {
            out.push(30);
            put_uvarint(out, *values as u64);
        }
        GraphNode::ExtVariant => out.push(26),''')
    source = replace_once(source, '        26 => Ok(GraphNode::ExtVariant),',
        '''        30 => Ok(GraphNode::SortIndices { values: check_ref(cur.uvar()?)? }),
        26 => Ok(GraphNode::ExtVariant),''')
    source = replace_once(source, '                GraphNode::KeyedSum { keys, deltas } => Val::Ints(keyed_sum(',
        '''                GraphNode::SortIndices { values } => {
                    let source = *values as usize;
                    let values = as_ints(&vals[source])?;
                    budget.check_alloc(values.len().saturating_mul(8), "SortIndices")?;
                    let order = if matches!(nodes[source], GraphNode::SortIndices { .. }) {
                        let mut inverse = vec![0i64; values.len()];
                        for (i, &v) in values.iter().enumerate() {
                            inverse[v as usize] = i as i64;
                        }
                        inverse
                    } else {
                        let mut order: Vec<i64> = (0..values.len() as i64).collect();
                        order.sort_unstable_by_key(|&i| (values[i as usize], i));
                        order
                    };
                    Val::Ints(order)
                }
                GraphNode::KeyedSum { keys, deltas } => Val::Ints(keyed_sum(''')
    path.write_text(source)
    path = dst / "crates/zform-core/src/inspect.rs"
    source = replace_once(path.read_text(), '            GraphNode::ExtVariant => "Variant".into(),',
        '            GraphNode::SortIndices { .. } => "SortIndices".into(),\n            GraphNode::ExtVariant => "Variant".into(),')
    path.write_text(source)


def add_ordered_sum(dst):
    """Compare an ordered scan with the separate sort/gather/scan composition."""
    path = dst / "crates/zform-core/src/payload/graph.rs"
    source = path.read_text()
    source = replace_once(source, 'pub enum GraphNode {',
        'pub enum GraphNode {\n    OrderedSum { keys: u16, deltas: u16, order: u16, zero: bool },')
    source = replace_once(source, '            Self::CellIds { cells } => f(*cells),',
        '''            Self::OrderedSum { keys, deltas, order, .. } => {
                [*keys, *deltas, *order].into_iter().for_each(f);
            }
            Self::CellIds { cells } => f(*cells),''')
    source = replace_once(source, '            Self::CellIds { cells } => Self::CellIds { cells: f(*cells) },',
        '''            Self::OrderedSum { keys, deltas, order, zero } => Self::OrderedSum {
                keys: f(*keys), deltas: f(*deltas), order: f(*order), zero: *zero,
            },
            Self::CellIds { cells } => Self::CellIds { cells: f(*cells) },''')
    source = replace_once(source, '        GraphNode::ExtVariant => out.push(26),',
        '''        GraphNode::OrderedSum { keys, deltas, order, zero } => {
            out.push(31);
            for r in [keys, deltas, order] { put_uvarint(out, *r as u64); }
            out.push(u8::from(*zero));
        }
        GraphNode::ExtVariant => out.push(26),''')
    source = replace_once(source, '        26 => Ok(GraphNode::ExtVariant),',
        '''        31 => Ok(GraphNode::OrderedSum {
            keys: check_ref(cur.uvar()?)?, deltas: check_ref(cur.uvar()?)?, order: check_ref(cur.uvar()?)?,
            zero: match cur.u8()? { 0 => false, 1 => true, _ => return Err(Error::Format("OrderedSum seed flag".into())) },
        }),
        26 => Ok(GraphNode::ExtVariant),''')
    source = replace_once(source, '                GraphNode::KeyedSum { keys, deltas } => Val::Ints(keyed_sum(',
        '''                GraphNode::OrderedSum { keys, deltas, order, zero } => Val::Ints(ordered_sum(
                    as_ints(&vals[*keys as usize])?, as_ints(&vals[*deltas as usize])?,
                    as_ints(&vals[*order as usize])?, *zero, budget,
                )?),
                GraphNode::KeyedSum { keys, deltas } => Val::Ints(keyed_sum(''')
    source += '''
fn ordered_sum(keys: &[i64], deltas: &[i64], order: &[i64], zero: bool, budget: &DecodeBudget) -> Result<Vec<i64>> {
    if keys.len() != deltas.len() || keys.len() != order.len() {
        return Err(Error::Format("OrderedSum length mismatch".into()));
    }
    let max = keys.iter().copied().max().unwrap_or(0);
    if keys.iter().any(|&k| k < 0) || max as u64 > keys.len() as u64 {
        return Err(Error::Format("OrderedSum key out of range".into()));
    }
    budget.check_alloc((max as usize + 1 + keys.len()).saturating_mul(16), "OrderedSum")?;
    let mut rows: Vec<usize> = (0..keys.len()).filter(|&i| keys[i] != 0).collect();
    rows.sort_unstable_by_key(|&i| (order[i], i));
    if zero {
        let mut state = vec![0i64; max as usize + 1];
        let mut out = deltas.to_vec();
        for i in rows {
            let s = &mut state[keys[i] as usize];
            *s = s.wrapping_add(deltas[i]);
            out[i] = *s;
        }
        return Ok(out);
    }
    let (mut state, mut last) = (vec![None::<i64>; max as usize + 1], 0i64);
    let mut out = deltas.to_vec();
    for i in rows {
        let s = &mut state[keys[i] as usize];
        last = s.unwrap_or(last).wrapping_add(deltas[i]);
        *s = Some(last);
        out[i] = last;
    }
    Ok(out)
}
'''
    path.write_text(source)
    path = dst / "crates/zform-core/src/inspect.rs"
    source = replace_once(path.read_text(), '            GraphNode::ExtVariant => "Variant".into(),',
        '            GraphNode::OrderedSum { .. } => "OrderedSum".into(),\n            GraphNode::ExtVariant => "Variant".into(),')
    path.write_text(source)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    binary = args.work / "src/target/release/zform"
    report = {
        "build": json.loads((args.work / "build.json").read_text()),
        "method": "Sequential warm-cache process timings; median of repeats; one thread; indexed archives; repeated encodes checked for determinism when repeats > 1; every variant SHA256 roundtripped; grep Deleting block checked against source order with CLI line-ending normalization.",
        "level": args.level, "context_mixing": not args.no_mixing,
        "repeats": args.repeats, "datasets": [],
    }
    output = args.output or args.work / "results.json"
    artifacts = args.work / output.stem
    artifacts.mkdir(parents=True, exist_ok=True)
    for src in args.inputs:
        raw_hash = digest(src)
        expected_grep = hashlib.sha256()
        with src.open("rb") as stream:
            for line in stream:
                if b"Deleting block" in line:
                    expected_grep.update(line.rstrip(b"\r\n") + b"\n")
        dataset = {"source": str(src), "raw_bytes": src.stat().st_size,
                   "raw_sha256": raw_hash, "variants": []}
        for mode in args.modes:
            env = dict(os.environ, ZFORM_RANKED_ABLATION=mode)
            stem = f"{len(report['datasets'])}-{mode}"
            archive = artifacts / f"{stem}.zfm"
            restored = artifacts / f"{stem}.restored"
            query = artifacts / f"{stem}.grep"
            command = [str(binary), "pack", "--threads", "1", "-l", str(args.level), "--hdfs-predictors"]
            if not args.no_mixing:
                command.append("--context-mixing")
            command += [str(src), "-o", str(archive)]
            enc_times, dec_times, hashes = [], [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                subprocess.run(command, env=env, check=True, capture_output=True, timeout=args.timeout)
                enc_times.append(time.perf_counter() - start)
                hashes.append(digest(archive))
                start = time.perf_counter()
                subprocess.run([str(binary), "unpack", str(archive), "-o", str(restored)],
                               check=True, capture_output=True, timeout=args.timeout)
                dec_times.append(time.perf_counter() - start)
                assert digest(restored) == raw_hash, (src, mode, "roundtrip")
                restored.unlink()
            assert len(set(hashes)) == 1, (src, mode, "nondeterministic")
            with query.open("wb") as stream:
                subprocess.run([str(binary), "grep", "Deleting block", str(archive)], stdout=stream,
                               stderr=subprocess.PIPE, check=True, timeout=args.timeout)
            assert digest(query) == expected_grep.hexdigest(), (src, mode, "grep")
            query.unlink()
            info = subprocess.check_output([str(binary), "inspect", "--json", str(archive)], timeout=args.timeout)
            (artifacts / f"{stem}.inspect.json").write_bytes(info)
            row = {"mode": mode, "bytes": archive.stat().st_size, "encode_seconds": statistics.median(enc_times),
                   "decode_seconds": statistics.median(dec_times), "encode_samples": enc_times,
                   "decode_samples": dec_times, "compressed_sha256": hashes[0], "roundtrip": True,
                   "grep": True, "determinism_checked": args.repeats > 1,
                   "contains_ranked_sum": b"RankedSum" in info,
                   "contains_sort_indices": b"SortIndices" in info,
                   "contains_ordered_sum": b"OrderedSum" in info}
            if mode != "ranked":
                assert not row["contains_ranked_sum"], (mode, "still emits RankedSum")
            dataset["variants"].append(row)
            output.write_text(json.dumps({**report, "datasets": report["datasets"] + [dataset]}, indent=2) + "\n")
            print(f"{src.name} {mode}: {row['bytes']:,} B, enc {row['encode_seconds']:.3f}s dec {row['decode_seconds']:.3f}s", flush=True)
        report["datasets"].append(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("build")
    make.add_argument("--work", required=True, type=pathlib.Path)
    make.add_argument("--revision", default="fc5e945", help="source revision before the KeyedSum extension")
    bench = sub.add_parser("run")
    bench.add_argument("--work", required=True, type=pathlib.Path)
    bench.add_argument("inputs", nargs="+", type=pathlib.Path)
    bench.add_argument("--output", type=pathlib.Path)
    bench.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    bench.add_argument("--level", type=int, default=6)
    bench.add_argument("--no-mixing", action="store_true")
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--timeout", type=float, default=40)
    args = parser.parse_args()
    args.work = args.work.resolve()
    if args.action == "run" and (args.repeats < 1 or args.timeout <= 0):
        parser.error("repeats and timeout must be positive")
    (build if args.action == "build" else run)(args)


if __name__ == "__main__":
    main()
