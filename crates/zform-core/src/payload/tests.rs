//! The engine against the reference evaluator and native generators.

use super::engine::CompiledProgram;
use super::graph::{CoreDecoder, GraphBuilder, GraphNode, PayloadGraph};
use super::ir::{ALL_OPS, Builder, Ins, Len, Op, Program, TableBuf, Width, eval_reference};
use super::library::{Harvest, LibEntry, Library, with_code_pricing};
use super::recur::{CompiledRecurrence, RecurrenceSpec, eval_scalar_u32};
use crate::packs::{PackRegistry, prog_fmt_int, recur_mt19937};
use crate::wire::{Cursor, DecodeBudget, put_ivarint, put_uvarint};
use std::sync::Arc;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[test]
fn program_wire_roundtrip_and_adversarial_rejection() {
    // The programs the built-in packs write for a few column shapes.
    let columns: [Vec<String>; 5] = [
        (0..4000i64)
            .map(|i| (i * i * 7_919 - 300).to_string())
            .collect(),
        (0..4000)
            .map(|i| {
                format!(
                    "2026-11-{:02}T{:02}:24:{:02}Z",
                    1 + i % 28,
                    i * 7 % 24,
                    i * 13 % 60
                )
            })
            .collect(),
        (0..4000u64)
            .map(|i| {
                format!(
                    "{:08x}-0000-4000-8000-{:012x}",
                    (i * 2_654_435_761) as u32,
                    i * 0x9e37_79b9 % (1 << 48)
                )
            })
            .collect(),
        (0..4000)
            .map(|i| {
                format!(
                    "10.250.{}.{}:{}",
                    i % 7,
                    i * 37 % 256,
                    50_000 + i * 31 % 9_000
                )
            })
            .collect(),
        (0..4000u64)
            .map(|i| format!("{:016x}", i.wrapping_mul(0x9e37_79b9_7f4a_7c15)))
            .collect(),
    ];
    let reg = PackRegistry::default();
    let mut progs = vec![prog_fmt_int(true, 42)];
    for col in &columns {
        let rows: Vec<&[u8]> = col.iter().map(String::as_bytes).collect();
        for node in reg.encode_best_graph(&rows).nodes {
            if let GraphNode::ProgramCells { prog, .. } | GraphNode::ProgramI64 { prog, .. } = node
            {
                progs.push(Program::clone(&prog));
            }
        }
    }
    assert!(progs.len() >= 5, "{} programs", progs.len());
    for p in &progs {
        let enc = p.encode();
        let dec = Program::decode(&enc).expect("valid program must decode");
        assert_eq!(&dec, p);
        for cut in 0..enc.len() {
            assert!(
                Program::decode(&enc[..cut]).is_err(),
                "truncated program at {cut}/{} should fail",
                enc.len()
            );
        }
        let mut trailing = enc.clone();
        trailing.push(0);
        assert!(Program::decode(&trailing).is_err());
    }

    // Random garbage bytes must never panic or pass unchecked bounds.
    let mut rng = 0xdead_beef_cafe_babe_u64;
    for len in 0..128 {
        let buf: Vec<u8> = (0..len).map(|_| splitmix64(&mut rng) as u8).collect();
        let _ = Program::decode(&buf);
        let _ = RecurrenceSpec::decode(&buf);
        let _ = PayloadGraph::decode(&buf, &DecodeBudget::default());
        let _ = CoreDecoder::decode_bytes(&buf, &DecodeBudget::default());
    }
}

/// A random valid program over one 256-byte table.
fn random_program(rng: &mut u64, width: Width, trial: usize) -> Program {
    let n_in = (1 + (splitmix64(rng) % 4)) as u32;
    let n_ops = 4 + (splitmix64(rng) % 40) as usize;
    let mut luts = Vec::new();
    for _ in 0..4 {
        let mut l = [0u8; 16];
        for b in &mut l {
            *b = splitmix64(rng) as u8;
        }
        luts.push(l);
    }

    let mut ops = Vec::with_capacity(n_ops);
    for i in 0..n_ops {
        let avail = n_in + i as u32;
        let op = loop {
            let cand = ALL_OPS[(splitmix64(rng) as usize) % ALL_OPS.len()];
            if width == Width::W32
                && matches!(
                    cand,
                    Op::MulHiI
                        | Op::F64Add
                        | Op::F64Sub
                        | Op::F64Mul
                        | Op::F64AddI
                        | Op::F64MulI
                        | Op::I64ToF64
                        | Op::F64DivI
                        | Op::U64ToF64
                        | Op::ScanF64Add
                )
            {
                continue;
            }
            // Avoid generating NaN inputs into F64 ops so bit-level comparison is deterministic.
            if matches!(
                cand,
                Op::F64Add
                    | Op::F64Sub
                    | Op::F64Mul
                    | Op::F64AddI
                    | Op::F64MulI
                    | Op::F64DivI
                    | Op::ScanF64Add
            ) {
                continue;
            }
            break cand;
        };
        let a = (splitmix64(rng) as u32) % avail;
        let b = if op.arity() >= 2 {
            (splitmix64(rng) as u32) % avail
        } else {
            0
        };
        let c = if op.arity() >= 3 {
            (splitmix64(rng) as u32) % avail
        } else {
            0
        };
        let aux = match op {
            Op::NibbleLut => (splitmix64(rng) % 4) as u8,
            Op::ByteLut32 => (splitmix64(rng) % 3) as u8,
            Op::Gather => ((splitmix64(rng) % 4) as u8) << 6,
            _ => 0,
        };
        let imm = if op.is_shift_imm() {
            (splitmix64(rng) % width.bits() as u64) as i64
        } else if op == Op::Gather {
            (splitmix64(rng) % 16) as i64
        } else if op.has_imm() {
            splitmix64(rng) as i64
        } else {
            0
        };
        ops.push(Ins {
            op,
            a,
            b,
            c,
            aux,
            imm,
        });
    }

    let total = n_in + n_ops as u32;
    let max_w = if width == Width::W32 { 4u8 } else { 8u8 };
    let n_sinks = 1 + (trial % 4);
    let mut sinks = Vec::with_capacity(n_sinks);
    for _ in 0..n_sinks {
        let v = (splitmix64(rng) as u32) % total;
        if splitmix64(rng) & 3 == 0 {
            let l = (splitmix64(rng) as u32) % total;
            sinks.push((v, Len::Lane(l)));
        } else {
            let w = 1 + (splitmix64(rng) as u8 % max_w);
            sinks.push((v, Len::Fixed(w)));
        }
    }

    Program {
        width,
        n_in,
        n_bufs: 1,
        luts,
        ops,
        sinks,
    }
}

#[test]
fn engine_matches_eval_reference_on_random_dags_w32_and_w64() {
    let mut rng = 0x1234_5678_90ab_cdef_u64;
    let tbl_bytes: Vec<u8> = (0..256).map(|_| splitmix64(&mut rng) as u8).collect();
    let bufs = [TableBuf::from_bytes(&tbl_bytes)];

    for &width in &[Width::W32, Width::W64] {
        for trial in 0..120 {
            let prog = random_program(&mut rng, width, trial);
            let n_in = prog.n_in;
            let enc = prog.encode();
            let dec = Program::decode(&enc).expect("random valid DAG must decode");
            assert_eq!(dec, prog);

            let rows = [0, 1, 7, 16, 31, 32, 33, 64, 65, 131, 300, 1100][trial % 12];
            let cols: Vec<Vec<i64>> = (0..n_in)
                .map(|_| (0..rows).map(|_| splitmix64(&mut rng) as i64).collect())
                .collect();
            let col_refs: Vec<&[i64]> = cols.iter().map(|c| c.as_slice()).collect();

            let want = eval_reference(&prog, &col_refs, &bufs);
            let cp = CompiledProgram::compile(&prog);
            let mut ends = vec![0u32; rows];
            let mut text = vec![0u8; rows * prog.row_bytes() + 16];
            let written = cp.eval_cells(&col_refs, &bufs, rows, &mut ends, &mut text);
            for r in 0..rows {
                let s = if r == 0 { 0 } else { ends[r - 1] as usize };
                let e = ends[r] as usize;
                assert_eq!(&text[s..e], want[r].as_slice(), "trial {trial} row {r}");
            }
            assert_eq!(written, ends.last().copied().unwrap_or(0) as usize);
        }
    }
}

#[test]
fn f64_ops_and_scans_are_bit_exact() {
    let b = Builder::new(2);
    let u = b.input(0);
    let d = b.input(1);
    let f = b.r(Op::U64ToF64, u);
    let scaled = b.rf(Op::F64MulI, f, 0.25);
    let biased = b.rf(Op::F64AddI, scaled, -1.5);
    let walk = b.scan_f64_add(d, 10.0);
    let sum = b.rr(Op::F64Add, biased, walk);
    let diff = b.rr(Op::F64Sub, sum, f);
    let prod = b.rr(Op::F64Mul, diff, scaled);
    let prog = b.finish(&[(prod, 8)]);

    let mut rng = 99u64;
    let rows = 100;
    let u_col: Vec<i64> = (0..rows)
        .map(|_| (splitmix64(&mut rng) >> 10) as i64)
        .collect();
    let d_col: Vec<i64> = (0..rows)
        .map(|i| ((i as f64) * 0.125).to_bits() as i64)
        .collect();
    let inputs = [u_col.as_slice(), d_col.as_slice()];

    let want = eval_reference(&prog, &inputs, &[]);
    let cp = CompiledProgram::compile(&prog);
    let mut got = vec![0i64; rows];
    cp.eval_i64(&inputs, &[], &mut got);
    for r in 0..rows {
        assert_eq!(&got[r].to_le_bytes()[..], want[r].as_slice());
    }
}

#[test]
fn recurrence_matches_native_generators_across_odd_lengths() {
    let seed = generators::mt_seed(5489);
    let mt_raw = CompiledRecurrence::compile(&recur_mt19937(false)).unwrap();
    let mt_tmp = CompiledRecurrence::compile(&recur_mt19937(true)).unwrap();
    let lfg = CompiledRecurrence::compile(&recur_lfg(3, 31)).unwrap();
    let rsum = CompiledRecurrence::compile(&recur_running_sum()).unwrap();

    for &n in &[0usize, 1, 5, 16, 31, 63, 64, 65, 624, 625, 2053] {
        let mut want = vec![0u32; n];
        let mut got = vec![0u32; n];
        generators::mt::<false>(5489, &mut want);
        mt_raw.eval_u32(&seed, &[], &mut got);
        assert_eq!(got, want, "MT raw mismatch at n={n}");

        generators::mt::<true>(5489, &mut want);
        mt_tmp.eval_u32(&seed, &[], &mut got);
        assert_eq!(got, want, "MT tempered mismatch at n={n}");

        let mut full_lfg = Vec::with_capacity(31 + n);
        full_lfg.extend_from_slice(&seed[..31]);
        full_lfg.resize(31 + n, 0);
        generators::lfg(&mut full_lfg);
        lfg.eval_u32(&seed[..31], &[], &mut got);
        assert_eq!(got, &full_lfg[31..], "LFG(3,31) mismatch at n={n}");

        let inp: Vec<i64> = (0..n).map(|i| (i as i64 * 37) - 100).collect();
        let mut want_s = vec![0i64; n];
        let mut got_s = vec![0i64; n];
        generators::running_sum(&inp, &mut want_s);
        rsum.eval_i64(&[0], &[inp.as_slice()], &mut got_s);
        assert_eq!(got_s, want_s, "running_sum mismatch at n={n}");
    }

    // Nonlinear short-lag recurrence (`lag = 5` and `lag = 19`): chunks of five words per step.
    let nb = Builder::new_u32(2);
    let a = nb.input(0);
    let c = nb.input(1);
    let x = nb.rr(Op::Xor, a, nb.ri(Op::ShrI, c, 3));
    let y = nb.ri(Op::AddI, x, 0x9e37_79b9u32 as i64);
    let nonlin = RecurrenceSpec {
        lags: vec![5, 19],
        state_prog: nb.finish(&[(y, 4)]),
        out_prog: None,
    };
    let comp_nl = CompiledRecurrence::compile(&nonlin).unwrap();
    let mut want_nl = vec![0u32; 500];
    let mut got_nl = vec![0u32; 500];
    eval_scalar_u32(&nonlin, &seed[..19], &[], &mut want_nl);
    comp_nl.eval_u32(&seed[..19], &[], &mut got_nl);
    assert_eq!(got_nl, want_nl);
}

/// Recurrence throughput for the specs the writer emits: `cargo test --release bench_recur --
/// --ignored --nocapture`.
#[test]
#[ignore]
fn bench_recurrences() {
    let best = |f: &mut dyn FnMut()| {
        (0..15)
            .map(|_| {
                let t = std::time::Instant::now();
                f();
                t.elapsed().as_nanos()
            })
            .min()
            .unwrap()
    };
    let seed = generators::mt_seed(5489);
    let mt = CompiledRecurrence::compile(&recur_mt19937(true)).unwrap();
    let mut out = vec![0u32; 1 << 20];
    let ns = best(&mut || mt.eval_u32(&seed, &[], &mut out));
    println!(
        "mt19937 tempered: {:.2} ns/word",
        ns as f64 / out.len() as f64
    );
    // Python's `init_by_array` first pass: lag 1 with a column.
    let b = Builder::new_u32(2);
    let prev = b.input(0);
    let mixed = b.ri(
        Op::MulI,
        b.rr(Op::Xor, prev, b.ri(Op::ShrI, prev, 30)),
        1_812_433_253,
    );
    let cur = b.rr(Op::Add, mixed, b.input(1));
    let spec = RecurrenceSpec {
        lags: vec![1],
        state_prog: b.finish(&[(cur, 4)]),
        out_prog: None,
    };
    let lag1 = CompiledRecurrence::compile(&spec).unwrap();
    let idx: Vec<u32> = (1..=623).collect();
    let mut out = vec![0u32; 623];
    let ns = best(&mut || {
        for _ in 0..100 {
            lag1.eval_u32(&[19_650_218], &[&idx], &mut out);
        }
    });
    println!(
        "init_by_array lag-1 (623 words): {:.2} ns/word",
        ns as f64 / 62_300.0
    );
    let ns = best(&mut || {
        for _ in 0..100 {
            eval_scalar_u32(&spec, &[19_650_218], &[&idx], &mut out);
        }
    });
    println!(
        "  scalar evaluator, same: {:.2} ns/word",
        ns as f64 / 62_300.0
    );
    // Lag `k` and `k + 1` add-xor generator: where chunking beats one word at a time.
    for k in [2u32, 4, 8, 16, 32] {
        let b = Builder::new_u32(2);
        let x = b.rr(Op::Add, b.input(0), b.ri(Op::ShlI, b.input(1), 3));
        let x = b.rr(Op::Xor, x, b.ri(Op::ShrI, x, 7));
        let spec = RecurrenceSpec {
            lags: vec![k, k + 1],
            state_prog: b.finish(&[(x, 4)]),
            out_prog: None,
        };
        let seed: Vec<u32> = (0..=k).map(|i| i * 2_654_435_761).collect();
        let c = CompiledRecurrence::compile(&spec).unwrap();
        let mut out = vec![0u32; 1 << 16];
        let fast = best(&mut || c.eval_u32(&seed, &[], &mut out));
        let slow = best(&mut || eval_scalar_u32(&spec, &seed, &[], &mut out));
        println!(
            "lag {k}: compiled {:.2} scalar {:.2} ns/word",
            fast as f64 / 65536.0,
            slow as f64 / 65536.0
        );
    }
}

/// `p` as a recurrence program: one full-width sink, no scans, gathers or buffers.
fn as_recurrence_prog(mut p: Program) -> Program {
    for ins in &mut p.ops {
        if ins.op.is_scan() || ins.op == Op::Gather {
            ins.op = Op::XorI;
        }
    }
    let full = if p.width == Width::W32 { 4 } else { 8 };
    p.n_bufs = 0;
    p.sinks = vec![(p.n_in + p.ops.len() as u32 - 1, Len::Fixed(full))];
    p
}

#[test]
fn random_recurrences_match_the_scalar_evaluator() {
    let mut rng = 0x5eed_0f_7ec0_u64;
    for trial in 0..160 {
        let width = if trial % 2 == 0 {
            Width::W32
        } else {
            Width::W64
        };
        let state_prog = as_recurrence_prog(random_program(&mut rng, width, 0));
        let n_lags = 1 + (splitmix64(&mut rng) % u64::from(state_prog.n_in)) as usize;
        let lags: Vec<u32> = (0..n_lags)
            .map(|_| [1, 2, 3, 5, 31, 64, 300][(splitmix64(&mut rng) % 7) as usize])
            .collect();
        let mut out_prog = None;
        if splitmix64(&mut rng) % 2 == 0 {
            let mut p = random_program(&mut rng, width, 0);
            p.n_in = 1;
            p.ops.iter_mut().for_each(|i| (i.a, i.b, i.c) = (0, 0, 0));
            out_prog = Some(as_recurrence_prog(p));
        }
        let spec = RecurrenceSpec {
            lags,
            state_prog,
            out_prog,
        };
        let n = [0, 1, 7, 300, 1100][trial % 5];
        let cols: Vec<Vec<i64>> = (0..spec.n_cols())
            .map(|_| (0..n).map(|_| splitmix64(&mut rng) as i64).collect())
            .collect();
        let seed: Vec<i64> = (0..spec.max_lag())
            .map(|_| splitmix64(&mut rng) as i64)
            .collect();
        let compiled = CompiledRecurrence::compile(&spec).unwrap();
        if width == Width::W32 {
            let narrow = |v: &[i64]| v.iter().map(|&x| x as u32).collect::<Vec<_>>();
            let cols: Vec<Vec<u32>> = cols.iter().map(|c| narrow(c)).collect();
            let cols: Vec<&[u32]> = cols.iter().map(|c| c.as_slice()).collect();
            let (mut want, mut got) = (vec![0; n], vec![0; n]);
            eval_scalar_u32(&spec, &narrow(&seed), &cols, &mut want);
            compiled.eval_u32(&narrow(&seed), &cols, &mut got);
            assert_eq!(got, want, "trial {trial}: {spec:?}");
        } else {
            let cols: Vec<&[i64]> = cols.iter().map(|c| c.as_slice()).collect();
            let (mut want, mut got) = (vec![0; n], vec![0; n]);
            super::recur::eval_scalar_i64(&spec, &seed, &cols, &mut want);
            compiled.eval_i64(&seed, &cols, &mut got);
            assert_eq!(got, want, "trial {trial}: {spec:?}");
        }
    }
}

fn recur_lfg(lag_a: u32, lag_b: u32) -> RecurrenceSpec {
    let b = Builder::new_u32(2);
    let sum = b.rr(Op::Add, b.input(0), b.input(1));
    RecurrenceSpec {
        lags: vec![lag_a, lag_b],
        state_prog: b.finish(&[(sum, 4)]),
        out_prog: None,
    }
}

/// Lag-1 running sum `s[i] = s[i - 1] + col[i]` in `W64`.
fn recur_running_sum() -> RecurrenceSpec {
    let b = Builder::new(2);
    let sum = b.rr(Op::Add, b.input(0), b.input(1));
    RecurrenceSpec {
        lags: vec![1],
        state_prog: b.finish(&[(sum, 8)]),
        out_prog: None,
    }
}

/// Native generators the recurrence engine must reproduce.
mod generators {
    const N: usize = 624;
    const M: usize = 397;

    /// MT19937's `init_genrand`.
    pub fn mt_seed(seed: u32) -> [u32; N] {
        let mut s = [seed; N];
        for i in 1..N {
            s[i] = (s[i - 1] ^ (s[i - 1] >> 30))
                .wrapping_mul(1_812_433_253)
                .wrapping_add(i as u32);
        }
        s
    }

    pub fn temper(mut y: u32) -> u32 {
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// The MT19937 state words that follow `mt_seed(seed)`, tempered into outputs if `TEMPER`.
    pub fn mt<const TEMPER: bool>(seed: u32, out: &mut [u32]) {
        let mut s = mt_seed(seed);
        let twist = |hi: u32, lo: u32, far: u32| {
            let y = (hi & (1 << 31)) | (lo & !(1 << 31));
            far ^ (y >> 1) ^ (0x9908_b0df * (y & 1))
        };
        for chunk in out.chunks_mut(N) {
            for i in 0..N - M {
                s[i] = twist(s[i], s[i + 1], s[i + M]);
            }
            for i in N - M..N - 1 {
                s[i] = twist(s[i], s[i + 1], s[i + M - N]);
            }
            s[N - 1] = twist(s[N - 1], s[0], s[M - 1]);
            for (o, &x) in chunk.iter_mut().zip(&s) {
                *o = if TEMPER { temper(x) } else { x };
            }
        }
    }

    /// `r[i] = r[i - 3] + r[i - 31]` mod 2^32, after 31 seed words.
    pub fn lfg(r: &mut [u32]) {
        for i in 31..r.len() {
            r[i] = r[i - 3].wrapping_add(r[i - 31]);
        }
    }

    pub fn running_sum(inp: &[i64], out: &mut [i64]) {
        let mut acc = 0i64;
        for (o, &x) in out.iter_mut().zip(inp) {
            acc = acc.wrapping_add(x);
            *o = acc;
        }
    }
}

/// Each row plus `add`, as its low `width` bytes.
fn offset_prog(add: i64, width: u8) -> Program {
    let b = Builder::new(1);
    let v = b.ri(Op::AddI, b.input(0), add);
    b.finish(&[(v, width)])
}

fn shift_prog(by: i64) -> Program {
    let b = Builder::new(1);
    let v = b.ri(Op::ShlI, b.input(0), by);
    b.finish(&[(v, 8)])
}

fn graph_of(prog: Program) -> PayloadGraph {
    let mut gb = GraphBuilder::new(4);
    let ints = gb.add_ints(&[1, 2, 3, 400]);
    let root = gb.add_prog_cells(prog, vec![ints], vec![]);
    gb.finish(root)
}

#[test]
fn a_library_program_is_named_with_its_constants_overridden() {
    let lib = Library::new(vec![LibEntry::Program(offset_prog(7, 2).into())]);
    let budget = DecodeBudget::default();
    // The entry itself, then other constants, another width, and both at once.
    let mut sizes = Vec::new();
    for (add, width) in [(7, 2), (1 << 40, 2), (7, 3), (-5, 8)] {
        let g = graph_of(offset_prog(add, width));
        let linked = g.encode_linked(Some(&lib));
        assert!(linked.len() <= g.encode().len(), "add={add} width={width}");
        assert_eq!(
            PayloadGraph::decode_linked(&linked, &lib, &budget).unwrap(),
            g
        );
        if linked != g.encode() {
            // The reference means nothing without the library it names.
            assert!(PayloadGraph::decode(&linked, &budget).is_err());
        } else {
            // A short lane chain can be smaller inline than its overrides.
            assert_eq!(PayloadGraph::decode(&linked, &budget).unwrap(), g);
        }
        sizes.push(linked.len());
    }
    assert!(sizes[0] < sizes[2], "{sizes:?}");

    // A program of another shape stays inline, and reads back with or without the library.
    let other = graph_of(shift_prog(3));
    let wire = other.encode_linked(Some(&lib));
    assert_eq!(wire, other.encode());
    assert_eq!(
        PayloadGraph::decode_linked(&wire, &lib, &budget).unwrap(),
        other
    );
    assert_eq!(PayloadGraph::decode(&wire, &budget).unwrap(), other);
}

#[test]
fn of_two_entries_of_one_shape_the_nearer_is_named() {
    let b = |x: i64, y: i64| {
        let b = Builder::new(1);
        let v = b.ri(Op::XorI, b.ri(Op::AddI, b.input(0), x), y);
        b.finish(&[(v, 4)])
    };
    let lib = Library::new(vec![
        LibEntry::Program(b(1, 2).into()),
        LibEntry::Program(b(30, 40).into()),
    ]);
    let exact = graph_of(b(30, 40)).encode_linked(Some(&lib));
    let near = graph_of(b(30, 41)).encode_linked(Some(&lib));
    let far = graph_of(b(31, 41)).encode_linked(Some(&lib));
    assert!(exact.len() < near.len() && near.len() < far.len());
    for (wire, prog) in [(exact, b(30, 40)), (near, b(30, 41)), (far, b(31, 41))] {
        let g = PayloadGraph::decode_linked(&wire, &lib, &DecodeBudget::default()).unwrap();
        assert_eq!(g, graph_of(prog));
    }
}

#[test]
fn tables_and_recurrences_are_named_whole() {
    let table: Vec<u8> = (0..200u8).collect();
    let graph = |table: &[u8]| {
        let b = Builder::new(1);
        let buf = b.add_buf();
        let v = b.gather(buf, 0, b.input(0), 0);
        let mut gb = GraphBuilder::new(4);
        let ints = gb.add_ints(&[0, 5, 9, 199]);
        let buf = gb.add_raw_bytes(table);
        let root = gb.add_prog_cells(b.finish(&[(v, 1)]), vec![ints], vec![buf]);
        gb.finish(root)
    };
    let g = graph(&table);
    let lib = Library::new(vec![LibEntry::Bytes(table.clone())]);
    let linked = g.encode_linked(Some(&lib));
    assert!(linked.len() + 190 < g.encode().len());
    let budget = DecodeBudget::default();
    let back = PayloadGraph::decode_linked(&linked, &lib, &budget).unwrap();
    assert_eq!(back, g);
    let cells = CoreDecoder::decode_cells(&back, &Default::default(), &budget).unwrap();
    assert_eq!(
        (0..4).map(|i| cells.get(i).unwrap()[0]).collect::<Vec<_>>(),
        [0, 5, 9, 199]
    );
    // A table one byte off is the block's own.
    let mut odd = table;
    odd[100] ^= 1;
    assert_eq!(graph(&odd).encode_linked(Some(&lib)), graph(&odd).encode());

    let spec = recur_mt19937(true);
    let lib = Library::new(vec![LibEntry::Recur(spec.clone())]);
    assert_eq!(lib.find_recur(&spec.encode()), Some(0));
    assert_eq!(lib.read_recur(0).unwrap(), spec);
    assert_eq!(lib.find_recur(&recur_mt19937(false).encode()), None);
}

#[test]
fn library_wire_round_trips_and_refuses_bad_references() {
    let lib = Library::new(vec![
        LibEntry::Program(offset_prog(7, 2).into()),
        LibEntry::Bytes(vec![1, 2, 3]),
        LibEntry::Recur(recur_mt19937(true)),
        LibEntry::Program(shift_prog(3).into()),
    ]);
    let mut wire = Vec::new();
    lib.encode(&mut wire);
    assert_eq!(Library::decode(&mut Cursor::new(&wire)).unwrap(), lib);
    for cut in 0..wire.len() {
        assert!(
            Library::decode(&mut Cursor::new(&wire[..cut])).is_err(),
            "cut {cut}"
        );
    }
    let mut bad_kind = wire.clone();
    bad_kind[1] = 3;
    assert!(Library::decode(&mut Cursor::new(&bad_kind)).is_err());
    assert!(Library::decode(&mut Cursor::new(&[0xff, 0xff, 0x7f])).is_err());

    // An entry of the wrong kind, or none at all.
    assert!(lib.read_program(1, &mut Cursor::new(&[0])).is_err());
    assert!(lib.read_program(4, &mut Cursor::new(&[0])).is_err());
    assert!(lib.read_program(u64::MAX, &mut Cursor::new(&[0])).is_err());
    assert!(lib.read_recur(0).is_err());
    assert!(lib.read_bytes(2).is_err());
    assert_eq!(lib.read_bytes(1).unwrap(), [1, 2, 3]);

    // `offset_prog` has two values: its constant and its width.
    let over = |bytes: &[u8]| lib.read_program(0, &mut Cursor::new(bytes));
    assert_eq!(*over(&[0]).unwrap(), offset_prog(7, 2));
    assert_eq!(*over(&[1, 0, 18]).unwrap(), offset_prog(9, 2));
    assert_eq!(*over(&[1, 1, 8]).unwrap(), offset_prog(7, 4));
    assert_eq!(*over(&[2, 0, 18, 0, 8]).unwrap(), offset_prog(9, 4));
    for bad in [
        &[1, 2, 0][..], // a value past the last
        &[2, 1, 8, 0, 8],
        &[3, 0, 0, 0, 0, 0, 0], // more overrides than values
        &[1, 0],                // cut short
        &[1, 1, 18],            // a width no sink can have
    ] {
        assert!(over(bad).is_err(), "{bad:?}");
    }
    // An override gets the checks an inline program gets: a shift past the lane is refused.
    assert_eq!(
        *lib.read_program(3, &mut Cursor::new(&[1, 0, 126])).unwrap(),
        shift_prog(63)
    );
    assert!(lib.read_program(3, &mut Cursor::new(&[1, 0, 128])).is_err());
}

#[test]
fn a_harvest_keeps_repeats_and_one_program_per_shape() {
    let wire = |p: Program| graph_of(p).encode();
    let mut h = Harvest::default();
    // One shape: a program two blocks repeat and one a single block uses.
    for p in [offset_prog(7, 2), offset_prog(9, 2), offset_prog(7, 2)] {
        h.add_blob(&wire(p));
    }
    // Another shape whose constant changes every block, and a third used once.
    for by in [1, 2, 3] {
        h.add_blob(&wire(shift_prog(by)));
    }
    let once = {
        let b = Builder::new(1);
        let v = b.ri(Op::MulI, b.input(0), 3);
        b.finish(&[(v, 8)])
    };
    h.add_blob(&wire(once.clone()));
    let lib = h.into_library(false);
    assert_eq!(
        lib.entries(),
        [
            LibEntry::Program(offset_prog(7, 2).into()),
            LibEntry::Program(shift_prog(1).into())
        ]
    );

    // When later blocks are expected, code seen once is kept too.
    let mut h = Harvest::default();
    h.add_blob(&wire(once.clone()));
    assert_eq!(
        h.into_library(true).entries(),
        [LibEntry::Program(once.into())]
    );
}

#[test]
fn shared_code_is_priced_at_its_reference() {
    let g = graph_of(offset_prog(9, 2));
    let full = g.encode().len();
    assert_eq!(g.cost(), full);
    let lib = Arc::new(Library::new(vec![LibEntry::Program(
        offset_prog(7, 2).into(),
    )]));
    let linked = g.encode_linked(Some(&lib)).len();
    assert_eq!(with_code_pricing(lib, 1, || g.cost()), linked);
    // With no library yet, code is priced as shared by the blocks expected to use it.
    let halved = with_code_pricing(Arc::default(), 2, || g.cost());
    assert!(halved < full && halved > linked, "{halved} of {full}");
    assert_eq!(g.cost(), full);
}

/// The cells of `prog`, run as a block does that names library entry 0 with the instruction
/// constants `set` replaced, checked against the reference evaluator.
fn check_named(lib: &Library, prog: &Program, set: &[(usize, i64)], rows: usize, rng: &mut u64) {
    let mut wire = Vec::new();
    put_uvarint(&mut wire, set.len() as u64);
    let mut variant = prog.clone();
    for (k, &(at, value)) in set.iter().enumerate() {
        let skip = if k == 0 { at } else { at - set[k - 1].0 - 1 };
        put_uvarint(&mut wire, skip as u64);
        put_ivarint(&mut wire, value);
        let ins = variant.ops.iter_mut().filter(|i| i.op.has_imm()).nth(at);
        ins.unwrap().imm = value;
    }
    let code = lib.read_program(0, &mut Cursor::new(&wire)).unwrap();
    assert_eq!(*code, variant);

    let table: Vec<u8> = (0..=255).collect();
    let bufs = [TableBuf::from_bytes(&table)];
    let cols: Vec<Vec<i64>> = (0..prog.n_in)
        .map(|_| (0..rows).map(|_| splitmix64(rng) as i64).collect())
        .collect();
    let cols: Vec<&[i64]> = cols.iter().map(|c| c.as_slice()).collect();
    let want = eval_reference(&variant, &cols, &bufs).concat();
    let got = CompiledProgram::with(&code, |cp| {
        let mut ends = vec![0u32; rows];
        let mut text = vec![0u8; rows * variant.row_bytes() + 16];
        let n = cp.eval_cells(&cols, &bufs, rows, &mut ends, &mut text);
        text.truncate(n);
        text
    });
    assert_eq!(got, want, "constants {set:?} in {prog:?}");
}

/// A constant instruction `at` of `prog` may carry.
fn random_imm(prog: &Program, at: usize, rng: &mut u64) -> i64 {
    let op = prog
        .ops
        .iter()
        .filter(|i| i.op.has_imm())
        .nth(at)
        .unwrap()
        .op;
    let x = splitmix64(rng);
    if op.is_shift_imm() {
        (x % u64::from(prog.width.bits())) as i64
    } else if op == Op::Gather {
        (x % 16) as i64
    } else if x & 1 == 0 {
        x as i64
    } else {
        (x % 7) as i64 - 3
    }
}

#[test]
fn a_named_program_runs_as_if_compiled_with_its_constants() {
    let mut rng = 0x0bad_5eed_cafe_f00d_u64;
    for &width in &[Width::W32, Width::W64] {
        for trial in 0..80 {
            let prog = random_program(&mut rng, width, trial);
            let lib = Library::new(vec![LibEntry::Program(prog.clone().into())]);
            let n_imm = prog.ops.iter().filter(|i| i.op.has_imm()).count();
            let rows = [1, 7, 31, 32, 33, 65][trial % 6];
            // One compiled entry serves every set of constants and then its own again.
            for round in 0..6 {
                let mut set: Vec<(usize, i64)> = Vec::new();
                for at in 0..n_imm {
                    if round < 5 && splitmix64(&mut rng) % (n_imm as u64).max(1) < 2 {
                        set.push((at, random_imm(&prog, at, &mut rng)));
                    }
                }
                check_named(&lib, &prog, &set, rows, &mut rng);
            }
        }
    }
}

#[test]
fn constants_of_combined_operations_take_overrides_alone_or_together() {
    let m = 0x00ff_00ff_00ff_00ff_i64;
    // Each program with its count of constants.
    let cases: Vec<(Program, usize)> = vec![
        // A constant lane, then one derived from it: only the last instruction's stands alone.
        {
            let b = Builder::new(1);
            let k = b.ri(Op::AddI, b.konst(b.input(0), 5), 7);
            (
                {
                    let out = b.rr(Op::Add, b.input(0), k);
                    b.finish(&[(out, 8)])
                },
                3,
            )
        },
        // Multiply-add: the addend, not the multiplier.
        {
            let b = Builder::new(1);
            let v = b.ri(Op::AddI, b.ri(Op::MulI, b.input(0), 3), 9);
            (b.finish(&[(v, 8)]), 2)
        },
        // Complementary masks.
        {
            let b = Builder::new(2);
            let lo = b.ri(Op::AndI, b.input(0), m);
            let hi = b.ri(Op::AndI, b.input(1), !m);
            let out = b.rr(Op::Or, lo, hi);
            (b.finish(&[(out, 8)]), 2)
        },
        // Xorshift steps, either way round.
        {
            let b = Builder::new(1);
            let x = b.rr(Op::Xor, b.ri(Op::ShrI, b.input(0), 7), b.input(0));
            let y = b.rr(Op::Xor, x, b.ri(Op::AndI, b.ri(Op::ShlI, x, 3), m));
            (b.finish(&[(y, 8)]), 3)
        },
        // A low bit spread into a mask.
        {
            let b = Builder::new(2);
            let top = b.ri(Op::ShlI, b.input(1), 63);
            let mask = b.ri(Op::AndI, b.ri(Op::SarI, top, 63), m);
            let v = b.rr(Op::Xor, b.input(0), mask);
            (b.finish(&[(v, 8)]), 3)
        },
        // A select on masked bits, and beside it a scan and a shift that stand alone.
        {
            let b = Builder::new(2);
            let cond = b.ri(Op::AndI, b.input(0), 1);
            let sum = b.ri(Op::ScanAdd, b.input(1), 100);
            let v = b.select(cond, b.ri(Op::ShrI, sum, 4), b.input(1));
            (b.finish(&[(v, 8)]), 3)
        },
    ];
    let mut rng = 7u64;
    for &(ref prog, n_imm) in &cases {
        assert_eq!(prog.ops.iter().filter(|i| i.op.has_imm()).count(), n_imm);
        let lib = Library::new(vec![LibEntry::Program(prog.clone().into())]);
        // Every constant alone, then all at once, then none.
        let mut sets: Vec<Vec<(usize, i64)>> = (0..n_imm)
            .map(|at| vec![(at, random_imm(prog, at, &mut rng))])
            .collect();
        sets.push(
            (0..n_imm)
                .map(|at| (at, random_imm(prog, at, &mut rng)))
                .collect(),
        );
        sets.push(Vec::new());
        for set in &sets {
            check_named(&lib, prog, set, 40, &mut rng);
        }
    }
}

#[test]
fn numeric_chains_match_reference_for_extremes_and_short_tails() {
    let values = [i64::MIN, i64::MAX, -1, 0, 1, 9_007_199_254_740_993, -17001];
    for rows in [0, 1, 2, 7, 15, 31, 65] {
        let input: Vec<_> = (0..rows).map(|i| values[i % values.len()]).collect();
        for scan in [false, true] {
            for factor in [1, 1000, -1, i64::MIN] {
                for scale in [
                    None,
                    Some((Op::F64DivI, 1000.)),
                    Some((Op::F64MulI, -0.001)),
                ] {
                    let b = Builder::new(1);
                    let mut v = b.input(0);
                    if scan {
                        v = b.scan_add(v, i64::MAX);
                    }
                    v = b.ri(Op::ShlI, v, 2);
                    if factor != 1 {
                        v = b.ri(Op::MulI, v, factor);
                    }
                    v = b.ri(Op::AddI, v, -17);
                    if let Some((op, divisor)) = scale {
                        v = b.rf(op, b.r(Op::I64ToF64, v), divisor);
                    }
                    let p = b.finish(&[(v, 8)]);
                    let want = eval_reference(&p, &[&input], &[]);
                    let mut fast = vec![0; rows];
                    assert!(super::chain::eval_numeric_chain(&p, &input, &mut fast));
                    let mut general = vec![0; rows];
                    CompiledProgram::compile(&p).eval_i64(&[&input], &[], &mut general);
                    assert_eq!(fast, general);
                    for (got, expected) in fast.iter().zip(want) {
                        assert_eq!(got.to_le_bytes().as_slice(), expected);
                    }
                }
            }
        }
    }
}

#[test]
fn common_factor_normalization_handles_signed_range_and_wrapping_products() {
    use super::graph::{CoreDecoder, ExtInputs, GraphBuilder};
    let budget = DecodeBudget::default();
    for (base, factor, span) in [
        (i64::MIN, 1000u64, 127),
        (-17001, 3000, 127),
        (i64::MIN, u64::MAX, 2),
        (i64::MIN, 1 << 63, 2),
    ] {
        let values: Vec<_> = (0..128u64)
            .map(|i| {
                (base as u64).wrapping_add(((i * 17 + i * i * 2) % span).wrapping_mul(factor))
                    as i64
            })
            .collect();
        let mut builder = GraphBuilder::new(values.len());
        let root = builder.add_ints_fast(&values);
        let graph = builder.finish(root).fuse_numeric_programs();
        assert_eq!(
            CoreDecoder::decode_ints(&graph, &ExtInputs::empty(), &budget).unwrap(),
            values
        );
        assert_eq!(
            CoreDecoder::decode_ints_bytes(&graph.encode(), &budget).unwrap(),
            values
        );
    }
}

#[test]
fn sparse_integer_transport_preserves_wide_exceptions_and_validates_framing() {
    use super::graph::IntStream;
    let mut values: Vec<_> = (0..257).map(|i| i % 23 - 11).collect();
    values[0] = i64::MIN;
    values[128] = i64::MAX;
    values[256] = -900_000;
    let stream = IntStream::encode_fast(&values);
    assert!(matches!(stream, IntStream::ZigZagBytes { .. }));
    let mut bytes = Vec::new();
    stream.encode(&mut bytes);
    let budget = DecodeBudget::default();
    let decoded = IntStream::decode_wire(&mut Cursor::new(&bytes), &budget, 0).unwrap();
    assert_eq!(decoded.unpack(&budget).unwrap(), values);
    for n in 0..bytes.len() {
        assert!(
            IntStream::decode_wire(&mut Cursor::new(&bytes[..n]), &budget, 0)
                .and_then(|s| s.unpack(&budget))
                .is_err()
        );
    }
}

#[test]
fn encoder_dictionary_hashes_distribute_padded_identifier_suffixes() {
    use std::hash::{Hash, Hasher};
    for width in [16, 32] {
        let mut buckets = [false; 1024];
        for n in 0..4096 {
            let key = format!("{n:0width$x}");
            let mut hash = super::graph::FastHasher::default();
            key.as_bytes().hash(&mut hash);
            buckets[hash.finish() as usize & 1023] = true;
        }
        assert!(buckets.iter().filter(|&&used| used).count() > 900);
    }
}

#[test]
fn compact_numeric_graphs_match_general_execution_and_reject_bad_frames() {
    use super::graph::{CoreDecoder, ExtInputs, GraphBuilder, PayloadGraph};
    let budget = DecodeBudget::default();
    for rows in [1, 7, 65] {
        let input: Vec<_> = (0..rows)
            .map(|i| (i as i64).wrapping_mul(i64::MAX / 3))
            .collect();
        for transform in 0..4 {
            let mut builder = GraphBuilder::new(rows);
            let source = builder.add_ints_fast(&input);
            let b = Builder::new(1);
            let value = match transform {
                0 => b.input(0),
                1 => b.scan_add(b.input(0), -17001),
                2 => b.rf(Op::F64DivI, b.r(Op::I64ToF64, b.input(0)), 1e9),
                _ => b.r(Op::Bswap, b.input(0)),
            };
            let program = b.finish(&[(value, 8)]);
            let root = builder.add_prog_i64(program, vec![source], vec![]);
            let graph = builder.finish(root).fuse_numeric_programs();
            let wire = graph.encode();
            let parsed = PayloadGraph::decode(&wire, &budget).unwrap();
            let expected = CoreDecoder::decode_ints(&graph, &ExtInputs::empty(), &budget).unwrap();
            assert_eq!(
                CoreDecoder::decode_ints_bytes(&wire, &budget).unwrap(),
                expected
            );
            assert_eq!(
                CoreDecoder::decode_ints(&parsed, &ExtInputs::empty(), &budget).unwrap(),
                expected
            );
            for end in 0..wire.len() {
                assert!(CoreDecoder::decode_ints_bytes(&wire[..end], &budget).is_err());
            }
            let mut trailing = wire.clone();
            trailing.push(0);
            assert!(CoreDecoder::decode_ints_bytes(&trailing, &budget).is_err());
            let tight = DecodeBudget {
                max_single_alloc_bytes: 1,
                ..budget.clone()
            };
            assert!(CoreDecoder::decode_ints_bytes(&wire, &tight).is_err());
        }
    }
    // One constant input, followed by a chain whose operation needs two operands.
    let bad = [2, 16, 0, 2, 0, 0x81, 1, Op::Add as u8];
    assert!(PayloadGraph::decode(&bad, &budget).is_err());
    assert!(CoreDecoder::decode_ints_bytes(&bad, &budget).is_err());
}

#[test]
fn shifted_constant_transport_preserves_extremes_and_rejects_overflow() {
    use super::graph::IntStream;
    let budget = DecodeBudget::default();
    for value in [
        0,
        -1,
        1,
        i64::MIN,
        i64::MAX,
        1i64 << 60,
        1.5f64.to_bits() as i64,
        (-3.0f64).to_bits() as i64,
    ] {
        let stream = IntStream::Const { val: value, len: 7 };
        let mut bytes = Vec::new();
        stream.encode(&mut bytes);
        assert_eq!(
            IntStream::decode_wire(&mut Cursor::new(&bytes), &budget, 0)
                .unwrap()
                .unpack(&budget)
                .unwrap(),
            vec![value; 7]
        );
        for n in 0..bytes.len() {
            assert!(IntStream::decode_wire(&mut Cursor::new(&bytes[..n]), &budget, 0).is_err());
        }
    }
    for bytes in [&[7, 1, 0, 2][..], &[7, 1, 64, 2], &[7, 1, 63, 4]] {
        assert!(IntStream::decode_wire(&mut Cursor::new(bytes), &budget, 0).is_err());
    }
}

#[test]
fn rice_integer_transport_validates_counts_framing_and_allocation_budget() {
    use super::graph::IntStream;
    let values: Vec<_> = (0..65).map(|i| (i * 17 % 31) - 15).collect();
    let bytes = super::rice::encode(&values, usize::MAX).unwrap();
    let stream = IntStream::Rice {
        len: values.len() as u32,
        bytes,
    };
    let mut wire = Vec::new();
    stream.encode(&mut wire);
    let budget = DecodeBudget::default();
    assert_eq!(
        IntStream::decode_wire(&mut Cursor::new(&wire), &budget, 0)
            .unwrap()
            .unpack(&budget)
            .unwrap(),
        values
    );
    for end in 0..wire.len() {
        assert!(
            IntStream::decode_wire(&mut Cursor::new(&wire[..end]), &budget, 0)
                .and_then(|s| s.unpack(&budget))
                .is_err()
        );
    }
    for count in [0, 1, 64, 66, 127] {
        let mut bad = wire.clone();
        bad[1] = count;
        assert!(
            IntStream::decode_wire(&mut Cursor::new(&bad), &budget, 0)
                .and_then(|s| s.unpack(&budget))
                .is_err()
        );
    }
    let tight = DecodeBudget {
        max_single_alloc_bytes: 64,
        ..budget
    };
    assert!(
        IntStream::decode_wire(&mut Cursor::new(&wire), &tight, 0)
            .and_then(|s| s.unpack(&tight))
            .is_err()
    );
}
