use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::ExitCode;

use zform::{
    Archive, DecodeBudget, EntropyBackend, FramingSpec, PackOptions, Predicate, Profile, Stats,
    TrainOptions, inspect_stream, logical, pack, train, train_seekable, unpack_with_budget,
};

fn usage() -> &'static str {
    "\
zform: lossless structure-aware compression, with queries on the compressed file

USAGE:
  zform pack    [OPTIONS] [INPUT] [-o OUTPUT]
  zform unpack  [OPTIONS] [INPUT] [-o OUTPUT]
  zform train   [OPTIONS] <INPUT> -o <PROFILE.zfp> [--extend <PRIOR.zfp>]
  zform inspect [INPUT]
  zform grep    [FILTERS] [SUBSTRING] <FILE>
  zform select  -f <f1,f2,...> [FILTERS] [--tsv] <FILE>
  zform [-d] [-c] [-t] [-l <1..19>] [--fast|--best|--lz4] [INPUT] [-o OUTPUT]

OPTIONS:
  -o, --output <FILE>       Output path (defaults to stdout for filter commands)
  -l, --level <1..19>       Compression level (default: 3)
  --fast                    Fast preset (level 1, 4,096 training rows)
  --best                    High-density preset (level 19, 65,536 training rows)
  --lz4                     Compress columns with LZ4 instead of zstd
  --profile <FILE.zfp>      Use (or extend on `train --extend`) a ZFP1 structural profile
  --fixed-profile           Disable online drift retraining when `--profile` is set
  --train-rows <N>          Max sampled rows during online/offline training (default: 16384)
  --chunk-bytes <BYTES>     Streaming frame raw chunk size (default: 16777216 = 16 MiB)
  --block-bytes <BYTES>     Raw bytes per block, the unit a query reads (default: 8388608 = 8 MiB)
  --block-rows <N>          Max records per block (default: 65536)
  --framing <MODE>          Record framing: auto (default), lf, crlf, nul, u32le, json, stride:<N>[:<HDR>]
  --no-checksum             Omit per-frame XXH64 content checksum
  --no-index                Omit the trailing index region (`grep` / `select` need it)
  -h, --help                Print this help message

FILTERS (`grep` and `select`; several are combined with AND):
  -e, --contains <BYTES>    Records containing these bytes
  -w, --word <BYTES>        The same, with no letter, digit or `_` on either side
  --eq, --where <F=VALUE>   Records whose field F equals VALUE
  --range <F=MIN..MAX>      Records whose integer field F is in the range; either end may be omitted
  --exists <F>              Records that have field F
  -f, --fields <F1,F2,...>  Fields `select` prints, as dotted paths
  --tsv                     `select` prints tab-separated values instead of JSON
  --stats                   Print how much of the file the query read
"
}

/// Lets a closed output pipe end the process quietly, as it does for other filters.
#[cfg(unix)]
fn default_sigpipe() {
    unsafe extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    const SIGPIPE: i32 = 13;
    const SIG_DFL: usize = 0;
    // SAFETY: restores the default disposition before any thread or handler exists.
    unsafe { signal(SIGPIPE, SIG_DFL) };
}

#[cfg(not(unix))]
fn default_sigpipe() {}

fn main() -> ExitCode {
    default_sigpipe();
    let args: Vec<String> = env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("zform: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        return Ok(());
    }
    match args[0].as_str() {
        "pack" => cmd_pack(&args[1..]),
        "unpack" => cmd_unpack(&args[1..], false),
        "test" => cmd_unpack(&args[1..], true),
        "train" => cmd_train(&args[1..]),
        "inspect" => cmd_inspect(&args[1..]),
        "grep" => cmd_grep(&args[1..]),
        "select" => cmd_select(&args[1..]),
        _ => {
            // Unix filter shorthand (`-d` = unpack, `-t` = test integrity, `--list` = inspect, default = pack)
            if args.iter().any(|a| a == "-t" || a == "--test") {
                let filtered: Vec<String> = args
                    .iter()
                    .filter(|a| *a != "-t" && *a != "--test")
                    .cloned()
                    .collect();
                cmd_unpack(&filtered, true)
            } else if args.iter().any(|a| a == "--list") {
                let filtered: Vec<String> =
                    args.iter().filter(|a| *a != "--list").cloned().collect();
                cmd_inspect(&filtered)
            } else if args.iter().any(|a| a == "-d" || a == "--decompress") {
                let filtered: Vec<String> = args
                    .iter()
                    .filter(|a| {
                        *a != "-d" && *a != "--decompress" && *a != "-c" && *a != "--stdout"
                    })
                    .cloned()
                    .collect();
                cmd_unpack(&filtered, false)
            } else {
                let filtered: Vec<String> = args
                    .iter()
                    .filter(|a| *a != "-c" && *a != "--stdout")
                    .cloned()
                    .collect();
                cmd_pack(&filtered)
            }
        }
    }
}

fn parse_framing(s: &str) -> Result<FramingSpec, String> {
    match s {
        "auto" => Ok(FramingSpec::Auto),
        "lf" => Ok(FramingSpec::Lf),
        "crlf" => Ok(FramingSpec::Crlf),
        "nul" => Ok(FramingSpec::Nul),
        "u32le" => Ok(FramingSpec::LengthU32Le),
        "json" => Ok(FramingSpec::Json),
        _ if s.starts_with("stride:") => {
            let rest = &s["stride:".len()..];
            let mut parts = rest.split(':');
            let stride: u32 = parts
                .next()
                .unwrap_or("")
                .parse()
                .map_err(|_| format!("invalid stride in {s}"))?;
            let header_bytes: u32 = match parts.next() {
                Some(h) => h
                    .parse()
                    .map_err(|_| format!("invalid header bytes in {s}"))?,
                None => 0,
            };
            Ok(FramingSpec::FixedStride {
                stride,
                header_bytes,
            })
        }
        _ => Err(format!("unknown framing mode '{s}'")),
    }
}

fn check_not_same_file(in_path: Option<&str>, out_path: Option<&str>) -> Result<(), String> {
    if let (Some(i), Some(o)) = (in_path, out_path)
        && i != "-"
        && o != "-"
        && Path::new(i).exists()
        && Path::new(o).exists()
        && same_file::is_same_file(i, o).unwrap_or(false)
    {
        return Err("input and output refer to the same file".into());
    }
    Ok(())
}

fn open_input(path: Option<&str>) -> Result<Box<dyn Read>, String> {
    match path {
        None | Some("-") => Ok(Box::new(BufReader::with_capacity(256 << 10, io::stdin()))),
        Some(p) => {
            let f = File::open(p).map_err(|e| format!("{p}: {e}"))?;
            Ok(Box::new(BufReader::with_capacity(256 << 10, f)))
        }
    }
}

fn open_output(path: Option<&str>) -> Result<Box<dyn Write>, String> {
    match path {
        None | Some("-") => Ok(Box::new(BufWriter::with_capacity(256 << 10, io::stdout()))),
        Some(p) => {
            let f = File::create(p).map_err(|e| format!("{p}: {e}"))?;
            Ok(Box::new(BufWriter::with_capacity(256 << 10, f)))
        }
    }
}

fn cmd_pack(args: &[String]) -> Result<(), String> {
    let mut opts = PackOptions::default();
    let mut input: Option<&str> = None;
    let mut output: Option<&str> = None;
    let mut profile_path: Option<&str> = None;
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-o" | "--output" => {
                i += 1;
                output = Some(args.get(i).ok_or("missing value for -o")?);
            }
            "-l" | "--level" => {
                i += 1;
                opts.level = args
                    .get(i)
                    .ok_or("missing value for -l")?
                    .parse()
                    .map_err(|_| "invalid compression level")?;
            }
            "--fast" => {
                opts.level = 1;
                opts.train.max_sample_rows = 4_096;
            }
            "--best" => {
                opts.level = 19;
                opts.train.max_sample_rows = 65_536;
                opts.train.max_sample_bytes = 64 << 20;
            }
            "--lz4" => opts.backend = EntropyBackend::Lz4,
            "--raw" => opts.backend = EntropyBackend::Raw,
            "--profile" => {
                i += 1;
                profile_path = Some(args.get(i).ok_or("missing value for --profile")?);
            }
            "--fixed-profile" => opts.allow_drift_retrain = false,
            "--train-rows" => {
                i += 1;
                opts.train.max_sample_rows = args
                    .get(i)
                    .ok_or("missing value for --train-rows")?
                    .parse()
                    .map_err(|_| "invalid --train-rows")?;
            }
            "--chunk-bytes" => {
                i += 1;
                opts.chunk_bytes = args
                    .get(i)
                    .ok_or("missing value for --chunk-bytes")?
                    .parse()
                    .map_err(|_| "invalid --chunk-bytes")?;
            }
            "--framing" => {
                i += 1;
                let f = parse_framing(args.get(i).ok_or("missing value for --framing")?)?;
                opts.framing = f;
                opts.train.framing = f;
            }
            "--no-checksum" => opts.checksum = false,
            "--block-bytes" => {
                i += 1;
                opts.block_bytes = args
                    .get(i)
                    .ok_or("missing value for --block-bytes")?
                    .parse()
                    .map_err(|_| "invalid --block-bytes")?;
            }
            "--block-rows" => {
                i += 1;
                opts.block_rows = args
                    .get(i)
                    .ok_or("missing value for --block-rows")?
                    .parse()
                    .map_err(|_| "invalid --block-rows")?;
            }
            "--no-index" => opts.index = false,
            "-f" | "--force" | "-q" | "--quiet" | "-k" | "--keep" | "-c" | "--stdout" => {}
            _ if a.starts_with('-')
                && a.len() >= 2
                && a[1..].chars().all(|c| c.is_ascii_digit()) =>
            {
                opts.level = a[1..].parse().map_err(|_| "invalid -<level>")?;
            }
            _ if a.starts_with('-') && a != "-" => return Err(format!("unknown flag '{a}'")),
            _ => {
                if input.is_some() {
                    return Err(format!("unexpected extra input argument '{a}'"));
                }
                input = Some(a);
            }
        }
        i += 1;
    }

    if let Some(pp) = profile_path {
        let bytes = std::fs::read(pp).map_err(|e| format!("{pp}: {e}"))?;
        opts.profile =
            Some(Profile::from_bytes(&bytes, &DecodeBudget::default()).map_err(|e| e.to_string())?);
    }

    check_not_same_file(input, output)?;
    let rdr = open_input(input)?;
    let mut wtr = open_output(output)?;
    pack(rdr, &mut wtr, &opts).map_err(|e| e.to_string())?;
    wtr.flush().map_err(|e| e.to_string())
}

fn cmd_unpack(args: &[String], test_only: bool) -> Result<(), String> {
    let mut budget = DecodeBudget::default();
    let mut input: Option<&str> = None;
    let mut output: Option<&str> = None;
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-o" | "--output" => {
                i += 1;
                output = Some(args.get(i).ok_or("missing value for -o")?);
            }
            "--no-verify" => budget.verify_checksums = false,
            "-f" | "--force" | "-q" | "--quiet" | "-k" | "--keep" | "-c" | "--stdout" => {}
            _ if a.starts_with('-') && a != "-" => return Err(format!("unknown flag '{a}'")),
            _ => {
                if input.is_some() {
                    return Err(format!("unexpected extra input argument '{a}'"));
                }
                input = Some(a);
            }
        }
        i += 1;
    }

    check_not_same_file(input, output)?;
    let rdr = open_input(input)?;
    if test_only {
        unpack_with_budget(rdr, io::sink(), &budget).map_err(|e| e.to_string())
    } else {
        let mut wtr = open_output(output)?;
        unpack_with_budget(rdr, &mut wtr, &budget).map_err(|e| e.to_string())?;
        wtr.flush().map_err(|e| e.to_string())
    }
}

fn cmd_train(args: &[String]) -> Result<(), String> {
    let mut opts = TrainOptions::default();
    let mut input: Option<&str> = None;
    let mut output: Option<&str> = None;
    let mut extend_path: Option<&str> = None;
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-o" | "--output" => {
                i += 1;
                output = Some(args.get(i).ok_or("missing value for -o")?);
            }
            "--extend" => {
                i += 1;
                extend_path = Some(args.get(i).ok_or("missing value for --extend")?);
            }
            "--train-rows" => {
                i += 1;
                opts.max_sample_rows = args
                    .get(i)
                    .ok_or("missing value for --train-rows")?
                    .parse()
                    .map_err(|_| "invalid --train-rows")?;
            }
            "--max-sample-bytes" => {
                i += 1;
                opts.max_sample_bytes = args
                    .get(i)
                    .ok_or("missing value for --max-sample-bytes")?
                    .parse()
                    .map_err(|_| "invalid --max-sample-bytes")?;
            }
            "--framing" => {
                i += 1;
                opts.framing = parse_framing(args.get(i).ok_or("missing value for --framing")?)?;
            }
            _ if a.starts_with('-') && a != "-" => return Err(format!("unknown flag '{a}'")),
            _ => {
                if input.is_some() {
                    return Err(format!("unexpected extra input argument '{a}'"));
                }
                input = Some(a);
            }
        }
        i += 1;
    }

    let out_p = output.ok_or("train requires -o <PROFILE.zfp>")?;
    check_not_same_file(input, Some(out_p))?;

    let prior = if let Some(ep) = extend_path {
        let b = std::fs::read(ep).map_err(|e| format!("{ep}: {e}"))?;
        Some(Profile::from_bytes(&b, &DecodeBudget::default()).map_err(|e| e.to_string())?)
    } else {
        None
    };

    let prof = match input {
        Some(p) if p != "-" => {
            let mut f = File::open(p).map_err(|e| format!("{p}: {e}"))?;
            train_seekable(&mut f, &opts, prior.as_ref()).map_err(|e| e.to_string())?
        }
        other => {
            let rdr = open_input(other)?;
            train(rdr, &opts, prior.as_ref()).map_err(|e| e.to_string())?
        }
    };

    std::fs::write(out_p, &prof.bytes).map_err(|e| format!("{out_p}: {e}"))?;
    eprintln!(
        "trained profile {} ({} bytes, {} columns, {} variants)",
        &prof.id_sha256[..16],
        prof.bytes.len(),
        prof.plan.columns.len(),
        prof.plan.variants.len()
    );
    Ok(())
}

fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");
    let input = args
        .iter()
        .find(|a| !a.starts_with('-') || a.as_str() == "-")
        .map(|s| s.as_str());
    if let Some(p) = input
        && p != "-"
        && let Ok(bytes) = std::fs::read(p)
        && bytes.starts_with(b"ZFP1")
    {
        let prof =
            Profile::from_bytes(&bytes, &DecodeBudget::default()).map_err(|e| e.to_string())?;
        println!(
            "{{\"kind\":\"ZFP1-profile\",\"sha256\":\"{}\",\"bytes\":{},\"columns\":{},\"variants\":{}}}",
            prof.id_sha256,
            prof.bytes.len(),
            prof.plan.columns.len(),
            prof.plan.variants.len()
        );
        return Ok(());
    }
    let rdr = open_input(input)?;
    let rep = inspect_stream(rdr, &DecodeBudget::default()).map_err(|e| e.to_string())?;
    let ratio = if rep.total_compressed_bytes > 0 {
        rep.total_uncompressed_bytes as f64 / rep.total_compressed_bytes as f64
    } else {
        0.0
    };
    println!(
        "{{\"kind\":\"ZFM1-stream\",\"frames\":{},\"sync_frames\":{},\"passthrough_frames\":{},\"records\":{},\"raw_bytes\":{},\"compressed_bytes\":{},\"ratio\":{:.3},\"index_bytes\":{},\"variants\":{},\"columns\":{}}}",
        rep.frames,
        rep.sync_frames,
        rep.passthrough_frames,
        rep.total_records,
        rep.total_uncompressed_bytes,
        rep.total_compressed_bytes,
        ratio,
        rep.index_bytes,
        rep.variant_count,
        rep.column_names.len()
    );
    if verbose {
        let mut sorted = rep.column_bytes.clone();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, bytes) in sorted.into_iter().take(20) {
            eprintln!("  {:>9} B  {}", bytes, name);
        }
    }
    Ok(())
}

fn parse_filter_flags(
    args: &[String],
    allow_positional_needle: bool,
) -> Result<
    (
        Predicate,
        Option<String>,
        Option<String>,
        bool,
        Vec<String>,
        bool,
    ),
    String,
> {
    let mut preds = Vec::<Predicate>::new();
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut show_stats = false;
    let mut fields = Vec::<String>::new();
    let mut tsv = false;
    let mut positionals = Vec::<String>::new();

    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-o" | "--output" => {
                i += 1;
                output = Some(args.get(i).ok_or("missing value for -o")?.clone());
            }
            "-f" | "--fields" => {
                i += 1;
                let spec = args.get(i).ok_or("missing value for -f")?;
                for part in spec.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        fields.push(trimmed.to_string());
                    }
                }
            }
            "--tsv" => tsv = true,
            "--stats" => show_stats = true,
            "--contains" | "-e" => {
                i += 1;
                let pat = args.get(i).ok_or("missing value for --contains")?;
                preds.push(Predicate::Contains(pat.as_bytes().to_vec()));
            }
            "--word" | "-w" => {
                i += 1;
                let pat = args.get(i).ok_or("missing value for --word")?;
                preds.push(Predicate::Word(pat.as_bytes().to_vec()));
            }
            "--eq" | "--where" => {
                i += 1;
                let kv = args.get(i).ok_or("missing value for --eq")?;
                let (k, v) = kv.split_once('=').ok_or("--eq expects field=value")?;
                preds.push(Predicate::FieldEq {
                    field: k.to_string(),
                    value: v.as_bytes().to_vec(),
                });
            }
            "--range" => {
                i += 1;
                let kv = args.get(i).ok_or("missing value for --range")?;
                let (k, r) = kv.split_once('=').ok_or("--range expects field=min..max")?;
                let (lo_s, hi_s) = r.split_once("..").ok_or("--range expects min..max")?;
                let min = if lo_s.is_empty() {
                    None
                } else {
                    Some(lo_s.parse::<i64>().map_err(|_| "invalid range min")?)
                };
                let max = if hi_s.is_empty() {
                    None
                } else {
                    Some(hi_s.parse::<i64>().map_err(|_| "invalid range max")?)
                };
                preds.push(Predicate::FieldIntRange {
                    field: k.to_string(),
                    min,
                    max,
                });
            }
            "--exists" => {
                i += 1;
                let f = args.get(i).ok_or("missing value for --exists")?;
                preds.push(Predicate::FieldExists(f.clone()));
            }
            _ if a.starts_with('-') && a != "-" => return Err(format!("unknown flag '{a}'")),
            _ => {
                positionals.push(a.to_string());
            }
        }
        i += 1;
    }

    if allow_positional_needle {
        match (preds.is_empty(), positionals.len()) {
            (true, 0) => {}
            (true, 1) => {
                preds.push(Predicate::Contains(positionals.remove(0).into_bytes()));
            }
            (true, 2) => {
                preds.push(Predicate::Contains(positionals.remove(0).into_bytes()));
                input = Some(positionals.remove(0));
            }
            (false, 0) => {}
            (false, 1) => {
                input = Some(positionals.remove(0));
            }
            _ => {
                return Err(format!(
                    "unexpected extra positional argument '{}'",
                    positionals.last().unwrap()
                ));
            }
        }
    } else if positionals.len() == 1 {
        input = Some(positionals.remove(0));
    } else if positionals.len() > 1 {
        return Err(format!("unexpected extra argument '{}'", positionals[1]));
    }
    let pred = match preds.len() {
        0 => Predicate::All,
        1 => preds.pop().unwrap(),
        _ => Predicate::And(preds),
    };
    Ok((pred, input, output, show_stats, fields, tsv))
}

fn open_archive(input: Option<&str>) -> Result<Archive<File>, String> {
    let p = input
        .filter(|p| *p != "-")
        .ok_or("queries read a file, not stdin")?;
    let f = File::open(p).map_err(|e| format!("{p}: {e}"))?;
    Archive::open(f, DecodeBudget::default()).map_err(|e| format!("{p}: {e}"))
}

fn print_stats(cmd: &str, matched: u64, arch: &Archive<File>, stats: &Stats) {
    eprintln!(
        "zform {cmd}: matched {matched}/{} records | block reads: {} pruned, {} in part, {} whole, over {} blocks | rebuilt {} records | read {}/{} data bytes",
        arch.file().rows(),
        stats.blocks_pruned,
        stats.blocks_partial,
        stats.blocks_whole,
        arch.file().blocks().count(),
        stats.rebuilt,
        arch.data_bytes_read,
        arch.file().data_len(),
    );
}

fn cmd_grep(args: &[String]) -> Result<(), String> {
    let (pred, input, output, show_stats, _, _) = parse_filter_flags(args, true)?;
    let mut arch = open_archive(input.as_deref())?;
    let mut wtr = open_output(output.as_deref())?;
    let mut matched = 0u64;
    let stats = arch
        .grep(&pred, |_, rec| {
            matched += 1;
            wtr.write_all(rec)?;
            wtr.write_all(b"\n")?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    wtr.flush().map_err(|e| e.to_string())?;
    if show_stats {
        print_stats("grep", matched, &arch, &stats);
    }
    Ok(())
}

/// Appends `v` as a JSON string.
fn push_json_string(v: &[u8], out: &mut Vec<u8>) {
    out.push(b'"');
    for &b in v {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0..=0x1f => out.extend_from_slice(format!("\\u{b:04x}").as_bytes()),
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

/// Appends one field of a `select` line: its values as TSV text, or as a JSON member.
fn push_field(
    name: &str,
    first: bool,
    vals: &[Vec<u8>],
    tsv: bool,
    doc: &mut zform::JsonDoc,
    out: &mut Vec<u8>,
) {
    if tsv {
        out.extend_from_slice(if first { b"" } else { b"\t" });
        for (i, v) in vals.iter().enumerate() {
            out.extend_from_slice(if i > 0 { b"," } else { b"" });
            out.extend_from_slice(&logical(v));
        }
        return;
    }
    out.push(if first { b'{' } else { b',' });
    push_json_string(name.as_bytes(), out);
    out.push(b':');
    // A value from a record that is not JSON is text.
    let mut value = |v: &[u8], out: &mut Vec<u8>| {
        if doc.parse_any(v).is_some() {
            out.extend_from_slice(v)
        } else {
            push_json_string(v, out)
        }
    };
    match vals {
        [] => out.extend_from_slice(b"null"),
        [v] => value(v, out),
        _ => {
            for (i, v) in vals.iter().enumerate() {
                out.push(if i > 0 { b',' } else { b'[' });
                value(v, out);
            }
            out.push(b']');
        }
    }
}

fn cmd_select(args: &[String]) -> Result<(), String> {
    let (pred, input, output, show_stats, fields, tsv) = parse_filter_flags(args, false)?;
    if fields.is_empty() {
        return Err("select requires -f <field1,field2,...>".into());
    }
    let mut arch = open_archive(input.as_deref())?;
    let mut wtr = open_output(output.as_deref())?;
    let paths: Vec<&str> = fields.iter().map(String::as_str).collect();
    let mut doc = zform::JsonDoc::default();
    let end: &[u8] = if tsv { b"\n" } else { b"}\n" };
    // The field being gathered, its values, and the line of its row so far.
    let mut cur = None::<(u64, usize)>;
    let mut vals = Vec::<Vec<u8>>::new();
    let mut line = Vec::with_capacity(256);
    let mut matched = 0u64;
    let stats = arch
        .select(&paths, &pred, |row, p, v| {
            if cur != Some((row, p)) {
                if let Some((_, prev)) = cur {
                    push_field(paths[prev], prev == 0, &vals, tsv, &mut doc, &mut line);
                    vals.clear();
                }
                if p == 0 {
                    if cur.is_some() {
                        line.extend_from_slice(end);
                        wtr.write_all(&line)?;
                        line.clear();
                    }
                    matched += 1;
                }
                cur = Some((row, p));
            }
            vals.extend(v.map(<[u8]>::to_vec));
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    if let Some((_, prev)) = cur {
        push_field(paths[prev], prev == 0, &vals, tsv, &mut doc, &mut line);
        line.extend_from_slice(end);
        wtr.write_all(&line).map_err(|e| e.to_string())?;
    }
    wtr.flush().map_err(|e| e.to_string())?;
    if show_stats {
        print_stats("select", matched, &arch, &stats);
    }
    Ok(())
}
