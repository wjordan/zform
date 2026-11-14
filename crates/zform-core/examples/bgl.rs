//! A fixed BGL header keeps its repeated identities in shared columns; messages get a model
//! of their own. The output carries both models and decodes with an ordinary zform reader.
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use zform_core::container::{PackOptions, pack_stream};
use zform_core::framing::FramingSpec;
use zform_core::grammar::{FrontEnd, ModelPlan, PlanVariant, Shape};
use zform_core::packs::{BglClockPack, BglNodePack};
use zform_core::plan::{Profile, TrainOptions};
use zform_core::{block, build_dictionary};

const USAGE: &str = "Usage: bgl INPUT OUTPUT [--level 1..19] [--raw-message] [--no-node-pack]
           [--train-rows N] [--max-templates N] [--dict-mib N]
           [--block-bytes N] [--block-rows N] [--combined-clock]
OUTPUT must be a new file. Restore it with zform unpack.";

struct Config {
    input: PathBuf,
    output: PathBuf,
    level: i32,
    nested: bool,
    nodes: bool,
    train_rows: usize,
    templates: usize,
    dict_mib: usize,
    block_bytes: usize,
    block_rows: usize,
    combined_clock: bool,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut args = env::args().skip(1);
        let input = args.next().ok_or(USAGE)?;
        let output = args.next().ok_or(USAGE)?;
        let mut c = Self {
            input: input.into(),
            output: output.into(),
            level: 3,
            nested: true,
            nodes: true,
            train_rows: 16384,
            templates: 128,
            dict_mib: 0,
            block_bytes: 8 << 20,
            block_rows: 65536,
            combined_clock: false,
        };
        while let Some(arg) = args.next() {
            if arg == "--combined-clock" {
                c.combined_clock = true;
                continue;
            }
            if arg == "--raw-message" {
                c.nested = false;
                continue;
            }
            if arg == "--no-node-pack" {
                c.nodes = false;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {arg}"))?;
            match arg.as_str() {
                "--level" => c.level = value.parse().map_err(|_| "invalid level")?,
                "--train-rows" => c.train_rows = value.parse().map_err(|_| "invalid row count")?,
                "--max-templates" => {
                    c.templates = value.parse().map_err(|_| "invalid template count")?
                }
                "--dict-mib" => {
                    c.dict_mib = value.parse().map_err(|_| "invalid dictionary budget")?
                }
                "--block-bytes" => {
                    c.block_bytes = value.parse().map_err(|_| "invalid block size")?
                }
                "--block-rows" => {
                    c.block_rows = value.parse().map_err(|_| "invalid block row count")?
                }
                _ => return Err(format!("unknown option: {arg}")),
            }
        }
        if !(1..=19).contains(&c.level)
            || c.train_rows < 2
            || c.templates == 0
            || c.block_bytes == 0
            || c.block_rows == 0
            || c.dict_mib > usize::MAX >> 20
        {
            return Err(
                "level must be 1..19; training needs at least two rows; budgets must fit usize"
                    .into(),
            );
        }
        Ok(c)
    }
}

fn sample_messages(path: &Path, max_rows: usize) -> std::io::Result<Vec<Vec<u8>>> {
    // Count first so the sample covers the file without holding all its records in memory.
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; 256 << 10];
    let mut count = 1;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        count += memchr::memchr_iter(b'\n', &buf[..n]).count();
    }
    let n = count.min(max_rows);
    let mut reader = BufReader::with_capacity(256 << 10, File::open(path)?);
    let mut messages = Vec::with_capacity(n);
    let (mut line, mut row, mut sample) = (Vec::new(), 0, 0);
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        if row == sample * (count - 1) / (n - 1).max(1) {
            let record = line.strip_suffix(b"\n").unwrap_or(&line);
            if let Some(body) = record.splitn(10, |&b| b == b' ').nth(9) {
                messages.push(body.to_vec());
            }
            sample += 1;
        }
        row += 1;
    }
    Ok(messages)
}

fn profile(c: &Config) -> std::io::Result<Profile> {
    let messages = sample_messages(&c.input, c.train_rows)?;
    let bodies: Vec<&[u8]> = messages.iter().map(Vec::as_slice).collect();
    let mut plan = ModelPlan::default();
    let mut slots = Vec::new();
    for name in [
        "label",
        "epoch",
        "date",
        "node",
        "time",
        "node_copy",
        "facility",
        "component",
        "severity",
        "message",
    ] {
        let id = plan.add_or_get_col(name);
        if name == "node" || name == "node_copy" {
            plan.columns[id as usize].dict_group = "node".into();
        }
        if name == "message" {
            plan.columns[id as usize].dict_group = "text".into();
        }
        plan.columns[id as usize].horiz_candidates = (1..id).rev().take(6).collect();
        slots.push(id);
    }
    let train = TrainOptions {
        max_templates: c.templates,
        max_sample_rows: c.train_rows,
        max_sample_bytes: 64 << 20,
        ..TrainOptions::default()
    };
    if c.nested {
        plan.columns[10].nested_plan = Some(Box::new(block::train(&bodies, &train, None)));
    }
    // Empty messages omit the final separator; both layouts share every header column.
    for fields in [10, 9] {
        let mut literals = vec![b" ".to_vec(); fields + 1];
        literals[0].clear();
        literals[fields].clear();
        plan.push_variant(
            PlanVariant::Shape(Shape {
                literals,
                slots: slots[..fields].to_vec(),
            }),
            FrontEnd::Template,
        );
    }
    plan.rebuild_indexes();
    Ok(Profile::from_plan(plan))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let c = Config::parse()?;
    let started = Instant::now();
    let mut opts = PackOptions {
        level: c.level,
        framing: FramingSpec::Lf,
        profile: Some(profile(&c)?),
        allow_drift_retrain: false,
        block_bytes: c.block_bytes,
        block_rows: c.block_rows,
        chunk_bytes: c.block_bytes.max(16 << 20),
        ..PackOptions::default()
    };
    if c.nodes {
        opts.packs.register(BglNodePack);
    }
    if c.combined_clock {
        opts.packs.register(BglClockPack);
    }
    opts.packs = opts.packs.with_plain_nested_fallback();
    if c.dict_mib > 0 {
        opts.dictionary =
            build_dictionary(File::open(&c.input)?, &opts, c.dict_mib << 20)?.map(Arc::new);
    }
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&c.output)?;
    pack_stream(File::open(&c.input)?, output, &opts)?;
    eprintln!(
        "raw={} compressed={} seconds={:.3}",
        fs::metadata(&c.input)?.len(),
        fs::metadata(&c.output)?.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    if env::args().any(|arg| arg == "--help") {
        println!("{USAGE}");
        return;
    }
    if let Err(e) = run() {
        eprintln!("bgl: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zform_core::{DecodeBudget, unpack_stream};

    #[test]
    fn archive_keeps_empty_messages_node_exceptions_and_unterminated_tail() {
        let input = env::temp_dir().join(format!("zform-bgl-test-{}", std::process::id()));
        let mut raw = Vec::new();
        for i in 0..32 {
            let node = if i == 17 {
                "RFF-MF-NF-I:J99-U99"
            } else {
                "R02-M1-N0-C:J12-U11"
            };
            let mut row = format!(
                "- {} 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.{i:02}.675872 {node} RAS KERNEL INFO",
                1117838520 + i
            );
            if i % 4 != 0 {
                row.push_str(&format!(" corrected {i} errors"));
            }
            raw.extend_from_slice(row.as_bytes());
            raw.push(b'\n');
        }
        raw.extend_from_slice(b"an unrelated short line without a final newline");
        fs::write(&input, &raw).unwrap();
        let c = Config {
            input: input.clone(),
            output: PathBuf::new(),
            level: 3,
            nested: true,
            nodes: true,
            train_rows: 16,
            templates: 128,
            dict_mib: 0,
            block_bytes: 8 << 20,
            block_rows: 65536,
            combined_clock: false,
        };
        let mut opts = PackOptions {
            profile: Some(profile(&c).unwrap()),
            allow_drift_retrain: false,
            framing: FramingSpec::Lf,
            ..PackOptions::default()
        };
        opts.packs.register(BglNodePack);
        opts.packs = opts.packs.with_plain_nested_fallback();
        fs::remove_file(input).unwrap();
        let mut compressed = Vec::new();
        pack_stream(raw.as_slice(), &mut compressed, &opts).unwrap();
        let mut restored = Vec::new();
        unpack_stream(
            compressed.as_slice(),
            &mut restored,
            &DecodeBudget::default(),
        )
        .unwrap();
        assert_eq!(restored, raw);
    }
}
