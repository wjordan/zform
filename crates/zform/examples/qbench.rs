//! Query bench over a file from `file::Writer`: bytes requested, round trips and time.
//!
//!   qbench pack   <base> <profile.zfp|-> <level> <block MiB> [filter bits]   < records.ndjson
//!   qbench fetch  <base> <field> <value>             records where field == value
//!   qbench among  <base> <field> <value>             the same, told which blocks hold such records,
//!                                                    as a host with its own index of the field is
//!   qbench grep   <base> <needle>                    how many records contain needle
//!   qbench newest <base> <needle> <field> <n> <end>  the n records containing needle with the latest
//!                                                    timestamp in field, none at or after end (epoch s)
//!   qbench top    <base> <field> <value> <path> <n>  the n most common values of path where field == value
//!   qbench filters <base>                            size and fill of the block filters
//!
//! `<base>.data` and `<base>.index` hold the two regions. Each query starts with nothing read.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::time::Instant;
use zform::file::{self, ByteRange, RangeCache, Region, RowSet, Step, Writer, drive};
use zform::{DecodeBudget, PackOptions, Predicate, Profile, Result, logical};

/// The two region files, with what has been read from them.
struct Disk {
    data: File,
    index: File,
    cache: RangeCache,
    data_bytes: u64,
    index_bytes: u64,
    trips: usize,
    /// Data ranges read, as (start, end).
    read: Vec<(u64, u64)>,
}

impl Disk {
    fn run<T>(&mut self, op: impl FnMut(&RangeCache) -> Result<Step<T>>) -> T {
        let Self {
            data,
            index,
            cache,
            data_bytes,
            index_bytes,
            trips,
            read,
        } = self;
        let fetch = |ranges: &[ByteRange]| {
            *trips += 1;
            let mut parts = Vec::with_capacity(ranges.len());
            for r in ranges {
                let (f, n) = match r.region {
                    Region::Data => {
                        read.push((r.off, r.end()));
                        (&*data, &mut *data_bytes)
                    }
                    Region::Index => (&*index, &mut *index_bytes),
                };
                let mut buf = vec![0u8; r.len as usize];
                f.read_exact_at(&mut buf, r.off)?;
                *n += r.len;
                parts.push(buf);
            }
            Ok(parts)
        };
        drive(cache, fetch, op).expect("query")
    }
}

/// Nanoseconds this thread has spent on a CPU.
fn cpu_ns() -> u64 {
    let s = std::fs::read_to_string("/proc/thread-self/schedstat").unwrap_or_default();
    s.split(' ')
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn pack(base: &str, args: &[String]) {
    let mut opts = PackOptions::default();
    if args[0] != "-" {
        let bytes = std::fs::read(&args[0]).expect("profile");
        opts.profile =
            Some(Profile::from_bytes(&bytes, &DecodeBudget::default()).expect("profile"));
        opts.allow_drift_retrain = false;
    }
    opts.level = args[1].parse().expect("level");
    opts.block_bytes = args[2].parse::<usize>().expect("block MiB") << 20;
    if let Some(bits) = args.get(3) {
        opts.filter_bits = bits.parse().expect("filter bits");
    }
    let out = BufWriter::with_capacity(
        1 << 20,
        File::create(format!("{base}.data")).expect("create"),
    );
    let mut w = Writer::new(out, opts);
    let (t, cpu) = (Instant::now(), cpu_ns());
    let (mut raw, mut residual) = (0u64, 0u64);
    let mut input = std::io::stdin().lock();
    let mut line = Vec::new();
    loop {
        line.clear();
        if input.read_until(b'\n', &mut line).expect("stdin") == 0 {
            break;
        }
        let rec = line.strip_suffix(b"\n").unwrap_or(&line);
        raw += rec.len() as u64;
        residual += w
            .push(rec)
            .expect("push")
            .map_or(0, |fit| fit.residual_bytes as u64);
    }
    let rows = w.rows();
    residual += w
        .cut()
        .expect("cut")
        .map_or(0, |fit| fit.residual_bytes as u64);
    let (mut out, index) = w.finish().expect("finish");
    out.flush().expect("flush");
    std::fs::write(format!("{base}.index"), &index).expect("index");
    let data = std::fs::metadata(format!("{base}.data")).unwrap().len();
    println!(
        "{{\"rows\":{rows},\"raw_bytes\":{raw},\"data_bytes\":{data},\"index_bytes\":{},\"residual_bytes\":{residual},\"wall_s\":{:.1},\"cpu_s\":{:.1}}}",
        index.len(),
        t.elapsed().as_secs_f64(),
        (cpu_ns() - cpu) as f64 / 1e9,
    );
}

/// The bytes of `read` that fall in each part of the blocks: row structure, shared dictionaries,
/// text pool, and columns with whatever else a block holds.
fn parts_read(base: &str, read: &mut [(u64, u64)]) -> [u64; 4] {
    use zform_core::index::{BlockDir, BlockKind, INDEX_TAIL_LEN, Root, locate_root};
    let index = std::fs::read(format!("{base}.index")).expect("index");
    let (off, len, crc) =
        locate_root(&index[index.len() - INDEX_TAIL_LEN..], index.len() as u64).expect("root");
    let root = Root::decode(
        &index[off as usize..][..len as usize],
        crc,
        &DecodeBudget::default(),
    )
    .expect("root");
    read.sort_unstable();
    let within = |s: u64, e: u64| -> u64 {
        let from = read.partition_point(|r| r.1 <= s);
        read[from..]
            .iter()
            .take_while(|r| r.0 < e)
            .map(|r| r.1.min(e) - r.0.max(s))
            .sum()
    };
    let mut parts = [0u64; 4];
    for b in &root.blocks {
        if let BlockKind::Structured { .. } = b.kind {
            let page = &index[b.dir_off as usize..][..b.dir_len as usize];
            let d = BlockDir::decode(page, &DecodeBudget::default()).expect("dir");
            parts[0] += within(b.off, b.off + d.struct_len as u64);
            for (o, l) in &d.dicts {
                parts[1] += within(b.off + *o as u64, b.off + (*o + *l) as u64);
            }
            if let Some((o, l)) = d.pool {
                parts[2] += within(b.off + o as u64, b.off + (o + l) as u64);
            }
        }
        parts[3] += within(b.off, b.off + b.len);
    }
    parts[3] -= parts[0] + parts[1] + parts[2];
    parts
}

fn filters(base: &str) {
    use zform_core::index::{INDEX_TAIL_LEN, Root, locate_root};
    let index = std::fs::read(format!("{base}.index")).expect("index");
    let (off, len, crc) =
        locate_root(&index[index.len() - INDEX_TAIL_LEN..], index.len() as u64).expect("root");
    let (off, len) = (off as usize, len as usize);
    let root = Root::decode(&index[off..off + len], crc, &DecodeBudget::default()).expect("root");
    let (mut dir, mut bytes, mut fill, mut pass) = (0u64, 0u64, 0f64, 0f64);
    // Where the data region's bytes are: row structure, shared dictionaries, text pool, columns.
    let mut parts = [0u64; 4];
    for b in &root.blocks {
        if let zform_core::index::BlockKind::Structured { .. } = b.kind {
            let page = &index[b.dir_off as usize..][..b.dir_len as usize];
            let d =
                zform_core::index::BlockDir::decode(page, &DecodeBudget::default()).expect("dir");
            parts[0] += d.struct_len as u64;
            parts[1] += d.dicts.iter().map(|x| x.1 as u64).sum::<u64>();
            parts[2] += d.pool.map_or(0, |x| x.1 as u64);
            parts[3] += d.col_spans().map(|c| c.1 as u64).sum::<u64>();
        }
    }
    eprintln!(
        "structure {} dicts {} pool {} columns {}",
        parts[0], parts[1], parts[2], parts[3]
    );
    for b in &root.blocks {
        let f = &index[b.filter_off as usize..][..b.filter_len as usize];
        let set: u32 = f.iter().map(|x| x.count_ones()).sum();
        let p = set as f64 / (f.len() * 8) as f64;
        dir += b.dir_len as u64;
        bytes += f.len() as u64;
        fill += p;
        // A needle of ten letters probes three grams of two bits each.
        pass += p.powi(6);
    }
    let n = root.blocks.len() as f64;
    println!(
        "{{\"blocks\":{n},\"root_bytes\":{len},\"dir_bytes\":{dir},\"filter_bytes\":{bytes},\"mean_fill\":{:.3},\"false_pass_10_letters\":{:.3}}}",
        fill / n,
        pass / n
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, base, rest) = (args[0].as_str(), args[1].as_str(), &args[2..]);
    match cmd {
        "pack" => return pack(base, rest),
        "filters" => return filters(base),
        _ => {}
    }
    let (mut t, mut cpu) = (Instant::now(), cpu_ns());
    let open = |ext: &str| File::open(format!("{base}.{ext}")).expect("open");
    let (data, index) = (open("data"), open("index"));
    let index_len = index.metadata().unwrap().len();
    let mut disk = Disk {
        data,
        index,
        cache: RangeCache::default(),
        data_bytes: 0,
        index_bytes: 0,
        trips: 0,
        read: Vec::new(),
    };
    let file = disk.run(|c| file::File::open(c, index_len, DecodeBudget::default()));
    let mut session = file.session();
    let eq = |field: &str, value: &str| Predicate::FieldEq {
        field: field.into(),
        value: value.into(),
    };

    let (hits, result);
    match cmd {
        "fetch" | "among" => {
            let pred = eq(&rest[0], &rest[1]);
            let within = (cmd == "among").then(|| {
                // Found outside the measurement: every row of the blocks that hold a match.
                let mut all = file.session();
                let found = disk.run(|c| all.matches(c, &pred, None));
                let mut within = RowSet::default();
                for (first, n) in file.blocks() {
                    let block = RowSet::range(first, first + n);
                    if !block.intersect(&found).is_empty() {
                        within.push_range(first, first + n);
                    }
                }
                disk.cache = RangeCache::default();
                (disk.data_bytes, disk.index_bytes, disk.trips) = (0, 0, 0);
                disk.read.clear();
                disk.run(|c| file::File::open(c, index_len, DecodeBudget::default()));
                (t, cpu) = (Instant::now(), cpu_ns());
                within
            });
            let rows = disk.run(|c| session.matches(c, &pred, within.as_ref()));
            let mut bytes = 0u64;
            disk.run(|c| {
                session.records(c, &rows, |_, rec| {
                    let _: () = bytes += rec.len() as u64;
                    Ok(())
                })
            });
            (hits, result) = (rows.len(), bytes.to_string());
        }
        "grep" => {
            let pred = Predicate::Contains(rest[0].clone().into_bytes());
            (hits, result) = (
                disk.run(|c| session.matches(c, &pred, None)).len(),
                String::new(),
            );
        }
        "newest" => {
            let needle = Predicate::Contains(rest[0].clone().into_bytes());
            let field = rest[1].as_str();
            let n: usize = rest[2].parse().expect("n");
            let end = rest[3].parse::<i64>().expect("end, epoch seconds") * 1_000_000_000;
            // Windows double back from the end until they hold n matches.
            let mut keyed = Vec::<(Vec<u8>, u64)>::new();
            let (mut hi, mut span) = (end, 3600 * 1_000_000_000i64);
            for _ in 0..16 {
                let window = Predicate::FieldTimeRange {
                    field: field.into(),
                    min_ns: Some(hi - span),
                    max_ns: Some(hi - 1),
                };
                let pred = Predicate::And(vec![window, needle.clone()]);
                let rows = disk.run(|c| session.matches(c, &pred, None));
                let before = keyed.len();
                disk.run(|c| {
                    keyed.truncate(before);
                    session.fields(c, &[field], &rows, |row, _, v| {
                        let _: () = keyed.extend(v.map(|v| (logical(v).into_owned(), row)));
                        Ok(())
                    })
                });
                if keyed.len() >= n {
                    break;
                }
                (hi, span) = (hi - span, span * 2);
            }
            keyed.sort_unstable();
            let mut newest: Vec<u64> = keyed.iter().rev().take(n).map(|k| k.1).collect();
            newest.sort_unstable();
            let newest = RowSet::from_sorted(newest);
            let mut bytes = 0u64;
            disk.run(|c| {
                session.records(c, &newest, |_, rec| {
                    let _: () = bytes += rec.len() as u64;
                    Ok(())
                })
            });
            (hits, result) = (newest.len(), bytes.to_string());
        }
        "top" => {
            let pred = eq(&rest[0], &rest[1]);
            let n: usize = rest[3].parse().expect("n");
            let rows = disk.run(|c| session.matches(c, &pred, None));
            let mut counts = HashMap::<Vec<u8>, u64>::new();
            disk.run(|c| {
                counts.clear();
                session.distinct(c, &rest[2], &rows, |v, rows| {
                    let _: () = *counts.entry(v.to_vec()).or_default() += rows.len() as u64;
                    Ok(())
                })
            });
            let mut top: Vec<(u64, Vec<u8>)> = counts.into_iter().map(|(v, n)| (n, v)).collect();
            top.sort_unstable_by(|a, b| b.cmp(a));
            top.truncate(n);
            hits = top.first().map_or(0, |t| t.0);
            let show =
                |t: &(u64, Vec<u8>)| format!("{} {}", String::from_utf8_lossy(&logical(&t.1)), t.0);
            result = top.iter().map(show).collect::<Vec<_>>().join(", ");
        }
        _ => panic!("unknown command {cmd}"),
    }
    let (wall, cpu) = (
        t.elapsed().as_secs_f64() * 1e3,
        (cpu_ns() - cpu) as f64 / 1e6,
    );
    let [structure, dicts, pool, columns] = parts_read(base, &mut disk.read);
    println!(
        "{{\"query\":\"{cmd}\",\"hits\":{hits},\"result\":\"{}\",\"data_bytes\":{},\"index_bytes\":{},\"round_trips\":{},\"wall_ms\":{wall:.1},\"cpu_ms\":{cpu:.1},\"structure\":{structure},\"dicts\":{dicts},\"pool\":{pool},\"columns\":{columns},\"blocks\":{},\"pruned\":{},\"partial\":{},\"whole\":{},\"rebuilt\":{}}}",
        result.replace('"', "'"),
        disk.data_bytes,
        disk.index_bytes,
        disk.trips,
        file.blocks().count(),
        session.stats.blocks_pruned,
        session.stats.blocks_partial,
        session.stats.blocks_whole,
        session.stats.rebuilt,
    );
}
