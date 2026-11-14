//! Manual BGL experiment with a native frontend and in-memory codec calls.
//! cargo build --release --manifest-path bench/bgl-native/Cargo.toml
//! bench/bgl-native/target/release/bgl-native pack BGL.log out.bcm --hybrid
//! bench/bgl-native/target/release/bgl-native unpack out.bcm restored.log
//! BGLCM1/BGLCM2 stay compatible with bench/bgl_cm.py; no Python or helper process is used.

#[allow(dead_code)]
#[path = "../../../crates/zform-core/examples/bgl_payload_probe.rs"]
mod clock;
#[allow(dead_code)]
#[path = "../../bgl_cm.rs"]
mod mixer;
mod parser;

use chrono::{Datelike, Timelike};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{BufWriter, Read, Write},
    path::Path,
    time::Instant,
};
use xz2::{
    read::XzDecoder,
    stream::{Check, Stream},
    write::XzEncoder,
};
use zform_core::{payload::graph::CoreDecoder, wire::DecodeBudget};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MAGIC: &[u8] = b"BGLCM1\n";
const HYBRID: &[u8] = b"BGLCM2\n";

fn xz(raw: &[u8]) -> Result<Vec<u8>> {
    let stream = Stream::new_easy_encoder(9 | (1 << 31), Check::Crc64)?;
    let mut enc = XzEncoder::new_stream(Vec::new(), stream);
    enc.write_all(raw)?;
    Ok(enc.finish()?)
}
fn unxz(raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    XzDecoder::new(raw).read_to_end(&mut out)?;
    Ok(out)
}
fn joined<I, T>(items: I) -> Vec<u8>
where
    I: IntoIterator<Item = T>,
    T: AsRef<[u8]>,
{
    let mut out = Vec::new();
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            out.push(b'\n');
        }
        out.extend_from_slice(item.as_ref());
    }
    out
}
fn tagged(value: Option<&[u8]>) -> Vec<u8> {
    match value {
        None => vec![b'-'],
        Some(v) => {
            let mut out = Vec::with_capacity(v.len() + 1);
            out.push(b'+');
            out.extend_from_slice(v);
            out
        }
    }
}
fn records(bytes: &[u8]) -> Vec<&[u8]> {
    if bytes.is_empty() {
        vec![]
    } else {
        bytes.split(|&b| b == b'\n').collect()
    }
}
fn untagged(bytes: &[u8]) -> Result<Option<&[u8]>> {
    match bytes {
        b"-" => Ok(None),
        [b'+', rest @ ..] => Ok(Some(rest)),
        _ => Err("invalid tagged record".into()),
    }
}
fn frame(writer: &mut impl Write, bytes: &[u8]) -> Result<()> {
    writer.write_all(&u32::try_from(bytes.len())?.to_le_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}
fn take_frame<'a>(input: &mut &'a [u8]) -> Result<&'a [u8]> {
    let len =
        u32::from_le_bytes(input.get(..4).ok_or("truncated frame size")?.try_into()?) as usize;
    let bytes = input.get(4..4 + len).ok_or("truncated frame")?;
    *input = &input[4 + len..];
    Ok(bytes)
}

fn pack(src: &Path, out: &Path, hybrid: bool) -> Result<()> {
    let start = Instant::now();
    let raw = fs::read(src)?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    let parser::Parsed {
        mut rows,
        names,
        heads,
        shapes,
        fallback,
        offsets,
        newline,
    } = parser::parse(&raw)?;
    let parse_seconds = start.elapsed().as_secs_f64();
    let lanes = std::iter::once(0)
        .chain(shapes.iter().map(|s| parser::lanes(s.as_deref())))
        .collect();
    let dict = parser::dictionaries(&names, lanes);
    let selected_templates = shapes.len();
    let fallback_rows = fallback.len();
    let n = rows.node.len();
    // Keep Python's spacing as well as its field order for byte-for-byte comparisons.
    let offset_json = offsets
        .iter()
        .map(|(o, n)| format!("[{o}, {n}]"))
        .collect::<Vec<_>>()
        .join(", ");
    let header = format!(
        "{{\"rows\": {n}, \"newline\": {newline}, \"sha256\": \"{digest}\", \"offsets\": [{offset_json}]}}"
    );
    let aux = [
        xz(&joined(names))?,
        xz(&joined(heads.iter().map(|h| {
            let mut f = h.0.to_vec();
            if let Some(other) = h.1 {
                f.push(other);
            }
            f.join(&b' ')
        })))?,
        xz(&joined(shapes.iter().map(|s| tagged(s.as_deref()))))?,
        xz(&joined(fallback.iter().map(|s| tagged(*s))))?,
    ];
    drop(raw);
    let frontend_seconds = start.elapsed().as_secs_f64();
    let tick = Instant::now();
    let clock = hybrid.then(|| clock::encode(&rows.ticks, 3, 10));
    let clock_seconds = tick.elapsed().as_secs_f64();
    let tick = Instant::now();
    let cm = mixer::encode_rows(&mut rows, &dict, hybrid, false);
    let mixer_seconds = tick.elapsed().as_secs_f64();
    let mut writer = BufWriter::new(fs::File::create(out)?);
    writer.write_all(if hybrid { HYBRID } else { MAGIC })?;
    frame(&mut writer, header.as_bytes())?;
    frame(&mut writer, &cm)?;
    for stream in &aux {
        frame(&mut writer, stream)?;
    }
    if let Some(clock) = &clock {
        frame(&mut writer, clock)?;
    }
    writer.flush()?;
    println!(
        "{}",
        json!({
            "rows": n, "total_bytes": fs::metadata(out)?.len(), "sha256":digest,
            "selected_templates":selected_templates, "fallback_rows":fallback_rows,
            "parse_seconds":parse_seconds, "frontend_seconds":frontend_seconds,
            "clock_seconds":clock_seconds, "mixer_seconds":mixer_seconds,
            "seconds":start.elapsed().as_secs_f64(),
            "streams":{"header":7+4+header.len(),"cm":4+cm.len(),"nodes":4+aux[0].len(),"heads":4+aux[1].len(),"shapes":4+aux[2].len(),"fallback":4+aux[3].len(),"clock":clock.as_ref().map_or(0,|b|4+b.len())}
        })
    );
    Ok(())
}

fn unpack(src: &Path, out: &Path) -> Result<()> {
    let start = Instant::now();
    let raw = fs::read(src)?;
    let hybrid = raw.starts_with(HYBRID);
    if !hybrid && !raw.starts_with(MAGIC) {
        return Err("not a bgl_cm archive".into());
    }
    let mut input = &raw[7..];
    let header: serde_json::Value = serde_json::from_slice(take_frame(&mut input)?)?;
    let n = usize::try_from(header["rows"].as_u64().ok_or("missing row count")?)?;
    if n == 0 || n > 16_000_000 {
        return Err("row count outside experiment budget".into());
    }
    let newline = header["newline"].as_bool().ok_or("missing newline flag")?;
    let expected = header["sha256"].as_str().ok_or("missing checksum")?;
    let offsets: Vec<(i64, usize)> = serde_json::from_value(header["offsets"].clone())?;
    if offsets.iter().any(|r| r.1 == 0)
        || offsets.iter().try_fold(0usize, |s, r| s.checked_add(r.1)) != Some(n)
    {
        return Err("invalid offset runs".into());
    }
    let cm = take_frame(&mut input)?;
    let aux = [
        unxz(take_frame(&mut input)?)?,
        unxz(take_frame(&mut input)?)?,
        unxz(take_frame(&mut input)?)?,
        unxz(take_frame(&mut input)?)?,
    ];
    let names = records(&aux[0]);
    let heads: Vec<Vec<&[u8]>> = records(&aux[1])
        .iter()
        .map(|r| r.split(|&b| b == b' ').collect())
        .collect();
    if names.is_empty() || heads.is_empty() || heads.iter().any(|h| h.len() != 4 && h.len() != 5) {
        return Err("invalid dictionaries".into());
    }
    let shapes = records(&aux[2])
        .into_iter()
        .map(untagged)
        .collect::<Result<Vec<_>>>()?;
    let fallback = records(&aux[3])
        .into_iter()
        .map(untagged)
        .collect::<Result<Vec<_>>>()?;
    let lanes: Vec<_> = std::iter::once(0)
        .chain(shapes.iter().map(|s| parser::lanes(*s)))
        .collect();
    let dict = parser::dictionaries(&names, lanes);
    if cm.len() < 16
        || (u64::from_le_bytes(cm[..8].try_into()?) & !(1 << 63)) != n as u64
        || u64::from_le_bytes(cm[8..16].try_into()?) != heads.len() as u64
    {
        return Err("archive and mixer headers disagree".into());
    }
    let ticks = if hybrid {
        let budget = DecodeBudget {
            max_records_per_block: 16_000_000,
            max_single_alloc_bytes: 512 << 20,
            ..DecodeBudget::default()
        };
        Some(CoreDecoder::decode_ints_bytes(
            take_frame(&mut input)?,
            &budget,
        )?)
    } else {
        None
    };
    if !input.is_empty() {
        return Err("trailing archive data".into());
    }
    let rows = mixer::decode_rows(cm, &dict, ticks, false)?;
    let codec_seconds = start.elapsed().as_secs_f64();
    let mut writer = BufWriter::new(fs::File::create(out)?);
    let mut digest = Sha256::new();
    let mut buf = Vec::with_capacity(1 << 20);
    let (mut pos, mut fb, mut run, mut remaining) = (0, 0, 0, offsets[0].1);
    let mut last = None;
    let (mut prefix, mut date) = (Vec::new(), Vec::new());
    for i in 0..n {
        let wall = rows.ticks[i].div_euclid(1_000_000);
        let us = rows.ticks[i].rem_euclid(1_000_000);
        if last != Some(wall) {
            let dt = chrono::DateTime::from_timestamp(wall, 0).ok_or("invalid timestamp")?;
            prefix.clear();
            date.clear();
            write!(
                prefix,
                "{:04}-{:02}-{:02}-{:02}.{:02}.{:02}",
                dt.year(),
                dt.month(),
                dt.day(),
                dt.hour(),
                dt.minute(),
                dt.second()
            )?;
            write!(date, "{:04}.{:02}.{:02}", dt.year(), dt.month(), dt.day())?;
            last = Some(wall);
        }
        let name = *names.get(rows.node[i]).ok_or("invalid node")?;
        let head = heads.get(rows.head[i]).ok_or("invalid head")?;
        let epoch = wall.checked_sub(offsets[run].0).ok_or("epoch overflow")?;
        buf.extend_from_slice(head[0]);
        write!(buf, " {epoch} ")?;
        buf.extend_from_slice(&date);
        buf.push(b' ');
        buf.extend_from_slice(name);
        buf.push(b' ');
        buf.extend_from_slice(&prefix);
        write!(buf, ".{us:06} ")?;
        buf.extend_from_slice(if head.len() == 5 { head[4] } else { name });
        for field in &head[1..4] {
            buf.push(b' ');
            buf.extend_from_slice(field);
        }
        let sel = rows.sel[i];
        if sel > 0 {
            let shape = shapes.get(sel - 1).ok_or("invalid selector")?;
            let count = dict.lanes[sel];
            let values = rows
                .nums
                .get(pos..pos + count)
                .ok_or("missing numeric lanes")?;
            if let Some(shape) = shape {
                buf.push(b' ');
                parser::render(shape, values, &mut buf)?;
            }
            pos += count;
        } else {
            if let Some(body) = fallback.get(fb).ok_or("missing fallback")? {
                buf.push(b' ');
                buf.extend_from_slice(body);
            }
            fb += 1;
        }
        if i + 1 < n || newline {
            buf.push(b'\n');
        }
        remaining -= 1;
        if remaining == 0 && i + 1 < n {
            run += 1;
            remaining = offsets[run].1;
        }
        if buf.len() >= 1 << 20 {
            digest.update(&buf);
            writer.write_all(&buf)?;
            buf.clear();
        }
    }
    if pos != rows.nums.len() || fb != fallback.len() {
        return Err("unused numeric or fallback rows".into());
    }
    digest.update(&buf);
    writer.write_all(&buf)?;
    writer.flush()?;
    let actual = format!("{:x}", digest.finalize());
    if actual != expected {
        return Err(format!("sha256 mismatch: restored {actual}, archive {expected}").into());
    }
    println!(
        "{}",
        json!({"rows":n,"sha256":"ok","codec_seconds":codec_seconds,"seconds":start.elapsed().as_secs_f64()})
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() == 4 && args[1] == "unpack" {
        return unpack(Path::new(&args[2]), Path::new(&args[3]));
    }
    if (args.len() == 4 || (args.len() == 5 && args[4] == "--hybrid")) && args[1] == "pack" {
        return pack(Path::new(&args[2]), Path::new(&args[3]), args.len() == 5);
    }
    Err("usage: bgl-native pack INPUT OUTPUT [--hybrid] | unpack INPUT OUTPUT".into())
}
