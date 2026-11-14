use crate::Result;
use crate::mixer::{Dict, Rows};
use chrono::NaiveDate;
use std::collections::HashMap;
use std::io::Write;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Head<'a>(pub [&'a [u8]; 4], pub Option<&'a [u8]>);

pub(crate) struct Parsed<'a> {
    pub rows: Rows,
    pub names: Vec<&'a [u8]>,
    pub heads: Vec<Head<'a>>,
    pub shapes: Vec<Option<Vec<u8>>>,
    pub fallback: Vec<Option<&'a [u8]>>,
    pub offsets: Vec<(i64, usize)>,
    pub newline: bool,
}

fn word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
fn literal(out: &mut Vec<u8>, bytes: &[u8]) {
    for &b in bytes {
        out.push(b);
        if b == b'%' {
            out.push(b);
        }
    }
}
fn number(bytes: &[u8], radix: u32) -> i64 {
    bytes.iter().fold(0, |v, &b| {
        v * radix as i64 + (b as char).to_digit(radix).unwrap() as i64
    })
}

// Preserve the reference regex's alternative priority and ASCII word boundaries.
pub(crate) fn tokenize(body: Option<&[u8]>) -> (Option<Vec<u8>>, Vec<i64>) {
    let Some(body) = body else {
        return (None, vec![]);
    };
    let (mut shape, mut values, mut at, mut literal_start) = (Vec::new(), Vec::new(), 0, 0);
    while at < body.len() {
        let mut end = at;
        if body[at] == b'0' && body.get(at + 1).is_some_and(|b| matches!(b, b'x' | b'X')) {
            end = at + 2;
            while end < body.len() && body[end].is_ascii_hexdigit() {
                end += 1;
            }
            if end == at + 2 {
                end = at;
            }
        }
        if end == at && (at == 0 || !word(body[at - 1])) && body[at].is_ascii_hexdigit() {
            let mut hex_end = at;
            while hex_end < body.len() && body[hex_end].is_ascii_hexdigit() {
                hex_end += 1;
            }
            if hex_end - at >= 8
                && (hex_end == body.len() || !word(body[hex_end]))
                && (body[at] == b'0' || body[at..hex_end].iter().any(u8::is_ascii_alphabetic))
            {
                end = hex_end;
            }
        }
        if end == at && body[at].is_ascii_digit() {
            end = at + 1;
            while end < body.len() && body[end].is_ascii_digit() {
                end += 1;
            }
        }
        if end == at {
            at += 1;
            continue;
        }
        literal(&mut shape, &body[literal_start..at]);
        let token = &body[at..end];
        let prefixed = token.starts_with(b"0x") || token.starts_with(b"0X");
        let hex = prefixed
            || (token.len() >= 8 && (token[0] == b'0' || !token.iter().all(u8::is_ascii_digit)));
        if hex {
            let digits = if prefixed { &token[2..] } else { token };
            let upper = digits.iter().any(u8::is_ascii_uppercase);
            let mixed = upper && digits.iter().any(u8::is_ascii_lowercase);
            if mixed || (prefixed && digits.len() > 15) {
                shape.extend_from_slice(token);
            } else {
                if prefixed {
                    shape.extend_from_slice(&token[..2]);
                }
                let kind = if upper { 'X' } else { 'x' };
                for chunk in digits.chunks(if prefixed { 15 } else { 14 }) {
                    write!(shape, "%0{}{kind}", chunk.len()).unwrap();
                    values.push(number(chunk, 16));
                }
            }
        } else if token.len() > 18 {
            shape.extend_from_slice(token);
        } else {
            if token.len() > 1 && token[0] == b'0' {
                write!(shape, "%0{}d", token.len()).unwrap();
            } else {
                shape.extend_from_slice(b"%d");
            }
            values.push(number(token, 10));
        }
        at = end;
        literal_start = end;
    }
    literal(&mut shape, &body[literal_start..]);
    (Some(shape), values)
}

pub(crate) fn lanes(shape: Option<&[u8]>) -> usize {
    let Some(shape) = shape else {
        return 0;
    };
    let (mut at, mut n) = (0, 0);
    while at < shape.len() {
        if shape[at] == b'%' {
            if shape.get(at + 1) == Some(&b'%') {
                at += 2;
                continue;
            }
            n += 1;
        }
        at += 1;
    }
    n
}

pub(crate) fn render(shape: &[u8], values: &[i64], out: &mut Vec<u8>) -> Result<()> {
    let (mut at, mut lane) = (0, 0);
    while at < shape.len() {
        if shape[at] != b'%' {
            out.push(shape[at]);
            at += 1;
            continue;
        }
        at += 1;
        if shape.get(at) == Some(&b'%') {
            out.push(b'%');
            at += 1;
            continue;
        }
        let mut width = 0usize;
        while shape.get(at).is_some_and(u8::is_ascii_digit) {
            width = width
                .checked_mul(10)
                .and_then(|w| w.checked_add((shape[at] - b'0') as usize))
                .ok_or("bad shape width")?;
            if width > 1024 {
                return Err("shape width exceeds tokenizer limit".into());
            }
            at += 1;
        }
        let value = *values.get(lane).ok_or("missing numeric lane")?;
        match shape.get(at) {
            Some(b'd') => write!(out, "{value:0width$}")?,
            Some(b'x') => write!(out, "{value:0width$x}")?,
            Some(b'X') => write!(out, "{value:0width$X}")?,
            _ => return Err("invalid shape conversion".into()),
        }
        at += 1;
        lane += 1;
    }
    if lane != values.len() {
        return Err("extra numeric lanes".into());
    }
    Ok(())
}

fn second(pre: &[u8]) -> Result<i64> {
    if pre.len() != 19
        || [4, 7, 10].iter().any(|&i| pre[i] != b'-')
        || [13, 16].iter().any(|&i| pre[i] != b'.')
    {
        return Err("non-canonical clock".into());
    }
    let digits = |s: &[u8]| -> Result<u32> {
        if !s.iter().all(u8::is_ascii_digit) {
            return Err("non-canonical clock".into());
        }
        Ok(number(s, 10) as u32)
    };
    let year = digits(&pre[..4])?;
    if year == 0 {
        return Err("non-canonical clock".into());
    }
    let date = NaiveDate::from_ymd_opt(year as i32, digits(&pre[5..7])?, digits(&pre[8..10])?)
        .ok_or("non-canonical clock")?;
    let hour = digits(&pre[11..13])?;
    let minute = digits(&pre[14..16])?;
    let second = digits(&pre[17..19])?;
    let dt = date
        .and_hms_opt(hour, minute, second)
        .ok_or("non-canonical clock")?;
    Ok(dt.and_utc().timestamp())
}

fn intern<'a>(
    map: &mut HashMap<&'a [u8], usize>,
    list: &mut Vec<&'a [u8]>,
    key: &'a [u8],
) -> usize {
    *map.entry(key).or_insert_with(|| {
        let id = list.len();
        list.push(key);
        id
    })
}

pub(crate) fn dictionaries(names: &[&[u8]], lanes: Vec<usize>) -> Dict {
    let mut maps: [HashMap<Vec<u8>, usize>; 4] = std::array::from_fn(|_| HashMap::new());
    let mut cols: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::with_capacity(names.len()));
    for name in names {
        let parts: Vec<_> = name.split(|&b| b == b'-').collect();
        let keys = [
            parts[..parts.len().min(3)].join(&b'-'),
            parts[..parts.len().min(2)].join(&b'-'),
            parts[..1].join(&b'-'),
            parts[parts.len().min(3)..].join(&b'-'),
        ];
        for (i, key) in keys.into_iter().enumerate() {
            let next = maps[i].len();
            cols[i].push(*maps[i].entry(key).or_insert(next));
        }
    }
    let [card, mid, rack, pos] = cols;
    Dict {
        card,
        mid,
        rack,
        pos,
        lanes,
    }
}

pub(crate) fn parse(raw: &[u8]) -> Result<Parsed<'_>> {
    if raw.is_empty() {
        return Err("empty input".into());
    }
    let newline = raw.ends_with(b"\n");
    let n = memchr::memchr_iter(b'\n', raw).count() + usize::from(!newline);
    let (mut node_ids, mut names, mut head_ids, mut heads) =
        (HashMap::new(), Vec::new(), HashMap::new(), Vec::new());
    let mut shape_ids = HashMap::new();
    let mut shapes = Vec::new();
    let mut shape_counts = Vec::new();
    let mut body_ids = HashMap::new();
    let mut bodies: Vec<(Option<&[u8]>, usize, Vec<i64>)> = Vec::new();
    let mut row_bodies = Vec::with_capacity(n);
    let mut seconds: HashMap<&[u8], (i64, [u8; 10], Option<&[u8]>, i64)> = HashMap::new();
    let mut offsets: Vec<(i64, usize)> = Vec::new();
    let mut rows = Rows {
        node: Vec::with_capacity(n),
        head: Vec::with_capacity(n),
        sel: Vec::with_capacity(n),
        ticks: Vec::with_capacity(n),
        nums: Vec::new(),
    };
    let mut line_start = 0;
    for (r, end) in memchr::memchr_iter(b'\n', raw)
        .chain((!newline).then_some(raw.len()))
        .enumerate()
    {
        let line = &raw[line_start..end];
        line_start = end + 1;
        let mut fields = [b"".as_slice(); 10];
        let mut count = 0;
        let mut start = 0;
        for space in memchr::memchr_iter(b' ', line).take(9) {
            fields[count] = &line[start..space];
            count += 1;
            start = space + 1;
        }
        fields[count] = &line[start..];
        count += 1;
        if count < 9 {
            return Err(format!("line {}: fewer than 9 fields", r + 1).into());
        }
        let f = fields;
        let clock = f[4];
        if clock.len() != 26 || clock[19] != b'.' || !clock[20..].iter().all(u8::is_ascii_digit) {
            return Err(format!("line {}: clock or date field not canonical", r + 1).into());
        }
        let pre = &clock[..19];
        let sec = match seconds.entry(pre) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let wall = second(pre)?;
                let mut date: [u8; 10] = pre[..10].try_into()?;
                date[4] = b'.';
                date[7] = b'.';
                e.insert((wall, date, None, 0))
            }
        };
        if f[2] != sec.1 {
            return Err(format!("line {}: clock or date field not canonical", r + 1).into());
        }
        if Some(f[1]) != sec.2 {
            let epoch: i64 = std::str::from_utf8(f[1])?.parse()?;
            if epoch.to_string().as_bytes() != f[1] {
                return Err(format!("line {}: epoch field not canonical", r + 1).into());
            }
            sec.2 = Some(f[1]);
            sec.3 = sec.0.checked_sub(epoch).ok_or("epoch offset overflow")?;
        }
        let tick = sec.0 * 1_000_000 + number(&clock[20..], 10);
        if rows.ticks.last().is_some_and(|&prev| prev > tick) {
            return Err("clock goes backwards".into());
        }
        rows.ticks.push(tick);
        if let Some(last) = offsets.last_mut().filter(|last| last.0 == sec.3) {
            last.1 += 1;
        } else {
            offsets.push((sec.3, 1));
        }
        rows.node.push(intern(&mut node_ids, &mut names, f[3]));
        let head = Head([f[0], f[6], f[7], f[8]], (f[3] != f[5]).then_some(f[5]));
        rows.head.push(*head_ids.entry(head).or_insert_with(|| {
            let id = heads.len();
            heads.push(head);
            id
        }));
        let body = (count == 10).then_some(f[9]);
        let id = *body_ids.entry(body).or_insert_with(|| {
            let (shape, vals) = tokenize(body);
            let sid = *shape_ids.entry(shape.clone()).or_insert_with(|| {
                let id = shapes.len();
                shapes.push(shape);
                shape_counts.push(0usize);
                id
            });
            let id = bodies.len();
            bodies.push((body, sid, vals));
            id
        });
        row_bodies.push(id);
        shape_counts[bodies[id].1] += 1;
    }
    let mut selection = vec![0; shapes.len()];
    let mut selected = Vec::new();
    for (i, shape) in shapes.into_iter().enumerate() {
        if shape_counts[i] >= 16 {
            selection[i] = selected.len() + 1;
            selected.push(shape);
        }
    }
    let mut fallback = Vec::new();
    for id in row_bodies {
        let (body, shape, values) = &bodies[id];
        let sel = selection[*shape];
        rows.sel.push(sel);
        if sel == 0 {
            fallback.push(*body);
        } else {
            rows.nums.extend_from_slice(values);
        }
    }
    Ok(Parsed {
        rows,
        names,
        heads,
        shapes: selected,
        fallback,
        offsets,
        newline,
    })
}
