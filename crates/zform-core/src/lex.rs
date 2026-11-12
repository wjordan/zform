//! Fast, roundtrip-verified lexical parsers and formatters for observability & structured data:
//! exact padded/signed integers, exact decimal floats, ISO-8601/RFC3339/CLF/Syslog timestamps,
//! hex/UUIDv4/UUIDv7/ULID/W3C `traceparent` identifiers, IPv4/IPv4:port, emails, paths, and JSON escapes.

// ============================================================================
// 1. Exact Signed & Zero-Padded Integers
// ============================================================================

/// Parses an ASCII integer (`"42"`, `"-17"`, `"007"`, `"+04"`, `"-0"`) into `(value, fmt_tag)`
/// if and only if [`format_exact_int`] reproduces `src` byte-for-byte.
#[inline]
pub fn parse_exact_int(src: &[u8]) -> Option<(i64, u8)> {
    if src.is_empty() || src.len() > 20 {
        return None;
    }
    let (sign_mode, digits) = match src[0] {
        b'+' => (1u8, &src[1..]),
        b'-' => (2u8, &src[1..]),
        _ => (0u8, src),
    };
    if digits.is_empty() || digits.len() > 19 {
        return None;
    }
    let mut u = 0u64;
    for &b in digits {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        u = u.checked_mul(10)?.checked_add(d as u64)?;
    }
    let val = if sign_mode == 2 {
        if u == 0 {
            0i64
        } else if u == (1u64 << 63) {
            i64::MIN
        } else if u < (1u64 << 63) {
            -(u as i64)
        } else {
            return None;
        }
    } else {
        i64::try_from(u).ok()?
    };

    let has_leading_zero = digits.len() > 1 && digits[0] == b'0';
    let pad_width = if has_leading_zero {
        digits.len() as u8
    } else {
        0
    };
    if pad_width > 20 {
        return None;
    }
    let eff_sign = if sign_mode == 2 && val < 0 {
        0u8 // negative sign is already carried by `val < 0`
    } else {
        sign_mode // 0 = normal, 1 = explicit '+', 2 = negative zero ("-0", "-00")
    };
    let fmt = (pad_width << 2) | eff_sign;
    let mut check = [0u8; 24];
    let n = format_exact_int_buf(val, fmt, &mut check);
    if &check[..n] == src {
        Some((val, fmt))
    } else {
        None
    }
}

#[inline]
pub fn format_exact_int(val: i64, fmt: u8, out: &mut Vec<u8>) {
    let mut buf = [0u8; 24];
    let n = format_exact_int_buf(val, fmt, &mut buf);
    out.extend_from_slice(&buf[..n]);
}

#[inline]
pub fn format_exact_int_buf(val: i64, fmt: u8, out: &mut [u8; 24]) -> usize {
    let sign_mode = fmt & 0x3;
    let pad_width = (fmt >> 2) as usize;
    let mut pos = 0usize;
    let mag: u64 = if val < 0 {
        out[pos] = b'-';
        pos += 1;
        val.wrapping_neg() as u64
    } else {
        if sign_mode == 1 {
            out[pos] = b'+';
            pos += 1;
        } else if sign_mode == 2 {
            out[pos] = b'-';
            pos += 1;
        }
        val as u64
    };
    let mut tmp = [0u8; 20];
    let mut t = 20usize;
    let mut x = mag;
    loop {
        t -= 1;
        tmp[t] = b'0' + (x % 10) as u8;
        x /= 10;
        if x == 0 {
            break;
        }
    }
    let dlen = 20 - t;
    if pad_width > dlen && pad_width <= 20 {
        let zeros = pad_width - dlen;
        for _ in 0..zeros {
            out[pos] = b'0';
            pos += 1;
        }
    }
    out[pos..pos + dlen].copy_from_slice(&tmp[t..]);
    pos + dlen
}

// ============================================================================
// 2. Exact Decimal Floats (Metrics-in-Logs, Latencies, Coordinates)
// ============================================================================

/// Parses fixed-point ASCII decimal floats (`"12.340"`, `"-0.005"`, `"-0.0"`, `"+3.5"`)
/// into `(scaled_mantissa, spec_u16)` with 100% byte-exact roundtrip verification.
#[inline]
pub fn parse_exact_decimal(src: &[u8]) -> Option<(i64, u16)> {
    if src.len() < 3 || src.len() > 21 {
        return None;
    }
    let (sign_mode, rest) = match src[0] {
        b'+' => (1u16, &src[1..]),
        b'-' => (2u16, &src[1..]),
        _ => (0u16, src),
    };
    let dot = rest.iter().position(|&b| b == b'.')?;
    let int_part = &rest[..dot];
    let frac_part = &rest[dot + 1..];
    if int_part.is_empty() || frac_part.is_empty() || frac_part.len() > 12 || int_part.len() > 15 {
        return None;
    }
    let mut u = 0u64;
    for &b in int_part.iter().chain(frac_part.iter()) {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        u = u.checked_mul(10)?.checked_add(d as u64)?;
    }
    let mantissa = if sign_mode == 2 {
        if u == 0 {
            0i64
        } else if u < (1u64 << 63) {
            -(u as i64)
        } else {
            return None;
        }
    } else {
        i64::try_from(u).ok()?
    };
    let int_pad = if int_part.len() > 1 && int_part[0] == b'0' {
        int_part.len() as u16
    } else {
        0u16
    };
    let eff_sign = if sign_mode == 2 && mantissa < 0 {
        0u16
    } else {
        sign_mode
    };
    // spec layout: bits 0..4 = frac_len (1..=12), bits 4..8 = int_pad (0..=15), bits 8..10 = eff_sign
    let spec = (frac_part.len() as u16) | (int_pad << 4) | (eff_sign << 8);
    let mut check = Vec::with_capacity(24);
    format_exact_decimal(mantissa, spec, &mut check);
    if check.as_slice() == src {
        Some((mantissa, spec))
    } else {
        None
    }
}

pub fn format_exact_decimal(mantissa: i64, spec: u16, out: &mut Vec<u8>) {
    let frac_len = (spec & 0x0f) as usize;
    let int_pad = ((spec >> 4) & 0x0f) as usize;
    let sign_mode = (spec >> 8) & 0x03;
    if !(1..=12).contains(&frac_len) {
        return;
    }
    let mag: u64 = if mantissa < 0 {
        out.push(b'-');
        mantissa.wrapping_neg() as u64
    } else {
        if sign_mode == 1 {
            out.push(b'+');
        } else if sign_mode == 2 {
            out.push(b'-');
        }
        mantissa as u64
    };
    let mut digits = [b'0'; 24];
    let mut t = 24usize;
    let mut x = mag;
    loop {
        t -= 1;
        digits[t] = b'0' + (x % 10) as u8;
        x /= 10;
        if x == 0 {
            break;
        }
    }
    let total_digits = 24 - t;
    let min_total = frac_len + 1;
    let start = if total_digits < min_total {
        24 - min_total
    } else {
        t
    };
    let int_end = 24 - frac_len;
    let cur_int_len = int_end - start;
    if int_pad > cur_int_len && int_pad <= 15 {
        for _ in 0..(int_pad - cur_int_len) {
            out.push(b'0');
        }
    }
    out.extend_from_slice(&digits[start..int_end]);
    out.push(b'.');
    out.extend_from_slice(&digits[int_end..24]);
}

// ============================================================================
// 3. Canonical Timestamps (ISO-8601 / RFC3339, Apache CLF, Syslog RFC3164)
// ============================================================================

const MONTHS: [&[u8; 3]; 12] = [
    b"Jan", b"Feb", b"Mar", b"Apr", b"May", b"Jun", b"Jul", b"Aug", b"Sep", b"Oct", b"Nov", b"Dec",
];

#[inline]
fn parse_2d(b: &[u8]) -> Option<u32> {
    if b.len() < 2 {
        return None;
    }
    let d0 = b[0].wrapping_sub(b'0');
    let d1 = b[1].wrapping_sub(b'0');
    if d0 <= 9 && d1 <= 9 {
        Some((d0 as u32) * 10 + (d1 as u32))
    } else {
        None
    }
}

#[inline]
fn parse_4d(b: &[u8]) -> Option<i32> {
    if b.len() < 4 {
        return None;
    }
    let hi = parse_2d(&b[0..2])?;
    let lo = parse_2d(&b[2..4])?;
    Some((hi * 100 + lo) as i32)
}

/// Howard Hinnant civil-from-days / days-from-civil (exact proleptic Gregorian).
#[inline]
fn days_from_civil(y: i32, m: u32, d: u32) -> Option<i64> {
    if !(1970..=2260).contains(&y) || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let is_leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    let max_d = match m {
        2 => {
            if is_leap {
                29
            } else {
                28
            }
        }
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if d > max_d {
        return None;
    }
    let y = y as i64 - i64::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = (if m > 2 { m - 3 } else { m + 9 }) as u64;
    let doy = (153 * mp + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146097 + doe as i64 - 719468)
}

#[inline]
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + i64::from(m <= 2);
    (y as i32, m as u32, d as u32)
}

/// Layout bitfield (`u32`) for lossless timestamp reconstruction:
/// - `bits 0..2`: kind (`0 = Iso8601`, `1 = ApacheClf`, `2 = Syslog3164`)
/// - `bit 2`: ISO date/time separator (`0 = 'T'`, `1 = ' '`), or Syslog space-padded single-digit day (`' 5'` vs `'05'`)
/// - `bit 3`: fractional separator (`0 = '.'`, `1 = ','`)
/// - `bits 4..8`: fractional digit width (`0..=9`)
/// - `bits 8..11`: tz style (`0 = none`, `1 = 'Z'`, `2 = 'z'`, `3 = ±HH:MM`, `4 = ±HHMM`, `5 = -00:00`, `6 = -0000`)
/// - `bits 11..19`: signed tz offset in 15-minute increments (`i8 as u8`, `-64..=64`)
pub fn parse_timestamp(src: &[u8]) -> Option<(i64, u32)> {
    let cand = if src.len() >= 19
        && src[4] == b'-'
        && src[7] == b'-'
        && src[13] == b':'
        && src[16] == b':'
    {
        parse_iso8601(src)
    } else if src.len() == 26 && src[2] == b'/' && src[6] == b'/' && src[11] == b':' {
        parse_clf(src)
    } else if src.len() == 15
        && src[3] == b' '
        && src[6] == b' '
        && src[9] == b':'
        && src[12] == b':'
    {
        parse_syslog(src)
    } else {
        None
    }?;
    let mut check = Vec::with_capacity(36);
    format_timestamp(cand.0, cand.1, &mut check);
    if check.as_slice() == src {
        Some(cand)
    } else {
        None
    }
}

fn parse_iso8601(src: &[u8]) -> Option<(i64, u32)> {
    if src.len() > 35 {
        return None;
    }
    let y = parse_4d(&src[0..4])?;
    let m = parse_2d(&src[5..7])?;
    let d = parse_2d(&src[8..10])?;
    let sep_space = match src[10] {
        b'T' => 0u32,
        b' ' => 1u32,
        _ => return None,
    };
    let hh = parse_2d(&src[11..13])?;
    let mm = parse_2d(&src[14..16])?;
    let ss = parse_2d(&src[17..19])?;
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    let days = days_from_civil(y, m, d)?;
    let mut pos = 19usize;
    let mut frac_comma = 0u32;
    let mut frac_len = 0u32;
    let mut subsec_ns = 0i64;
    if pos < src.len() && (src[pos] == b'.' || src[pos] == b',') {
        frac_comma = u32::from(src[pos] == b',');
        pos += 1;
        let fstart = pos;
        while pos < src.len() && src[pos].is_ascii_digit() {
            pos += 1;
        }
        let flen = pos - fstart;
        if !(1..=9).contains(&flen) {
            return None;
        }
        frac_len = flen as u32;
        let mut v = 0i64;
        for &b in &src[fstart..pos] {
            v = v * 10 + (b - b'0') as i64;
        }
        for _ in flen..9 {
            v *= 10;
        }
        subsec_ns = v;
    }
    let rem = &src[pos..];
    let (tz_style, tz_q15): (u32, i8) = if rem.is_empty() {
        (0, 0)
    } else if rem == b"Z" {
        (1, 0)
    } else if rem == b"z" {
        (2, 0)
    } else if (rem.len() == 6 && rem[3] == b':') || rem.len() == 5 {
        let neg = match rem[0] {
            b'+' => false,
            b'-' => true,
            _ => return None,
        };
        let th = parse_2d(&rem[1..3])? as i32;
        let tm = if rem.len() == 6 {
            parse_2d(&rem[4..6])? as i32
        } else {
            parse_2d(&rem[3..5])? as i32
        };
        if th > 15 || tm > 59 || tm % 15 != 0 {
            return None;
        }
        let q = (th * 4 + tm / 15) * if neg { -1 } else { 1 };
        let style = if q == 0 && neg {
            if rem.len() == 6 { 5 } else { 6 }
        } else if rem.len() == 6 {
            3
        } else {
            4
        };
        (style, q as i8)
    } else {
        return None;
    };

    let local_secs = days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64);
    let utc_secs = local_secs - (tz_q15 as i64) * 900;
    let epoch_ns = utc_secs
        .checked_mul(1_000_000_000)?
        .checked_add(subsec_ns)?;
    let layout = (sep_space << 2)
        | (frac_comma << 3)
        | (frac_len << 4)
        | (tz_style << 8)
        | (((tz_q15 as u8) as u32) << 11);
    Some((epoch_ns, layout))
}

fn parse_clf(src: &[u8]) -> Option<(i64, u32)> {
    // DD/Mmm/YYYY:HH:MM:SS ±HHMM (len == 26)
    let d = parse_2d(&src[0..2])?;
    let mon = MONTHS.iter().position(|m| *m == &src[3..6])? as u32 + 1;
    let y = parse_4d(&src[7..11])?;
    if src[14] != b':' || src[17] != b':' || src[20] != b' ' {
        return None;
    }
    let hh = parse_2d(&src[12..14])?;
    let mm = parse_2d(&src[15..17])?;
    let ss = parse_2d(&src[18..20])?;
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    let neg = match src[21] {
        b'+' => false,
        b'-' => true,
        _ => return None,
    };
    let th = parse_2d(&src[22..24])? as i32;
    let tm = parse_2d(&src[24..26])? as i32;
    if th > 15 || tm > 59 || tm % 15 != 0 {
        return None;
    }
    let q = (th * 4 + tm / 15) * if neg { -1 } else { 1 };
    let tz_style = if q == 0 && neg { 6u32 } else { 4u32 };
    let days = days_from_civil(y, mon, d)?;
    let utc_secs =
        days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64) - (q as i64) * 900;
    let epoch_ns = utc_secs.checked_mul(1_000_000_000)?;
    let layout = 1u32 | (tz_style << 8) | (((q as u8) as u32) << 11);
    Some((epoch_ns, layout))
}

fn parse_syslog(src: &[u8]) -> Option<(i64, u32)> {
    // Mmm [ D]D HH:MM:SS (len == 15) — anchored on leap year 2000 so Feb 29 is valid too!
    let mon = MONTHS.iter().position(|m| *m == &src[0..3])? as u32 + 1;
    let (space_pad, d) = if src[4] == b' ' {
        let d1 = src[5].wrapping_sub(b'0');
        if !(1..=9).contains(&d1) {
            return None;
        }
        (1u32, d1 as u32)
    } else {
        (0u32, parse_2d(&src[4..6])?)
    };
    let hh = parse_2d(&src[7..9])?;
    let mm = parse_2d(&src[10..12])?;
    let ss = parse_2d(&src[13..15])?;
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    let days = days_from_civil(2000, mon, d)?;
    let epoch_ns =
        (days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64)) * 1_000_000_000;
    let layout = 2u32 | (space_pad << 2);
    Some((epoch_ns, layout))
}

#[inline]
fn push_2d(out: &mut Vec<u8>, v: u32) {
    out.push(b'0' + ((v / 10) % 10) as u8);
    out.push(b'0' + (v % 10) as u8);
}

#[inline]
fn push_4d(out: &mut Vec<u8>, v: i32) {
    let v = v.unsigned_abs();
    out.push(b'0' + ((v / 1000) % 10) as u8);
    out.push(b'0' + ((v / 100) % 10) as u8);
    out.push(b'0' + ((v / 10) % 10) as u8);
    out.push(b'0' + (v % 10) as u8);
}

pub fn format_timestamp(epoch_ns: i64, layout: u32, out: &mut Vec<u8>) {
    let kind = layout & 0x03;
    let flag2 = (layout >> 2) & 1;
    let frac_comma = (layout >> 3) & 1;
    let frac_len = ((layout >> 4) & 0x0f) as usize;
    let tz_style = (layout >> 8) & 0x07;
    let tz_q15 = ((layout >> 11) & 0xff) as u8 as i8;

    let utc_secs = epoch_ns.div_euclid(1_000_000_000);
    let subsec_ns = epoch_ns.rem_euclid(1_000_000_000) as u64;
    let local_secs = utc_secs + (tz_q15 as i64) * 900;
    let days = local_secs.div_euclid(86_400);
    let sod = local_secs.rem_euclid(86_400) as u32;
    let (y, m, d) = civil_from_days(days);
    let hh = sod / 3600;
    let mm = (sod % 3600) / 60;
    let ss = sod % 60;

    match kind {
        0 => {
            push_4d(out, y);
            out.push(b'-');
            push_2d(out, m);
            out.push(b'-');
            push_2d(out, d);
            out.push(if flag2 != 0 { b' ' } else { b'T' });
            push_2d(out, hh);
            out.push(b':');
            push_2d(out, mm);
            out.push(b':');
            push_2d(out, ss);
            if (1..=9).contains(&frac_len) {
                out.push(if frac_comma != 0 { b',' } else { b'.' });
                let mut fbuf = [b'0'; 9];
                let mut x = subsec_ns;
                for i in (0..9).rev() {
                    fbuf[i] = b'0' + (x % 10) as u8;
                    x /= 10;
                }
                out.extend_from_slice(&fbuf[..frac_len]);
            }
            emit_tz(out, tz_style, tz_q15);
        }
        1 => {
            push_2d(out, d);
            out.push(b'/');
            out.extend_from_slice(MONTHS[(m.saturating_sub(1) as usize).min(11)]);
            out.push(b'/');
            push_4d(out, y);
            out.push(b':');
            push_2d(out, hh);
            out.push(b':');
            push_2d(out, mm);
            out.push(b':');
            push_2d(out, ss);
            out.push(b' ');
            emit_tz(out, tz_style, tz_q15);
        }
        2 => {
            out.extend_from_slice(MONTHS[(m.saturating_sub(1) as usize).min(11)]);
            out.push(b' ');
            if flag2 != 0 && d < 10 {
                out.push(b' ');
                out.push(b'0' + d as u8);
            } else {
                push_2d(out, d);
            }
            out.push(b' ');
            push_2d(out, hh);
            out.push(b':');
            push_2d(out, mm);
            out.push(b':');
            push_2d(out, ss);
        }
        _ => {}
    }
}

fn emit_tz(out: &mut Vec<u8>, tz_style: u32, tz_q15: i8) {
    match tz_style {
        1 => out.push(b'Z'),
        2 => out.push(b'z'),
        3..=6 => {
            let neg = tz_q15 < 0 || tz_style == 5 || tz_style == 6;
            out.push(if neg { b'-' } else { b'+' });
            let q = tz_q15.unsigned_abs() as u32;
            push_2d(out, q / 4);
            if tz_style == 3 || tz_style == 5 {
                out.push(b':');
            }
            push_2d(out, (q % 4) * 15);
        }
        _ => {}
    }
}

// ============================================================================
// 4. Observability Identifiers (Hex, UUIDv4/v7, ULID, W3C traceparent, IPv4)
// ============================================================================

#[inline]
pub fn hex_val_lower(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    }
}

#[inline]
pub fn hex_val_upper(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parses uniform-case even-length hex (`4..=64` chars -> `2..=32` bytes).
/// Note: All-digit strings (`"12345678"`) are still valid lowercase hex (`upper = false`),
/// though integer codecs take precedence unless mixed with hex rows in the same column!
pub fn parse_hex_exact(src: &[u8]) -> Option<(bool, [u8; 32], u8)> {
    if src.len() < 4 || src.len() > 64 || !src.len().is_multiple_of(2) {
        return None;
    }
    let has_upper = src.iter().any(|b| b.is_ascii_uppercase());
    let has_lower = src.iter().any(|b| b.is_ascii_lowercase());
    if has_upper && has_lower {
        return None; // Mixed-case hex cannot roundtrip from 1 case bit
    }
    let mut out = [0u8; 32];
    let n = src.len() / 2;
    for i in 0..n {
        let (hi, lo) = if has_upper {
            (hex_val_upper(src[2 * i])?, hex_val_upper(src[2 * i + 1])?)
        } else {
            (hex_val_lower(src[2 * i])?, hex_val_lower(src[2 * i + 1])?)
        };
        out[i] = (hi << 4) | lo;
    }
    Some((has_upper, out, n as u8))
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const UNBASE64: [u8; 256] = {
    let mut t = [0xff; 256];
    let mut i = 0;
    while i < 64 {
        t[BASE64[i] as usize] = i as u8;
        i += 1;
    }
    t
};

/// Appends the bytes of padded standard Base64. `None`, with `out` in any state, unless
/// [`format_base64`] writes exactly `src` for them.
pub fn parse_base64(src: &[u8], out: &mut Vec<u8>) -> Option<()> {
    if src.is_empty() || !src.len().is_multiple_of(4) {
        return None;
    }
    let pad = src[src.len() - 2..].iter().filter(|&&b| b == b'=').count();
    let mut bits = 0u32;
    for (i, &c) in src[..src.len() - pad].iter().enumerate() {
        let d = UNBASE64[c as usize];
        if d == 0xff {
            return None;
        }
        bits = bits << 6 | u32::from(d);
        if i % 4 == 3 {
            out.extend_from_slice(&bits.to_be_bytes()[1..]);
            bits = 0;
        }
    }
    // The bits of the last digit that no byte takes are zero.
    match pad {
        1 if bits & 3 == 0 => out.extend_from_slice(&[(bits >> 10) as u8, (bits >> 2) as u8]),
        2 if bits & 15 == 0 => out.push((bits >> 4) as u8),
        0 => {}
        _ => return None,
    }
    Some(())
}

pub fn format_base64(bytes: &[u8], out: &mut Vec<u8>) {
    for chunk in bytes.chunks(3) {
        let mut b = [0u8; 4];
        b[1..1 + chunk.len()].copy_from_slice(chunk);
        let bits = u32::from_be_bytes(b);
        for i in 0..4 {
            let digit = BASE64[(bits >> (18 - 6 * i)) as usize & 63];
            out.push(if i <= chunk.len() { digit } else { b'=' });
        }
    }
}

#[inline]
pub fn format_hex_bytes(bytes: &[u8], upper: bool, out: &mut Vec<u8>) {
    const LO: &[u8; 16] = b"0123456789abcdef";
    const UP: &[u8; 16] = b"0123456789ABCDEF";
    let tab = if upper { UP } else { LO };
    for &b in bytes {
        out.push(tab[(b >> 4) as usize]);
        out.push(tab[(b & 0x0f) as usize]);
    }
}

/// Parses standard 36-byte hyphenated UUID (`8-4-4-4-12`, uniform lower or upper case).
pub fn parse_uuid(src: &[u8]) -> Option<(bool, [u8; 16])> {
    if src.len() != 36 || src[8] != b'-' || src[13] != b'-' || src[18] != b'-' || src[23] != b'-' {
        return None;
    }
    let mut compact = [0u8; 32];
    compact[0..8].copy_from_slice(&src[0..8]);
    compact[8..12].copy_from_slice(&src[9..13]);
    compact[12..16].copy_from_slice(&src[14..18]);
    compact[16..20].copy_from_slice(&src[19..23]);
    compact[20..32].copy_from_slice(&src[24..36]);
    let (upper, buf, _) = parse_hex_exact(&compact)?;
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&buf[..16]);
    Some((upper, uuid))
}

#[inline]
pub fn format_uuid(uuid: &[u8; 16], upper: bool, out: &mut Vec<u8>) {
    format_hex_bytes(&uuid[0..4], upper, out);
    out.push(b'-');
    format_hex_bytes(&uuid[4..6], upper, out);
    out.push(b'-');
    format_hex_bytes(&uuid[6..8], upper, out);
    out.push(b'-');
    format_hex_bytes(&uuid[8..10], upper, out);
    out.push(b'-');
    format_hex_bytes(&uuid[10..16], upper, out);
}

/// Parses W3C `traceparent`: `00-<32 lower hex>-<16 lower hex>-<2 lower hex>` (55 bytes).
pub fn parse_traceparent(src: &[u8]) -> Option<([u8; 16], [u8; 8], u8)> {
    if src.len() != 55 || &src[0..3] != b"00-" || src[35] != b'-' || src[52] != b'-' {
        return None;
    }
    let (u1, t_buf, _) = parse_hex_exact(&src[3..35])?;
    let (u2, s_buf, _) = parse_hex_exact(&src[36..52])?;
    if u1 || u2 {
        return None;
    }
    let f_hi = hex_val_lower(src[53])?;
    let f_lo = hex_val_lower(src[54])?;
    let mut tid = [0u8; 16];
    let mut sid = [0u8; 8];
    tid.copy_from_slice(&t_buf[..16]);
    sid.copy_from_slice(&s_buf[..8]);
    Some((tid, sid, (f_hi << 4) | f_lo))
}

#[inline]
pub fn format_traceparent(tid: &[u8; 16], sid: &[u8; 8], flags: u8, out: &mut Vec<u8>) {
    out.extend_from_slice(b"00-");
    format_hex_bytes(tid, false, out);
    out.push(b'-');
    format_hex_bytes(sid, false, out);
    out.push(b'-');
    format_hex_bytes(&[flags], false, out);
}

const CROCKFORD_ENC: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Parses canonical 26-char uppercase Crockford Base32 ULID into `(ts_ms_48b, rand_10B)`.
pub fn parse_ulid(src: &[u8]) -> Option<(u64, [u8; 10])> {
    if src.len() != 26 || src[0] > b'7' {
        return None;
    }
    let mut vals = [0u8; 26];
    for (i, &b) in src.iter().enumerate() {
        vals[i] = match b {
            b'0'..=b'9' => b - b'0',
            b'A'..=b'H' => b - b'A' + 10,
            b'J'..=b'K' => b - b'J' + 18,
            b'M'..=b'N' => b - b'M' + 20,
            b'P'..=b'T' => b - b'P' + 22,
            b'V'..=b'Z' => b - b'V' + 27,
            _ => return None,
        };
    }
    // First 10 chars = 50 bits (top 2 bits must be 0 since src[0] <= '7') -> 48-bit ms timestamp
    let mut ts = 0u64;
    for &v in &vals[0..10] {
        ts = (ts << 5) | (v as u64);
    }
    // Next 16 chars = 80 bits -> 10 bytes
    let mut acc = 0u128;
    for &v in &vals[10..26] {
        acc = (acc << 5) | (v as u128);
    }
    let rand_be = acc.to_be_bytes();
    let mut rand10 = [0u8; 10];
    rand10.copy_from_slice(&rand_be[6..16]);
    Some((ts, rand10))
}

#[inline]
pub fn format_ulid(ts_ms: u64, rand10: &[u8; 10], out: &mut Vec<u8>) {
    for i in (0..10).rev() {
        let v = ((ts_ms >> (i * 5)) & 0x1f) as usize;
        out.push(CROCKFORD_ENC[v]);
    }
    let mut be = [0u8; 16];
    be[6..16].copy_from_slice(rand10);
    let acc = u128::from_be_bytes(be);
    for i in (0..16).rev() {
        let v = ((acc >> (i * 5)) & 0x1f) as usize;
        out.push(CROCKFORD_ENC[v]);
    }
}

/// Parses canonical IPv4 (`"10.0.1.42"`) and optional `:port` (`"10.0.1.42:8080"`).
/// `port == 0` when no `:port` suffix is present (or rejects `:0` to keep `0` unambiguous).
pub fn parse_ipv4_or_port(src: &[u8]) -> Option<([u8; 4], u16)> {
    if src.len() < 7 || src.len() > 21 {
        return None;
    }
    let (ip_part, port) = if let Some(colon) = src.iter().position(|&b| b == b':') {
        let p_slice = &src[colon + 1..];
        if p_slice.is_empty() || p_slice.len() > 5 || p_slice[0] == b'0' {
            return None;
        }
        let mut p = 0u32;
        for &b in p_slice {
            let d = b.wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            p = p * 10 + d as u32;
        }
        if p == 0 || p > 65535 {
            return None;
        }
        (&src[..colon], p as u16)
    } else {
        (src, 0u16)
    };
    let mut octets = [0u8; 4];
    let mut idx = 0usize;
    let mut pos = 0usize;
    while idx < 4 {
        let start = pos;
        while pos < ip_part.len() && ip_part[pos] != b'.' {
            pos += 1;
        }
        let seg = &ip_part[start..pos];
        if seg.is_empty() || seg.len() > 3 || (seg.len() > 1 && seg[0] == b'0') {
            return None;
        }
        let mut v = 0u32;
        for &b in seg {
            let d = b.wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            v = v * 10 + d as u32;
        }
        if v > 255 {
            return None;
        }
        octets[idx] = v as u8;
        idx += 1;
        if idx < 4 {
            if pos >= ip_part.len() || ip_part[pos] != b'.' {
                return None;
            }
            pos += 1;
        } else if pos != ip_part.len() {
            return None;
        }
    }
    Some((octets, port))
}

#[inline]
pub fn format_ipv4_or_port(ip: [u8; 4], port: u16, out: &mut Vec<u8>) {
    let mut tmp = [0u8; 24];
    for i in 0..4 {
        if i > 0 {
            out.push(b'.');
        }
        let n = format_exact_int_buf(ip[i] as i64, 0, &mut tmp);
        out.extend_from_slice(&tmp[..n]);
    }
    if port != 0 {
        out.push(b':');
        let n = format_exact_int_buf(port as i64, 0, &mut tmp);
        out.extend_from_slice(&tmp[..n]);
    }
}

// ============================================================================
// 5. Structured Strings (`local@domain`, Hierarchical Paths/URLs/Pod-Names, JSON Escapes)
// ============================================================================

/// Splits `local@domain` email/identity strings when exactly one `@` is present.
#[inline]
pub fn split_email(src: &[u8]) -> Option<(&[u8], &[u8])> {
    if src.len() < 5 || src.len() > 256 || src.contains(&b' ') {
        return None;
    }
    let at = src.iter().position(|&b| b == b'@')?;
    if at == 0 || at + 2 >= src.len() {
        return None;
    }
    let domain = &src[at + 1..];
    if domain.contains(&b'@') || !domain.contains(&b'.') {
        return None;
    }
    Some((&src[..at], domain))
}

/// Splits hierarchical strings (URLs, file paths `/a/b/leaf`, or K8s `<deploy>-<rs>-<pod>`)
/// at the last `/` (preferred) or last `-` when the prefix is shared across rows.
#[inline]
pub fn split_hierarchical(src: &[u8]) -> Option<(&[u8], &[u8])> {
    if src.len() < 6 {
        return None;
    }
    if let Some(pos) = src.iter().rposition(|&b| b == b'/')
        && pos >= 2
        && pos + 1 < src.len()
    {
        // Include trailing `/` in prefix so reconstruction is plain concatenation!
        return Some((&src[..=pos], &src[pos + 1..]));
    }
    if let Some(pos) = src.iter().rposition(|&b| b == b'-')
        && pos >= 4
        && pos + 1 < src.len()
    {
        return Some((&src[..=pos], &src[pos + 1..]));
    }
    None
}

/// Losslessly unescapes the interior of a JSON string literal (`raw_inner` excludes outer `"`s)
/// if and only if [`escape_json_str_exact`] reproduces `raw_inner` byte-for-byte.
pub fn unescape_json_str_exact(raw_inner: &[u8]) -> Option<Vec<u8>> {
    if !raw_inner.contains(&b'\\') {
        // Ensure no chars that `escape_json_str_exact` would escape (`"`, `\`, control < 0x20)
        if raw_inner.iter().any(|&b| b == b'"' || b < 0x20) {
            return None;
        }
        return Some(raw_inner.to_vec());
    }
    let mut out = Vec::with_capacity(raw_inner.len());
    let mut i = 0usize;
    while i < raw_inner.len() {
        let b = raw_inner[i];
        if b == b'\\' {
            i += 1;
            let esc = *raw_inner.get(i)?;
            match esc {
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                _ => return None, // `\uXXXX`, `\/`, `\b`, `\f` fall back to exact raw string!
            }
            i += 1;
        } else {
            if b == b'"' || b < 0x20 {
                return None;
            }
            out.push(b);
            i += 1;
        }
    }
    let mut check = Vec::with_capacity(raw_inner.len());
    escape_json_str_exact(&out, &mut check);
    if check.as_slice() == raw_inner {
        Some(out)
    } else {
        None
    }
}

pub fn escape_json_str_exact(unescaped: &[u8], out: &mut Vec<u8>) {
    for &b in unescaped {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            _ => out.push(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_exact_ints_and_decimals() {
        for s in [
            "0",
            "-0",
            "+0",
            "007",
            "-0042",
            "+123",
            "9223372036854775807",
            "-9223372036854775808",
        ] {
            let (v, f) = parse_exact_int(s.as_bytes()).expect(s);
            let mut out = Vec::new();
            format_exact_int(v, f, &mut out);
            assert_eq!(out, s.as_bytes());
        }
        for s in ["0.0", "-0.00", "+12.340", "003.14159", "-99.999000"] {
            let (m, sp) = parse_exact_decimal(s.as_bytes()).expect(s);
            let mut out = Vec::new();
            format_exact_decimal(m, sp, &mut out);
            assert_eq!(out, s.as_bytes());
        }
    }

    #[test]
    fn base64_is_exact_or_refused() {
        for text in ["TWFu", "TWE=", "TQ==", "AAECAwQFBgcICQ==", "+/+/"] {
            let (mut bytes, mut back) = (Vec::new(), Vec::new());
            parse_base64(text.as_bytes(), &mut bytes).expect(text);
            format_base64(&bytes, &mut back);
            assert_eq!(back, text.as_bytes());
        }
        // Wrong length, alphabet or padding, and spare bits a writer would have left zero.
        for text in [
            "", "TWF", "TW-_", "TWE", "=WE=", "T===", "TW=u", "TWF=", "TR==",
        ] {
            assert!(
                parse_base64(text.as_bytes(), &mut Vec::new()).is_none(),
                "{text}"
            );
        }
    }

    #[test]
    fn roundtrip_timestamps_and_ids() {
        for ts in [
            "2026-11-10T11:24:02Z",
            "2026-11-10T11:24:02.123456789Z",
            "2026-11-10 11:24:02,450+02:00",
            "2026-02-28T23:59:59-0800",
            "10/Nov/2026:11:24:02 +0000",
            "Nov  5 08:12:04",
            "Nov 10 11:24:02",
        ] {
            let (ns, lay) = parse_timestamp(ts.as_bytes()).expect(ts);
            let mut out = Vec::new();
            format_timestamp(ns, lay, &mut out);
            assert_eq!(out, ts.as_bytes(), "failed on {ts}");
        }

        let u = b"018f5a3e-7c10-7b2a-9f01-020304050607";
        let (up, bytes) = parse_uuid(u).unwrap();
        let mut out = Vec::new();
        format_uuid(&bytes, up, &mut out);
        assert_eq!(out, u);

        let ulid = b"01ARZ3NDEKTSV4RRFFQ69G5FAV";
        let (tms, r10) = parse_ulid(ulid).unwrap();
        out.clear();
        format_ulid(tms, &r10, &mut out);
        assert_eq!(out, ulid);

        let tp = b"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let (tid, sid, fl) = parse_traceparent(tp).unwrap();
        out.clear();
        format_traceparent(&tid, &sid, fl, &mut out);
        assert_eq!(out, tp);
    }
}
