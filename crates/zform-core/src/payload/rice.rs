//! Centered Rice transport for signed integer lanes.
use crate::wire::{Cursor, DecodeBudget, Error, Result, put_ivarint};

pub(super) fn encode(values: &[i64], budget: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    encode_rice(values, budget, &mut Vec::new(), &mut Vec::new(), &mut out).then_some(out)
}

pub(super) fn decode(bytes: &[u8], count: usize, budget: &DecodeBudget) -> Result<Vec<i64>> {
    budget.check_alloc(count.saturating_mul(8), "Rice lanes")?;
    let mut out = Vec::with_capacity(count);
    decode_rice(bytes, count, |v| {
        out.push(v);
        Ok(())
    })
    .map_err(Error::Format)?;
    Ok(out)
}

fn encode_rice(
    values: &[i64],
    budget: usize,
    median: &mut Vec<i64>,
    centered: &mut Vec<u64>,
    out: &mut Vec<u8>,
) -> bool {
    if values.is_empty() || budget < 3 {
        return false;
    }
    median.clear();
    median.extend_from_slice(values);
    let center = *median.select_nth_unstable(values.len() / 2).1;
    centered.clear();
    let mut sum = 0u128;
    for &value in values {
        let Some(delta) = value.checked_sub(center) else {
            return false;
        };
        let code = ((delta as u64) << 1) ^ ((delta >> 63) as u64);
        centered.push(code);
        sum += code as u128;
    }
    out.clear();
    put_ivarint(out, center);
    out.push(0);
    let header = out.len();
    let budget = budget.min(values.len() * 10 + 11);
    if header >= budget {
        return false;
    }
    let mean = (sum / values.len() as u128) as u64;
    let near = 63 - mean.max(1).leading_zeros();
    let mut best_bits = ((budget - header - 1) as u128) * 8 + 1;
    let mut best_k = 0;
    // Probe near log2(mean), with an early exit when a width cannot beat the budget.
    for k in near.saturating_sub(1)..=(near + 1).min(63) {
        let mut bits = values.len() as u128 * (k as u128 + 1);
        if bits >= best_bits {
            continue;
        }
        for &code in centered.iter() {
            bits += (code >> k) as u128;
            if bits >= best_bits {
                break;
            }
        }
        if bits < best_bits {
            best_bits = bits;
            best_k = k;
        }
    }
    let bytes = best_bits.div_ceil(8) as usize;
    if header + bytes >= budget {
        return false;
    }
    out[header - 1] = best_k as u8;
    out.resize(header + bytes, 0);
    let mut writer = RiceWriter {
        bytes: &mut out[header..],
        at: 0,
        word: 0,
        used: 0,
    };
    let mask = (1u64 << best_k) - 1;
    for &code in centered.iter() {
        writer.unary(code >> best_k);
        writer.bits(code & mask, best_k);
    }
    writer.finish();
    true
}

fn decode_rice(
    src: &[u8],
    count: usize,
    mut emit: impl FnMut(i64) -> std::result::Result<(), String>,
) -> std::result::Result<(), String> {
    if count == 0 || src.len() > count.saturating_mul(10).saturating_add(11) {
        return Err("Rice residual count or bytes exceed bound".into());
    }
    let mut reader = Cursor::new(src);
    let center = reader.ivar().map_err(|e| e.to_string())?;
    let k = reader.u8().map_err(|e| e.to_string())? as u32;
    if k > 63 {
        return Err("invalid Rice parameter".into());
    }
    let bytes = reader
        .slice(reader.remaining())
        .map_err(|e| e.to_string())?;
    if count.saturating_mul(k as usize + 1) > bytes.len().saturating_mul(8) {
        return Err("truncated Rice residuals".into());
    }
    let mut reader = RiceReader {
        bytes,
        at: 0,
        word: 0,
        available: 0,
    };
    for _ in 0..count {
        let quotient = reader.unary(u64::MAX >> k)?;
        let code = (quotient << k) | reader.bits(k)?;
        let delta = ((code >> 1) as i64) ^ -((code & 1) as i64);
        emit(center.checked_add(delta).ok_or("Rice residual overflow")?)?;
    }
    reader.finish()?;
    Ok(())
}

struct RiceWriter<'a> {
    bytes: &'a mut [u8],
    at: usize,
    word: u64,
    used: u32,
}
impl RiceWriter<'_> {
    #[inline]
    fn flush(&mut self) {
        self.bytes[self.at..self.at + 8].copy_from_slice(&self.word.to_le_bytes());
        self.at += 8;
        self.word = 0;
        self.used = 0;
    }

    #[inline]
    fn unary(&mut self, mut zeroes: u64) {
        let free = 64 - self.used;
        if zeroes >= free as u64 {
            zeroes -= free as u64;
            self.flush();
            // The output was zero-filled after its exact byte budget was checked.
            self.at += (zeroes / 64) as usize * 8;
            zeroes %= 64;
        }
        self.used += zeroes as u32;
        self.word |= 1u64 << self.used;
        self.used += 1;
        if self.used == 64 {
            self.flush();
        }
    }

    #[inline]
    fn bits(&mut self, value: u64, width: u32) {
        self.word |= value << self.used;
        let total = self.used + width;
        if total >= 64 {
            let remaining = value >> (64 - self.used);
            self.flush();
            self.word = remaining;
            self.used = total - 64;
        } else {
            self.used = total;
        }
    }

    fn finish(self) {
        let tail = self.used.div_ceil(8) as usize;
        self.bytes[self.at..self.at + tail].copy_from_slice(&self.word.to_le_bytes()[..tail]);
        debug_assert_eq!(self.at + tail, self.bytes.len());
    }
}

struct RiceReader<'a> {
    bytes: &'a [u8],
    at: usize,
    word: u64,
    available: u32,
}
impl RiceReader<'_> {
    #[inline]
    fn refill(&mut self) -> std::result::Result<(), String> {
        let remaining = &self.bytes[self.at..];
        if remaining.len() >= 8 {
            self.word = u64::from_le_bytes(remaining[..8].try_into().unwrap());
            self.available = 64;
            self.at += 8;
        } else {
            if remaining.is_empty() {
                return Err("truncated Rice residuals".into());
            }
            let mut tail = [0u8; 8];
            tail[..remaining.len()].copy_from_slice(remaining);
            self.word = u64::from_le_bytes(tail);
            self.available = remaining.len() as u32 * 8;
            self.at = self.bytes.len();
        }
        Ok(())
    }

    #[inline]
    fn unary(&mut self, limit: u64) -> std::result::Result<u64, String> {
        let mut quotient = 0u64;
        loop {
            if self.available == 0 {
                self.refill()?;
            }
            let zeroes = self.word.trailing_zeros().min(self.available);
            quotient = quotient
                .checked_add(zeroes as u64)
                .ok_or("Rice quotient overflow")?;
            if quotient > limit {
                return Err("Rice quotient overflow".into());
            }
            if zeroes < self.available {
                let consumed = zeroes + 1;
                self.word = if consumed == 64 {
                    0
                } else {
                    self.word >> consumed
                };
                self.available -= consumed;
                return Ok(quotient);
            }
            self.available = 0;
        }
    }

    #[inline]
    fn bits(&mut self, width: u32) -> std::result::Result<u64, String> {
        let mask = (1u64 << width) - 1;
        if width <= self.available {
            let value = self.word & mask;
            self.word >>= width;
            self.available -= width;
            return Ok(value);
        }
        let low = self.word;
        let have = self.available;
        self.refill()?;
        let need = width - have;
        if need > self.available {
            return Err("truncated Rice remainder".into());
        }
        let value = low | ((self.word & ((1u64 << need) - 1)) << have);
        self.word >>= need;
        self.available -= need;
        Ok(value)
    }

    fn finish(self) -> std::result::Result<(), String> {
        let bits = self.at * 8 - self.available as usize;
        if bits.div_ceil(8) != self.bytes.len() {
            return Err("trailing Rice residual bytes".into());
        }
        if self.word != 0 {
            return Err("nonzero Rice residual padding".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rice_word_stream_matches_bitwise_layout_at_all_widths() {
        for k in 0..64u32 {
            let mask = (1u64 << k) - 1;
            let codes: Vec<_> = [0u64, 1, 63, 64, 65, 127, 128]
                .into_iter()
                .enumerate()
                .map(|(i, q)| (q.min(u64::MAX >> k) << k) | (mask >> (i % 3)))
                .collect();
            let bit_count: usize = codes
                .iter()
                .map(|&v| (v >> k) as usize + 1 + k as usize)
                .sum();
            let mut expected = vec![0; bit_count.div_ceil(8)];
            let mut bit = 0;
            for &code in &codes {
                bit += (code >> k) as usize;
                expected[bit / 8] |= 1 << (bit % 8);
                bit += 1;
                for j in 0..k {
                    expected[bit / 8] |= (((code >> j) & 1) as u8) << (bit % 8);
                    bit += 1;
                }
            }
            let mut actual = vec![0; expected.len()];
            let mut writer = RiceWriter {
                bytes: &mut actual,
                at: 0,
                word: 0,
                used: 0,
            };
            for &code in &codes {
                writer.unary(code >> k);
                writer.bits(code & mask, k);
            }
            writer.finish();
            assert_eq!(actual, expected);
            let mut reader = RiceReader {
                bytes: &actual,
                at: 0,
                word: 0,
                available: 0,
            };
            for &code in &codes {
                let quotient = reader.unary(u64::MAX >> k).unwrap();
                assert_eq!((quotient << k) | reader.bits(k).unwrap(), code);
            }
            reader.finish().unwrap();
        }
    }

    #[test]
    fn rice_round_trips_centered_residuals_and_strict_budgets() {
        for count in [1, 2, 31, 32, 33, 65, 128] {
            for values in [
                vec![i64::MIN; count],
                vec![i64::MAX; count],
                (0..count)
                    .map(|i| i64::MAX - (i * 17 % 19) as i64)
                    .collect(),
                (0..count)
                    .map(|i| ((i * 17 + i * i) % 31) as i64 - 15)
                    .collect(),
            ] {
                let encoded = encode(&values, usize::MAX).unwrap();
                let mut decoded = Vec::new();
                decode_rice(&encoded, count, |v| {
                    decoded.push(v);
                    Ok(())
                })
                .unwrap();
                assert_eq!(decoded, values);
                assert!(encode(&values, encoded.len()).is_none());
                assert_eq!(encode(&values, encoded.len() + 1).unwrap(), encoded);
            }
        }
        assert!(encode(&[i64::MIN, i64::MAX], usize::MAX).is_none());
        let values: Vec<_> = (0..33).map(|i| [i64::MIN, i64::MAX, 0][i % 3]).collect();
        let encoded = encode(&values, usize::MAX).unwrap();
        let mut decoded = Vec::new();
        decode_rice(&encoded, values.len(), |v| {
            decoded.push(v);
            Ok(())
        })
        .unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn rice_parameter_search_matches_exhaustive_bit_cost() {
        for seed in 0..64u64 {
            let values: Vec<_> = (0..63)
                .map(|i| {
                    let magnitude = ((seed.wrapping_mul(131) + i * 79) % 251) << (seed % 48);
                    if i % 3 == 0 {
                        -(magnitude as i64)
                    } else {
                        magnitude as i64
                    }
                })
                .collect();
            let mut sorted = values.clone();
            sorted.sort_unstable();
            let center = sorted[values.len() / 2];
            let codes: Vec<_> = values
                .iter()
                .map(|v| {
                    let d = v - center;
                    ((d as u64) << 1) ^ ((d >> 63) as u64)
                })
                .collect();
            let best = (0..64)
                .map(|k| {
                    codes
                        .iter()
                        .map(|v| (*v >> k) as u128 + 1 + k as u128)
                        .sum::<u128>()
                })
                .min()
                .unwrap();
            let encoded = encode(&values, usize::MAX).unwrap();
            let mut reader = Cursor::new(&encoded);
            assert_eq!(reader.ivar().unwrap(), center);
            let k = reader.u8().unwrap();
            let actual: u128 = codes
                .iter()
                .map(|v| (*v >> k) as u128 + 1 + k as u128)
                .sum();
            assert_eq!(actual, best);
        }
    }

    #[test]
    fn rice_decoder_rejects_invalid_bounds_frames_padding_and_overflow() {
        assert!(decode_rice(&[0, 0, 1], 0, |_| Ok(())).is_err());
        assert!(decode_rice(&[0, 0, 1], usize::MAX, |_| Ok(())).is_err());
        assert!(decode_rice(&[0, 64, 1], 1, |_| Ok(())).is_err());
        assert!(decode_rice(&[0, 0, 0], 1, |_| Ok(())).is_err());
        assert!(decode_rice(&[0, 0, 0x87], 3, |_| Ok(())).is_err());
        let mut overflow = Vec::new();
        put_ivarint(&mut overflow, i64::MAX);
        overflow.extend_from_slice(&[0, 4]);
        assert!(
            decode_rice(&overflow, 1, |_| Ok(()))
                .unwrap_err()
                .contains("overflow")
        );
        let mut quotient = vec![0, 63, 4];
        quotient.resize(11, 0);
        assert!(
            decode_rice(&quotient, 1, |_| Ok(()))
                .unwrap_err()
                .contains("overflow")
        );
        let encoded = encode(&[-100, 4, 8, 9, 9, 200, 5], usize::MAX).unwrap();
        for end in 0..encoded.len() {
            assert!(decode_rice(&encoded[..end], 7, |_| Ok(())).is_err());
        }
        let mut extra = encoded.clone();
        extra.push(0);
        assert!(decode_rice(&extra, 7, |_| Ok(())).is_err());
        assert_eq!(
            decode_rice(&encoded, 7, |_| Err("callback".into())).unwrap_err(),
            "callback"
        );
    }
}
