//! Pack bounded unsigned digits into little-endian words without rounding each digit to a bit width.
use crate::wire::{Error, Result};

pub(super) struct Layout {
    pub(super) radix: u32,
    digits: usize,
    width: usize,
}

impl Layout {
    pub fn new(radix: u32) -> Option<Self> {
        if !(3..=65535).contains(&radix) || radix.is_power_of_two() {
            return None;
        }
        let mut best = Self {
            radix,
            digits: 1,
            width: 4,
        };
        let mut power = 1u64;
        for digits in 1.. {
            let Some(next) = power.checked_mul(u64::from(radix)) else {
                break;
            };
            power = next;
            let width = ((64 - (power - 1).leading_zeros()) as usize).div_ceil(8);
            if width * best.digits < best.width * digits {
                best.digits = digits;
                best.width = width;
            }
        }
        Some(best)
    }

    fn tail_bytes(&self, digits: usize) -> usize {
        ((64 - (u64::from(self.radix).pow(digits as u32) - 1).leading_zeros()) as usize).div_ceil(8)
    }

    pub fn size(&self, n: usize) -> usize {
        n / self.digits * self.width + self.tail_bytes(n % self.digits)
    }

    pub fn pack(&self, values: &[i64], min: i64) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size(values.len()));
        for chunk in values.chunks(self.digits) {
            let mut word = 0u64;
            for &v in chunk.iter().rev() {
                word = word * u64::from(self.radix) + (v - min) as u64;
            }
            let width = if chunk.len() == self.digits {
                self.width
            } else {
                self.tail_bytes(chunk.len())
            };
            bytes.extend_from_slice(&word.to_le_bytes()[..width]);
        }
        bytes
    }

    pub fn unpack(&self, bytes: &[u8], n: usize, min: i64) -> Result<Vec<i64>> {
        if min.checked_add(i64::from(self.radix) - 1).is_none() || bytes.len() != self.size(n) {
            return Err(Error::Format("invalid radix bounds".into()));
        }
        let mut values = Vec::with_capacity(n);
        let radix = u64::from(self.radix);
        let reciprocal = u64::MAX / radix;
        for chunk in bytes.chunks(self.width) {
            let digits = self.digits.min(n - values.len());
            let mut le = [0u8; 8];
            le[..chunk.len()].copy_from_slice(chunk);
            let mut word = u64::from_le_bytes(le);
            for _ in 0..digits {
                let quotient = ((u128::from(word) * u128::from(reciprocal)) >> 64) as u64;
                let rem = word - quotient * radix;
                let carry = u64::from(rem >= radix);
                values.push(min + (rem - carry * radix) as i64);
                word = quotient + carry;
            }
            if word != 0 {
                return Err(Error::Format("radix word exceeds digit count".into()));
            }
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_words_round_trip_tails_and_extreme_bounds() {
        for radix in [3, 5, 9, 13, 17, 255, 1001, 65535] {
            let layout = Layout::new(radix).unwrap();
            for min in [i64::MIN, 0, i64::MAX - i64::from(radix - 1)] {
                for len in 0..=2 * layout.digits + 1 {
                    let values: Vec<_> = (0..len)
                        .map(|i| min + i64::from(if i % 2 == 0 { radix - 1 } else { 0 }))
                        .collect();
                    let bytes = layout.pack(&values, min);
                    assert_eq!(bytes.len(), layout.size(len));
                    assert_eq!(layout.unpack(&bytes, len, min).unwrap(), values);
                }
            }
        }
    }

    #[test]
    fn corrupt_tail_and_overflowing_bounds_fail() {
        let layout = Layout::new(3).unwrap();
        assert!(layout.unpack(&[3], 1, 0).is_err());
        assert!(layout.unpack(&[], 1, 0).is_err());
        assert!(layout.unpack(&[0], 1, i64::MAX).is_err());
    }
}
