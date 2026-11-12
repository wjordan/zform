//! Native numeric lanes carried by self-describing payload programs.
use crate::{CoreDecoder, DecodeBudget, Error, PayloadGraph, Result};
use zform_core::payload::{
    graph::{ExtInputs, GraphBuilder},
    ir::{Builder, Op},
};

pub fn encode_i64(values: &[i64]) -> Vec<u8> {
    zform_core::payload::graph::with_encode_opts(crate::EntropyBackend::Zstd, 1, || {
        let mut graph = GraphBuilder::new(values.len());
        let root = graph.add_ints_fast(values);
        graph.finish(root).fuse_numeric_programs().encode()
    })
}

pub fn decode_i64(bytes: &[u8], count: usize, budget: &DecodeBudget) -> Result<Vec<i64>> {
    let values = CoreDecoder::decode_ints_bytes(bytes, budget)?;
    if values.len() != count {
        return Err(Error::Format("numeric row count mismatch".into()));
    }
    Ok(values)
}

#[derive(Clone, Default)]
pub struct FloatHint {
    scale: Option<(u8, bool)>,
    failures: u8,
    skip: u8,
}
impl FloatHint {
    pub fn is_empty(&self) -> bool {
        self.scale.is_none() && self.failures == 0
    }
}

/// Keeps numeric scratch across calls. Hints avoid repeated decimal probes; every encoded
/// value is validated and residual bits preserve NaNs, signed zero, and precision changes.
#[derive(Default)]
pub struct FloatEncoder {
    bits: Vec<i64>,
    integers: Vec<i64>,
    residuals: Vec<i64>,
}
impl FloatEncoder {
    pub fn encode(&mut self, values: &[f64], hint: &mut FloatHint) -> Vec<u8> {
        self.bits.clear();
        self.bits.extend(values.iter().map(|v| v.to_bits() as i64));
        if values.is_empty() || self.bits.iter().all(|v| *v == self.bits[0]) {
            return encode_i64(&self.bits);
        }
        let runs = 1 + self.bits.windows(2).filter(|w| w[0] != w[1]).count();
        if runs * 16 < values.len() {
            return encode_i64(&self.bits);
        }
        let fits = |e: u8, multiply: bool| {
            let scale = 10f64.powi(e as i32);
            let probe = values.len().min(32);
            let exact = values
                .iter()
                .take(probe)
                .filter(|&&v| {
                    let i = (v * scale).round();
                    i.is_finite()
                        && i >= i64::MIN as f64
                        && i < -(i64::MIN as f64)
                        && reconstruct(i as i64, scale, multiply).to_bits() == v.to_bits()
                })
                .count();
            exact * 8 >= probe * 7
        };
        let scale = hint.scale.filter(|&(e, m)| fits(e, m)).or_else(|| {
            if hint.skip > 0 {
                hint.skip -= 1;
                return None;
            }
            (0..=9)
                .flat_map(|e| [(e, false), (e, true)])
                .find(|&(e, m)| fits(e, m))
        });
        let Some((exponent, multiply)) = scale else {
            hint.scale = None;
            hint.failures = hint.failures.saturating_add(1).min(6);
            hint.skip = (1 << hint.failures) - 1;
            return self.raw_bits();
        };
        hint.scale = scale;
        hint.failures = 0;
        hint.skip = 0;
        let factor = 10f64.powi(exponent as i32);
        self.integers.clear();
        self.residuals.clear();
        for &v in values {
            let i = (v * factor).round() as i64;
            self.integers.push(i);
            self.residuals
                .push((v.to_bits() ^ reconstruct(i, factor, multiply).to_bits()) as i64);
        }
        let mut graph = GraphBuilder::new(values.len());
        let ints = graph.add_ints_fast(&self.integers);
        let exceptions = self.residuals.iter().any(|v| *v != 0);
        let b = Builder::new(if exceptions { 2 } else { 1 });
        let mut output = b.r(Op::I64ToF64, b.input(0));
        if exponent != 0 {
            output = b.rf(
                if multiply { Op::F64MulI } else { Op::F64DivI },
                output,
                if multiply { 1.0 / factor } else { factor },
            );
        }
        let mut inputs = vec![ints];
        if exceptions {
            inputs.push(graph.add_ints_fast(&self.residuals));
            output = b.rr(Op::Xor, output, b.input(1));
        }
        let root = graph.add_prog_i64(b.finish(&[(output, 8)]), inputs, vec![]);
        let candidate = graph.finish(root).fuse_numeric_programs().encode();
        if candidate.len() < values.len() * 2 {
            return candidate;
        }
        let raw = self.raw_bits();
        if candidate.len() < raw.len() {
            candidate
        } else {
            raw
        }
    }
    fn raw_bits(&self) -> Vec<u8> {
        zform_core::payload::graph::with_encode_opts(crate::EntropyBackend::Zstd, 1, || {
            let n = self.bits.len();
            let mut bytes = vec![0; n * 8];
            for (i, value) in self.bits.iter().enumerate() {
                for (k, byte) in value.to_le_bytes().iter().enumerate() {
                    bytes[k * n + i] = *byte;
                }
            }
            let mut graph = GraphBuilder::new(n);
            let source = graph.add_bytes(&bytes);
            let source = graph.add_byte_unshuffle(source, 8);
            let root = graph.add_byte_chunks(source, 8);
            graph.finish(root).fuse_numeric_programs().encode()
        })
    }
}

fn reconstruct(i: i64, scale: f64, multiply: bool) -> f64 {
    if multiply {
        i as f64 * (1.0 / scale)
    } else {
        i as f64 / scale
    }
}

pub fn decode_f64(
    bytes: &[u8],
    count: usize,
    range: std::ops::Range<usize>,
    budget: &DecodeBudget,
) -> Result<Vec<f64>> {
    if range.start > range.end || range.end > count {
        return Err(Error::InvalidArgument("numeric selection range".into()));
    }
    let mut values = decode_i64(bytes, count, budget)?;
    values.truncate(range.end);
    if range.start > 0 {
        values.drain(..range.start);
    }
    Ok(values
        .into_iter()
        .map(|v| f64::from_bits(v as u64))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_bits_survive_decimal_hints_exceptions_and_precision_drift() {
        let mut encoder = FloatEncoder::default();
        let mut hint = FloatHint::default();
        let specials = [
            0,
            1,
            u64::MAX,
            (-0.0f64).to_bits(),
            f64::INFINITY.to_bits(),
            0x7ff0000000000002,
        ];
        for round in 0..4 {
            let values: Vec<_> = (0..256)
                .map(|i| {
                    if i % 37 == 0 {
                        f64::from_bits(specials[i / 37 % specials.len()])
                    } else {
                        (i as f64 - 150.)
                            / if round % 2 == 0 {
                                1000.
                            } else {
                                1_000_000_000.
                            }
                    }
                })
                .collect();
            let bytes = encoder.encode(&values, &mut hint);
            for range in [0..values.len(), 33..70, 0..0] {
                let got = decode_f64(
                    &bytes,
                    values.len(),
                    range.clone(),
                    &DecodeBudget::default(),
                )
                .unwrap();
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    values[range]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            for n in 0..bytes.len() {
                assert!(
                    decode_f64(
                        &bytes[..n],
                        values.len(),
                        0..values.len(),
                        &DecodeBudget::default()
                    )
                    .is_err()
                );
            }
        }
        let times = [i64::MIN, i64::MAX, -1, 0, 9_007_199_254_740_993];
        assert_eq!(
            decode_i64(&encode_i64(&times), times.len(), &DecodeBudget::default()).unwrap(),
            times
        );
    }
}

/// Encodes exact differences from a caller-supplied base column. The payload carries the
/// addition program; the decoder only supplies the base as an external byte pool.
pub fn encode_f64_with_base(values: &[f64], base: &[f64]) -> Result<Vec<u8>> {
    if values.len() != base.len() {
        return Err(Error::InvalidArgument("numeric base length".into()));
    }
    let mut differences = Vec::with_capacity(values.len());
    for (&value, &prior) in values.iter().zip(base) {
        let delta = value - prior;
        let integer = delta as i64;
        if !delta.is_finite()
            || integer as f64 != delta
            || (prior + integer as f64).to_bits() != value.to_bits()
        {
            return Err(Error::InvalidArgument(
                "inexact numeric base difference".into(),
            ));
        }
        differences.push(integer);
    }
    let mut graph = GraphBuilder::new(values.len());
    let input = graph.add_ints_fast(&differences);
    let pool = graph.add_ext_pool();
    let prior = graph.add_byte_chunks(pool, 8);
    let b = Builder::new(2);
    let delta = b.r(Op::I64ToF64, b.input(0));
    let value = b.rr(Op::F64Add, b.input(1), delta);
    let root = graph.add_prog_i64(b.finish(&[(value, 8)]), vec![input, prior], vec![]);
    Ok(graph.finish(root).fuse_numeric_programs().encode())
}

pub fn decode_f64_with_base(
    bytes: &[u8],
    count: usize,
    base: Option<&[f64]>,
    budget: &DecodeBudget,
) -> Result<Vec<f64>> {
    let graph = PayloadGraph::decode(bytes, budget)?;
    if graph.rows as usize != count {
        return Err(Error::Format("numeric row count mismatch".into()));
    }
    let pool = if let Some(base) = base {
        if base.len() != count {
            return Err(Error::InvalidArgument("numeric base length".into()));
        }
        budget.check_alloc(count.saturating_mul(8), "numeric base")?;
        Some(std::sync::Arc::new(
            base.iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        ))
    } else {
        None
    };
    let bits = CoreDecoder::decode_ints(
        &graph,
        &ExtInputs {
            pool: pool.as_ref(),
            ..ExtInputs::empty()
        },
        budget,
    )?;
    Ok(bits.into_iter().map(|b| f64::from_bits(b as u64)).collect())
}

#[cfg(test)]
mod base_tests {
    use super::*;
    #[test]
    fn dependent_numeric_programs_reconstruct_from_the_supplied_base() {
        let base: Vec<_> = (0..65).map(|i| (i * 13 % 47) as f64).collect();
        let values: Vec<_> = base
            .iter()
            .enumerate()
            .map(|(i, &v)| v + (i * 7 % 11) as f64)
            .collect();
        let bytes = encode_f64_with_base(&values, &base).unwrap();
        let budget = DecodeBudget::default();
        assert_eq!(
            decode_f64_with_base(&bytes, values.len(), Some(&base), &budget).unwrap(),
            values
        );
        assert!(decode_f64_with_base(&bytes, values.len(), None, &budget).is_err());
        assert!(decode_f64_with_base(&bytes, values.len(), Some(&base[..5]), &budget).is_err());
        assert!(encode_f64_with_base(&[0.25], &[0.]).is_err());
        assert!(encode_f64_with_base(&[-0.], &[0.]).is_err());
    }
}
