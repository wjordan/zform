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
        graph.finish(root).fuse_numeric_programs().encode_numeric()
    })
}

pub fn decode_i64(bytes: &[u8], count: usize, budget: &DecodeBudget) -> Result<Vec<i64>> {
    let values = CoreDecoder::decode_ints_bytes(bytes, budget)?;
    if values.len() != count {
        return Err(Error::Format("numeric row count mismatch".into()));
    }
    Ok(values)
}

#[derive(Clone, Copy)]
enum Scale {
    Divide,
    Multiply,
    MilliTwice,
}

#[derive(Clone, Default)]
pub struct FloatHint {
    scale: Option<(u8, Scale)>,
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
        self.integers.clear();
        if values.iter().all(|&v| {
            let i = v as i64;
            if (i as f64).to_bits() != v.to_bits() {
                return false;
            }
            self.integers.push(i);
            true
        }) {
            return zform_core::payload::graph::with_encode_opts(
                crate::EntropyBackend::Zstd,
                1,
                || {
                    let mut graph = GraphBuilder::new(values.len());
                    let ints = graph.add_ints_fast(&self.integers);
                    let b = Builder::new(1);
                    let value = b.r(Op::I64ToF64, b.input(0));
                    let root = graph.add_prog_i64(b.finish(&[(value, 8)]), vec![ints], vec![]);
                    graph.finish(root).fuse_numeric_programs().encode_numeric()
                },
            );
        }
        const POW10: [f64; 10] = [1., 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9];
        let probe = values.len().min(32);
        let max_fb_misses = probe - (probe * 7).div_ceil(8);
        let fits = |e: u8, mode: Scale, max_misses: usize| -> Option<usize> {
            let scale = POW10[e as usize];
            let inv = 1.0 / scale;
            let mut misses = 0usize;
            for &v in &values[..probe] {
                let i = (v * scale).round();
                let ok = i.is_finite()
                    && i >= i64::MIN as f64
                    && i < -(i64::MIN as f64)
                    && match mode {
                        Scale::Divide => (i as i64 as f64 / scale).to_bits() == v.to_bits(),
                        Scale::Multiply => (i as i64 as f64 * inv).to_bits() == v.to_bits(),
                        Scale::MilliTwice => {
                            (i as i64 as f64 / 1000. / 1000.).to_bits() == v.to_bits()
                        }
                    };
                if !ok {
                    misses += 1;
                    if misses > max_misses {
                        return None;
                    }
                }
            }
            Some(probe - misses)
        };
        let scale = hint
            .scale
            .filter(|&(e, m)| fits(e, m, 0).is_some())
            .or_else(|| {
                if hint.skip > 0 {
                    hint.skip -= 1;
                    return None;
                }
                let mut fallback = None;
                for e in 0..=9 {
                    for mode in [Scale::Divide, Scale::Multiply, Scale::MilliTwice] {
                        if matches!(mode, Scale::MilliTwice) && e != 6 {
                            continue;
                        }
                        let cap = if fallback.is_none() { max_fb_misses } else { 0 };
                        let Some(exact) = fits(e, mode, cap) else {
                            continue;
                        };
                        if exact == probe {
                            return Some((e, mode));
                        }
                        if fallback.is_none() && exact * 8 >= probe * 7 {
                            fallback = Some((e, mode));
                        }
                    }
                    if fallback.is_some() {
                        return fallback;
                    }
                }
                None
            });
        let Some((exponent, mode)) = scale else {
            hint.scale = None;
            hint.failures = hint.failures.saturating_add(1).min(6);
            hint.skip = (1 << hint.failures) - 1;
            return self.raw_bits();
        };
        hint.scale = scale;
        hint.failures = 0;
        hint.skip = 0;
        let factor = POW10[exponent as usize];
        self.integers.clear();
        self.residuals.clear();
        for &v in values {
            let i = (v * factor).round() as i64;
            self.integers.push(i);
            self.residuals
                .push((v.to_bits() ^ reconstruct(i, factor, mode).to_bits()) as i64);
        }
        let candidate =
            zform_core::payload::graph::with_encode_opts(crate::EntropyBackend::Zstd, 1, || {
                let mut graph = GraphBuilder::new(values.len());
                let ints = graph.add_ints_fast(&self.integers);
                let exceptions = self.residuals.iter().any(|v| *v != 0);
                let b = Builder::new(if exceptions { 2 } else { 1 });
                let mut output = b.r(Op::I64ToF64, b.input(0));
                if exponent != 0 {
                    output = match mode {
                        Scale::Divide => b.rf(Op::F64DivI, output, factor),
                        Scale::Multiply => b.rf(Op::F64MulI, output, 1.0 / factor),
                        Scale::MilliTwice => {
                            b.rf(Op::F64DivI, b.rf(Op::F64DivI, output, 1000.), 1000.)
                        }
                    };
                }
                let mut inputs = vec![ints];
                if exceptions {
                    inputs.push(graph.add_ints_fast(&self.residuals));
                    output = b.rr(Op::Xor, output, b.input(1));
                }
                let root = graph.add_prog_i64(b.finish(&[(output, 8)]), inputs, vec![]);
                graph.finish(root).fuse_numeric_programs().encode_numeric()
            });
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
            let total = n * 8;
            let mut bytes = Vec::with_capacity(total);
            let ptr: *mut u8 = bytes.as_mut_ptr();
            // SAFETY: `bytes` has capacity `n * 8`; every write targets `k * n + i < n * 8`.
            unsafe {
                let p0 = ptr;
                let p1 = ptr.add(n);
                let p2 = ptr.add(n * 2);
                let p3 = ptr.add(n * 3);
                let p4 = ptr.add(n * 4);
                let p5 = ptr.add(n * 5);
                let p6 = ptr.add(n * 6);
                let p7 = ptr.add(n * 7);
                for (i, &value) in self.bits.iter().enumerate() {
                    let b = value.to_le_bytes();
                    *p0.add(i) = b[0];
                    *p1.add(i) = b[1];
                    *p2.add(i) = b[2];
                    *p3.add(i) = b[3];
                    *p4.add(i) = b[4];
                    *p5.add(i) = b[5];
                    *p6.add(i) = b[6];
                    *p7.add(i) = b[7];
                }
                bytes.set_len(total);
            }
            let mut graph = GraphBuilder::new(n);
            let root = graph.add_planes(&bytes, 8);
            graph.finish(root).fuse_numeric_programs().encode_numeric()
        })
    }
}

fn reconstruct(i: i64, scale: f64, mode: Scale) -> f64 {
    match mode {
        Scale::Divide => i as f64 / scale,
        Scale::Multiply => i as f64 * (1.0 / scale),
        Scale::MilliTwice => i as f64 / 1000. / 1000.,
    }
}

pub fn decode_f64(
    bytes: &[u8],
    count: usize,
    range: std::ops::Range<usize>,
    budget: &DecodeBudget,
) -> Result<Vec<f64>> {
    let mut wire = zform_core::wire::Cursor::new(bytes);
    if wire.uvar()? != count as u64 {
        return Err(Error::Format("numeric row count mismatch".into()));
    }
    decode_f64_body(wire.slice(wire.remaining())?, count, range, budget)
}

/// Decodes the bytes following the row count when an enclosing frame supplies that count.
pub fn decode_f64_body(
    body: &[u8],
    count: usize,
    range: std::ops::Range<usize>,
    budget: &DecodeBudget,
) -> Result<Vec<f64>> {
    if range.start > range.end || range.end > count {
        return Err(Error::InvalidArgument("numeric selection range".into()));
    }
    let values = CoreDecoder::decode_ints_body(body, count as u64, budget)?;
    if values.len() != count {
        return Err(Error::Format("numeric row count mismatch".into()));
    }
    Ok(values[range.start..range.end]
        .iter()
        .map(|&v| f64::from_bits(v as u64))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decimal_recipes_avoid_residuals_and_revalidate_hints() {
        let mut encoder = FloatEncoder::default();
        let mut hint = FloatHint::default();
        for mode in [
            Scale::Divide,
            Scale::Multiply,
            Scale::MilliTwice,
            Scale::Divide,
        ] {
            let values: Vec<_> = (0..128)
                .map(|i| reconstruct(5_620_066_438 + (i * 7919 % 12347), 1e6, mode))
                .collect();
            let bytes = encoder.encode(&values, &mut hint);
            assert!(
                matches!(hint.scale, Some((6, actual)) if std::mem::discriminant(&actual) == std::mem::discriminant(&mode))
            );
            let decoded = decode_f64(
                &bytes,
                values.len(),
                0..values.len(),
                &DecodeBudget::default(),
            )
            .unwrap();
            assert_eq!(
                decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let graph =
                zform_core::payload::graph::PayloadGraph::decode(&bytes, &DecodeBudget::default())
                    .unwrap();
            assert_eq!(
                graph.nodes.len(),
                2,
                "exact scaling should need one integer stream and one program"
            );
        }
    }

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
    if differences.iter().all(|&d| d == 0) {
        let mut graph = GraphBuilder::new(values.len());
        let pool = graph.add_ext_pool();
        let root = graph.add_byte_chunks(pool, 8);
        return Ok(graph.finish(root).encode_numeric());
    }
    Ok(zform_core::payload::graph::with_encode_opts(
        crate::EntropyBackend::Zstd,
        1,
        || {
            let mut graph = GraphBuilder::new(values.len());
            let input = graph.add_ints_fast(&differences);
            let pool = graph.add_ext_pool();
            let prior = graph.add_byte_chunks(pool, 8);
            let b = Builder::new(2);
            let delta = b.r(Op::I64ToF64, b.input(0));
            let value = b.rr(Op::F64Add, b.input(1), delta);
            let root = graph.add_prog_i64(b.finish(&[(value, 8)]), vec![input, prior], vec![]);
            graph.finish(root).fuse_numeric_programs().encode_numeric()
        },
    ))
}

pub fn decode_f64_with_base(
    bytes: &[u8],
    count: usize,
    base: Option<&[f64]>,
    budget: &DecodeBudget,
) -> Result<Vec<f64>> {
    use zform_core::payload::graph::GraphNode;
    use zform_core::payload::ir::{Len, Width};
    let graph = PayloadGraph::decode(bytes, budget)?;
    if graph.rows as usize != count {
        return Err(Error::Format("numeric row count mismatch".into()));
    }
    // Fast path for the two canonical `encode_f64_with_base` shapes without copying `base` into a byte Pool.
    if let Some(base) = base {
        if base.len() != count {
            return Err(Error::InvalidArgument("numeric base length".into()));
        }
        match &graph.nodes[graph.root as usize] {
            GraphNode::ByteChunks {
                buf,
                chunk_bytes: 8,
            } if matches!(graph.nodes[*buf as usize], GraphNode::ExtPool) => {
                return Ok(base.to_vec());
            }
            GraphNode::ProgramI64 { prog, inputs, bufs }
                if bufs.is_empty()
                    && inputs.len() == 2
                    && prog.width == Width::W64
                    && prog.n_in == 2
                    && prog.ops.len() == 2
                    && prog.ops[0].op == Op::I64ToF64
                    && prog.ops[0].a == 0
                    && prog.ops[1].op == Op::F64Add
                    && prog.ops[1].a == 1
                    && prog.ops[1].b == 2
                    && prog.sinks == [(3, Len::Fixed(8))] =>
            {
                if let (
                    GraphNode::IntStream(delta_stream),
                    GraphNode::ByteChunks {
                        buf,
                        chunk_bytes: 8,
                    },
                ) = (
                    &graph.nodes[inputs[0] as usize],
                    &graph.nodes[inputs[1] as usize],
                ) && matches!(graph.nodes[*buf as usize], GraphNode::ExtPool)
                    && delta_stream.len() == count
                {
                    let deltas = delta_stream.unpack(budget)?;
                    let mut out = Vec::with_capacity(count);
                    for (&b, &d) in base.iter().zip(&deltas) {
                        out.push(b + d as f64);
                    }
                    return Ok(out);
                }
            }
            _ => {}
        }
    }
    let pool = if let Some(base) = base {
        budget.check_alloc(count.saturating_mul(8), "numeric base")?;
        Some(std::sync::Arc::new(zform_core::cells::Pool::whole(
            base.iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        )))
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
