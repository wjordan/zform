//! Standalone columns using the same payload programs as structured record files.
use crate::{CoreDecoder, DecodeBudget, Error, PackRegistry, PayloadGraph, Result};
use std::{collections::HashMap, sync::Arc};
use zform_core::{
    cells::{Cells, Col},
    payload::{
        graph::{ExtInputs, GraphBuilder},
        ir::Builder,
    },
};

#[derive(Default)]
pub struct Encoder {
    registry: PackRegistry,
}

impl Encoder {
    pub fn encode(&self, values: &[&[u8]]) -> Vec<u8> {
        self.registry.encode_best(values)
    }

    /// Selects a standalone payload or a payload referencing peer slot zero. The caller
    /// names that dependency in its own column directory when `uses_peer` is true.
    pub fn encode_with_peer(&self, values: &[&[u8]], peer: &[&[u8]]) -> (Vec<u8>, bool) {
        let plain = self.encode(values);
        if peer.is_empty() || values.is_empty() {
            return (plain, false);
        }
        let mut positions = HashMap::with_capacity(peer.len());
        for (i, value) in peer.iter().enumerate() {
            positions.entry(*value).or_insert(i);
        }
        let mut indices = Vec::with_capacity(values.len());
        let mut patches = Vec::new();
        let mut literals = Vec::new();
        for (i, value) in values.iter().enumerate() {
            match positions.get(value) {
                Some(&row) => indices.push(row as i64),
                None => {
                    indices.push(0);
                    patches.push(i);
                    literals.push(*value);
                }
            }
        }
        if patches.len() == values.len() {
            return (plain, false);
        }
        let mut graph = GraphBuilder::new(values.len());
        let source = graph.add_ext_peer(0);
        let rows = graph.add_ints(&indices);
        let root = graph.add_index_cells(source, rows);
        let root = graph.add_patch(root, &patches, &literals);
        let candidate = graph.finish(root).encode();
        if candidate.len() < plain.len() {
            (candidate, true)
        } else {
            (plain, false)
        }
    }

    /// Native signed lanes with an eight-byte little-endian output program.
    pub fn encode_i64(&self, values: &[i64]) -> Vec<u8> {
        let mut graph = GraphBuilder::new(values.len());
        let ints = graph.add_ints(values);
        let program = Builder::new(1);
        let value = program.input(0);
        let root = graph.add_prog_cells(program.finish(&[(value, 8)]), vec![ints], vec![]);
        graph.finish(root).encode()
    }
}

pub fn decode(
    bytes: &[u8],
    count: usize,
    peers: &[Arc<Cells>],
    budget: &DecodeBudget,
) -> Result<Col> {
    let graph = PayloadGraph::decode(bytes, budget)?;
    if graph.rows as usize != count {
        return Err(Error::Format("column row count mismatch".into()));
    }
    CoreDecoder::decode_col(
        &graph,
        &ExtInputs {
            peers,
            ..ExtInputs::empty()
        },
        budget,
    )
}

pub fn decode_i64(bytes: &[u8], count: usize, budget: &DecodeBudget) -> Result<Vec<i64>> {
    let column = decode(bytes, count, &[], budget)?;
    let mut scratch = Vec::new();
    (0..count)
        .map(|r| {
            let bytes = column.get(r, &mut scratch)?;
            Ok(i64::from_le_bytes(bytes.try_into().map_err(|_| {
                Error::Format("integer column width".into())
            })?))
        })
        .collect()
}

/// Evaluate a predicate once per dictionary entry, then apply its row ordinals.
pub fn matching(column: &Col, mut test: impl FnMut(&[u8]) -> bool) -> Result<Vec<bool>> {
    match column {
        Col::Const { val, n } => Ok(vec![test(val); *n]),
        Col::Dict { dict, ords } => {
            let matches: Vec<_> = dict.iter().map(&mut test).collect();
            Ok(ords.iter().map(|&i| matches[i as usize]).collect())
        }
        _ => {
            let mut scratch = Vec::new();
            (0..column.len())
                .map(|r| Ok(test(column.get(r, &mut scratch)?)))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_lanes_preserve_signed_extremes_and_sparse_selection() {
        let values = [i64::MIN, i64::MAX, -1, 0, 1, 9_007_199_254_740_993];
        let bytes = Encoder::default().encode_i64(&values);
        assert_eq!(
            decode_i64(&bytes, values.len(), &DecodeBudget::default()).unwrap(),
            values
        );
        assert!(decode_i64(&bytes, values.len() + 1, &DecodeBudget::default()).is_err());
    }

    #[test]
    fn external_rows_and_exception_patches_use_the_generic_decoder() {
        let peer: Vec<Vec<u8>> = (0..256u64)
            .map(|i| {
                let mut x = i + 1;
                (0..40)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        x as u8
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<_> = peer.iter().map(Vec::as_slice).collect();
        let mut rows: Vec<_> = (0..256).map(|i| refs[(i + 73) % 256]).collect();
        rows[7] = b"an exception";
        let (bytes, uses_peer) = Encoder::default().encode_with_peer(&rows, &refs);
        assert!(uses_peer);
        let mut cells = Cells::default();
        for v in &refs {
            cells.push(v);
        }
        let decoded = decode(
            &bytes,
            rows.len(),
            &[Arc::new(cells)],
            &DecodeBudget::default(),
        )
        .unwrap();
        assert_eq!(decoded.to_cells().unwrap().iter().collect::<Vec<_>>(), rows);
        assert!(decode(&bytes, rows.len(), &[], &DecodeBudget::default()).is_err());
    }
}
