//! Bounded standalone byte blocks using the payload entropy framing.
use crate::{DecodeBudget, Error, Result};
use zform_core::wire::{self, Cursor, EntropyBackend};

pub fn encode(src: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::new();
    encode_into(src, level, &mut out);
    out
}

pub fn encode_into(src: &[u8], level: i32, out: &mut Vec<u8>) {
    out.clear();
    wire::compress_leaf(out, src, EntropyBackend::Zstd, level);
}

/// Checks the complete frame before a caller reserves its decoded bytes.
pub fn decoded_size(src: &[u8], limit: usize) -> Result<usize> {
    let mut cursor = Cursor::new(src);
    let backend = EntropyBackend::from_u8(cursor.u8()?)?;
    let size = usize::try_from(cursor.uvar()?).map_err(|_| Error::BudgetExceeded("block size"))?;
    if size > limit {
        return Err(Error::BudgetExceeded("block size"));
    }
    let stored =
        usize::try_from(cursor.uvar()?).map_err(|_| Error::BudgetExceeded("block payload"))?;
    cursor.slice(stored)?;
    if cursor.remaining() != 0 || (backend == EntropyBackend::Raw && size != stored) {
        return Err(Error::Format("invalid block length".into()));
    }
    Ok(size)
}

pub fn decode(src: &[u8], limit: usize) -> Result<Vec<u8>> {
    decoded_size(src, limit)?;
    let budget = DecodeBudget {
        max_single_alloc_bytes: limit,
        max_block_decoded_bytes: limit,
        ..DecodeBudget::default()
    };
    wire::decompress_leaf(&mut Cursor::new(src), &budget)
}

pub fn decode_exact(src: &[u8], size: usize) -> Result<Vec<u8>> {
    if decoded_size(src, size)? != size {
        return Err(Error::Format("block decoded size mismatch".into()));
    }
    decode(src, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_blocks_reject_truncation_trailing_bytes_and_wrong_size() {
        for raw in [vec![], b"hello".to_vec(), vec![42; 10000]] {
            let bytes = encode(&raw, 1);
            assert_eq!(decode_exact(&bytes, raw.len()).unwrap(), raw);
            for n in 0..bytes.len() {
                assert!(decode(&bytes[..n], raw.len()).is_err());
            }
            let mut bad = bytes.clone();
            bad.push(0);
            assert!(decode(&bad, raw.len()).is_err());
            assert!(decode_exact(&bytes, raw.len() + 1).is_err());
            if !raw.is_empty() {
                assert!(decode(&bytes, raw.len() - 1).is_err());
            }
        }
    }
}
