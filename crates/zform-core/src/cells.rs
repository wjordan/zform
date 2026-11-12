//! Decoded columns. Most codecs keep their compact form and produce a cell's bytes on demand, so
//! decoding a block costs little for the rows nobody reads.

use crate::wire::{Error, Result};
use std::sync::Arc;

/// Byte strings stored back to back.
#[derive(Debug, Clone, Default)]
pub struct Cells {
    buf: Vec<u8>,
    ends: Vec<u32>,
}

impl Cells {
    pub fn with_capacity(cells: usize, bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            ends: Vec::with_capacity(cells),
        }
    }

    /// Cells of the given lengths laid out in `buf`, which they must exactly cover.
    pub fn from_lens(buf: Vec<u8>, lens: &[u32]) -> Result<Self> {
        let ends = ends_of(lens, buf.len())?;
        Ok(Self { buf, ends })
    }

    /// Cells with precomputed monotonic cumulative `ends`, which must end at `buf.len()`.
    pub fn from_parts(buf: Vec<u8>, ends: Vec<u32>) -> Result<Self> {
        if buf.len() > u32::MAX as usize {
            return Err(Error::BudgetExceeded("column bytes"));
        }
        let mut prev = 0u32;
        for &e in &ends {
            if e < prev {
                return Err(Error::Format("non-monotonic cell ends".into()));
            }
            prev = e;
        }
        if prev as usize != buf.len() {
            return Err(Error::Format("cell ends do not cover their bytes".into()));
        }
        Ok(Self { buf, ends })
    }

    /// `rows` fixed-width cells of `width` bytes laid back to back in `buf`.
    pub fn from_fixed(rows: usize, width: usize, buf: Vec<u8>) -> Result<Self> {
        let total = rows
            .checked_mul(width)
            .ok_or(Error::BudgetExceeded("column bytes"))?;
        if total != buf.len() || total > u32::MAX as usize {
            return Err(Error::Format("fixed-width buffer length mismatch".into()));
        }
        let mut ends = Vec::with_capacity(rows);
        let w = width as u32;
        let mut pos = 0u32;
        for _ in 0..rows {
            pos += w;
            ends.push(pos);
        }
        Ok(Self { buf, ends })
    }

    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    #[inline]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        let end = *self.ends.get(i)? as usize;
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        Some(&self.buf[start..end])
    }

    pub fn push(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
        self.seal();
    }

    /// The buffer the next cell is written into; `seal` ends that cell.
    pub fn tail(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }

    pub fn seal(&mut self) {
        self.ends.push(self.buf.len() as u32);
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn ends(&self) -> &[u32] {
        &self.ends
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        (0..self.len()).map(|i| self.get(i).unwrap())
    }
}

/// Running ends of cells with the given lengths, which must sum to `total`.
pub(crate) fn ends_of(lens: &[u32], total: usize) -> Result<Vec<u32>> {
    if total > u32::MAX as usize {
        return Err(Error::BudgetExceeded("column bytes"));
    }
    let mut ends = Vec::with_capacity(lens.len());
    let mut pos = 0u64;
    for &l in lens {
        pos += l as u64;
        ends.push(pos as u32);
    }
    if pos != total as u64 {
        return Err(Error::Format(
            "cell lengths do not cover their bytes".into(),
        ));
    }
    Ok(ends)
}

/// Marks a `Quoted` cell that is the bare token `null`.
pub(crate) const NULL_CELL: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub enum Col {
    Flat(Cells),
    Const {
        val: Vec<u8>,
        n: usize,
    },
    /// `ords` are checked against the dictionary when the column is decoded.
    Dict {
        dict: Arc<Cells>,
        ords: Vec<u32>,
    },
    /// Cells laid back to back in the block's text pool from `base`.
    Pool {
        pool: Arc<Vec<u8>>,
        base: usize,
        ends: Vec<u32>,
    },
    /// `"inner"` per cell. With a non-empty `map`, cell `i` is `inner[map[i]]`, or `null`.
    Quoted {
        inner: Box<Col>,
        map: Vec<u32>,
    },
}

pub(crate) fn underflow() -> Error {
    Error::Format("column occurrence underflow".into())
}

impl Col {
    pub fn len(&self) -> usize {
        match self {
            Col::Flat(c) => c.len(),
            Col::Const { n, .. } => *n,
            Col::Dict { ords, .. } => ords.len(),
            Col::Pool { ends, .. } => ends.len(),
            Col::Quoted { inner, map } => {
                if map.is_empty() {
                    inner.len()
                } else {
                    map.len()
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The cell's bytes when they exist as one stored slice.
    #[inline]
    fn slice(&self, i: usize) -> Option<Result<&[u8]>> {
        Some(match self {
            Col::Flat(c) => c.get(i).ok_or_else(underflow),
            Col::Const { val, n } => {
                if i < *n {
                    Ok(val.as_slice())
                } else {
                    Err(underflow())
                }
            }
            Col::Dict { dict, ords } => ords
                .get(i)
                .and_then(|&o| dict.get(o as usize))
                .ok_or_else(underflow),
            Col::Pool { pool, base, ends } => match ends.get(i) {
                Some(&e) => {
                    let s = if i == 0 { 0 } else { ends[i - 1] as usize };
                    Ok(&pool[base + s..base + e as usize])
                }
                None => Err(underflow()),
            },
            Col::Quoted { .. } => return None,
        })
    }

    /// The cells' bytes laid back to back with each cell's end, when the column holds them so.
    pub(crate) fn laid(&self) -> Option<(&[u8], &[u32])> {
        match self {
            Col::Flat(c) => Some((c.bytes(), c.ends())),
            Col::Pool { pool, base, ends } => Some((&pool[*base..], ends)),
            _ => None,
        }
    }

    /// Appends cell `i` to `out`.
    #[inline]
    pub fn write(&self, i: usize, out: &mut Vec<u8>) -> Result<()> {
        if let Some(s) = self.slice(i) {
            out.extend_from_slice(s?);
            return Ok(());
        }
        match self {
            Col::Quoted { inner, map } => {
                let j = if map.is_empty() {
                    i
                } else {
                    *map.get(i).ok_or_else(underflow)? as usize
                };
                if j == NULL_CELL as usize {
                    out.extend_from_slice(b"null");
                } else {
                    out.push(b'"');
                    inner.write(j, out)?;
                    out.push(b'"');
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Cell `i`, built in `scratch` when the column does not store it as a slice.
    #[inline]
    pub fn get<'a>(&'a self, i: usize, scratch: &'a mut Vec<u8>) -> Result<&'a [u8]> {
        if let Some(s) = self.slice(i) {
            return s;
        }
        scratch.clear();
        self.write(i, scratch)?;
        Ok(scratch.as_slice())
    }

    /// Every cell written out.
    pub fn to_cells(&self) -> Result<Cells> {
        let n = self.len();
        let mut out = Cells::with_capacity(n, n * 8);
        for i in 0..n {
            self.write(i, out.tail())?;
            out.seal();
        }
        Ok(out)
    }
}
