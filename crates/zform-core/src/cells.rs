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
        // Each end is at most `total`, which fits in u32.
        let ends = (1..=rows).map(|i| (i * width) as u32).collect();
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

/// The span of a paged cell not held, which reads as `NotLoaded`.
pub(crate) const NOT_HELD: (u32, u32) = (u32::MAX, 0);

/// Marks a `Quoted` cell that is the bare token `null`.
pub(crate) const NULL_CELL: u32 = u32::MAX;

/// A block's text pool, which a reader may hold only some pages of. A cell in a page it lacks is
/// `NotLoaded`.
#[derive(Debug, Default)]
pub struct Pool {
    bytes: Vec<u8>,
    /// Byte ranges not held, ascending and disjoint.
    missing: Vec<(u32, u32)>,
}

impl Pool {
    pub fn whole(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            ..Self::default()
        }
    }

    pub(crate) fn partial(bytes: Vec<u8>, missing: Vec<(u32, u32)>) -> Self {
        Self { bytes, missing }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn is_whole(&self) -> bool {
        self.missing.is_empty()
    }

    /// All the bytes, when the pool is whole.
    pub fn all(&self) -> Result<&[u8]> {
        match self.missing.is_empty() {
            true => Ok(&self.bytes),
            false => Err(Error::NotLoaded),
        }
    }

    /// The bytes, those not held zeroed.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Whether bytes `s..e` are held.
    pub(crate) fn held(&self, s: usize, e: usize) -> bool {
        let i = self.missing.partition_point(|m| (m.1 as usize) <= s);
        !self.missing.get(i).is_some_and(|m| (m.0 as usize) < e)
    }

}

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
    /// Cells laid back to back in the block's text pool, which is whole, from `base`.
    Pool {
        pool: Arc<Pool>,
        base: usize,
        ends: Vec<u32>,
    },
    /// Cells at `(start, end)` in the block's paged text pool, or in a pool of only the cells a
    /// reader asked for; `NOT_HELD` for one not held.
    Paged {
        pool: Arc<Pool>,
        spans: Vec<(u32, u32)>,
    },
    /// `"inner"` per cell. With a non-empty `map`, cell `i` is `inner[map[i]]`, or `null`.
    Quoted {
        inner: Box<Col>,
        map: Vec<u32>,
    },
}

/// Appends `buf[s..e]` to `out`. Most cells are short, and a fixed 32-byte copy costs less than a
/// memmove call; bytes past `e` land in `out`'s spare capacity and do not count.
#[inline(always)]
pub(crate) fn put(out: &mut Vec<u8>, buf: &[u8], s: usize, e: usize) {
    let src = &buf[s..e];
    if src.len() <= 32 && buf.len() - s >= 32 {
        out.reserve(32);
        // SAFETY: `buf` holds 32 bytes from `s`, and `out` has room for 32 past its length.
        unsafe {
            let len = out.len();
            std::ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr().add(len), 32);
            out.set_len(len + src.len());
        }
    } else {
        out.extend_from_slice(src);
    }
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
            Col::Paged { spans, .. } => spans.len(),
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

    /// The cell's bytes when they exist as one stored slice: the buffer holding them, and their
    /// range in it.
    #[inline(always)]
    fn span(&self, i: usize) -> Option<Result<(&[u8], usize, usize)>> {
        let laid = |buf, ends: &[u32], base: usize, i: usize| match ends.get(i) {
            Some(&e) => {
                let s = if i == 0 { 0 } else { ends[i - 1] as usize };
                Ok((buf, base + s, base + e as usize))
            }
            None => Err(underflow()),
        };
        Some(match self {
            Col::Flat(c) => laid(&c.buf[..], &c.ends, 0, i),
            Col::Const { val, n } => match i < *n {
                true => Ok((val.as_slice(), 0, val.len())),
                false => Err(underflow()),
            },
            Col::Dict { dict, ords } => match ords.get(i) {
                Some(&o) => laid(&dict.buf[..], &dict.ends, 0, o as usize),
                None => Err(underflow()),
            },
            Col::Pool { pool, base, ends } => laid(pool.bytes(), ends, *base, i),
            Col::Paged { pool, spans } => match spans.get(i) {
                Some(&(s, e)) if s <= e && e as usize <= pool.bytes.len() => {
                    Ok((&pool.bytes[..], s as usize, e as usize))
                }
                Some(_) => Err(Error::NotLoaded),
                None => Err(underflow()),
            },
            Col::Quoted { .. } => return None,
        })
    }

    #[inline(always)]
    fn slice(&self, i: usize) -> Option<Result<&[u8]>> {
        Some(self.span(i)?.map(|(buf, s, e)| &buf[s..e]))
    }

    /// The cells' bytes laid back to back with each cell's end, when the column holds them so.
    pub(crate) fn laid(&self) -> Option<(&[u8], &[u32])> {
        match self {
            Col::Flat(c) => Some((c.bytes(), c.ends())),
            Col::Pool { pool, base, ends } => Some((&pool.bytes()[*base..], ends)),
            _ => None,
        }
    }

    /// Appends cell `i` to `out`.
    #[inline(always)]
    pub fn write(&self, i: usize, out: &mut Vec<u8>) -> Result<()> {
        match self.span(i) {
            Some(span) => {
                let (buf, s, e) = span?;
                put(out, buf, s, e);
                Ok(())
            }
            None => self.write_quoted(i, out),
        }
    }

    #[inline(never)]
    fn write_quoted(&self, i: usize, out: &mut Vec<u8>) -> Result<()> {
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
        if let Col::Flat(c) = self {
            return Ok(c.clone());
        }
        let n = self.len();
        let mut out = Cells::with_capacity(n, n * 8);
        for i in 0..n {
            self.write(i, out.tail())?;
            out.seal();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_cells_copy_exactly_near_buffer_ends() {
        // Lengths across the 32-byte chunk, so cells end before, at and past the buffer's last
        // 32 bytes, written into vectors with no spare room.
        let lens: Vec<u32> = (0..=40).chain((0..=40).rev()).collect();
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        let buf: Vec<u8> = (0..total).map(|i| (i * 7 + 1) as u8).collect();
        let col = Col::Flat(Cells::from_lens(buf.clone(), &lens).unwrap());
        let mut all = Vec::new();
        for i in 0..col.len() {
            let mut one = Vec::new();
            col.write(i, &mut one).unwrap();
            assert_eq!(one, col.slice(i).unwrap().unwrap());
            col.write(i, &mut all).unwrap();
        }
        assert_eq!(all, buf);
    }
}
