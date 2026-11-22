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

    /// Internal constructor when `ends` is already known to be monotonic and terminate at `buf.len()`.
    #[inline]
    pub(crate) fn from_parts_unchecked(buf: Vec<u8>, ends: Vec<u32>) -> Self {
        debug_assert!(ends.last().copied().unwrap_or(0) as usize == buf.len());
        Self { buf, ends }
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

/// Appends a short literal `src` to `out` without hitting libc `memmove` for common 1..=32 byte
/// JSON/log literals.
#[inline(always)]
pub(crate) fn put_lit(out: &mut Vec<u8>, src: &[u8]) {
    let n = src.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        out.push(src[0]);
        return;
    }
    if n <= 32 {
        out.reserve(n);
        // SAFETY: `out` has room for `n` bytes past `len`, and `src` does not alias `out`.
        unsafe {
            let len = out.len();
            let dst = out.as_mut_ptr().add(len);
            let sp = src.as_ptr();
            if n <= 8 {
                if n >= 4 {
                    std::ptr::copy_nonoverlapping(sp, dst, 4);
                    std::ptr::copy_nonoverlapping(sp.add(n - 4), dst.add(n - 4), 4);
                } else {
                    std::ptr::copy_nonoverlapping(sp, dst, 2);
                    *dst.add(n - 1) = *sp.add(n - 1);
                }
            } else if n <= 16 {
                std::ptr::copy_nonoverlapping(sp, dst, 8);
                std::ptr::copy_nonoverlapping(sp.add(n - 8), dst.add(n - 8), 8);
            } else {
                std::ptr::copy_nonoverlapping(sp, dst, 16);
                std::ptr::copy_nonoverlapping(sp.add(n - 16), dst.add(n - 16), 16);
            }
            out.set_len(len + n);
        }
        return;
    }
    out.extend_from_slice(src);
}

/// Appends `buf[s..e]` to `out`. Most cells are short, and a fixed 32-byte copy costs less than a
/// memmove call; bytes past `e` land in `out`'s spare capacity and do not count.
#[inline(always)]
pub(crate) fn put(out: &mut Vec<u8>, buf: &[u8], s: usize, e: usize) {
    let n = e - s;
    if n <= 32 && buf.len() - s >= 32 {
        out.reserve(32);
        // SAFETY: `buf` holds 32 bytes from `s`, and `out` has room for 32 past its length.
        unsafe {
            let len = out.len();
            std::ptr::copy_nonoverlapping(buf.as_ptr().add(s), out.as_mut_ptr().add(len), 32);
            out.set_len(len + n);
        }
    } else {
        put_lit(out, &buf[s..e]);
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
    pub(crate) fn span(&self, i: usize) -> Option<Result<(&[u8], usize, usize)>> {
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

    #[inline(always)]
    fn write_quoted(&self, i: usize, out: &mut Vec<u8>) -> Result<()> {
        match self {
            Col::Quoted { inner, map } => {
                let j = if map.is_empty() {
                    i
                } else {
                    *map.get(i).ok_or_else(underflow)? as usize
                };
                if j == NULL_CELL as usize {
                    put_lit(out, b"null");
                } else if let Some(span) = inner.span(j) {
                    let (buf, s, e) = span?;
                    let n = e - s;
                    if n <= 32 && buf.len() - s >= 32 {
                        out.reserve(34);
                        // SAFETY: `buf` holds 32 bytes from `s`, `out` has 34 spare bytes.
                        unsafe {
                            let len = out.len();
                            let dst = out.as_mut_ptr().add(len);
                            *dst = b'"';
                            std::ptr::copy_nonoverlapping(buf.as_ptr().add(s), dst.add(1), 32);
                            *dst.add(1 + n) = b'"';
                            out.set_len(len + n + 2);
                        }
                    } else {
                        out.reserve(n + 2);
                        out.push(b'"');
                        put_lit(out, &buf[s..e]);
                        out.push(b'"');
                    }
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

/// Register-cached write cursor over a `Vec<u8>` for whole-block reconstruction, keeping `(ptr,
/// len, cap)` in CPU registers and syncing `Vec::len` only on cold growth or drop.
pub(crate) struct FastBuf<'a> {
    vec: &'a mut Vec<u8>,
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

impl<'a> FastBuf<'a> {
    #[inline]
    pub(crate) fn new(vec: &'a mut Vec<u8>) -> Self {
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        let cap = vec.capacity();
        Self { vec, ptr, len, cap }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr[..len]` is always initialized and valid while `self` borrows `vec`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self, need: usize) {
        // SAFETY: `self.len <= self.cap` and `ptr[..len]` is initialized.
        unsafe { self.vec.set_len(self.len) };
        self.vec.reserve(need.max(64));
        self.ptr = self.vec.as_mut_ptr();
        self.cap = self.vec.capacity();
    }

    #[inline(always)]
    pub(crate) fn put_byte(&mut self, b: u8) {
        if self.len == self.cap {
            self.grow(1);
        }
        unsafe {
            *self.ptr.add(self.len) = b;
        }
        self.len += 1;
    }

    #[inline(always)]
    pub(crate) fn put_lit(&mut self, src: &[u8]) {
        let n = src.len();
        if n == 0 {
            return;
        }
        if self.len + n > self.cap {
            self.grow(n);
        }
        // SAFETY: `len + n <= cap`, `src` does not alias `self.ptr`.
        unsafe {
            let dst = self.ptr.add(self.len);
            let sp = src.as_ptr();
            if n == 1 {
                *dst = *sp;
            } else if n <= 8 {
                if n >= 4 {
                    std::ptr::copy_nonoverlapping(sp, dst, 4);
                    std::ptr::copy_nonoverlapping(sp.add(n - 4), dst.add(n - 4), 4);
                } else {
                    std::ptr::copy_nonoverlapping(sp, dst, 2);
                    *dst.add(n - 1) = *sp.add(n - 1);
                }
            } else if n <= 16 {
                std::ptr::copy_nonoverlapping(sp, dst, 8);
                std::ptr::copy_nonoverlapping(sp.add(n - 8), dst.add(n - 8), 8);
            } else if n <= 32 {
                std::ptr::copy_nonoverlapping(sp, dst, 16);
                std::ptr::copy_nonoverlapping(sp.add(n - 16), dst.add(n - 16), 16);
            } else {
                std::ptr::copy_nonoverlapping(sp, dst, n);
            }
        }
        self.len += n;
    }

    /// Appends `arena[off..off + n]` where `arena` is guaranteed to have at least 32 bytes of
    /// trailing padding past `off + n` (zero length branches for all `n <= 32`).
    #[inline(always)]
    pub(crate) fn put_padded(&mut self, arena: &[u8], off: usize, n: usize) {
        if n <= 32 {
            if self.len + 32 > self.cap {
                self.grow(32);
            }
            // SAFETY: `arena` has >= 32 readable bytes from `off`, `self` has >= 32 spare bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(arena.as_ptr().add(off), self.ptr.add(self.len), 32);
            }
            self.len += n;
        } else {
            self.put_lit_long(&arena[off..off + n]);
        }
    }

    #[cold]
    #[inline(never)]
    fn put_lit_long(&mut self, src: &[u8]) {
        let n = src.len();
        if self.len + n > self.cap {
            self.grow(n);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(self.len), n);
        }
        self.len += n;
    }

    /// Branchlessly writes `buf[s..e]` (unquoted when `q == 0`, or `"..."` when `q == 1`).
    #[inline(always)]
    pub(crate) fn put_span_q(&mut self, buf: &[u8], s: usize, e: usize, q: usize) {
        let n = e - s;
        if n <= 32 && buf.len() - s >= 32 {
            if self.len + 34 > self.cap {
                self.grow(34);
            }
            // SAFETY: `buf` has >=32 bytes from `s`, `self` has >=34 spare bytes.
            // When `q == 0`: `*dst = b'"'` is immediately overwritten by the 32-byte copy and
            // `*dst.add(n) = b'"'` lands in spare capacity past `len + n`.
            // When `q == 1`: opening `"` sits at `dst[0]`, bytes at `dst[1..1+n]`, closing `"` at `dst[1+n]`.
            unsafe {
                let dst = self.ptr.add(self.len);
                *dst = b'"';
                std::ptr::copy_nonoverlapping(buf.as_ptr().add(s), dst.add(q), 32);
                *dst.add(q + n) = b'"';
            }
            self.len += n + (q << 1);
        } else {
            self.put_span_q_slow(&buf[s..e], q);
        }
    }

    #[cold]
    #[inline(never)]
    fn put_span_q_slow(&mut self, src: &[u8], q: usize) {
        let n = src.len();
        let need = n + (q << 1);
        if self.len + need > self.cap {
            self.grow(need);
        }
        unsafe {
            let dst = self.ptr.add(self.len);
            if q != 0 {
                *dst = b'"';
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.add(1), n);
                *dst.add(1 + n) = b'"';
            } else {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
            }
        }
        self.len += need;
    }

    #[inline(always)]
    pub(crate) fn put_quoted_span(&mut self, buf: &[u8], s: usize, e: usize) {
        self.put_span_q(buf, s, e, 1);
    }

    #[inline(always)]
    pub(crate) fn put_within(&mut self, rs: usize, re: usize) {
        let n = re - rs;
        if n == 0 {
            return;
        }
        if n <= 32 && self.len - rs >= 32 {
            if self.len + 32 > self.cap {
                self.grow(32);
            }
            // SAFETY: `self.len - rs >= 32`, so `ptr[rs..rs+32]` and `ptr[len..len+32]` are
            // completely disjoint and within bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.add(rs), self.ptr.add(self.len), 32);
            }
            self.len += n;
            return;
        }
        self.put_within_slow(rs, re);
    }

    #[cold]
    #[inline(never)]
    fn put_within_slow(&mut self, rs: usize, re: usize) {
        let n = re - rs;
        if self.len + n > self.cap {
            self.grow(n);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.add(rs), self.ptr.add(self.len), n);
        }
        self.len += n;
    }

    /// Temporarily syncs into the underlying `Vec<u8>` for cold-path writers.
    #[inline(never)]
    pub(crate) fn with_vec<R>(&mut self, f: impl FnOnce(&mut Vec<u8>) -> R) -> R {
        unsafe { self.vec.set_len(self.len) };
        let res = f(self.vec);
        self.ptr = self.vec.as_mut_ptr();
        self.len = self.vec.len();
        self.cap = self.vec.capacity();
        res
    }
}

impl Drop for FastBuf<'_> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `self.len <= self.cap` and all bytes in `..self.len` were initialized.
        unsafe { self.vec.set_len(self.len) };
    }
}

/// Pre-resolved direct column cursor unifying unquoted/quoted `Flat`/`Pool`, `Dict`, and `Const`
/// into branchless `q` (0 or 1) variants so quoted/unquoted alternation never mispredicts.
pub(crate) enum FastCol<'a> {
    /// Covers `Col::Flat`, `Col::Pool`, `Col::Quoted(Flat, [])`, and `Col::Quoted(Pool, [])`.
    Flat {
        buf: &'a [u8],
        ends: &'a [u32],
        pos: u32,
        prev_end: u32,
        q: u8,
    },
    /// Covers `Col::Dict` and `Col::Quoted(Dict, [])`.
    Dict {
        buf: &'a [u8],
        ends: &'a [u32],
        ords: &'a [u32],
        pos: u32,
        q: u8,
    },
    /// Covers `Col::Const` and `Col::Quoted(Const, [])`.
    Const {
        val: &'a [u8],
        rem: u32,
        q: u8,
    },
    Paged {
        buf: &'a [u8],
        spans: &'a [(u32, u32)],
        pos: u32,
    },
    QuotedFlatMap {
        buf: &'a [u8],
        ends: &'a [u32],
        map: &'a [u32],
        pos: u32,
    },
    General {
        col: &'a Col,
        pos: usize,
    },
    Missing,
}

impl<'a> FastCol<'a> {
    pub(crate) fn from_col(col: &'a Col, start_pos: usize) -> Self {
        let p = start_pos as u32;
        match col {
            Col::Flat(c) => {
                let prev_end = if start_pos == 0 {
                    0
                } else {
                    c.ends.get(start_pos - 1).copied().unwrap_or(0)
                };
                FastCol::Flat {
                    buf: &c.buf,
                    ends: &c.ends,
                    pos: p,
                    prev_end,
                    q: 0,
                }
            }
            Col::Pool { pool, base, ends } => {
                let prev_end = if start_pos == 0 {
                    0
                } else {
                    ends.get(start_pos - 1).copied().unwrap_or(0)
                };
                FastCol::Flat {
                    buf: &pool.bytes()[*base..],
                    ends,
                    pos: p,
                    prev_end,
                    q: 0,
                }
            }
            Col::Const { val, n } => FastCol::Const {
                val,
                rem: n.saturating_sub(start_pos) as u32,
                q: 0,
            },
            Col::Dict { dict, ords } => FastCol::Dict {
                buf: &dict.buf,
                ends: &dict.ends,
                ords,
                pos: p,
                q: 0,
            },
            Col::Paged { pool, spans } => FastCol::Paged {
                buf: pool.bytes(),
                spans,
                pos: p,
            },
            Col::Quoted { inner, map } => match inner.as_ref() {
                Col::Flat(c) if map.is_empty() => {
                    let prev_end = if start_pos == 0 {
                        0
                    } else {
                        c.ends.get(start_pos - 1).copied().unwrap_or(0)
                    };
                    FastCol::Flat {
                        buf: &c.buf,
                        ends: &c.ends,
                        pos: p,
                        prev_end,
                        q: 1,
                    }
                }
                Col::Pool { pool, base, ends } if map.is_empty() => {
                    let prev_end = if start_pos == 0 {
                        0
                    } else {
                        ends.get(start_pos - 1).copied().unwrap_or(0)
                    };
                    FastCol::Flat {
                        buf: &pool.bytes()[*base..],
                        ends,
                        pos: p,
                        prev_end,
                        q: 1,
                    }
                }
                Col::Flat(c) => FastCol::QuotedFlatMap {
                    buf: &c.buf,
                    ends: &c.ends,
                    map,
                    pos: p,
                },
                Col::Dict { dict, ords } if map.is_empty() => FastCol::Dict {
                    buf: &dict.buf,
                    ends: &dict.ends,
                    ords,
                    pos: p,
                    q: 1,
                },
                Col::Const { val, n } if map.is_empty() => FastCol::Const {
                    val,
                    rem: n.saturating_sub(start_pos) as u32,
                    q: 1,
                },
                _ => FastCol::General {
                    col,
                    pos: start_pos,
                },
            },
        }
    }

    #[inline(always)]
    pub(crate) fn write_next(&mut self, out: &mut FastBuf<'_>) -> bool {
        match self {
            FastCol::Flat {
                buf,
                ends,
                pos,
                prev_end,
                q,
            } => {
                let p = *pos as usize;
                let Some(&e) = ends.get(p) else {
                    return false;
                };
                let s = *prev_end as usize;
                *prev_end = e;
                *pos = (p + 1) as u32;
                out.put_span_q(buf, s, e as usize, *q as usize);
                true
            }
            FastCol::Dict {
                buf,
                ends,
                ords,
                pos,
                q,
            } => {
                let p = *pos as usize;
                let Some(&o) = ords.get(p) else {
                    return false;
                };
                *pos = (p + 1) as u32;
                let oi = o as usize;
                let e = ends[oi] as usize;
                let s = if oi == 0 { 0 } else { ends[oi - 1] as usize };
                out.put_span_q(buf, s, e, *q as usize);
                true
            }
            FastCol::Const { val, rem, q } => {
                if *rem == 0 {
                    return false;
                }
                *rem -= 1;
                if *q == 0 {
                    out.put_lit(val);
                } else {
                    out.put_quoted_span(val, 0, val.len());
                }
                true
            }
            FastCol::Paged { buf, spans, pos } => {
                let p = *pos as usize;
                let Some(&(s, e)) = spans.get(p) else {
                    return false;
                };
                if s > e || e as usize > buf.len() {
                    return false;
                }
                *pos = (p + 1) as u32;
                out.put_span_q(buf, s as usize, e as usize, 0);
                true
            }
            FastCol::QuotedFlatMap {
                buf,
                ends,
                map,
                pos,
            } => {
                let p = *pos as usize;
                let Some(&m) = map.get(p) else {
                    return false;
                };
                *pos = (p + 1) as u32;
                if m == NULL_CELL {
                    out.put_lit(b"null");
                } else {
                    let j = m as usize;
                    let Some(&e) = ends.get(j) else {
                        return false;
                    };
                    let s = if j == 0 { 0 } else { ends[j - 1] as usize };
                    out.put_quoted_span(buf, s, e as usize);
                }
                true
            }
            FastCol::General { col, pos } => {
                let p = *pos;
                *pos = p + 1;
                out.with_vec(|v| col.write(p, v)).is_ok()
            }
            FastCol::Missing => false,
        }
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
