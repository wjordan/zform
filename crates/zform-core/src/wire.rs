//! Wire primitives, integrity hashes (XXH64, CRC32, SHA-256), explicit decode budgets,
//! and portable entropy backends (Raw, Zstd via `zstd-safe`, LZ4 via `lz4_flex`).

use std::cell::RefCell;
use std::fmt;

thread_local! {
    static ZSTD_ENC_SCRATCH: RefCell<(Option<zstd_safe::CCtx<'static>>, Vec<u8>)> =
        const { RefCell::new((None, Vec::new())) };
    static ZSTD_DEC_SCRATCH: RefCell<Option<zstd_safe::DCtx<'static>>> =
        const { RefCell::new(None) };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Format(String),
    ChecksumMismatch(&'static str),
    BudgetExceeded(&'static str),
    Io(String),
    InvalidArgument(String),
    /// A reader touched a part of a block it was not given.
    NotLoaded,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Format(s) => write!(f, "format error: {s}"),
            Error::ChecksumMismatch(ctx) => write!(f, "checksum mismatch in {ctx}"),
            Error::BudgetExceeded(ctx) => write!(f, "decode budget exceeded: {ctx}"),
            Error::Io(s) => write!(f, "io error: {s}"),
            Error::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Error::NotLoaded => write!(f, "block part not loaded"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Explicit resource bounds enforced during untrusted stream/frame decoding.
/// Replaces legacy hidden thread-local state with deterministic tracking.
#[derive(Debug, Clone, Copy)]
pub struct DecodeBudget {
    pub max_frame_compressed_bytes: usize,
    pub max_single_alloc_bytes: usize,
    pub max_block_decoded_bytes: usize,
    pub max_records_per_block: usize,
    pub verify_checksums: bool,
}

impl Default for DecodeBudget {
    fn default() -> Self {
        Self {
            max_frame_compressed_bytes: 512 << 20, // 512 MiB
            max_single_alloc_bytes: 256 << 20,     // 256 MiB
            max_block_decoded_bytes: 512 << 20,    // 512 MiB
            max_records_per_block: 4_000_000,
            verify_checksums: true,
        }
    }
}

impl DecodeBudget {
    #[inline]
    pub fn check_alloc(&self, bytes: usize, _ctx: &'static str) -> Result<()> {
        if bytes > self.max_single_alloc_bytes {
            return Err(Error::BudgetExceeded("single allocation limit"));
        }
        Ok(())
    }

    #[inline]
    pub fn alloc_vec(&self, len: usize, ctx: &'static str) -> Result<Vec<u8>> {
        self.check_alloc(len, ctx)?;
        let mut v = Vec::new();
        v.try_reserve_exact(len)
            .map_err(|_| Error::BudgetExceeded("out of memory"))?;
        v.resize(len, 0);
        Ok(v)
    }
}

#[inline]
pub fn zigzag_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
pub fn zigzag_decode(u: u64) -> i64 {
    ((u >> 1) as i64) ^ (-((u & 1) as i64))
}

#[inline]
pub fn put_uvarint(out: &mut Vec<u8>, mut x: u64) {
    while x >= 0x80 {
        out.push((x as u8) | 0x80);
        x >>= 7;
    }
    out.push(x as u8);
}

#[inline]
pub fn put_ivarint(out: &mut Vec<u8>, v: i64) {
    put_uvarint(out, zigzag_encode(v));
}

#[inline]
pub fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_uvarint(out, b.len() as u64);
    out.extend_from_slice(b);
}

#[inline]
pub fn get_uvarint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut x = 0u64;
    let mut shift = 0u32;
    for _ in 0..10 {
        let &b = buf
            .get(*pos)
            .ok_or_else(|| Error::Format("truncated varint".into()))?;
        *pos += 1;
        if shift == 63 && b > 1 {
            return Err(Error::Format("varint overflow".into()));
        }
        x |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            return Ok(x);
        }
        shift += 7;
    }
    Err(Error::Format("varint too long".into()))
}

#[inline]
pub fn get_ivarint(buf: &[u8], pos: &mut usize) -> Result<i64> {
    Ok(zigzag_decode(get_uvarint(buf, pos)?))
}

pub struct Cursor<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    #[inline]
    pub fn u8(&mut self) -> Result<u8> {
        let &b = self
            .buf
            .get(self.pos)
            .ok_or_else(|| Error::Format("unexpected EOF (u8)".into()))?;
        self.pos += 1;
        Ok(b)
    }

    #[inline]
    pub fn u16_le(&mut self) -> Result<u16> {
        let s = self.slice(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    #[inline]
    pub fn u32_le(&mut self) -> Result<u32> {
        let s = self.slice(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    #[inline]
    pub fn u64_le(&mut self) -> Result<u64> {
        let s = self.slice(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }

    #[inline]
    pub fn uvar(&mut self) -> Result<u64> {
        get_uvarint(self.buf, &mut self.pos)
    }

    #[inline]
    pub fn ivar(&mut self) -> Result<i64> {
        get_ivarint(self.buf, &mut self.pos)
    }

    #[inline]
    pub fn slice(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| Error::Format("slice length overflow".into()))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| Error::Format("unexpected EOF (slice)".into()))?;
        self.pos = end;
        Ok(s)
    }

    #[inline]
    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.uvar()? as usize;
        self.slice(len)
    }

    #[inline]
    pub fn str_utf8(&mut self) -> Result<&'a str> {
        let b = self.bytes()?;
        std::str::from_utf8(b).map_err(|_| Error::Format("invalid utf-8 in metadata".into()))
    }
}

// --- Pluggable Entropy Backends ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntropyBackend {
    Raw = 0,
    Zstd = 1,
    Lz4 = 2,
}

impl EntropyBackend {
    pub fn from_u8(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Zstd),
            2 => Ok(Self::Lz4),
            _ => Err(Error::Format(format!("unknown entropy backend id {b}"))),
        }
    }
}

/// Compresses `src` with `backend` (falling back to `Raw` if compression does not shrink `src`).
/// Writes `[tag: u8][raw_len: varint][payload_len: varint][payload]` to `out`.
pub fn compress_leaf(out: &mut Vec<u8>, src: &[u8], backend: EntropyBackend, zstd_level: i32) {
    if src.len() < 32 || backend == EntropyBackend::Raw {
        out.push(EntropyBackend::Raw as u8);
        put_uvarint(out, src.len() as u64);
        put_uvarint(out, src.len() as u64);
        out.extend_from_slice(src);
        return;
    }
    match backend {
        EntropyBackend::Raw => unreachable!(),
        EntropyBackend::Zstd => {
            let bound = zstd_safe::compress_bound(src.len());
            let wrote = ZSTD_ENC_SCRATCH.with(|cell| {
                let mut borrow = cell.borrow_mut();
                let (cctx_opt, tmp) = &mut *borrow;
                if cctx_opt.is_none() {
                    *cctx_opt = zstd_safe::CCtx::try_create();
                }
                let Some(ctx) = cctx_opt.as_mut() else {
                    return false;
                };
                let _ = ctx.reset(zstd_safe::ResetDirective::SessionOnly);
                let _ = ctx.set_parameter(zstd_safe::CParameter::CompressionLevel(zstd_level));
                let _ = ctx.set_parameter(zstd_safe::CParameter::ChecksumFlag(false));
                if tmp.len() < bound {
                    tmp.resize(bound, 0);
                }
                if let Ok(n) = ctx.compress2(&mut tmp[..bound], src)
                    && n + 2 < src.len()
                {
                    out.push(EntropyBackend::Zstd as u8);
                    put_uvarint(out, src.len() as u64);
                    put_uvarint(out, n as u64);
                    out.extend_from_slice(&tmp[..n]);
                    return true;
                }
                false
            });
            if wrote {
                return;
            }
        }
        EntropyBackend::Lz4 => {
            let tmp = lz4_flex::block::compress(src);
            if tmp.len() + 2 < src.len() {
                out.push(EntropyBackend::Lz4 as u8);
                put_uvarint(out, src.len() as u64);
                put_uvarint(out, tmp.len() as u64);
                out.extend_from_slice(&tmp);
                return;
            }
        }
    }
    out.push(EntropyBackend::Raw as u8);
    put_uvarint(out, src.len() as u64);
    put_uvarint(out, src.len() as u64);
    out.extend_from_slice(src);
}

/// Decompresses a leaf written by [`compress_leaf`].
pub fn decompress_leaf(cur: &mut Cursor<'_>, budget: &DecodeBudget) -> Result<Vec<u8>> {
    let tag = EntropyBackend::from_u8(cur.u8()?)?;
    let raw_len = cur.uvar()? as usize;
    let comp_len = cur.uvar()? as usize;
    let payload = cur.slice(comp_len)?;
    match tag {
        EntropyBackend::Raw => {
            if comp_len != raw_len {
                return Err(Error::Format("raw leaf length mismatch".into()));
            }
            budget.check_alloc(raw_len, "raw leaf")?;
            Ok(payload.to_vec())
        }
        EntropyBackend::Zstd => {
            let mut dst = budget.alloc_vec(raw_len, "zstd leaf")?;
            let n = ZSTD_DEC_SCRATCH.with(|cell| -> Result<usize> {
                let mut borrow = cell.borrow_mut();
                if borrow.is_none() {
                    *borrow = zstd_safe::DCtx::try_create();
                }
                let dctx = borrow
                    .as_mut()
                    .ok_or_else(|| Error::Format("failed to create zstd DCtx".into()))?;
                let _ = dctx.reset(zstd_safe::ResetDirective::SessionOnly);
                dctx.decompress(&mut dst[..], payload).map_err(|code| {
                    Error::Format(format!(
                        "zstd decompress error: {}",
                        zstd_safe::get_error_name(code)
                    ))
                })
            })?;
            if n != raw_len {
                return Err(Error::Format(format!(
                    "zstd decoded {n} != expected {raw_len}"
                )));
            }
            Ok(dst)
        }
        EntropyBackend::Lz4 => {
            budget.check_alloc(raw_len, "lz4 leaf")?;
            let dst = lz4_flex::block::decompress(payload, raw_len)
                .map_err(|e| Error::Format(format!("lz4 decompress error: {e}")))?;
            if dst.len() != raw_len {
                return Err(Error::Format("lz4 decoded length mismatch".into()));
            }
            Ok(dst)
        }
    }
}

/// Skips a leaf written by [`compress_leaf`] without decompressing! Essential for fast projection.
#[inline]
pub fn skip_leaf(cur: &mut Cursor<'_>) -> Result<usize> {
    let _tag = EntropyBackend::from_u8(cur.u8()?)?;
    let raw_len = cur.uvar()? as usize;
    let comp_len = cur.uvar()? as usize;
    let _ = cur.slice(comp_len)?;
    Ok(raw_len)
}

// --- Integrity Hashes: XXH64, CRC32 (IEEE), and SHA-256 ---

pub fn xxh64(data: &[u8], seed: u64) -> u64 {
    const P1: u64 = 11400714785074694791;
    const P2: u64 = 14029467366897019727;
    const P3: u64 = 1609587929392839161;
    const P4: u64 = 9650029242287828579;
    const P5: u64 = 2870177450012600261;

    #[inline(always)]
    fn round(acc: u64, input: u64) -> u64 {
        acc.wrapping_add(input.wrapping_mul(P2))
            .rotate_left(31)
            .wrapping_mul(P1)
    }
    #[inline(always)]
    fn merge_round(mut acc: u64, val: u64) -> u64 {
        acc ^= round(0, val);
        acc.wrapping_mul(P1).wrapping_add(P4)
    }

    let mut p = 0usize;
    let len = data.len();
    let mut h: u64 = if len >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);
        while p + 32 <= len {
            v1 = round(v1, u64::from_le_bytes(data[p..p + 8].try_into().unwrap()));
            v2 = round(
                v2,
                u64::from_le_bytes(data[p + 8..p + 16].try_into().unwrap()),
            );
            v3 = round(
                v3,
                u64::from_le_bytes(data[p + 16..p + 24].try_into().unwrap()),
            );
            v4 = round(
                v4,
                u64::from_le_bytes(data[p + 24..p + 32].try_into().unwrap()),
            );
            p += 32;
        }
        let mut h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        merge_round(h, v4)
    } else {
        seed.wrapping_add(P5)
    };

    h = h.wrapping_add(len as u64);
    while p + 8 <= len {
        let k1 = round(0, u64::from_le_bytes(data[p..p + 8].try_into().unwrap()));
        h ^= k1;
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        p += 8;
    }
    if p + 4 <= len {
        h ^= (u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as u64).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        p += 4;
    }
    while p < len {
        h ^= (data[p] as u64).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
        p += 1;
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
    let mut out = String::with_capacity(64);
    for word in state {
        use std::fmt::Write;
        let _ = write!(&mut out, "{word:08x}");
    }
    out
}
