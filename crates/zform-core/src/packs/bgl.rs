use super::*;

/// Ordered microsecond clocks retain their cadence across second boundaries.
pub struct BglClockPack;

impl CodecPack for BglClockPack {
    fn name(&self) -> &'static str {
        "pack/bgl-clock"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 8 && p.any_iso_ts
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        // Separate near-uniform jitter bits from the burst-dependent upper lane.
        Self::encode(rows, &[0, 10])
    }
}

impl BglClockPack {
    fn encode(rows: &[&[u8]], widths: &[i64]) -> Option<PayloadGraph> {
        let (layout, first_sec, first_sub) = TimestampIsoPack::parse_iso(rows.first()?)?;
        if layout.frac_digits != 6 {
            return None;
        }
        let mut last = first_sec * 1_000_000 + first_sub;
        let mut deltas = Vec::with_capacity(rows.len());
        for row in rows {
            let (other, sec, sub) = TimestampIsoPack::parse_iso(row)?;
            if other != layout {
                return None;
            }
            let ticks = sec * 1_000_000 + sub;
            deltas.push(ticks - last);
            last = ticks;
        }
        let program = TimestampIsoPack::prog_iso_inner(&layout, first_sec, first_sub, true);
        let mut best = None;
        for &bits in widths {
            let mut gb = GraphBuilder::new(rows.len());
            let input = if bits == 0 {
                gb.add_ints(&deltas)
            } else {
                let low: Vec<_> = deltas.iter().map(|v| v & ((1 << bits) - 1)).collect();
                let high: Vec<_> = deltas.iter().map(|v| v >> bits).collect();
                let low = gb.add_ints(&low);
                let high = gb.add_ints(&high);
                let b = Builder::new(2);
                let value = b.rr(Op::Add, b.input(0), b.ri(Op::ShlI, b.input(1), bits));
                gb.add_prog_i64(b.finish(&[(value, 8)]), vec![low, high], vec![])
            };
            let root = gb.add_prog_cells(program.clone(), vec![input], vec![]);
            let graph = gb.finish(root);
            let cost = graph.priced().1;
            if best.as_ref().is_none_or(|(size, _)| cost < *size) {
                best = Some((cost, graph));
            }
        }
        best.map(|(_, graph)| graph)
    }
}

/// BGL hardware locations, packed as coordinates rather than text dictionary entries.
pub struct BglNodePack;

impl BglNodePack {
    fn parse(s: &[u8]) -> Option<i64> {
        if s.len() != 19
            || s[0] != b'R'
            || &s[3..5] != b"-M"
            || &s[6..8] != b"-N"
            || s[9] != b'-'
            || &s[11..13] != b":J"
            || &s[15..17] != b"-U"
            || !matches!(s[10], b'C' | b'I')
        {
            return None;
        }
        let hex = |v: u8| {
            if v.is_ascii_lowercase() {
                None
            } else {
                hex_val(v).map(i64::from)
            }
        };
        let rack = hex(s[1])? * 16 + hex(s[2])?;
        let mid = hex(s[5])?;
        let node = hex(s[8])?;
        let card = i64::from(TimestampIsoPack::parse_two(&s[13..15])?);
        let chip = i64::from(TimestampIsoPack::parse_two(&s[17..19])?);
        Some(
            rack | (mid << 8)
                | (node << 12)
                | (card << 16)
                | (chip << 23)
                | (i64::from(s[10] == b'I') << 30),
        )
    }

    fn program() -> Program {
        let b = Builder::new(1);
        let v = b.input(0);
        let field = |shift, mask| b.ri(Op::AndI, b.ri(Op::ShrI, v, shift), mask);
        let lut = b.add_lut16(*b"0123456789ABCDEF");
        let hex = |v| b.ri(Op::AndI, b.nibble_lut(v, lut), 255);
        let rack_hi = hex(field(4, 15));
        let rack_lo = hex(field(0, 15));
        let mid = hex(field(8, 15));
        let node = hex(field(12, 15));
        let kind = b.ri(Op::AddI, b.ri(Op::MulI, field(30, 1), 6), b'C' as i64);
        let card = two_digits(&b, field(16, 127));
        let chip = two_digits(&b, field(23, 127));
        let s0 = b.ri(
            Op::OrI,
            b.rr(
                Op::Or,
                b.rr(
                    Op::Or,
                    b.ri(Op::ShlI, rack_hi, 8),
                    b.ri(Op::ShlI, rack_lo, 16),
                ),
                b.ri(Op::ShlI, mid, 40),
            ),
            u64::from_le_bytes(*b"R\0\0-M\0-N") as i64,
        );
        let s1 = b.ri(
            Op::OrI,
            b.rr(
                Op::Or,
                b.rr(Op::Or, node, b.ri(Op::ShlI, kind, 16)),
                b.ri(Op::ShlI, card, 40),
            ),
            u64::from_le_bytes(*b"\0-\0:J\0\0-") as i64,
        );
        let s2 = b.ri(Op::OrI, b.ri(Op::ShlI, chip, 8), b'U' as i64);
        b.finish(&[(s0, 8), (s1, 8), (s2, 3)])
    }
}

impl CodecPack for BglNodePack {
    fn name(&self) -> &'static str {
        "pack/bgl-node"
    }

    fn precheck(&self, p: &ColumnProfile) -> bool {
        p.n_rows >= 8 && p.all_ascii && p.max_len == 19 && !p.any_digit_start
    }

    fn try_encode(&self, rows: &[&[u8]]) -> Option<PayloadGraph> {
        let probe = rows.len().min(16);
        if probe < 8
            || rows[..probe]
                .iter()
                .filter(|r| Self::parse(r).is_some())
                .count()
                * 2
                < probe
        {
            return None;
        }
        let mut values = Vec::with_capacity(rows.len());
        let (mut exc_idx, mut exc_vals) = (Vec::new(), Vec::new());
        let mut last = 0;
        for (i, &row) in rows.iter().enumerate() {
            if let Some(v) = Self::parse(row) {
                last = v;
            } else {
                exc_idx.push(i);
                exc_vals.push(row);
                if exc_idx.len() * 8 > rows.len() {
                    return None;
                }
            }
            values.push(last);
        }
        let mut gb = GraphBuilder::new(rows.len());
        let ids = gb.add_mtf_ints(&values);
        let cells = gb.add_prog_cells(Self::program(), vec![ids], vec![]);
        let root = gb.add_patch(cells, &exc_idx, &exc_vals);
        Some(gb.finish(root))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{payload::graph::CoreDecoder, wire::DecodeBudget};

    #[test]
    fn clock_keeps_rollovers_backwards_steps_and_layout() {
        for suffix in ["", "Z", "+01:00"] {
            let mut rows = Vec::new();
            for i in 0..24 {
                let time = match i % 6 {
                    0 => "2005-12-31-23.59.59.999999",
                    1 => "2006-01-01-00.00.00.000001",
                    2 => "2006-01-01-00.00.00.000001",
                    3 => "1970-01-01-00.00.00.000000",
                    4 => "9999-12-31-23.59.59.999999",
                    _ => "2005-12-31-23.59.59.000003",
                };
                rows.push(format!("{time}{suffix}").into_bytes());
            }
            let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
            for bits in [0, 10] {
                let graph = BglClockPack::encode(&refs, &[bits]).unwrap();
                let back =
                    CoreDecoder::decode_bytes(&graph.encode(), &DecodeBudget::default()).unwrap();
                for (i, want) in rows.iter().enumerate() {
                    assert_eq!(back.get(i).unwrap(), want);
                }
            }
            rows[7] = b"malformed".to_vec();
            let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
            assert!(BglClockPack.try_encode(&refs).is_none());
        }
        assert!(BglClockPack.try_encode(&[]).is_none());
        assert!(
            BglClockPack
                .try_encode(&[b"2005-12-31-23.59.59.999"])
                .is_none()
        );
    }

    #[test]
    fn locations_keep_coordinates_kind_and_malformed_rows() {
        let mut rows: Vec<Vec<u8>> = (0..64)
            .map(|i| {
                format!(
                    "R{i:02X}-M{:X}-N{:X}-{}:J{:02}-U{:02}",
                    i % 2,
                    i % 16,
                    if i % 3 == 0 { 'I' } else { 'C' },
                    i % 100,
                    (i * 13) % 100
                )
                .into_bytes()
            })
            .collect();
        rows[7] = b"-".to_vec();
        rows[21] = b"R00-M0-Na-C:J00-U00".to_vec();
        rows[63] = b"RFF-MF-NF-I:J99-U99".to_vec();
        let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
        let graph = BglNodePack.try_encode(&refs).unwrap();
        let back = CoreDecoder::decode_bytes(&graph.encode(), &DecodeBudget::default()).unwrap();
        for (i, want) in rows.iter().enumerate() {
            assert_eq!(back.get(i).unwrap(), want);
        }
        assert!(
            BglNodePack
                .try_encode(&[b"R00-M0-N0-C:J0x-U00".as_slice(); 16])
                .is_none()
        );
    }
}
