//! `java.util.Random.nextLong()` outputs stored as positions on the generator's cycle.
//!
//! The generator is a 48-bit LCG with full period: every instance walks the same cycle and a
//! seed only picks where it starts. An output is 64 bits drawn from one 48-bit state, so the
//! state is recovered from it, and the output is replaced by the state's distance from an
//! anchor. Ids drawn one after another (Hadoop block ids before 2.1) become small integers,
//! and an id from an unrelated instance still fits in 48 bits.

use super::*;

const MASK: u64 = (1 << 48) - 1;
const MULT: u64 = 0x5deece66d;
const ADD: u64 = 11;

/// `JUMP[0][i]` moves a state `2^i` steps forward as `s * a + c`, `JUMP[1][i]` as far back.
const JUMP: [[(u64, u64); 48]; 2] = {
    // Newton's iteration doubles the correct low bits of the inverse; `MULT` itself has 3.
    let mut inv = MULT;
    let mut i = 0;
    while i < 4 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(MULT.wrapping_mul(inv)));
        i += 1;
    }
    let steps = [
        (MULT, ADD),
        (inv & MASK, ADD.wrapping_neg().wrapping_mul(inv) & MASK),
    ];
    let mut table = [[(0, 0); 48]; 2];
    let mut dir = 0;
    while dir < 2 {
        let (mut a, mut c) = steps[dir];
        let mut i = 0;
        while i < 48 {
            table[dir][i] = (a, c);
            c = c.wrapping_mul(a.wrapping_add(1)) & MASK;
            a = a.wrapping_mul(a) & MASK;
            i += 1;
        }
        dir += 1;
    }
    table
};

/// The value `nextLong()` returns from `state`, the state its high half is taken from.
fn output(state: u64) -> i64 {
    let next = state.wrapping_mul(MULT).wrapping_add(ADD) & MASK;
    // The low half is sign extended before the addition.
    ((state >> 16) << 32).wrapping_add((next >> 16) as i32 as u64) as i64
}

/// The state whose output is `value`. One 64-bit value in 65,536 has one, and never two.
fn state_of(value: i64) -> Option<u64> {
    let low = value as u32 as u64;
    let high = ((value as u64 >> 32) as u32).wrapping_add((low >> 31) as u32) as u64;
    // The state is `high << 16 | x`. Its successor's top 32 bits are `low` when `MULT * x`
    // lies in `[t, t + 2^16)` modulo 2^48, and `MULT * x < 2^51` leaves seven such ranges
    // with at most one multiple of `MULT` in each.
    let t = (low << 16).wrapping_sub((high << 16).wrapping_mul(MULT).wrapping_add(ADD)) & MASK;
    for k in -1..=5i64 {
        let base = t as i64 + (k << 48);
        let x = (base.max(0) as u64).div_ceil(MULT);
        if x >> 16 != 0 {
            break;
        }
        if ((x * MULT) as i64) < base + (1 << 16) {
            return Some(high << 16 | x);
        }
    }
    None
}

/// Steps from `from` to `to`, in `-2^47..2^47`.
fn distance(from: u64, to: u64) -> i64 {
    // 2^i steps flip bit i of a state and keep the bits below it, so the count comes out
    // lowest bit first.
    let (mut state, mut steps) = (from, 0u64);
    for (i, &(a, c)) in JUMP[0].iter().enumerate() {
        if state == to {
            break;
        }
        if (state ^ to) >> i & 1 != 0 {
            state = state.wrapping_mul(a).wrapping_add(c) & MASK;
            steps |= 1 << i;
        }
    }
    (steps << 16) as i64 >> 16
}

/// The state `steps` from `state`, taken modulo 2^48 and walked the shorter way round.
fn jump(mut state: u64, steps: i64) -> u64 {
    let forward = steps as u64 & MASK;
    let (table, mut left) = if forward >> 47 == 0 {
        (&JUMP[0], forward)
    } else {
        (&JUMP[1], (1 << 48) - forward)
    };
    while left != 0 {
        let (a, c) = table[left.trailing_zeros() as usize];
        state = state.wrapping_mul(a).wrapping_add(c) & MASK;
        left &= left - 1;
    }
    state
}

/// `None` unless most of `vals` are generator outputs.
pub(super) fn encode(vals: &[i64], backend: EntropyBackend, level: i32) -> Option<Vec<u8>> {
    let mut anchor = None;
    let mut steps = Vec::with_capacity(vals.len());
    let (mut other_rows, mut others) = (Vec::new(), Vec::new());
    for (row, &v) in vals.iter().enumerate() {
        // Other data almost never passes, so 16 values settle it.
        if row == 16 && others.len() > 8 {
            return None;
        }
        match state_of(v) {
            Some(state) => steps.push(distance(*anchor.get_or_insert(state), state)),
            None => {
                other_rows.push(row as u32);
                others.push(v);
            }
        }
    }
    if others.len() > steps.len() {
        return None;
    }
    let mut out = anchor?.to_le_bytes()[..6].to_vec();
    encode_u32_seq(&other_rows, backend, level, &mut out);
    if !others.is_empty() {
        encode_i64_seq_with(&others, backend, level, 0, &mut out);
    }
    encode_i64_seq_with(&steps, backend, level, 0, &mut out);
    Some(out)
}

pub(super) fn decode(
    cur: &mut Cursor<'_>,
    n: usize,
    budget: &DecodeBudget,
    depth: u8,
) -> Result<Vec<i64>> {
    let mut anchor = [0u8; 8];
    anchor[..6].copy_from_slice(cur.slice(6)?);
    let anchor = u64::from_le_bytes(anchor);
    let other_rows = decode_u32_seq(cur, budget)?;
    let others = if other_rows.is_empty() {
        Vec::new()
    } else {
        decode_i64_seq_at(cur, budget, depth + 1)?
    };
    let steps = decode_i64_seq_at(cur, budget, depth + 1)?;
    if others.len() != other_rows.len() || steps.len().checked_add(others.len()) != Some(n) {
        return Err(Error::Format("generator value count mismatch".into()));
    }
    budget.check_alloc(n * 8, "generator output")?;
    let mut out = Vec::with_capacity(n);
    let mut steps = steps.into_iter();
    let mut others = other_rows.into_iter().zip(others).peekable();
    for row in 0..n {
        out.push(match others.next_if(|&(r, _)| r as usize == row) {
            Some((_, v)) => v,
            // Rows out of order leave too few steps for the rows that remain.
            None => output(jump(
                anchor,
                steps
                    .next()
                    .ok_or_else(|| Error::Format("generator rows out of order".into()))?,
            )),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const Z: EntropyBackend = EntropyBackend::Zstd;

    /// `new Random(seed)`, as the state its first `nextLong()` reads.
    fn seeded(seed: u64) -> u64 {
        jump((seed ^ MULT) & MASK, 1)
    }

    fn outputs(mut state: u64, n: usize) -> Vec<i64> {
        (0..n)
            .map(|_| {
                let v = output(state);
                state = jump(state, 2);
                v
            })
            .collect()
    }

    fn roundtrip(vals: &[i64]) -> Vec<u8> {
        let mut enc = Vec::new();
        encode_i64_seq(vals, Z, 3, &mut enc);
        let budget = DecodeBudget::default();
        assert_eq!(
            decode_i64_seq(&mut Cursor::new(&enc), &budget).unwrap(),
            vals
        );
        for end in 0..enc.len() {
            assert!(decode_i64_seq(&mut Cursor::new(&enc[..end]), &budget).is_err());
        }
        enc
    }

    fn mode(enc: &[u8]) -> u8 {
        let mut cur = Cursor::new(enc);
        cur.uvar().unwrap();
        cur.u8().unwrap()
    }

    #[test]
    fn states_positions_and_outputs_agree() {
        // The first values of `new Random(0)` and `new Random(42)`.
        assert_eq!(output(seeded(0)), -4962768465676381896);
        assert_eq!(output(seeded(42)), -5025562857975149833);
        let start = seeded(0x1234_5678_9abc);
        let mut state = start;
        for i in 0..2000i64 {
            assert_eq!(state_of(output(state)), Some(state));
            assert_eq!(distance(start, state), i);
            assert_eq!(jump(start, i), state);
            assert_eq!(jump(state, -i), start);
            assert_eq!(distance(state, start), -i);
            state = jump(state, 1);
        }
        for steps in [
            i64::MIN,
            -(1 << 47),
            -1,
            0,
            (1 << 47) - 1,
            1 << 47,
            i64::MAX,
        ] {
            let there = jump(start, steps);
            assert_eq!(jump(there, steps.wrapping_neg()), start);
            assert_eq!(distance(start, there), (steps << 16) >> 16);
        }
        // What makes `state_of` unambiguous: two states with one output would differ by a
        // multiple of `MULT` within 2^16 of a multiple of 2^48.
        for m in 1..=6u64 {
            let r = (m << 48) % MULT;
            assert!(r.min(MULT - r) >= 1 << 16);
        }
        // A state with zero low bits is the multiple of `MULT` in the range that starts below zero.
        for high in [0, 1, 0xdead_beef, 0xffff_ffffu64] {
            assert_eq!(state_of(output(high << 16)), Some(high << 16));
        }
        let golden = 0x9e37_79b9_7f4a_7c15u64 as i64;
        let on_cycle = (0..1i64 << 18).filter(|v| state_of(v.wrapping_mul(golden)).is_some());
        assert!(on_cycle.count() < 32);
    }

    #[test]
    fn interleaved_ids_cost_their_order() {
        // Ids referenced a dozen times each while 64 are live, first seen slightly out of order.
        let ids = outputs(seeded(7), 2000);
        let vals: Vec<i64> = (0..24_000usize)
            .map(|i| ids[(i / 12 + (i * 37) % 64).min(1999)])
            .collect();
        let enc = roundtrip(&vals);
        assert_eq!(mode(&enc), 8);
        let mut plain = Vec::new();
        encode_i64_seq_with(&vals, Z, 3, ALLOW_RECENCY, &mut plain);
        assert!(
            enc.len() + 2000 * 6 < plain.len(),
            "{} {}",
            enc.len(),
            plain.len()
        );
    }

    #[test]
    fn distinct_far_and_foreign_values() {
        // Each value once, some from 2^40 steps away and from another instance, three of them
        // not outputs at all.
        let mut vals = outputs(seeded(1), 300);
        vals.extend(outputs(jump(seeded(1), -(1 << 40)), 20));
        vals.extend(outputs(seeded(0xdead_beef), 20));
        vals.reverse();
        for (row, v) in [(5, i64::MIN), (77, 1), (339, i64::MAX)] {
            assert_eq!(state_of(v), None);
            vals[row] = v;
        }
        let enc = roundtrip(&vals);
        assert_eq!(mode(&enc), 7);
        assert!(enc.len() < 600, "{}", enc.len());
        // Unrelated instances still save the 16 bits a state lacks.
        let lone: Vec<i64> = (0..256).map(|i| output(seeded(i * 0x9e37_79b9))).collect();
        let enc = roundtrip(&lone);
        assert_eq!(mode(&enc), 7);
        assert!(enc.len() < 256 * 6 + 64, "{}", enc.len());
    }

    #[test]
    fn other_integers_are_left_alone() {
        let mut vals = outputs(seeded(3), 8);
        vals.extend((0..120i64).map(|i| i.wrapping_mul(0x0123_4567_89ab_cdef)));
        assert!(encode(&vals, Z, 3).is_none());
        assert!(encode(&[1, 2, 3], Z, 3).is_none());
        assert!(encode(&[], Z, 3).is_none());
    }

    #[test]
    fn rejects_malformed_payloads() {
        let budget = DecodeBudget::default();
        let vals = outputs(seeded(9), 4);
        let build = |rows: &[u32], others: &[i64], steps: &[i64]| {
            let mut b = vec![0u8; 6];
            encode_u32_seq(rows, Z, 3, &mut b);
            if !rows.is_empty() {
                encode_i64_seq_with(others, Z, 3, 0, &mut b);
            }
            encode_i64_seq_with(steps, Z, 3, 0, &mut b);
            b
        };
        let ok = build(&[1], &[5], &[0, 2, 4]);
        assert_eq!(decode(&mut Cursor::new(&ok), 4, &budget, 0).unwrap()[1], 5);
        for bad in [
            build(&[], &[], &[0, 2, 4]),
            build(&[], &[], &[0, 2, 4, 6, 8]),
            build(&[1], &[5, 6], &[0, 2, 4]),
            build(&[4], &[5], &[0, 2, 4]),
            build(&[2, 1], &[5, 6], &[0, 2]),
            build(&[1, 1], &[5, 6], &[0, 2]),
        ] {
            assert!(decode(&mut Cursor::new(&bad), 4, &budget, 0).is_err());
        }
        // A sequence nested in two others holds no further ones.
        let mut nested = Vec::new();
        encode_i64_seq(&vals, Z, 3, &mut nested);
        assert_eq!(mode(&nested), 7);
        let mut cur = Cursor::new(&nested);
        assert_eq!(decode_i64_seq_at(&mut cur, &budget, 1).unwrap(), vals);
        assert!(decode_i64_seq_at(&mut Cursor::new(&nested), &budget, 2).is_err());
        let tight = DecodeBudget {
            max_single_alloc_bytes: 16,
            ..DecodeBudget::default()
        };
        assert!(matches!(
            decode_i64_seq(&mut Cursor::new(&nested), &tight),
            Err(Error::BudgetExceeded(_))
        ));
    }
}
