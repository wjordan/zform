//! Compatibility decoding for payload operations no longer emitted by built-in packs.

use crate::cells::Cells;
use crate::wire::{DecodeBudget, Error, Result};
use super::graph::{FastMap, parse_cell_ascii_i64};

/// State required by archives written before ordered keyed sums.
#[derive(Default)]
struct Ranked<'a> {
    ids: FastMap<&'a [u8], u32>,
    keys: Vec<(Option<i64>, bool)>,
    last: i64,
}

impl<'a> Ranked<'a> {
    /// A group's candidates, in pick order, as `left`.
    fn open(&mut self, cells: impl Iterator<Item = &'a [u8]>, left: &mut Vec<u32>) {
        left.clear();
        for v in cells.filter(|v| !v.is_empty()) {
            let next = self.keys.len() as u32;
            let k = *self.ids.entry(v).or_insert(next);
            if k == next {
                self.keys.push((None, false));
            }
            if !left.contains(&k) {
                left.push(k);
            }
        }
        // Stable: candidates alike stay in cell order.
        left.sort_by_key(|&k| !self.keys[k as usize].1);
    }

    fn sum(&self, k: u32) -> i64 {
        self.keys[k as usize].0.unwrap_or(self.last)
    }

    fn step(&mut self, k: u32, delta: i64) -> i64 {
        self.last = self.sum(k).wrapping_add(delta);
        self.keys[k as usize].0 = Some(self.last);
        self.last
    }

    /// Ends a group of candidates `all`, `left` of them untaken.
    fn close(&mut self, all: &[u32], left: &[u32]) {
        for &k in all {
            self.keys[k as usize].1 = !left.contains(&k);
        }
    }
}

pub(super) fn ranked_sum(
    at: &[i64],
    picks: &[i64],
    deltas: &[i64],
    order: &Cells,
    cands: &Cells,
    width: usize,
    budget: &DecodeBudget,
) -> Result<Vec<i64>> {
    if at.len() != picks.len() || at.len() != deltas.len() {
        return Err(Error::Format("RankedSum length mismatch".into()));
    }
    budget.check_alloc(at.len().saturating_mul(40), "RankedSum")?;
    let mut rows = Vec::new();
    for (i, (&a, &p)) in at.iter().zip(picks).enumerate() {
        if p != 0 {
            let cell = usize::try_from(a).ok().and_then(|a| order.get(a));
            let cell = cell.ok_or_else(|| Error::Format("RankedSum row out of range".into()))?;
            rows.push((parse_cell_ascii_i64(cell), i));
        }
    }
    rows.sort_unstable();
    let mut out = deltas.to_vec();
    let mut state = Ranked::default();
    let (mut left, mut all) = (Vec::new(), Vec::new());
    let mut j = 0;
    while j < rows.len() {
        let end = j + rows[j..].iter().take_while(|r| r.0 == rows[j].0).count();
        let from = (at[rows[j].1] as usize).saturating_mul(width);
        state.open((from..from.saturating_add(width)).filter_map(|c| cands.get(c)), &mut left);
        all.clone_from(&left);
        for &(_, i) in &rows[j..end] {
            let p = usize::try_from(picks[i] - 1).ok().filter(|&p| p < left.len());
            let p = p.ok_or_else(|| Error::Format("RankedSum pick out of range".into()))?;
            out[i] = state.step(left.remove(p), deltas[i]);
        }
        state.close(&all, &left);
        j = end;
    }
    Ok(out)
}
