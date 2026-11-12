//! One block of records as a value: the API for embedding the codec in another container.
//!
//! A system with its own storage layout keeps a model once and each block's bytes and row count
//! wherever it likes, and hands them back to read. Nothing here does IO or spawns threads; the only
//! state kept between calls is a per-thread zstd context. The `ZFM1` stream is one such container.
//! A block read this way is read whole; `file` reads parts of blocks through an index.

use crate::container::{decode_single_block_records, encode_block_fit, learn_library};
use crate::framing::ResolvedFraming;
use crate::grammar::ModelPlan;
use crate::packs::PackRegistry;
use crate::plan::{TrainOptions, train_from_records};
use crate::wire::{DecodeBudget, EntropyBackend, Result};

#[derive(Debug, Clone)]
pub struct BlockOptions {
    pub backend: EntropyBackend,
    pub level: i32,
    pub packs: PackRegistry,
}

impl Default for BlockOptions {
    fn default() -> Self {
        Self {
            backend: EntropyBackend::Zstd,
            level: 3,
            packs: PackRegistry::default(),
        }
    }
}

/// How much of a block its model could not place: the signal for retraining.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fit {
    pub rows: usize,
    pub residual_rows: usize,
    pub bytes: usize,
    pub residual_bytes: usize,
}

impl Fit {
    /// The larger of the residual row share and the residual byte share; 0 for an empty block.
    pub fn residual_rate(&self) -> f64 {
        let share = |part: usize, all: usize| {
            if all == 0 {
                0.0
            } else {
                part as f64 / all as f64
            }
        };
        share(self.residual_rows, self.rows).max(share(self.residual_bytes, self.bytes))
    }
}

/// Learns a model from sample records, keeping what `prior` already covers. The model carries
/// the decode programs the samples' columns use, so each block names them instead of storing them.
pub fn train(samples: &[&[u8]], opts: &TrainOptions, prior: Option<&ModelPlan>) -> ModelPlan {
    let mut model = train_from_records(
        samples,
        ResolvedFraming::Lf { open_end: false },
        opts,
        prior,
    );
    let block = BlockOptions::default();
    let sample = &samples[..samples.len().min(LIBRARY_SAMPLE_ROWS)];
    model.library = learn_library(sample, &model, &block.packs, block.backend, block.level);
    model
}

/// Rows of the training sample encoded to see which programs its columns use.
const LIBRARY_SAMPLE_ROWS: usize = 4096;

/// Encodes `records` under `model`. A record the model cannot place is stored verbatim, so any
/// model encodes any input; `Fit` reports how much fell through.
pub fn encode_block(records: &[&[u8]], model: &ModelPlan, opts: &BlockOptions) -> (Vec<u8>, Fit) {
    let eb = encode_block_fit(records, model, &opts.packs, opts.backend, opts.level, None);
    let fit = Fit {
        rows: records.len(),
        residual_rows: eb.residual_rows,
        bytes: records.iter().map(|r| r.len()).sum(),
        residual_bytes: eb.residual_bytes,
    };
    (eb.bytes, fit)
}

/// Restores every record of a block, in order.
pub fn decode_block(
    model: &ModelPlan,
    bytes: &[u8],
    n_rows: usize,
    budget: &DecodeBudget,
) -> Result<Vec<Vec<u8>>> {
    decode_single_block_records(bytes, n_rows, model, budget)
}
