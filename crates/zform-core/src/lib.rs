//! The implementation behind the `zform` crate: `ZFM1` streams and `ZFP1` profiles, the model
//! trainer, column codecs, and the block, file and query layers. Depend on `zform` instead; the
//! items here change without notice.

pub mod block;
pub mod cells;
pub mod codecs;
pub mod container;
pub mod file;
pub mod framing;
pub mod grammar;
pub mod index;
pub mod json_tree;
pub mod lex;
pub mod plan;
pub mod query;
pub mod wire;

pub use container::{
    FrameFit, InspectReport, PackOptions, StreamEncoder, inspect_stream, pack_stream, unpack_stream,
};
pub use framing::{FramingSpec, ResolvedFraming};
pub use grammar::ModelPlan;
pub use plan::{Profile, TrainOptions, train_from_bytes};
pub use query::{Predicate, eval_raw_record, logical, raw_field_values};
pub use wire::{DecodeBudget, EntropyBackend, Error, Result};
