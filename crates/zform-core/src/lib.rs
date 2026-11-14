//! The implementation behind the `zform` crate: `ZFM1` streams and `ZFP1` profiles, the model
//! trainer, column codecs, and the block, file and query layers. Depend on `zform` instead; the
//! items here change without notice.

pub mod block;
pub mod cells;
pub mod codecs;
pub mod container;
pub mod dictionary;
pub mod file;
pub mod framing;
pub mod grammar;
pub mod index;
pub mod inspect;
pub mod json_tree;
pub mod lex;
pub mod packs;
pub mod payload;
pub mod plan;
pub mod query;
pub mod wire;

pub use container::{
    FrameFit, InspectReport, PackOptions, StreamEncoder, build_dictionary, inspect_stream,
    pack_stream, unpack_stream,
};
pub use dictionary::{Dictionary, DictionaryBuilder};
pub use framing::{FramingSpec, ResolvedFraming};
pub use grammar::ModelPlan;
pub use packs::{
    CodecPack, ColumnProfile, HdfsRelations, PackRegistry, RelationContext, RelationPack,
};
pub use payload::graph::{CoreDecoder, PayloadGraph};
pub use plan::{Profile, TrainOptions, train_from_bytes};
pub use query::{Predicate, eval_raw_record, logical, raw_field_values};
pub use wire::{DecodeBudget, EntropyBackend, Error, Result};
