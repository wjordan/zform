//! Self-describing column payloads and their decoder.
//!
//! Writer-side codec packs emit [`graph::PayloadGraph`] nodes producing byte buffers, integer
//! lanes, and byte-string cells. [`graph::CoreDecoder`] executes those generic operations without
//! the pack's domain-specific code. Graph headers declare shared block data dependencies;
//! [`library`] supplies programs, recurrences, and byte tables shared by a model's blocks.
//!
//! [`ir`] defines portable lane programs, [`engine`] executes them, and [`recur`] evaluates
//! lagged recurrences.

mod chain;
pub mod engine;
pub mod graph;
pub mod ir;
pub mod library;
pub mod recur;
mod rice;

#[cfg(test)]
mod tests;
