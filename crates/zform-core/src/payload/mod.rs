//! Self-describing payload codec engine: portable SSA lane IR (`ir`), hardware SIMD tile layer
//! (`simd_arch`), Direct-Threaded SIMD lane executor (`engine`), lagged recurrence executor
//! (`recur`), and domain-agnostic DAG payload evaluator (`graph`).

pub mod engine;
pub mod graph;
pub mod ir;
pub mod library;
pub mod recur;
pub mod simd_arch;

#[cfg(test)]
mod tests;
