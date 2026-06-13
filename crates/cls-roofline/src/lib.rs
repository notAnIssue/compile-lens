//! `cls-roofline` — the kernel roofline cost model for Tool 5 (`kernel-roofline`).
//!
//! The model has three layers: a theoretical lower bound (Williams roofline), an empirical
//! predictor, and a calibrated pruning decision. This crate currently implements **Layer 1** and
//! the **GPU spec registry** it draws on; Layers 2 and 3 land in later changes.
//!
//! Entry point: [`RooflineCostModel::for_gpu`] — bind the model to a registered GPU by name, then
//! analyze a kernel's features into a `cls_schema::RooflinePrediction`.

pub mod gpu;
mod predictor;
mod pruning;
mod roofline;

pub use gpu::{known_names, lookup, GpuSpec};
pub use predictor::{
    block_size_penalty, occupancy, occupancy_penalty, register_pressure_penalty_us,
    EmpiricalPrediction, LAUNCH_OVERHEAD_US,
};
pub use pruning::{prune_mode, pruning_decision, spearman_correlation, PruneMode};
pub use roofline::{BoundType, RooflineCostModel};
