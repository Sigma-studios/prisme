//! Colour science for Prisme.
//!
//! The document working space is **scene-linear Rec.2020 (D65)**. This crate provides
//! analytic conversions for well known spaces (exact f32 math, used for untagged and
//! CICP-tagged images and for UI colour pickers), ICC transforms through `moxcms` for
//! embedded profiles, OKLab/OKLCh for perceptual operations, and display LUT baking.

pub mod hdr;
pub mod icc;
pub mod lut;
pub mod math;
pub mod spaces;

pub use icc::{IccProfile, IccTransform};
pub use math::Mat3;
pub use spaces::*;
