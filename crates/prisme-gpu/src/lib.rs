//! GPU engine: tiled layer uploads, node-graph evaluation with compute kernels,
//! display transform and readbacks. Works identically on native (Metal/Vulkan/DX12)
//! and the web (WebGPU): every kernel writes storage textures and samples manually.

pub mod context;
pub mod display;
pub mod engine;
pub mod image;
pub mod kernels;
pub mod levels;
pub mod readback;
pub mod scopes;
pub mod upload;

pub use context::GpuContext;
pub use display::{DisplayOutput, DisplayParams, DisplayRenderer, HighlightMode};
pub use engine::{Engine, EvalFormat, EvalRequest, LayerRasterizer, RenderSource};
pub use image::GpuImage;
pub use readback::Readback;

pub use wgpu;
