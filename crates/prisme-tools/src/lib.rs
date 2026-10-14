//! Editing tools as input-driven state machines, plus the pixel algorithms they use.
//!
//! Tools never talk to the UI toolkit: they receive [`ToolInput`] (document-space
//! positions, pressure, tilt, modifiers — tablet ready), mutate a *working copy* of the
//! document, describe their on-canvas feedback as [`Overlay`] shapes and report a
//! [`ToolOutcome`] telling the host to preview, commit to history, cancel, etc. Dab tools
//! can instead hand their strokes to a GPU host ([`stroke`]).

pub mod flatten;
pub mod input;
pub mod liquify;
pub mod options;
pub mod paint;
pub mod region;
pub mod stroke;
pub mod tools;
pub mod transform;

pub use liquify::{LiquifyMode, LiquifyOptions, LiquifySpec};
pub use input::{Modifiers, Phase, PointerKind, ToolInput, ToolKey};
pub use options::ToolOptions;
pub use stroke::{PaintTarget, StrokeOp, StrokeSpec};
pub use tools::{make_tool, Overlay, Tool, ToolCursor, ToolCx, ToolKind, ToolOutcome};

/// Parallel-for over a mutable slice on native, sequential on wasm.
pub(crate) fn par_for_each<T: Send>(items: &mut [T], f: impl Fn(&mut T) + Sync + Send) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        items.par_iter_mut().for_each(f);
    }
    #[cfg(target_arch = "wasm32")]
    items.iter_mut().for_each(f);
}
