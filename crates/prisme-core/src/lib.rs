//! Prisme document model.
//!
//! This crate is deliberately free of GPU, UI and IO dependencies. Every change to a
//! [`Document`] goes through value-semantics editing (clone, mutate, commit) and the
//! [`History`] keeps structurally shared snapshots, which makes undo/redo cheap.
//!
//! Pixel convention: all raster tiles store **straight-alpha, scene-linear** values in
//! the document working space (linear Rec.2020 by default).

pub mod document;
pub mod geom;
pub mod gradient;
pub mod graph;
pub mod history;
pub mod ids;
pub mod layer;
pub mod mask;
pub mod tile;

pub use document::{Document, DocumentInfo};
pub use geom::IRect;
pub use gradient::{Gradient, GradientInterp, GradientKind, GradientStop};
pub use graph::{Graph, Link, Node, ParamValue};
pub use history::History;
pub use ids::{LayerId, MaskId, NodeId};
pub use layer::{BlendMode, FillLayer, Layer, LayerKind, LockFlags, TextLayer, VectorLayer};
pub use mask::{Mask, SavedMask, SelectionOp};
pub use tile::{PixelFormat, Tile, TileCoord, TileGrid, TILE_SIZE};
