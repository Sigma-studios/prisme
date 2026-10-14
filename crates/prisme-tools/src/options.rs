use prisme_brush::{presets, BrushSettings};
use prisme_core::gradient::{GradientGeometry, GradientKind, GradientRepeat};
use prisme_core::{Gradient, SelectionOp};
use serde::{Deserialize, Serialize};

/// Settings for all tools (edited in the options bar, persisted between sessions).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolOptions {
    // selection
    pub sel_op: SelectionOp,
    pub sel_feather: f32,
    pub sel_antialias: bool,
    /// Fixed aspect / size for marquee (0 = free).
    pub sel_fixed_ratio: f32,
    // colour/blob select & fill
    pub tolerance: f32,
    pub sample_all_layers: bool,
    pub fill_antialias: bool,
    pub fill_opacity: f32,
    // brushes (each tool keeps its own brush)
    pub brush: BrushSettings,
    pub eraser: BrushSettings,
    pub clone_brush: BrushSettings,
    pub retouch_brush: BrushSettings,
    pub brush_preset: String,
    /// Smudge / blur strength.
    pub retouch_strength: f32,
    /// Clone: keep source offset between strokes.
    pub clone_aligned: bool,
    /// Paint onto the active layer's mask instead of its pixels.
    pub edit_mask: bool,
    /// Paint onto this saved mask (Masks panel "Edit"); takes precedence over layers.
    #[serde(skip)]
    pub edit_saved_mask: Option<prisme_core::MaskId>,
    // gradient
    pub gradient: Gradient,
    pub gradient_kind: GradientKind,
    pub gradient_repeat: GradientRepeat,
    pub gradient_reverse: bool,
    pub gradient_dither: bool,
    pub gradient_as_layer: bool,
    pub gradient_opacity: f32,
    pub gradient_use_fg_bg: bool,
    // crop
    pub crop_delete_outside: bool,
    // picker
    pub picker_sample_size: u32,
    // move
    pub move_auto_select: bool,
}

impl Default for ToolOptions {
    fn default() -> Self {
        let p = presets();
        let find = |n: &str| p.iter().find(|b| b.name == n).map(|b| b.settings.clone()).unwrap_or_default();
        let _ = GradientGeometry::default();
        Self {
            sel_op: SelectionOp::Replace,
            sel_feather: 0.0,
            sel_antialias: true,
            sel_fixed_ratio: 0.0,
            tolerance: 0.12,
            sample_all_layers: false,
            fill_antialias: true,
            fill_opacity: 1.0,
            brush: find("Round Hard"),
            eraser: find("Round Soft"),
            clone_brush: BrushSettings { hardness: 0.5, size: 40.0, ..find("Round Soft") },
            retouch_brush: BrushSettings { hardness: 0.3, size: 40.0, spacing: 0.15, ..find("Round Soft") },
            brush_preset: "Round Hard".into(),
            retouch_strength: 0.5,
            clone_aligned: true,
            edit_mask: false,
            edit_saved_mask: None,
            gradient: Gradient::default(),
            gradient_kind: GradientKind::Linear,
            gradient_repeat: GradientRepeat::Pad,
            gradient_reverse: false,
            gradient_dither: true,
            gradient_as_layer: false,
            gradient_opacity: 1.0,
            gradient_use_fg_bg: true,
            crop_delete_outside: false,
            picker_sample_size: 1,
            move_auto_select: false,
        }
    }
}
