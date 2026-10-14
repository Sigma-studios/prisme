//! Dab-based brush engine.
//!
//! A [`Stroke`] turns raw input samples (position, pressure, tilt, time — mouse input
//! simply reports pressure 1.0) into evenly spaced [`Dab`]s using the brush's spacing
//! and dynamics. Dabs accumulate into a per-stroke coverage buffer which is then
//! composited onto a snapshot of the layer taken at stroke start; this gives correct
//! "opacity caps the stroke, flow builds up" behaviour and lets erase/paint share code.
//!
//! Tablet support only needs a new event source: dynamics already map from abstract
//! inputs (pressure, tilt, velocity, direction, random).

pub mod dab;
pub mod presets;
pub mod stroke;

pub use dab::{Dab, TipShape};
pub use presets::{presets, BrushPreset};
pub use stroke::{BlendTarget, Stroke, StrokeCanvas};

use serde::{Deserialize, Serialize};

/// Input that can drive a dynamic parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum InputSource {
    #[default]
    None,
    Pressure,
    Velocity,
    Tilt,
    Random,
    /// Stroke direction (only meaningful for angle).
    Direction,
}

impl InputSource {
    pub const ALL: [InputSource; 6] =
        [InputSource::None, InputSource::Pressure, InputSource::Velocity, InputSource::Tilt, InputSource::Random, InputSource::Direction];

    pub fn name(self) -> &'static str {
        match self {
            InputSource::None => "Off",
            InputSource::Pressure => "Pressure",
            InputSource::Velocity => "Velocity",
            InputSource::Tilt => "Tilt",
            InputSource::Random => "Random",
            InputSource::Direction => "Direction",
        }
    }
}

/// Maps an input in [0,1] to a multiplier in [min, 1] with a gamma curve.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dynamic {
    pub source: InputSource,
    pub min: f32,
    pub gamma: f32,
}

impl Default for Dynamic {
    fn default() -> Self {
        Self { source: InputSource::None, min: 0.0, gamma: 1.0 }
    }
}

impl Dynamic {
    pub fn pressure(min: f32) -> Self {
        Self { source: InputSource::Pressure, min, gamma: 1.0 }
    }

    pub fn eval(&self, input: f32) -> f32 {
        if self.source == InputSource::None {
            return 1.0;
        }
        self.min + (1.0 - self.min) * input.clamp(0.0, 1.0).powf(self.gamma.max(0.05))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BrushMode {
    #[default]
    Paint,
    Erase,
}

/// Full brush description. Sizes are in document pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushSettings {
    pub tip: TipShape,
    /// Diameter in document pixels.
    pub size: f32,
    /// 0 = very soft falloff, 1 = hard edge.
    pub hardness: f32,
    /// Distance between dabs as a fraction of the diameter.
    pub spacing: f32,
    /// Per-dab opacity (builds up within a stroke).
    pub flow: f32,
    /// Maximum opacity of the whole stroke.
    pub opacity: f32,
    /// Tip rotation in radians.
    pub angle: f32,
    /// Tip aspect ratio (1 = circle).
    pub roundness: f32,
    /// Aliased, pixel-grid snapped dabs (pixel art).
    pub pixel_perfect: bool,
    pub size_dynamic: Dynamic,
    pub opacity_dynamic: Dynamic,
    pub flow_dynamic: Dynamic,
    pub angle_source: InputSource,
    /// Random variation amounts in [0,1].
    pub jitter_size: f32,
    pub jitter_angle: f32,
    pub jitter_opacity: f32,
    /// Random offset perpendicular/along the stroke as a fraction of the diameter.
    pub scatter: f32,
    /// Paper-like grain strength in [0,1] (chalk/charcoal).
    pub grain: f32,
    /// Position smoothing in [0,1] (lazy-mouse style).
    pub smoothing: f32,
    /// Airbrush: keep emitting dabs while the pointer is held still.
    pub airbrush: bool,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            tip: TipShape::Round,
            size: 20.0,
            hardness: 0.8,
            spacing: 0.1,
            flow: 1.0,
            opacity: 1.0,
            angle: 0.0,
            roundness: 1.0,
            pixel_perfect: false,
            size_dynamic: Dynamic::pressure(0.2),
            opacity_dynamic: Dynamic::default(),
            flow_dynamic: Dynamic::default(),
            angle_source: InputSource::None,
            jitter_size: 0.0,
            jitter_angle: 0.0,
            jitter_opacity: 0.0,
            scatter: 0.0,
            grain: 0.0,
            smoothing: 0.0,
            airbrush: false,
        }
    }
}

/// One raw input sample from the pointer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputSample {
    pub pos: [f32; 2],
    /// 0..1; mouse reports 1.0.
    pub pressure: f32,
    /// Tilt magnitude 0..1 and direction (radians); zero for mice.
    pub tilt: f32,
    pub tilt_dir: f32,
    /// Seconds.
    pub time: f64,
}

impl InputSample {
    pub fn mouse(pos: [f32; 2], time: f64) -> Self {
        Self { pos, pressure: 1.0, tilt: 0.0, tilt_dir: 0.0, time }
    }
}
