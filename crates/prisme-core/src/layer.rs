use crate::geom::Affine2;
use crate::gradient::{Gradient, GradientGeometry};
use crate::ids::LayerId;
use crate::tile::TileGrid;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Layer blend modes. Blending happens in linear light in the working space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u32)]
pub enum BlendMode {
    #[default]
    Normal = 0,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Add,
    Subtract,
    Divide,
    LinearBurn,
    VividLight,
    LinearLight,
    PinLight,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl BlendMode {
    pub const ALL: [BlendMode; 23] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Add,
        BlendMode::Subtract,
        BlendMode::Divide,
        BlendMode::LinearBurn,
        BlendMode::VividLight,
        BlendMode::LinearLight,
        BlendMode::PinLight,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
            BlendMode::Darken => "Darken",
            BlendMode::Lighten => "Lighten",
            BlendMode::ColorDodge => "Color Dodge",
            BlendMode::ColorBurn => "Color Burn",
            BlendMode::HardLight => "Hard Light",
            BlendMode::SoftLight => "Soft Light",
            BlendMode::Difference => "Difference",
            BlendMode::Exclusion => "Exclusion",
            BlendMode::Add => "Add (Linear Dodge)",
            BlendMode::Subtract => "Subtract",
            BlendMode::Divide => "Divide",
            BlendMode::LinearBurn => "Linear Burn",
            BlendMode::VividLight => "Vivid Light",
            BlendMode::LinearLight => "Linear Light",
            BlendMode::PinLight => "Pin Light",
            BlendMode::Hue => "Hue",
            BlendMode::Saturation => "Saturation",
            BlendMode::Color => "Color",
            BlendMode::Luminosity => "Luminosity",
        }
    }

    pub fn from_index(i: u32) -> BlendMode {
        Self::ALL.get(i as usize).copied().unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LockFlags {
    pub pixels: bool,
    pub position: bool,
    pub alpha: bool,
}

impl LockFlags {
    pub fn any(&self) -> bool {
        self.pixels || self.position || self.alpha
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerMask {
    /// Single-channel coverage (0 = hidden, 1 = visible).
    pub grid: Arc<TileGrid>,
    pub enabled: bool,
    /// When a mask is missing a tile, it reveals (true) or hides (false).
    pub default_visible: bool,
}

/// An SVG kept as vector source until render time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorLayer {
    pub svg: Arc<String>,
    /// Natural size of the SVG in user units.
    pub size: [f32; 2],
    /// Placement: SVG user space -> document pixels.
    pub transform: Affine2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FillContent {
    Solid([f32; 4]),
    Gradient(Gradient, GradientGeometry),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FillLayer {
    pub content: FillContent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TextFill {
    Solid([f32; 4]),
    /// Gradient across the text bounds at `angle` (radians).
    Gradient { gradient: Gradient, angle: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextOutline {
    pub width: f32,
    pub color: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextGlow {
    pub radius: f32,
    pub color: [f32; 4],
    pub strength: f32,
    pub inner: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextShadow {
    pub offset: [f32; 2],
    pub blur: f32,
    pub color: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum TextWarp {
    #[default]
    None,
    /// Bend along an arc; `bend` in [-1, 1].
    Arc { bend: f32 },
    /// Vertical sine wave.
    Wave { amplitude: f32, wavelength: f32 },
    /// Bulge outwards in the middle.
    Bulge { amount: f32 },
    /// Flag-like wave that grows along the line.
    Flag { amplitude: f32, wavelength: f32 },
}

/// 3D extruded text wrapped onto a parametric sphere (flat shading, no lighting).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Text3D {
    pub enabled: bool,
    /// Extrusion depth in text pixels.
    pub depth: f32,
    /// Sphere radius in document pixels.
    pub radius: f32,
    /// Placement of the text centre on the sphere (radians).
    pub longitude: f32,
    pub latitude: f32,
    /// Object rotation (yaw, pitch, roll) in radians.
    pub rotation: [f32; 3],
    /// Perspective strength (0 = orthographic).
    pub perspective: f32,
    pub side_color: [f32; 4],
    /// Optional gradient along the extrusion (front -> back). Overrides `side_color`.
    pub side_gradient: Option<Gradient>,
    pub show_sphere: bool,
    pub sphere_color: [f32; 4],
}

impl Default for Text3D {
    fn default() -> Self {
        Self {
            enabled: false,
            depth: 20.0,
            radius: 300.0,
            longitude: 0.0,
            latitude: 0.0,
            rotation: [0.0, 0.0, 0.0],
            perspective: 0.6,
            side_color: [0.15, 0.15, 0.2, 1.0],
            side_gradient: None,
            show_sphere: false,
            sphere_color: [0.2, 0.25, 0.35, 1.0],
        }
    }
}

/// Parametric text, re-rendered at any resolution; always editable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextLayer {
    pub text: String,
    pub font_family: String,
    pub weight: u16,
    pub italic: bool,
    /// Font size in document pixels.
    pub size: f32,
    pub line_height: f32,
    pub letter_spacing: f32,
    pub align: TextAlign,
    /// Top-left of the text block in document space.
    pub position: [f32; 2],
    pub rotation: f32,
    pub fill: TextFill,
    pub outline: Option<TextOutline>,
    pub glow: Option<TextGlow>,
    pub shadow: Option<TextShadow>,
    pub warp: TextWarp,
    pub three_d: Text3D,
}

impl Default for TextLayer {
    fn default() -> Self {
        Self {
            text: "Text".into(),
            font_family: "sans-serif".into(),
            weight: 400,
            italic: false,
            size: 64.0,
            line_height: 1.2,
            letter_spacing: 0.0,
            align: TextAlign::Left,
            position: [0.0, 0.0],
            rotation: 0.0,
            fill: TextFill::Solid([0.0, 0.0, 0.0, 1.0]),
            outline: None,
            glow: None,
            shadow: None,
            warp: TextWarp::None,
            three_d: Text3D::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupLayer {
    pub children: Vec<Layer>,
    pub expanded: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LayerKind {
    Raster(Arc<TileGrid>),
    Vector(VectorLayer),
    Text(TextLayer),
    Fill(FillLayer),
    Group(GroupLayer),
}

impl LayerKind {
    pub fn type_name(&self) -> &'static str {
        match self {
            LayerKind::Raster(_) => "Raster",
            LayerKind::Vector(_) => "Vector",
            LayerKind::Text(_) => "Text",
            LayerKind::Fill(_) => "Fill",
            LayerKind::Group(_) => "Group",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub lock: LockFlags,
    pub mask: Option<LayerMask>,
    pub kind: LayerKind,
}

impl Layer {
    pub fn new(id: LayerId, name: impl Into<String>, kind: LayerKind) -> Self {
        Self {
            id,
            name: name.into(),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            lock: LockFlags::default(),
            mask: None,
            kind,
        }
    }

    pub fn raster(&self) -> Option<&Arc<TileGrid>> {
        match &self.kind {
            LayerKind::Raster(g) => Some(g),
            _ => None,
        }
    }

    pub fn raster_mut(&mut self) -> Option<&mut TileGrid> {
        match &mut self.kind {
            LayerKind::Raster(g) => Some(Arc::make_mut(g)),
            _ => None,
        }
    }

    pub fn is_group(&self) -> bool {
        matches!(self.kind, LayerKind::Group(_))
    }

    pub fn children(&self) -> &[Layer] {
        match &self.kind {
            LayerKind::Group(g) => &g.children,
            _ => &[],
        }
    }

    pub fn children_mut(&mut self) -> Option<&mut Vec<Layer>> {
        match &mut self.kind {
            LayerKind::Group(g) => Some(&mut g.children),
            _ => None,
        }
    }
}
