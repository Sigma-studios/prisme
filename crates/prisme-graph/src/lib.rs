//! Node kind registry: sockets, parameter schemas (which drive UI generation and
//! serialization defaults) and region-of-interest padding. GPU kernels are registered
//! separately in `prisme-gpu`, keyed by the same kind id.
//!
//! Adding a node = adding a [`NodeKindDesc`] here and a kernel in `prisme-gpu`.

use prisme_core::graph::{Node, ParamValue, KIND_LAYER_STACK, KIND_OUTPUT};
use prisme_core::Gradient;
use std::collections::BTreeMap;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocketType {
    Image,
    Mask,
}

#[derive(Clone, Debug)]
pub struct SocketDesc {
    pub name: &'static str,
    pub ty: SocketType,
    pub optional: bool,
}

const fn img(name: &'static str) -> SocketDesc {
    SocketDesc { name, ty: SocketType::Image, optional: false }
}
const fn mask_opt(name: &'static str) -> SocketDesc {
    SocketDesc { name, ty: SocketType::Mask, optional: true }
}
const fn mask(name: &'static str) -> SocketDesc {
    SocketDesc { name, ty: SocketType::Mask, optional: false }
}

#[derive(Clone, Debug)]
pub enum ParamKind {
    Float { min: f32, max: f32, default: f32, logarithmic: bool },
    Int { min: i32, max: i32, default: i32 },
    Bool(bool),
    Color([f32; 4]),
    Choice { options: &'static [&'static str], default: u32 },
    Vec2([f32; 2]),
    /// A colour-wheel offset (OKLab a/b offset + lightness), edited with a wheel widget.
    Wheel,
    Curve,
    Gradient,
    Layer,
    LayerSet,
    Mask,
    Text,
    /// Placeholder parameter rendered by a custom widget that owns several keys.
    Custom(&'static str),
}

#[derive(Clone, Debug)]
pub struct ParamDesc {
    pub key: &'static str,
    pub label: &'static str,
    pub kind: ParamKind,
}

const fn p(key: &'static str, label: &'static str, kind: ParamKind) -> ParamDesc {
    ParamDesc { key, label, kind }
}
const fn float(key: &'static str, label: &'static str, min: f32, max: f32, default: f32) -> ParamDesc {
    p(key, label, ParamKind::Float { min, max, default, logarithmic: false })
}
const fn float_log(key: &'static str, label: &'static str, min: f32, max: f32, default: f32) -> ParamDesc {
    p(key, label, ParamKind::Float { min, max, default, logarithmic: true })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    Input,
    Tonal,
    Color,
    Filter,
    Composite,
    Utility,
}

impl Category {
    pub const ALL: [Category; 6] =
        [Category::Input, Category::Tonal, Category::Color, Category::Filter, Category::Composite, Category::Utility];

    pub fn name(self) -> &'static str {
        match self {
            Category::Input => "Inputs",
            Category::Tonal => "Tone",
            Category::Color => "Color",
            Category::Filter => "Filters",
            Category::Composite => "Compositing",
            Category::Utility => "Utility",
        }
    }
}

pub struct NodeKindDesc {
    pub id: &'static str,
    pub name: &'static str,
    pub category: Category,
    pub description: &'static str,
    pub inputs: Vec<SocketDesc>,
    pub outputs: Vec<SocketDesc>,
    pub params: Vec<ParamDesc>,
    /// Pixels of input context needed around the output region, at full resolution.
    pub pad: fn(&Node) -> f32,
    /// Pointwise nodes can be evaluated on any sub-rect independently.
    pub pointwise: bool,
}

fn no_pad(_: &Node) -> f32 {
    0.0
}

pub const LEVEL_CHANNELS: [&str; 4] = ["m", "r", "g", "b"];
pub const CURVE_CHANNELS: [&str; 5] = ["m", "r", "g", "b", "l"];

pub struct Registry {
    pub kinds: Vec<NodeKindDesc>,
}

impl Registry {
    pub fn get(&self, id: &str) -> Option<&NodeKindDesc> {
        self.kinds.iter().find(|k| k.id == id)
    }

    pub fn by_category(&self, c: Category) -> impl Iterator<Item = &NodeKindDesc> {
        self.kinds.iter().filter(move |k| k.category == c && k.id != KIND_OUTPUT)
    }
}

pub fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(build_registry)
}

/// Parameters shown directly on the node in the graph editor: up to three simple
/// (scalar / choice / colour / layer / mask) parameters, in schema order.
pub fn main_params(kind: &str) -> Vec<&'static ParamDesc> {
    let Some(desc) = registry().get(kind) else { return vec![] };
    let limit = match kind {
        "channel_mixer" | "black_white" => 0,
        _ => 3,
    };
    desc.params
        .iter()
        .filter(|p| {
            matches!(
                p.kind,
                ParamKind::Float { .. } | ParamKind::Int { .. } | ParamKind::Bool(_) | ParamKind::Choice { .. } | ParamKind::Color(_) | ParamKind::Layer | ParamKind::Mask
            )
        })
        .take(limit)
        .collect()
}

/// True when the kind has parameters that need the full Properties panel (curves, wheels…).
pub fn has_complex_params(kind: &str) -> bool {
    registry().get(kind).is_some_and(|d| {
        d.params.iter().any(|p| matches!(p.kind, ParamKind::Custom(_) | ParamKind::Curve | ParamKind::Gradient | ParamKind::Wheel | ParamKind::LayerSet | ParamKind::Text))
            || d.params.len() > main_params(kind).len()
    })
}

/// Default parameter values for a kind.
pub fn default_params(kind: &str) -> BTreeMap<String, ParamValue> {
    let mut m = BTreeMap::new();
    let Some(desc) = registry().get(kind) else { return m };
    for pd in &desc.params {
        let v = match &pd.kind {
            ParamKind::Float { default, .. } => ParamValue::Float(*default),
            ParamKind::Int { default, .. } => ParamValue::Int(*default),
            ParamKind::Bool(b) => ParamValue::Bool(*b),
            ParamKind::Color(c) => ParamValue::Color(*c),
            ParamKind::Choice { default, .. } => ParamValue::Choice(*default),
            ParamKind::Vec2(v) => ParamValue::Vec2(*v),
            ParamKind::Wheel => ParamValue::Vec3([0.0, 0.0, 0.0]),
            ParamKind::Curve => ParamValue::Curve(vec![[0.0, 0.0], [1.0, 1.0]]),
            ParamKind::Gradient => ParamValue::Gradient(Gradient::default()),
            ParamKind::Layer => ParamValue::Layer(None),
            ParamKind::LayerSet => ParamValue::LayerSet(vec![]),
            ParamKind::Mask => ParamValue::Mask(None),
            ParamKind::Text => ParamValue::Text(String::new()),
            ParamKind::Custom("levels") => {
                for ch in LEVEL_CHANNELS {
                    m.insert(format!("{ch}.in_black"), ParamValue::Float(0.0));
                    m.insert(format!("{ch}.in_white"), ParamValue::Float(1.0));
                    m.insert(format!("{ch}.gamma"), ParamValue::Float(1.0));
                    m.insert(format!("{ch}.out_black"), ParamValue::Float(0.0));
                    m.insert(format!("{ch}.out_white"), ParamValue::Float(1.0));
                }
                continue;
            }
            ParamKind::Custom("curves") => {
                for ch in CURVE_CHANNELS {
                    m.insert(format!("curve.{ch}"), ParamValue::Curve(vec![[0.0, 0.0], [1.0, 1.0]]));
                }
                continue;
            }
            ParamKind::Custom(_) => continue,
        };
        m.insert(pd.key.to_string(), v);
    }
    m
}

fn build_registry() -> Registry {
    use Category::*;
    let k = |id, name, category, description, inputs: Vec<SocketDesc>, params: Vec<ParamDesc>| NodeKindDesc {
        id,
        name,
        category,
        description,
        inputs,
        outputs: vec![img("Image")],
        params,
        pad: no_pad,
        pointwise: true,
    };
    let adj = |id, name, category, description, params| {
        k(id, name, category, description, vec![img("Image"), mask_opt("Mask")], params)
    };
    let mut kinds = vec![
        // ---------------------------------------------------------------- inputs
        NodeKindDesc {
            outputs: vec![],
            ..k(KIND_OUTPUT, "Output", Utility, "Final document image.", vec![img("Image")], vec![])
        },
        k(
            KIND_LAYER_STACK,
            "Layer Stack",
            Input,
            "Composite of the layer stack. Excluded layers are skipped.",
            vec![],
            vec![p("exclude", "Exclude", ParamKind::LayerSet)],
        ),
        k(
            "layer",
            "Layer",
            Input,
            "A single layer's pixels (with its own mask and opacity).",
            vec![],
            vec![
                p("layer", "Layer", ParamKind::Layer),
                p("apply_mask", "Apply layer mask", ParamKind::Bool(true)),
                p("apply_opacity", "Apply opacity", ParamKind::Bool(false)),
            ],
        ),
        NodeKindDesc {
            outputs: vec![mask("Mask")],
            ..k("mask", "Saved Mask", Input, "A saved selection mask.", vec![], vec![p("mask", "Mask", ParamKind::Mask)])
        },
        NodeKindDesc {
            outputs: vec![mask("Mask")],
            ..k("selection", "Selection", Input, "The live selection as a mask.", vec![], vec![])
        },
        k(
            "color",
            "Solid Color",
            Input,
            "Constant colour.",
            vec![],
            vec![p("color", "Color", ParamKind::Color([0.5, 0.5, 0.5, 1.0]))],
        ),
        k(
            "gradient",
            "Gradient",
            Input,
            "Gradient in document space.",
            vec![],
            vec![
                p("gradient", "Gradient", ParamKind::Gradient),
                p(
                    "shape",
                    "Shape",
                    ParamKind::Choice { options: &["Linear", "Radial", "Angular", "Reflected", "Diamond"], default: 0 },
                ),
                p("start", "Start", ParamKind::Vec2([0.0, 0.0])),
                p("end", "End", ParamKind::Vec2([512.0, 0.0])),
            ],
        ),
        // ---------------------------------------------------------------- tonal
        adj("levels", "Levels", Tonal, "Input/output levels and gamma per channel.", vec![p("levels", "Levels", ParamKind::Custom("levels"))]),
        adj("curves", "Curves", Tonal, "Tone curves for RGB, each channel and OKLab lightness.", vec![p("curves", "Curves", ParamKind::Custom("curves"))]),
        adj(
            "exposure",
            "Exposure",
            Tonal,
            "Exposure in stops, black offset and gamma.",
            vec![float("exposure", "Exposure (EV)", -10.0, 10.0, 0.0), float("offset", "Offset", -0.5, 0.5, 0.0), float("gamma", "Gamma", 0.1, 5.0, 1.0)],
        ),
        adj(
            "brightness_contrast",
            "Brightness / Contrast",
            Tonal,
            "Perceptual brightness and contrast around a pivot.",
            vec![float("brightness", "Brightness", -1.0, 1.0, 0.0), float("contrast", "Contrast", -1.0, 1.0, 0.0), float("pivot", "Pivot", 0.0, 1.0, 0.5)],
        ),
        NodeKindDesc {
            pad: |n| n.f("radius", 30.0) * 3.0,
            pointwise: false,
            ..adj(
                "shadows_highlights",
                "Shadows / Highlights",
                Tonal,
                "Local tone mapping that lifts shadows and recovers highlights.",
                vec![float("shadows", "Shadows", -1.0, 1.0, 0.3), float("highlights", "Highlights", -1.0, 1.0, 0.0), float_log("radius", "Radius", 1.0, 200.0, 30.0), p("edges", "Edges", ParamKind::Choice { options: &["Extend (keep edges)", "Transparent"], default: 0 })],
            )
        },
        // ---------------------------------------------------------------- color
        adj(
            "color_wheels",
            "Color Wheels",
            Color,
            "Lift / gamma / gain three-way colour correction.",
            vec![
                p("lift", "Lift (shadows)", ParamKind::Wheel),
                p("gamma", "Gamma (midtones)", ParamKind::Wheel),
                p("gain", "Gain (highlights)", ParamKind::Wheel),
                p("offset", "Offset", ParamKind::Wheel),
                float("saturation", "Saturation", 0.0, 3.0, 1.0),
            ],
        ),
        adj(
            "hsl",
            "Hue / Saturation",
            Color,
            "Hue rotation, saturation and lightness in OKLCh.",
            vec![
                float("hue", "Hue (°)", -180.0, 180.0, 0.0),
                float("saturation", "Saturation", -1.0, 1.0, 0.0),
                float("lightness", "Lightness", -1.0, 1.0, 0.0),
                p("colorize", "Colorize", ParamKind::Bool(false)),
                p("colorize_color", "Colorize color", ParamKind::Color([0.8, 0.5, 0.3, 1.0])),
            ],
        ),
        adj(
            "vibrance",
            "Vibrance",
            Color,
            "Saturation boost that protects already saturated colours.",
            vec![float("vibrance", "Vibrance", -1.0, 1.0, 0.3), float("saturation", "Saturation", -1.0, 1.0, 0.0)],
        ),
        adj(
            "white_balance",
            "White Balance",
            Color,
            "Temperature and tint (von Kries in LMS).",
            vec![float("temperature", "Temperature", -1.0, 1.0, 0.0), float("tint", "Tint", -1.0, 1.0, 0.0)],
        ),
        adj(
            "color_balance",
            "Color Balance",
            Color,
            "Shift colour in shadows, midtones and highlights.",
            vec![
                p("shadows", "Shadows", ParamKind::Wheel),
                p("midtones", "Midtones", ParamKind::Wheel),
                p("highlights", "Highlights", ParamKind::Wheel),
                p("preserve_luminosity", "Preserve luminosity", ParamKind::Bool(true)),
            ],
        ),
        adj(
            "channel_mixer",
            "Channel Mixer",
            Color,
            "Mix input channels into each output channel.",
            vec![
                float("rr", "Red ← R", -2.0, 2.0, 1.0),
                float("rg", "Red ← G", -2.0, 2.0, 0.0),
                float("rb", "Red ← B", -2.0, 2.0, 0.0),
                float("gr", "Green ← R", -2.0, 2.0, 0.0),
                float("gg", "Green ← G", -2.0, 2.0, 1.0),
                float("gb", "Green ← B", -2.0, 2.0, 0.0),
                float("br", "Blue ← R", -2.0, 2.0, 0.0),
                float("bg", "Blue ← G", -2.0, 2.0, 0.0),
                float("bb", "Blue ← B", -2.0, 2.0, 1.0),
                p("monochrome", "Monochrome", ParamKind::Bool(false)),
            ],
        ),
        adj(
            "selective_color",
            "Selective Color",
            Color,
            "Adjust hue, saturation and lightness of one hue range.",
            vec![
                p(
                    "range",
                    "Range",
                    ParamKind::Choice { options: &["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas"], default: 0 },
                ),
                float("width", "Range width", 0.05, 1.0, 0.35),
                float("hue", "Hue (°)", -180.0, 180.0, 0.0),
                float("saturation", "Saturation", -1.0, 1.0, 0.0),
                float("lightness", "Lightness", -1.0, 1.0, 0.0),
            ],
        ),
        adj(
            "black_white",
            "Black & White",
            Color,
            "Monochrome conversion with per-hue weights and optional tint.",
            vec![
                float("reds", "Reds", -2.0, 3.0, 0.4),
                float("yellows", "Yellows", -2.0, 3.0, 0.6),
                float("greens", "Greens", -2.0, 3.0, 0.4),
                float("cyans", "Cyans", -2.0, 3.0, 0.6),
                float("blues", "Blues", -2.0, 3.0, 0.2),
                float("magentas", "Magentas", -2.0, 3.0, 0.8),
                p("tint", "Tint", ParamKind::Bool(false)),
                p("tint_color", "Tint color", ParamKind::Color([0.9, 0.75, 0.55, 1.0])),
            ],
        ),
        adj(
            "gradient_map",
            "Gradient Map",
            Color,
            "Map luminance through a gradient.",
            vec![p("gradient", "Gradient", ParamKind::Gradient)],
        ),
        adj(
            "lut",
            "3D LUT",
            Color,
            "Apply an Adobe .cube LUT (sRGB-encoded domain).",
            vec![p("cube", "LUT file", ParamKind::Text), float("strength", "Strength", 0.0, 1.0, 1.0)],
        ),
        adj("invert", "Invert", Color, "Invert colours (perceptually, in sRGB-encoded space).", vec![]),
        adj("threshold", "Threshold", Color, "Black/white threshold on luminance.", vec![float("level", "Level", 0.0, 1.0, 0.5)]),
        adj("posterize", "Posterize", Color, "Reduce tonal levels per channel.", vec![p("levels", "Levels", ParamKind::Int { min: 2, max: 64, default: 6 })]),
        // ---------------------------------------------------------------- filters
        NodeKindDesc {
            pad: |n| n.f("radius", 4.0) * 3.0,
            pointwise: false,
            ..adj("gaussian_blur", "Gaussian Blur", Filter, "Separable gaussian blur.", vec![float_log("radius", "Radius", 0.1, 500.0, 4.0), p("edges", "Edges", ParamKind::Choice { options: &["Extend (keep edges)", "Transparent"], default: 0 })])
        },
        NodeKindDesc {
            pad: |n| n.f("radius", 4.0),
            pointwise: false,
            ..adj("box_blur", "Box Blur", Filter, "Separable box blur.", vec![float_log("radius", "Radius", 0.5, 500.0, 4.0), p("edges", "Edges", ParamKind::Choice { options: &["Extend (keep edges)", "Transparent"], default: 0 })])
        },
        NodeKindDesc {
            pad: |n| n.f("distance", 20.0),
            pointwise: false,
            ..adj(
                "motion_blur",
                "Motion Blur",
                Filter,
                "Directional blur.",
                vec![float("angle", "Angle (°)", -180.0, 180.0, 0.0), float_log("distance", "Distance", 1.0, 500.0, 20.0), p("edges", "Edges", ParamKind::Choice { options: &["Extend (keep edges)", "Transparent"], default: 0 })],
            )
        },
        NodeKindDesc {
            pad: |n| n.f("radius", 2.0) * 3.0,
            pointwise: false,
            ..adj(
                "unsharp_mask",
                "Unsharp Mask",
                Filter,
                "Sharpen by adding back high frequencies.",
                vec![float("amount", "Amount", 0.0, 5.0, 1.0), float_log("radius", "Radius", 0.1, 100.0, 2.0), float("threshold", "Threshold", 0.0, 0.5, 0.0), p("edges", "Edges", ParamKind::Choice { options: &["Extend (keep edges)", "Transparent"], default: 0 })],
            )
        },
        NodeKindDesc {
            pad: |n| n.f("radius", 10.0) * 3.0,
            pointwise: false,
            ..adj("high_pass", "High Pass", Filter, "Keep detail above a frequency (centred on grey).", vec![float_log("radius", "Radius", 0.1, 200.0, 10.0), p("edges", "Edges", ParamKind::Choice { options: &["Extend (keep edges)", "Transparent"], default: 0 })])
        },
        adj(
            "noise",
            "Add Noise",
            Filter,
            "Gaussian-ish film grain.",
            vec![float("amount", "Amount", 0.0, 1.0, 0.1), p("mono", "Monochromatic", ParamKind::Bool(true)), p("seed", "Seed", ParamKind::Int { min: 0, max: 9999, default: 1 })],
        ),
        adj(
            "vignette",
            "Vignette",
            Filter,
            "Darken (or lighten) the image edges.",
            vec![
                float("amount", "Amount", -1.0, 1.0, -0.5),
                float("midpoint", "Midpoint", 0.0, 1.0, 0.5),
                float("roundness", "Roundness", 0.0, 1.0, 1.0),
                float("feather", "Feather", 0.01, 1.0, 0.5),
            ],
        ),
        // ---------------------------------------------------------------- composite
        k(
            "blend",
            "Blend",
            Composite,
            "Blend Top over Base with a blend mode, opacity and optional mask.",
            vec![img("Base"), img("Top"), mask_opt("Mask")],
            vec![
                p("mode", "Mode", ParamKind::Choice { options: BLEND_NAMES, default: 0 }),
                float("opacity", "Opacity", 0.0, 1.0, 1.0),
            ],
        ),
        k(
            "mix",
            "Mix by Mask",
            Composite,
            "A where the mask is black, B where it is white.",
            vec![img("A"), img("B"), mask("Mask")],
            vec![float("factor", "Factor", 0.0, 1.0, 1.0)],
        ),
        NodeKindDesc {
            pointwise: false,
            ..k(
                "transform",
                "Transform",
                Composite,
                "Translate, rotate and scale around the document centre.",
                vec![img("Image")],
                vec![
                    p("translate", "Translate", ParamKind::Vec2([0.0, 0.0])),
                    float("rotate", "Rotate (°)", -180.0, 180.0, 0.0),
                    float_log("scale", "Scale", 0.05, 20.0, 1.0),
                ],
            )
        },
        k(
            "crop",
            "Crop",
            Composite,
            "Make everything outside a rectangle transparent.",
            vec![img("Image")],
            vec![p("min", "Top-left", ParamKind::Vec2([0.0, 0.0])), p("max", "Bottom-right", ParamKind::Vec2([512.0, 512.0]))],
        ),
        // ---------------------------------------------------------------- utility
        NodeKindDesc {
            outputs: vec![mask("Mask")],
            ..k(
                "channel",
                "Extract Channel",
                Utility,
                "Turn a channel (or luminance) into a mask.",
                vec![img("Image")],
                vec![p("channel", "Channel", ParamKind::Choice { options: &["Red", "Green", "Blue", "Alpha", "Luminance"], default: 4 })],
            )
        },
        k(
            "combine",
            "Combine Channels",
            Utility,
            "Build an image from up to four masks.",
            vec![mask_opt("R"), mask_opt("G"), mask_opt("B"), mask_opt("A")],
            vec![],
        ),
        NodeKindDesc {
            outputs: vec![mask("Mask")],
            ..k("mask_invert", "Invert Mask", Utility, "1 - mask.", vec![mask("Mask")], vec![])
        },
        NodeKindDesc {
            outputs: vec![mask("Mask")],
            ..k(
                "mask_levels",
                "Mask Range",
                Utility,
                "Remap a mask between low and high with softness.",
                vec![mask("Mask")],
                vec![float("low", "Low", 0.0, 1.0, 0.0), float("high", "High", 0.0, 1.0, 1.0)],
            )
        },
        k(
            "math",
            "Math",
            Utility,
            "Per-channel arithmetic with a constant.",
            vec![img("Image")],
            vec![
                p("op", "Operation", ParamKind::Choice { options: &["Add", "Multiply", "Power", "Min", "Max", "Clamp 0-1"], default: 1 }),
                float("value", "Value", -4.0, 4.0, 1.0),
            ],
        ),
        adj("set_alpha", "Set Alpha", Utility, "Use the mask as the image alpha.", vec![]),
    ];
    // Opacity applied by `set_alpha` uses the optional mask input; no params.
    kinds.shrink_to_fit();
    Registry { kinds }
}

pub const BLEND_NAMES: &[&str] = &[
    "Normal", "Multiply", "Screen", "Overlay", "Darken", "Lighten", "Color Dodge", "Color Burn", "Hard Light",
    "Soft Light", "Difference", "Exclusion", "Add", "Subtract", "Divide", "Linear Burn", "Vivid Light",
    "Linear Light", "Pin Light", "Hue", "Saturation", "Color", "Luminosity",
];

/// Parse an Adobe .cube 3D LUT. Returns (size, rgb data in r-fastest order).
pub fn parse_cube(text: &str) -> Option<(usize, Vec<[f32; 3]>)> {
    let mut size = 0usize;
    let mut data = vec![];
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some(s) = l.strip_prefix("LUT_3D_SIZE") {
            size = s.trim().parse().ok()?;
            continue;
        }
        if l.starts_with(|c: char| c.is_ascii_alphabetic()) {
            continue;
        }
        let v: Vec<f32> = l.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if v.len() == 3 {
            data.push([v[0], v[1], v[2]]);
        }
    }
    (size >= 2 && data.len() == size * size * size).then_some((size, data))
}

/// Evaluate a monotone cubic (Fritsch–Carlson) curve through control points.
pub fn eval_curve(points: &[[f32; 2]], x: f32) -> f32 {
    let mut pts: Vec<[f32; 2]> = points.to_vec();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if pts.is_empty() {
        return x;
    }
    if pts.len() == 1 {
        return pts[0][1];
    }
    if x <= pts[0][0] {
        return pts[0][1];
    }
    if x >= pts[pts.len() - 1][0] {
        return pts[pts.len() - 1][1];
    }
    let n = pts.len();
    let mut d = vec![0.0f32; n - 1];
    for i in 0..n - 1 {
        let dx = (pts[i + 1][0] - pts[i][0]).max(1e-6);
        d[i] = (pts[i + 1][1] - pts[i][1]) / dx;
    }
    let mut m = vec![0.0f32; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for i in 1..n - 1 {
        m[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) / 2.0 };
    }
    for i in 0..n - 1 {
        if d[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let a = m[i] / d[i];
            let b = m[i + 1] / d[i];
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[i] = t * a * d[i];
                m[i + 1] = t * b * d[i];
            }
        }
    }
    let i = pts.windows(2).position(|w| x <= w[1][0]).unwrap_or(n - 2);
    let h = (pts[i + 1][0] - pts[i][0]).max(1e-6);
    let t = (x - pts[i][0]) / h;
    let (t2, t3) = (t * t, t * t * t);
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * pts[i][1] + h10 * h * m[i] + h01 * pts[i + 1][1] + h11 * h * m[i + 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_curve() {
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            assert!((eval_curve(&[[0.0, 0.0], [1.0, 1.0]], x) - x).abs() < 1e-6);
        }
    }

    #[test]
    fn monotone_curve_no_overshoot() {
        let pts = [[0.0, 0.0], [0.3, 0.8], [0.35, 0.82], [1.0, 1.0]];
        let mut prev = -1.0;
        for i in 0..=1000 {
            let y = eval_curve(&pts, i as f32 / 1000.0);
            assert!(y >= prev - 1e-6 && y <= 1.0 + 1e-6);
            prev = y;
        }
    }

    #[test]
    fn all_kinds_have_defaults() {
        for k in &registry().kinds {
            let _ = default_params(k.id);
        }
        assert!(default_params("levels").contains_key("r.gamma"));
    }
}
