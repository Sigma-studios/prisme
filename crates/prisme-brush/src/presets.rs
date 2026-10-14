use crate::dab::TipShape;
use crate::{BrushSettings, Dynamic, InputSource};

#[derive(Clone, Debug)]
pub struct BrushPreset {
    pub name: &'static str,
    pub settings: BrushSettings,
}

/// Built-in brush presets. Dynamics default to pressure so tablets work out of the box;
/// with a mouse (pressure 1.0) they behave like plain brushes.
pub fn presets() -> Vec<BrushPreset> {
    let base = BrushSettings::default();
    vec![
        BrushPreset { name: "Round Soft", settings: BrushSettings { hardness: 0.0, size: 40.0, spacing: 0.08, ..base.clone() } },
        BrushPreset { name: "Round Hard", settings: BrushSettings { hardness: 1.0, size: 20.0, ..base.clone() } },
        BrushPreset { name: "Square", settings: BrushSettings { tip: TipShape::Square, hardness: 1.0, size: 20.0, ..base.clone() } },
        BrushPreset {
            name: "Pixel",
            settings: BrushSettings {
                size: 1.0,
                hardness: 1.0,
                pixel_perfect: true,
                spacing: 0.25,
                size_dynamic: Dynamic::default(),
                ..base.clone()
            },
        },
        BrushPreset {
            name: "Pencil",
            settings: BrushSettings {
                size: 3.0,
                hardness: 0.9,
                spacing: 0.15,
                flow: 0.7,
                tip: TipShape::Chalk,
                grain: 0.35,
                opacity_dynamic: Dynamic::pressure(0.3),
                size_dynamic: Dynamic::pressure(0.6),
                ..base.clone()
            },
        },
        BrushPreset {
            name: "Ink Pen",
            settings: BrushSettings {
                size: 8.0,
                hardness: 1.0,
                spacing: 0.05,
                smoothing: 0.5,
                size_dynamic: Dynamic { source: InputSource::Pressure, min: 0.1, gamma: 1.4 },
                ..base.clone()
            },
        },
        BrushPreset {
            name: "Airbrush",
            settings: BrushSettings {
                size: 80.0,
                hardness: 0.0,
                flow: 0.08,
                spacing: 0.1,
                airbrush: true,
                size_dynamic: Dynamic::default(),
                flow_dynamic: Dynamic::pressure(0.0),
                ..base.clone()
            },
        },
        BrushPreset {
            name: "Marker",
            settings: BrushSettings {
                size: 24.0,
                hardness: 0.85,
                roundness: 0.35,
                angle: 0.6,
                flow: 0.35,
                opacity: 0.85,
                spacing: 0.06,
                size_dynamic: Dynamic::default(),
                ..base.clone()
            },
        },
        BrushPreset {
            name: "Chalk",
            settings: BrushSettings {
                size: 30.0,
                hardness: 0.7,
                tip: TipShape::Chalk,
                grain: 0.6,
                spacing: 0.12,
                jitter_angle: 0.3,
                opacity_dynamic: Dynamic::pressure(0.2),
                ..base.clone()
            },
        },
        BrushPreset {
            name: "Charcoal",
            settings: BrushSettings {
                size: 26.0,
                hardness: 0.6,
                tip: TipShape::Charcoal,
                grain: 0.5,
                roundness: 0.6,
                angle_source: InputSource::Direction,
                spacing: 0.08,
                ..base.clone()
            },
        },
        BrushPreset {
            name: "Splatter",
            settings: BrushSettings {
                size: 60.0,
                hardness: 1.0,
                tip: TipShape::Splatter,
                spacing: 0.6,
                scatter: 0.3,
                jitter_size: 0.5,
                angle_source: InputSource::Random,
                size_dynamic: Dynamic::default(),
                ..base
            },
        },
    ]
}
