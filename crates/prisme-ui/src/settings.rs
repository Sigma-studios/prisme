use crate::actions::Keymap;
use crate::theme::ThemeKind;
use crate::widgets::color_wheel::WheelMode;
use prisme_tools::ToolOptions;
use serde::{Deserialize, Serialize};

/// Persisted user settings.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub theme: ThemeKind,
    pub keymap: Keymap,
    pub tools: ToolOptions,
    pub wheel_mode: WheelMode,
    pub use_display_profile: bool,
    pub pixel_grid: bool,
    pub fg: [f32; 4],
    pub bg: [f32; 4],
    pub export: prisme_export::PngOptions,
    pub autosave_minutes: f32,
    pub recent_colors: Vec<[f32; 4]>,
    /// Recently opened/saved files, most recent first (native paths).
    pub recent_files: Vec<String>,
    pub dither_display: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: ThemeKind::Dark,
            keymap: Keymap::default(),
            tools: ToolOptions::default(),
            wheel_mode: WheelMode::Hsv,
            use_display_profile: true,
            pixel_grid: true,
            fg: [0.0, 0.0, 0.0, 1.0],
            bg: [1.0, 1.0, 1.0, 1.0],
            export: prisme_export::PngOptions::default(),
            autosave_minutes: 3.0,
            recent_colors: vec![],
            recent_files: vec![],
            dither_display: true,
        }
    }
}

pub const SETTINGS_KEY: &str = "prisme-settings";
