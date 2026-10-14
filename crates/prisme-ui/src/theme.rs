//! Prisme visual theme.

use egui::{Color32, CornerRadius, Stroke, Visuals};

pub const ACCENT: Color32 = Color32::from_rgb(110, 140, 255);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(70, 92, 170);
pub const BG0: Color32 = Color32::from_rgb(24, 25, 28);
pub const BG1: Color32 = Color32::from_rgb(32, 33, 37);
pub const BG2: Color32 = Color32::from_rgb(42, 43, 48);
pub const BG3: Color32 = Color32::from_rgb(54, 56, 62);
pub const TEXT: Color32 = Color32::from_rgb(222, 224, 230);
pub const TEXT_DIM: Color32 = Color32::from_rgb(150, 153, 162);
pub const WARN: Color32 = Color32::from_rgb(240, 180, 80);
pub const CANVAS_BG: [f32; 3] = [0.082, 0.086, 0.094];

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum ThemeKind {
    #[default]
    Dark,
    Light,
}

pub fn apply(ctx: &egui::Context, kind: ThemeKind) {
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 5.0);
    style.spacing.button_padding = egui::vec2(8.0, 3.0);
    style.spacing.interact_size.y = 22.0;
    style.spacing.slider_width = 140.0;
    style.spacing.window_margin = egui::Margin::same(10);
    style.spacing.menu_margin = egui::Margin::same(6);
    style.interaction.selectable_labels = false;
    style.visuals = match kind {
        ThemeKind::Dark => dark(),
        ThemeKind::Light => Visuals::light(),
    };
    ctx.set_global_style(style);
}

fn dark() -> Visuals {
    let mut v = Visuals::dark();
    let r = CornerRadius::same(5);
    v.panel_fill = BG1;
    v.window_fill = BG1;
    v.extreme_bg_color = BG0;
    v.faint_bg_color = BG2;
    v.code_bg_color = BG0;
    v.window_stroke = Stroke::new(1.0, BG3);
    v.window_corner_radius = CornerRadius::same(8);
    v.menu_corner_radius = CornerRadius::same(6);
    v.selection.bg_fill = ACCENT_DIM;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.widgets.noninteractive.bg_fill = BG1;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BG3);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_DIM);
    v.widgets.noninteractive.corner_radius = r;
    v.widgets.inactive.bg_fill = BG2;
    v.widgets.inactive.weak_bg_fill = BG2;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.inactive.corner_radius = r;
    v.widgets.hovered.bg_fill = BG3;
    v.widgets.hovered.weak_bg_fill = BG3;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_DIM);
    v.widgets.hovered.corner_radius = r;
    v.widgets.active.bg_fill = ACCENT_DIM;
    v.widgets.active.weak_bg_fill = ACCENT_DIM;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.corner_radius = r;
    v.widgets.open.bg_fill = BG3;
    v.widgets.open.corner_radius = r;
    v
}

/// Working-space linear RGBA -> display sRGB Color32 (for swatches; unmultiplied alpha).
pub fn to_color32(c: [f32; 4]) -> Color32 {
    let s = prisme_color::srgb().from_working([c[0], c[1], c[2]]);
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgba_unmultiplied(q(s[0]), q(s[1]), q(s[2]), q(c[3]))
}

/// sRGB Color32 -> working-space linear RGBA.
pub fn from_color32(c: Color32) -> [f32; 4] {
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    let w = prisme_color::srgb().to_working([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]);
    [w[0], w[1], w[2], a as f32 / 255.0]
}

/// Draw a checkerboard (transparency backdrop) into a rect.
pub fn checker(painter: &egui::Painter, rect: egui::Rect, size: f32) {
    painter.rect_filled(rect, 0.0, Color32::from_gray(190));
    let nx = (rect.width() / size).ceil() as i32;
    let ny = (rect.height() / size).ceil() as i32;
    for y in 0..ny {
        for x in 0..nx {
            if (x + y) % 2 == 0 {
                let min = rect.min + egui::vec2(x as f32 * size, y as f32 * size);
                let r = egui::Rect::from_min_size(min, egui::vec2(size, size)).intersect(rect);
                painter.rect_filled(r, 0.0, Color32::from_gray(140));
            }
        }
    }
}
