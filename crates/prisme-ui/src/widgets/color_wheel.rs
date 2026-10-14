//! Colour wheel: hue ring with an inner saturation/value square (HSV, display sRGB) or
//! an OKLCh perceptual mode (lightness/chroma plane, out-of-gamut shown hatched-grey).
//! Smooth sub-pixel dragging; Shift for fine adjustment.

use egui::{Color32, Mesh, Pos2, Rect, Sense, Shape, Stroke, Ui, Vec2};
use prisme_color::{hsv_to_rgb, oklab_to_working, oklch_to_oklab, rgb_to_hsv, srgb, working_to_oklab, oklab_to_oklch};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum WheelMode {
    #[default]
    Hsv,
    Oklch,
}

#[derive(Clone, Copy, Default)]
struct State {
    /// Persisted hue/sat so they don't jump when the colour is grey/black.
    hsv: [f32; 3],
    lch: [f32; 3],
    last: [f32; 4],
    drag: u8, // 0 none, 1 ring, 2 inner
}

fn srgb8(rgb: [f32; 3]) -> Color32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(q(rgb[0]), q(rgb[1]), q(rgb[2]))
}

fn lch_to_srgb(lch: [f32; 3]) -> ([f32; 3], bool) {
    let w = oklab_to_working(oklch_to_oklab(lch));
    let s = srgb().from_working(w);
    let ok = s.iter().all(|&v| (-0.001..=1.001).contains(&v));
    (s, ok)
}

/// Returns true when `color` (linear working RGBA) changed.
pub fn color_wheel(ui: &mut Ui, id: egui::Id, color: &mut [f32; 4], mode: WheelMode, size: f32) -> bool {
    let mut st: State = ui.data(|d| d.get_temp(id)).unwrap_or_default();
    if st.last != *color {
        let s = srgb().from_working([color[0], color[1], color[2]]);
        let hsv = rgb_to_hsv([s[0].clamp(0.0, 1.0), s[1].clamp(0.0, 1.0), s[2].clamp(0.0, 1.0)]);
        st.hsv = [if hsv[1] > 1e-4 && hsv[2] > 1e-4 { hsv[0] } else { st.hsv[0] }, if hsv[2] > 1e-4 { hsv[1] } else { st.hsv[1] }, hsv[2]];
        let lch = oklab_to_oklch(working_to_oklab([color[0], color[1], color[2]]));
        st.lch = [lch[0], lch[1], if lch[1] > 1e-4 { lch[2] } else { st.lch[2] }];
    }
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click_and_drag());
    let painter = ui.painter_at(rect.expand(4.0));
    let c = rect.center();
    let r_out = size * 0.5 - 2.0;
    let r_in = r_out * 0.80;
    let fine = ui.input(|i| i.modifiers.shift);

    // ---- hue ring
    let mut mesh = Mesh::default();
    let n = 96;
    for i in 0..=n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let col = match mode {
            WheelMode::Hsv => srgb8(hsv_to_rgb([a / std::f32::consts::TAU, 1.0, 1.0])),
            WheelMode::Oklch => srgb8(lch_to_srgb([0.72, 0.13, a]).0),
        };
        let d = Vec2::angled(a);
        mesh.colored_vertex(c + d * r_in, col);
        mesh.colored_vertex(c + d * r_out, col);
        if i > 0 {
            let b = (i as u32 - 1) * 2;
            mesh.add_triangle(b, b + 1, b + 2);
            mesh.add_triangle(b + 1, b + 2, b + 3);
        }
    }
    painter.add(mesh);

    // ---- inner plane
    let half = r_in * 0.68;
    let sq = Rect::from_center_size(c, Vec2::splat(half * 2.0));
    let grid = 24;
    let mut mesh = Mesh::default();
    for j in 0..=grid {
        for i in 0..=grid {
            let (u, v) = (i as f32 / grid as f32, j as f32 / grid as f32);
            let col = match mode {
                WheelMode::Hsv => srgb8(hsv_to_rgb([st.hsv[0], u, 1.0 - v])),
                WheelMode::Oklch => {
                    let (s, ok) = lch_to_srgb([1.0 - v, u * 0.37, st.lch[2]]);
                    if ok { srgb8(s) } else { Color32::from_gray(((1.0 - v) * 90.0 + 30.0) as u8) }
                }
            };
            mesh.colored_vertex(Pos2::new(sq.left() + u * sq.width(), sq.top() + v * sq.height()), col);
        }
    }
    let w = grid as u32 + 1;
    for j in 0..grid as u32 {
        for i in 0..grid as u32 {
            let a = j * w + i;
            mesh.add_triangle(a, a + 1, a + w);
            mesh.add_triangle(a + 1, a + w, a + w + 1);
        }
    }
    painter.add(mesh);
    painter.rect_stroke(sq, 0.0, Stroke::new(1.0, Color32::from_gray(20)), egui::StrokeKind::Outside);

    // ---- interaction
    let mut changed = false;
    if (resp.drag_started() || resp.clicked())
        && let Some(p) = resp.interact_pointer_pos() {
            let d = (p - c).length();
            st.drag = if d >= r_in - 4.0 && d <= r_out + 6.0 { 1 } else if sq.expand(6.0).contains(p) { 2 } else { 0 };
        }
    if let Some(p) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
        match st.drag {
            1 => {
                let mut a = (p - c).angle();
                if a < 0.0 {
                    a += std::f32::consts::TAU;
                }
                match mode {
                    WheelMode::Hsv => {
                        let h = a / std::f32::consts::TAU;
                        st.hsv[0] = if fine { st.hsv[0] + (h - st.hsv[0]) * 0.2 } else { h };
                    }
                    WheelMode::Oklch => st.lch[2] = if fine { st.lch[2] + (a - st.lch[2]) * 0.2 } else { a },
                }
                changed = true;
            }
            2 => {
                let u = ((p.x - sq.left()) / sq.width()).clamp(0.0, 1.0);
                let v = ((p.y - sq.top()) / sq.height()).clamp(0.0, 1.0);
                match mode {
                    WheelMode::Hsv => {
                        let (s, val) = (u, 1.0 - v);
                        if fine {
                            st.hsv[1] += (s - st.hsv[1]) * 0.2;
                            st.hsv[2] += (val - st.hsv[2]) * 0.2;
                        } else {
                            st.hsv[1] = s;
                            st.hsv[2] = val;
                        }
                    }
                    WheelMode::Oklch => {
                        st.lch[0] = 1.0 - v;
                        st.lch[1] = u * 0.37;
                    }
                }
                changed = true;
            }
            _ => {}
        }
    }
    if resp.drag_stopped() {
        st.drag = 0;
    }
    if changed {
        let w = match mode {
            WheelMode::Hsv => srgb().to_working(hsv_to_rgb(st.hsv)),
            WheelMode::Oklch => {
                // Clip chroma into gamut so the picked colour is displayable.
                let mut lch = st.lch;
                for _ in 0..24 {
                    if lch_to_srgb(lch).1 {
                        break;
                    }
                    lch[1] *= 0.92;
                }
                oklab_to_working(oklch_to_oklab(lch))
            }
        };
        color[0] = w[0];
        color[1] = w[1];
        color[2] = w[2];
    }

    // ---- markers
    let (ha, (mx, my)) = match mode {
        WheelMode::Hsv => (st.hsv[0] * std::f32::consts::TAU, (st.hsv[1], 1.0 - st.hsv[2])),
        WheelMode::Oklch => (st.lch[2], ((st.lch[1] / 0.37).clamp(0.0, 1.0), (1.0 - st.lch[0]).clamp(0.0, 1.0))),
    };
    let hp = c + Vec2::angled(ha) * (r_in + r_out) * 0.5;
    painter.circle_stroke(hp, (r_out - r_in) * 0.5 + 1.0, Stroke::new(2.0, Color32::WHITE));
    painter.circle_stroke(hp, (r_out - r_in) * 0.5 + 2.5, Stroke::new(1.0, Color32::BLACK));
    let ip = Pos2::new(sq.left() + mx * sq.width(), sq.top() + my * sq.height());
    painter.add(Shape::circle_stroke(ip, 6.0, Stroke::new(2.0, Color32::WHITE)));
    painter.add(Shape::circle_stroke(ip, 7.5, Stroke::new(1.0, Color32::BLACK)));

    st.last = *color;
    ui.data_mut(|d| d.insert_temp(id, st));
    changed
}
