//! Keyboard Shortcuts editor: its own dock tab, opened beside the Canvas tab.

use crate::actions::{shortcut_matches, Action, Binding, Keymap};
use crate::app::AppState;
use crate::theme;
use egui::{Button, RichText, TextEdit, Ui, Vec2};
use egui_phosphor::regular as ph;

pub const TITLE: &str = "Keyboard Shortcuts";

#[derive(Default)]
pub struct ShortcutsTab {
    /// Pending dock change: Some(true) = add/show the tab, Some(false) = remove it.
    pub request: Option<bool>,
    /// The tab is showing (active in its dock leaf); set by the app each frame.
    pub focused: bool,
    /// The tab is in the dock leaf that has focus (Cmd+W closes it).
    pub dock_focused: bool,
    pub search: String,
    pub rebinding: Option<Action>,
    /// Captured binding awaiting Replace/Cancel: (target, binding, current holder).
    pub conflict: Option<(Action, Binding, Action)>,
    /// Bring the Canvas dock tab forward on the next frame (a document was opened/selected).
    pub reveal: bool,
    focus_search: bool,
}

/// Open (or focus) the shortcuts tab.
pub fn open(s: &mut AppState) {
    if let Some(d) = s.doc_mut() {
        d.cancel_working();
    }
    let st = &mut s.dialogs.shortcuts;
    st.request = Some(true);
    st.focus_search = true;
}

pub fn close(s: &mut AppState) {
    s.dialogs.shortcuts.request = Some(false);
    stop_editing(s);
}

/// A document was opened or selected: show the Canvas tab.
pub fn unfocus(s: &mut AppState) {
    s.dialogs.shortcuts.reveal = true;
}

/// Drop an in-progress rebind / conflict prompt (tab hidden or closed).
pub fn stop_editing(s: &mut AppState) {
    let st = &mut s.dialogs.shortcuts;
    st.rebinding = None;
    st.conflict = None;
}

/// Run at the start of the frame, before shortcut dispatch. Returns true while capturing a
/// key for rebinding (all keyboard input this frame is swallowed).
pub fn capture(s: &mut AppState, ctx: &egui::Context) -> bool {
    let st = &mut s.dialogs.shortcuts;
    if !st.focused {
        return false;
    }
    let esc = ctx.input(|i| i.key_pressed(egui::Key::Escape));
    let Some(a) = st.rebinding else {
        // Esc: dismiss a pending conflict, else clear the search.
        if esc && (st.conflict.is_some() || !st.search.is_empty()) {
            if st.conflict.take().is_none() {
                st.search.clear();
            }
            ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        }
        return false;
    };
    let captured = ctx.input(|i| {
        i.events.iter().find_map(|e| match e {
            egui::Event::Key { key, pressed: true, modifiers, .. } if *key != egui::Key::Escape => {
                Some(Binding { key: *key, command: modifiers.command, shift: modifiers.shift, alt: modifiers.alt })
            }
            _ => None,
        })
    });
    if let Some(b) = captured {
        st.rebinding = None;
        assign(st, &mut s.settings.keymap, a, Some(b));
    } else if esc {
        st.rebinding = None;
    }
    // Keep the key from reaching menus, focused buttons or the search field.
    ctx.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Key { .. } | egui::Event::Text(_))));
    true
}

/// Bind `a` to `b`, or stage a conflict if another action already uses `b`.
fn assign(st: &mut ShortcutsTab, km: &mut Keymap, a: Action, b: Option<Binding>) {
    st.conflict = None;
    match b.and_then(|b| km.holder(b, a).map(|h| (b, h))) {
        Some((b, h)) => st.conflict = Some((a, b, h)),
        None => km.set(a, b),
    }
}

fn display_name(a: Action) -> String {
    match a {
        Action::Tool(t) => t.name().to_string(),
        _ => a.name(),
    }
}

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    let defaults = Keymap::default();
    let (st, km) = (&mut s.dialogs.shortcuts, &mut s.settings.keymap);
    egui::Frame::new().inner_margin(egui::Margin::symmetric(16, 10)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{} {TITLE}", ph::KEYBOARD)).size(18.0).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(format!("{} Reset all", ph::ARROW_COUNTER_CLOCKWISE)).clicked() {
                    *km = Keymap::default();
                    st.conflict = None;
                    st.rebinding = None;
                }
            });
        });
        ui.label(RichText::new("Click a shortcut to rebind (Esc cancels) · right-click to clear").small().color(theme::TEXT_DIM));
        ui.add_space(4.0);
        let r = ui.add(
            TextEdit::singleline(&mut st.search)
                .hint_text(format!("{} Search actions or keys (brush, ⌘S, ctrl+s…)", ph::MAGNIFYING_GLASS))
                .desired_width(f32::INFINITY),
        );
        if std::mem::take(&mut st.focus_search) {
            r.request_focus();
        }
        ui.add_space(4.0);
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let mut any = false;
            for cat in Action::CATEGORIES {
                let rows: Vec<Action> = Action::all()
                    .into_iter()
                    .filter(|a| a.category() == cat)
                    .filter(|a| {
                        let b = km.get(*a);
                        shortcut_matches(&a.name(), b, &b.map(|b| b.label(&ctx)).unwrap_or_default(), &st.search)
                    })
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                any = true;
                ui.add_space(6.0);
                ui.label(RichText::new(cat.to_uppercase()).small().strong().color(theme::TEXT_DIM));
                egui::Grid::new(("keys", cat)).striped(true).num_columns(3).min_col_width(24.0).show(ui, |ui| {
                    for a in rows {
                        row(ui, &ctx, st, km, &defaults, a);
                    }
                });
            }
            if !any {
                ui.add_space(24.0);
                ui.vertical_centered(|ui| ui.label(RichText::new("No matching actions").color(theme::TEXT_DIM)));
            }
        });
    });
}

fn row(ui: &mut Ui, ctx: &egui::Context, st: &mut ShortcutsTab, km: &mut Keymap, defaults: &Keymap, a: Action) {
    ui.add_sized([240.0, 20.0], egui::Label::new(display_name(a)).truncate());
    let pending = st.conflict.filter(|c| c.0 == a);
    let capturing = st.rebinding == Some(a);
    let (txt, col) = if capturing {
        ("press a key…".to_string(), theme::ACCENT)
    } else if let Some((_, b, _)) = pending {
        (b.label(ctx), theme::WARN)
    } else {
        match km.label(ctx, a) {
            l if l.is_empty() => ("—".to_string(), theme::TEXT_DIM),
            l => (l, theme::TEXT),
        }
    };
    let r = ui.add(Button::new(RichText::new(txt).color(col)).min_size(Vec2::new(140.0, 0.0)).selected(capturing)).on_hover_text("Click to rebind · right-click to clear");
    if r.clicked() {
        st.rebinding = Some(a);
        st.conflict = None;
    }
    if r.secondary_clicked() {
        km.set(a, None);
        st.rebinding = None;
    }
    let def = defaults.get(a);
    if km.get(a) != def {
        let tip = format!("Reset to {}", def.map(|b| b.label(ctx)).unwrap_or_else(|| "none".into()));
        if ui.small_button(ph::ARROW_COUNTER_CLOCKWISE).on_hover_text(tip).clicked() {
            assign(st, km, a, def);
        }
    } else {
        ui.label("");
    }
    ui.end_row();
    if let Some((_, b, holder)) = pending {
        ui.label(RichText::new(format!("{} Already used by {}", ph::WARNING, display_name(holder))).color(theme::WARN));
        ui.horizontal(|ui| {
            if ui.button("Replace").clicked() {
                km.set(a, Some(b)); // unbinds the holder
                st.conflict = None;
            }
            if ui.button("Cancel").clicked() {
                st.conflict = None;
            }
        });
        ui.label("");
        ui.end_row();
    }
}
