use crate::app::AppState;
use crate::theme;
use egui::{RichText, Ui};

pub fn ui(s: &mut AppState, ui: &mut Ui) {
    let Some(d) = s.doc_mut() else { return super::no_doc(ui) };
    let cursor = d.history.cursor();
    let labels: Vec<String> = d.history.entries().iter().map(|e| e.label.clone()).collect();
    let mut jump = None;
    ui.horizontal(|ui| {
        if ui.add_enabled(d.history.can_undo(), egui::Button::new(format!("{} Undo", egui_phosphor::regular::ARROW_COUNTER_CLOCKWISE))).clicked() {
            jump = Some(cursor - 1);
        }
        if ui.add_enabled(d.history.can_redo(), egui::Button::new(format!("{} Redo", egui_phosphor::regular::ARROW_CLOCKWISE))).clicked() {
            jump = Some(cursor + 1);
        }
        ui.label(RichText::new(format!("{:.0} MB", d.history.memory_bytes() as f64 / 1e6)).small().color(theme::TEXT_DIM))
            .on_hover_text("Unique pixel memory held by history");
    });
    ui.separator();
    egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        for (i, l) in labels.iter().enumerate() {
            let future = i > cursor;
            let txt = RichText::new(format!("{}  {}", if i == 0 { egui_phosphor::regular::FILE_IMAGE } else { "•" }, l)).color(if future { theme::TEXT_DIM } else { theme::TEXT });
            if ui.selectable_label(i == cursor, txt).clicked() {
                jump = Some(i);
            }
        }
    });
    if let Some(i) = jump {
        d.working = None;
        d.history.jump(i);
        d.touch();
    }
}
