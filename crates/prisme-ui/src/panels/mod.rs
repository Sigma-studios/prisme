pub mod brushes;
pub mod color;
pub mod history;
pub mod layers;
pub mod masks;
pub mod navigator;
pub mod nodes;
pub mod options;
pub mod properties;
pub mod scopes;

use egui::{RichText, Ui};

pub fn section(ui: &mut Ui, title: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(title.to_uppercase()).small().strong().color(crate::theme::TEXT_DIM));
}

pub fn no_doc(ui: &mut Ui) {
    ui.centered_and_justified(|ui| ui.label(RichText::new("No document").color(crate::theme::TEXT_DIM)));
}
