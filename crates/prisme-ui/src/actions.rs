//! Action registry and rebindable keymap.

use egui::{Key, KeyboardShortcut, Modifiers};
use prisme_tools::ToolKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    New,
    Open,
    Save,
    SaveAs,
    Export,
    Close,
    Undo,
    Redo,
    Cut,
    Copy,
    CopyMerged,
    Paste,
    PasteNewDoc,
    Delete,
    FillFg,
    FillBg,
    SelectAll,
    Deselect,
    InvertSelection,
    SaveSelectionMask,
    NewLayer,
    DuplicateLayer,
    DeleteLayer,
    MergeDown,
    GroupLayer,
    AddMask,
    ImageSize,
    CanvasSize,
    ZoomIn,
    ZoomOut,
    ZoomFit,
    Zoom100,
    RotateViewLeft,
    RotateViewRight,
    ResetView,
    ToggleNodeOutput,
    SwapColors,
    ResetColors,
    BrushSmaller,
    BrushBigger,
    PickScreenColor,
    Preferences,
    Tool(ToolKind),
}

impl Action {
    pub fn name(&self) -> String {
        match self {
            Action::Tool(t) => format!("Tool: {}", t.name()),
            other => format!("{other:?}")
                .chars()
                .enumerate()
                .flat_map(|(i, c)| if i > 0 && c.is_uppercase() { vec![' ', c] } else { vec![c] })
                .collect(),
        }
    }

    pub fn all() -> Vec<Action> {
        use Action::*;
        let mut v = vec![
            New, Open, Save, SaveAs, Export, Close, Undo, Redo, Cut, Copy, CopyMerged, Paste, PasteNewDoc, Delete, FillFg, FillBg,
            SelectAll, Deselect, InvertSelection, SaveSelectionMask, NewLayer, DuplicateLayer, DeleteLayer, MergeDown,
            GroupLayer, AddMask, ImageSize, CanvasSize, ZoomIn, ZoomOut, ZoomFit, Zoom100, RotateViewLeft, RotateViewRight,
            ResetView, ToggleNodeOutput, SwapColors, ResetColors, BrushSmaller, BrushBigger, PickScreenColor, Preferences,
        ];
        v.extend(ToolKind::ALL.iter().map(|t| Tool(*t)));
        v
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub key: Key,
    pub command: bool,
    pub shift: bool,
    pub alt: bool,
}

impl Binding {
    pub const fn new(key: Key) -> Self {
        Self { key, command: false, shift: false, alt: false }
    }
    pub const fn cmd(key: Key) -> Self {
        Self { key, command: true, shift: false, alt: false }
    }
    pub const fn cmd_shift(key: Key) -> Self {
        Self { key, command: true, shift: true, alt: false }
    }
    pub const fn cmd_alt(key: Key) -> Self {
        Self { key, command: true, shift: false, alt: true }
    }
    pub const fn shift(key: Key) -> Self {
        Self { key, command: false, shift: true, alt: false }
    }

    pub fn shortcut(&self) -> KeyboardShortcut {
        let mut m = Modifiers::NONE;
        if self.command {
            m |= Modifiers::COMMAND;
        }
        if self.shift {
            m |= Modifiers::SHIFT;
        }
        if self.alt {
            m |= Modifiers::ALT;
        }
        KeyboardShortcut::new(m, self.key)
    }

    pub fn label(&self, ctx: &egui::Context) -> String {
        ctx.format_shortcut(&self.shortcut())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keymap {
    pub bindings: BTreeMap<String, Binding>,
}

fn key_of(c: char) -> Option<Key> {
    Key::from_name(&c.to_ascii_uppercase().to_string())
}

impl Default for Keymap {
    fn default() -> Self {
        use Action::*;
        let mut m = BTreeMap::new();
        let mut add = |a: Action, b: Binding| {
            m.insert(format!("{a:?}"), b);
        };
        add(New, Binding::cmd(Key::N));
        add(Open, Binding::cmd(Key::O));
        add(Save, Binding::cmd(Key::S));
        add(SaveAs, Binding::cmd_shift(Key::S));
        add(Export, Binding::cmd_alt(Key::E));
        add(Close, Binding::cmd(Key::W));
        add(Undo, Binding::cmd(Key::Z));
        add(Redo, Binding::cmd_shift(Key::Z));
        add(Cut, Binding::cmd(Key::X));
        add(Copy, Binding::cmd(Key::C));
        add(CopyMerged, Binding::cmd_shift(Key::C));
        add(Paste, Binding::cmd(Key::V));
        add(PasteNewDoc, Binding::cmd_alt(Key::V));
        add(Delete, Binding::new(Key::Delete));
        add(FillFg, Binding { key: Key::Backspace, command: false, shift: false, alt: true });
        add(FillBg, Binding::cmd(Key::Backspace));
        add(SelectAll, Binding::cmd(Key::A));
        add(Deselect, Binding::cmd(Key::D));
        add(InvertSelection, Binding::cmd_shift(Key::I));
        add(NewLayer, Binding::cmd_shift(Key::N));
        add(DuplicateLayer, Binding::cmd(Key::J));
        add(MergeDown, Binding::cmd(Key::E));
        add(GroupLayer, Binding::cmd(Key::G));
        add(ImageSize, Binding::cmd_alt(Key::I));
        add(CanvasSize, Binding::cmd_alt(Key::C));
        add(ZoomIn, Binding::cmd(Key::Equals));
        add(ZoomOut, Binding::cmd(Key::Minus));
        add(ZoomFit, Binding::cmd(Key::Num0));
        add(Zoom100, Binding::cmd(Key::Num1));
        add(RotateViewLeft, Binding::shift(Key::OpenBracket));
        add(RotateViewRight, Binding::shift(Key::CloseBracket));
        add(ResetView, Binding::new(Key::Escape).with_shift());
        add(ToggleNodeOutput, Binding::new(Key::Backslash));
        add(SwapColors, Binding::new(Key::X));
        add(ResetColors, Binding::new(Key::D));
        add(BrushSmaller, Binding::new(Key::OpenBracket));
        add(BrushBigger, Binding::new(Key::CloseBracket));
        add(PickScreenColor, Binding::shift(Key::I));
        add(Preferences, Binding::cmd(Key::Comma));
        for t in ToolKind::ALL {
            if let Some(c) = t.shortcut()
                && let Some(k) = key_of(c) {
                    add(Tool(t), Binding::new(k));
                }
        }
        add(Tool(ToolKind::SelectEllipse), Binding::shift(Key::M));
        add(Tool(ToolKind::PolyLasso), Binding::shift(Key::L));
        add(Tool(ToolKind::ColorSelect), Binding::shift(Key::W));
        add(Tool(ToolKind::Blur), Binding::shift(Key::R));
        Keymap { bindings: m }
    }
}

impl Binding {
    const fn with_shift(mut self) -> Self {
        self.shift = true;
        self
    }
}

impl Keymap {
    pub fn get(&self, a: Action) -> Option<Binding> {
        self.bindings.get(&format!("{a:?}")).copied()
    }

    pub fn set(&mut self, a: Action, b: Option<Binding>) {
        let k = format!("{a:?}");
        match b {
            Some(b) => {
                // Unbind any other action using this binding.
                self.bindings.retain(|_, v| *v != b);
                self.bindings.insert(k, b);
            }
            None => {
                self.bindings.remove(&k);
            }
        }
    }

    pub fn label(&self, ctx: &egui::Context, a: Action) -> String {
        self.get(a).map(|b| b.label(ctx)).unwrap_or_default()
    }

    /// Consume triggered actions from this frame's input.
    pub fn triggered(&self, ctx: &egui::Context) -> Vec<Action> {
        let mut out = vec![];
        // Longest modifier combos first so Cmd+Shift+S beats Cmd+S.
        let mut entries: Vec<(Action, Binding)> = Action::all().into_iter().filter_map(|a| self.get(a).map(|b| (a, b))).collect();
        entries.sort_by_key(|(_, b)| std::cmp::Reverse(b.command as u8 + b.shift as u8 + b.alt as u8));
        ctx.input_mut(|i| {
            for (a, b) in entries {
                if i.consume_shortcut(&b.shortcut()) {
                    out.push(a);
                }
            }
        });
        out
    }
}
