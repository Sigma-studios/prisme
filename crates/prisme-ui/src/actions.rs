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
    ExportSelection,
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
            New, Open, Save, SaveAs, Export, ExportSelection, Close, Undo, Redo, Cut, Copy, CopyMerged, Paste, PasteNewDoc, Delete, FillFg, FillBg,
            SelectAll, Deselect, InvertSelection, SaveSelectionMask, NewLayer, DuplicateLayer, DeleteLayer, MergeDown,
            GroupLayer, AddMask, ImageSize, CanvasSize, ZoomIn, ZoomOut, ZoomFit, Zoom100, RotateViewLeft, RotateViewRight,
            ResetView, ToggleNodeOutput, SwapColors, ResetColors, BrushSmaller, BrushBigger, PickScreenColor, Preferences,
        ];
        v.extend(ToolKind::ALL.iter().map(|t| Tool(*t)));
        v
    }

    /// Display order of `category()` groups.
    pub const CATEGORIES: [&'static str; 9] = ["Tools", "File", "Edit", "Select", "Layer", "Image", "View", "Color", "App"];

    /// Doesn't touch the document (view, colours, tool settings): an open Liquify session
    /// stays open across it.
    pub fn view_only(&self) -> bool {
        use Action::*;
        matches!(
            self,
            Undo | Redo | ZoomIn | ZoomOut | ZoomFit | Zoom100 | RotateViewLeft | RotateViewRight | ResetView | ToggleNodeOutput | SwapColors
                | ResetColors | BrushSmaller | BrushBigger | PickScreenColor | Preferences | Tool(_)
        )
    }

    pub fn category(&self) -> &'static str {
        use Action::*;
        match self {
            Tool(_) | BrushSmaller | BrushBigger => "Tools",
            New | Open | Save | SaveAs | Export | ExportSelection | Close => "File",
            Undo | Redo | Cut | Copy | CopyMerged | Paste | PasteNewDoc | Delete | FillFg | FillBg => "Edit",
            SelectAll | Deselect | InvertSelection | SaveSelectionMask => "Select",
            NewLayer | DuplicateLayer | DeleteLayer | MergeDown | GroupLayer | AddMask => "Layer",
            ImageSize | CanvasSize => "Image",
            ZoomIn | ZoomOut | ZoomFit | Zoom100 | RotateViewLeft | RotateViewRight | ResetView | ToggleNodeOutput => "View",
            SwapColors | ResetColors | PickScreenColor => "Color",
            Preferences => "App",
        }
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

    /// Keycap parts in platform order: mac `⌥ ⇧ ⌘ K` (Phosphor glyphs: the UI font lacks ⇧/⌥),
    /// elsewhere `Ctrl Alt Shift K`. Keys use their symbol where one exists (`[`, not "OpenBracket").
    pub fn parts(&self, mac: bool) -> Vec<String> {
        let mut v: Vec<String> = vec![];
        let mut push = |on: bool, s: &str| {
            if on {
                v.push(s.to_string());
            }
        };
        if mac {
            push(self.alt, egui_phosphor::regular::OPTION);
            push(self.shift, egui_phosphor::regular::ARROW_FAT_UP);
            push(self.command, egui_phosphor::regular::COMMAND);
        } else {
            push(self.command, "Ctrl");
            push(self.alt, "Alt");
            push(self.shift, "Shift");
        }
        v.push(self.key.symbol_or_name().to_string());
        v
    }

    /// Text form for menus and search (egui picks symbols the font can draw).
    pub fn label(&self, ctx: &egui::Context) -> String {
        ctx.format_shortcut(&self.shortcut()).replace(self.key.name(), self.key.symbol_or_name())
    }

    /// Does query token `t` (lowercase) name one of this binding's modifiers or its key?
    fn token_matches(&self, t: &str) -> bool {
        match t {
            "cmd" | "command" | "ctrl" | "control" | "ctl" | "super" | "meta" | "⌘" | "^" => self.command,
            "shift" | "⇧" => self.shift,
            "alt" | "option" | "opt" | "⌥" => self.alt,
            _ => t == self.key.name().to_lowercase() || t == self.key.symbol_or_name().to_lowercase(),
        }
    }
}

/// Fuzzy-ish, case-insensitive match of `q` against an action name and its shortcut
/// (`label` is the platform-formatted one, e.g. "⌘S" / "Ctrl+S").
pub fn shortcut_matches(name: &str, b: Option<Binding>, label: &str, q: &str) -> bool {
    let q = q.trim().to_lowercase();
    if q.is_empty() {
        return true;
    }
    let name = name.to_lowercase();
    let compact = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    let (cq, cn) = (compact(&q), compact(&name));
    // Name: every word is a substring, or the whole query is an in-order subsequence.
    if q.split_whitespace().all(|t| name.contains(t)) {
        return true;
    }
    let mut it = cn.chars();
    if cq.chars().count() >= 3 && cq.chars().all(|c| it.any(|n| n == c)) {
        return true;
    }
    let Some(b) = b else { return false };
    // Shortcut: formatted label ("⌘⇧s", "ctrl+s"), or every token is a modifier/key of the binding.
    let label = label.to_lowercase();
    if !label.is_empty() && (label.contains(&cq) || compact(&label.replace('+', "")) == compact(&cq.replace('+', ""))) {
        return true;
    }
    // "+" separates tokens unless it is the key itself ("cmd +").
    let toks: Vec<&str> = q.split(|c: char| c.is_whitespace() || c == '+').filter(|t| !t.is_empty()).collect();
    let toks = if toks.is_empty() { vec!["+"] } else { toks };
    toks.iter().all(|t| b.token_matches(t))
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
        add(Tool(ToolKind::Liquify), Binding::cmd_shift(Key::X));
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

    /// The action other than `except` currently bound to `b`.
    pub fn holder(&self, b: Binding, except: Action) -> Option<Action> {
        Action::all().into_iter().find(|a| *a != except && self.get(*a) == Some(b))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_action_has_a_listed_category() {
        for a in Action::all() {
            assert!(Action::CATEGORIES.contains(&a.category()), "{a:?}");
        }
    }

    #[test]
    fn search_by_name_and_shortcut() {
        let save = Some(Binding::cmd(Key::S));
        assert!(shortcut_matches("Save", save, "⌘S", "sav"));
        assert!(shortcut_matches("Tool: Brush", None, "", "BRUSH"));
        assert!(shortcut_matches("Save Selection Mask", None, "", "sel mask"));
        assert!(shortcut_matches("Save", save, "⌘S", "⌘s"));
        assert!(shortcut_matches("Save", save, "Ctrl+S", "ctrl+s"));
        assert!(shortcut_matches("Save", save, "⌘S", "cmd s"));
        assert!(shortcut_matches("Save", save, "⌘S", "Ctrl+S"));
        assert!(!shortcut_matches("Save", save, "⌘S", "shift s"));
        assert!(!shortcut_matches("Save", save, "⌘S", "brush"));
        assert!(!shortcut_matches("Open", None, "", "cmd o"));
        let zoom = Some(Binding::cmd(Key::Equals));
        assert!(shortcut_matches("Zoom In", zoom, "⌘=", "cmd ="));
        let rot = Some(Binding::shift(Key::OpenBracket));
        assert!(shortcut_matches("Rotate View Left", rot, "⇧[", "shift+["));
    }

    #[test]
    fn holder_finds_conflicts() {
        let k = Keymap::default();
        assert_eq!(k.holder(Binding::cmd(Key::S), Action::Open), Some(Action::Save));
        assert_eq!(k.holder(Binding::cmd(Key::S), Action::Save), None);
    }
}
