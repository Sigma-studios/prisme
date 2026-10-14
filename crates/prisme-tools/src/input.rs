#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub alt: bool,
    /// Cmd on macOS, Ctrl elsewhere.
    pub primary: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Down,
    Move,
    Up,
    /// Pointer moving with no button pressed.
    Hover,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PointerKind {
    #[default]
    Mouse,
    Pen,
    Eraser,
    Touch,
}

/// One pointer event in document space.
#[derive(Clone, Copy, Debug)]
pub struct ToolInput {
    pub phase: Phase,
    pub pos: [f32; 2],
    /// 0..1, mouse = 1.0.
    pub pressure: f32,
    pub tilt: f32,
    pub tilt_dir: f32,
    pub pointer: PointerKind,
    pub mods: Modifiers,
    /// Seconds.
    pub time: f64,
    /// Screen pixels per document pixel (for hit-test tolerances).
    pub zoom: f32,
    /// Secondary (right) button.
    pub secondary: bool,
}

impl ToolInput {
    pub fn mouse(phase: Phase, pos: [f32; 2]) -> Self {
        Self {
            phase,
            pos,
            pressure: 1.0,
            tilt: 0.0,
            tilt_dir: 0.0,
            pointer: PointerKind::Mouse,
            mods: Modifiers::default(),
            time: 0.0,
            zoom: 1.0,
            secondary: false,
        }
    }

    /// Hit tolerance in document pixels for a given screen-pixel radius.
    pub fn tol(&self, screen_px: f32) -> f32 {
        screen_px / self.zoom.max(1e-3)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolKey {
    Enter,
    Escape,
    Delete,
    Backspace,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Char(char),
    Text(String),
}
