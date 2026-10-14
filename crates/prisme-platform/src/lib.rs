//! # prisme-platform
//!
//! Every platform-specific behaviour of Prisme lives behind this crate, so the rest of the
//! application is platform-agnostic. Anything that may block returns a [`Pending`] that the
//! egui update loop polls once per frame (call [`set_wake_callback`] with
//! `ctx.request_repaint()` so the UI wakes up when results arrive). Nothing here blocks on the
//! web.
//!
//! ## Support matrix
//!
//! | Feature | macOS | Windows | Linux X11 | Linux Wayland | Web |
//! |---|---|---|---|---|---|
//! | [`spawn`] / [`spawn_async`] | thread | thread | thread | thread | microtask / `spawn_local` |
//! | [`spawn_worker`] (heavy CPU jobs) | thread | thread | thread | thread | Web Worker pool (fallback: microtask) |
//! | [`open_files`] | rfd (sheet) | rfd | rfd (portal/GTK-less) | rfd (portal) | rfd (`<input type=file>`) |
//! | [`save_file`] | rfd + write | rfd + write | rfd + write | rfd + write | `showSaveFilePicker` (Chromium), else `<a download>` |
//! | [`write_file_path`] | yes | yes | yes | yes | error (no paths) |
//! | clipboard text | arboard | arboard | arboard | arboard (XWayland) | async Clipboard API |
//! | clipboard image | arboard | arboard | arboard | arboard (XWayland) | `ClipboardItem` `image/png` |
//! | [`pick_screen_color`] | `NSColorSampler` | full-screen capture overlay | portal, else X11 pointer grab | portal (`xdg-portal` feature) | `EyeDropper` (Chromium only) |
//! | [`global_cursor_position`] / [`global_mouse_down`] | CGEvent | GetCursorPos / GetAsyncKeyState | x11rb `QueryPointer` | `None` | `None` |
//! | [`screen_pixel_at`] | CGWindowListCreateImage (needs Screen Recording permission, else `None`) | GetPixel | x11rb `GetImage` | `None` | `None` |
//! | [`display_icc_profile`] | CGDisplayCopyColorSpace | GetICMProfileW | `_ICC_PROFILE` root atom | `None` | `None` |
//! | [`display_is_wide_gamut_web`] | false | false | false | false | `(color-gamut: p3)` |
//! | storage (small) | `<data_dir>/prisme/` | `%APPDATA%\prisme\` | `$XDG_DATA_HOME/prisme/` | same | `localStorage` (base64) |
//! | storage (large) | file on thread | file on thread | file on thread | file on thread | OPFS (fallback `localStorage`) |
//! | [`open_url`] | `open` | ShellExecuteW | `xdg-open` | `xdg-open` | `window.open` |
//!
//! ## Threading caveats
//!
//! * Call the dialog / clipboard / picker functions from the UI thread. On the web several of
//!   them (`save_file`, `clipboard_set_image`, `pick_screen_color`, `open_files`) must run
//!   inside a user gesture (a click handler — in egui: the frame in which the button was
//!   clicked), which is why they start their work synchronously before returning.
//! * macOS: `NSColorSampler` must be shown on the main thread. [`pick_screen_color`] does this
//!   directly when called from the main thread (eframe's update loop runs there) and otherwise
//!   dispatches to the main queue, which requires a running AppKit run loop. rfd dialogs are
//!   attached as sheets to the main window.
//! * Windows: the picker runs its own overlay window + message loop on a worker thread.
//! * Linux: the portal call uses zbus' internal executor driven by `pollster` on a worker
//!   thread. The X11 primitives share one cached connection guarded by a mutex.

mod clipboard;
mod files;
mod pending;
mod storage;
mod worker;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(windows)]
mod windows;

pub use clipboard::{
    ClipboardImage, clipboard_get_image, clipboard_get_text, clipboard_set_image,
    clipboard_set_text, decode_png, encode_png,
};
pub use files::{
    FileFilter, PickedFile, open_files, save_file, save_file_deferred, write_file_path,
};
pub use pending::{
    Completer, MaybeSend, Pending, Wait, pending_pair, set_wake_callback, spawn, spawn_async,
};
pub use worker::{
    Buffer, JobControl, JobCtx, RETIRE_BYTES, Task, WorkerHandler, WorkerJob, WorkerMsg,
    run_via_messages, spawn_job, spawn_worker, worker_entry, worker_parallelism,
};
#[cfg(not(target_arch = "wasm32"))]
pub use storage::storage_dir;
pub use storage::{
    storage_delete, storage_read, storage_read_large, storage_write, storage_write_large,
};

#[cfg(not(any(
    target_os = "macos",
    windows,
    target_os = "linux",
    target_arch = "wasm32"
)))]
mod fallback {
    use crate::Pending;
    pub fn pick_screen_color() -> Pending<Option<[u8; 3]>> {
        Pending::ready(None)
    }
    pub fn open_url(_url: &str) {}
}

// ---------------------------------------------------------------------------------------------
// Screen colour picking

/// Whether [`pick_screen_color`] can work on this platform / browser.
pub fn screen_picker_supported() -> bool {
    #[cfg(any(target_os = "macos", windows))]
    {
        true
    }
    #[cfg(target_os = "linux")]
    {
        linux::screen_picker_supported()
    }
    #[cfg(target_arch = "wasm32")]
    {
        web_picker::supported()
    }
    #[cfg(not(any(
        target_os = "macos",
        windows,
        target_os = "linux",
        target_arch = "wasm32"
    )))]
    {
        false
    }
}

/// Let the user pick a colour anywhere on screen (outside the app window too).
///
/// Resolves to the sRGB-encoded 8-bit colour as displayed, or `None` if cancelled/unsupported.
/// Call from the UI thread (macOS: main thread preferred; web: inside a user gesture).
pub fn pick_screen_color() -> Pending<Option<[u8; 3]>> {
    #[cfg(target_os = "macos")]
    {
        macos::pick_screen_color()
    }
    #[cfg(windows)]
    {
        windows::pick_screen_color()
    }
    #[cfg(target_os = "linux")]
    {
        linux::pick_screen_color()
    }
    #[cfg(target_arch = "wasm32")]
    {
        web_picker::pick()
    }
    #[cfg(not(any(
        target_os = "macos",
        windows,
        target_os = "linux",
        target_arch = "wasm32"
    )))]
    {
        fallback::pick_screen_color()
    }
}

/// Global cursor position in desktop coordinates (macOS: points from the top-left of the main
/// display; Windows: virtual-screen pixels; X11: root-window pixels). `None` where unsupported
/// (Wayland, web).
pub fn global_cursor_position() -> Option<(i32, i32)> {
    #[cfg(target_os = "macos")]
    {
        macos::global_cursor_position()
    }
    #[cfg(windows)]
    {
        windows::global_cursor_position()
    }
    #[cfg(target_os = "linux")]
    {
        linux::global_cursor_position()
    }
    #[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
    {
        None
    }
}

/// Whether the primary mouse button is currently held, globally. `None` where unsupported.
pub fn global_mouse_down() -> Option<bool> {
    #[cfg(target_os = "macos")]
    {
        macos::global_mouse_down()
    }
    #[cfg(windows)]
    {
        windows::global_mouse_down()
    }
    #[cfg(target_os = "linux")]
    {
        linux::global_mouse_down()
    }
    #[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
    {
        None
    }
}

/// sRGB 8-bit colour of the screen pixel at desktop coordinates (same space as
/// [`global_cursor_position`]). Synchronous but cheap (one pixel). `None` if unsupported or
/// not permitted (macOS without the Screen Recording permission).
pub fn screen_pixel_at(x: i32, y: i32) -> Option<[u8; 3]> {
    #[cfg(target_os = "macos")]
    {
        macos::screen_pixel_at(x, y)
    }
    #[cfg(windows)]
    {
        windows::screen_pixel_at(x, y)
    }
    #[cfg(target_os = "linux")]
    {
        linux::screen_pixel_at(x, y)
    }
    #[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
    {
        let _ = (x, y);
        None
    }
}

/// macOS: whether the Screen Recording permission (needed by [`screen_pixel_at`]) is granted.
/// Always `true` elsewhere.
pub fn screen_capture_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::screen_capture_permission()
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// macOS: ask for the Screen Recording permission (shows the system prompt once; the app
/// usually needs a restart afterwards). Returns the current state. No-op elsewhere.
pub fn request_screen_capture_permission() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::request_screen_capture_permission()
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

// ---------------------------------------------------------------------------------------------
// Display colour

/// ICC profile bytes of the main display. `None` if unavailable (Wayland, web, errors).
pub fn display_icc_profile() -> Option<Vec<u8>> {
    #[cfg(target_os = "macos")]
    {
        macos::display_icc_profile()
    }
    #[cfg(windows)]
    {
        windows::display_icc_profile()
    }
    #[cfg(target_os = "linux")]
    {
        linux::display_icc_profile()
    }
    #[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
    {
        None
    }
}

/// ICC profile of the display containing the given desktop point (macOS: per display;
/// elsewhere: same as [`display_icc_profile`]).
pub fn display_icc_profile_at(x: i32, y: i32) -> Option<Vec<u8>> {
    #[cfg(target_os = "macos")]
    {
        macos::display_icc_profile_at(x, y)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (x, y);
        display_icc_profile()
    }
}

/// Web: whether the display covers (approximately) Display P3 (`(color-gamut: p3)`).
/// Always `false` on native (use [`display_icc_profile`] there).
pub fn display_is_wide_gamut_web() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        web::window()
            .and_then(|w| w.match_media("(color-gamut: p3)").ok().flatten())
            .is_some_and(|m| m.matches())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------------------------
// Misc

/// Monotonic seconds since the first call (web-time `Instant`; works on wasm).
pub fn now_seconds() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<web_time::Instant> = OnceLock::new();
    START
        .get_or_init(web_time::Instant::now)
        .elapsed()
        .as_secs_f64()
}

/// Tell the platform whether any document has unsaved changes. Web: while set, closing or
/// reloading the tab shows the browser's "Leave site?" prompt. Native: no-op (the app
/// intercepts window close itself).
pub fn set_unsaved_changes(dirty: bool) {
    #[cfg(target_arch = "wasm32")]
    web::set_unsaved_changes(dirty);
    #[cfg(not(target_arch = "wasm32"))]
    let _ = dirty;
}

/// Open a URL in the default browser (web: a new tab).
pub fn open_url(url: &str) {
    #[cfg(target_os = "macos")]
    macos::open_url(url);
    #[cfg(windows)]
    windows::open_url(url);
    #[cfg(target_os = "linux")]
    linux::open_url(url);
    #[cfg(target_arch = "wasm32")]
    if let Some(w) = web::window() {
        let _ = w.open_with_url_and_target(url, "_blank");
    }
    #[cfg(not(any(
        target_os = "macos",
        windows,
        target_os = "linux",
        target_arch = "wasm32"
    )))]
    fallback::open_url(url);
}

/// `"macOS"`, `"Windows"`, `"Linux"`, `"Web"` (or `"Unknown"`).
pub fn platform_name() -> &'static str {
    if cfg!(target_arch = "wasm32") {
        "Web"
    } else if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(windows) {
        "Windows"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else {
        "Unknown"
    }
}

/// Whether the user is on a Mac (native macOS, or a browser on macOS / iOS).
pub fn is_mac() -> bool {
    #[cfg(target_os = "macos")]
    {
        true
    }
    #[cfg(target_arch = "wasm32")]
    {
        use std::sync::OnceLock;
        static MAC: OnceLock<bool> = OnceLock::new();
        *MAC.get_or_init(|| {
            web::window()
                .map(|w| {
                    let nav = w.navigator();
                    let s = format!(
                        "{} {}",
                        nav.platform().unwrap_or_default(),
                        nav.user_agent().unwrap_or_default()
                    );
                    s.contains("Mac") || s.contains("iPhone") || s.contains("iPad")
                })
                .unwrap_or(false)
        })
    }
    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    {
        false
    }
}

/// Label of the primary shortcut modifier: `"⌘"` on Macs (native or browser), else `"Ctrl"`.
pub fn primary_modifier_name() -> &'static str {
    if is_mac() { "⌘" } else { "Ctrl" }
}

// ---------------------------------------------------------------------------------------------
// Web EyeDropper

#[cfg(target_arch = "wasm32")]
mod web_picker {
    use crate::pending::{Pending, pending_pair};
    use crate::web::*;
    use wasm_bindgen::{JsCast, JsValue};

    pub fn supported() -> bool {
        window().is_some_and(|w| get(&w, "EyeDropper").is_some())
    }

    pub(crate) fn parse_css_color(s: &str) -> Option<[u8; 3]> {
        let s = s.trim();
        if let Some(hex) = s.strip_prefix('#') {
            let h = |i: usize| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok();
            return match hex.len() {
                6 | 8 => Some([h(0)?, h(2)?, h(4)?]),
                3 | 4 => {
                    let d = |i: usize| {
                        u8::from_str_radix(hex.get(i..i + 1)?, 16)
                            .ok()
                            .map(|v| v * 17)
                    };
                    Some([d(0)?, d(1)?, d(2)?])
                }
                _ => None,
            };
        }
        let inner = s
            .strip_prefix("rgb(")
            .or_else(|| s.strip_prefix("rgba("))?
            .strip_suffix(')')?;
        let mut it = inner.split([',', ' ', '/']).filter(|p| !p.is_empty());
        let mut c = [0u8; 3];
        for v in &mut c {
            *v = it
                .next()?
                .trim()
                .parse::<f32>()
                .ok()?
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        Some(c)
    }

    pub fn pick() -> Pending<Option<[u8; 3]>> {
        let Some(win) = window() else {
            return Pending::ready(None);
        };
        // Construct + open synchronously: `open()` requires transient user activation.
        let promise = (|| -> Result<JsValue, JsValue> {
            let ctor: js_sys::Function =
                get(&win, "EyeDropper").ok_or("no EyeDropper")?.dyn_into()?;
            let dropper = js_sys::Reflect::construct(&ctor, &js_sys::Array::new())?;
            call_method(&dropper, "open", &[])
        })();
        let promise = match promise {
            Ok(p) => p,
            Err(e) => {
                log::info!("EyeDropper unavailable: {}", err_string(&e));
                return Pending::ready(None);
            }
        };
        let (c, p) = pending_pair();
        wasm_bindgen_futures::spawn_local(async move {
            let r = match await_promise(promise).await {
                Ok(res) => get(&res, "sRGBHex")
                    .and_then(|v| v.as_string())
                    .and_then(|s| parse_css_color(&s)),
                Err(e) => {
                    if !is_abort(&e) {
                        log::warn!("EyeDropper: {}", err_string(&e));
                    }
                    None
                }
            };
            c.complete(r);
        });
        p
    }
}

#[cfg(test)]
mod tests;
