//! Linux implementations: X11 via x11rb (pure Rust), colour picking via xdg-desktop-portal.
//!
//! Under a Wayland session the X11 primitives return `None` (XWayland only sees X clients,
//! and Wayland deliberately exposes no global cursor/pixels); the portal still works there.

use crate::pending::{Pending, spawn};
use std::sync::{Mutex, OnceLock};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{self, ConnectionExt as _};
use x11rb::rust_connection::RustConnection;

fn is_wayland_session() -> bool {
    std::env::var("XDG_SESSION_TYPE")
        .map(|s| s.eq_ignore_ascii_case("wayland"))
        .unwrap_or(false)
        || (std::env::var_os("WAYLAND_DISPLAY").is_some() && std::env::var_os("DISPLAY").is_none())
}

struct X {
    conn: RustConnection,
    screen: usize,
}

fn x11() -> Option<&'static Mutex<X>> {
    static X11: OnceLock<Option<Mutex<X>>> = OnceLock::new();
    X11.get_or_init(|| {
        if is_wayland_session() || std::env::var_os("DISPLAY").is_none() {
            return None;
        }
        match x11rb::connect(None) {
            Ok((conn, screen)) => Some(Mutex::new(X { conn, screen })),
            Err(e) => {
                log::info!("prisme-platform: no X11 connection: {e}");
                None
            }
        }
    })
    .as_ref()
}

fn with_x<R>(f: impl FnOnce(&X) -> Option<R>) -> Option<R> {
    let m = x11()?;
    let g = m.lock().unwrap_or_else(|e| e.into_inner());
    f(&g)
}

fn root_of(x: &X) -> Option<&xproto::Screen> {
    x.conn.setup().roots.get(x.screen)
}

pub fn x11_available() -> bool {
    x11().is_some()
}

pub fn global_cursor_position() -> Option<(i32, i32)> {
    with_x(|x| {
        let root = root_of(x)?.root;
        let r = x.conn.query_pointer(root).ok()?.reply().ok()?;
        Some((r.root_x as i32, r.root_y as i32))
    })
}

pub fn global_mouse_down() -> Option<bool> {
    with_x(|x| {
        let root = root_of(x)?.root;
        let r = x.conn.query_pointer(root).ok()?.reply().ok()?;
        Some(u16::from(r.mask) & u16::from(xproto::KeyButMask::BUTTON1) != 0)
    })
}

fn pixel_at(x: &X, px: i32, py: i32) -> Option<[u8; 3]> {
    let setup = x.conn.setup();
    let screen = root_of(x)?;
    if px < 0
        || py < 0
        || px >= screen.width_in_pixels as i32
        || py >= screen.height_in_pixels as i32
    {
        return None;
    }
    let img = x
        .conn
        .get_image(
            xproto::ImageFormat::Z_PIXMAP,
            screen.root,
            px as i16,
            py as i16,
            1,
            1,
            !0,
        )
        .ok()?
        .reply()
        .ok()?;
    let bpp = setup
        .pixmap_formats
        .iter()
        .find(|f| f.depth == img.depth)
        .map(|f| f.bits_per_pixel)
        .unwrap_or(32);
    let nbytes = (bpp as usize).div_ceil(8);
    if img.data.len() < nbytes || nbytes == 0 || nbytes > 4 {
        return None;
    }
    let bytes = &img.data[..nbytes];
    let mut v: u32 = 0;
    if setup.image_byte_order == xproto::ImageOrder::LSB_FIRST {
        for (i, b) in bytes.iter().enumerate() {
            v |= (*b as u32) << (8 * i);
        }
    } else {
        for b in bytes {
            v = (v << 8) | *b as u32;
        }
    }
    let vis = screen
        .allowed_depths
        .iter()
        .flat_map(|d| d.visuals.iter())
        .find(|vt| vt.visual_id == img.visual)
        .or_else(|| {
            screen
                .allowed_depths
                .iter()
                .flat_map(|d| d.visuals.iter())
                .find(|vt| vt.visual_id == screen.root_visual)
        })?;
    let chan = |mask: u32| -> u8 {
        if mask == 0 {
            return 0;
        }
        let shift = mask.trailing_zeros();
        let max = mask >> shift;
        let c = (v & mask) >> shift;
        ((c * 255 + max / 2) / max) as u8
    };
    Some([
        chan(vis.red_mask),
        chan(vis.green_mask),
        chan(vis.blue_mask),
    ])
}

pub fn screen_pixel_at(px: i32, py: i32) -> Option<[u8; 3]> {
    with_x(|x| pixel_at(x, px, py))
}

pub fn display_icc_profile() -> Option<Vec<u8>> {
    with_x(|x| {
        let root = root_of(x)?.root;
        // ICC Profiles in X Specification: `_ICC_PROFILE` on the root window describes the
        // first monitor (`_ICC_PROFILE_n` the others).
        let atom = x
            .conn
            .intern_atom(true, b"_ICC_PROFILE")
            .ok()?
            .reply()
            .ok()?
            .atom;
        if atom == x11rb::NONE {
            return None;
        }
        let r = x
            .conn
            .get_property(
                false,
                root,
                atom,
                xproto::AtomEnum::ANY,
                0,
                64 * 1024 * 1024 / 4,
            )
            .ok()?
            .reply()
            .ok()?;
        (r.format == 8 && !r.value.is_empty()).then_some(r.value)
    })
}

/// X11 picker: grab the pointer with a crosshair cursor, read the pixel under the next click.
/// Uses its own connection so that blocking on events does not hold the shared one.
fn x11_pick() -> Option<[u8; 3]> {
    x11()?;
    let (conn, screen_num) = x11rb::connect(None).ok()?;
    let screen = conn.setup().roots.get(screen_num)?.clone();
    let font = conn.generate_id().ok()?;
    conn.open_font(font, b"cursor").ok()?;
    let cursor = conn.generate_id().ok()?;
    // XC_crosshair = 34
    conn.create_glyph_cursor(cursor, font, font, 34, 35, 0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF)
        .ok()?;
    let grab = conn
        .grab_pointer(
            false,
            screen.root,
            xproto::EventMask::BUTTON_PRESS,
            xproto::GrabMode::ASYNC,
            xproto::GrabMode::ASYNC,
            x11rb::NONE,
            cursor,
            x11rb::CURRENT_TIME,
        )
        .ok()?
        .reply()
        .ok()?;
    if grab.status != xproto::GrabStatus::SUCCESS {
        return None;
    }
    let _ = conn.grab_keyboard(
        false,
        screen.root,
        x11rb::CURRENT_TIME,
        xproto::GrabMode::ASYNC,
        xproto::GrabMode::ASYNC,
    );
    conn.flush().ok()?;
    let result = loop {
        match conn.wait_for_event() {
            Ok(x11rb::protocol::Event::ButtonPress(ev)) => {
                if ev.detail == 1 {
                    let x = X {
                        conn,
                        screen: screen_num,
                    };
                    let rgb = {
                        let _ = x.conn.ungrab_pointer(x11rb::CURRENT_TIME);
                        let _ = x.conn.ungrab_keyboard(x11rb::CURRENT_TIME);
                        let _ = x.conn.flush();
                        pixel_at(&x, ev.root_x as i32, ev.root_y as i32)
                    };
                    return rgb;
                } else {
                    break None;
                }
            }
            Ok(x11rb::protocol::Event::KeyPress(_)) => break None, // any key (Esc) cancels
            Ok(_) => continue,
            Err(_) => return None,
        }
    };
    let _ = conn.ungrab_pointer(x11rb::CURRENT_TIME);
    let _ = conn.ungrab_keyboard(x11rb::CURRENT_TIME);
    let _ = conn.free_cursor(cursor);
    let _ = conn.close_font(font);
    let _ = conn.flush();
    result
}

#[cfg(feature = "xdg-portal")]
fn portal_pick() -> Result<Option<[u8; 3]>, String> {
    pollster::block_on(async {
        let req = ashpd::desktop::Color::pick()
            .send()
            .await
            .map_err(|e| e.to_string())?;
        match req.response() {
            Ok(c) => {
                let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                Ok(Some([q(c.red()), q(c.green()), q(c.blue())]))
            }
            Err(ashpd::Error::Response(ashpd::desktop::ResponseError::Cancelled)) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    })
}

pub fn screen_picker_supported() -> bool {
    cfg!(feature = "xdg-portal") || x11_available()
}

pub fn pick_screen_color() -> Pending<Option<[u8; 3]>> {
    spawn(|| {
        #[cfg(feature = "xdg-portal")]
        match portal_pick() {
            Ok(r) => return r,
            Err(e) => log::info!("portal colour picker unavailable ({e}); trying X11"),
        }
        x11_pick()
    })
}

pub fn open_url(url: &str) {
    if let Err(e) = std::process::Command::new("xdg-open").arg(url).spawn() {
        log::warn!("open_url({url}): {e}");
    }
}
