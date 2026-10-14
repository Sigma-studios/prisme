//! macOS implementations (AppKit / CoreGraphics via objc2).

#![allow(deprecated)] // CGWindowListCreateImage: still functional, ScreenCaptureKit is async-only.

use crate::pending::{Pending, pending_pair};
use objc2::MainThreadMarker;
use objc2_app_kit::{NSColor, NSColorSampler, NSColorSpace};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGDirectDisplayID, CGDisplayCopyColorSpace,
    CGEvent, CGEventSource, CGEventSourceStateID, CGGetDisplaysWithPoint, CGImageAlphaInfo,
    CGMainDisplayID, CGMouseButton, CGPreflightScreenCaptureAccess, CGRequestScreenCaptureAccess,
    CGWindowImageOption, CGWindowListCreateImage, CGWindowListOption, kCGColorSpaceSRGB,
    kCGNullWindowID,
};
use std::sync::Mutex;

// ---------------------------------------------------------------------------------------------
// NSColorSampler

fn ns_color_to_srgb8(color: &NSColor) -> Option<[u8; 3]> {
    let srgb = color.colorUsingColorSpace(&NSColorSpace::sRGBColorSpace())?;
    let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Some([
        q(srgb.redComponent()),
        q(srgb.greenComponent()),
        q(srgb.blueComponent()),
    ])
}

fn show_sampler(_mtm: MainThreadMarker, completer: crate::pending::Completer<Option<[u8; 3]>>) {
    let slot = Mutex::new(Some(completer));
    let handler = block2::RcBlock::new(move |color: *mut NSColor| {
        // SAFETY: AppKit passes either nil or a valid NSColor for the duration of the call.
        let rgb = unsafe { color.as_ref() }.and_then(ns_color_to_srgb8);
        if let Some(c) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
            c.complete(rgb);
        }
    });
    let sampler = NSColorSampler::new();
    // SAFETY: called on the main thread with a valid block; the sampler retains itself
    // until the session ends.
    unsafe { sampler.showSamplerWithSelectionHandler(&handler) };
}

pub fn pick_screen_color() -> Pending<Option<[u8; 3]>> {
    let (c, p) = pending_pair();
    if let Some(mtm) = MainThreadMarker::new() {
        show_sampler(mtm, c);
    } else {
        // Hop to the main queue (serviced by the winit/AppKit run loop).
        dispatch2::DispatchQueue::main().exec_async(move || {
            // SAFETY: blocks on the main queue run on the main thread.
            let mtm = unsafe { MainThreadMarker::new_unchecked() };
            show_sampler(mtm, c);
        });
    }
    p
}

// ---------------------------------------------------------------------------------------------
// Global cursor / buttons / pixels

/// Global cursor position in points, origin top-left of the main display.
pub fn global_cursor_position() -> Option<(i32, i32)> {
    let ev = CGEvent::new(None)?;
    let p = CGEvent::location(Some(&ev));
    Some((p.x.floor() as i32, p.y.floor() as i32))
}

pub fn global_mouse_down() -> Option<bool> {
    Some(CGEventSource::button_state(
        CGEventSourceStateID::CombinedSessionState,
        CGMouseButton::Left,
    ))
}

pub fn screen_capture_permission() -> bool {
    CGPreflightScreenCaptureAccess()
}

pub fn request_screen_capture_permission() -> bool {
    CGRequestScreenCaptureAccess()
}

fn srgb_space() -> Option<objc2_core_foundation::CFRetained<CGColorSpace>> {
    // SAFETY: reading an immutable framework constant.
    CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }))
}

/// Pixel at global point coordinates (same space as [`global_cursor_position`]).
pub fn screen_pixel_at(x: i32, y: i32) -> Option<[u8; 3]> {
    // Without the Screen Recording permission CoreGraphics silently returns only the desktop
    // picture and our own windows, which would give wrong colours: refuse instead.
    if !CGPreflightScreenCaptureAccess() {
        return None;
    }
    let rect = CGRect::new(CGPoint::new(x as f64, y as f64), CGSize::new(1.0, 1.0));
    let image = CGWindowListCreateImage(
        rect,
        CGWindowListOption::OptionOnScreenOnly,
        kCGNullWindowID,
        CGWindowImageOption::NominalResolution,
    )?;
    // Draw into a 1x1 sRGB bitmap: CoreGraphics colour-converts from the display profile.
    let space = srgb_space()?;
    let mut px = [0u8; 4];
    // SAFETY: `px` outlives the context, 1x1 RGBA8 with 4 bytes per row.
    let ctx = unsafe {
        CGBitmapContextCreate(
            px.as_mut_ptr().cast(),
            1,
            1,
            8,
            4,
            Some(&space),
            CGImageAlphaInfo::PremultipliedLast.0,
        )
    }?;
    CGContext::draw_image(
        Some(&ctx),
        CGRect::new(CGPoint::new(0.0, 0.0), CGSize::new(1.0, 1.0)),
        Some(&image),
    );
    drop(ctx);
    let a = px[3];
    if a == 0 {
        return None;
    }
    if a == 255 {
        Some([px[0], px[1], px[2]])
    } else {
        let un = |c: u8| ((c as u32 * 255 + a as u32 / 2) / a as u32).min(255) as u8;
        Some([un(px[0]), un(px[1]), un(px[2])])
    }
}

// ---------------------------------------------------------------------------------------------
// Display profile

fn icc_of_display(id: CGDirectDisplayID) -> Option<Vec<u8>> {
    let space = CGDisplayCopyColorSpace(id);
    let data = CGColorSpace::icc_data(Some(&space))?;
    let v = data.to_vec();
    (!v.is_empty()).then_some(v)
}

pub fn display_icc_profile() -> Option<Vec<u8>> {
    icc_of_display(CGMainDisplayID())
}

/// ICC profile of the display containing the global point (points, top-left origin).
pub fn display_icc_profile_at(x: i32, y: i32) -> Option<Vec<u8>> {
    let mut id: CGDirectDisplayID = 0;
    let mut n: u32 = 0;
    // SAFETY: valid out pointers for one display.
    let err =
        unsafe { CGGetDisplaysWithPoint(CGPoint::new(x as f64, y as f64), 1, &mut id, &mut n) };
    if err.0 != 0 || n == 0 {
        return display_icc_profile();
    }
    icc_of_display(id)
}

// ---------------------------------------------------------------------------------------------

pub fn open_url(url: &str) {
    if let Err(e) = std::process::Command::new("/usr/bin/open").arg(url).spawn() {
        log::warn!("open_url({url}): {e}");
    }
}

/// winit's app delegate doesn't implement `applicationShouldTerminate:`, so Cmd+Q / Dock ▸ Quit
/// terminate without a window-close event. Add the method to its class: it cancels the
/// termination while documents are unsaved (the app then shows its own quit prompt).
pub fn install_quit_guard() {
    use objc2::runtime::{AnyClass, AnyObject, Sel};
    unsafe extern "C-unwind" fn should_terminate(_this: *mut AnyObject, _sel: Sel, _sender: *mut AnyObject) -> usize {
        // NSTerminateNow = 1, NSTerminateCancel = 0.
        if crate::quit_allowed() { 1 } else { 0 }
    }
    let Some(cls) = AnyClass::get(c"WinitApplicationDelegate") else {
        log::warn!("prisme-platform: winit app delegate not found; Cmd+Q won't ask about unsaved changes");
        return;
    };
    unsafe {
        let imp: objc2::runtime::Imp = std::mem::transmute(
            should_terminate as unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject) -> usize,
        );
        // Fails (returns NO) if the method already exists, which is fine.
        objc2::ffi::class_addMethod(cls as *const AnyClass as *mut AnyClass, objc2::sel!(applicationShouldTerminate:), imp, c"Q@:@".as_ptr());
    }
}

pub fn request_app_terminate() {
    let Some(mtm) = MainThreadMarker::new() else { return };
    unsafe { objc2_app_kit::NSApplication::sharedApplication(mtm).terminate(None) };
}
