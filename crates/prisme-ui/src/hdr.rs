//! HDR / extended-range display output.
//!
//! **Surface.** egui-wgpu is patched (see `vendor/README.md`) so that the app can pick the
//! surface format and colour space. On platforms with an *encoded* extended-range sRGB
//! colour space we create an `Rgba16Float` surface once at startup and then switch its colour
//! space at runtime between:
//! * `Srgb` — SDR, identical to the classic 8-bit path (values in [0, 1], ICC display LUT,
//!   dithering), and
//! * `ExtendedSrgb` — HDR: sRGB-encoded, 1.0 = SDR reference white, values above 1.0 are
//!   brighter than white up to the display's headroom, negatives encode colours outside sRGB.
//!
//! The *encoded* (not linear scRGB) variant is used on purpose: egui's shader writes
//! sRGB-encoded UI colours to any non-`*Srgb` format, which is exactly what `ExtendedSrgb`
//! expects, so UI colours and blending stay identical and UI white stays at SDR white.
//!
//! | Platform | Result |
//! |---|---|
//! | macOS (Metal) | `Rgba16Float` + `Srgb` ⇄ `ExtendedSrgb` (EDR); headroom from `NSScreen` |
//! | Web, Chromium 129+ (fp16 canvas) | same, via canvas `toneMapping: "extended"`; headroom unknown → user setting |
//! | Web, other browsers | SDR (no fp16 canvas → no `Rgba16Float`); browsers that accept fp16 but ignore `toneMapping` clip at SDR white |
//! | Windows | SDR. Only scRGB (linear) is offered there; egui would need a linear/white-level-scaled UI shader. |
//! | Linux / others | SDR |
//!
//! `PRISME_HDR=off` (native) keeps the classic 8-bit surface.

use crate::app::AppState;
use egui_wgpu::wgpu;
use prisme_gpu::{DisplayOutput, HighlightMode};
use serde::{Deserialize, Serialize};

/// User preference for HDR output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HdrMode {
    /// HDR when the surface supports it and the display reports HDR capability.
    #[default]
    Auto,
    /// HDR whenever the surface supports it.
    On,
    /// Always SDR.
    Off,
}

impl HdrMode {
    pub const ALL: [HdrMode; 3] = [HdrMode::Auto, HdrMode::On, HdrMode::Off];

    pub fn label(self) -> &'static str {
        match self {
            HdrMode::Auto => "Auto",
            HdrMode::On => "On",
            HdrMode::Off => "Off",
        }
    }
}

/// `WgpuConfiguration::target_format_selector`: an `Rgba16Float` surface (initially SDR) when
/// it can be switched to encoded extended-range sRGB at runtime; otherwise egui's default.
pub fn select_surface(adapter: &wgpu::Adapter, surface: &wgpu::Surface<'_>) -> Option<(wgpu::TextureFormat, wgpu::SurfaceColorSpace)> {
    use wgpu::{SurfaceColorSpace, SurfaceColorSpaces, TextureFormat};
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var("PRISME_HDR").is_ok_and(|v| v.eq_ignore_ascii_case("off")) {
        log::info!("HDR: disabled by PRISME_HDR=off; using the default SDR surface");
        return None;
    }
    let caps = surface.get_capabilities(adapter);
    let spaces = caps.color_spaces(TextureFormat::Rgba16Float);
    log::info!("HDR: surface formats {:?}; Rgba16Float colour spaces {spaces:?}", caps.formats);
    if !cfg!(any(target_os = "macos", target_arch = "wasm32")) {
        log::info!("HDR: extended-range output not enabled on this platform; using the default SDR surface");
        return None;
    }
    if spaces.contains(SurfaceColorSpaces::SRGB | SurfaceColorSpaces::EXTENDED_SRGB) {
        log::info!("HDR: using an Rgba16Float surface (Srgb <-> ExtendedSrgb switchable at runtime)");
        Some((TextureFormat::Rgba16Float, SurfaceColorSpace::Srgb))
    } else {
        log::info!("HDR: no switchable extended-range sRGB colour space; using the default SDR surface");
        None
    }
}

/// Snapshot of the HDR state for the UI and the display pass.
#[derive(Clone, Debug, Default)]
pub struct HdrStatus {
    /// The surface can present extended range (HDR can be turned on at runtime).
    pub capable: bool,
    /// Why HDR is unavailable or inactive (empty when active).
    pub reason: String,
    /// The surface is currently configured for extended range.
    pub active: bool,
    /// Current headroom reported by the OS (× SDR white), if any.
    pub headroom: Option<f32>,
    /// Best-case headroom (macOS), if reported.
    pub potential: Option<f32>,
    /// Display reports HDR capability (macOS: potential headroom > 1; web: `dynamic-range: high`).
    pub display_hdr: Option<bool>,
    /// Headroom used by the display pass (reported value or the user's fallback).
    pub peak: f32,
    pub format: Option<wgpu::TextureFormat>,
    pub color_space: Option<wgpu::SurfaceColorSpace>,
}

impl HdrStatus {
    pub fn output(&self, highlights: HighlightMode) -> DisplayOutput {
        if self.active { DisplayOutput::ExtendedSrgb { peak: self.peak, highlights } } else { DisplayOutput::Sdr }
    }

    /// Linear range for the extended scopes: the next power of two ≥ the peak.
    pub fn scope_range(&self) -> f32 {
        if self.active { self.peak.max(1.0).log2().ceil().exp2().clamp(1.0, 16.0) } else { 1.0 }
    }
}

/// Per-frame update: decide the requested surface colour space from the settings and the
/// display, and refresh [`AppState::hdr`]. Logs transitions.
pub fn update(s: &mut AppState, ctx: &egui::Context) {
    let Some(g) = &s.gpu else {
        s.hdr = HdrStatus { reason: "GPU unavailable".into(), peak: 1.0, ..Default::default() };
        return;
    };
    let format = g.rs.target_format;
    let mut st = g.rs.surface_color.write();
    let ext = wgpu::SurfaceColorSpaces::EXTENDED_SRGB;
    let capable = format == wgpu::TextureFormat::Rgba16Float && st.supported.contains(ext | wgpu::SurfaceColorSpaces::SRGB);
    let d = st.display.clone();
    let headroom = d.headroom.and_then(|h| h.current).filter(|h| h.is_finite());
    let potential = d.headroom.and_then(|h| h.potential).filter(|h| h.is_finite());
    let display_hdr = potential.map(|p| p > 1.0).or(d.coarse.and_then(|c| c.high_dynamic_range));
    let want = capable
        && match s.settings.hdr_mode {
            HdrMode::Off => false,
            HdrMode::On => true,
            HdrMode::Auto => display_hdr == Some(true),
        };
    if capable {
        st.requested = if want { wgpu::SurfaceColorSpace::ExtendedSrgb } else { wgpu::SurfaceColorSpace::Srgb };
    }
    let active = st.configured.is_hdr();
    let peak = match d.tone_map_headroom() {
        Some(h) if h.is_finite() => h,
        _ => s.settings.hdr_peak_fallback,
    }
    .clamp(1.0, 64.0);
    let reason = if !capable {
        if cfg!(target_arch = "wasm32") {
            "This browser offers no extended-range (fp16 + toneMapping \"extended\") canvas".to_string()
        } else if cfg!(target_os = "macos") {
            "The window surface offers no extended-range colour space".to_string()
        } else {
            "HDR output is implemented for macOS (EDR) and WebGPU browsers only".to_string()
        }
    } else if active {
        String::new()
    } else if s.settings.hdr_mode == HdrMode::Off {
        "Turned off".into()
    } else if want {
        "Switching…".into()
    } else {
        "The display doesn't report HDR capability (choose On to force)".into()
    };
    let status = HdrStatus {
        capable,
        reason,
        active,
        headroom,
        potential,
        display_hdr,
        peak,
        format: Some(format),
        color_space: Some(st.configured),
    };
    drop(st);
    // Log changes of the surface state / headroom (not every frame).
    let prev = &s.hdr;
    let changed = prev.format != status.format
        || prev.color_space != status.color_space
        || prev.capable != status.capable
        || (prev.peak - status.peak).abs() > 0.05 * status.peak
        || prev.potential != status.potential;
    if changed {
        log::info!(
            "HDR: surface {:?} / {:?}, capable={}, active={}, display headroom current={:?} potential={:?} display_hdr={:?}, peak used={:.2}{}",
            status.format.unwrap_or(format),
            status.color_space.unwrap_or_default(),
            status.capable,
            status.active,
            status.headroom,
            status.potential,
            status.display_hdr,
            status.peak,
            if status.reason.is_empty() { String::new() } else { format!(" ({})", status.reason) }
        );
    }
    if status.active || (capable && s.settings.hdr_mode == HdrMode::Auto) {
        // The painter refreshes the display info only while painting: keep a slow tick so
        // brightness / display changes are picked up while idle.
        ctx.request_repaint_after(std::time::Duration::from_secs(1));
    }
    if want != status.active {
        // A colour-space switch is applied on the next paint.
        ctx.request_repaint();
    }
    s.hdr = status;
}
