//! This crates provides bindings between [`egui`](https://github.com/emilk/egui) and [wgpu](https://crates.io/crates/wgpu).
//!
//! If you're targeting WebGL you also need to turn on the
//! `webgl` feature of the `wgpu` crate:
//!
//! ```toml
//! # Enable both WebGL and WebGPU backends on web.
//! wgpu = { version = "*", features = ["webgpu", "webgl"] }
//! ```
//!
//! You can control whether WebGL or WebGPU will be picked at runtime by configuring
//! [`WgpuConfiguration::wgpu_setup`].
//! The default is to prefer WebGPU and fall back on WebGL.
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
//!

pub use wgpu;

/// Low-level painting of [`egui`](https://github.com/emilk/egui) on [`wgpu`].
mod renderer;

mod setup;

pub use renderer::*;
pub use setup::{
    EguiDisplayHandle, NativeAdapterSelectorMethod, WgpuSetup, WgpuSetupCreateNew,
    WgpuSetupExisting,
};

/// Helpers for capturing screenshots of the UI.
#[cfg(feature = "capture")]
pub mod capture;

/// Module for painting [`egui`](https://github.com/emilk/egui) with [`wgpu`] on [`winit`].
#[cfg(feature = "winit")]
pub mod winit;

use std::sync::Arc;

use epaint::mutex::RwLock;

/// An error produced by egui-wgpu.
#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error(transparent)]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),

    #[error("Adapter selection failed: {0}")]
    CustomNativeAdapterSelectionError(String),

    #[error("There was no valid format for the surface at all.")]
    NoSurfaceFormatsAvailable,

    #[error(transparent)]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),

    #[error(transparent)]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),

    #[cfg(feature = "winit")]
    #[error(transparent)]
    HandleError(#[from] ::winit::raw_window_handle::HandleError),
}

/// Runtime-mutable subset of [`WgpuConfiguration`].
///
/// Edit any field to have the surface reconfigured on the next paint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceConfig {
    /// Present mode used for the primary surface.
    pub present_mode: wgpu::PresentMode,

    /// Desired maximum number of frames that the presentation engine should queue in advance.
    ///
    /// Use `1` for low-latency, and `2` for high-throughput.
    ///
    /// See [`wgpu::SurfaceConfiguration::desired_maximum_frame_latency`] for details.
    ///
    /// `None` => Let `wgpu` pick a default (currently `2`).
    pub desired_maximum_frame_latency: Option<u32>,
}

impl SurfaceConfig {
    /// Good default for GUIs with very little (or no) extra GPU work.
    pub const LOW_LATENCY: Self = Self {
        present_mode: wgpu::PresentMode::AutoVsync,

        desired_maximum_frame_latency: if cfg!(target_os = "ios") {
            None // The default is good on iOS, while `Some(1)` cuts FPS in half
        } else {
            Some(1)
        },
    };

    /// Good default for GUIs with a lot of extra GPU work,
    /// or that want to prioritize smoothness over latency.
    pub const HIGH_THROUGHPUT: Self = Self {
        present_mode: wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: Some(2), // High-throughput.
    };
}

/// Access to the render state for egui.
#[derive(Clone)]
pub struct RenderState {
    /// Wgpu adapter used for rendering.
    pub adapter: wgpu::Adapter,

    /// All the available adapters.
    ///
    /// This is not available on web.
    /// On web, we always select WebGPU is available, then fall back to WebGL if not.
    #[cfg(not(target_arch = "wasm32"))]
    pub available_adapters: Vec<wgpu::Adapter>,

    /// Wgpu instance used for creating surfaces and adapters.
    pub instance: wgpu::Instance,

    /// Wgpu device used for rendering, created from the adapter.
    pub device: wgpu::Device,

    /// Wgpu queue used for rendering, created from the adapter.
    pub queue: wgpu::Queue,

    /// The target texture format used for presenting to the window.
    pub target_format: wgpu::TextureFormat,

    /// Egui renderer responsible for drawing the UI.
    pub renderer: Arc<RwLock<Renderer>>,

    /// Runtime-mutable subset of the wgpu configuration.
    ///
    /// Update this to have the surface reconfigured on the next paint.
    pub surface_config: SurfaceConfig,

    /// PRISME PATCH: colour space of the presentation surface and HDR info of the display
    /// it is on. Shared by every clone of this [`RenderState`] (unlike `surface_config`,
    /// whose clones in `eframe::Frame` never reach the painter), so the app can request a
    /// different colour space at runtime. See [`SurfaceColorState`].
    pub surface_color: Arc<RwLock<SurfaceColorState>>,
}

async fn request_adapter(
    instance: &wgpu::Instance,
    power_preference: wgpu::PowerPreference,
    compatible_surface: Option<&wgpu::Surface<'_>>,
    available_adapters: &[wgpu::Adapter],
) -> Result<wgpu::Adapter, WgpuError> {
    profiling::function_scope!();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            compatible_surface,
            // We don't expose this as an option right now since it's fairly rarely useful:
            // * only has an effect on native
            // * fails if there's no software rasterizer available
            // * can achieve the same with `native_adapter_selector`
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .inspect_err(|_err| {
            if cfg!(target_arch = "wasm32") {
                // Nothing to add here
            } else if available_adapters.is_empty() {
                if std::env::var("DYLD_LIBRARY_PATH").is_ok() {
                    // DYLD_LIBRARY_PATH can sometimes lead to loading dylibs that cause
                    // us to find zero adapters. Very strange.
                    // I don't want to debug this again.
                    // See https://github.com/rerun-io/rerun/issues/11351 for more
                    log::warn!(
                        "No wgpu adapter found. This could be because DYLD_LIBRARY_PATH causes dylibs to be loaded that interfere with Metal device creation. Try restarting with DYLD_LIBRARY_PATH=''"
                    );
                } else {
                    log::info!("No wgpu adapter found");
                }
            } else if available_adapters.len() == 1 {
                log::info!(
                    "The only available wgpu adapter was not suitable: {}",
                    adapter_info_summary(&available_adapters[0].get_info())
                );
            } else {
                log::info!(
                    "No suitable wgpu adapter found out of the {} available ones: {}",
                    available_adapters.len(),
                    describe_adapters(available_adapters)
                );
            }
        })?;

    if 1 < available_adapters.len() {
        log::info!(
            "There are {} available wgpu adapters: {}",
            available_adapters.len(),
            describe_adapters(available_adapters)
        );
    }

    Ok(adapter)
}

impl RenderState {
    /// Creates a new [`RenderState`], containing everything needed for drawing egui with wgpu.
    ///
    /// # Errors
    /// Wgpu initialization may fail due to incompatible hardware or driver for a given config.
    pub async fn create(
        config: &WgpuConfiguration,
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'static>>,
        options: RendererOptions,
    ) -> Result<Self, WgpuError> {
        profiling::scope!("RenderState::create"); // async yield give bad names using `profile_function`

        // This is always an empty list on web.
        #[cfg(not(target_arch = "wasm32"))]
        let available_adapters = {
            let backends = if let WgpuSetup::CreateNew(create_new) = &config.wgpu_setup {
                create_new.instance_descriptor.backends
            } else {
                wgpu::Backends::all()
            };

            instance.enumerate_adapters(backends).await
        };

        let (instance, adapter, device, queue) = match config.wgpu_setup.clone() {
            WgpuSetup::CreateNew(WgpuSetupCreateNew {
                instance_descriptor: _,
                display_handle: _,
                power_preference,
                native_adapter_selector: _native_adapter_selector,
                device_descriptor,
            }) => {
                let adapter = {
                    #[cfg(target_arch = "wasm32")]
                    {
                        request_adapter(instance, power_preference, compatible_surface, &[]).await
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(native_adapter_selector) = _native_adapter_selector {
                        native_adapter_selector(&available_adapters, compatible_surface)
                            .map_err(WgpuError::CustomNativeAdapterSelectionError)
                    } else {
                        request_adapter(
                            instance,
                            power_preference,
                            compatible_surface,
                            &available_adapters,
                        )
                        .await
                    }
                }?;

                let (device, queue) = {
                    profiling::scope!("request_device");
                    adapter
                        .request_device(&(*device_descriptor)(&adapter))
                        .await?
                };

                (instance.clone(), adapter, device, queue)
            }
            WgpuSetup::Existing(WgpuSetupExisting {
                instance,
                adapter,
                device,
                queue,
            }) => (instance, adapter, device, queue),
        };

        log_adapter_info(&adapter.get_info());

        let surface_formats = {
            profiling::scope!("get_capabilities");
            compatible_surface.map_or_else(
                || vec![wgpu::TextureFormat::Rgba8Unorm],
                |s| s.get_capabilities(&adapter).formats,
            )
        };
        let mut target_format = crate::preferred_framebuffer_format(&surface_formats)?;

        // PRISME PATCH: let the application pick the surface format + colour space (e.g. an
        // `Rgba16Float` extended-range surface for HDR). Rejected if unsupported.
        let mut surface_color = SurfaceColorState::default();
        if let Some(surface) = compatible_surface {
            if let Some(select) = &config.target_format_selector
                && let Some((format, color_space)) = select(&adapter, surface)
            {
                let supported = surface.get_capabilities(&adapter).color_spaces(format);
                if color_space
                    .to_color_spaces()
                    .is_some_and(|flag| supported.contains(flag))
                {
                    target_format = format;
                    surface_color.requested = color_space;
                    surface_color.configured = color_space;
                } else {
                    log::warn!(
                        "Requested surface {format:?} / {color_space:?} is unsupported; using {target_format:?}"
                    );
                }
            }
            surface_color.supported = surface.get_capabilities(&adapter).color_spaces(target_format);
        }

        let renderer = Renderer::new(&device, target_format, options);

        // On wasm, depending on feature flags, wgpu objects may or may not implement sync.
        // It doesn't make sense to switch to Rc for that special usecase, so simply disable the lint.
        #[allow(clippy::allow_attributes, clippy::arc_with_non_send_sync)] // For wasm
        Ok(Self {
            instance,
            adapter,
            #[cfg(not(target_arch = "wasm32"))]
            available_adapters,
            device,
            queue,
            target_format,
            renderer: Arc::new(RwLock::new(renderer)),
            surface_config: config.surface,
            surface_color: Arc::new(RwLock::new(surface_color)),
        })
    }
}

fn describe_adapters(adapters: &[wgpu::Adapter]) -> String {
    if adapters.is_empty() {
        "(none)".to_owned()
    } else if adapters.len() == 1 {
        adapter_info_summary(&adapters[0].get_info())
    } else {
        adapters
            .iter()
            .map(|a| format!("{{{}}}", adapter_info_summary(&a.get_info())))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Specifies which action should be taken as consequence of a surface error.
pub enum SurfaceErrorAction {
    /// Do nothing and skip the current frame.
    SkipFrame,

    /// Reconfigure the existing surface, then skip the current frame.
    ///
    /// Calls [`wgpu::Surface::configure`] on the current surface object.
    /// Use for [`wgpu::CurrentSurfaceTexture::Outdated`].
    Reconfigure,

    /// Drop the surface, create a new one via [`wgpu::Instance::create_surface`], configure it,
    /// then skip the current frame.
    ///
    /// Use for [`wgpu::CurrentSurfaceTexture::Lost`], where reconfiguring the same surface
    /// object cannot recover.
    RecreateSurface,
}

/// Configuration for using wgpu with eframe or the egui-wgpu winit feature.
#[derive(Clone)]
pub struct WgpuConfiguration {
    /// Runtime-mutable configuration for the surface (present mode, frame latency).
    ///
    /// These are the fields exposed via [`RenderState::surface_config`] for live
    /// reconfiguration at runtime.
    pub surface: SurfaceConfig,

    /// How to create the wgpu adapter & device
    pub wgpu_setup: WgpuSetup,

    /// Callback for surface status changes.
    ///
    /// Called with the [`wgpu::CurrentSurfaceTexture`] result whenever acquiring a frame
    /// does not return [`wgpu::CurrentSurfaceTexture::Success`]. For
    /// [`wgpu::CurrentSurfaceTexture::Suboptimal`], egui uses the frame as-is and
    /// defers surface reconfiguration to the next frame — the callback is not invoked
    /// in that case either.
    pub on_surface_status:
        Arc<dyn Fn(&wgpu::CurrentSurfaceTexture) -> SurfaceErrorAction + Send + Sync>,

    /// PRISME PATCH: optionally choose the surface texture format and initial colour space
    /// (e.g. `Rgba16Float` + an extended-range colour space for HDR output). Called once with
    /// the adapter and the first surface. `None` (the default) or a `None` / unsupported result
    /// keeps egui's usual choice (`Rgba8Unorm`/`Bgra8Unorm`, colour space `Auto`).
    ///
    /// egui's own shader picks its output encoding from the *format* only: for non-`*Srgb`
    /// formats it writes sRGB-encoded (gamma) values. That is correct for `Srgb`,
    /// `ExtendedSrgb` (encoded extended range) and `DisplayP3`-family spaces' transfer, but
    /// NOT for the linear `ExtendedSrgbLinear` (scRGB), where the UI would look too dark.
    pub target_format_selector: Option<TargetFormatSelector>,
}

/// PRISME PATCH: see [`WgpuConfiguration::target_format_selector`].
pub type TargetFormatSelector = Arc<
    dyn Fn(&wgpu::Adapter, &wgpu::Surface<'_>) -> Option<(wgpu::TextureFormat, wgpu::SurfaceColorSpace)>
        + Send
        + Sync,
>;

/// PRISME PATCH: runtime colour-space state of the presentation surface, shared through
/// [`RenderState::surface_color`].
///
/// The app writes `requested`; the painter applies it on the next paint (if it is in
/// `supported`) and reflects the result in `configured`. The painter also refreshes
/// `display` (the [`wgpu::DisplayHdrInfo`] of the display the window is on) at most every
/// 500 ms while painting, so moving the window to another display or changing brightness is
/// picked up.
#[derive(Clone, Debug)]
pub struct SurfaceColorState {
    /// Colour space the application wants.
    pub requested: wgpu::SurfaceColorSpace,
    /// Colour space the surface is configured with (`Auto` = wgpu's default resolution).
    pub configured: wgpu::SurfaceColorSpace,
    /// Colour spaces the surface supports for [`RenderState::target_format`].
    pub supported: wgpu::SurfaceColorSpaces,
    /// Latest HDR info of the display hosting the surface.
    pub display: wgpu::DisplayHdrInfo,
    /// Whether `display` has been queried at least once.
    pub display_known: bool,
    last_query: Option<web_time::Instant>,
}

impl Default for SurfaceColorState {
    fn default() -> Self {
        Self {
            requested: wgpu::SurfaceColorSpace::Auto,
            configured: wgpu::SurfaceColorSpace::Auto,
            supported: wgpu::SurfaceColorSpaces::empty(),
            display: wgpu::DisplayHdrInfo::default(),
            display_known: false,
            last_query: None,
        }
    }
}

/// PRISME PATCH: called by the painters once per frame (must be on the main thread for
/// Metal's `display_hdr_info`). Refreshes the display info (throttled) and returns the new
/// colour space when a supported change was requested (the caller must then reconfigure
/// the surface with it).
pub fn sync_surface_color(
    state: &RwLock<SurfaceColorState>,
    surface: &wgpu::Surface<'_>,
    adapter: &wgpu::Adapter,
) -> Option<wgpu::SurfaceColorSpace> {
    let mut s = state.write();
    let now = web_time::Instant::now();
    if s.last_query
        .is_none_or(|t| now.duration_since(t) >= std::time::Duration::from_millis(500))
    {
        s.display = surface.display_hdr_info(adapter);
        s.display_known = true;
        s.last_query = Some(now);
    }
    if s.requested == s.configured {
        return None;
    }
    let ok = s.requested == wgpu::SurfaceColorSpace::Auto
        || s.requested
            .to_color_spaces()
            .is_some_and(|flag| s.supported.contains(flag));
    if ok {
        s.configured = s.requested;
        Some(s.configured)
    } else {
        log::warn!(
            "Surface colour space {:?} is unsupported for this surface; keeping {:?}",
            s.requested,
            s.configured
        );
        s.requested = s.configured;
        None
    }
}

#[test]
fn wgpu_config_impl_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WgpuConfiguration>();
}

impl core::fmt::Debug for WgpuConfiguration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Self {
            surface,
            wgpu_setup,
            on_surface_status: _,
            target_format_selector: _,
        } = self;
        f.debug_struct("WgpuConfiguration")
            .field("surface", &surface)
            .field("wgpu_setup", &wgpu_setup)
            .finish_non_exhaustive()
    }
}

impl WgpuConfiguration {
    #[inline]
    pub fn with_surface_config(mut self, surface_config: SurfaceConfig) -> Self {
        self.surface = surface_config;
        self
    }
}

impl Default for WgpuConfiguration {
    fn default() -> Self {
        Self {
            surface: SurfaceConfig::HIGH_THROUGHPUT,

            // No display handle available at this point — callers should replace this with
            // `WgpuSetup::from_display_handle(...)` before creating the instance if one is available.
            wgpu_setup: WgpuSetup::without_display_handle(),
            on_surface_status: Arc::new(|status| match status {
                wgpu::CurrentSurfaceTexture::Outdated => {
                    // The compositor changed the surface (resize, scale, output, …). wgpu
                    // requires us to reconfigure before the next acquire. Skipping would mean
                    // we are stuck in `Outdated` forever.
                    log::trace!("Dropped frame with error: {status:?}");
                    SurfaceErrorAction::Reconfigure
                }
                wgpu::CurrentSurfaceTexture::Lost => {
                    // The underlying surface is gone and we need a fresh one from the `wgpu::Instance`.
                    log::debug!("Dropped frame with error: {status:?}");
                    SurfaceErrorAction::RecreateSurface
                }
                wgpu::CurrentSurfaceTexture::Occluded => {
                    // App is hidden (minimized / behind another window). Skip silently.
                    log::trace!("Skipping frame due to occlusion.");
                    SurfaceErrorAction::SkipFrame
                }
                _ => {
                    log::warn!("Dropped frame with error: {status:?}");
                    SurfaceErrorAction::SkipFrame
                }
            }),
            target_format_selector: None,
        }
    }
}

/// Find the framebuffer format that egui prefers
///
/// # Errors
/// Returns [`WgpuError::NoSurfaceFormatsAvailable`] if the given list of formats is empty.
pub fn preferred_framebuffer_format(
    formats: &[wgpu::TextureFormat],
) -> Result<wgpu::TextureFormat, WgpuError> {
    for &format in formats {
        if matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
        ) {
            return Ok(format);
        }
    }

    formats
        .first()
        .copied()
        .ok_or(WgpuError::NoSurfaceFormatsAvailable)
}

/// Take's epi's depth/stencil bits and returns the corresponding wgpu format.
pub fn depth_format_from_bits(depth_buffer: u8, stencil_buffer: u8) -> Option<wgpu::TextureFormat> {
    match (depth_buffer, stencil_buffer) {
        (0, 8) => Some(wgpu::TextureFormat::Stencil8),
        (16, 0) => Some(wgpu::TextureFormat::Depth16Unorm),
        (24, 0) => Some(wgpu::TextureFormat::Depth24Plus),
        (24, 8) => Some(wgpu::TextureFormat::Depth24PlusStencil8),
        (32, 0) => Some(wgpu::TextureFormat::Depth32Float),
        (32, 8) => Some(wgpu::TextureFormat::Depth32FloatStencil8),
        _ => None,
    }
}

// ---------------------------------------------------------------------------

fn log_adapter_info(info: &wgpu::AdapterInfo) {
    let summary = adapter_info_summary(info);

    let is_test = cfg!(test); // Software rasterizers are expected (and preferred) during testing!

    if info.device_type == wgpu::DeviceType::Cpu && !is_test {
        log::warn!("Software rasterizer detected - loss of performance expected. {summary}");
    } else {
        log::debug!("wgpu adapter: {summary}");
    }
}

/// A human-readable summary about an adapter
pub fn adapter_info_summary(info: &wgpu::AdapterInfo) -> String {
    let wgpu::AdapterInfo {
        name,
        vendor,
        device,
        device_type,
        driver,
        driver_info,
        backend,
        device_pci_bus_id,
        subgroup_min_size,
        subgroup_max_size,
        transient_saves_memory,
        limit_bucket,
    } = &info;

    // Example values:
    // > name: "llvmpipe (LLVM 16.0.6, 256 bits)", device_type: Cpu, backend: Vulkan, driver: "llvmpipe", driver_info: "Mesa 23.1.6-arch1.4 (LLVM 16.0.6)"
    // > name: "Apple M1 Pro", device_type: IntegratedGpu, backend: Metal, driver: "", driver_info: ""
    // > name: "ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)", device_type: IntegratedGpu, backend: Gl, driver: "", driver_info: ""

    use core::fmt::Write as _;

    let mut summary = format!("backend: {backend:?}, device_type: {device_type:?}");

    if !name.is_empty() {
        write!(summary, ", name: {name:?}").ok();
    }
    if !driver.is_empty() {
        write!(summary, ", driver: {driver:?}").ok();
    }
    if !driver_info.is_empty() {
        write!(summary, ", driver_info: {driver_info:?}").ok();
    }
    if *vendor != 0 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            write!(
                summary,
                ", vendor: {} (0x{vendor:04X})",
                parse_vendor_id(*vendor)
            )
            .ok();
        }
        #[cfg(target_arch = "wasm32")]
        {
            write!(summary, ", vendor: 0x{vendor:04X}").ok();
        }
    }
    if *device != 0 {
        write!(summary, ", device: 0x{device:02X}").ok();
    }
    if !device_pci_bus_id.is_empty() {
        write!(summary, ", pci_bus_id: {device_pci_bus_id:?}").ok();
    }
    if *subgroup_min_size != 0 || *subgroup_max_size != 0 {
        write!(
            summary,
            ", subgroup_size: {subgroup_min_size}..={subgroup_max_size}"
        )
        .ok();
    }
    write!(
        summary,
        ", transient_saves_memory: {transient_saves_memory:?}"
    )
    .ok();
    write!(summary, ", limit_bucket: {limit_bucket:?}").ok();

    summary
}

/// Tries to parse the adapter's vendor ID to a human-readable string.
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_vendor_id(vendor_id: u32) -> &'static str {
    match vendor_id {
        wgpu::hal::auxil::db::amd::VENDOR => "AMD",
        wgpu::hal::auxil::db::apple::VENDOR => "Apple",
        wgpu::hal::auxil::db::arm::VENDOR => "ARM",
        wgpu::hal::auxil::db::broadcom::VENDOR => "Broadcom",
        wgpu::hal::auxil::db::imgtec::VENDOR => "Imagination Technologies",
        wgpu::hal::auxil::db::intel::VENDOR => "Intel",
        wgpu::hal::auxil::db::mesa::VENDOR => "Mesa",
        wgpu::hal::auxil::db::nvidia::VENDOR => "NVIDIA",
        wgpu::hal::auxil::db::qualcomm::VENDOR => "Qualcomm",
        _ => "Unknown",
    }
}
