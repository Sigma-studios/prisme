//! Prisme user interface (egui). Platform-agnostic: the desktop and web apps only
//! create the eframe runner and hand it [`PrismeApp`].

pub mod actions;
pub mod app;
pub mod canvas;
pub mod commands;
pub mod dialogs;
pub mod doc;
pub mod hdr;
pub mod jobs;
pub mod offload;
pub mod outline;
pub mod perf;
pub mod panels;
pub mod rasterizer;
pub mod script;
pub mod settings;
pub mod shortcuts;
pub mod stroke;
pub mod theme;
pub mod widgets;

pub use app::PrismeApp;

/// wgpu configuration shared by native and web runners: WebGPU / native backends only
/// (compute shaders are required), with Prisme's limits.
pub fn wgpu_options() -> egui_wgpu::WgpuConfiguration {
    use egui_wgpu::{WgpuSetup, WgpuSetupCreateNew};
    let mut create = WgpuSetupCreateNew::without_display_handle();
    create.instance_descriptor.backends = wgpu_backends();
    create.device_descriptor = std::sync::Arc::new(|adapter: &egui_wgpu::wgpu::Adapter| egui_wgpu::wgpu::DeviceDescriptor {
        label: Some("prisme"),
        required_limits: prisme_gpu::GpuContext::required_limits(&adapter.limits()),
        // GPU timings in the Performance panel, where supported.
        required_features: adapter.features() & egui_wgpu::wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    });
    egui_wgpu::WgpuConfiguration {
        wgpu_setup: WgpuSetup::CreateNew(create),
        // HDR: Rgba16Float surface with a runtime-switchable colour space where available.
        target_format_selector: Some(std::sync::Arc::new(hdr::select_surface)),
        ..Default::default()
    }
}

fn wgpu_backends() -> egui_wgpu::wgpu::Backends {
    use egui_wgpu::wgpu::Backends;
    if cfg!(target_arch = "wasm32") { Backends::BROWSER_WEBGPU } else { Backends::PRIMARY }
}
