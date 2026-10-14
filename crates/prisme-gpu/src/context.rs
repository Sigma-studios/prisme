/// Device + queue shared by the engine. Cloning is cheap (wgpu handles are refcounted).
#[derive(Clone, Debug)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        // wgpu panics on uncaptured errors by default. In release builds log them instead: a
        // failed allocation (out of VRAM) shouldn't take unsaved documents down with it.
        #[cfg(not(debug_assertions))]
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| log::error!("wgpu: {e}")));
        Self { device, queue }
    }

    /// Limits Prisme needs on top of WebGPU defaults (none beyond defaults today).
    pub fn required_limits(adapter_limits: &wgpu::Limits) -> wgpu::Limits {
        // Clamp the defaults to what the adapter supports: requesting more fails device creation.
        let mut l = wgpu::Limits::default().or_worse_values_from(adapter_limits);
        // Use the largest textures the adapter offers (documents can be big).
        l.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;
        l.max_buffer_size = adapter_limits.max_buffer_size;
        l.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        l
    }

    /// Create a headless context (tests, CLI tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn headless() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("prisme-headless"),
                required_limits: Self::required_limits(&adapter.limits()),
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Self { device, queue })
    }

    pub fn max_texture_size(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }
}
