//! GPU scopes: histogram, RGB waveform and vectorscope.

use crate::image::GpuImage;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

#[derive(Clone, Default)]
pub struct ScopeData {
    /// R, G, B, luma histograms (256 bins each).
    pub hist: Vec<[u32; 256]>,
    /// 3 x 256 x 256 (row-major, top = bright).
    pub waveform: Vec<u32>,
    /// 256 x 256 (OKLab a/b).
    pub vectorscope: Vec<u32>,
}

const HIST: u64 = 1024 * 4;
const WAVE: u64 = 196_608 * 4;
const VECS: u64 = 65_536 * 4;

pub struct Scopes {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    pending: Option<(wgpu::Buffer, Arc<Mutex<Option<bool>>>)>,
    pub latest: Option<ScopeData>,
}

impl Scopes {
    pub fn new(device: &wgpu::Device) -> Self {
        let (w2lms, lms2lab, _, _) = prisme_color::oklab_gpu_matrices();
        let mat = |m: &prisme_color::Mat3| {
            let c = m.to_wgsl_padded();
            format!("mat3x3<f32>({}, {}, {}, {}, {}, {}, {}, {}, {})", c[0], c[1], c[2], c[4], c[5], c[6], c[8], c[9], c[10])
        };
        let src = include_str!("shaders/scopes.wgsl").replace("MAT_W2LMS", &mat(&w2lms)).replace("MAT_LMS2LAB", &mat(&lms2lab));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("scopes"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let storage = |b| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scopes"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(2),
                storage(3),
                storage(4),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("scopes"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scopes"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Scopes { pipeline, bgl, pending: None, latest: None }
    }

    pub fn busy(&self) -> bool {
        self.pending.is_some()
    }

    /// Start computing scopes for `image` (skipped if a previous request is in flight).
    pub fn request(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: &GpuImage) {
        if self.pending.is_some() {
            return;
        }
        let w = image.rect.width().max(1) as u32;
        let h = image.rect.height().max(1) as u32;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scopes-params"),
            contents: bytemuck::cast_slice(&[w, h, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mk = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scopes-acc"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let (hb, wb, vb) = (mk(HIST), mk(WAVE), mk(VECS));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scopes"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(image.view()) },
                wgpu::BindGroupEntry { binding: 2, resource: hb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: vb.as_entire_binding() },
            ],
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scopes-read"),
            size: HIST + WAVE + VECS,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.clear_buffer(&hb, 0, None);
        enc.clear_buffer(&wb, 0, None);
        enc.clear_buffer(&vb, 0, None);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&hb, 0, &read, 0, HIST);
        enc.copy_buffer_to_buffer(&wb, 0, &read, HIST, WAVE);
        enc.copy_buffer_to_buffer(&vb, 0, &read, HIST + WAVE, VECS);
        queue.submit([enc.finish()]);
        let state = Arc::new(Mutex::new(None));
        let s2 = state.clone();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| *s2.lock().unwrap() = Some(r.is_ok()));
        self.pending = Some((read, state));
    }

    /// Poll for completion; returns true when new data arrived.
    pub fn poll(&mut self, device: &wgpu::Device) -> bool {
        let Some((buf, state)) = &self.pending else { return false };
        let _ = device.poll(wgpu::PollType::Poll);
        let done = *state.lock().unwrap();
        match done {
            None => false,
            Some(ok) => {
                if ok {
                    let Ok(data) = buf.slice(..).get_mapped_range() else { self.pending = None; return false };
                    let u: &[u32] = bytemuck::cast_slice(&data);
                    let mut hist = vec![[0u32; 256]; 4];
                    for (c, h) in hist.iter_mut().enumerate() {
                        h.copy_from_slice(&u[c * 256..(c + 1) * 256]);
                    }
                    let waveform = u[1024..1024 + 196_608].to_vec();
                    let vectorscope = u[1024 + 196_608..].to_vec();
                    drop(data);
                    buf.unmap();
                    self.latest = Some(ScopeData { hist, waveform, vectorscope });
                }
                self.pending = None;
                ok
            }
        }
    }
}
