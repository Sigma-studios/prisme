//! Compute kernel library: one WGSL module (specialized per output format) with many
//! entry points sharing a single bind group layout.

use crate::image::GpuImage;
use prisme_core::IRect;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

const SOURCE: &str = include_str!("shaders/kernels.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutFormat {
    Rgba16F,
    Rgba32F,
    Rgba8,
}

impl OutFormat {
    pub fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            OutFormat::Rgba16F => wgpu::TextureFormat::Rgba16Float,
            OutFormat::Rgba32F => wgpu::TextureFormat::Rgba32Float,
            OutFormat::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
        }
    }

    fn wgsl(self) -> &'static str {
        match self {
            OutFormat::Rgba16F => "rgba16float",
            OutFormat::Rgba32F => "rgba32float",
            OutFormat::Rgba8 => "rgba8unorm",
        }
    }

    pub fn of(format: wgpu::TextureFormat) -> OutFormat {
        match format {
            wgpu::TextureFormat::Rgba32Float => OutFormat::Rgba32F,
            wgpu::TextureFormat::Rgba8Unorm => OutFormat::Rgba8,
            _ => OutFormat::Rgba16F,
        }
    }
}

/// Per-dispatch global values.
#[derive(Clone, Copy, Debug)]
pub struct Globals {
    pub level_scale: f32,
    pub mip: u32,
    pub doc_w: f32,
    pub doc_h: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InfoUniform {
    out_origin: [i32; 2],
    out_size: [i32; 2],
    inputs: [[i32; 4]; 4],
    connected: [u32; 4],
    level_scale: f32,
    mip: f32,
    doc_w: f32,
    doc_h: f32,
}

pub struct Kernels {
    device: wgpu::Device,
    layouts: HashMap<OutFormat, (wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    modules: HashMap<OutFormat, wgpu::ShaderModule>,
    pipelines: HashMap<(OutFormat, &'static str), wgpu::ComputePipeline>,
    dummy_view: wgpu::TextureView,
    dummy_lut: wgpu::Buffer,
}

fn mat_wgsl(m: &prisme_color::Mat3) -> String {
    let c = m.to_wgsl_padded();
    format!(
        "mat3x3<f32>({:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9}, {:.9})",
        c[0], c[1], c[2], c[4], c[5], c[6], c[8], c[9], c[10]
    )
}

fn make_layout(device: &wgpu::Device, fmt: OutFormat) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let uniform = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("prisme-kernels"),
        entries: &[
            uniform(0),
            uniform(1),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: fmt.texture_format(),
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            tex_entry(3),
            tex_entry(4),
            tex_entry(5),
            tex_entry(6),
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("prisme-kernels"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    (bgl, layout)
}

impl Kernels {
    pub fn new(device: &wgpu::Device) -> Self {
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_lut = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy-lut"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            device: device.clone(),
            layouts: HashMap::new(),
            modules: HashMap::new(),
            pipelines: HashMap::new(),
            dummy_view: dummy.create_view(&Default::default()),
            dummy_lut,
        }
    }

    fn layout(&mut self, fmt: OutFormat) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let device = &self.device;
        self.layouts.entry(fmt).or_insert_with(|| make_layout(device, fmt)).clone()
    }

    fn module(&mut self, fmt: OutFormat) -> &wgpu::ShaderModule {
        let device = &self.device;
        self.modules.entry(fmt).or_insert_with(|| {
            let (w2lms, lms2lab, lab2lms, lms2w) = prisme_color::oklab_gpu_matrices();
            let src = SOURCE
                .replace("OUT_FORMAT", fmt.wgsl())
                .replace("MAT_W2LMS", &mat_wgsl(&w2lms))
                .replace("MAT_LMS2LAB", &mat_wgsl(&lms2lab))
                .replace("MAT_LAB2LMS", &mat_wgsl(&lab2lms))
                .replace("MAT_LMS2W", &mat_wgsl(&lms2w));
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("prisme-kernels"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        })
    }

    fn pipeline(&mut self, fmt: OutFormat, entry: &'static str) -> wgpu::ComputePipeline {
        if let Some(p) = self.pipelines.get(&(fmt, entry)) {
            return p.clone();
        }
        let module = self.module(fmt).clone();
        let (_, layout) = self.layout(fmt);
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipelines.insert((fmt, entry), p.clone());
        p
    }

    /// Record a kernel dispatch writing `out` over `region` (defaults to the whole image rect).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        entry: &'static str,
        out: &GpuImage,
        region: Option<IRect>,
        inputs: [Option<&GpuImage>; 4],
        params: &[f32],
        lut: Option<&wgpu::Buffer>,
        g: &Globals,
    ) {
        let fmt = OutFormat::of(out.format());
        let pipeline = self.pipeline(fmt, entry);
        let region = region.unwrap_or(out.rect);
        // Output texel (0,0) == out.rect origin; region must be inside out.rect.
        let off = [region.x0 - out.rect.x0, region.y0 - out.rect.y0];
        debug_assert!(off[0] >= 0 && off[1] >= 0);
        let mut info = InfoUniform {
            out_origin: [region.x0, region.y0],
            out_size: [region.width(), region.height()],
            inputs: [[0; 4]; 4],
            connected: [0; 4],
            level_scale: g.level_scale,
            mip: g.mip as f32,
            doc_w: g.doc_w,
            doc_h: g.doc_h,
        };
        for (i, inp) in inputs.iter().enumerate() {
            if let Some(img) = inp {
                info.inputs[i] = [img.rect.x0, img.rect.y0, img.rect.width(), img.rect.height()];
                info.connected[i] = 1;
            }
        }
        // Kernels store at the dispatch-relative coordinate, so regions start at the origin.
        assert_eq!(off, [0, 0], "sub-region dispatch must start at the image origin");
        let mut p = [0.0f32; 64];
        p[..params.len().min(64)].copy_from_slice(&params[..params.len().min(64)]);
        let info_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("info"),
            contents: bytemuck::bytes_of(&info),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (bgl, _) = self.layout(fmt);
        let view = |i: usize| inputs[i].map(|img| img.view()).unwrap_or(&self.dummy_view);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: info_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: param_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(out.view()) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(view(0)) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(view(1)) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(view(2)) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(view(3)) },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: lut.unwrap_or(&self.dummy_lut).as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((region.width() as u32).div_ceil(8), (region.height() as u32).div_ceil(8), 1);
    }

    pub fn lut_buffer(&self, data: &[[f32; 4]]) -> wgpu::Buffer {
        let mut v: Vec<f32> = data.iter().flatten().copied().collect();
        if v.len() < 16 {
            v.resize(16, 0.0);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lut"),
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }
}
