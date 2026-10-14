use crate::image::GpuImage;
use prisme_color::{Mat3, RgbSpace, Transfer};
use prisme_core::geom::Affine2;
use wgpu::util::DeviceExt;

/// How highlights above the display's HDR headroom are handled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HighlightMode {
    /// Clip each channel at the display peak.
    Clip,
    /// Smooth shoulder from 0.75·peak to the peak, preserving hue
    /// (see [`prisme_color::hdr::soft_rolloff`]).
    #[default]
    SoftRolloff,
}

/// What the presentation surface expects from the display pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DisplayOutput {
    /// Standard dynamic range: output clamped to [0, 1] in `display_space` (or through the
    /// baked display ICC LUT when one is installed). Surface colour space `Srgb`.
    #[default]
    Sdr,
    /// Extended-range sRGB-*encoded* surface (`SurfaceColorSpace::ExtendedSrgb`: Metal EDR,
    /// WebGPU `toneMapping: "extended"`). Working Rec.2020 is converted to linear BT.709
    /// primaries (negative = outside sRGB, kept), working 1.0 maps to SDR reference white
    /// (1.0), highlights are limited to `peak` (headroom, × SDR white), then encoded with the
    /// sign-mirrored, unbounded sRGB curve. The OS/browser colour-manages from there.
    ExtendedSrgb { peak: f32, highlights: HighlightMode },
}

impl DisplayOutput {
    pub fn is_hdr(self) -> bool {
        !matches!(self, DisplayOutput::Sdr)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DisplayParams {
    /// Maps framebuffer pixels to document pixels.
    pub fb_to_doc: Affine2,
    pub level_scale: f32,
    pub nearest: bool,
    /// Page size; negative = no page (infinite canvas: the checkerboard fills the view).
    pub doc_size: [f32; 2],
    pub checker_px: f32,
    /// Target space used when no display LUT is loaded.
    pub display_space: RgbSpace,
    pub exposure: f32,
    /// Dithering (SDR only: an fp16 extended-range surface doesn't need it, and noise sized
    /// for 8-bit SDR steps would be pointless there).
    pub dither: bool,
    /// SDR or extended-range (HDR) output.
    pub output: DisplayOutput,
    /// Background / checkerboard colours, sRGB-encoded in [0, 1]: they stay at (or below)
    /// SDR white on an HDR surface too.
    pub bg: [f32; 3],
    pub checker_a: [f32; 3],
    pub checker_b: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewUniform {
    fb_to_doc0: [f32; 4],
    fb_to_doc1: [f32; 4],
    img_origin: [f32; 2],
    img_size: [f32; 2],
    tex_size: [f32; 2],
    level_scale: f32,
    nearest: f32,
    doc_size: [f32; 2],
    checker: f32,
    has_lut: f32,
    disp: [[f32; 4]; 3],
    transfer: f32,
    target_srgb: f32,
    exposure: f32,
    dither: f32,
    bg: [f32; 4],
    chk_a: [f32; 4],
    chk_b: [f32; 4],
    /// x: 0 SDR, 1 extended sRGB (HDR) · y: peak (× SDR white) · z: 1 soft roll-off, 0 clip ·
    /// w: roll-off knee.
    hdr: [f32; 4],
}

/// Draws the evaluated canvas into an existing render pass (egui paint callback).
pub struct DisplayRenderer {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    lut_sampler: wgpu::Sampler,
    lut_view: wgpu::TextureView,
    has_lut: bool,
    target_srgb: bool,
    bind_group: Option<wgpu::BindGroup>,
}

impl DisplayRenderer {
    pub fn new(device: &wgpu::Device, target: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format: target, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("display-lin"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("display-lut"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let lut_view = Self::make_lut(device, None, 2);
        DisplayRenderer {
            pipeline,
            bgl,
            sampler,
            lut_sampler,
            lut_view,
            has_lut: false,
            target_srgb: target.is_srgb(),
            bind_group: None,
        }
    }

    fn make_lut(device: &wgpu::Device, data: Option<(&wgpu::Queue, &[f32])>, size: u32) -> wgpu::TextureView {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display-lut"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: size },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        if let Some((queue, d)) = data {
            let h: Vec<half::f16> = d.iter().map(|&v| half::f16::from_f32(v)).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&h),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * 8), rows_per_image: Some(size) },
                wgpu::Extent3d { width: size, height: size, depth_or_array_layers: size },
            );
        }
        tex.create_view(&Default::default())
    }

    /// Install (or clear) a baked display LUT (`size³` RGBA f32).
    pub fn set_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, lut: Option<(&[f32], u32)>) {
        match lut {
            Some((data, size)) => {
                self.lut_view = Self::make_lut(device, Some((queue, data)), size);
                self.has_lut = true;
            }
            None => {
                self.lut_view = Self::make_lut(device, None, 2);
                self.has_lut = false;
            }
        }
    }

    pub fn has_lut(&self) -> bool {
        self.has_lut
    }

    pub fn prepare(&mut self, device: &wgpu::Device, p: &DisplayParams, image: &GpuImage) {
        let m = p.fb_to_doc.m;
        // HDR always uses the analytic transform to linear BT.709 (extended sRGB): the baked ICC
        // LUT targets the display's *device* RGB with a [0, 1]-clamped domain, whereas on an
        // extended-range surface the OS/browser colour-manages from extended sRGB itself and
        // values above 1.0 / below 0.0 must survive.
        let hdr = match p.output {
            DisplayOutput::Sdr => [0.0; 4],
            DisplayOutput::ExtendedSrgb { peak, highlights } => {
                let peak = peak.max(1.0);
                [1.0, peak, (highlights == HighlightMode::SoftRolloff) as i32 as f32, prisme_color::hdr::rolloff_knee(peak)]
            }
        };
        let space = if p.output.is_hdr() { RgbSpace::Srgb } else { p.display_space };
        let disp: Mat3 = space.from_working_matrix();
        let d = disp.to_wgsl_padded();
        let transfer = match space.transfer() {
            Transfer::Srgb => 0.0,
            Transfer::Linear => 1.0,
            Transfer::Gamma(g) if (g - 2.2).abs() < 0.05 => 2.0,
            Transfer::Rec709 => 3.0,
            _ => 0.0,
        };
        let u = ViewUniform {
            fb_to_doc0: [m[0], m[1], m[2], 0.0],
            fb_to_doc1: [m[3], m[4], m[5], 0.0],
            img_origin: [image.rect.x0 as f32, image.rect.y0 as f32],
            img_size: [image.rect.width() as f32, image.rect.height() as f32],
            tex_size: [image.tex.w as f32, image.tex.h as f32],
            level_scale: p.level_scale,
            nearest: p.nearest as i32 as f32,
            doc_size: p.doc_size,
            checker: p.checker_px.max(1.0),
            has_lut: self.has_lut as i32 as f32,
            disp: [[d[0], d[1], d[2], 0.0], [d[4], d[5], d[6], 0.0], [d[8], d[9], d[10], 0.0]],
            transfer,
            target_srgb: self.target_srgb as i32 as f32,
            exposure: p.exposure,
            dither: (p.dither && !p.output.is_hdr()) as i32 as f32,
            bg: [p.bg[0], p.bg[1], p.bg[2], 1.0],
            chk_a: [p.checker_a[0], p.checker_a[1], p.checker_a[2], 1.0],
            chk_b: [p.checker_b[0], p.checker_b[1], p.checker_b[2], 1.0],
            hdr,
        };
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("display-view"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(image.view()) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.lut_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.lut_sampler) },
            ],
        }));
    }

    pub fn paint(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(bg) = &self.bind_group {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
