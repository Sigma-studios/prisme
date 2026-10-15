//! f16 landing of GPU tiles: stroke tiles (f32 textures) converted to their f16 bits on the
//! GPU (`shaders/pack.wgsl`, exactly the `half` crate's rounding) into storage buffers, then
//! read back as ready-made [`Tile`]s: half the bytes of an f32 read-back and no conversion on
//! the CPU. The buffers stay alive so the same bits can be handed to the tile store
//! ([`crate::Engine::adopt_tiles`]) instead of uploading the landed tiles again.

use crate::image::GpuImage;
use prisme_core::tile::{PixelFormat, Tile, TileCoord, TILE_PIXELS, TILE_SIZE};
use std::sync::{Arc, Mutex};

/// Tiles per buffer (32 MiB for RGBA).
const CHUNK: usize = 64;

pub(crate) struct PackPipelines {
    layout: wgpu::BindGroupLayout,
    rgba: wgpu::ComputePipeline,
    mask: wgpu::ComputePipeline,
}

impl PackPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let src = include_str!("shaders/pack.wgsl").replace("TILE_SIZE_U", &format!("{TILE_SIZE}u"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("pack"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pack"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("pack"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pack"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self { rgba: pipeline("k_pack4"), mask: pipeline("k_pack1"), layout }
    }
}

struct Chunk {
    /// The f16 bits (kept for the tile store).
    bits: wgpu::Buffer,
    read: wgpu::Buffer,
    first: usize,
    count: usize,
    state: Arc<Mutex<Option<bool>>>,
}

/// A set of tiles converted to f16 on the GPU, being read back.
pub struct PackedTiles {
    pub coords: Vec<TileCoord>,
    channels: u8,
    chunks: Vec<Chunk>,
}

/// Bytes of one f16 tile.
pub fn tile_bytes(channels: u8) -> u64 {
    TILE_PIXELS as u64 * channels as u64 * 2
}

impl PackedTiles {
    /// Convert `tiles` (f32 textures, one tile each, `channels` 1 or 4) and start reading them.
    pub(crate) fn start(device: &wgpu::Device, queue: &wgpu::Queue, p: &PackPipelines, tiles: &[(TileCoord, GpuImage)], channels: u8) -> PackedTiles {
        let per = tile_bytes(channels);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pack") });
        let mut chunks = vec![];
        for (n, part) in tiles.chunks(CHUNK).enumerate() {
            let size = per * part.len() as u64;
            let bits = device.create_buffer(&wgpu::BufferDescriptor { label: Some("pack-bits"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = device.create_buffer(&wgpu::BufferDescriptor { label: Some("pack-read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let groups: Vec<wgpu::BindGroup> = part
                .iter()
                .enumerate()
                .map(|(i, (_, img))| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("pack"),
                        layout: &p.layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(img.view()) },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &bits, offset: per * i as u64, size: wgpu::BufferSize::new(per) }),
                            },
                        ],
                    })
                })
                .collect();
            {
                // One pass for the whole chunk.
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pack"), timestamp_writes: None });
                pass.set_pipeline(if channels == 1 { &p.mask } else { &p.rgba });
                let words = (TILE_PIXELS as u32 * channels as u32 / 2).div_ceil(64);
                for g in &groups {
                    pass.set_bind_group(0, g, &[]);
                    pass.dispatch_workgroups(words, 1, 1);
                }
            }
            enc.copy_buffer_to_buffer(&bits, 0, &read, 0, size);
            chunks.push(Chunk { bits, read, first: n * CHUNK, count: part.len(), state: Arc::new(Mutex::new(None)) });
        }
        queue.submit([enc.finish()]);
        for c in &chunks {
            let s = c.state.clone();
            c.read.slice(..).map_async(wgpu::MapMode::Read, move |r| *s.lock().unwrap() = Some(r.is_ok()));
        }
        PackedTiles { coords: tiles.iter().map(|(c, _)| *c).collect(), channels, chunks }
    }

    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Non-blocking: whether [`PackedTiles::take`] would return the tiles.
    pub fn ready(&self, device: &wgpu::Device) -> bool {
        let _ = device.poll(wgpu::PollType::Poll);
        self.chunks.iter().all(|c| c.state.lock().unwrap().is_some())
    }

    /// Block until read back (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        while !self.ready(device) {
            std::thread::yield_now();
        }
    }

    /// The f16 tiles in `coords` order once read back (None if not ready or a map failed).
    pub fn take(&self, device: &wgpu::Device) -> Option<Vec<Tile>> {
        if !self.ready(device) || self.chunks.iter().any(|c| *c.state.lock().unwrap() != Some(true)) {
            return None;
        }
        let per = tile_bytes(self.channels) as usize;
        let mut out = Vec::with_capacity(self.coords.len());
        for c in &self.chunks {
            {
                let data = c.read.slice(..).get_mapped_range().ok()?;
                // The copy is bound by first-touch page faults on the new tile buffers: spread
                // it over the cores on native.
                let ch = self.channels;
                let one = |i: usize| Tile::from_bytes(PixelFormat::F16, ch, &data[i * per..(i + 1) * per]);
                #[cfg(not(target_arch = "wasm32"))]
                let tiles: Option<Vec<Tile>> = {
                    use rayon::prelude::*;
                    (0..c.count).into_par_iter().map(one).collect()
                };
                #[cfg(target_arch = "wasm32")]
                let tiles: Option<Vec<Tile>> = (0..c.count).map(one).collect();
                out.extend(tiles?);
            }
            c.read.unmap();
        }
        Some(out)
    }

    /// Buffer and byte offset holding tile `i`'s bits.
    pub(crate) fn bits(&self, i: usize) -> Option<(&wgpu::Buffer, u64)> {
        let c = self.chunks.iter().find(|c| i >= c.first && i < c.first + c.count)?;
        Some((&c.bits, (i - c.first) as u64 * tile_bytes(self.channels)))
    }
}
