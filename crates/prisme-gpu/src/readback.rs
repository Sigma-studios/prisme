//! Asynchronous texture readback (works on native and web).

use crate::image::GpuImage;
use prisme_core::IRect;
use std::sync::{Arc, Mutex};

pub struct Readback {
    buffer: wgpu::Buffer,
    pub rect: IRect,
    bytes_per_row: u32,
    bpp: u32,
    format: wgpu::TextureFormat,
    state: Arc<Mutex<Option<bool>>>,
}

impl Readback {
    /// Start reading `region` (level rect inside `image.rect`).
    pub fn start(device: &wgpu::Device, queue: &wgpu::Queue, image: &GpuImage, region: IRect) -> Readback {
        let region = region.intersect(&image.rect);
        let format = image.format();
        let bpp = format.block_copy_size(None).unwrap_or(8);
        let w = region.width().max(1) as u32;
        let h = region.height().max(1) as u32;
        let bytes_per_row = (w * bpp).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        if !region.is_empty() {
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &image.tex.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: (region.x0 - image.rect.x0) as u32,
                        y: (region.y0 - image.rect.y0) as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(h) },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        queue.submit([enc.finish()]);
        let state = Arc::new(Mutex::new(None));
        let s2 = state.clone();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            *s2.lock().unwrap() = Some(r.is_ok());
        });
        Readback { buffer, rect: region, bytes_per_row, bpp, format, state }
    }

    /// Non-blocking poll. Returns RGBA f32 (row-major) once ready.
    pub fn try_take(&self, device: &wgpu::Device) -> Option<Vec<f32>> {
        let _ = device.poll(wgpu::PollType::Poll);
        match *self.state.lock().unwrap() {
            Some(true) => Some(self.decode()),
            Some(false) => Some(vec![]),
            None => None,
        }
    }

    /// Block until complete (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self, device: &wgpu::Device) -> Vec<f32> {
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        loop {
            if let Some(v) = self.try_take(device) {
                return v;
            }
            std::thread::yield_now();
        }
    }

    fn decode(&self) -> Vec<f32> {
        let w = self.rect.width() as usize;
        let h = self.rect.height() as usize;
        let Ok(data) = self.buffer.slice(..).get_mapped_range() else { return vec![] };
        let ch = match self.format {
            wgpu::TextureFormat::R16Float | wgpu::TextureFormat::R32Float => 1,
            _ => 4,
        };
        let mut out = Vec::with_capacity(w * h * ch);
        for y in 0..h {
            let row = &data[y * self.bytes_per_row as usize..][..w * self.bpp as usize];
            match self.format {
                wgpu::TextureFormat::Rgba32Float | wgpu::TextureFormat::R32Float => {
                    out.extend(row.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                }
                wgpu::TextureFormat::Rgba8Unorm => out.extend(row.iter().map(|&b| b as f32 / 255.0)),
                _ => out.extend(row.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())),
            }
        }
        drop(data);
        self.buffer.unmap();
        out
    }
}
