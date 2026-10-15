use prisme_core::IRect;
use std::sync::Arc;

pub struct PooledTex {
    pub tex: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub w: u32,
    pub h: u32,
    pub format: wgpu::TextureFormat,
}

/// A GPU image covering `rect` (level-space pixels). Texel (0,0) maps to `rect.x0, rect.y0`.
/// The backing texture may be larger than the rect (pooling).
#[derive(Clone)]
pub struct GpuImage {
    pub tex: Arc<PooledTex>,
    pub rect: IRect,
    /// True when colour is straight alpha (raw layer uploads); computed images are premultiplied.
    pub straight: bool,
}

impl GpuImage {
    pub fn view(&self) -> &wgpu::TextureView {
        &self.tex.view
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.tex.format
    }

    pub fn byte_size(&self) -> u64 {
        let bpp = self.tex.format.block_copy_size(None).unwrap_or(8) as u64;
        self.tex.w as u64 * self.tex.h as u64 * bpp
    }
}


/// Reuses textures whose only owner is the pool.
pub struct TexturePool {
    device: wgpu::Device,
    textures: Vec<Arc<PooledTex>>,
}

impl TexturePool {
    pub fn new(device: wgpu::Device) -> Self {
        Self { device, textures: vec![] }
    }

    pub fn acquire(&mut self, w: u32, h: u32, format: wgpu::TextureFormat) -> Arc<PooledTex> {
        let w = w.max(1);
        let h = h.max(1);
        // Best fit among free textures that are large enough but not wastefully so.
        let mut best: Option<usize> = None;
        for (i, t) in self.textures.iter().enumerate() {
            if Arc::strong_count(t) == 1 && t.format == format && t.w >= w && t.h >= h {
                let waste = t.w as u64 * t.h as u64;
                if waste <= (w as u64 * h as u64) * 2 + 65536
                    && best.is_none_or(|b| waste < self.textures[b].w as u64 * self.textures[b].h as u64)
                {
                    best = Some(i);
                }
            }
        }
        if let Some(i) = best {
            // Most recently used last (`trim` drops from the front).
            let t = self.textures.remove(i);
            self.textures.push(t.clone());
            return t;
        }
        // Quarter-octave size buckets (…, 4096, 5120, 6144, 7168, 8192, …): the canvas
        // evaluates between 1x and 2x the screen per axis whatever the zoom, so a view growing
        // a little each frame (zooming out) lands in a few sizes it can reuse instead of
        // allocating (and clearing) new textures every frame.
        let round = |v: u32| {
            let p = 1u32 << (31 - v.max(64).leading_zeros());
            let step = (p / 4).max(64);
            v.div_ceil(step) * step
        };
        let (tw, th) = (round(w), round(h));
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prisme-pool"),
            size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let t = Arc::new(PooledTex { tex, view, w: tw, h: th, format });
        self.textures.push(t.clone());
        t
    }

    pub fn image(&mut self, rect: IRect, format: wgpu::TextureFormat) -> GpuImage {
        let tex = self.acquire(rect.width() as u32, rect.height() as u32, format);
        GpuImage { tex, rect, straight: false }
    }

    /// Drop free textures beyond a budget (bytes), least recently used first.
    pub fn trim(&mut self, keep_free_bytes: u64) {
        let mut free = 0u64;
        let mut keep: Vec<bool> = self
            .textures
            .iter()
            .rev()
            .map(|t| {
                if Arc::strong_count(t) > 1 {
                    return true;
                }
                let bpp = t.format.block_copy_size(None).unwrap_or(8) as u64;
                free += t.w as u64 * t.h as u64 * bpp;
                free <= keep_free_bytes
            })
            .collect();
        keep.reverse();
        let mut k = keep.into_iter();
        self.textures.retain(|_| k.next().unwrap());
    }

    pub fn total_bytes(&self) -> u64 {
        self.textures
            .iter()
            .map(|t| t.w as u64 * t.h as u64 * t.format.block_copy_size(None).unwrap_or(8) as u64)
            .sum()
    }
}
