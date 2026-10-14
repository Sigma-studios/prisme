//! Headless GPU tests for HDR display output and extended-range scopes.

use prisme_core::geom::Affine2;
use prisme_core::{Document, IRect};
use prisme_gpu::image::{GpuImage, PooledTex};
use prisme_gpu::scopes::Scopes;
use prisme_gpu::{DisplayOutput, DisplayParams, DisplayRenderer, Engine, EvalFormat, EvalRequest, GpuContext, HighlightMode, Readback, RenderSource};
use std::sync::Arc;

fn ctx() -> Option<GpuContext> {
    pollster::block_on(GpuContext::headless())
}

/// Document filled with a working-space grey of `v` (× SDR white).
fn image(engine: &mut Engine, v: f32) -> GpuImage {
    let doc = Document::with_background("hdr", 64, 64, Some([v, v, v, 1.0]));
    engine.evaluate(&EvalRequest { doc: &doc, rect: doc.bounds(), mip: 0, format: EvalFormat::F16, source: RenderSource::LayerStack }).unwrap()
}

fn draw(ctx: &GpuContext, src: &GpuImage, output: DisplayOutput) -> [f32; 4] {
    let format = wgpu::TextureFormat::Rgba16Float;
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr-target"),
        size: wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let mut r = DisplayRenderer::new(&ctx.device, format);
    let p = DisplayParams {
        fb_to_doc: Affine2::IDENTITY,
        level_scale: 1.0,
        nearest: true,
        doc_size: [64.0, 64.0],
        checker_px: 8.0,
        display_space: prisme_color::RgbSpace::Srgb,
        exposure: 0.0,
        dither: false,
        output,
        bg: [0.0; 3],
        checker_a: [0.8; 3],
        checker_b: [0.6; 3],
    };
    r.prepare(&ctx.device, &p, src);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let mut pass = pass.forget_lifetime();
        r.paint(&mut pass);
    }
    ctx.queue.submit([enc.finish()]);
    let img = GpuImage { tex: Arc::new(PooledTex { tex, view, w: 16, h: 16, format }), rect: IRect::new(0, 0, 16, 16), straight: false };
    let px = Readback::start(&ctx.device, &ctx.queue, &img, IRect::new(4, 4, 5, 5)).wait(&ctx.device);
    [px[0], px[1], px[2], px[3]]
}

#[test]
fn hdr_display_keeps_highlights() {
    let Some(ctx) = ctx() else { eprintln!("no GPU adapter; skipping"); return };
    let mut engine = Engine::new(ctx.clone());
    let src = image(&mut engine, 4.0);
    let close = |a: f32, b: f32| (a - b).abs() < 4e-3 * b.abs().max(1.0);
    let enc = prisme_color::hdr::encode_extended_srgb;

    // SDR: clamped at white.
    let sdr = draw(&ctx, &src, DisplayOutput::Sdr);
    assert!(close(sdr[0], 1.0), "{sdr:?}");

    // HDR with plenty of headroom: 4× SDR white passes through (encoded).
    let hdr = draw(&ctx, &src, DisplayOutput::ExtendedSrgb { peak: 8.0, highlights: HighlightMode::SoftRolloff });
    assert!(close(hdr[0], enc(4.0)) && close(hdr[2], enc(4.0)), "{hdr:?} vs {}", enc(4.0));

    // Peak 2: clip vs soft roll-off.
    let clip = draw(&ctx, &src, DisplayOutput::ExtendedSrgb { peak: 2.0, highlights: HighlightMode::Clip });
    assert!(close(clip[0], enc(2.0)), "{clip:?}");
    let soft = draw(&ctx, &src, DisplayOutput::ExtendedSrgb { peak: 2.0, highlights: HighlightMode::SoftRolloff });
    let expect = enc(prisme_color::hdr::soft_rolloff(4.0, 2.0));
    assert!(close(soft[0], expect) && soft[0] < enc(2.0), "{soft:?} vs {expect}");

    // SDR content is unchanged by HDR mode.
    let mid = image(&mut engine, 0.5);
    let a = draw(&ctx, &mid, DisplayOutput::Sdr);
    let b = draw(&ctx, &mid, DisplayOutput::ExtendedSrgb { peak: 4.0, highlights: HighlightMode::SoftRolloff });
    assert!(close(a[0], b[0]) && close(a[0], enc(0.5)), "{a:?} {b:?}");
}

#[test]
fn scopes_report_values_above_white() {
    let Some(ctx) = ctx() else { return };
    let mut engine = Engine::new(ctx.clone());
    let mut scopes = Scopes::new(&ctx.device);
    let wait = |scopes: &mut Scopes| {
        let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        while !scopes.poll(&ctx.device) {
            std::thread::yield_now();
        }
        scopes.latest.clone().unwrap()
    };
    // 2× white over a 4× range: lands at encoded 2/4 → bin round(255·enc(2)/enc(4)).
    scopes.request(&ctx.device, &ctx.queue, &image(&mut engine, 2.0), 4.0);
    let d = wait(&mut scopes);
    assert_eq!(d.total, 64 * 64);
    assert_eq!(d.over, [d.total; 4], "{:?}", d.over);
    assert_eq!(d.range, 4.0);
    let enc = prisme_color::hdr::encode_extended_srgb;
    let bin = (enc(2.0) / enc(4.0) * 255.0).round() as usize;
    let peak = d.hist_ext[3].iter().enumerate().max_by_key(|(_, v)| **v).unwrap().0;
    assert!(peak.abs_diff(bin) <= 1, "luma peak bin {peak}, expected {bin}");
    // The SDR histogram clamps to the top bin.
    assert_eq!(d.hist[0][255], d.total);

    // SDR content: nothing above white.
    scopes.request(&ctx.device, &ctx.queue, &image(&mut engine, 0.5), 1.0);
    let d = wait(&mut scopes);
    assert_eq!(d.over, [0; 4]);
    assert_eq!(d.hist, d.hist_ext);
}
