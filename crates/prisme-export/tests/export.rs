use prisme_color::{RgbSpace, SpaceConverter, srgb};
use prisme_export::*;
use std::io::Cursor;

// ------------------------------------------------------------------ helpers

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32
    }
}

/// Build a working-space image from sRGB-encoded RGBA values in [0,1].
fn from_srgb(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> Vec<f32> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = f(x, y);
            let l = srgb().to_working([p[0], p[1], p[2]]);
            v.extend([l[0], l[1], l[2], p[3]]);
        }
    }
    v
}

fn random_srgb8(w: u32, h: u32, seed: u64) -> (Vec<f32>, Vec<[u8; 3]>) {
    let mut r = Lcg(seed);
    let cols: Vec<[u8; 3]> = (0..w * h).map(|_| [r.next() as u8, r.next() as u8, r.next() as u8]).collect();
    let px = from_srgb(w, h, |x, y| {
        let c = cols[(y * w + x) as usize];
        [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0]
    });
    (px, cols)
}

/// Smooth gradient + noise (photo-like but with high entropy).
fn noisy(w: u32, h: u32, amp: f32, seed: u64) -> Vec<f32> {
    let mut r = Lcg(seed);
    let mut n = || (r.unit() - 0.5) * amp;
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let base = [x as f32 / w as f32, y as f32 / h as f32, 0.5 + 0.4 * ((x + y) as f32 * 0.02).sin()];
            let c = [base[0] + n(), base[1] + n(), base[2] + n()].map(|c| c.clamp(0.0, 1.0));
            let l = srgb().to_working(c);
            v.extend([l[0], l[1], l[2], 1.0]);
        }
    }
    v
}

struct Decoded {
    width: u32,
    height: u32,
    color: png::ColorType,
    depth: u8,
    interlaced: bool,
    palette_len: Option<usize>,
    chunks: Vec<[u8; 4]>,
    cicp: Option<[u8; 4]>,
    /// Expanded samples (palette -> RGB(A)), native depth (16-bit as big-endian pairs).
    data: Vec<u8>,
    out_color: png::ColorType,
}

fn chunk_list(bytes: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
    let mut pos = 8;
    let mut out = vec![];
    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let name: [u8; 4] = bytes[pos + 4..pos + 8].try_into().unwrap();
        out.push((name, bytes[pos + 8..pos + 8 + len].to_vec()));
        pos += 12 + len;
    }
    out
}

fn decode(bytes: &[u8]) -> Decoded {
    let chunks = chunk_list(bytes);
    let cicp = chunks.iter().find(|c| &c.0 == b"cICP").map(|c| c.1[..4].try_into().unwrap());
    let mut dec = png::Decoder::new(Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info().expect("valid PNG");
    let (color, depth, interlaced, width, height, palette_len) = {
        let i = reader.info();
        (i.color_type, i.bit_depth as u8, i.interlaced, i.width, i.height, i.palette.as_ref().map(|p| p.len() / 3))
    };
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let out = reader.next_frame(&mut buf).expect("decodes");
    buf.truncate(out.buffer_size());
    Decoded {
        width,
        height,
        color,
        depth,
        interlaced,
        palette_len,
        chunks: chunks.iter().map(|c| c.0).collect(),
        cicp,
        data: buf,
        out_color: out.color_type,
    }
}

fn has(d: &Decoded, name: &[u8; 4]) -> bool {
    d.chunks.iter().any(|c| c == name)
}

fn enc(px: &[f32], w: u32, h: u32, o: &PngOptions) -> ExportResult {
    encode_png(px, w, h, o, &mut |_, _| {}, &|| false).expect("encode")
}

fn channels(c: png::ColorType) -> usize {
    match c {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => 1,
    }
}

/// RGBA8 view of an expanded 8-bit decode.
fn rgba8(d: &Decoded) -> Vec<[u8; 4]> {
    assert!(d.depth <= 8 || d.out_color == png::ColorType::Indexed);
    let ch = channels(d.out_color);
    // EXPAND unpacks low bit-depth gray to 8 bits.
    d.data
        .chunks(ch)
        .map(|p| match ch {
            1 => [p[0], p[0], p[0], 255],
            2 => [p[0], p[0], p[0], p[1]],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        })
        .collect()
}

// ------------------------------------------------------------------ tests

#[test]
fn rgb8_pixel_accuracy_and_icc() {
    let (w, h) = (64, 48);
    let (px, cols) = random_srgb8(w, h, 1);
    let o = PngOptions { color_mode: ColorMode::Rgb, bit_depth: BitDepth::Eight, ..Default::default() };
    let r = enc(&px, w, h, &o);
    let d = decode(&r.bytes);
    assert_eq!((d.width, d.height), (w, h));
    assert_eq!((r.width, r.height), (w, h));
    assert_eq!(d.color, png::ColorType::Rgb);
    assert_eq!(d.depth, 8);
    assert_eq!(r.color_mode, ColorMode::Rgb);
    assert!(has(&d, b"iCCP") && !has(&d, b"sRGB") && !has(&d, b"cICP"));
    for (p, c) in rgba8(&d).iter().zip(&cols) {
        for i in 0..3 {
            assert!((p[i] as i32 - c[i] as i32).abs() <= 1, "{p:?} vs {c:?}");
        }
    }
}

#[test]
fn auto_drops_alpha_for_opaque() {
    let (px, _) = random_srgb8(40, 30, 2);
    let r = enc(&px, 40, 30, &PngOptions::default());
    let d = decode(&r.bytes);
    assert_eq!(d.color, png::ColorType::Rgb);
    assert_eq!(r.color_mode, ColorMode::Rgb);
    assert_eq!(r.bit_depth, 8);
}

#[test]
fn auto_keeps_alpha_and_cleans_transparent() {
    let (w, h) = (32, 32);
    let px = from_srgb(w, h, |x, _| {
        let a = if x < 8 { 0.0 } else { 0.5 };
        [r_unit(x), 0.3, 0.7, a]
    });
    fn r_unit(x: u32) -> f32 {
        (x * 37 % 101) as f32 / 100.0
    }
    let res = enc(&px, w, h, &PngOptions { color_mode: ColorMode::Rgba, ..Default::default() });
    let d = decode(&res.bytes);
    assert_eq!(d.color, png::ColorType::Rgba);
    for (i, p) in rgba8(&d).iter().enumerate() {
        let x = i as u32 % w;
        if x < 8 {
            assert_eq!(p[3], 0);
        } else {
            assert!((p[3] as i32 - 128).abs() <= 1);
        }
    }
}

#[test]
fn auto_gray_and_srgb_chunk_for_gray_icc() {
    let (w, h) = (50, 20);
    let px = from_srgb(w, h, |x, y| {
        let g = ((x * 7 + y * 3) % 256) as f32 / 255.0;
        [g, g, g, 1.0]
    });
    let r = enc(&px, w, h, &PngOptions::default());
    let d = decode(&r.bytes);
    assert_eq!(d.color, png::ColorType::Grayscale);
    assert_eq!(r.color_mode, ColorMode::Gray);
    // Gray PNG can't carry an RGB ICC profile: sRGB chunk instead.
    assert!(has(&d, b"sRGB") && !has(&d, b"iCCP"));
    for (i, p) in rgba8(&d).iter().enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        assert!((p[0] as i32 - ((x * 7 + y * 3) % 256) as i32).abs() <= 1);
    }
    // For Adobe RGB + ICC, gray reduction must be avoided (RGB ICC profile kept).
    let r = enc(&px, w, h, &PngOptions { target_space: RgbSpace::AdobeRgb, ..Default::default() });
    let d = decode(&r.bytes);
    assert!(matches!(d.color, png::ColorType::Rgb | png::ColorType::Indexed), "{:?}", d.color);
    assert!(has(&d, b"iCCP"));
}

#[test]
fn auto_lossless_palette_for_few_colors() {
    let (w, h) = (60, 40);
    let cols = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [10, 20, 30]];
    let px = from_srgb(w, h, |x, y| {
        let c = cols[((x / 7 + y / 5) % 5) as usize];
        [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0]
    });
    let r = enc(&px, w, h, &PngOptions::default());
    let d = decode(&r.bytes);
    assert_eq!(d.color, png::ColorType::Indexed);
    assert!(d.palette_len.unwrap() <= 5);
    assert!(d.depth <= 4);
    assert_eq!(r.color_mode, ColorMode::Palette);
    for (i, p) in rgba8(&d).iter().enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let c = cols[((x / 7 + y / 5) % 5) as usize];
        for k in 0..3 {
            assert!((p[k] as i32 - c[k] as i32).abs() <= 1, "{p:?} {c:?}");
        }
    }
}

#[test]
fn sixteen_bit_accuracy() {
    for space in [RgbSpace::Srgb, RgbSpace::Rec2020, RgbSpace::Rec2100Pq] {
        let (w, h) = (256, 16);
        let px: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let t = i as f32 / (w * h) as f32;
                [t * 0.9, (1.0 - t) * 0.8, (t * 13.0).fract() * 0.7, 1.0]
            })
            .collect();
        let o = PngOptions { bit_depth: BitDepth::Sixteen, target_space: space, ..Default::default() };
        let r = enc(&px, w, h, &o);
        let d = decode(&r.bytes);
        assert_eq!(d.depth, 16);
        assert_eq!(r.bit_depth, 16);
        assert_eq!(d.color, png::ColorType::Rgb);
        let conv = SpaceConverter::new(space);
        for (i, p) in d.data.chunks(6).enumerate() {
            let e = conv.from_working([px[i * 4], px[i * 4 + 1], px[i * 4 + 2]]);
            for k in 0..3 {
                let got = u16::from_be_bytes([p[2 * k], p[2 * k + 1]]) as f64 / 65535.0;
                let exp = e[k].clamp(0.0, 1.0) as f64;
                assert!((got - exp).abs() <= 1.0 / 65535.0 + 1e-6, "{space:?} px {i} ch {k}: {got} vs {exp}");
            }
        }
    }
}

#[test]
fn color_tags() {
    let (px, _) = random_srgb8(16, 16, 4);
    let t = |tag, space| {
        let r = enc(&px, 16, 16, &PngOptions { color_tag: tag, target_space: space, ..Default::default() });
        (decode(&r.bytes), r.notes)
    };
    let (d, _) = t(ColorTag::Cicp, RgbSpace::DisplayP3);
    assert_eq!(d.cicp, Some([12, 13, 0, 1]));
    assert!(!has(&d, b"iCCP"));
    let (d, _) = t(ColorTag::Cicp, RgbSpace::Rec2100Pq);
    assert_eq!(d.cicp, Some([9, 16, 0, 1]));
    let (d, _) = t(ColorTag::Cicp, RgbSpace::Rec2100Hlg);
    assert_eq!(d.cicp, Some([9, 18, 0, 1]));
    let (d, notes) = t(ColorTag::Cicp, RgbSpace::AdobeRgb);
    assert!(has(&d, b"iCCP") && d.cicp.is_none());
    assert!(!notes.is_empty());
    let (d, _) = t(ColorTag::SrgbChunk, RgbSpace::Srgb);
    assert!(has(&d, b"sRGB") && !has(&d, b"iCCP"));
    let (d, _) = t(ColorTag::SrgbChunk, RgbSpace::DisplayP3);
    assert!(has(&d, b"iCCP") && !has(&d, b"sRGB"));
    let (d, _) = t(ColorTag::Icc, RgbSpace::ProPhoto);
    assert!(has(&d, b"iCCP"));
    let (d, _) = t(ColorTag::None, RgbSpace::Srgb);
    assert!(!has(&d, b"iCCP") && !has(&d, b"sRGB") && !has(&d, b"cICP"));
    // Tag chunks precede IDAT; text chunk only when metadata is kept.
    let r = enc(&px, 16, 16, &PngOptions { strip_metadata: false, ..Default::default() });
    let d = decode(&r.bytes);
    let pos = |n: &[u8; 4]| d.chunks.iter().position(|c| c == n).unwrap();
    assert!(pos(b"iCCP") < pos(b"IDAT"));
    assert!(has(&d, b"tEXt"));
    assert!(!has(&enc(&px, 16, 16, &PngOptions::default()).bytes.pipe(|b| decode(&b)), b"tEXt"));
}

trait Pipe: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> Pipe for T {}

#[test]
fn palette_respects_color_limit() {
    let (w, h) = (96, 64);
    let px = noisy(w, h, 0.3, 5);
    for n in [2u16, 16, 100] {
        let o = PngOptions { color_mode: ColorMode::Palette, palette_colors: n, ..Default::default() };
        let r = enc(&px, w, h, &o);
        let d = decode(&r.bytes);
        assert_eq!(d.color, png::ColorType::Indexed);
        assert!(d.palette_len.unwrap() <= n as usize, "{n}: {:?}", d.palette_len);
        assert!(r.palette_colors.unwrap() <= n);
        let mut uniq: Vec<[u8; 4]> = rgba8(&d);
        uniq.sort();
        uniq.dedup();
        assert!(uniq.len() <= n as usize);
    }
}

#[test]
fn palette_with_alpha() {
    let (w, h) = (64, 64);
    let px = from_srgb(w, h, |x, y| {
        let a = if x < 16 { 0.0 } else { y as f32 / (h - 1) as f32 };
        [x as f32 / w as f32, 0.5, 1.0 - y as f32 / h as f32, a]
    });
    let o = PngOptions { color_mode: ColorMode::Palette, palette_colors: 32, dither: 0.5, ..Default::default() };
    let r = enc(&px, w, h, &o);
    let d = decode(&r.bytes);
    assert_eq!(d.color, png::ColorType::Indexed);
    assert!(d.palette_len.unwrap() <= 32);
    assert!(has(&d, b"tRNS"));
    let p = rgba8(&d);
    for y in 0..h {
        for x in 0..16 {
            assert_eq!(p[(y * w + x) as usize][3], 0);
        }
        // Opaque bottom row stays (nearly) opaque.
    }
    let bottom = &p[((h - 1) * w + 16) as usize..(h * w) as usize];
    assert!(bottom.iter().all(|q| q[3] > 200), "{:?}", &bottom[..4]);
}

#[test]
fn low_bit_depths() {
    let (w, h) = (37, 21);
    let px = noisy(w, h, 0.1, 6);
    // 1-bit gray
    let r = enc(&px, w, h, &PngOptions { color_mode: ColorMode::Gray, bit_depth: BitDepth::One, ..Default::default() });
    let d = decode(&r.bytes);
    assert_eq!((d.color, d.depth), (png::ColorType::Grayscale, 1));
    assert!(rgba8(&d).iter().all(|p| p[0] == 0 || p[0] == 255));
    // 4-bit palette
    let r = enc(&px, w, h, &PngOptions { bit_depth: BitDepth::Four, ..Default::default() });
    let d = decode(&r.bytes);
    assert_eq!(d.color, png::ColorType::Indexed);
    assert!(d.depth <= 4 && d.palette_len.unwrap() <= 16);
    // 2-bit palette
    let r = enc(&px, w, h, &PngOptions { color_mode: ColorMode::Palette, bit_depth: BitDepth::Two, ..Default::default() });
    let d = decode(&r.bytes);
    assert_eq!((d.color, d.depth), (png::ColorType::Indexed, 2));
    assert!(d.palette_len.unwrap() <= 4);
}

#[test]
fn interlace_and_filters() {
    let (w, h) = (45, 33);
    let (px, cols) = random_srgb8(w, h, 7);
    for filter in [FilterStrategy::Auto, FilterStrategy::Paeth, FilterStrategy::MinSum, FilterStrategy::Brute] {
        let o = PngOptions { interlace: true, filter, effort: 1, ..Default::default() };
        let r = enc(&px, w, h, &o);
        let d = decode(&r.bytes);
        assert!(d.interlaced);
        for (p, c) in rgba8(&d).iter().zip(&cols) {
            for i in 0..3 {
                assert!((p[i] as i32 - c[i] as i32).abs() <= 1);
            }
        }
    }
}

#[test]
fn effort_and_zopfli_shrink_or_equal() {
    let (w, h) = (128, 96);
    let px = noisy(w, h, 0.05, 8);
    let s0 = enc(&px, w, h, &PngOptions { effort: 0, ..Default::default() }).bytes.len();
    let s4 = enc(&px, w, h, &PngOptions { effort: 4, ..Default::default() }).bytes.len();
    let sz = enc(&px, w, h, &PngOptions { effort: 2, zopfli: true, zopfli_iterations: 5, ..Default::default() })
        .bytes
        .len();
    assert!(s4 <= s0, "{s4} > {s0}");
    assert!(sz <= s0 + 64, "{sz} > {s0}");
}

#[test]
fn explicit_scale() {
    let (px, _) = random_srgb8(40, 20, 9);
    let r = enc(&px, 40, 20, &PngOptions { scale: 0.5, ..Default::default() });
    assert_eq!((r.width, r.height), (20, 10));
    let d = decode(&r.bytes);
    assert_eq!((d.width, d.height), (20, 10));
    assert!((r.scale - 0.5).abs() < 1e-6);
}

#[test]
fn target_size_palette_only() {
    let (w, h) = (512, 512);
    let px = noisy(w, h, 0.25, 10);
    let max = 40_000;
    let lossless = enc(&px, w, h, &PngOptions::default()).bytes.len();
    assert!(lossless > max as usize);
    let o = PngOptions {
        target: Some(TargetSize {
            max_bytes: max,
            allow_resize: false,
            min_scale: 0.25,
            priority: TargetPriority::Balanced,
            allow_lossy_palette: true,
        }),
        ..Default::default()
    };
    let r = enc(&px, w, h, &o);
    assert_eq!(r.met_target, Some(true), "{:?}", r.notes);
    assert!(r.bytes.len() as u64 <= max, "{}", r.bytes.len());
    assert_eq!((r.width, r.height), (w, h));
    assert_eq!(r.color_mode, ColorMode::Palette);
    let d = decode(&r.bytes);
    assert_eq!((d.width, d.height), (w, h));
    eprintln!("palette-only: {} bytes, notes {:?}", r.bytes.len(), r.notes);
}

#[test]
fn target_size_with_resize() {
    let (w, h) = (512, 512);
    let px = noisy(w, h, 0.25, 11);
    let max = 40_000;
    for (pal, prio) in [
        (false, TargetPriority::Balanced),
        (true, TargetPriority::ColorsFirst),
        (true, TargetPriority::ResolutionFirst),
        (true, TargetPriority::Balanced),
    ] {
        let o = PngOptions {
            target: Some(TargetSize {
                max_bytes: max,
                allow_resize: true,
                min_scale: 0.1,
                priority: prio,
                allow_lossy_palette: pal,
            }),
            ..Default::default()
        };
        let r = enc(&px, w, h, &o);
        assert_eq!(r.met_target, Some(true), "{pal} {prio:?}: {:?}", r.notes);
        assert!(r.bytes.len() as u64 <= max);
        let d = decode(&r.bytes);
        assert_eq!((d.width, d.height), (r.width, r.height));
        if !pal {
            assert!(r.scale < 1.0 && r.width < w, "{:?}", r.notes);
        }
        eprintln!("{pal} {prio:?}: {} bytes {}x{} scale {:.3} {:?}", r.bytes.len(), r.width, r.height, r.scale, r.notes);
    }
}

#[test]
fn target_size_unreachable() {
    let (w, h) = (128, 128);
    let px = noisy(w, h, 1.0, 12);
    let o = PngOptions {
        target: Some(TargetSize {
            max_bytes: 100,
            allow_resize: false,
            min_scale: 1.0,
            priority: TargetPriority::Balanced,
            allow_lossy_palette: true,
        }),
        ..Default::default()
    };
    let r = enc(&px, w, h, &o);
    assert_eq!(r.met_target, Some(false));
    assert!(!r.notes.is_empty());
    decode(&r.bytes);
}

#[test]
fn target_already_fits() {
    let (px, _) = random_srgb8(20, 20, 13);
    let o = PngOptions {
        target: Some(TargetSize {
            max_bytes: 1_000_000,
            allow_resize: true,
            min_scale: 0.5,
            priority: TargetPriority::Balanced,
            allow_lossy_palette: true,
        }),
        ..Default::default()
    };
    let r = enc(&px, 20, 20, &o);
    assert_eq!(r.met_target, Some(true));
    assert_eq!(r.scale, 1.0);
    assert_eq!(r.color_mode, ColorMode::Rgb);
}

#[test]
fn cancel_works() {
    let (px, _) = random_srgb8(64, 64, 14);
    let e = encode_png(&px, 64, 64, &PngOptions::default(), &mut |_, _| {}, &|| true);
    assert!(matches!(e, Err(ExportError::Cancelled)));
    // Cancel in the middle of a target-size search.
    let px = noisy(256, 256, 0.3, 15);
    let calls = std::cell::Cell::new(0);
    let o = PngOptions {
        target: Some(TargetSize {
            max_bytes: 5_000,
            allow_resize: true,
            min_scale: 0.1,
            priority: TargetPriority::Balanced,
            allow_lossy_palette: true,
        }),
        ..Default::default()
    };
    let mut progressed = 0;
    let e = encode_png(&px, 256, 256, &o, &mut |_, _| progressed += 1, &|| {
        calls.set(calls.get() + 1);
        calls.get() > 4
    });
    assert!(matches!(e, Err(ExportError::Cancelled)));
    assert!(progressed > 0);
}

#[test]
fn invalid_input() {
    let e = encode_png(&[0.0; 10], 2, 2, &PngOptions::default(), &mut |_, _| {}, &|| false);
    assert!(matches!(e, Err(ExportError::InvalidInput(_))));
    assert_eq!(estimate_size(&[0.0; 10], 2, 2, &PngOptions::default()), 0);
}

#[test]
fn resize_constant_stays_constant() {
    let c = [0.2f32, 0.5, 0.9, 0.75];
    let px: Vec<f32> = (0..17 * 13).flat_map(|_| c).collect();
    for f in [ResizeFilter::Lanczos3, ResizeFilter::Mitchell, ResizeFilter::Bilinear, ResizeFilter::Nearest] {
        for (nw, nh) in [(5, 4), (40, 31), (17, 6), (1, 1)] {
            let out = resize_rgba(&px, 17, 13, nw, nh, f);
            assert_eq!(out.len(), (nw * nh * 4) as usize);
            for p in out.chunks(4) {
                for k in 0..4 {
                    assert!((p[k] - c[k]).abs() < 1e-4, "{f:?} {nw}x{nh}: {p:?}");
                }
            }
        }
    }
}

#[test]
fn resize_checker_averages() {
    let px = [1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    for f in [ResizeFilter::Lanczos3, ResizeFilter::Mitchell, ResizeFilter::Bilinear] {
        let out = resize_rgba(&px, 2, 2, 1, 1, f);
        for k in 0..3 {
            assert!((out[k] - 0.5).abs() < 1e-5, "{f:?}: {out:?}");
        }
        assert!((out[3] - 1.0).abs() < 1e-5);
    }
    // Larger checker 2x downscale averages too.
    let n = 16u32;
    let big: Vec<f32> = (0..n * n)
        .flat_map(|i| {
            let v = ((i % n + i / n) % 2) as f32;
            [v, v, v, 1.0]
        })
        .collect();
    let m = (n / 2) as usize;
    for f in [ResizeFilter::Lanczos3, ResizeFilter::Mitchell, ResizeFilter::Bilinear] {
        let out = resize_rgba(&big, n, n, n / 2, n / 2, f);
        // Interior samples (edges are renormalized over the clipped support).
        for y in 2..m - 2 {
            for x in 2..m - 2 {
                let v = out[(y * m + x) * 4];
                assert!((v - 0.5).abs() < 1e-4, "{f:?} ({x},{y}) = {v}");
            }
        }
    }
}

#[test]
fn resize_premultiplied_no_halo() {
    // Opaque red next to fully transparent green: the colour must stay red.
    let px = [1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let out = resize_rgba(&px, 2, 1, 1, 1, ResizeFilter::Bilinear);
    assert!((out[0] - 1.0).abs() < 1e-5 && out[1].abs() < 1e-5, "{out:?}");
    assert!((out[3] - 0.5).abs() < 1e-5);
}

#[test]
fn estimate_is_reasonable() {
    let (w, h) = (1200, 900);
    let px = noisy(w, h, 0.08, 16);
    for o in [
        PngOptions::default(),
        PngOptions { effort: 4, ..Default::default() },
        PngOptions { color_mode: ColorMode::Palette, palette_colors: 64, ..Default::default() },
        PngOptions { scale: 0.5, ..Default::default() },
    ] {
        let est = estimate_size(&px, w, h, &o) as f64;
        let real = enc(&px, w, h, &o).bytes.len() as f64;
        let ratio = est / real;
        eprintln!("estimate {est} real {real} ratio {ratio:.3}");
        assert!((0.7..1.4).contains(&ratio), "estimate {est} vs real {real}");
    }
}

/// Timing report (run with `cargo test --release -- --ignored --nocapture perf`).
#[test]
#[ignore]
fn perf_12mp() {
    let (w, h) = (4000, 3000);
    let px = noisy(w, h, 0.04, 17);
    let t = std::time::Instant::now();
    let est = estimate_size(&px, w, h, &PngOptions::default());
    let te = t.elapsed();
    let t = std::time::Instant::now();
    let est2 = estimate_size(&px, w, h, &PngOptions::default());
    let te2 = t.elapsed();
    for effort in [0u8, 2] {
        let t = std::time::Instant::now();
        let r = enc(&px, w, h, &PngOptions { effort, ..Default::default() });
        eprintln!("12MP encode effort {effort}: {:?}, {} bytes", t.elapsed(), r.bytes.len());
    }
    let t = std::time::Instant::now();
    let r = enc(&px, w, h, &PngOptions { color_mode: ColorMode::Palette, ..Default::default() });
    eprintln!("12MP palette-256 effort 2: {:?}, {} bytes", t.elapsed(), r.bytes.len());
    eprintln!("estimate_size: {te:?} (warm {te2:?}) -> {est} / {est2} bytes");
}
