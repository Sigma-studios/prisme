//! Decoder tests on small in-memory images.

use super::*;
use image::{DynamicImage, ImageBuffer, ImageFormat};
use prisme_color::{srgb, IccProfile, RgbSpace, SpaceConverter};
use std::io::Cursor;

fn raster(d: Decoded) -> DecodedImage {
    match d {
        Decoded::Raster(i) => i,
        other => panic!("expected raster, got {other:?}"),
    }
}

fn px(img: &DecodedImage, x: u32, y: u32) -> [f32; 4] {
    let i = ((y * img.width + x) * 4) as usize;
    img.pixels[i..i + 4].try_into().unwrap()
}

fn assert_close(a: [f32; 4], b: [f32; 4], tol: f32) {
    for i in 0..4 {
        assert!((a[i] - b[i]).abs() <= tol, "{a:?} vs {b:?} (tol {tol})");
    }
}

fn srgb8(r: u8, g: u8, b: u8, a: u8) -> [f32; 4] {
    let w = srgb().to_working([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]);
    [w[0], w[1], w[2], a as f32 / 255.0]
}

fn encode_png(w: u32, h: u32, ct: png::ColorType, depth: png::BitDepth, data: &[u8], icc: Option<Vec<u8>>) -> Vec<u8> {
    let mut out = vec![];
    {
        let mut info = png::Info::with_size(w, h);
        info.color_type = ct;
        info.bit_depth = depth;
        if let Some(icc) = icc {
            info.icc_profile = Some(icc.into());
        }
        let enc = png::Encoder::with_info(&mut out, info).unwrap();
        let mut wr = enc.write_header().unwrap();
        wr.write_image_data(data).unwrap();
    }
    out
}

#[test]
fn png_8bit_untagged_is_srgb() {
    let data = [255, 0, 0, 255, 0, 255, 0, 128, 10, 20, 30, 0, 200, 150, 100, 255];
    let bytes = encode_png(2, 2, png::ColorType::Rgba, png::BitDepth::Eight, &data, None);
    let img = raster(decode(&bytes, "x.png").unwrap());
    assert_eq!((img.width, img.height, img.source_bit_depth), (2, 2, 8));
    assert_eq!(img.format, PixelFormat::F16);
    assert_eq!(img.color_info, "sRGB (untagged)");
    assert_close(px(&img, 0, 0), srgb8(255, 0, 0, 255), 1e-5);
    assert_close(px(&img, 1, 0), srgb8(0, 255, 0, 128), 1e-5);
    assert_close(px(&img, 1, 1), srgb8(200, 150, 100, 255), 1e-5);
}

#[test]
fn png_16bit_is_lossless() {
    let (w, h) = (17u32, 9u32);
    let vals: Vec<u16> = (0..w * h * 4).map(|i| ((i * 7919 + 13) % 65536) as u16).collect();
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_be_bytes()).collect();
    let bytes = encode_png(w, h, png::ColorType::Rgba, png::BitDepth::Sixteen, &data, None);
    let img = raster(decode(&bytes, "deep.png").unwrap());
    assert_eq!(img.format, PixelFormat::F32);
    assert_eq!(img.source_bit_depth, 16);
    for (p, v) in img.pixels.chunks_exact(4).zip(vals.chunks_exact(4)) {
        let s = srgb().from_working([p[0], p[1], p[2]]);
        for c in 0..3 {
            let back = (s[c] * 65535.0).round();
            assert!((back - v[c] as f32).abs() <= 1.0, "{back} vs {}", v[c]);
        }
        assert_eq!((p[3] * 65535.0).round() as u16, v[3]);
    }
}

#[test]
fn png_icc_profile_is_honoured() {
    let icc = IccProfile::encode_standard(RgbSpace::DisplayP3).unwrap();
    let data = [255, 0, 0, 30, 180, 90];
    let bytes = encode_png(2, 1, png::ColorType::Rgb, png::BitDepth::Eight, &data, Some(icc));
    let img = raster(decode(&bytes, "p3.png").unwrap());
    assert_ne!(img.color_info, "sRGB (untagged)");
    let p3 = SpaceConverter::new(RgbSpace::DisplayP3);
    for (x, rgb) in [(0, [255u8, 0, 0]), (1, [30, 180, 90])] {
        let w = p3.to_working(rgb.map(|v| v as f32 / 255.0));
        assert_close(px(&img, x, 0), [w[0], w[1], w[2], 1.0], 4e-3);
    }
}

#[test]
fn png_gray_and_palette() {
    let bytes = encode_png(2, 1, png::ColorType::GrayscaleAlpha, png::BitDepth::Eight, &[128, 255, 0, 64], None);
    let img = raster(decode(&bytes, "g.png").unwrap());
    assert_close(px(&img, 0, 0), srgb8(128, 128, 128, 255), 1e-5);
    assert_close(px(&img, 1, 0), srgb8(0, 0, 0, 64), 1e-5);

    let mut out = vec![];
    {
        let mut enc = png::Encoder::new(&mut out, 2, 1);
        enc.set_color(png::ColorType::Indexed);
        enc.set_palette(vec![255, 0, 0, 0, 0, 255]);
        enc.set_trns(vec![255, 0]);
        enc.write_header().unwrap().write_image_data(&[0, 1]).unwrap();
    }
    let img = raster(decode(&out, "pal.png").unwrap());
    assert_close(px(&img, 0, 0), srgb8(255, 0, 0, 255), 1e-5);
    assert_eq!(px(&img, 1, 0)[3], 0.0);
}

/// Build a JPEG with four solid quadrants and an EXIF orientation tag.
fn jpeg_with_orientation(orientation: u16) -> Vec<u8> {
    let (w, h) = (64u32, 32u32);
    let img = ImageBuffer::from_fn(w, h, |x, _| if x < w / 2 { image::Rgb([250u8, 10, 10]) } else { image::Rgb([10, 10, 250]) });
    let mut jpg = vec![];
    DynamicImage::ImageRgb8(img).write_to(&mut Cursor::new(&mut jpg), ImageFormat::Jpeg).unwrap();
    // APP1 Exif segment (big-endian TIFF, one IFD entry).
    let mut tiff = b"MM\0*\0\0\0\x08\0\x01".to_vec();
    tiff.extend_from_slice(&[0x01, 0x12, 0, 3, 0, 0, 0, 1]);
    tiff.extend_from_slice(&orientation.to_be_bytes());
    tiff.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(&tiff);
    let mut seg = vec![0xFF, 0xE1];
    seg.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    seg.extend_from_slice(&payload);
    let mut out = jpg[..2].to_vec();
    out.extend_from_slice(&seg);
    out.extend_from_slice(&jpg[2..]);
    out
}

#[test]
fn jpeg_exif_orientation_rotates_pixels() {
    let bytes = jpeg_with_orientation(1);
    let img = raster(decode(&bytes, "a.jpg").unwrap());
    assert_eq!((img.width, img.height), (64, 32));
    let red = srgb8(250, 10, 10, 255);
    assert_close(px(&img, 4, 16), red, 0.03);

    // Orientation 6 = rotate 90° clockwise: the left (red) half ends up on top.
    let img = raster(decode(&jpeg_with_orientation(6), "a.jpg").unwrap());
    assert_eq!((img.width, img.height), (32, 64));
    assert_close(px(&img, 16, 4), red, 0.03);
    assert_close(px(&img, 16, 60), srgb8(10, 10, 250, 255), 0.03);

    let prev = decode_preview(&jpeg_with_orientation(6), "a.jpg", 16).unwrap();
    assert!(prev.width.max(prev.height) <= 16);
    assert!(prev.height > prev.width);
    assert_close(px(&prev, prev.width / 2, 1), red, 0.05);
}

#[test]
fn jpeg_preview_skipped_when_it_would_be_a_full_decode() {
    // 64 px wide: halving gives 32 < 40, so only a full-size decode could serve the preview.
    assert!(decode_preview(&jpeg_with_orientation(1), "a.jpg", 40).is_none());
    assert!(decode_preview(&jpeg_with_orientation(1), "a.jpg", 32).is_some());
}

fn tiff_page_rgb8(enc: &mut tiff::encoder::TiffEncoder<&mut Cursor<Vec<u8>>>, w: u32, h: u32, v: u8) {
    let data = vec![v; (w * h * 3) as usize];
    enc.write_image::<tiff::encoder::colortype::RGB8>(w, h, &data).unwrap();
}

#[test]
fn tiff_multipage_becomes_layers() {
    let mut buf = Cursor::new(Vec::new());
    {
        let mut enc = tiff::encoder::TiffEncoder::new(&mut buf).unwrap();
        tiff_page_rgb8(&mut enc, 8, 4, 50);
        tiff_page_rgb8(&mut enc, 6, 6, 200);
    }
    let bytes = buf.into_inner();
    match decode(&bytes, "pages.tif").unwrap() {
        Decoded::Layers(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!((v[0].1.width, v[0].1.height), (8, 4));
            assert_close(px(&v[1].1, 1, 1), srgb8(200, 200, 200, 255), 1e-5);
            let doc = document_from_decoded("pages.tif", Decoded::Layers(v));
            assert_eq!((doc.info.width, doc.info.height), (8, 6));
            assert_eq!(doc.layers.len(), 2);
            assert_eq!(doc.layers[0].name, "Page 1");
        }
        other => panic!("expected layers, got {other:?}"),
    }
}

#[test]
fn tiff_16bit_and_float() {
    let mut buf = Cursor::new(Vec::new());
    let vals: Vec<u16> = vec![0, 1000, 65535, 40000, 123, 9999];
    tiff::encoder::TiffEncoder::new(&mut buf)
        .unwrap()
        .write_image::<tiff::encoder::colortype::RGB16>(2, 1, &vals)
        .unwrap();
    let img = raster(decode(&buf.into_inner(), "d.tiff").unwrap());
    assert_eq!(img.format, PixelFormat::F32);
    let s = srgb().from_working([px(&img, 1, 0)[0], px(&img, 1, 0)[1], px(&img, 1, 0)[2]]);
    assert!((s[0] * 65535.0 - 40000.0).abs() < 1.0);

    let mut buf = Cursor::new(Vec::new());
    let f = vec![0.5f32, 2.0, 0.0, 1.0, 1.0, 1.0];
    tiff::encoder::TiffEncoder::new(&mut buf)
        .unwrap()
        .write_image::<tiff::encoder::colortype::RGB32Float>(2, 1, &f)
        .unwrap();
    let img = raster(decode(&buf.into_inner(), "f.tif").unwrap());
    let lin = SpaceConverter::new(RgbSpace::LinearSrgb).to_working([0.5, 2.0, 0.0]);
    assert_close(px(&img, 0, 0), [lin[0], lin[1], lin[2], 1.0], 1e-5);
    assert!(img.color_info.contains("Linear"));
}

#[test]
fn image_crate_formats() {
    let src = ImageBuffer::from_fn(5, 3, |x, y| image::Rgba([(x * 50) as u8, (y * 80) as u8, 77, 255]));
    for (fmt, name) in [(ImageFormat::Bmp, "a.bmp"), (ImageFormat::Qoi, "a.qoi"), (ImageFormat::WebP, "a.webp")] {
        let mut bytes = vec![];
        DynamicImage::ImageRgba8(src.clone()).write_to(&mut Cursor::new(&mut bytes), fmt).unwrap();
        // Wrong extension: magic bytes must win.
        let img = raster(decode(&bytes, "misnamed.dat").unwrap_or_else(|e| panic!("{name}: {e}")));
        assert_eq!((img.width, img.height), (5, 3), "{name}");
        assert_close(px(&img, 4, 2), srgb8(200, 160, 77, 255), 1e-5);
    }
}

#[test]
fn exr_is_linear_rec709() {
    use exr::prelude::*;
    let img = exr::image::Image::from_channels(
        (2, 1),
        SpecificChannels::rgba(|p: Vec2<usize>| if p.x() == 0 { (0.5f32, 0.25f32, 4.0f32, 1.0f32) } else { (0.2, 0.2, 0.2, 0.5) }),
    );
    let mut buf = Cursor::new(Vec::new());
    img.write().to_buffered(&mut buf).unwrap();
    let d = raster(decode(&buf.into_inner(), "hdr.exr").unwrap());
    assert_eq!(d.format, PixelFormat::F32);
    let c = SpaceConverter::new(RgbSpace::LinearSrgb);
    let w = c.to_working([0.5, 0.25, 4.0]);
    assert_close(px(&d, 0, 0), [w[0], w[1], w[2], 1.0], 1e-4);
    // Premultiplied in the file -> straight alpha in memory.
    let w = c.to_working([0.4, 0.4, 0.4]);
    assert_close(px(&d, 1, 0), [w[0], w[1], w[2], 0.5], 1e-4);
}

#[test]
fn svg_size_from_attributes_and_viewbox() {
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="120" height="45.5"><rect width="10" height="10"/></svg>"#;
    match decode(svg.as_bytes(), "a.svg").unwrap() {
        Decoded::Vector { width, height, svg: s } => {
            assert_eq!((width, height), (120.0, 45.5));
            assert_eq!(s, svg);
        }
        other => panic!("{other:?}"),
    }
    let svg = r#"<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200"/>"#;
    let d = decode(svg.as_bytes(), "noext").unwrap();
    let Decoded::Vector { width, height, .. } = &d else { panic!() };
    assert_eq!((*width, *height), (300.0, 200.0));
    let doc = document_from_decoded("logo.svg", d);
    assert_eq!((doc.info.width, doc.info.height, doc.info.name.as_str()), (300, 200, "logo"));
    assert!(matches!(doc.layers[0].kind, LayerKind::Vector(_)));
}

#[test]
fn document_building() {
    let pixels: Vec<f32> = (0..300 * 270).flat_map(|i| [0.5, (i % 7) as f32 / 7.0, 0.1, 1.0]).collect();
    let img = DecodedImage::new(300, 270, pixels.clone(), "sRGB (untagged)".into(), 16, false);
    let doc = document_from_decoded("dir/photo.final.png", Decoded::Raster(img.clone()));
    assert_eq!(doc.info.name, "photo.final");
    assert_eq!(doc.info.format, PixelFormat::F32);
    assert_eq!(doc.info.source_profile.as_deref(), Some("sRGB (untagged)"));
    let l = doc.active().unwrap();
    assert_eq!(l.raster().unwrap().read_rect(&doc.bounds()), pixels);

    let mut doc = Document::with_background("bg", 500, 400, Some([1.0; 4]));
    let ids = add_decoded_as_layers(&mut doc, "paste.png", Decoded::Raster(img));
    assert_eq!(ids.len(), 1);
    assert_eq!(doc.layers.last().unwrap().id, ids[0]);
    assert_eq!(doc.active_layer, Some(ids[0]));
    assert_eq!(doc.info.format, PixelFormat::F32);
    // Centred: (500-300)/2 = 100, (400-270)/2 = 65.
    let g = doc.layer(ids[0]).unwrap().raster().unwrap();
    assert_eq!(g.content_bounds(), IRect::new(100, 65, 400, 335));
}

#[test]
fn unsupported_and_registry() {
    assert!(matches!(decode(b"definitely not an image", "x.xyz"), Err(IoError::Unsupported(_))));
    let mut heic = vec![0, 0, 0, 24];
    heic.extend_from_slice(b"ftypheic\0\0\0\0mif1heic");
    if cfg!(not(feature = "heic")) {
        assert!(matches!(decode(&heic, "a.heic"), Err(IoError::Unsupported(_))));
    }
    assert!(decode(&[0x89, b'P', b'N', b'G', 1, 2, 3], "broken.png").is_err());
    let exts = supported_extensions();
    for e in ["png", "jpg", "tif", "webp", "exr", "jxl", "avif", "heic", "cr3", "dng", "svg", "prisme"] {
        assert!(exts.contains(&e), "{e}");
    }
    assert!(is_project_file("a/b/My File.PRISME"));
    assert!(!is_project_file("a.png"));
}

#[test]
fn avif_property_scan() {
    // meta (full box) > iprp > ipco > [colr nclx, irot]
    fn bx(ty: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut v = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(ty);
        v.extend_from_slice(payload);
        v
    }
    let mut colr = b"nclx".to_vec();
    colr.extend_from_slice(&[0, 9, 0, 16, 0, 9, 0x80]);
    let ipco = [bx(b"colr", &colr), bx(b"irot", &[1])].concat();
    let meta = [vec![0, 0, 0, 0], bx(b"iprp", &bx(b"ipco", &ipco))].concat();
    let file = [bx(b"ftyp", b"avif\0\0\0\0avifmif1"), bx(b"meta", &meta)].concat();
    let p = formats::avif_props_for_test(&file);
    assert_eq!(p, (Some((9, 16, 9, true)), 1));
}

/// Compare decodes of real files against reference PNGs (`name.ext` vs `name.png`) found
/// in `$PRISME_IO_SAMPLES`. Run with `cargo test -p prisme-io -- --ignored samples`.
#[test]
#[ignore]
fn samples_match_reference_pngs() {
    let Ok(dir) = std::env::var("PRISME_IO_SAMPLES") else { return };
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if ext == "png" {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(&path).unwrap();
        let t = std::time::Instant::now();
        let img = match decode(&bytes, &name) {
            Ok(Decoded::Raster(i)) => i,
            Ok(other) => {
                println!("{name}: non-raster {:?}", std::mem::discriminant(&other));
                continue;
            }
            Err(e) => {
                println!("{name}: ERROR {e}");
                continue;
            }
        };
        let dt = t.elapsed();
        let t = std::time::Instant::now();
        let doc = document_from_decoded(&name, Decoded::Raster(img.clone()));
        let dt_doc = t.elapsed();
        if let Ok(dump) = std::env::var("PRISME_IO_DUMP") {
            let small = img.clone().downscaled(800);
            let rgba: Vec<u8> = small
                .pixels
                .chunks_exact(4)
                .flat_map(|p| {
                    let s = srgb().from_working([p[0], p[1], p[2]]);
                    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    [q(s[0]), q(s[1]), q(s[2]), q(p[3])]
                })
                .collect();
            let file = std::path::Path::new(&dump).join(format!("{name}.png"));
            image::RgbaImage::from_raw(small.width, small.height, rgba).unwrap().save(file).unwrap();
        }
        let t = std::time::Instant::now();
        let saved = container::save(&prisme_core::History::new(doc), true).unwrap();
        let dt_save = t.elapsed();
        let t = std::time::Instant::now();
        container::load(&saved).unwrap();
        println!("  .prisme: {} bytes, save {dt_save:?}, load {:?}", saved.len(), t.elapsed());
        let prev = decode_preview(&bytes, &name, 256).map(|p| (p.width, p.height));
        let reference = path.with_extension("png");
        let err = std::fs::read(&reference).ok().and_then(|r| {
            let r = raster(decode(&r, "ref.png").unwrap());
            if (r.width, r.height) != (img.width, img.height) {
                println!("  reference size {}x{} differs", r.width, r.height);
                return None;
            }
            let mut sum = 0.0f64;
            for (a, b) in img.pixels.chunks_exact(4).zip(r.pixels.chunks_exact(4)) {
                let sa = srgb().from_working([a[0], a[1], a[2]]);
                let sb = srgb().from_working([b[0], b[1], b[2]]);
                sum += (0..3).map(|c| (sa[c] - sb[c]).abs() as f64).sum::<f64>() / 3.0;
            }
            Some(sum / (img.width as f64 * img.height as f64) * 255.0)
        });
        println!(
            "{name}: {}x{} {:?} '{}' {}-bit in {dt:?} (+{dt_doc:?} to tiles), preview {prev:?}, mean abs err vs png: {err:?} (8-bit units)",
            img.width, img.height, img.format, img.color_info, img.source_bit_depth
        );
    }
}
