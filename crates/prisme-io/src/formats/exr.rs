//! OpenEXR through the `exr` crate. EXR is scene-linear with Rec.709 primaries unless a
//! `chromaticities` attribute says otherwise; alpha is premultiplied in the file.

use super::ImageDecoder;
use crate::color::{self, Analytic, ColorSource, Layout, Samples};
use crate::{Decoded, DecodedImage, IoError};
use exr::prelude::{read, ReadChannels, ReadLayers, RgbaChannels, Vec2};
use prisme_color::{RgbSpace, Transfer};
use std::io::Cursor;

pub struct Exr;

impl ImageDecoder for Exr {
    fn name(&self) -> &'static str {
        "OpenEXR"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["exr"]
    }

    fn sniff(&self, h: &[u8]) -> bool {
        h.starts_with(&[0x76, 0x2F, 0x31, 0x01])
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        decode_exr(bytes).map(Decoded::Raster)
    }
}

struct Rgba {
    width: usize,
    data: Vec<f32>,
}

fn source_for(chroma: Option<&exr::meta::attribute::Chromaticities>) -> (ColorSource, String) {
    let rec709 = RgbSpace::LinearSrgb.primaries();
    match chroma {
        Some(c) => {
            let p = |v: Vec2<f32>| [v.0 as f64, v.1 as f64];
            let (r, g, b, w) = (p(c.red), p(c.green), p(c.blue), p(c.white));
            let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3;
            if close(r, rec709.0) && close(g, rec709.1) && close(b, rec709.2) && close(w, rec709.3) {
                (ColorSource::space(RgbSpace::LinearSrgb), "Linear Rec.709".into())
            } else {
                let a = Analytic::primaries(Transfer::Linear, r, g, b, w);
                (ColorSource::Analytic(a), format!("Linear, primaries R{r:?} G{g:?} B{b:?} W{w:?}"))
            }
        }
        None => (ColorSource::space(RgbSpace::LinearSrgb), "Linear Rec.709".into()),
    }
}

fn decode_exr(bytes: &[u8]) -> Result<DecodedImage, IoError> {
    let rgba = read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |size, _: &RgbaChannels| Rgba { width: size.width(), data: vec![0.0; size.width() * size.height() * 4] },
            |px: &mut Rgba, pos: Vec2<usize>, (r, g, b, a): (f32, f32, f32, f32)| {
                let i = (pos.y() * px.width + pos.x()) * 4;
                px.data[i..i + 4].copy_from_slice(&[r, g, b, a]);
            },
        )
        .first_valid_layer()
        .all_attributes()
        .from_buffered(Cursor::new(bytes));
    let (w, h, layout, data, chroma) = match rgba {
        Ok(img) => {
            let size = img.layer_data.size;
            (size.width(), size.height(), Layout::Rgba, img.layer_data.channel_data.pixels.data, img.attributes.chromaticities)
        }
        Err(e) => match decode_luminance(bytes) {
            Some(r) => r,
            None => return Err(IoError::decode(e)),
        },
    };
    let (source, desc) = source_for(chroma.as_ref());
    let (w32, h32) = (w as u32, h as u32);
    let mut pixels = color::to_working(Samples::F32(&data), w32, h32, layout, &source);
    color::unpremultiply(&mut pixels, w32);
    Ok(DecodedImage::new(w32, h32, pixels, desc, 32, true))
}

/// Luminance-only ("Y", optional "A") EXR files.
type Plane = (usize, usize, Layout, Vec<f32>, Option<exr::meta::attribute::Chromaticities>);

fn decode_luminance(bytes: &[u8]) -> Option<Plane> {
    let img = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .from_buffered(Cursor::new(bytes))
        .ok()?;
    let size = img.layer_data.size;
    let chans = &img.layer_data.channel_data.list;
    let find = |name: &str| chans.iter().find(|c| c.name.eq(name)).map(|c| c.sample_data.values_as_f32().collect::<Vec<f32>>());
    let y = find("Y")?;
    let n = size.width() * size.height();
    let data: Vec<f32> = match find("A") {
        Some(a) => (0..n).flat_map(|i| [y[i], a[i]]).collect(),
        None => y,
    };
    let layout = if data.len() == n * 2 { Layout::GrayAlpha } else { Layout::Gray };
    // The pipeline treats gray as equal RGB, which keeps luminance for any primaries.
    Some((size.width(), size.height(), layout, data, img.attributes.chromaticities))
}
