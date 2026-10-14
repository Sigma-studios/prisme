//! JPEG XL through `jxl-oxide` (pure Rust). XYB and enum-encoded images are rendered
//! straight into linear Rec.2020; ICC-tagged and grayscale images go through the ICC path.

use super::ImageDecoder;
use crate::color::{self, ColorSource, Layout, Samples};
use crate::{Decoded, DecodedImage, IoError};
use jxl_oxide::color::{ColourEncoding, ColourSpace, Primaries, TransferFunction, WhitePoint};
use jxl_oxide::{EnumColourEncoding, JxlImage, PixelFormat, RenderingIntent};
use std::io::Cursor;

pub struct Jxl;

impl ImageDecoder for Jxl {
    fn name(&self) -> &'static str {
        "JPEG XL"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["jxl"]
    }

    fn sniff(&self, h: &[u8]) -> bool {
        h.starts_with(&[0xFF, 0x0A]) || h.starts_with(&[0, 0, 0, 0x0C, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A])
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        decode_jxl(bytes).map(Decoded::Raster)
    }
}

fn decode_jxl(bytes: &[u8]) -> Result<DecodedImage, IoError> {
    let builder = JxlImage::builder();
    #[cfg(target_arch = "wasm32")]
    let builder = builder.pool(jxl_oxide::JxlThreadPool::none());
    let mut image = builder.read(Cursor::new(bytes)).map_err(IoError::decode)?;
    let meta = &image.image_header().metadata;
    let (bits, float) = match meta.bit_depth {
        jxl_oxide::image::BitDepth::IntegerSample { bits_per_sample } => (bits_per_sample, false),
        jxl_oxide::image::BitDepth::FloatSample { bits_per_sample, .. } => (bits_per_sample, true),
    };
    let associated_alpha = meta.alpha().and_then(|i| meta.ec_info[i].alpha_associated()).unwrap_or(false);
    let gray = meta.grayscale();
    let enum_encoded = matches!(meta.colour_encoding, ColourEncoding::Enum(_));
    let original_desc = describe(&image);
    // Render directly in the working space whenever jxl-oxide can do so without a CMS.
    let direct = !gray && (meta.xyb_encoded || enum_encoded);
    if direct {
        image.request_color_encoding(EnumColourEncoding {
            colour_space: ColourSpace::Rgb,
            white_point: WhitePoint::D65,
            primaries: Primaries::Bt2100,
            tf: TransferFunction::Linear,
            rendering_intent: RenderingIntent::Relative,
        });
    }
    let layout = match image.pixel_format() {
        PixelFormat::Gray => Layout::Gray,
        PixelFormat::Graya => Layout::GrayAlpha,
        PixelFormat::Rgb => Layout::Rgb,
        PixelFormat::Rgba => Layout::Rgba,
        PixelFormat::Cmyk => Layout::Cmyk,
        PixelFormat::Cmyka => Layout::Cmyka,
    };
    let render = image.render_frame(0).map_err(IoError::decode)?;
    let mut stream = render.stream();
    let (w, h, ch) = (stream.width(), stream.height(), stream.channels() as usize);
    if ch != layout.channels() {
        return Err(IoError::Unsupported(format!("JPEG XL with {ch} rendered channels")));
    }
    let mut buf = vec![0.0f32; w as usize * h as usize * ch];
    stream.write_to_buffer(&mut buf);
    let source = if direct {
        ColorSource::space(prisme_color::RgbSpace::LinearRec2020)
    } else {
        let icc = image.rendered_icc();
        ColorSource::from_icc(&icc).map(|s| s.0).unwrap_or_else(ColorSource::srgb)
    };
    let mut pixels = color::to_working(Samples::F32(&buf), w, h, layout, &source);
    if associated_alpha && layout.has_alpha() {
        color::unpremultiply(&mut pixels, w);
    }
    let bits = bits.min(32) as u8;
    Ok(DecodedImage::new(w, h, pixels, original_desc, bits, float))
}

fn describe(image: &JxlImage) -> String {
    if let Some(d) = image.original_icc().and_then(ColorSource::from_icc).map(|s| s.1) {
        return d;
    }
    match &image.image_header().metadata.colour_encoding {
        ColourEncoding::Enum(e) if e.is_srgb() => "sRGB".into(),
        ColourEncoding::Enum(e) => match e.cicp() {
            Some([p, t, ..]) => color::cicp(p, t).map(|c| c.1).unwrap_or_else(|| format!("JPEG XL CICP {p}/{t}")),
            None => "JPEG XL custom colour encoding".into(),
        },
        ColourEncoding::IccProfile(_) => "JPEG XL (ICC)".into(),
    }
}
