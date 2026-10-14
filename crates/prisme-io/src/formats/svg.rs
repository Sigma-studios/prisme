//! SVG kept as a vector source; `usvg` parses it for validation and its size.

use super::ImageDecoder;
use crate::{Decoded, IoError};

pub struct Svg;

impl ImageDecoder for Svg {
    fn name(&self) -> &'static str {
        "SVG"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["svg", "svgz"]
    }

    fn sniff(&self, h: &[u8]) -> bool {
        let s = String::from_utf8_lossy(h);
        let s = s.trim_start_matches('\u{feff}').trim_start();
        s.starts_with("<svg") || (s.starts_with("<?xml") && s.contains("<svg"))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Decoded, IoError> {
        decode_svg(bytes)
    }
}

fn decode_svg(bytes: &[u8]) -> Result<Decoded, IoError> {
    let opts = usvg::Options::default();
    let tree = usvg::Tree::from_data(bytes, &opts).map_err(|e| IoError::Decode(format!("SVG: {e}")))?;
    let size = tree.size();
    // Keep the original (decompressed) source so the renderer sees exactly what was authored.
    let svg = if bytes.starts_with(&[0x1f, 0x8b]) {
        let raw = usvg::decompress_svgz(bytes).map_err(|e| IoError::Decode(format!("SVGZ: {e}")))?;
        String::from_utf8(raw).map_err(|e| IoError::Decode(format!("SVG is not UTF-8: {e}")))?
    } else {
        String::from_utf8(bytes.to_vec()).map_err(|e| IoError::Decode(format!("SVG is not UTF-8: {e}")))?
    };
    Ok(Decoded::Vector { svg, width: size.width(), height: size.height() })
}
