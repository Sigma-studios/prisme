//! System clipboard (text and images).

use crate::pending::Pending;

/// A clipboard image: straight (non-premultiplied) alpha, sRGB-encoded 8-bit RGBA, row-major,
/// `rgba.len() == width * height * 4`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipboardImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Encode a clipboard image as PNG.
pub fn encode_png(img: &ClipboardImage) -> Result<Vec<u8>, String> {
    let expected = img.width as usize * img.height as usize * 4;
    if img.rgba.len() != expected || img.width == 0 || img.height == 0 {
        return Err(format!(
            "bad image buffer: {}x{} needs {expected} bytes, got {}",
            img.width,
            img.height,
            img.rgba.len()
        ));
    }
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, img.width, img.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        enc.set_compression(png::Compression::Fast);
        let mut w = enc.write_header().map_err(|e| e.to_string())?;
        w.write_image_data(&img.rgba).map_err(|e| e.to_string())?;
        w.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Decode any PNG into straight-alpha RGBA8 (16-bit is truncated, gray/palette expanded).
pub fn decode_png(bytes: &[u8]) -> Result<ClipboardImage, String> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| "png too large".to_string())?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width, info.height);
    let px = w as usize * h as usize;
    let data = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => data.to_vec(),
        png::ColorType::Rgb => data
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => data
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[0], c[0], c[1]])
            .collect(),
        png::ColorType::Grayscale => data.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return Err("unexpanded indexed png".into()),
    };
    if rgba.len() != px * 4 {
        return Err("png decode size mismatch".into());
    }
    Ok(ClipboardImage {
        width: w,
        height: h,
        rgba,
    })
}

// ---------------------------------------------------------------------------------------------
// Native (arboard)

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::ClipboardImage;
    use crate::pending::{Pending, spawn};
    use std::sync::Mutex;

    // One long-lived clipboard instance: on Linux/X11 arboard serves the clipboard contents
    // from a background thread that lives as long as a `Clipboard` exists.
    static CLIPBOARD: Mutex<Option<arboard::Clipboard>> = Mutex::new(None);

    fn with<R>(
        f: impl FnOnce(&mut arboard::Clipboard) -> Result<R, arboard::Error>,
    ) -> Result<R, String> {
        let mut g = CLIPBOARD.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            *g = Some(arboard::Clipboard::new().map_err(|e| e.to_string())?);
        }
        f(g.as_mut().unwrap()).map_err(|e| e.to_string())
    }

    pub fn set_image(img: ClipboardImage) -> Pending<Result<(), String>> {
        spawn(move || {
            let expected = img.width as usize * img.height as usize * 4;
            if img.rgba.len() != expected {
                return Err("bad image buffer size".to_string());
            }
            with(|c| {
                c.set_image(arboard::ImageData {
                    width: img.width as usize,
                    height: img.height as usize,
                    bytes: img.rgba.into(),
                })
            })
        })
    }

    pub fn get_image() -> Pending<Option<ClipboardImage>> {
        spawn(|| {
            with(|c| c.get_image()).ok().map(|d| ClipboardImage {
                width: d.width as u32,
                height: d.height as u32,
                rgba: d.bytes.into_owned(),
            })
        })
    }

    pub fn set_text(s: String) {
        let _ = spawn(move || {
            if let Err(e) = with(|c| c.set_text(s)) {
                log::warn!("clipboard_set_text: {e}");
            }
        });
    }

    pub fn get_text() -> Pending<Option<String>> {
        spawn(|| with(|c| c.get_text()).ok())
    }
}

// ---------------------------------------------------------------------------------------------
// Web (async Clipboard API)

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::{ClipboardImage, decode_png, encode_png};
    use crate::pending::{Pending, pending_pair};
    use crate::web::*;
    use wasm_bindgen::JsValue;

    fn clipboard() -> Option<web_sys::Clipboard> {
        let nav = window()?.navigator();
        // `navigator.clipboard` is undefined in insecure contexts.
        get(&nav, "clipboard")?;
        Some(nav.clipboard())
    }

    pub fn set_image(img: ClipboardImage) -> Pending<Result<(), String>> {
        let Some(cb) = clipboard() else {
            return Pending::ready(Err("clipboard API unavailable".into()));
        };
        let png = match encode_png(&img) {
            Ok(p) => p,
            Err(e) => return Pending::ready(Err(e)),
        };
        // Everything up to `write()` runs synchronously so it happens within the user gesture
        // (Safari is strict about this).
        let promise = (|| -> Result<js_sys::Promise, JsValue> {
            let blob = bytes_to_blob(&png, "image/png")?;
            let rec = object();
            set(&rec, "image/png", &blob);
            let item = web_sys::ClipboardItem::new_with_record_from_str_to_blob_promise(&rec)?;
            Ok(cb.write(&js_sys::Array::of1(&item)))
        })();
        let promise = match promise {
            Ok(p) => p,
            Err(e) => return Pending::ready(Err(err_string(&e))),
        };
        let (c, p) = pending_pair();
        wasm_bindgen_futures::spawn_local(async move {
            let r = wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map(|_| ())
                .map_err(|e| err_string(&e));
            c.complete(r);
        });
        p
    }

    pub fn get_image() -> Pending<Option<ClipboardImage>> {
        let Some(cb) = clipboard() else {
            return Pending::ready(None);
        };
        let promise = cb.read();
        let (c, p) = pending_pair();
        wasm_bindgen_futures::spawn_local(async move {
            let res = async {
                let items = wasm_bindgen_futures::JsFuture::from(promise).await.ok()?;
                let items: js_sys::Array = items.into();
                for item in items.iter() {
                    let types: js_sys::Array = get(&item, "types")?.into();
                    let has_png = types
                        .iter()
                        .any(|t| t.as_string().as_deref() == Some("image/png"));
                    if !has_png {
                        continue;
                    }
                    let blob =
                        await_promise(call_method(&item, "getType", &["image/png".into()]).ok()?)
                            .await
                            .ok()?;
                    let bytes = blob_bytes(&blob).await.ok()?;
                    match decode_png(&bytes) {
                        Ok(img) => return Some(img),
                        Err(e) => log::warn!("clipboard png decode: {e}"),
                    }
                }
                None
            };
            c.complete(res.await);
        });
        p
    }

    pub fn set_text(s: String) {
        if let Some(cb) = clipboard() {
            let promise = cb.write_text(&s);
            wasm_bindgen_futures::spawn_local(async move {
                if let Err(e) = wasm_bindgen_futures::JsFuture::from(promise).await {
                    log::warn!("clipboard_set_text: {}", err_string(&e));
                }
            });
        }
    }

    pub fn get_text() -> Pending<Option<String>> {
        let Some(cb) = clipboard() else {
            return Pending::ready(None);
        };
        let promise = cb.read_text();
        let (c, p) = pending_pair();
        wasm_bindgen_futures::spawn_local(async move {
            let r = wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .ok()
                .and_then(|v| v.as_string());
            c.complete(r);
        });
        p
    }
}

/// Put an image on the clipboard.
/// Native: arboard (on a background thread). Web: `navigator.clipboard.write` with an
/// `image/png` `ClipboardItem` (call inside a user gesture).
pub fn clipboard_set_image(img: ClipboardImage) -> Pending<Result<(), String>> {
    imp::set_image(img)
}

/// Read an image from the clipboard; `None` if there is none (or permission was denied).
/// Web: `navigator.clipboard.read()` (the browser may show a permission prompt).
pub fn clipboard_get_image() -> Pending<Option<ClipboardImage>> {
    imp::get_image()
}

/// Put text on the clipboard (fire and forget).
pub fn clipboard_set_text(s: String) {
    imp::set_text(s)
}

/// Read text from the clipboard.
pub fn clipboard_get_text() -> Pending<Option<String>> {
    imp::get_text()
}
