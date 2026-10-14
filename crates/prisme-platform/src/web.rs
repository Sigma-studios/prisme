//! Small JS interop helpers (wasm32 only).

use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

pub(crate) fn window() -> Option<web_sys::Window> {
    web_sys::window()
}

pub(crate) fn get(obj: &JsValue, key: &str) -> Option<JsValue> {
    Reflect::get(obj, &JsValue::from_str(key))
        .ok()
        .filter(|v| !v.is_undefined() && !v.is_null())
}

pub(crate) fn set(obj: &JsValue, key: &str, value: &JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

/// Call `obj[method](...args)`.
pub(crate) fn call_method(
    obj: &JsValue,
    method: &str,
    args: &[JsValue],
) -> Result<JsValue, JsValue> {
    let f: Function = get(obj, method)
        .ok_or_else(|| JsValue::from_str(&format!("no method {method}")))?
        .dyn_into()?;
    let arr = js_sys::Array::new();
    for a in args {
        arr.push(a);
    }
    f.apply(obj, &arr)
}

pub(crate) async fn await_promise(v: JsValue) -> Result<JsValue, JsValue> {
    let p: Promise = v.dyn_into()?;
    JsFuture::from(p).await
}

pub(crate) fn err_string(e: &JsValue) -> String {
    if let Some(s) = e.as_string() {
        return s;
    }
    if let Some(m) = get(e, "message").and_then(|m| m.as_string()) {
        let name = get(e, "name")
            .and_then(|n| n.as_string())
            .unwrap_or_default();
        return if name.is_empty() {
            m
        } else {
            format!("{name}: {m}")
        };
    }
    format!("{e:?}")
}

pub(crate) fn is_abort(e: &JsValue) -> bool {
    get(e, "name").and_then(|n| n.as_string()).as_deref() == Some("AbortError")
}

pub(crate) fn bytes_to_blob(bytes: &[u8], mime: &str) -> Result<web_sys::Blob, JsValue> {
    let arr = Uint8Array::from(bytes);
    let parts = js_sys::Array::of1(&arr.buffer());
    let opts = web_sys::BlobPropertyBag::new();
    opts.set_type(mime);
    web_sys::Blob::new_with_buffer_source_sequence_and_options(&parts, &opts)
}

/// `await blob.arrayBuffer()` -> bytes.
pub(crate) async fn blob_bytes(blob: &JsValue) -> Result<Vec<u8>, JsValue> {
    let buf = await_promise(call_method(blob, "arrayBuffer", &[])?).await?;
    Ok(Uint8Array::new(&buf).to_vec())
}

pub(crate) fn object() -> Object {
    Object::new()
}

pub(crate) fn mime_for_extension(ext: &str) -> &'static str {
    match ext.trim_start_matches('.').to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "avif" => "image/avif",
        "gif" => "image/gif",
        "tif" | "tiff" => "image/tiff",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "exr" => "image/x-exr",
        "jxl" => "image/jxl",
        "pdf" => "application/pdf",
        "json" => "application/json",
        "txt" => "text/plain",
        _ => "application/octet-stream",
    }
}
