//! File open / save dialogs.

use crate::pending::{Pending, spawn_async};

/// A file picked by the user, already read into memory.
#[derive(Clone, Debug)]
pub struct PickedFile {
    /// File name (without directories).
    pub name: String,
    pub bytes: Vec<u8>,
    /// Full path on native platforms; always `None` on the web.
    pub path: Option<std::path::PathBuf>,
}

/// A named group of extensions for open/save dialogs, e.g. `("Images", ["png", "jpg"])`.
/// Extensions are given without the leading dot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileFilter {
    pub name: String,
    pub extensions: Vec<String>,
}

impl FileFilter {
    pub fn new(name: impl Into<String>, extensions: &[&str]) -> Self {
        Self {
            name: name.into(),
            extensions: extensions
                .iter()
                .map(|e| e.trim_start_matches('.').to_string())
                .collect(),
        }
    }
}

fn dialog(title: &str, filters: &[FileFilter]) -> rfd::AsyncFileDialog {
    let mut d = rfd::AsyncFileDialog::new().set_title(title);
    for f in filters {
        let exts: Vec<&str> = f.extensions.iter().map(|s| s.as_str()).collect();
        d = d.add_filter(&f.name, &exts);
    }
    d
}

/// Show an open-file dialog and read the selected file(s).
/// Resolves to an empty `Vec` if the user cancelled.
///
/// Must be called from the UI thread (on macOS the panel is attached as a sheet to the main
/// window; on the web the call must happen inside a user gesture, e.g. a button click).
pub fn open_files(title: &str, filters: &[FileFilter], multiple: bool) -> Pending<Vec<PickedFile>> {
    let d = dialog(title, filters);
    // Create the dialog futures now, on the calling (UI) thread (rfd starts the dialog when
    // the future is constructed); only the awaiting happens in the background.
    let (many, one) = if multiple {
        (Some(d.pick_files()), None)
    } else {
        (None, Some(d.pick_file()))
    };
    let handles = async move {
        match (many, one) {
            (Some(f), _) => f.await.unwrap_or_default(),
            (_, Some(f)) => f.await.into_iter().collect(),
            _ => Vec::new(),
        }
    };
    spawn_async(async move {
        let mut out = Vec::new();
        for h in handles.await {
            let bytes = h.read().await;
            out.push(PickedFile {
                name: h.file_name(),
                bytes,
                #[cfg(not(target_arch = "wasm32"))]
                path: Some(h.path().to_path_buf()),
                #[cfg(target_arch = "wasm32")]
                path: None,
            });
        }
        out
    })
}

/// Save bytes to a user-chosen location.
///
/// * Native: shows a save dialog, then writes the file. Resolves to `Ok(Some(full path))`.
/// * Web: uses the File System Access API (`showSaveFilePicker`, Chromium) when available,
///   otherwise triggers a browser download through an `<a download>` blob URL (in that case
///   it resolves to `Ok(Some(suggested_name))` immediately, the browser handles the rest).
///
/// `Ok(None)` means the user cancelled. Call from the UI thread (inside a user gesture on web).
pub fn save_file(
    title: &str,
    suggested_name: &str,
    filters: &[FileFilter],
    bytes: Vec<u8>,
) -> Pending<Result<Option<String>, String>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let fut = dialog(title, filters)
            .set_file_name(suggested_name)
            .save_file();
        spawn_async(async move {
            match fut.await {
                None => Ok(None),
                Some(h) => {
                    let path = h.path().to_path_buf();
                    write_file_path(&path, &bytes)?;
                    Ok(Some(path.display().to_string()))
                }
            }
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = title;
        web_save::save(suggested_name, filters, bytes)
    }
}

/// Write bytes to a path directly (native only). Writes to a temporary sibling file first
/// and renames it into place, so a crash never leaves a truncated file.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_file_path(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".prisme-tmp");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

/// Not available on the web: always returns an error.
#[cfg(target_arch = "wasm32")]
pub fn write_file_path(path: &std::path::Path, _bytes: &[u8]) -> Result<(), String> {
    Err(format!(
        "cannot write {} : direct file paths are not available on the web",
        path.display()
    ))
}

#[cfg(target_arch = "wasm32")]
mod web_save {
    use super::FileFilter;
    use crate::pending::{Pending, pending_pair};
    use crate::web::*;
    use wasm_bindgen::{JsCast, JsValue};

    pub(super) fn save(
        suggested_name: &str,
        filters: &[FileFilter],
        bytes: Vec<u8>,
    ) -> Pending<Result<Option<String>, String>> {
        let Some(win) = window() else {
            return Pending::ready(Err("no window".into()));
        };
        if get(&win, "showSaveFilePicker").is_some() {
            // Build options and call synchronously so the user activation is still valid.
            let opts = object();
            set(&opts, "suggestedName", &JsValue::from_str(suggested_name));
            let types = js_sys::Array::new();
            for f in filters {
                let accept = object();
                // Group extensions per MIME type.
                let mut by_mime: Vec<(&str, js_sys::Array)> = Vec::new();
                for e in &f.extensions {
                    let m = mime_for_extension(e);
                    let ext = JsValue::from_str(&format!(".{}", e.trim_start_matches('.')));
                    if let Some((_, a)) = by_mime.iter().find(|(mm, _)| *mm == m) {
                        a.push(&ext);
                    } else {
                        let a = js_sys::Array::new();
                        a.push(&ext);
                        by_mime.push((m, a));
                    }
                }
                if by_mime.is_empty() {
                    continue;
                }
                for (m, a) in by_mime {
                    set(&accept, m, &a);
                }
                let t = object();
                set(&t, "description", &JsValue::from_str(&f.name));
                set(&t, "accept", &accept);
                types.push(&t);
            }
            if types.length() > 0 {
                set(&opts, "types", &types);
            }
            match call_method(&win, "showSaveFilePicker", &[opts.into()]) {
                Ok(promise) => {
                    let (c, p) = pending_pair();
                    let name = suggested_name.to_string();
                    wasm_bindgen_futures::spawn_local(async move {
                        c.complete(write_via_handle(promise, bytes, name).await);
                    });
                    return p;
                }
                Err(e) => {
                    log::warn!(
                        "showSaveFilePicker failed ({}), falling back to download",
                        err_string(&e)
                    );
                }
            }
        }
        Pending::ready(download(suggested_name, &bytes).map(Some))
    }

    async fn write_via_handle(
        promise: JsValue,
        bytes: Vec<u8>,
        fallback_name: String,
    ) -> Result<Option<String>, String> {
        let handle = match await_promise(promise).await {
            Ok(h) => h,
            Err(e) if is_abort(&e) => return Ok(None),
            Err(e) => return Err(err_string(&e)),
        };
        let name = get(&handle, "name")
            .and_then(|n| n.as_string())
            .unwrap_or(fallback_name);
        let inner = async {
            let writable = await_promise(call_method(&handle, "createWritable", &[])?).await?;
            let arr = js_sys::Uint8Array::from(bytes.as_slice());
            await_promise(call_method(&writable, "write", &[arr.into()])?).await?;
            await_promise(call_method(&writable, "close", &[])?).await?;
            Ok::<_, JsValue>(())
        };
        inner.await.map_err(|e| err_string(&e))?;
        Ok(Some(name))
    }

    /// Trigger a download through a temporary `<a download>` element.
    pub(crate) fn download(name: &str, bytes: &[u8]) -> Result<String, String> {
        let win = window().ok_or("no window")?;
        let doc = win.document().ok_or("no document")?;
        let ext = name.rsplit('.').next().unwrap_or("");
        let blob = bytes_to_blob(bytes, mime_for_extension(ext)).map_err(|e| err_string(&e))?;
        let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(|e| err_string(&e))?;
        let a: web_sys::HtmlAnchorElement = doc
            .create_element("a")
            .map_err(|e| err_string(&e))?
            .dyn_into()
            .map_err(|_| "not an anchor".to_string())?;
        a.set_href(&url);
        a.set_download(name);
        let _ = a.style().set_property("display", "none");
        if let Some(body) = doc.body() {
            let _ = body.append_child(&a);
            a.click();
            let _ = body.remove_child(&a);
        } else {
            a.click();
        }
        // Revoke later: revoking synchronously can cancel the download in some browsers.
        let cb = wasm_bindgen::closure::Closure::once_into_js(move || {
            let _ = web_sys::Url::revoke_object_url(&url);
        });
        let _ =
            win.set_timeout_with_callback_and_timeout_and_arguments_0(cb.unchecked_ref(), 60_000);
        Ok(name.to_string())
    }
}
