//! Persistent key/value app storage (settings, autosave, crash recovery).
//!
//! Keys are short strings; `/` may be used to group keys (`"autosave/doc1"`). Any character
//! outside `[A-Za-z0-9._-]` (per segment) is replaced by `_`, and `..`/empty segments are
//! dropped, so keys can never escape the storage root.

use crate::pending::Pending;

fn sanitize_segments(key: &str) -> Vec<String> {
    key.split(['/', '\\'])
        .filter(|s| !s.is_empty() && *s != "." && *s != "..")
        .map(|s| {
            s.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                        c
                    } else {
                        '_'
                    }
                })
                .collect()
        })
        .collect()
}

/// Flattened, sanitized key (segments joined with `__`); used on the web.
#[allow(dead_code)]
fn flat_key(key: &str) -> String {
    let s = sanitize_segments(key).join("__");
    if s.is_empty() { "_".into() } else { s }
}

// ---------------------------------------------------------------------------------------------
// Native: files under dirs::data_dir()/prisme/

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::sanitize_segments;
    use crate::pending::{Pending, spawn};
    use std::path::PathBuf;

    pub fn root() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("PRISME_DATA_DIR") {
            return Some(PathBuf::from(p));
        }
        dirs::data_dir().map(|d| d.join("prisme"))
    }

    pub fn path_for(key: &str) -> Result<PathBuf, String> {
        let segs = sanitize_segments(key);
        if segs.is_empty() {
            return Err(format!("invalid storage key {key:?}"));
        }
        let mut p = root().ok_or("no data directory on this system")?;
        for s in segs {
            p.push(s);
        }
        Ok(p)
    }

    pub fn write(key: &str, bytes: &[u8]) -> Result<(), String> {
        crate::files::write_file_path(&path_for(key)?, bytes)
    }

    pub fn read(key: &str) -> Option<Vec<u8>> {
        std::fs::read(path_for(key).ok()?).ok()
    }

    pub fn write_large(key: &str, bytes: Vec<u8>) -> Pending<Result<(), String>> {
        let key = key.to_string();
        spawn(move || write(&key, &bytes))
    }

    pub fn read_large(key: &str) -> Pending<Option<Vec<u8>>> {
        let key = key.to_string();
        spawn(move || read(&key))
    }

    pub fn delete(key: &str) {
        if let Ok(p) = path_for(key) {
            let _ = std::fs::remove_file(p);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Web: localStorage (base64) for small values, OPFS for large blobs

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::flat_key;
    use crate::pending::{Pending, pending_pair};
    use crate::web::*;
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD as B64;
    use wasm_bindgen::{JsCast, JsValue};

    const PREFIX: &str = "prisme:";

    fn local_storage() -> Option<web_sys::Storage> {
        window()?.local_storage().ok().flatten()
    }

    pub fn write(key: &str, bytes: &[u8]) -> Result<(), String> {
        let ls = local_storage().ok_or("localStorage unavailable")?;
        ls.set_item(&format!("{PREFIX}{}", flat_key(key)), &B64.encode(bytes))
            .map_err(|e| err_string(&e))
    }

    pub fn read(key: &str) -> Option<Vec<u8>> {
        let s = local_storage()?
            .get_item(&format!("{PREFIX}{}", flat_key(key)))
            .ok()??;
        B64.decode(s).ok()
    }

    async fn opfs_root() -> Result<web_sys::FileSystemDirectoryHandle, JsValue> {
        let nav = window().ok_or("no window")?.navigator();
        let storage = get(&nav, "storage").ok_or("navigator.storage unavailable")?;
        let dir = await_promise(call_method(&storage, "getDirectory", &[])?).await?;
        let root: web_sys::FileSystemDirectoryHandle = dir.dyn_into()?;
        // Keep our files in a subdirectory.
        let opts = web_sys::FileSystemGetDirectoryOptions::new();
        opts.set_create(true);
        let sub = wasm_bindgen_futures::JsFuture::from(
            root.get_directory_handle_with_options("prisme", &opts),
        )
        .await?;
        sub.dyn_into()
    }

    async fn opfs_write(name: String, bytes: Vec<u8>) -> Result<(), JsValue> {
        let dir = opfs_root().await?;
        let opts = web_sys::FileSystemGetFileOptions::new();
        opts.set_create(true);
        let fh =
            wasm_bindgen_futures::JsFuture::from(dir.get_file_handle_with_options(&name, &opts))
                .await?;
        let writable = await_promise(call_method(&fh, "createWritable", &[])?).await?;
        let arr = js_sys::Uint8Array::from(bytes.as_slice());
        await_promise(call_method(&writable, "write", &[arr.into()])?).await?;
        await_promise(call_method(&writable, "close", &[])?).await?;
        Ok(())
    }

    async fn opfs_read(name: String) -> Result<Vec<u8>, JsValue> {
        let dir = opfs_root().await?;
        let fh = wasm_bindgen_futures::JsFuture::from(dir.get_file_handle(&name)).await?;
        let file = await_promise(call_method(&fh, "getFile", &[])?).await?;
        blob_bytes(&file).await
    }

    pub fn write_large(key: &str, bytes: Vec<u8>) -> Pending<Result<(), String>> {
        let name = flat_key(key);
        let key = key.to_string();
        let (c, p) = pending_pair();
        wasm_bindgen_futures::spawn_local(async move {
            let r = match opfs_write(name, bytes.clone()).await {
                Ok(()) => {
                    // Remove any stale small copy so reads are unambiguous.
                    if let Some(ls) = local_storage() {
                        let _ = ls.remove_item(&format!("{PREFIX}{}", flat_key(&key)));
                    }
                    Ok(())
                }
                Err(e) => {
                    // No OPFS / no createWritable (older Safari): fall back to localStorage.
                    log::warn!(
                        "OPFS write failed ({}), falling back to localStorage",
                        err_string(&e)
                    );
                    write(&key, &bytes)
                }
            };
            c.complete(r);
        });
        p
    }

    pub fn read_large(key: &str) -> Pending<Option<Vec<u8>>> {
        let name = flat_key(key);
        let key = key.to_string();
        let (c, p) = pending_pair();
        wasm_bindgen_futures::spawn_local(async move {
            let r = match opfs_read(name).await {
                Ok(b) => Some(b),
                Err(_) => read(&key),
            };
            c.complete(r);
        });
        p
    }

    pub fn delete(key: &str) {
        let name = flat_key(key);
        if let Some(ls) = local_storage() {
            let _ = ls.remove_item(&format!("{PREFIX}{name}"));
        }
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(dir) = opfs_root().await {
                let _ = wasm_bindgen_futures::JsFuture::from(dir.remove_entry(&name)).await;
            }
        });
    }
}

/// Store a small value synchronously.
/// Native: file `dirs::data_dir()/prisme/<key>` (override the root with `PRISME_DATA_DIR`),
/// written atomically. Web: `localStorage` (base64; keep values well below ~1 MB).
pub fn storage_write(key: &str, bytes: &[u8]) -> Result<(), String> {
    imp::write(key, bytes)
}

/// Read a value written with [`storage_write`] (or, on native, [`storage_write_large`]).
pub fn storage_read(key: &str) -> Option<Vec<u8>> {
    imp::read(key)
}

/// Store a large blob asynchronously. Native: file write on a background thread.
/// Web: Origin Private File System (falls back to `localStorage` if OPFS is unavailable).
pub fn storage_write_large(key: &str, bytes: Vec<u8>) -> Pending<Result<(), String>> {
    imp::write_large(key, bytes)
}

/// Read a blob written with [`storage_write_large`].
pub fn storage_read_large(key: &str) -> Pending<Option<Vec<u8>>> {
    imp::read_large(key)
}

/// Delete a key from both the small and the large store (best effort, never fails).
pub fn storage_delete(key: &str) {
    imp::delete(key)
}

/// Native only: the directory backing the storage (`<data dir>/prisme`).
#[cfg(not(target_arch = "wasm32"))]
pub fn storage_dir() -> Option<std::path::PathBuf> {
    imp::root()
}

#[cfg(test)]
pub(crate) fn sanitize_for_test(key: &str) -> Vec<String> {
    sanitize_segments(key)
}
