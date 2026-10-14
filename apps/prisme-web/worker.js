// Prisme job worker (module worker). Runs its own instance of the app's wasm module and
// serves heavy CPU jobs (decoding, .prisme encode/decode, PNG export) posted by the page.
// No shared memory: data moves with postMessage + transferable ArrayBuffers, so no
// COOP/COEP headers are needed (works from a cross-origin iframe, e.g. itch.io).
//
// Protocol (see crates/prisme-platform/src/worker.rs):
//   page -> worker  {t:"init", module: WebAssembly.Module}   (once; the compiled module is reused)
//   worker -> page  {t:"ready"} | {t:"init-error", msg}
//   page -> worker  {t:"job", id, header: Uint8Array, bufs: [{k:"u8"|"f32", b: ArrayBuffer}]}
//   worker -> page  {t:"progress", id, f, msg} … then {t:"done", id, ok, err?, header, bufs, mem}
//
// All paths are relative to this file, so the build works from any subpath. The file names
// are stable because Trunk.toml sets `filehash = false`.
import init from "./prisme-web.js";

self.onmessage = async (e) => {
  if (!e.data || e.data.t !== "init") return;
  self.onmessage = null; // the wasm side installs its own handler once started
  try {
    // The wasm start function detects the worker scope and posts {t:"ready"}.
    await init({ module_or_path: e.data.module });
  } catch (err) {
    try {
      // Fall back to fetching the module ourselves (e.g. if Module cloning is unsupported).
      await init({ module_or_path: new URL("./prisme-web_bg.wasm", import.meta.url) });
    } catch (err2) {
      self.postMessage({ t: "init-error", msg: String((err2 && err2.stack) || err2) });
    }
  }
};
