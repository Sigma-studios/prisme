# Prisme

An all-Rust, GPU-accelerated, colour-accurate image editor with non-destructive node-based editing.
Runs on Windows, macOS, Linux and the web (WebGPU).

## Run

```sh
cargo run --release -p prisme-desktop -- [image files…]   # desktop
cd apps/prisme-web && trunk serve --release               # web (needs a WebGPU browser)
cargo test --workspace --release                          # tests (GPU tests use any available adapter)
```

## Architecture

| Crate | Role |
|---|---|
| `prisme-core` | Document model: sparse copy-on-write tiles (f16/f32), layers (raster, vector, text, fill, group), masks & selections, node-graph data, structurally shared undo history. No GPU/UI/IO. |
| `prisme-color` | Colour science: working space = scene-linear Rec.2020 (f32), analytic spaces, OKLab/OKLCh, ICC via `moxcms`, display LUT baking, ΔE2000. |
| `prisme-graph` | Node kind registry: sockets, parameter schemas (drive UI + defaults), ROI padding. |
| `prisme-gpu` | wgpu engine: incremental tile uploads, CPU mip pyramid, viewport-driven graph evaluation with per-node content-hash caching, WGSL compute kernels, GPU brush strokes, display transform pass, scopes, readbacks. |
| `prisme-io` | Decoders (PNG, JPEG, WebP, GIF, BMP, ICO, TGA, QOI, PNM, HDR, TIFF, JXL, EXR, AVIF, camera RAW, SVG; HEIC behind a feature) and the `.prisme` container (chunked, LZ4, deduplicated tiles, full history). |
| `prisme-export` | PNG export: bit depth/colour mode/palette (quantette), dithering, filters, interlace, oxipng/zopfli, ICC/cICP tagging, explicit resize, target-file-size search with optional resolution reduction. |
| `prisme-brush` | Dab brush engine: tablet-ready dynamics and dab placement, presets, stroke-buffer semantics (opacity caps the stroke, flow builds up). Dabs are rasterized/composited on the GPU (`prisme-gpu`), read back and landed in the tiles as one history edit when the stroke ends; the CPU compositor is the reference/fallback. |
| `prisme-tools` | Tool state machines (selection, paint, fill, gradient, transform, crop, text, picker, clone/heal, smudge/blur, ruler…) and their pixel algorithms. |
| `prisme-text` | Fonts (system + bundled + user), shaping, 2D effects, warps, 3D extrusion on a sphere. |
| `prisme-vector` | SVG layers rendered from source at any zoom. |
| `prisme-platform` | Platform services: async file dialogs, clipboard, screen colour picker, display ICC, storage, task spawning. |
| `prisme-ui` | egui application: docking panels, canvas, node editor, colour wheels, curves/levels, dialogs, keymap. |

Extension points: add a node = `NodeKindDesc` in `prisme-graph` + a kernel in `prisme-gpu`; add a tool = a `Tool`
impl; add a format = an `ImageDecoder` impl; add a shortcut = an `Action`.

### Node model
The document graph ends in a single **Output** node; by default `Layer Stack → Output`. Use **Layer** nodes to pull
individual layers into processing chains and **Blend** nodes to recombine (blending is explicit). Destructive edits
(painting, fill…) modify layer pixels directly. `\` toggles the view between node output and the raw layer stack.

### Background jobs and the web build
Heavy CPU work goes through `prisme_platform::spawn_worker(job)` (jobs live in `crates/prisme-ui/src/offload.rs`):
decoding images on open/place (colour conversion and tile building included), `.prisme` load/save and autosave
encoding, and PNG export encoding (quantization, oxipng/zopfli, target-size search, with progress and Cancel).
It returns a `Task` (a `Pending` result plus a `JobControl` for progress/cancellation) on every target:

* **Native**: the job runs on an OS thread (Rayon inside, as before).
* **Web**: the job runs in a **Web Worker**, without shared memory (no `SharedArrayBuffer`, no COOP/COEP headers,
  stable toolchain). Each worker runs its own instance of the *same* `prisme-web` wasm: `apps/prisme-web/worker.js`
  imports the wasm-bindgen glue and the page posts it the already compiled `WebAssembly.Module`, so nothing is
  downloaded or compiled twice; the wasm `start` function notices it runs in a worker and serves jobs instead of
  starting the UI. Requests and replies are a small postcard header plus raw `ArrayBuffer`s moved with
  `postMessage` transfer lists (pixels as raw `f32`, documents as uncompressed tiles via
  `prisme_io::container::to_transport`). Workers form a lazily grown pool of `min(hardwareConcurrency − 1, 4)`
  (several files decode in parallel, an export can encode while a project saves); a worker whose wasm memory grew past
  256 MB is retired after its job since wasm memory never shrinks. Cancelling an export terminates its worker. If
  workers can't be created (CSP, missing `worker.js`…), jobs fall back to running on the UI thread through the same
  message path. The export size *estimate* still runs on the UI thread on the web (it only encodes a sampled mosaic).

`trunk build --release` in `apps/prisme-web` produces a self-contained `dist/` with relative URLs only
(`public_url = "./"`, `filehash = false` so `worker.js` can import `./prisme-web.js` / `./prisme-web_bg.wasm`):
`index.html`, `prisme-web.js`, `prisme-web_bg.wasm`, `worker.js` (plus `snippets/`). Zip the contents of `dist/` and
upload it as an HTML game on itch.io (or any static host, any subpath); no special headers are needed. It needs a
WebGPU browser with module-worker support (all current WebGPU browsers have it).

### Automation
`PRISME_SCRIPT="open:img.jpg;node:levels;param:m.gamma=1.3;export:out.png;quit"` drives the desktop app for
end-to-end checks; `tool:Brush;stroke:x0,y0,x1,y1,…` paints (see `crates/prisme-ui/src/script.rs`).
