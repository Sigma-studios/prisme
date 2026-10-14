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
| `prisme-gpu` | wgpu engine: incremental tile uploads, CPU mip pyramid, viewport-driven graph evaluation with per-node content-hash caching, WGSL compute kernels, display transform pass, scopes, readbacks. |
| `prisme-io` | Decoders (PNG, JPEG, WebP, GIF, BMP, ICO, TGA, QOI, PNM, HDR, TIFF, JXL, EXR, AVIF, camera RAW, SVG; HEIC behind a feature) and the `.prisme` container (chunked, LZ4, deduplicated tiles, full history). |
| `prisme-export` | PNG export: bit depth/colour mode/palette (quantette), dithering, filters, interlace, oxipng/zopfli, ICC/cICP tagging, explicit resize, target-file-size search with optional resolution reduction. |
| `prisme-brush` | Dab brush engine with stroke buffer (opacity caps the stroke, flow builds up), presets, tablet-ready dynamics. |
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

### Automation
`PRISME_SCRIPT="open:img.jpg;node:levels;param:m.gamma=1.3;export:out.png;quit"` drives the desktop app for
end-to-end checks (see `crates/prisme-ui/src/script.rs`).
