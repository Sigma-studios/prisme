# Vendored crates

Verbatim copies of crates.io sources, wired in through `[patch.crates-io]` in the workspace
`Cargo.toml`. Every change is marked `PRISME PATCH`; `diff -r` against
`~/.cargo/registry/src/*/<crate>-<version>` shows the full diff. Remove a crate here once
upstream offers the same feature.

## egui-wgpu 0.36.2 — HDR surface colour space

egui-wgpu always configures the surface with egui's preferred 8-bit format and
`SurfaceColorSpace::Auto`, and offers no hook to change that.

- `lib.rs`: `WgpuConfiguration::target_format_selector`. This optional callback picks the
  surface format and initial colour space in `RenderState::create`. Results the surface
  doesn't support are ignored.
- `lib.rs`: `RenderState::surface_color: Arc<RwLock<SurfaceColorState>>` is runtime
  colour-space state shared by every clone of the render state. `eframe::Frame`'s
  `set_wgpu_surface_config` writes to a clone the painter never reads, so it can't carry
  this. The app writes `requested`. The painter applies it and reports `configured`, the
  `supported` set, and the display's `wgpu::DisplayHdrInfo`, which is re-queried at most
  every 500 ms while painting.
- `lib.rs`: `sync_surface_color()` is the per-frame helper shared by the native and web
  painters.
- `winit.rs`: `configure_surface` uses `surface_color.configured`. `paint_and_update_textures`
  calls `sync_surface_color` and reconfigures when the colour space changes.
- `capture.rs`: screenshots of `Rgba16Float` surfaces. They use an 8-byte texel padding and
  f16 → u8 conversion, with HDR values clipped. Without this, `ViewportCommand::Screenshot`
  (used by `PRISME_SCRIPT shot:`) fails on HDR surfaces.

egui's own shader is unchanged. For non-`*Srgb` formats it writes sRGB-encoded values, which
is correct for `Srgb` and `ExtendedSrgb` surfaces (Prisme uses only these). It would be
wrong for linear `ExtendedSrgbLinear` (scRGB).

## eframe 0.36.2 — web painter

- `web/web_painter_wgpu.rs`: the canvas is configured with `surface_color.configured`.
  Each paint calls `egui_wgpu::sync_surface_color` and reconfigures on change. On WebGPU,
  `ExtendedSrgb` becomes canvas `toneMapping: { mode: "extended" }`.

## eframe 0.36.2 — clipboard shortcut keys (native)

egui-winit turns Cmd/Ctrl+C/X/V into `Event::Copy/Cut/Paste` and drops the key event, and
sends nothing at all for Cmd/Ctrl+V when the clipboard holds no text (an image). Apps can't bind
those shortcuts (Paste image, Copy Merged = Cmd+Shift+C, Paste as New Image = Cmd+Alt+V).

- `native/epi_integration.rs`: `EpiIntegration::on_window_event` also forwards those key
  presses as `Event::Key` (tracking modifiers from `ModifiersChanged`); `clipboard_key` picks
  the key like egui-winit (logical, physical fallback for non-Latin layouts). Text fields keep
  using the clipboard events. The web runner already forwards the key.

## eframe 0.36.2 — pasted files (web)

The web runner reads only text from the browser's `paste` event, and the async clipboard API
can't see files copied in a file manager (Finder, Files…).

- `web/events.rs`: a `paste` event carrying files (copied files, or image data, which browsers
  expose as a file) pushes them to `raw.dropped_files`, like a drop. Prisme places them as
  layers (Cmd+Alt+V or Shift: separately) and leaves Cmd+V's clipboard-API Paste to the menu.
