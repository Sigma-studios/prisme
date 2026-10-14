//! Heavy CPU jobs run through [`prisme_platform::spawn_worker`]: decoding images and
//! projects, `.prisme` encoding and PNG export encoding.
//!
//! Native: each job's `run` executes on an OS thread (Rayon inside, as before). Web: the job is
//! turned into a [`WorkerMsg`] (postcard header + raw transferable buffers), executed by
//! [`dispatch`] inside a Web Worker running this same wasm, and the reply is converted back
//! on the UI thread. Documents cross the boundary as a [`prisme_io::container::Transport`]
//! (uncompressed tiles) which is cheap on both sides.

use prisme_core::{Document, History};
use prisme_export::{ExportResult, PngOptions};
use prisme_io::container::{self, Transport};
use prisme_io::PreparedLayers;
use prisme_platform::{Buffer, JobCtx, WorkerHandler, WorkerJob, WorkerMsg};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Result of opening a file.
pub enum Loaded {
    Doc(Document),
    History(History),
    /// Layers to add to an open document ("Place").
    Place(PreparedLayers),
}

#[derive(Serialize, Deserialize)]
enum Request {
    /// buffers: [file bytes]
    Decode { name: String, place_into: Option<(u32, u32)> },
    /// buffers: [transport doc, transport tiles] of the history to save
    Save,
    /// buffers: [straight-alpha f32 RGBA pixels]
    EncodePng { width: u32, height: u32, opts: PngOptions },
}

#[derive(Serialize, Deserialize)]
enum Reply {
    /// buffers: [transport doc, transport tiles]
    Doc,
    History,
    Place,
    /// buffers: [bytes]
    Bytes,
    /// `result.bytes` is moved to buffers[0]
    Png(ExportResult),
}

fn header<T: Serialize>(v: &T) -> Result<Vec<u8>, String> {
    postcard::to_allocvec(v).map_err(|e| format!("message encoding: {e}"))
}

fn parse<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, String> {
    postcard::from_bytes(bytes).map_err(|e| format!("message decoding: {e}"))
}

fn transport_msg<T: Serialize>(reply: &Reply, value: &T) -> Result<WorkerMsg, String> {
    let Transport { doc, tiles } = container::to_transport(value).map_err(|e| e.to_string())?;
    Ok(WorkerMsg::new(header(reply)?, vec![Buffer::Bytes(doc), Buffer::Bytes(tiles)]))
}

fn from_transport<T: serde::de::DeserializeOwned>(buffers: Vec<Buffer>) -> Result<T, String> {
    let mut it = buffers.into_iter();
    let doc = it.next().ok_or("missing document buffer")?.into_bytes();
    let tiles = it.next().ok_or("missing tile buffer")?.into_bytes();
    container::from_transport(&doc, &tiles).map_err(|e| e.to_string())
}

fn first_buffer(buffers: Vec<Buffer>) -> Result<Buffer, String> {
    buffers.into_iter().next().ok_or_else(|| "missing buffer".to_string())
}

/// The worker-side handler for every job kind (also the on-thread fallback on the web).
pub fn dispatch(msg: WorkerMsg, ctx: &JobCtx) -> Result<WorkerMsg, String> {
    match parse::<Request>(&msg.header)? {
        Request::Decode { name, place_into } => {
            let bytes = first_buffer(msg.buffers)?.into_bytes();
            match decode(&name, bytes, place_into)? {
                Loaded::Doc(d) => transport_msg(&Reply::Doc, &d),
                Loaded::History(h) => transport_msg(&Reply::History, &h),
                Loaded::Place(p) => transport_msg(&Reply::Place, &p),
            }
        }
        Request::Save => {
            let mut it = msg.buffers.into_iter();
            let doc = it.next().ok_or("missing document buffer")?.into_bytes();
            let tiles = it.next().ok_or("missing tile buffer")?;
            let bytes = container::save_transport(doc, tiles.as_bytes(), None).map_err(|e| e.to_string())?;
            Ok(WorkerMsg::new(header(&Reply::Bytes)?, vec![Buffer::Bytes(bytes)]))
        }
        Request::EncodePng { width, height, opts } => {
            let pixels = first_buffer(msg.buffers)?.into_f32();
            let mut res = encode_png(&pixels, width, height, &opts, ctx)?;
            drop(pixels);
            let bytes = std::mem::take(&mut res.bytes);
            Ok(WorkerMsg::new(header(&Reply::Png(res))?, vec![Buffer::Bytes(bytes)]))
        }
    }
}

/// Start `prisme_platform`'s worker loop if this wasm instance runs inside a Web Worker.
/// Returns `true` in that case (the caller must not start the UI).
pub fn worker_entry() -> bool {
    prisme_platform::worker_entry(dispatch)
}

// ------------------------------------------------------------------ job bodies

fn decode(name: &str, bytes: Vec<u8>, place_into: Option<(u32, u32)>) -> Result<Loaded, String> {
    if prisme_io::is_project_file(name) {
        return container::load(&bytes).map(Loaded::History).map_err(|e| e.to_string());
    }
    let decoded = prisme_io::decode(&bytes, name).map_err(|e| e.to_string())?;
    drop(bytes);
    Ok(match place_into {
        Some(size) => Loaded::Place(prisme_io::prepare_layers(name, decoded, size)),
        None => Loaded::Doc(prisme_io::document_from_decoded(name, decoded)),
    })
}

fn encode_png(pixels: &[f32], w: u32, h: u32, opts: &PngOptions, ctx: &JobCtx) -> Result<ExportResult, String> {
    prisme_export::encode_png(pixels, w, h, opts, &mut |f, stage| ctx.progress(f, stage), &|| ctx.cancelled())
        .map_err(|e| e.to_string())
}

// ------------------------------------------------------------------ jobs

/// Decode an image or `.prisme` project. `place_into` = size of the document the image is
/// placed into (layers are built centred for it).
pub struct DecodeJob {
    pub name: String,
    pub bytes: Vec<u8>,
    pub place_into: Option<(u32, u32)>,
}

impl WorkerJob for DecodeJob {
    type Output = Result<Loaded, String>;

    fn run(self, _ctx: &JobCtx) -> Self::Output {
        decode(&self.name, self.bytes, self.place_into)
    }

    fn into_msg(self) -> Result<WorkerMsg, String> {
        let h = header(&Request::Decode { name: self.name, place_into: self.place_into })?;
        Ok(WorkerMsg::new(h, vec![Buffer::Bytes(self.bytes)]))
    }

    fn handler() -> WorkerHandler {
        dispatch
    }

    fn output_from_msg(reply: Result<WorkerMsg, String>) -> Self::Output {
        let msg = reply?;
        match parse::<Reply>(&msg.header)? {
            Reply::Doc => from_transport(msg.buffers).map(Loaded::Doc),
            Reply::History => from_transport(msg.buffers).map(Loaded::History),
            Reply::Place => from_transport(msg.buffers).map(Loaded::Place),
            _ => Err("unexpected worker reply".into()),
        }
    }
}

/// Encode a `.prisme` project (whole history, or only the current state).
pub struct SaveJob {
    pub history: History,
    pub include_history: bool,
}

impl WorkerJob for SaveJob {
    type Output = Result<Vec<u8>, String>;

    fn run(self, _ctx: &JobCtx) -> Self::Output {
        container::save(&self.history, self.include_history).map_err(|e| e.to_string())
    }

    fn into_msg(self) -> Result<WorkerMsg, String> {
        // Serializing the structure and copying raw tiles is cheap; the LZ4 work happens in
        // the worker.
        let h = container::history_for_save(&self.history, self.include_history);
        let Transport { doc, tiles } = container::to_transport(&h).map_err(|e| e.to_string())?;
        Ok(WorkerMsg::new(header(&Request::Save)?, vec![Buffer::Bytes(doc), Buffer::Bytes(tiles)]))
    }

    fn handler() -> WorkerHandler {
        dispatch
    }

    fn output_from_msg(reply: Result<WorkerMsg, String>) -> Self::Output {
        let msg = reply?;
        match parse::<Reply>(&msg.header)? {
            Reply::Bytes => Ok(first_buffer(msg.buffers)?.into_bytes()),
            _ => Err("unexpected worker reply".into()),
        }
    }
}

/// PNG export encoding (quantization, filters, oxipng/zopfli, target-size search), with
/// progress and cancellation.
pub struct EncodePngJob {
    pub pixels: Arc<Vec<f32>>,
    pub width: u32,
    pub height: u32,
    pub opts: PngOptions,
}

impl WorkerJob for EncodePngJob {
    type Output = Result<ExportResult, String>;

    fn run(self, ctx: &JobCtx) -> Self::Output {
        encode_png(&self.pixels, self.width, self.height, &self.opts, ctx)
    }

    fn into_msg(self) -> Result<WorkerMsg, String> {
        let h = header(&Request::EncodePng { width: self.width, height: self.height, opts: self.opts })?;
        Ok(WorkerMsg::new(h, vec![Buffer::SharedF32(self.pixels)]))
    }

    fn handler() -> WorkerHandler {
        dispatch
    }

    fn output_from_msg(reply: Result<WorkerMsg, String>) -> Self::Output {
        let msg = reply?;
        match parse::<Reply>(&msg.header)? {
            Reply::Png(mut res) => {
                res.bytes = first_buffer(msg.buffers)?.into_bytes();
                Ok(res)
            }
            _ => Err("unexpected worker reply".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prisme_platform::run_via_messages;

    fn sample_png() -> Vec<u8> {
        let (w, h) = (300u32, 200u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for (i, p) in rgba.chunks_mut(4).enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            p.copy_from_slice(&[(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8, 255]);
        }
        prisme_platform::encode_png(&prisme_platform::ClipboardImage { width: w, height: h, rgba }).unwrap()
    }

    fn pixels(doc: &Document) -> Vec<Vec<f32>> {
        let b = doc.bounds();
        doc.all_layers().iter().filter_map(|l| l.raster().map(|g| g.read_rect(&b))).collect()
    }

    #[test]
    fn decode_via_messages_matches_direct() {
        let ctx = JobCtx::detached();
        let png = sample_png();
        let job = || DecodeJob { name: "a.png".into(), bytes: png.clone(), place_into: None };
        let (Ok(Loaded::Doc(a)), Ok(Loaded::Doc(b))) = (job().run(&ctx), run_via_messages(job(), &ctx)) else {
            panic!("decode failed")
        };
        assert_eq!((a.info.width, a.info.height), (b.info.width, b.info.height));
        assert_eq!(pixels(&a), pixels(&b));

        let place = DecodeJob { name: "a.png".into(), bytes: png, place_into: Some((500, 500)) };
        let Ok(Loaded::Place(p)) = run_via_messages(place, &ctx) else { panic!("place failed") };
        assert_eq!(p.layers.len(), 1);
        let mut doc = Document::new("d", 500, 500, prisme_core::PixelFormat::F16);
        assert_eq!(prisme_io::add_prepared_layers(&mut doc, p).len(), 1);
    }

    #[test]
    fn save_and_load_project_via_messages() {
        let ctx = JobCtx::detached();
        let Ok(Loaded::Doc(doc)) = DecodeJob { name: "a.png".into(), bytes: sample_png(), place_into: None }.run(&ctx) else {
            panic!()
        };
        let mut history = History::new(doc.clone());
        let mut d2 = doc.clone();
        d2.info.name = "renamed".into();
        history.push("Rename", d2);
        let direct = SaveJob { history: history.clone(), include_history: true }.run(&ctx).unwrap();
        let via = run_via_messages(SaveJob { history: history.clone(), include_history: true }, &ctx).unwrap();
        assert_eq!(direct, via, "worker save must produce the same file");

        let Ok(Loaded::History(h)) = run_via_messages(DecodeJob { name: "p.prisme".into(), bytes: via, place_into: None }, &ctx) else {
            panic!("load failed")
        };
        assert_eq!(h.entries().len(), 2);
        assert_eq!(h.current().info.name, "renamed");
        assert_eq!(pixels(h.current()), pixels(&doc));

        let current = run_via_messages(SaveJob { history, include_history: false }, &ctx).unwrap();
        assert_eq!(container::load(&current).unwrap().entries().len(), 1);
    }

    #[test]
    fn encode_png_via_messages() {
        let ctx = JobCtx::detached();
        let (w, h) = (64u32, 48u32);
        let px: Vec<f32> = (0..w * h).flat_map(|i| [(i % w) as f32 / w as f32, 0.5, (i / w) as f32 / h as f32, 1.0]).collect();
        let job = || EncodePngJob { pixels: Arc::new(px.clone()), width: w, height: h, opts: PngOptions::default() };
        let a = job().run(&ctx).unwrap();
        let b = run_via_messages(job(), &ctx).unwrap();
        assert_eq!(a.bytes, b.bytes);
        assert_eq!((b.width, b.height), (w, h));
        assert!(ctx_progress_reported());
    }

    fn ctx_progress_reported() -> bool {
        let task = prisme_platform::spawn_job(|ctx| {
            ctx.progress(0.5, "half");
        });
        task.pending.block();
        task.control.progress().is_some_and(|(f, s)| f == 0.5 && s == "half")
    }

    #[test]
    fn errors_cross_the_boundary() {
        let ctx = JobCtx::detached();
        let r = run_via_messages(DecodeJob { name: "x.png".into(), bytes: vec![1, 2, 3], place_into: None }, &ctx);
        assert!(r.is_err());
    }
}
