//! Heavy CPU jobs off the UI thread: OS threads on native, Web Workers on the web.
//!
//! A job is a [`WorkerJob`]: it knows how to run directly ([`WorkerJob::run`], used on native
//! threads) and how to cross a Web Worker boundary as a [`WorkerMsg`] (a small serialized
//! header plus raw byte buffers that are *transferred*, not structured-cloned). [`spawn_worker`]
//! returns a [`Task`]: the usual [`Pending`] result plus a [`JobControl`] for progress and
//! cancellation.
//!
//! ## Web
//!
//! No shared memory is used (no `SharedArrayBuffer`, so no COOP/COEP headers are needed and
//! the app runs from a cross-origin iframe such as itch.io). Every worker runs its own
//! instance of the *same* wasm module as the page: `worker.js` (copied next to `index.html`)
//! imports the wasm-bindgen glue, and the page posts it the already compiled
//! `WebAssembly.Module`, so nothing is downloaded or compiled twice. The app's
//! `#[wasm_bindgen(start)]` calls [`worker_entry`], which detects the worker global scope and
//! starts serving requests instead of starting the UI.
//!
//! Workers form a small lazily grown pool (`min(hardwareConcurrency - 1, 4)`, at least 1).
//! Idle workers are reused; a worker whose wasm memory grew beyond [`RETIRE_BYTES`] (wasm
//! memory never shrinks) is terminated after its job. Cancelling a running job terminates its
//! worker (a busy worker cannot receive messages). A worker that panics fails only its job
//! (the [`Pending`] is abandoned). If workers cannot be created at all (CSP, missing
//! `worker.js`, no module-worker support), jobs fall back to running on the UI thread through
//! the very same message path.

use crate::pending::Pending;
#[cfg(target_arch = "wasm32")]
use crate::pending::pending_pair;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Wasm memory size above which a worker is retired after finishing its job.
pub const RETIRE_BYTES: f64 = 256.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------------------------
// Messages

/// A raw buffer crossing the worker boundary (moved, not serialized).
#[derive(Debug, Clone)]
pub enum Buffer {
    Bytes(Vec<u8>),
    F32(Vec<f32>),
    /// Shared pixels on the UI side (copied once into the outgoing `ArrayBuffer`).
    SharedF32(Arc<Vec<f32>>),
}

impl Buffer {
    /// The buffer's bytes (native endianness; wasm is little-endian).
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Buffer::Bytes(v) => v,
            Buffer::F32(v) => f32_bytes(v),
            Buffer::SharedF32(v) => f32_bytes(v),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        match self {
            Buffer::Bytes(v) => v,
            other => other.as_bytes().to_vec(),
        }
    }

    pub fn into_f32(self) -> Vec<f32> {
        match self {
            Buffer::F32(v) => v,
            Buffer::SharedF32(v) => Arc::try_unwrap(v).unwrap_or_else(|v| v.as_ref().clone()),
            Buffer::Bytes(b) => b
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns; u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// A request or reply: a small serialized `header` plus raw `buffers`.
#[derive(Debug, Clone, Default)]
pub struct WorkerMsg {
    pub header: Vec<u8>,
    pub buffers: Vec<Buffer>,
}

impl WorkerMsg {
    pub fn new(header: Vec<u8>, buffers: Vec<Buffer>) -> Self {
        Self { header, buffers }
    }
}

/// Executes a request message (inside a worker, or on the UI thread as a fallback) and returns
/// the reply. One function serves all job kinds: it dispatches on the header.
pub type WorkerHandler = fn(WorkerMsg, &JobCtx) -> Result<WorkerMsg, String>;

// ---------------------------------------------------------------------------------------------
// Control / context

#[derive(Default)]
struct ControlInner {
    cancelled: AtomicBool,
    progress: Mutex<Option<(f32, String)>>,
    /// Web: id of the pool task (0 = none).
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    task: AtomicU64,
}

/// UI-side handle of a running job: progress and cancellation.
#[derive(Clone, Default)]
pub struct JobControl(Arc<ControlInner>);

impl JobControl {
    /// Last reported `(fraction 0..=1, stage)`.
    pub fn progress(&self) -> Option<(f32, String)> {
        self.0.progress.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set_progress(&self, f: f32, msg: &str) {
        let mut p = self.0.progress.lock().unwrap_or_else(|e| e.into_inner());
        let changed = match &*p {
            Some((old, m)) => (f - old).abs() >= 0.01 || m != msg,
            None => true,
        };
        if changed {
            *p = Some((f, msg.to_string()));
            drop(p);
            crate::pending::wake();
        }
    }

    /// Ask the job to stop. Native: the job sees [`JobCtx::cancelled`] and returns early. Web:
    /// a queued job is dropped and a running one has its worker terminated. Either way the
    /// [`Pending`] may end up abandoned instead of yielding a value.
    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Relaxed);
        #[cfg(target_arch = "wasm32")]
        {
            let id = self.0.task.load(Ordering::Relaxed);
            if id != 0 {
                web_pool::cancel(id);
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Relaxed)
    }
}

/// What a running job sees: report progress, check for cancellation.
pub struct JobCtx {
    control: JobControl,
    /// Web worker: forwards progress to the UI thread with `postMessage`.
    sink: Option<Box<dyn Fn(f32, &str)>>,
}

impl JobCtx {
    /// A context not connected to anything (tests, synchronous use).
    pub fn detached() -> Self {
        Self { control: JobControl::default(), sink: None }
    }

    fn local(control: JobControl) -> Self {
        Self { control, sink: None }
    }

    pub fn progress(&self, fraction: f32, stage: &str) {
        match &self.sink {
            Some(s) => s(fraction, stage),
            None => self.control.set_progress(fraction, stage),
        }
    }

    /// Native: true once [`JobControl::cancel`] was called. Inside a worker always false (the
    /// worker is terminated instead).
    pub fn cancelled(&self) -> bool {
        self.control.is_cancelled()
    }
}

// ---------------------------------------------------------------------------------------------
// Jobs

/// A heavy job that can run on a native thread or in a Web Worker.
pub trait WorkerJob: Send + 'static {
    type Output: Send + 'static;

    /// Run directly (native threads).
    fn run(self, ctx: &JobCtx) -> Self::Output;

    /// Web: the request message for the worker.
    fn into_msg(self) -> Result<WorkerMsg, String>;

    /// Web: the function executing request messages (in the worker, or on the UI thread when
    /// workers are unavailable). Must be the handler the worker was started with.
    fn handler() -> WorkerHandler;

    /// Web: the output from the worker's reply (or a transport / job error), on the UI thread.
    fn output_from_msg(reply: Result<WorkerMsg, String>) -> Self::Output;
}

/// A started job: its result and its control handle.
pub struct Task<T> {
    pub pending: Pending<T>,
    pub control: JobControl,
}

/// Run a closure on a new OS thread with a [`JobCtx`] (native). On the web this runs on the UI
/// thread (deferred to a microtask) like [`crate::spawn`].
pub fn spawn_job<T: Send + 'static>(
    f: impl FnOnce(&JobCtx) -> T + Send + 'static,
) -> Task<T> {
    let control = JobControl::default();
    let ctl = control.clone();
    let pending = crate::spawn(move || f(&JobCtx::local(ctl)));
    Task { pending, control }
}

/// Run a heavy job off the UI thread: a new OS thread on native, a Web Worker on the web
/// (falling back to the UI thread if workers are unavailable).
pub fn spawn_worker<J: WorkerJob>(job: J) -> Task<J::Output> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        spawn_job(move |ctx| job.run(ctx))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let control = JobControl::default();
        let msg = match job.into_msg() {
            Ok(m) => m,
            Err(e) => {
                return Task { pending: Pending::ready(J::output_from_msg(Err(e))), control };
            }
        };
        let (c, pending) = pending_pair();
        let finish: Box<dyn FnOnce(Result<WorkerMsg, String>)> =
            Box::new(move |reply| c.complete(J::output_from_msg(reply)));
        web_pool::submit(J::handler(), msg, finish, control.clone());
        Task { pending, control }
    }
}

/// Run a job through its message path on the current thread: exactly what a worker does, plus
/// the conversions on both sides. This is the web fallback; also handy to test the protocol.
pub fn run_via_messages<J: WorkerJob>(job: J, ctx: &JobCtx) -> J::Output {
    let reply = job.into_msg().and_then(|m| (J::handler())(m, ctx));
    J::output_from_msg(reply)
}

/// Web: call first thing in the wasm start function. Returns `true` when running inside a
/// Web Worker, after installing the request loop for `handler`; the caller must then return
/// without starting the UI. Always `false` on native and on the page itself.
pub fn worker_entry(handler: WorkerHandler) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        web_worker::entry(handler)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = handler;
        false
    }
}

/// Number of workers the pool may use (web), or available threads (native).
pub fn worker_parallelism() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        web_pool::max_workers()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }
}

// ---------------------------------------------------------------------------------------------
// Web: JS conversions shared by both sides

#[cfg(target_arch = "wasm32")]
mod js {
    use super::{Buffer, WorkerMsg};
    use crate::web::{get, set};
    use js_sys::{Array, Float32Array, Uint8Array};
    use wasm_bindgen::JsValue;

    /// Fill `obj.header` / `obj.bufs` from `msg`; returns the transfer list.
    pub fn put_msg(obj: &JsValue, msg: WorkerMsg) -> Array {
        let transfer = Array::new();
        let header = Uint8Array::new_with_length(msg.header.len() as u32);
        header.copy_from(&msg.header);
        transfer.push(&header.buffer());
        set(obj, "header", &header);
        let bufs = Array::new();
        for b in msg.buffers {
            let entry = js_sys::Object::new();
            let (kind, ab) = match &b {
                Buffer::Bytes(v) => {
                    let a = Uint8Array::new_with_length(v.len() as u32);
                    a.copy_from(v);
                    ("u8", a.buffer())
                }
                Buffer::F32(v) => {
                    let a = Float32Array::new_with_length(v.len() as u32);
                    a.copy_from(v);
                    ("f32", a.buffer())
                }
                Buffer::SharedF32(v) => {
                    let a = Float32Array::new_with_length(v.len() as u32);
                    a.copy_from(v);
                    ("f32", a.buffer())
                }
            };
            drop(b); // free wasm memory early
            set(&entry, "k", &JsValue::from_str(kind));
            set(&entry, "b", &ab);
            transfer.push(&ab);
            bufs.push(&entry);
        }
        set(obj, "bufs", &bufs);
        transfer
    }

    /// Read `obj.header` / `obj.bufs` (copying once into wasm memory).
    pub fn take_msg(obj: &JsValue) -> WorkerMsg {
        let header = get(obj, "header").map(|h| Uint8Array::new(&h).to_vec()).unwrap_or_default();
        let mut buffers = vec![];
        if let Some(bufs) = get(obj, "bufs") {
            for entry in Array::from(&bufs).iter() {
                let Some(ab) = get(&entry, "b") else { continue };
                let kind = get(&entry, "k").and_then(|k| k.as_string()).unwrap_or_default();
                buffers.push(if kind == "f32" {
                    Buffer::F32(Float32Array::new(&ab).to_vec())
                } else {
                    Buffer::Bytes(Uint8Array::new(&ab).to_vec())
                });
            }
        }
        WorkerMsg { header, buffers }
    }

    pub fn num(obj: &JsValue, key: &str) -> f64 {
        get(obj, key).and_then(|v| v.as_f64()).unwrap_or(0.0)
    }

    pub fn string(obj: &JsValue, key: &str) -> String {
        get(obj, key).and_then(|v| v.as_string()).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------------------------
// Web: worker side

#[cfg(target_arch = "wasm32")]
mod web_worker {
    use super::js;
    use super::{JobControl, JobCtx, WorkerHandler};
    use crate::web::set;
    use std::cell::Cell;
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    pub fn entry(handler: WorkerHandler) -> bool {
        if web_sys::window().is_some() {
            return false;
        }
        let Ok(scope) = js_sys::global().dyn_into::<web_sys::DedicatedWorkerGlobalScope>() else {
            return false;
        };
        let s2 = scope.clone();
        let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
            let data = e.data();
            if js::string(&data, "t") != "job" {
                return;
            }
            let id = js::num(&data, "id");
            let msg = js::take_msg(&data);
            drop(data);
            let ctx = JobCtx { control: JobControl::default(), sink: Some(progress_sink(s2.clone(), id)) };
            let result = handler(msg, &ctx);
            let out = js_sys::Object::new();
            set(&out, "t", &JsValue::from_str("done"));
            set(&out, "id", &JsValue::from_f64(id));
            let transfer = match result {
                Ok(reply) => {
                    set(&out, "ok", &JsValue::TRUE);
                    js::put_msg(&out, reply)
                }
                Err(err) => {
                    set(&out, "ok", &JsValue::FALSE);
                    set(&out, "err", &JsValue::from_str(&err));
                    js_sys::Array::new()
                }
            };
            let mem = wasm_bindgen::memory()
                .dyn_into::<js_sys::WebAssembly::Memory>()
                .map(|m| m.buffer().unchecked_into::<js_sys::ArrayBuffer>().byte_length() as f64)
                .unwrap_or(0.0);
            set(&out, "mem", &JsValue::from_f64(mem));
            if let Err(e) = s2.post_message_with_transfer(&out, &transfer) {
                log::error!("worker: postMessage failed: {}", crate::web::err_string(&e));
            }
        });
        scope.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        on_message.forget();
        let ready = js_sys::Object::new();
        set(&ready, "t", &JsValue::from_str("ready"));
        let _ = scope.post_message(&ready);
        true
    }

    /// Posts `{t:"progress"}` messages, throttled to 1% steps or stage changes.
    fn progress_sink(scope: web_sys::DedicatedWorkerGlobalScope, id: f64) -> Box<dyn Fn(f32, &str)> {
        let last = Rc::new(Cell::new(-1.0f32));
        let stage = Rc::new(std::cell::RefCell::new(String::new()));
        Box::new(move |f: f32, msg: &str| {
            if (f - last.get()).abs() < 0.01 && *stage.borrow() == msg {
                return;
            }
            last.set(f);
            *stage.borrow_mut() = msg.to_string();
            let o = js_sys::Object::new();
            set(&o, "t", &JsValue::from_str("progress"));
            set(&o, "id", &JsValue::from_f64(id));
            set(&o, "f", &JsValue::from_f64(f as f64));
            set(&o, "msg", &JsValue::from_str(msg));
            let _ = scope.post_message(&o);
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Web: UI-side pool

#[cfg(target_arch = "wasm32")]
mod web_pool {
    use super::js;
    use super::{JobControl, JobCtx, RETIRE_BYTES, WorkerHandler, WorkerMsg};
    use crate::web::set;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::atomic::Ordering;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    /// Worker script, relative to the page (copied by Trunk next to `index.html`).
    const WORKER_SCRIPT: &str = "./worker.js";
    const MAX_WORKERS: usize = 4;
    /// Consecutive start-up failures after which workers are considered unavailable.
    const MAX_START_FAILURES: u32 = 2;

    struct PoolTask {
        id: u64,
        handler: WorkerHandler,
        msg: WorkerMsg,
        finish: Box<dyn FnOnce(Result<WorkerMsg, String>)>,
        control: JobControl,
    }

    struct Running {
        id: u64,
        finish: Box<dyn FnOnce(Result<WorkerMsg, String>)>,
        control: JobControl,
    }

    struct Slot {
        key: u32,
        worker: web_sys::Worker,
        ready: bool,
        running: Option<Running>,
        _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
        _on_error: Closure<dyn FnMut(web_sys::Event)>,
    }

    #[derive(Default)]
    struct Pool {
        slots: Vec<Slot>,
        queue: VecDeque<PoolTask>,
        next_id: u64,
        next_key: u32,
        start_failures: u32,
        broken: bool,
    }

    thread_local! {
        static POOL: RefCell<Pool> = RefCell::new(Pool::default());
    }

    pub fn max_workers() -> usize {
        let hc = web_sys::window().map_or(2.0, |w| w.navigator().hardware_concurrency());
        let hc = if hc.is_finite() && hc >= 1.0 { hc as usize } else { 2 };
        hc.saturating_sub(1).clamp(1, MAX_WORKERS)
    }

    pub fn submit(
        handler: WorkerHandler,
        msg: WorkerMsg,
        finish: Box<dyn FnOnce(Result<WorkerMsg, String>)>,
        control: JobControl,
    ) {
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            p.next_id += 1;
            let id = p.next_id;
            control.0.task.store(id, Ordering::Relaxed);
            p.queue.push_back(PoolTask { id, handler, msg, finish, control });
        });
        pump();
    }

    pub fn cancel(id: u64) {
        let dropped = POOL.with(|p| {
            let mut p = p.borrow_mut();
            if let Some(i) = p.queue.iter().position(|t| t.id == id) {
                return p.queue.remove(i).map(|t| (None, Some(t)));
            }
            let i = p.slots.iter().position(|s| s.running.as_ref().is_some_and(|r| r.id == id))?;
            let slot = p.slots.remove(i);
            slot.worker.terminate();
            log::info!("worker job {id} cancelled: worker terminated");
            Some((Some(slot), None))
        });
        // Dropping the task/slot outside the borrow abandons the job's Pending.
        if let Some((slot, task)) = dropped {
            drop(task);
            if let Some(slot) = slot {
                bury(slot);
            }
        }
        pump();
    }

    /// Drop a slot later: its closures may be on the stack right now.
    fn bury(slot: Slot) {
        wasm_bindgen_futures::spawn_local(async move { drop(slot) });
    }

    fn pump() {
        loop {
            enum Step {
                Done,
                Local(PoolTask),
                Post(web_sys::Worker, PoolTask),
                Spawn,
            }
            let step = POOL.with(|p| {
                let mut p = p.borrow_mut();
                if p.queue.is_empty() {
                    return Step::Done;
                }
                if p.broken {
                    return Step::Local(p.queue.pop_front().expect("non-empty"));
                }
                if let Some(i) = p.slots.iter().position(|s| s.ready && s.running.is_none()) {
                    let task = p.queue.pop_front().expect("non-empty");
                    let slot = &mut p.slots[i];
                    return Step::Post(slot.worker.clone(), task);
                }
                let starting = p.slots.iter().filter(|s| !s.ready).count();
                if p.slots.len() < max_workers() && p.queue.len() > starting {
                    return Step::Spawn;
                }
                Step::Done
            });
            match step {
                Step::Done => return,
                Step::Local(task) => run_local(task),
                Step::Spawn => {
                    if let Err(e) = spawn_worker() {
                        log::warn!("Web Workers unavailable ({e}); running jobs on the UI thread");
                        POOL.with(|p| p.borrow_mut().broken = true);
                    } else {
                        return; // wait for "ready"
                    }
                }
                Step::Post(worker, task) => post(worker, task),
            }
        }
    }

    fn run_local(task: PoolTask) {
        let PoolTask { handler, msg, finish, control, .. } = task;
        wasm_bindgen_futures::spawn_local(async move {
            let ctx = JobCtx::local(control);
            finish(handler(msg, &ctx));
        });
    }

    fn post(worker: web_sys::Worker, task: PoolTask) {
        let PoolTask { id, handler, msg, finish, control } = task;
        let obj = js_sys::Object::new();
        set(&obj, "t", &JsValue::from_str("job"));
        set(&obj, "id", &JsValue::from_f64(id as f64));
        let transfer = js::put_msg(&obj, msg);
        match worker.post_message_with_transfer(&obj, &transfer) {
            Ok(()) => POOL.with(|p| {
                let mut p = p.borrow_mut();
                if let Some(s) = p.slots.iter_mut().find(|s| s.worker == worker) {
                    s.running = Some(Running { id, finish, control });
                }
            }),
            Err(e) => {
                // The message was consumed by put_msg; nothing sensible to retry with.
                log::error!("postMessage to worker failed: {}", crate::web::err_string(&e));
                let _ = handler;
                finish(Err(format!("could not send job to worker: {}", crate::web::err_string(&e))));
            }
        }
    }

    fn spawn_worker() -> Result<(), String> {
        let opts = web_sys::WorkerOptions::new();
        opts.set_type(web_sys::WorkerType::Module);
        opts.set_name("prisme-worker");
        let worker = web_sys::Worker::new_with_options(WORKER_SCRIPT, &opts)
            .map_err(|e| crate::web::err_string(&e))?;
        let key = POOL.with(|p| {
            let mut p = p.borrow_mut();
            p.next_key += 1;
            p.next_key
        });
        let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
            on_message(key, e.data());
        });
        let on_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
            let msg = e
                .dyn_ref::<web_sys::ErrorEvent>()
                .map(|e| format!("{} ({}:{})", e.message(), e.filename(), e.lineno()))
                .unwrap_or_else(|| "worker error".into());
            on_failure(key, msg);
        });
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        worker.set_onmessageerror(Some(on_error.as_ref().unchecked_ref()));
        // Hand over the compiled module so the worker neither downloads nor compiles it again.
        let init = js_sys::Object::new();
        set(&init, "t", &JsValue::from_str("init"));
        set(&init, "module", &wasm_bindgen::module());
        worker.post_message(&init).map_err(|e| crate::web::err_string(&e))?;
        POOL.with(|p| {
            p.borrow_mut().slots.push(Slot {
                key,
                worker,
                ready: false,
                running: None,
                _on_message: on_message,
                _on_error: on_error,
            })
        });
        Ok(())
    }

    fn on_message(key: u32, data: JsValue) {
        match js::string(&data, "t").as_str() {
            "ready" => {
                POOL.with(|p| {
                    let mut p = p.borrow_mut();
                    p.start_failures = 0;
                    if let Some(s) = p.slots.iter_mut().find(|s| s.key == key) {
                        s.ready = true;
                    }
                });
                pump();
            }
            "progress" => POOL.with(|p| {
                let p = p.borrow();
                if let Some(r) = p.slots.iter().find(|s| s.key == key).and_then(|s| s.running.as_ref()) {
                    r.control.set_progress(js::num(&data, "f") as f32, &js::string(&data, "msg"));
                }
            }),
            "done" => {
                let (running, retired) = POOL.with(|p| {
                    let mut p = p.borrow_mut();
                    let Some(i) = p.slots.iter().position(|s| s.key == key) else { return (None, None) };
                    let running = p.slots[i].running.take();
                    let retired = (js::num(&data, "mem") > RETIRE_BYTES).then(|| {
                        let s = p.slots.remove(i);
                        s.worker.terminate();
                        s
                    });
                    (running, retired)
                });
                if let Some(slot) = retired {
                    bury(slot);
                }
                if let Some(r) = running {
                    let reply = if data_ok(&data) {
                        Ok(js::take_msg(&data))
                    } else {
                        Err(js::string(&data, "err"))
                    };
                    drop(data);
                    let _ = r.id;
                    r.control.0.task.store(0, Ordering::Relaxed);
                    (r.finish)(reply);
                }
                pump();
            }
            "init-error" => on_failure(key, js::string(&data, "msg")),
            _ => {}
        }
    }

    fn data_ok(data: &JsValue) -> bool {
        crate::web::get(data, "ok").and_then(|v| v.as_bool()).unwrap_or(false)
    }

    /// A worker failed to start or crashed (panic / out of memory): terminate it, abandon its
    /// job, and give up on workers after repeated start-up failures.
    fn on_failure(key: u32, msg: String) {
        let removed = POOL.with(|p| {
            let mut p = p.borrow_mut();
            let i = p.slots.iter().position(|s| s.key == key)?;
            let slot = p.slots.remove(i);
            slot.worker.terminate();
            if !slot.ready {
                p.start_failures += 1;
                if p.start_failures >= MAX_START_FAILURES {
                    p.broken = true;
                }
            }
            Some(slot)
        });
        if let Some(mut slot) = removed {
            if slot.ready {
                log::error!("worker crashed: {msg}");
            } else {
                log::warn!("worker failed to start: {msg}");
            }
            drop(slot.running.take()); // abandons the job's Pending
            bury(slot);
        }
        pump();
    }
}
