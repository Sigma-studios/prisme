//! Pollable results of background work ([`Pending`]) and the spawners that produce them.

use std::future::Future;
use std::sync::{Arc, Condvar, Mutex, RwLock};

/// `Send` on native targets, no bound at all on wasm (where everything runs on one thread).
///
/// Use it to write futures/closures that are portable: a future that is `MaybeSend`
/// compiles on wasm even if it holds `JsValue`s, and on native it must be `Send`.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send> MaybeSend for T {}

/// `Send` on native targets, no bound at all on wasm (where everything runs on one thread).
#[cfg(target_arch = "wasm32")]
pub trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSend for T {}

enum State<T> {
    Waiting,
    Ready(T),
    Taken,
    /// The producer went away (panicked / dropped its [`Completer`]) without a value.
    Abandoned,
}

struct Shared<T> {
    state: Mutex<State<T>>,
    cv: Condvar,
}

/// A pollable result of a background operation.
///
/// Call [`Pending::poll`] once per frame; it returns `Some(value)` exactly once, when the
/// operation has completed. Dropping a `Pending` simply discards the result.
///
/// Native: backed by a thread. Web: backed by `spawn_local` + JS promises.
pub struct Pending<T> {
    shared: Arc<Shared<T>>,
}

/// The producing half of a [`Pending`]. Dropping it without calling
/// [`Completer::complete`] marks the `Pending` as abandoned.
pub struct Completer<T> {
    shared: Option<Arc<Shared<T>>>,
}

/// Create a connected `(Completer, Pending)` pair. Useful for callback-based APIs.
pub fn pending_pair<T>() -> (Completer<T>, Pending<T>) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State::Waiting),
        cv: Condvar::new(),
    });
    (
        Completer {
            shared: Some(shared.clone()),
        },
        Pending { shared },
    )
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl<T> Completer<T> {
    /// Resolve the paired [`Pending`] and fire the wake callback (see [`set_wake_callback`]).
    pub fn complete(mut self, value: T) {
        if let Some(shared) = self.shared.take() {
            *lock(&shared.state) = State::Ready(value);
            shared.cv.notify_all();
            wake();
        }
    }
}

impl<T> Drop for Completer<T> {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.take() {
            let mut st = lock(&shared.state);
            if matches!(*st, State::Waiting) {
                *st = State::Abandoned;
            }
            drop(st);
            shared.cv.notify_all();
            wake();
        }
    }
}

impl<T> Pending<T> {
    /// An already-resolved `Pending`.
    pub fn ready(value: T) -> Self {
        let (c, p) = pending_pair();
        c.complete(value);
        p
    }

    /// Take the result if it is available. Returns `Some` exactly once.
    pub fn poll(&mut self) -> Option<T> {
        let mut st = lock(&self.shared.state);
        if matches!(*st, State::Ready(_)) {
            match std::mem::replace(&mut *st, State::Taken) {
                State::Ready(v) => Some(v),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    /// `true` once the operation has finished (result available, already taken, or abandoned).
    pub fn is_done(&self) -> bool {
        !matches!(*lock(&self.shared.state), State::Waiting)
    }

    /// `true` if the producer died without producing a value (e.g. the worker panicked).
    /// Such a `Pending` will never yield a value; drop it.
    pub fn is_abandoned(&self) -> bool {
        matches!(*lock(&self.shared.state), State::Abandoned)
    }

    /// Block the current thread until the result is available (native only; never call this
    /// from the UI thread). Returns `None` if abandoned or already taken.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn block(mut self) -> Option<T> {
        {
            let mut st = lock(&self.shared.state);
            while matches!(*st, State::Waiting) {
                st = self.shared.cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }
        self.poll()
    }
}

impl<T> std::fmt::Debug for Pending<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending")
            .field("done", &self.is_done())
            .finish()
    }
}

// ---------------------------------------------------------------------------------------------
// Wake callback

type WakeFn = Arc<dyn Fn() + Send + Sync>;
static WAKE: RwLock<Option<WakeFn>> = RwLock::new(None);

/// Register a callback fired (from any thread) whenever any [`Pending`] resolves.
/// Typically `move || egui_ctx.request_repaint()` so the UI polls promptly.
pub fn set_wake_callback(f: impl Fn() + Send + Sync + 'static) {
    *WAKE.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(f));
}

fn wake() {
    let f = WAKE.read().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(f) = f {
        f();
    }
}

// ---------------------------------------------------------------------------------------------
// Spawners

/// Run a closure in the background.
///
/// Native: on a new OS thread. Web: deferred to a microtask on the (only) thread — the
/// closure still runs on the UI thread, so keep it short there.
pub fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Pending<T> {
    let (c, p) = pending_pair();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let res = std::thread::Builder::new()
            .name("prisme-bg".into())
            .spawn(move || c.complete(f()));
        if let Err(e) = res {
            log::error!("prisme-platform: failed to spawn thread: {e}");
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move { c.complete(f()) });
    }
    p
}

/// Drive a future to completion in the background.
///
/// Native: `pollster::block_on` on a new OS thread (so the future must be `Send`, which
/// [`MaybeSend`] enforces only on native). Web: `wasm_bindgen_futures::spawn_local`.
pub fn spawn_async<T: MaybeSend + 'static>(
    fut: impl Future<Output = T> + MaybeSend + 'static,
) -> Pending<T> {
    let (c, p) = pending_pair();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let res = std::thread::Builder::new()
            .name("prisme-async".into())
            .spawn(move || c.complete(pollster::block_on(fut)));
        if let Err(e) = res {
            log::error!("prisme-platform: failed to spawn thread: {e}");
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move { c.complete(fut.await) });
    }
    p
}
