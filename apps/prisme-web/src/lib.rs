//! Prisme web entry point (WebGPU). Built with `trunk build --release` in this directory.

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
        wasm_logger::init(wasm_logger::Config::new(log::Level::Info));
        // The same wasm also runs inside the job Web Workers (see `worker.js`): there, serve
        // job requests instead of starting the UI.
        if prisme_ui::offload::worker_entry() {
            return;
        }
        wasm_bindgen_futures::spawn_local(async {
            let window = web_sys::window().expect("no window");
            let document = window.document().expect("no document");
            let missing_webgpu = js_sys_reflect_gpu(&window.navigator());
            if missing_webgpu {
                if let Some(el) = document.get_element_by_id("no-webgpu") {
                    el.set_inner_html("Prisme needs WebGPU (Chrome/Edge 113+, Safari 26+, Firefox 141+ on Windows).");
                }
                return;
            }
            let canvas = document
                .get_element_by_id("prisme-canvas")
                .expect("missing #prisme-canvas")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("not a canvas");
            let options = eframe::WebOptions { wgpu_options: prisme_ui::wgpu_options(), ..Default::default() };
            let result = eframe::WebRunner::new()
                .start(canvas, options, Box::new(|cc| Ok(Box::new(prisme_ui::PrismeApp::new(cc)))))
                .await;
            if let Some(loading) = document.get_element_by_id("loading") {
                loading.remove();
            }
            log::info!("Prisme web runner started: ok={}", result.is_ok());
            if let Err(e) = result {
                log::error!("failed to start Prisme: {e:?}");
            }
        });
    }

    /// True when `navigator.gpu` is missing.
    fn js_sys_reflect_gpu(nav: &web_sys::Navigator) -> bool {
        let v: &JsValue = nav.as_ref();
        let gpu = js_sys::Reflect::get(v, &JsValue::from_str("gpu")).unwrap_or(JsValue::UNDEFINED);
        gpu.is_undefined() || gpu.is_null()
    }
}
