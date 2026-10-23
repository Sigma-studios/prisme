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
            let navigator = window.navigator();
            // wgpu drops WebGPU when the browser gives no adapter, and Prisme has no other
            // backend: ask first, so the page can say why instead of failing later.
            let problem = match gpu(&navigator) {
                None => Some(super::help::Problem::NoWebGpu),
                Some(gpu) if !has_adapter(&gpu).await => Some(super::help::Problem::NoAdapter),
                Some(_) => None,
            };
            if let Some(problem) = problem {
                log::error!("Prisme can't start: {problem:?}");
                show_help(&document, &navigator, &problem);
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
            log::info!("Prisme web runner started: ok={}", result.is_ok());
            match result {
                Ok(()) => {
                    if let Some(loading) = document.get_element_by_id("loading") {
                        loading.remove();
                    }
                }
                Err(e) => {
                    log::error!("failed to start Prisme: {e:?}");
                    let message = e.as_string().unwrap_or_else(|| format!("{e:?}"));
                    show_help(&document, &navigator, &super::help::Problem::StartFailed(message));
                }
            }
        });
    }

    /// `navigator.gpu`, when the browser has it.
    fn gpu(nav: &web_sys::Navigator) -> Option<JsValue> {
        let gpu = js_sys::Reflect::get(nav.as_ref(), &JsValue::from_str("gpu")).ok()?;
        (!gpu.is_undefined() && !gpu.is_null()).then_some(gpu)
    }

    /// Whether `navigator.gpu.requestAdapter()` gives an adapter (the check wgpu makes).
    async fn has_adapter(gpu: &JsValue) -> bool {
        let request = js_sys::Reflect::get(gpu, &JsValue::from_str("requestAdapter"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
        let Some(promise) = request.and_then(|f| f.call0(gpu).ok()) else {
            return false;
        };
        wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise))
            .await
            .is_ok_and(|adapter| !adapter.is_undefined() && !adapter.is_null())
    }

    fn show_help(document: &web_sys::Document, nav: &web_sys::Navigator, problem: &super::help::Problem) {
        let user_agent = nav.user_agent().unwrap_or_default();
        let brave = js_sys::Reflect::get(nav.as_ref(), &JsValue::from_str("brave")).is_ok_and(|b| !b.is_undefined());
        let html = super::help::html(problem, &super::help::Browser::detect(&user_agent, brave));
        if let Some(el) = document.get_element_by_id("loading") {
            el.set_inner_html(&html);
            let _ = el.set_attribute("class", "help");
        }
    }
}

/// What to tell someone whose browser can't run Prisme: platform-independent, so it is
/// unit-tested natively.
#[cfg(any(target_arch = "wasm32", test))]
mod help {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Problem {
        /// No `navigator.gpu`.
        NoWebGpu,
        /// `navigator.gpu` is there but `requestAdapter()` gave nothing.
        NoAdapter,
        /// WebGPU looked fine but starting the GPU failed.
        StartFailed(String),
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Engine {
        /// Chrome, Edge, Vivaldi, Brave, Opera… with the scheme of their internal pages.
        Chromium { scheme: &'static str, name: &'static str },
        Firefox,
        Safari,
        Other,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Os {
        Linux,
        Windows,
        MacOs,
        Android,
        Ios,
        Other,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Browser {
        pub engine: Engine,
        pub os: Os,
    }

    impl Browser {
        pub fn detect(user_agent: &str, brave: bool) -> Self {
            let ua = user_agent;
            let os = if ua.contains("Android") {
                Os::Android
            } else if ua.contains("iPhone") || ua.contains("iPad") {
                Os::Ios
            } else if ua.contains("Windows") {
                Os::Windows
            } else if ua.contains("Mac OS X") || ua.contains("Macintosh") {
                Os::MacOs
            } else if ua.contains("Linux") || ua.contains("X11") || ua.contains("CrOS") {
                Os::Linux
            } else {
                Os::Other
            };
            let engine = if ua.contains("Firefox/") || ua.contains("FxiOS/") {
                Engine::Firefox
            } else if ua.contains("Edg/") {
                Engine::Chromium { scheme: "edge", name: "Edge" }
            } else if ua.contains("OPR/") {
                Engine::Chromium { scheme: "opera", name: "Opera" }
            } else if ua.contains("Vivaldi/") {
                Engine::Chromium { scheme: "vivaldi", name: "Vivaldi" }
            } else if brave {
                Engine::Chromium { scheme: "brave", name: "Brave" }
            } else if ua.contains("Chrome/") || ua.contains("Chromium/") || ua.contains("CriOS/") {
                // Vivaldi no longer names itself; chrome:// pages work in every Chromium.
                Engine::Chromium { scheme: "chrome", name: "your browser" }
            } else if ua.contains("Safari/") {
                Engine::Safari
            } else {
                Engine::Other
            };
            Self { engine, os }
        }
    }

    const SUPPORT: &str = "https://github.com/gpuweb/gpuweb/wiki/Implementation-Status";

    /// An internal page (`chrome://…`, `about:config`): pages can't link to those, so it is
    /// shown with a button copying it, to paste in the address bar.
    fn internal(url: &str) -> String {
        format!(
            "<code>{url}</code> <button type=\"button\" onclick=\"navigator.clipboard.writeText('{url}');this.textContent='Copied'\">Copy</button>"
        )
    }

    fn escape(s: &str) -> String {
        s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
    }

    /// The help shown in place of the loading screen.
    pub fn html(problem: &Problem, browser: &Browser) -> String {
        let what = match problem {
            Problem::NoWebGpu => "Prisme needs WebGPU, which this browser doesn't offer.".to_string(),
            Problem::NoAdapter => "Prisme needs WebGPU. This browser has it, but gave Prisme no GPU to use.".to_string(),
            Problem::StartFailed(e) => format!("Prisme couldn't start the GPU: <code>{}</code>", escape(e)),
        };
        let mut steps: Vec<String> = Vec::new();
        match (browser.engine, browser.os) {
            (Engine::Chromium { scheme, .. }, Os::Linux) => {
                steps.push(format!(
                    "Open {} and set it to <b>Enabled</b>, then the same for {}; relaunch.",
                    internal(&format!("{scheme}://flags/#enable-unsafe-webgpu")),
                    internal(&format!("{scheme}://flags/#enable-vulkan")),
                ));
                steps.push(
                    "Still nothing? Start the browser with <code>--enable-features=Vulkan</code> (on Wayland, \
                     <code>--ozone-platform=x11</code> may be needed too)."
                        .into(),
                );
                steps.push(format!(
                    "{} should then list WebGPU as <b>Hardware accelerated</b>. It needs working Vulkan drivers; \
                     on NixOS, a browser bundling its own Vulkan loader can't find them.",
                    internal(&format!("{scheme}://gpu")),
                ));
            }
            (Engine::Chromium { scheme, name }, _) => {
                if *problem == Problem::NoWebGpu {
                    steps.push(format!("Update {name}: WebGPU is on since Chrome and Edge 113."));
                } else {
                    steps.push(format!(
                        "Turn on <b>Use graphics acceleration when available</b> in {name}'s system settings, \
                         and relaunch."
                    ));
                }
                steps.push(format!(
                    "{} shows whether WebGPU is blocked for this GPU or driver; updating the graphics driver often helps.",
                    internal(&format!("{scheme}://gpu")),
                ));
            }
            (Engine::Firefox, Os::Windows) => {
                steps.push("Update Firefox: WebGPU is on since Firefox 141 on Windows.".into());
            }
            (Engine::Firefox, _) => {
                steps.push(format!(
                    "Firefox's WebGPU is still experimental here: set <code>dom.webgpu.enabled</code> to true in {} \
                     and restart, or use Chrome or Edge.",
                    internal("about:config"),
                ));
            }
            (Engine::Safari, _) => {
                steps.push("Update Safari: WebGPU is on in Safari 26 (macOS, iOS and iPadOS 26).".into());
            }
            (Engine::Other, _) => {
                steps.push("Use a current Chrome or Edge (113+), Safari 26+, or Firefox 141+ on Windows.".into());
            }
        }
        let steps: String = steps.iter().map(|s| format!("<li>{s}</li>")).collect();
        format!(
            "<h1>◆ Prisme</h1><p>{what}</p><ol>{steps}</ol>\
             <p class=\"small\">Internal pages like these can't be opened from a link: paste them in the \
             address bar. <a href=\"{SUPPORT}\" target=\"_blank\" rel=\"noopener\">Which browsers support WebGPU</a></p>"
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const VIVALDI_LINUX: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36";
        const EDGE_WINDOWS: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0";
        const FIREFOX_MAC: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:144.0) Gecko/20100101 Firefox/144.0";
        const SAFARI_MAC: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15";

        #[test]
        fn detects_browsers() {
            assert_eq!(
                Browser::detect(VIVALDI_LINUX, false),
                Browser { engine: Engine::Chromium { scheme: "chrome", name: "your browser" }, os: Os::Linux }
            );
            assert_eq!(Browser::detect(VIVALDI_LINUX, true).engine, Engine::Chromium { scheme: "brave", name: "Brave" });
            assert_eq!(
                Browser::detect(EDGE_WINDOWS, false),
                Browser { engine: Engine::Chromium { scheme: "edge", name: "Edge" }, os: Os::Windows }
            );
            assert_eq!(Browser::detect(FIREFOX_MAC, false), Browser { engine: Engine::Firefox, os: Os::MacOs });
            assert_eq!(Browser::detect(SAFARI_MAC, false), Browser { engine: Engine::Safari, os: Os::MacOs });
        }

        #[test]
        fn chromium_on_linux_gets_the_flags() {
            let html = html(&Problem::NoAdapter, &Browser::detect(VIVALDI_LINUX, false));
            assert!(html.contains("gave Prisme no GPU"));
            assert!(html.contains("<code>chrome://flags/#enable-unsafe-webgpu</code>"));
            assert!(html.contains("chrome://flags/#enable-vulkan"));
            assert!(html.contains("NixOS"));
            assert!(html.contains(SUPPORT));
        }

        #[test]
        fn other_platforms_get_their_own_advice() {
            let edge = html(&Problem::NoAdapter, &Browser::detect(EDGE_WINDOWS, false));
            assert!(edge.contains("graphics acceleration") && edge.contains("edge://gpu"));
            assert!(!edge.contains("enable-unsafe-webgpu"));
            assert!(html(&Problem::NoWebGpu, &Browser::detect(FIREFOX_MAC, false)).contains("dom.webgpu.enabled"));
            assert!(html(&Problem::NoWebGpu, &Browser::detect(SAFARI_MAC, false)).contains("Safari 26"));
        }

        #[test]
        fn start_errors_are_shown_escaped() {
            let html = html(&Problem::StartFailed("no <surface>".into()), &Browser::detect(VIVALDI_LINUX, false));
            assert!(html.contains("no &lt;surface&gt;"));
        }
    }
}
