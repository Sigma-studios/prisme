//! Prisme desktop entry point (Windows, macOS, Linux).

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,prisme=info")).init();
    // `--list-formats`: openable extensions, one per line (used by shell completion).
    if std::env::args().nth(1).as_deref() == Some("--list-formats") {
        for e in prisme_io::supported_extensions() {
            println!("{e}");
        }
        return Ok(());
    }
    let start = std::time::Instant::now();
    // Files passed on the command line (also used by "Open with…").
    let files: Vec<std::path::PathBuf> = std::env::args().skip(1).map(Into::into).collect();
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Prisme")
            .with_inner_size([1480.0, 920.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: prisme_ui::wgpu_options(),
        persist_window: true,
        ..Default::default()
    };
    eframe::run_native(
        "Prisme",
        options,
        Box::new(move |cc| {
            let mut app = prisme_ui::PrismeApp::new(cc);
            for f in files {
                match std::fs::read(&f) {
                    Ok(bytes) => {
                        let name = f.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                        app.open_bytes(name, bytes, std::fs::canonicalize(&f).ok());
                    }
                    Err(e) => log::error!("cannot read {}: {e}", f.display()),
                }
            }
            log::info!("window ready in {:.0} ms", start.elapsed().as_secs_f64() * 1000.0);
            Ok(Box::new(app))
        }),
    )
}
