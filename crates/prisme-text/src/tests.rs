use super::*;
use prisme_core::Gradient;
use prisme_core::layer::{Text3D, TextGlow, TextOutline, TextShadow, TextWarp};

fn renderer() -> TextRenderer {
    TextRenderer::new(FontDb::new())
}

fn base() -> TextLayer {
    TextLayer {
        text: "Hello".into(),
        font_family: "sans-serif".into(),
        size: 64.0,
        position: [10.0, 20.0],
        fill: TextFill::Solid([0.8, 0.1, 0.05, 1.0]),
        ..Default::default()
    }
}

fn alpha_sum(g: &TileGrid) -> f32 {
    g.tiles
        .values()
        .map(|t| t.f32_view().iter().skip(3).step_by(4).sum::<f32>())
        .sum()
}

fn opaque_pixel(g: &TileGrid) -> Option<(i32, i32, [f32; 4])> {
    let b = g.content_bounds();
    for y in b.y0..b.y1 {
        for x in b.x0..b.x1 {
            let p = g.get_pixel(x, y);
            if p[3] >= 0.999 {
                return Some((x, y, p));
            }
        }
    }
    None
}

fn within(inner: IRect, outer: [f32; 4]) -> bool {
    inner.x0 as f32 >= outer[0].floor() - 1.0
        && inner.y0 as f32 >= outer[1].floor() - 1.0
        && inner.x1 as f32 <= outer[2].ceil() + 1.0
        && inner.y1 as f32 <= outer[3].ceil() + 1.0
}

#[test]
fn basic_text_coverage_and_colour() {
    let r = renderer();
    let l = base();
    let g = r.rasterize(&l, 0).expect("non-empty");
    let cb = g.content_bounds();
    assert!(!cb.is_empty());
    let b = r.bounds(&l);
    assert!(within(cb, b), "{cb:?} vs {b:?}");
    assert!(cb.width() > 100 && cb.height() > 30, "{cb:?}");
    // Text starts near the position.
    assert!(
        cb.x0 >= 10 && cb.x0 < 30 && cb.y0 >= 20 && cb.y0 < 50,
        "{cb:?}"
    );
    let (_, _, p) = opaque_pixel(&g).expect("some fully covered pixel");
    for k in 0..3 {
        assert!((p[k] - [0.8, 0.1, 0.05][k]).abs() < 2e-3, "{p:?}");
    }
}

#[test]
fn empty_text_is_none() {
    let r = renderer();
    let l = TextLayer {
        text: "  \n ".into(),
        ..base()
    };
    assert!(r.rasterize(&l, 0).is_none());
    assert_eq!(r.caret_positions(&l).len(), 5);
}

#[test]
fn outline_grows_bounds() {
    let r = renderer();
    let l = base();
    let mut lo = base();
    lo.outline = Some(TextOutline {
        width: 6.0,
        color: [0.0, 0.0, 1.0, 1.0],
    });
    let (b0, b1) = (r.bounds(&l), r.bounds(&lo));
    assert!(b1[0] < b0[0] - 5.0 && b1[2] > b0[2] + 5.0, "{b0:?} {b1:?}");
    let (g0, g1) = (r.rasterize(&l, 0).unwrap(), r.rasterize(&lo, 0).unwrap());
    let (c0, c1) = (g0.content_bounds(), g1.content_bounds());
    assert!(c1.x0 <= c0.x0 - 5 && c1.x1 >= c0.x1 + 5, "{c0:?} {c1:?}");
    assert!(within(c1, b1));
    // Outline colour appears.
    let found = (c1.y0..c1.y1).any(|y| {
        (c1.x0..c1.x1).any(|x| {
            let p = g1.get_pixel(x, y);
            p[3] > 0.99 && p[2] > 0.99 && p[0] < 0.01
        })
    });
    assert!(found);
}

#[test]
fn glow_is_soft_outside_glyphs() {
    let r = renderer();
    let l = base();
    let mut lg = base();
    lg.glow = Some(TextGlow {
        radius: 12.0,
        color: [1.0, 1.0, 0.0, 1.0],
        strength: 1.0,
        inner: false,
    });
    let g0 = r.rasterize(&l, 0).unwrap();
    let g1 = r.rasterize(&lg, 0).unwrap();
    let b = g1.content_bounds();
    let mut soft = 0;
    for y in b.y0..b.y1 {
        for x in b.x0..b.x1 {
            if g0.get_pixel(x, y)[3] == 0.0 {
                let a = g1.get_pixel(x, y)[3];
                if a > 0.02 && a < 0.98 {
                    soft += 1;
                }
            }
        }
    }
    assert!(soft > 500, "soft glow pixels: {soft}");
    assert!(within(b, r.bounds(&lg)), "{b:?} {:?}", r.bounds(&lg));
    // inner glow stays inside the glyphs
    let mut li = base();
    li.glow = Some(TextGlow {
        radius: 6.0,
        color: [1.0, 1.0, 0.0, 1.0],
        strength: 1.0,
        inner: true,
    });
    let gi = r.rasterize(&li, 0).unwrap();
    assert_eq!(gi.content_bounds(), g0.content_bounds());
}

#[test]
fn shadow_offsets() {
    let r = renderer();
    let mut l = base();
    l.shadow = Some(TextShadow {
        offset: [20.0, 15.0],
        blur: 0.0,
        color: [0.0, 0.0, 0.0, 1.0],
    });
    let g0 = r.rasterize(&base(), 0).unwrap().content_bounds();
    let g1 = r.rasterize(&l, 0).unwrap().content_bounds();
    assert_eq!(g1.x0, g0.x0);
    assert!(
        (g1.x1 - (g0.x1 + 20)).abs() <= 1 && (g1.y1 - (g0.y1 + 15)).abs() <= 1,
        "{g0:?} {g1:?}"
    );
}

#[test]
fn gradient_varies_across_x() {
    let r = renderer();
    let mut l = base();
    l.text = "IIIIIIII".into();
    l.fill = TextFill::Gradient {
        gradient: Gradient::two([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
        angle: 0.0,
    };
    let g = r.rasterize(&l, 0).unwrap();
    let b = g.content_bounds();
    let y = (b.y0 + b.y1) / 2;
    let row: Vec<[f32; 4]> = (b.x0..b.x1)
        .map(|x| g.get_pixel(x, y))
        .filter(|p| p[3] > 0.99)
        .collect();
    let (first, last) = (row.first().unwrap(), row.last().unwrap());
    assert!(first[0] > 0.8 && first[2] < 0.2, "{first:?}");
    assert!(last[2] > 0.8 && last[0] < 0.2, "{last:?}");
}

#[test]
fn warp_changes_shape() {
    let r = renderer();
    let mut l = base();
    l.text = "Warped text".into();
    let g0 = r.rasterize(&l, 0).unwrap();
    for w in [
        TextWarp::Arc { bend: 0.6 },
        TextWarp::Wave {
            amplitude: 10.0,
            wavelength: 80.0,
        },
        TextWarp::Bulge { amount: 0.5 },
        TextWarp::Flag {
            amplitude: 12.0,
            wavelength: 60.0,
        },
    ] {
        l.warp = w;
        let g1 = r.rasterize(&l, 0).unwrap();
        let (c0, c1) = (g0.content_bounds(), g1.content_bounds());
        assert!(c0 != c1, "{w:?} kept bounds {c0:?}");
        assert!(
            within(c1, r.bounds(&l)),
            "{w:?}: {c1:?} vs {:?}",
            r.bounds(&l)
        );
    }
}

#[test]
fn rotation_rotates() {
    let r = renderer();
    let mut l = base();
    l.text = "Long line of text".into();
    let c0 = r.rasterize(&l, 0).unwrap().content_bounds();
    l.rotation = std::f32::consts::FRAC_PI_2;
    let c1 = r.rasterize(&l, 0).unwrap().content_bounds();
    assert!(
        c1.height() > c0.width() - 10 && c1.width() < c0.height() + 10,
        "{c0:?} {c1:?}"
    );
}

#[test]
fn three_d_has_coverage_and_sides() {
    let r = renderer();
    let mut l = base();
    l.text = "3D".into();
    l.size = 120.0;
    l.fill = TextFill::Solid([1.0, 1.0, 1.0, 1.0]);
    l.three_d = Text3D {
        enabled: true,
        depth: 30.0,
        radius: 400.0,
        rotation: [0.6, 0.4, 0.0],
        side_color: [0.0, 1.0, 0.0, 1.0],
        ..Default::default()
    };
    let g = r.rasterize(&l, 0).expect("3D coverage");
    let b = g.content_bounds();
    assert!(b.width() > 50 && b.height() > 50, "{b:?}");
    assert!(within(b, r.bounds(&l)), "{b:?} {:?}", r.bounds(&l));
    let (mut side, mut front) = (0, 0);
    for y in b.y0..b.y1 {
        for x in b.x0..b.x1 {
            let p = g.get_pixel(x, y);
            if p[3] > 0.99 {
                if p[1] > 0.95 && p[0] < 0.05 {
                    side += 1;
                }
                if p[0] > 0.95 && p[1] > 0.95 {
                    front += 1;
                }
            }
        }
    }
    assert!(side > 200 && front > 200, "side {side} front {front}");

    // sphere + flat + orthographic + gradient sides + outline on the silhouette
    l.three_d.show_sphere = true;
    l.three_d.perspective = 0.0;
    l.three_d.side_gradient = Some(Gradient::two([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]));
    l.outline = Some(TextOutline {
        width: 4.0,
        color: [1.0, 0.0, 1.0, 1.0],
    });
    let g2 = r.rasterize(&l, 0).unwrap();
    let b2 = g2.content_bounds();
    assert!(b2.width() >= 800 - 2, "sphere drawn: {b2:?}");
    l.three_d.radius = 0.0;
    l.three_d.show_sphere = false;
    assert!(r.rasterize(&l, 0).is_some());
}

#[test]
fn mip1_is_half_size() {
    let r = renderer();
    let mut l = base();
    l.size = 100.0;
    l.text = "Mip test".into();
    let c0 = r.rasterize(&l, 0).unwrap().content_bounds();
    let c1 = r.rasterize(&l, 1).unwrap().content_bounds();
    assert!(
        (c1.width() as f32 - c0.width() as f32 / 2.0).abs() <= 2.0,
        "{c0:?} {c1:?}"
    );
    assert!(
        (c1.height() as f32 - c0.height() as f32 / 2.0).abs() <= 2.0,
        "{c0:?} {c1:?}"
    );
    assert!((c1.x0 - c0.x0 / 2).abs() <= 1);
    let a0 = alpha_sum(&r.rasterize(&l, 0).unwrap());
    let a1 = alpha_sum(&r.rasterize(&l, 1).unwrap());
    assert!((a1 * 4.0 / a0 - 1.0).abs() < 0.05, "{a0} {a1}");
}

#[test]
fn big_text_chunks_and_multiline() {
    let r = renderer();
    let mut l = base();
    l.text = "Big\nTwo lines".into();
    l.size = 600.0;
    l.align = prisme_core::layer::TextAlign::Center;
    l.glow = Some(TextGlow {
        radius: 20.0,
        color: [0.0, 0.0, 0.0, 1.0],
        strength: 0.5,
        inner: false,
    });
    let g = r.rasterize(&l, 0).unwrap();
    let b = g.content_bounds();
    assert!(b.width() > 1500 && b.height() > 1000, "{b:?}");
    assert!(within(b, r.bounds(&l)));
    let carets = r.caret_positions(&l);
    assert_eq!(carets.len(), l.text.chars().count() + 1);
    assert!(carets[4][1] > carets[0][1], "second line below the first");
    assert!(carets[1][0] > carets[0][0] && carets[2][0] > carets[1][0]);
}

#[test]
fn add_font_data_returns_family() {
    let db = FontDb::new();
    let fams = db.add_font_data(epaint_default_fonts::UBUNTU_LIGHT.to_vec());
    assert_eq!(fams, vec!["Ubuntu".to_string()]);
    assert!(db.add_font_data(vec![1, 2, 3]).is_empty());
    assert!(db.families().contains(&"Hack".to_string()));
    assert!(!db.styles("Ubuntu").is_empty());
    assert!(!db.styles("sans-serif").is_empty());
}

#[test]
fn unknown_family_falls_back() {
    let r = renderer();
    let mut l = base();
    l.font_family = "No Such Font 123".into();
    assert!(r.rasterize(&l, 0).is_some());
    l.font_family = "monospace".into();
    l.weight = 700;
    l.italic = true; // synthetic bold + italic on Hack
    assert!(r.rasterize(&l, 0).is_some());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn system_fonts_load() {
    let db = FontDb::new();
    let n0 = db.families().len();
    let t0 = std::time::Instant::now();
    db.load_system_fonts_async();
    while !db.system_fonts_loaded() {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let load = t0.elapsed();
    assert!(db.families().len() >= n0);
    let r = TextRenderer::new(db.clone());
    let mut l = base();
    l.text = "Headline Text".into();
    l.size = 200.0;
    let t1 = std::time::Instant::now();
    assert!(r.rasterize(&l, 0).is_some());
    let first = t1.elapsed();
    let t2 = std::time::Instant::now();
    assert!(r.rasterize(&l, 0).is_some());
    println!(
        "system fonts: {} families in {load:?}; headline first {first:?}, cached {:?}",
        db.families().len(),
        t2.elapsed()
    );
}

/// Timing of a 200 px headline (run with `--release -- --nocapture`).
#[test]
fn headline_perf() {
    let r = renderer();
    let mut l = base();
    l.text = "Headline Text".into();
    l.size = 200.0;
    let t0 = std::time::Instant::now();
    let _ = r.rasterize(&l, 0).unwrap();
    let first = t0.elapsed();
    let t1 = std::time::Instant::now();
    let n = 10;
    for _ in 0..n {
        let _ = r.rasterize(&l, 0).unwrap();
    }
    let avg = t1.elapsed() / n;
    l.outline = Some(TextOutline {
        width: 4.0,
        color: [0.0, 0.0, 0.0, 1.0],
    });
    l.glow = Some(TextGlow {
        radius: 16.0,
        color: [1.0, 1.0, 0.0, 1.0],
        strength: 0.8,
        inner: false,
    });
    l.shadow = Some(TextShadow {
        offset: [8.0, 8.0],
        blur: 10.0,
        color: [0.0, 0.0, 0.0, 0.6],
    });
    let t2 = std::time::Instant::now();
    let _ = r.rasterize(&l, 0).unwrap();
    let fx = t2.elapsed();
    l.three_d = Text3D {
        enabled: true,
        rotation: [0.5, 0.3, 0.0],
        ..Default::default()
    };
    let t3 = std::time::Instant::now();
    let _ = r.rasterize(&l, 0).unwrap();
    let td = t3.elapsed();
    println!(
        "headline 200px: first {first:?}, cached avg {avg:?}, with effects {fx:?}, 3D+effects {td:?}"
    );
    if !cfg!(debug_assertions) {
        assert!(avg.as_millis() < 30, "{avg:?}");
    }
}
