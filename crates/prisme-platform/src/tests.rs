#![cfg(not(target_arch = "wasm32"))]

use super::*;
use std::time::Duration;

#[test]
fn spawn_and_poll() {
    let mut p = spawn(|| {
        std::thread::sleep(Duration::from_millis(20));
        21 * 2
    });
    let start = std::time::Instant::now();
    let v = loop {
        if let Some(v) = p.poll() {
            break v;
        }
        assert!(start.elapsed() < Duration::from_secs(5), "timed out");
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(v, 42);
    assert!(p.is_done());
    assert_eq!(p.poll(), None, "poll yields the value only once");
}

#[test]
fn wait_future_chains_jobs() {
    let a = spawn(|| {
        std::thread::sleep(Duration::from_millis(20));
        20
    });
    let b = spawn_async(async move { a.wait().await.map(|v| v + 1) });
    assert_eq!(b.block(), Some(Some(21)));
    // Abandoned producer -> None.
    let (c, p) = pending_pair::<u8>();
    drop(c);
    assert_eq!(pollster::block_on(p.wait()), None);
}

#[test]
fn spawn_job_progress_and_cancel() {
    let task = spawn_job(|ctx| {
        ctx.progress(0.25, "quarter");
        let start = std::time::Instant::now();
        while !ctx.cancelled() {
            assert!(start.elapsed() < Duration::from_secs(5), "cancel not seen");
            std::thread::sleep(Duration::from_millis(1));
        }
        "stopped"
    });
    let start = std::time::Instant::now();
    while task.control.progress().is_none() {
        assert!(start.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(task.control.progress(), Some((0.25, "quarter".to_string())));
    task.control.cancel();
    assert!(task.control.is_cancelled());
    assert_eq!(task.pending.block(), Some("stopped"));
}

#[test]
fn buffers_convert() {
    let f = vec![1.0f32, -2.5, 3.25];
    let bytes = Buffer::F32(f.clone()).into_bytes();
    assert_eq!(bytes.len(), 12);
    assert_eq!(Buffer::Bytes(bytes).into_f32(), f);
    let shared = std::sync::Arc::new(f.clone());
    assert_eq!(Buffer::SharedF32(shared.clone()).into_f32(), f);
}

#[test]
fn spawn_async_block_and_ready() {
    let p = spawn_async(async { "hello".to_string() });
    assert_eq!(p.block().as_deref(), Some("hello"));
    let mut r = Pending::ready(5u8);
    assert!(r.is_done());
    assert_eq!(r.poll(), Some(5));
}

#[test]
fn abandoned_on_panic() {
    let p: Pending<u32> = spawn(|| panic!("intentional test panic"));
    assert_eq!(p.block(), None);
    let (c, p2) = pending_pair::<u32>();
    drop(c);
    assert!(p2.is_abandoned());
}

#[test]
fn now_seconds_monotonic() {
    let a = now_seconds();
    std::thread::sleep(Duration::from_millis(5));
    let b = now_seconds();
    assert!(b > a, "{b} > {a}");
    assert!(b - a >= 0.004);
}

#[test]
fn storage_roundtrip() {
    let key = format!(
        "test-tmp/roundtrip-{}-{}",
        std::process::id(),
        now_seconds().to_bits()
    );
    let data: Vec<u8> = (0..=255u8).cycle().take(10_000).collect();
    storage_write(&key, &data).unwrap();
    assert_eq!(storage_read(&key).as_deref(), Some(&data[..]));
    storage_delete(&key);
    assert_eq!(storage_read(&key), None);

    let big: Vec<u8> = (0..1_000_000u32).map(|i| (i * 7) as u8).collect();
    storage_write_large(&key, big.clone())
        .block()
        .unwrap()
        .unwrap();
    assert_eq!(
        storage_read_large(&key).block().unwrap().as_deref(),
        Some(&big[..])
    );
    storage_delete(&key);
    assert_eq!(storage_read_large(&key).block().unwrap(), None);
    // Clean up the test directory if empty.
    if let Some(d) = storage_dir() {
        let _ = std::fs::remove_dir(d.join("test-tmp"));
    }
}

#[test]
fn storage_keys_cannot_escape() {
    let p = storage::sanitize_for_test("../../etc/pass wd");
    assert_eq!(p, vec!["etc".to_string(), "pass_wd".to_string()]);
}

#[test]
fn png_roundtrip() {
    let img = ClipboardImage {
        width: 3,
        height: 2,
        rgba: (0..24).map(|i| (i * 10) as u8).collect(),
    };
    let png = encode_png(&img).unwrap();
    assert_eq!(decode_png(&png).unwrap(), img);
}

#[test]
fn platform_strings() {
    assert!(["macOS", "Windows", "Linux"].contains(&platform_name()));
    #[cfg(target_os = "macos")]
    assert_eq!(primary_modifier_name(), "⌘");
}

#[cfg(target_os = "macos")]
#[test]
fn display_icc_profile_parses() {
    let bytes = display_icc_profile().expect("main display has an ICC profile");
    let prof = prisme_color::IccProfile::parse(&bytes).expect("parseable ICC profile");
    println!(
        "main display ICC profile: {:?} ({} bytes)",
        prof.description(),
        bytes.len()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn global_cursor_works() {
    // Works without any permission on macOS.
    assert!(global_cursor_position().is_some());
    assert!(global_mouse_down().is_some());
    println!(
        "cursor {:?}, screen capture permission: {}, pixel: {:?}",
        global_cursor_position(),
        screen_capture_permission(),
        global_cursor_position().and_then(|(x, y)| screen_pixel_at(x, y))
    );
}
