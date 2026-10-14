//! Windows implementations (Win32 via windows-sys).

use crate::pending::{Pending, spawn};
use std::ptr::{null, null_mut};
use windows_sys::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{GetDC, GetPixel, ReleaseDC};
use windows_sys::Win32::UI::ColorSystem::GetICMProfileW;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SetFocus, VK_ESCAPE, VK_LBUTTON, VK_RBUTTON,
};
use windows_sys::Win32::UI::Shell::ShellExecuteW;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

const CLR_INVALID: COLORREF = 0xFFFF_FFFF;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn global_cursor_position() -> Option<(i32, i32)> {
    let mut p = POINT { x: 0, y: 0 };
    // SAFETY: valid out pointer.
    (unsafe { GetCursorPos(&mut p) } != 0).then_some((p.x, p.y))
}

/// State of the *primary* (logical left) mouse button, honouring swapped buttons.
pub fn global_mouse_down() -> Option<bool> {
    // SAFETY: plain Win32 queries.
    let swapped = unsafe { GetSystemMetrics(SM_SWAPBUTTON) } != 0;
    let vk = if swapped { VK_RBUTTON } else { VK_LBUTTON };
    Some(unsafe { GetAsyncKeyState(vk as i32) } as u16 & 0x8000 != 0)
}

/// Pixel at virtual-screen (physical pixel, if the process is DPI aware) coordinates.
pub fn screen_pixel_at(x: i32, y: i32) -> Option<[u8; 3]> {
    // SAFETY: screen DC acquired and released here.
    unsafe {
        let dc = GetDC(null_mut());
        if dc.is_null() {
            return None;
        }
        let c = GetPixel(dc, x, y);
        ReleaseDC(null_mut(), dc);
        if c == CLR_INVALID {
            return None;
        }
        Some([
            (c & 0xFF) as u8,
            ((c >> 8) & 0xFF) as u8,
            ((c >> 16) & 0xFF) as u8,
        ])
    }
}

pub fn display_icc_profile() -> Option<Vec<u8>> {
    // SAFETY: screen DC acquired and released here; buffer sized as requested by the API.
    let path = unsafe {
        let dc = GetDC(null_mut());
        if dc.is_null() {
            return None;
        }
        let mut len: u32 = 0;
        GetICMProfileW(dc, &mut len, null_mut());
        let mut buf = vec![0u16; len.max(260) as usize + 1];
        len = buf.len() as u32;
        let ok = GetICMProfileW(dc, &mut len, buf.as_mut_ptr());
        ReleaseDC(null_mut(), dc);
        if ok == 0 {
            return None;
        }
        let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        String::from_utf16_lossy(&buf[..end])
    };
    std::fs::read(path).ok().filter(|v| !v.is_empty())
}

pub fn open_url(url: &str) {
    let op = wide("open");
    let file = wide(url);
    // SAFETY: valid NUL-terminated wide strings.
    unsafe {
        ShellExecuteW(
            null_mut(),
            op.as_ptr(),
            file.as_ptr(),
            null(),
            null(),
            SW_SHOWNORMAL,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Screen colour picker: a nearly transparent, top-most, borderless window covering the whole
// virtual screen captures the next click (so the click does not reach the app underneath),
// with a crosshair cursor. The window is excluded from screen capture (Windows 10 2004+) so
// GetPixel sees what is underneath it; on older systems the 1/255 alpha changes the read
// value by at most one level. Esc or right click cancels.

const PICK_PENDING: isize = -1;
const PICK_CANCEL: isize = -2;

thread_local! {
    static PICK_RESULT: std::cell::Cell<isize> = const { std::cell::Cell::new(PICK_PENDING) };
}

unsafe extern "system" fn picker_wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_LBUTTONDOWN => {
            let rgb = global_cursor_position().and_then(|(x, y)| screen_pixel_at(x, y));
            let v = rgb.map_or(PICK_CANCEL, |[r, g, b]| {
                (r as isize) | ((g as isize) << 8) | ((b as isize) << 16)
            });
            PICK_RESULT.with(|c| c.set(v));
            unsafe { DestroyWindow(hwnd) };
            0
        }
        WM_RBUTTONDOWN => {
            PICK_RESULT.with(|c| c.set(PICK_CANCEL));
            unsafe { DestroyWindow(hwnd) };
            0
        }
        WM_KEYDOWN if wp == VK_ESCAPE as usize => {
            PICK_RESULT.with(|c| c.set(PICK_CANCEL));
            unsafe { DestroyWindow(hwnd) };
            0
        }
        WM_KILLFOCUS | WM_ACTIVATEAPP if msg == WM_KILLFOCUS || wp == 0 => {
            // Alt-tab / focus loss cancels, so the overlay can never get stuck.
            PICK_RESULT.with(|c| {
                if c.get() == PICK_PENDING {
                    c.set(PICK_CANCEL)
                }
            });
            unsafe { DestroyWindow(hwnd) };
            0
        }
        WM_SETCURSOR => {
            unsafe { SetCursor(LoadCursorW(null_mut(), IDC_CROSS)) };
            1
        }
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

fn run_picker() -> Option<[u8; 3]> {
    // SAFETY: standard Win32 window creation and message loop, all on this thread.
    unsafe {
        PICK_RESULT.with(|c| c.set(PICK_PENDING));
        let hinst = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(null());
        let class = wide("PrismeScreenColorPicker");
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: 0,
            lpfnWndProc: Some(picker_wndproc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: hinst,
            hIcon: null_mut(),
            hCursor: LoadCursorW(null_mut(), IDC_CROSS),
            hbrBackground: windows_sys::Win32::Graphics::Gdi::GetStockObject(
                windows_sys::Win32::Graphics::Gdi::BLACK_BRUSH,
            ),
            lpszMenuName: null(),
            lpszClassName: class.as_ptr(),
            hIconSm: null_mut(),
        };
        RegisterClassExW(&wc); // may fail if already registered: fine.

        let x = GetSystemMetrics(SM_XVIRTUALSCREEN);
        let y = GetSystemMetrics(SM_YVIRTUALSCREEN);
        let w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        let h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        let title = wide("Pick a colour");
        let hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            class.as_ptr(),
            title.as_ptr(),
            WS_POPUP,
            x,
            y,
            w,
            h,
            null_mut(),
            null_mut(),
            hinst,
            null(),
        );
        if hwnd.is_null() {
            return None;
        }
        SetLayeredWindowAttributes(hwnd, 0, 1, LWA_ALPHA);
        SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
        ShowWindow(hwnd, SW_SHOW);
        SetForegroundWindow(hwnd);
        SetFocus(hwnd);

        let mut msg: MSG = std::mem::zeroed();
        while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        let v = PICK_RESULT.with(|c| c.get());
        (v >= 0).then_some([
            (v & 0xFF) as u8,
            ((v >> 8) & 0xFF) as u8,
            ((v >> 16) & 0xFF) as u8,
        ])
    }
}

pub fn pick_screen_color() -> Pending<Option<[u8; 3]>> {
    spawn(run_picker)
}
