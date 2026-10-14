//! Font database: bundled fallback fonts, optional system fonts (native only) and
//! user-provided font files. Internally synchronized (`Send + Sync`).

use fontdb::{Database, Family, ID, Query, Source, Style};
use rustybuzz::ttf_parser;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub(crate) type FontData = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// Bundled family used when nothing else matches (always present, also on the web).
pub const BUNDLED_SANS: &str = "Ubuntu";
/// Bundled monospace family.
pub const BUNDLED_MONO: &str = "Hack";

struct Inner {
    db: Database,
    sans: String,
    serif: String,
    mono: String,
}

/// Resolved face for a layer's family/weight/style request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ResolvedFace {
    pub id: ID,
    /// Value for the `wght` variation axis, when the face is variable.
    pub wght: Option<f32>,
    pub synth_bold: bool,
    pub synth_italic: bool,
}

/// Font collection shared by every text renderer.
pub struct FontDb {
    inner: RwLock<Inner>,
    data: Mutex<HashMap<ID, (FontData, u32)>>,
    system_loaded: AtomicBool,
    loading: AtomicBool,
    generation: AtomicU64,
    warned: Mutex<HashSet<String>>,
    fallback: Mutex<HashMap<(u64, char), Option<ID>>>,
}

fn bundled() -> [&'static [u8]; 3] {
    [
        epaint_default_fonts::UBUNTU_LIGHT,
        epaint_default_fonts::HACK_REGULAR,
        epaint_default_fonts::NOTO_EMOJI_REGULAR,
    ]
}

#[cfg(not(target_arch = "wasm32"))]
/// Family names tried for the generic `sans-serif` (first present wins).
const SANS_CANDIDATES: &[&str] = &[
    "Helvetica Neue",
    "Helvetica",
    "Segoe UI",
    "Arial",
    "Inter",
    "Noto Sans",
    "DejaVu Sans",
    "Liberation Sans",
    "Cantarell",
    BUNDLED_SANS,
];
#[cfg(not(target_arch = "wasm32"))]
const SERIF_CANDIDATES: &[&str] = &[
    "Times New Roman",
    "Times",
    "Georgia",
    "Noto Serif",
    "DejaVu Serif",
    "Liberation Serif",
    BUNDLED_SANS,
];
#[cfg(not(target_arch = "wasm32"))]
const MONO_CANDIDATES: &[&str] = &[
    "Menlo",
    "SF Mono",
    "Consolas",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Noto Sans Mono",
    BUNDLED_MONO,
];
/// Broad-coverage families searched when a character is missing from the chosen font.
const FALLBACK_FAMILIES: &[&str] = &[
    "Arial Unicode MS",
    "Helvetica Neue",
    "Segoe UI",
    "Segoe UI Symbol",
    "Noto Sans",
    "DejaVu Sans",
    "PingFang SC",
    "Hiragino Sans",
    "Apple SD Gothic Neo",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Noto Sans Arabic",
    "Noto Sans Hebrew",
    "Noto Sans Devanagari",
    "Geeza Pro",
    "Arial Hebrew",
    "Kohinoor Devanagari",
    "Apple Symbols",
    "Noto Emoji",
];

fn find_family<'a>(db: &'a Database, name: &str) -> Option<&'a str> {
    db.faces().find_map(|f| {
        f.families
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(n, _)| n.as_str())
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn pick(db: &Database, candidates: &[&str]) -> String {
    candidates
        .iter()
        .find_map(|c| find_family(db, c))
        .unwrap_or(BUNDLED_SANS)
        .to_string()
}

impl FontDb {
    /// Instant: contains only the bundled fallback fonts (Ubuntu Light, Hack, Noto Emoji).
    /// Call [`FontDb::load_system_fonts_async`] to enumerate system fonts in the background.
    pub fn new() -> Arc<Self> {
        let mut db = Database::new();
        for b in bundled() {
            db.load_font_source(Source::Binary(Arc::new(b)));
        }
        Arc::new(Self {
            inner: RwLock::new(Inner {
                db,
                sans: BUNDLED_SANS.into(),
                serif: BUNDLED_SANS.into(),
                mono: BUNDLED_MONO.into(),
            }),
            data: Mutex::new(HashMap::new()),
            system_loaded: AtomicBool::new(false),
            loading: AtomicBool::new(false),
            generation: AtomicU64::new(1),
            warned: Mutex::new(HashSet::new()),
            fallback: Mutex::new(HashMap::new()),
        })
    }

    /// Enumerate system fonts now (blocking; native only — a no-op on wasm).
    pub fn load_system_fonts_blocking(&self) {
        if self.system_loaded.load(Ordering::Acquire) {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut sys = Database::new();
            sys.load_system_fonts();
            let infos: Vec<fontdb::FaceInfo> = sys.faces().cloned().collect();
            let mut inner = self.inner.write().unwrap();
            if !self.system_loaded.load(Ordering::Acquire) {
                for info in infos {
                    inner.db.push_face_info(info);
                }
                inner.sans = pick(&inner.db, SANS_CANDIDATES);
                inner.serif = pick(&inner.db, SERIF_CANDIDATES);
                inner.mono = pick(&inner.db, MONO_CANDIDATES);
                log::info!(
                    "loaded {} font faces; sans-serif = {}, serif = {}, monospace = {}",
                    inner.db.len(),
                    inner.sans,
                    inner.serif,
                    inner.mono
                );
                self.generation.fetch_add(1, Ordering::AcqRel);
            }
        }
        self.system_loaded.store(true, Ordering::Release);
    }

    /// Enumerate system fonts on a background thread (native). On wasm there are no
    /// system fonts: this just marks loading as complete.
    pub fn load_system_fonts_async(self: &Arc<Self>) {
        if self.system_loaded.load(Ordering::Acquire) || self.loading.swap(true, Ordering::AcqRel) {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let me = self.clone();
            let spawned = std::thread::Builder::new()
                .name("prisme-fonts".into())
                .spawn(move || {
                    me.load_system_fonts_blocking();
                });
            if spawned.is_err() {
                self.load_system_fonts_blocking();
            }
        }
        #[cfg(target_arch = "wasm32")]
        self.system_loaded.store(true, Ordering::Release);
    }

    /// True once system font enumeration has finished (always true after the async load
    /// on wasm, where there are no system fonts).
    pub fn system_fonts_loaded(&self) -> bool {
        self.system_loaded.load(Ordering::Acquire)
    }

    /// Counter bumped whenever the set of fonts changes (system fonts loaded, user font
    /// added). Callers caching rendered text should re-render when it changes.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Add a user-provided font file (TTF/OTF/TTC). Returns the family names it adds.
    pub fn add_font_data(&self, bytes: Vec<u8>) -> Vec<String> {
        let data: FontData = Arc::new(bytes);
        let mut inner = self.inner.write().unwrap();
        let ids = inner.db.load_font_source(Source::Binary(data));
        let mut names: Vec<String> = ids
            .iter()
            .filter_map(|id| inner.db.face(*id))
            .filter_map(|f| f.families.first().map(|(n, _)| n.clone()))
            .collect();
        names.sort();
        names.dedup();
        if !ids.is_empty() {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
        names
    }

    /// Sorted, deduplicated family names (bundled + system + user).
    pub fn families(&self) -> Vec<String> {
        let inner = self.inner.read().unwrap();
        let mut v: Vec<String> = inner
            .db
            .faces()
            .filter_map(|f| f.families.first().map(|(n, _)| n.clone()))
            .collect();
        v.sort_by_key(|s| s.to_lowercase());
        v.dedup();
        v
    }

    /// Available `(weight, italic)` styles of a family (generic names resolve first).
    pub fn styles(&self, family: &str) -> Vec<(u16, bool)> {
        let inner = self.inner.read().unwrap();
        let name = match Self::generic(&inner, family) {
            Some(n) => n.to_string(),
            None => family.to_string(),
        };
        let mut v: Vec<(u16, bool)> = inner
            .db
            .faces()
            .filter(|f| {
                f.families
                    .iter()
                    .any(|(n, _)| n.eq_ignore_ascii_case(&name))
            })
            .map(|f| (f.weight.0, f.style != Style::Normal))
            .collect();
        v.sort();
        v.dedup();
        v
    }

    fn generic<'a>(inner: &'a Inner, family: &str) -> Option<&'a str> {
        match family.trim().to_ascii_lowercase().as_str() {
            "sans-serif" | "sans" | "system-ui" | "ui-sans-serif" | "" => Some(&inner.sans),
            "serif" | "ui-serif" => Some(&inner.serif),
            "monospace" | "ui-monospace" => Some(&inner.mono),
            "cursive" | "fantasy" => Some(&inner.sans),
            _ => None,
        }
    }

    /// Resolve a family/weight/style request to a concrete face.
    pub(crate) fn resolve(&self, family: &str, weight: u16, italic: bool) -> Option<ResolvedFace> {
        let inner = self.inner.read().unwrap();
        let name: String = match Self::generic(&inner, family) {
            Some(n) => n.to_string(),
            None => match find_family(&inner.db, family) {
                Some(n) => n.to_string(),
                None => {
                    let mut w = self.warned.lock().unwrap();
                    if w.insert(family.to_string()) {
                        log::warn!("font family '{family}' not found; using '{}'", inner.sans);
                    }
                    inner.sans.clone()
                }
            },
        };
        let style = if italic { Style::Italic } else { Style::Normal };
        let families = [
            Family::Name(&name),
            Family::Name(&inner.sans),
            Family::Name(BUNDLED_SANS),
        ];
        let q = Query {
            families: &families,
            weight: fontdb::Weight(weight),
            style,
            ..Default::default()
        };
        let id = inner.db.query(&q)?;
        let info = inner.db.face(id)?;
        let (face_weight, face_style) = (info.weight.0, info.style);
        drop(inner);
        let wght = self.with_face(id, |f| {
            let axis = f
                .variation_axes()
                .into_iter()
                .find(|a| a.tag == ttf_parser::Tag::from_bytes(b"wght"))?;
            Some((weight as f32).clamp(axis.min_value, axis.max_value))
        })?;
        Some(ResolvedFace {
            id,
            wght,
            synth_bold: wght.is_none() && weight >= 600 && face_weight + 150 <= weight,
            synth_italic: italic && face_style == Style::Normal,
        })
    }

    #[allow(clippy::infallible_destructuring_match)] // only `Source::Binary` exists on wasm
    pub(crate) fn face_data(&self, id: ID) -> Option<(FontData, u32)> {
        if let Some(d) = self.data.lock().unwrap().get(&id) {
            return Some(d.clone());
        }
        let (src, index) = self.inner.read().unwrap().db.face_source(id)?;
        let data: FontData = match src {
            Source::Binary(d) => d,
            #[cfg(not(target_arch = "wasm32"))]
            Source::File(path) => Arc::new(std::fs::read(&path).ok()?),
            #[cfg(not(target_arch = "wasm32"))]
            Source::SharedFile(_, d) => d,
        };
        self.data.lock().unwrap().insert(id, (data.clone(), index));
        Some((data, index))
    }

    /// Run `f` with a parsed face.
    pub(crate) fn with_face<T>(
        &self,
        id: ID,
        f: impl FnOnce(&mut rustybuzz::Face<'_>) -> T,
    ) -> Option<T> {
        let (data, index) = self.face_data(id)?;
        let bytes: &[u8] = (*data).as_ref();
        let mut face = rustybuzz::Face::from_slice(bytes, index)?;
        Some(f(&mut face))
    }

    /// Face covering `c` when the primary face lacks it (cached per generation).
    #[allow(irrefutable_let_patterns)] // only `Source::Binary` exists on wasm
    pub(crate) fn fallback_for(&self, c: char, primary: ID) -> Option<ID> {
        let gen_ = self.generation();
        if let Some(r) = self.fallback.lock().unwrap().get(&(gen_, c)) {
            return *r;
        }
        let candidates: Vec<ID> = {
            let inner = self.inner.read().unwrap();
            let mut fams: Vec<String> = vec![inner.sans.clone()];
            fams.extend(FALLBACK_FAMILIES.iter().map(|s| s.to_string()));
            fams.push(BUNDLED_SANS.into());
            fams.push(BUNDLED_MONO.into());
            let mut ids: Vec<ID> = Vec::new();
            for fam in &fams {
                let Some(name) = find_family(&inner.db, fam) else {
                    continue;
                };
                let families = [Family::Name(name)];
                if let Some(id) = inner.db.query(&Query {
                    families: &families,
                    ..Default::default()
                }) {
                    ids.push(id);
                }
            }
            // User-provided fonts (binary sources not part of the bundle) are also candidates.
            for f in inner.db.faces() {
                if let Source::Binary(d) = &f.source {
                    let p = (**d).as_ref().as_ptr();
                    if !bundled().iter().any(|b| b.as_ptr() == p) {
                        ids.push(f.id);
                    }
                }
            }
            ids
        };
        let found = candidates
            .into_iter()
            .filter(|id| *id != primary)
            .find(|id| {
                self.with_face(*id, |f| f.glyph_index(c).is_some())
                    .unwrap_or(false)
            });
        self.fallback.lock().unwrap().insert((gen_, c), found);
        found
    }
}
