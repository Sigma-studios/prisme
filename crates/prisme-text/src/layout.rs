//! Text shaping and line layout (explicit line breaks only; text layers have no box).

use crate::fonts::{FontDb, ResolvedFace};
use prisme_core::layer::{TextAlign, TextLayer};
use rustybuzz::ttf_parser;

/// One positioned glyph. Coordinates in text space: document pixels relative to the
/// top-left of the text block, y down, `y` = baseline.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlacedGlyph {
    pub face: fontdb::ID,
    pub gid: u16,
    pub x: f32,
    pub y: f32,
    /// Font units → document pixels.
    pub scale: f32,
    pub wght: Option<f32>,
    pub synth_bold: bool,
    pub synth_italic: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Layout {
    pub glyphs: Vec<PlacedGlyph>,
    /// Block size in document pixels.
    pub width: f32,
    pub height: f32,
    /// Per char index (plus one end position): `[x, y_top, y_bottom]` in text space.
    pub carets: Vec<[f32; 3]>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct LayoutKey {
    pub text: String,
    pub family: String,
    pub weight: u16,
    pub italic: bool,
    pub size: u32,
    pub line_height: u32,
    pub letter_spacing: u32,
    pub align: u8,
    pub generation: u64,
}

impl LayoutKey {
    pub fn of(l: &TextLayer, generation: u64) -> Self {
        Self {
            text: l.text.clone(),
            family: l.font_family.clone(),
            weight: l.weight,
            italic: l.italic,
            size: l.size.to_bits(),
            line_height: l.line_height.to_bits(),
            letter_spacing: l.letter_spacing.to_bits(),
            align: l.align as u8,
            generation,
        }
    }
}

struct ShapedLine {
    glyphs: Vec<PlacedGlyph>,
    /// (byte offset of cluster start within the line, x start, x end)
    clusters: Vec<(usize, f32, f32)>,
    width: f32,
}

/// Shape one line (left-aligned at x = 0, baseline at y = 0).
fn shape_line(
    fonts: &FontDb,
    line: &str,
    primary: ResolvedFace,
    size: f32,
    spacing: f32,
    italic: bool,
) -> ShapedLine {
    // Split into runs by covering face (primary first, then fallbacks).
    let primary_cover: Vec<bool> = fonts
        .with_face(primary.id, |f| {
            line.chars().map(|c| f.glyph_index(c).is_some()).collect()
        })
        .unwrap_or_else(|| vec![false; line.chars().count()]);
    let mut runs: Vec<(usize, usize, ResolvedFace)> = Vec::new();
    for (k, (i, c)) in line.char_indices().enumerate() {
        let face = if c.is_whitespace() || c.is_control() {
            runs.last().map(|r| r.2).unwrap_or(primary)
        } else if primary_cover[k] {
            primary
        } else if let Some(id) = fonts.fallback_for(c, primary.id) {
            ResolvedFace {
                id,
                wght: None,
                synth_bold: primary.synth_bold,
                synth_italic: italic,
            }
        } else {
            primary
        };
        let end = i + c.len_utf8();
        match runs.last_mut() {
            Some(r) if r.2 == face && r.1 == i => r.1 = end,
            _ => runs.push((i, end, face)),
        }
    }

    let mut out = ShapedLine {
        glyphs: Vec::new(),
        clusters: Vec::new(),
        width: 0.0,
    };
    let mut pen = 0.0f32;
    for (start, end, rf) in runs {
        let run = &line[start..end];
        let shaped = fonts.with_face(rf.id, |face| {
            if let Some(w) = rf.wght {
                face.set_variations(&[rustybuzz::Variation {
                    tag: ttf_parser::Tag::from_bytes(b"wght"),
                    value: w,
                }]);
            }
            let upem = face.units_per_em() as f32;
            let mut buf = rustybuzz::UnicodeBuffer::new();
            buf.push_str(run);
            buf.guess_segment_properties();
            let gb = rustybuzz::shape(face, &[], buf);
            let infos: Vec<(u32, u32)> = gb
                .glyph_infos()
                .iter()
                .map(|g| (g.glyph_id, g.cluster))
                .collect();
            let pos: Vec<(i32, i32, i32)> = gb
                .glyph_positions()
                .iter()
                .map(|p| (p.x_advance, p.x_offset, p.y_offset))
                .collect();
            (upem, infos, pos)
        });
        let Some((upem, infos, pos)) = shaped else {
            continue;
        };
        let scale = size / upem.max(1.0);
        let n = infos.len();
        for k in 0..n {
            let (gid, cluster) = infos[k];
            let (adv, xo, yo) = pos[k];
            out.glyphs.push(PlacedGlyph {
                face: rf.id,
                gid: gid as u16,
                x: pen + xo as f32 * scale,
                y: -(yo as f32) * scale,
                scale,
                wght: rf.wght,
                synth_bold: rf.synth_bold,
                synth_italic: rf.synth_italic,
            });
            let x0 = pen;
            pen += adv as f32 * scale;
            let last_of_cluster = k + 1 == n || infos[k + 1].1 != cluster;
            if last_of_cluster && adv != 0 {
                pen += spacing;
            }
            let cl = start + cluster as usize;
            match out.clusters.last_mut() {
                Some(c) if c.0 == cl => c.2 = pen,
                _ => out.clusters.push((cl, x0, pen)),
            }
        }
    }
    // Trailing letter spacing does not count towards the line width.
    out.width = if out.glyphs.is_empty() {
        0.0
    } else {
        (pen - spacing).max(0.0)
    };
    out
}

pub(crate) fn layout(fonts: &FontDb, l: &TextLayer) -> Layout {
    let size = l.size.max(0.01);
    let Some(primary) = fonts.resolve(&l.font_family, l.weight, l.italic) else {
        return Layout::default();
    };
    let (asc, desc) = fonts
        .with_face(primary.id, |f| {
            let upem = f.units_per_em() as f32;
            (f.ascender() as f32 / upem, -(f.descender() as f32) / upem)
        })
        .unwrap_or((0.8, 0.2));
    let line_adv = (l.line_height.max(0.0) * size).max(1e-3);
    let content = (asc + desc) * size;
    let base_off = (line_adv - content) * 0.5 + asc * size;

    let text = l.text.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = text.split('\n').collect();
    let shaped: Vec<ShapedLine> = lines
        .iter()
        .map(|s| shape_line(fonts, s, primary, size, l.letter_spacing, l.italic))
        .collect();
    let width = shaped.iter().map(|s| s.width).fold(0.0f32, f32::max);
    let height = line_adv * lines.len() as f32;

    let mut out = Layout {
        width,
        height,
        ..Default::default()
    };
    for (li, (line, sl)) in lines.iter().zip(&shaped).enumerate() {
        let top = li as f32 * line_adv;
        let dx = match l.align {
            TextAlign::Left => 0.0,
            TextAlign::Center => (width - sl.width) * 0.5,
            TextAlign::Right => width - sl.width,
        };
        let baseline = top + base_off;
        for g in &sl.glyphs {
            out.glyphs.push(PlacedGlyph {
                x: g.x + dx,
                y: g.y + baseline,
                ..*g
            });
        }
        // Caret positions for each char of the line, then the line end (the '\n' slot).
        let chars: Vec<usize> = line.char_indices().map(|(i, _)| i).collect();
        for (ci, &b) in chars.iter().enumerate() {
            let x = match sl.clusters.iter().rposition(|c| c.0 <= b) {
                Some(k) => {
                    let (cs, x0, x1) = sl.clusters[k];
                    let ce = sl
                        .clusters
                        .get(k + 1)
                        .map(|c| c.0)
                        .unwrap_or(line.len())
                        .max(cs + 1);
                    // Chars inside a multi-char cluster (ligatures) split the advance.
                    let n_in = chars.iter().filter(|&&o| o >= cs && o < ce).count().max(1);
                    let idx = chars[..ci].iter().filter(|&&o| o >= cs).count();
                    x0 + (x1 - x0) * idx as f32 / n_in as f32
                }
                None => 0.0,
            };
            out.carets.push([x + dx, top, top + line_adv]);
        }
        out.carets.push([sl.width + dx, top, top + line_adv]);
    }
    out
}
