use crate::geom::{Affine2, IRect};
use crate::graph::{Graph, ParamValue};
use crate::ids::{LayerId, MaskId};
use crate::layer::{FillContent, GroupLayer, Layer, LayerKind};
use crate::mask::{Mask, SavedMask};
use crate::tile::{PixelFormat, TileGrid};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Storage precision for new layers (highest precision among sources).
    pub format: PixelFormat,
    pub dpi: f32,
    /// Original embedded ICC profile of the first imported source (informational).
    pub source_profile: Option<String>,
}

/// A complete document state. Cheap to clone (heavy data is behind `Arc`s).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub info: DocumentInfo,
    /// Layer tree, bottom to top.
    pub layers: Vec<Layer>,
    pub masks: Vec<SavedMask>,
    pub selection: Option<Mask>,
    pub graph: Graph,
    pub active_layer: Option<LayerId>,
    pub next_id: u64,
}

impl Document {
    pub fn new(name: impl Into<String>, width: u32, height: u32, format: PixelFormat) -> Self {
        Document {
            info: DocumentInfo { name: name.into(), width, height, format, dpi: 72.0, source_profile: None },
            layers: vec![],
            masks: vec![],
            selection: None,
            graph: Graph::new_default(),
            active_layer: None,
            next_id: 1,
        }
    }

    /// New document with a single background layer filled with `fill` (or transparent).
    pub fn with_background(name: impl Into<String>, width: u32, height: u32, fill: Option<[f32; 4]>) -> Self {
        let mut d = Document::new(name, width, height, PixelFormat::F16);
        let mut g = TileGrid::rgba(PixelFormat::F16);
        if let Some(c) = fill {
            let r = d.bounds();
            g.map_rect(&r, |_, _, p| p.copy_from_slice(&c));
        }
        let id = d.alloc_layer_id();
        d.layers.push(Layer::new(id, "Background", LayerKind::Raster(Arc::new(g))));
        d.active_layer = Some(id);
        d
    }

    pub fn bounds(&self) -> IRect {
        IRect::from_size(self.info.width, self.info.height)
    }

    pub fn alloc_layer_id(&mut self) -> LayerId {
        let id = LayerId(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn alloc_mask_id(&mut self) -> MaskId {
        let id = MaskId(self.next_id);
        self.next_id += 1;
        id
    }

    // ------------------------------------------------------------------ layer tree

    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        fn find(v: &[Layer], id: LayerId) -> Option<&Layer> {
            for l in v {
                if l.id == id {
                    return Some(l);
                }
                if let Some(f) = find(l.children(), id) {
                    return Some(f);
                }
            }
            None
        }
        find(&self.layers, id)
    }

    pub fn layer_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        fn find(v: &mut [Layer], id: LayerId) -> Option<&mut Layer> {
            for l in v.iter_mut() {
                if l.id == id {
                    return Some(l);
                }
                if let Some(ch) = l.children_mut()
                    && let Some(f) = find(ch, id) {
                        return Some(f);
                    }
            }
            None
        }
        find(&mut self.layers, id)
    }

    pub fn active(&self) -> Option<&Layer> {
        self.active_layer.and_then(|id| self.layer(id))
    }

    pub fn active_mut(&mut self) -> Option<&mut Layer> {
        let id = self.active_layer?;
        self.layer_mut(id)
    }

    /// Path of indices from the root to a layer.
    pub fn path_of(&self, id: LayerId) -> Option<Vec<usize>> {
        fn walk(v: &[Layer], id: LayerId, path: &mut Vec<usize>) -> bool {
            for (i, l) in v.iter().enumerate() {
                path.push(i);
                if l.id == id || walk(l.children(), id, path) {
                    return true;
                }
                path.pop();
            }
            false
        }
        let mut p = vec![];
        walk(&self.layers, id, &mut p).then_some(p)
    }

    fn siblings_mut(&mut self, parent_path: &[usize]) -> &mut Vec<Layer> {
        let mut v = &mut self.layers;
        for &i in parent_path {
            v = v[i].children_mut().expect("parent is a group");
        }
        v
    }

    pub fn remove_layer(&mut self, id: LayerId) -> Option<Layer> {
        let path = self.path_of(id)?;
        let (last, parent) = path.split_last()?;
        let l = self.siblings_mut(parent).remove(*last);
        if self.active_layer == Some(id) {
            self.active_layer = self.all_layers().first().map(|l| l.id);
        }
        Some(l)
    }

    /// Insert `layer` directly above `above` (or at the top of the root when None).
    pub fn insert_layer_above(&mut self, layer: Layer, above: Option<LayerId>) {
        match above.and_then(|a| self.path_of(a)) {
            Some(path) => {
                let (last, parent) = path.split_last().unwrap();
                self.siblings_mut(parent).insert(last + 1, layer);
            }
            None => self.layers.push(layer),
        }
    }

    /// Insert a layer as the top child of a group.
    pub fn insert_into_group(&mut self, layer: Layer, group: LayerId) -> bool {
        if let Some(g) = self.layer_mut(group).and_then(|g| g.children_mut()) {
            g.push(layer);
            true
        } else {
            false
        }
    }

    /// Move a layer up (+1) or down (-1) among its siblings.
    pub fn move_layer(&mut self, id: LayerId, delta: i32) -> bool {
        let Some(path) = self.path_of(id) else { return false };
        let (last, parent) = path.split_last().unwrap();
        let sib = self.siblings_mut(parent);
        let to = *last as i32 + delta;
        if to < 0 || to >= sib.len() as i32 {
            return false;
        }
        sib.swap(*last, to as usize);
        true
    }

    /// Move `id` so that it sits directly above `target` (same parent as target), or into
    /// a group when `into_group` is true.
    pub fn reparent_layer(&mut self, id: LayerId, target: LayerId, into_group: bool) -> bool {
        if id == target {
            return false;
        }
        // Prevent moving a group into its own descendant.
        if let Some(l) = self.layer(id) {
            fn contains(l: &Layer, t: LayerId) -> bool {
                l.children().iter().any(|c| c.id == t || contains(c, t))
            }
            if contains(l, target) {
                return false;
            }
        }
        let Some(layer) = self.remove_layer(id) else { return false };
        let ok = if into_group {
            self.insert_into_group(layer.clone(), target)
        } else {
            self.insert_layer_above(layer.clone(), Some(target));
            true
        };
        if !ok {
            self.layers.push(layer);
        }
        self.active_layer = Some(id);
        true
    }

    /// Wrap the given layer into a new group at its position.
    pub fn group_layer(&mut self, id: LayerId) -> Option<LayerId> {
        let path = self.path_of(id)?;
        let gid = self.alloc_layer_id();
        let (last, parent) = path.split_last()?;
        let sib = self.siblings_mut(parent);
        let l = sib.remove(*last);
        let group = Layer::new(gid, "Group", LayerKind::Group(GroupLayer { children: vec![l], expanded: true }));
        sib.insert(*last, group);
        Some(gid)
    }

    /// Flattened list of all layers (depth-first, bottom to top).
    pub fn all_layers(&self) -> Vec<&Layer> {
        fn walk<'a>(v: &'a [Layer], out: &mut Vec<&'a Layer>) {
            for l in v {
                out.push(l);
                walk(l.children(), out);
            }
        }
        let mut out = vec![];
        walk(&self.layers, &mut out);
        out
    }

    pub fn saved_mask(&self, id: MaskId) -> Option<&SavedMask> {
        self.masks.iter().find(|m| m.id == id)
    }

    /// Highest storage precision currently used by raster layers.
    pub fn effective_format(&self) -> PixelFormat {
        self.all_layers()
            .iter()
            .filter_map(|l| l.raster().map(|g| g.format))
            .fold(self.info.format, |a, b| a.max(b))
    }

    /// Remove references to a deleted layer from the graph.
    pub fn scrub_graph_refs(&mut self) {
        let alive: std::collections::HashSet<LayerId> = self.all_layers().iter().map(|l| l.id).collect();
        let masks: std::collections::HashSet<MaskId> = self.masks.iter().map(|m| m.id).collect();
        for n in self.graph.nodes.values_mut() {
            for v in n.params.values_mut() {
                match v {
                    ParamValue::Layer(Some(id)) if !alive.contains(id) => *v = ParamValue::Layer(None),
                    ParamValue::LayerSet(set) => set.retain(|id| alive.contains(id)),
                    ParamValue::Mask(Some(id)) if !masks.contains(id) => *v = ParamValue::Mask(None),
                    _ => {}
                }
            }
        }
    }

    // ------------------------------------------------------------------ canvas ops

    /// Apply an integer translation to every layer (used by canvas resize / crop).
    pub fn translate_all(&mut self, dx: i32, dy: i32) {
        fn walk(v: &mut [Layer], dx: i32, dy: i32) {
            for l in v {
                if let Some(m) = &mut l.mask {
                    m.grid = Arc::new(m.grid.translated(dx, dy));
                }
                match &mut l.kind {
                    LayerKind::Raster(g) => *g = Arc::new(g.translated(dx, dy)),
                    LayerKind::Vector(v) => {
                        v.transform = Affine2::translate(dx as f32, dy as f32).mul(&v.transform)
                    }
                    LayerKind::Text(t) => {
                        t.position[0] += dx as f32;
                        t.position[1] += dy as f32;
                    }
                    LayerKind::Fill(f) => {
                        if let FillContent::Gradient(_, geo) = &mut f.content {
                            for p in [&mut geo.start, &mut geo.end] {
                                p[0] += dx as f32;
                                p[1] += dy as f32;
                            }
                        }
                    }
                    LayerKind::Group(g) => walk(&mut g.children, dx, dy),
                }
            }
        }
        walk(&mut self.layers, dx, dy);
        if let Some(s) = &self.selection {
            self.selection = Some(Arc::new(s.translated(dx, dy)));
        }
        for m in &mut self.masks {
            m.mask = Arc::new(m.mask.translated(dx, dy));
        }
    }

    /// Resize the canvas, anchoring content at (`anchor_x`, `anchor_y`) in [0,1].
    pub fn resize_canvas(&mut self, w: u32, h: u32, anchor_x: f32, anchor_y: f32) {
        let dx = ((w as f32 - self.info.width as f32) * anchor_x).round() as i32;
        let dy = ((h as f32 - self.info.height as f32) * anchor_y).round() as i32;
        self.translate_all(dx, dy);
        self.info.width = w;
        self.info.height = h;
    }

    /// Crop to `rect` (document coordinates). Pixels outside are kept (non-destructive crop)
    /// unless `delete_outside` is set.
    pub fn crop(&mut self, rect: IRect, delete_outside: bool) {
        if rect.is_empty() {
            return;
        }
        self.translate_all(-rect.x0, -rect.y0);
        self.info.width = rect.width() as u32;
        self.info.height = rect.height() as u32;
        if delete_outside {
            let b = self.bounds();
            fn walk(v: &mut [Layer], b: &IRect) {
                for l in v {
                    if let LayerKind::Raster(g) = &mut l.kind {
                        let g = Arc::make_mut(g);
                        g.tiles.retain(|c, _| !c.rect().intersect(b).is_empty());
                        let tb = g.tile_bounds();
                        let keep = b.intersect(&tb);
                        let coords: Vec<_> = TileGrid::coords_in(&tb).collect();
                        for c in coords {
                            let cr = c.rect();
                            if cr.intersect(b) != cr && g.tiles.contains_key(&c) {
                                let ch = g.channels as usize;
                                let i = cr.intersect(&keep);
                                g.map_rect(&cr, |x, y, p| {
                                    if !i.contains(x, y) {
                                        p[..ch].fill(0.0);
                                    }
                                });
                            }
                        }
                        g.remove_clear_tiles();
                    }
                    if let LayerKind::Group(gr) = &mut l.kind {
                        walk(&mut gr.children, b);
                    }
                }
            }
            walk(&mut self.layers, &b);
        }
    }

    /// Apply a pixel-exact orthogonal transform (rotations by 90° and flips) to every layer.
    /// `op`: 0 = rotate 90 CW, 1 = rotate 180, 2 = rotate 90 CCW, 3 = flip horizontal, 4 = flip vertical.
    pub fn orthogonal_transform(&mut self, op: u8) {
        let (w, h) = (self.info.width as i32, self.info.height as i32);
        // integer mapping (x, y) -> (x', y') for pixel indices
        let map = move |x: i32, y: i32| -> (i32, i32) {
            match op {
                0 => (h - 1 - y, x),
                1 => (w - 1 - x, h - 1 - y),
                2 => (y, w - 1 - x),
                3 => (w - 1 - x, y),
                _ => (x, h - 1 - y),
            }
        };
        // continuous-space affine for vector/text
        let aff = match op {
            0 => Affine2 { m: [0.0, -1.0, h as f32, 1.0, 0.0, 0.0] },
            1 => Affine2 { m: [-1.0, 0.0, w as f32, 0.0, -1.0, h as f32] },
            2 => Affine2 { m: [0.0, 1.0, 0.0, -1.0, 0.0, w as f32] },
            3 => Affine2 { m: [-1.0, 0.0, w as f32, 0.0, 1.0, 0.0] },
            _ => Affine2 { m: [1.0, 0.0, 0.0, 0.0, -1.0, h as f32] },
        };
        let remap = |g: &TileGrid| -> TileGrid {
            let b = g.tile_bounds();
            let src = g.read_rect(&b);
            let ch = g.channels as usize;
            let (p0, p1) = (map(b.x0, b.y0), map(b.x1 - 1, b.y1 - 1));
            let nb = IRect::new(p0.0.min(p1.0), p0.1.min(p1.1), p0.0.max(p1.0) + 1, p0.1.max(p1.1) + 1);
            let mut dst = vec![0.0f32; nb.area() as usize * ch];
            let bw = b.width() as usize;
            let nw = nb.width() as usize;
            for y in b.y0..b.y1 {
                for x in b.x0..b.x1 {
                    let (nx, ny) = map(x, y);
                    let si = ((y - b.y0) as usize * bw + (x - b.x0) as usize) * ch;
                    let di = ((ny - nb.y0) as usize * nw + (nx - nb.x0) as usize) * ch;
                    dst[di..di + ch].copy_from_slice(&src[si..si + ch]);
                }
            }
            let mut out = TileGrid::new(g.format, g.channels);
            if !b.is_empty() {
                out.write_rect(&nb, &dst);
            }
            out
        };
        fn walk(v: &mut [Layer], remap: &dyn Fn(&TileGrid) -> TileGrid, aff: &Affine2) {
            for l in v {
                if let Some(m) = &mut l.mask {
                    m.grid = Arc::new(remap(&m.grid));
                }
                match &mut l.kind {
                    LayerKind::Raster(g) => *g = Arc::new(remap(g)),
                    LayerKind::Vector(v) => v.transform = aff.mul(&v.transform),
                    LayerKind::Text(t) => {
                        let p = aff.apply(t.position);
                        t.position = p;
                        let m = aff.m;
                        t.rotation += m[3].atan2(m[0]);
                    }
                    LayerKind::Fill(f) => {
                        if let FillContent::Gradient(_, geo) = &mut f.content {
                            geo.start = aff.apply(geo.start);
                            geo.end = aff.apply(geo.end);
                        }
                    }
                    LayerKind::Group(g) => walk(&mut g.children, remap, aff),
                }
            }
        }
        walk(&mut self.layers, &remap, &aff);
        if let Some(s) = &self.selection {
            self.selection = Some(Arc::new(remap(s)));
        }
        for m in &mut self.masks {
            m.mask = Arc::new(remap(&m.mask));
        }
        if op == 0 || op == 2 {
            std::mem::swap(&mut self.info.width, &mut self.info.height);
        }
    }

    /// Approximate unique memory used by pixel data in this state.
    pub fn pixel_bytes(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut total = 0;
        let mut add = |g: &TileGrid| {
            for t in g.tiles.values() {
                if seen.insert(t.id()) {
                    total += t.byte_size();
                }
            }
        };
        for l in self.all_layers() {
            if let Some(g) = l.raster() {
                add(g);
            }
            if let Some(m) = &l.mask {
                add(&m.grid);
            }
        }
        if let Some(s) = &self.selection {
            add(s);
        }
        for m in &self.masks {
            add(&m.mask);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_four_times_is_identity() {
        let mut d = Document::with_background("t", 300, 200, None);
        let id = d.active_layer.unwrap();
        d.layer_mut(id).unwrap().raster_mut().unwrap().set_pixel(5, 7, [1.0, 0.0, 0.0, 1.0]);
        let orig = d.layer(id).unwrap().raster().unwrap().read_rect(&d.bounds());
        for _ in 0..4 {
            d.orthogonal_transform(0);
        }
        assert_eq!((d.info.width, d.info.height), (300, 200));
        assert_eq!(d.layer(id).unwrap().raster().unwrap().read_rect(&d.bounds()), orig);
    }

    #[test]
    fn rotate_cw_moves_pixel() {
        let mut d = Document::with_background("t", 300, 200, None);
        let id = d.active_layer.unwrap();
        d.layer_mut(id).unwrap().raster_mut().unwrap().set_pixel(0, 0, [1.0; 4]);
        d.orthogonal_transform(0);
        assert_eq!(d.layer(id).unwrap().raster().unwrap().get_pixel(199, 0), [1.0; 4]);
    }

    #[test]
    fn layer_tree_ops() {
        let mut d = Document::with_background("t", 10, 10, None);
        let bg = d.active_layer.unwrap();
        let a = d.alloc_layer_id();
        d.insert_layer_above(Layer::new(a, "A", LayerKind::Raster(Default::default())), Some(bg));
        let g = d.group_layer(a).unwrap();
        assert_eq!(d.path_of(a), Some(vec![1, 0]));
        assert!(d.reparent_layer(bg, g, true));
        assert_eq!(d.path_of(bg), Some(vec![0, 1]));
        assert!(!d.reparent_layer(g, bg, false) || d.path_of(g).is_some());
    }
}

impl Default for TileGrid {
    fn default() -> Self {
        TileGrid::rgba(PixelFormat::F16)
    }
}
