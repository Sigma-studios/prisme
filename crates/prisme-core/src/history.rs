//! Linear undo history of structurally shared document snapshots.

use crate::document::Document;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub label: String,
    pub doc: Document,
    /// Consecutive pushes with the same merge key collapse into one entry (e.g. slider drags).
    #[serde(skip)]
    pub merge_key: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct History {
    entries: Vec<HistoryEntry>,
    cursor: usize,
    /// Pixel memory budget for history in bytes; oldest states are dropped beyond it.
    pub budget_bytes: usize,
    /// Monotonic counter bumped on every change (render caches can key on it).
    #[serde(skip)]
    revision: u64,
}

impl History {
    pub fn new(doc: Document) -> Self {
        History {
            entries: vec![HistoryEntry { label: "Open".into(), doc, merge_key: None }],
            cursor: 0,
            budget_bytes: 2 << 30,
            revision: 0,
        }
    }

    pub fn current(&self) -> &Document {
        &self.entries[self.cursor].doc
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn entries(&self) -> &[HistoryEntry] {
        &self.entries
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Commit a new state, discarding any redo states.
    pub fn push(&mut self, label: impl Into<String>, doc: Document) {
        self.push_merge(label, doc, None);
    }

    pub fn push_merge(&mut self, label: impl Into<String>, doc: Document, merge_key: Option<u64>) {
        self.entries.truncate(self.cursor + 1);
        let label = label.into();
        if let (Some(k), Some(top)) = (merge_key, self.entries.last_mut())
            && top.merge_key == Some(k) && self.cursor > 0 {
                top.doc = doc;
                top.label = label;
                self.revision += 1;
                return;
            }
        self.entries.push(HistoryEntry { label, doc, merge_key });
        self.cursor = self.entries.len() - 1;
        self.revision += 1;
        self.enforce_budget();
    }

    /// Replace the current state without creating a history entry (e.g. selecting a layer).
    pub fn replace_current(&mut self, doc: Document) {
        self.entries[self.cursor].doc = doc;
        self.revision += 1;
    }

    /// Stop merging into the top entry (call when a drag ends).
    pub fn seal(&mut self) {
        if let Some(top) = self.entries.last_mut() {
            top.merge_key = None;
        }
    }

    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }

    pub fn can_redo(&self) -> bool {
        self.cursor + 1 < self.entries.len()
    }

    pub fn undo(&mut self) -> bool {
        if self.can_undo() {
            self.cursor -= 1;
            self.seal_all();
            self.revision += 1;
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        if self.can_redo() {
            self.cursor += 1;
            self.revision += 1;
            true
        } else {
            false
        }
    }

    pub fn jump(&mut self, index: usize) {
        if index < self.entries.len() && index != self.cursor {
            self.cursor = index;
            self.seal_all();
            self.revision += 1;
        }
    }

    fn seal_all(&mut self) {
        for e in &mut self.entries {
            e.merge_key = None;
        }
    }

    /// Unique pixel bytes referenced by all entries.
    pub fn memory_bytes(&self) -> usize {
        let mut seen = HashSet::new();
        let mut total = 0;
        for e in &self.entries {
            let mut add = |g: &crate::tile::TileGrid| {
                for t in g.tiles.values() {
                    if seen.insert(t.id()) {
                        total += t.byte_size();
                    }
                }
            };
            for l in e.doc.all_layers() {
                if let Some(g) = l.raster() {
                    add(g);
                }
                if let Some(m) = &l.mask {
                    add(&m.grid);
                }
            }
            if let Some(s) = &e.doc.selection {
                add(s);
            }
            for m in &e.doc.masks {
                add(&m.mask);
            }
        }
        total
    }

    fn enforce_budget(&mut self) {
        // Cheap early-out: only measure when there are many entries.
        while self.entries.len() > 2 && self.cursor > 0 && self.memory_bytes() > self.budget_bytes {
            self.entries.remove(0);
            self.cursor -= 1;
        }
        const MAX_ENTRIES: usize = 500;
        while self.entries.len() > MAX_ENTRIES && self.cursor > 0 {
            self.entries.remove(0);
            self.cursor -= 1;
        }
    }

    /// Restore from deserialized parts (file load).
    pub fn from_parts(entries: Vec<HistoryEntry>, cursor: usize) -> Self {
        let cursor = cursor.min(entries.len().saturating_sub(1));
        History { entries, cursor, budget_bytes: 2 << 30, revision: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::TileCoord;
    use rand::{Rng, SeedableRng};

    /// Random edit sequences followed by undo-all must restore the original pixels exactly,
    /// and redo-all must restore the final state.
    #[test]
    fn undo_all_restores_original() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..20 {
            let doc = Document::with_background("t", 600, 400, Some([0.2, 0.3, 0.4, 1.0]));
            let id = doc.active_layer.unwrap();
            let snapshot = |d: &Document| d.layer(id).unwrap().raster().unwrap().read_rect(&d.bounds());
            let original = snapshot(&doc);
            let mut h = History::new(doc);
            let n = rng.random_range(1..30);
            for i in 0..n {
                let mut d = h.current().clone();
                let g = d.layer_mut(id).unwrap().raster_mut().unwrap();
                for _ in 0..50 {
                    let (x, y) = (rng.random_range(0..600), rng.random_range(0..400));
                    g.set_pixel(x, y, [rng.random(), rng.random(), rng.random(), 1.0]);
                }
                if rng.random_bool(0.2) {
                    d.orthogonal_transform(1);
                }
                h.push(format!("edit {i}"), d);
            }
            let fin = snapshot(h.current());
            while h.undo() {}
            assert_eq!(snapshot(h.current()), original);
            while h.redo() {}
            assert_eq!(snapshot(h.current()), fin);
        }
    }

    #[test]
    fn merge_collapses() {
        let doc = Document::with_background("t", 10, 10, None);
        let mut h = History::new(doc.clone());
        h.push_merge("a", doc.clone(), Some(1));
        h.push_merge("a", doc.clone(), Some(1));
        assert_eq!(h.entries().len(), 2);
        h.seal();
        h.push_merge("a", doc, Some(1));
        assert_eq!(h.entries().len(), 3);
    }

    #[test]
    fn snapshots_share_tiles() {
        let doc = Document::with_background("t", 2048, 2048, Some([1.0; 4]));
        let id = doc.active_layer.unwrap();
        let mut h = History::new(doc);
        let base = h.memory_bytes();
        let mut d = h.current().clone();
        d.layer_mut(id).unwrap().raster_mut().unwrap().set_pixel(1, 1, [0.0; 4]);
        h.push("dot", d);
        let tile = h.current().layer(id).unwrap().raster().unwrap().get_tile(TileCoord::new(0, 0)).unwrap().byte_size();
        assert_eq!(h.memory_bytes(), base + tile);
    }
}
