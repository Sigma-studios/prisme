//! Node graph data model. Node *kinds* (sockets, params, kernels) live in
//! `prisme-graph` / `prisme-gpu`; here we only store instances and links.

use crate::gradient::Gradient;
use crate::ids::{LayerId, MaskId, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const KIND_OUTPUT: &str = "output";
pub const KIND_LAYER_STACK: &str = "layer_stack";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    /// Linear working RGBA.
    Color([f32; 4]),
    Choice(u32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    /// Monotone curve control points in [0,1]².
    Curve(Vec<[f32; 2]>),
    Gradient(Gradient),
    Layer(Option<LayerId>),
    /// Layers excluded from a stack.
    LayerSet(Vec<LayerId>),
    Mask(Option<MaskId>),
    Text(String),
}

impl ParamValue {
    pub fn as_f32(&self) -> f32 {
        match self {
            ParamValue::Float(v) => *v,
            ParamValue::Int(v) => *v as f32,
            ParamValue::Bool(b) => *b as i32 as f32,
            ParamValue::Choice(c) => *c as f32,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: String,
    pub params: BTreeMap<String, ParamValue>,
    pub pos: [f32; 2],
    /// Bypassed nodes pass their first image input through unchanged.
    pub bypass: bool,
    /// UI: inline parameters folded away in the node editor.
    #[serde(default)]
    pub collapsed: bool,
}

impl Node {
    pub fn f(&self, key: &str, default: f32) -> f32 {
        self.params.get(key).map(|p| p.as_f32()).unwrap_or(default)
    }

    pub fn b(&self, key: &str, default: bool) -> bool {
        match self.params.get(key) {
            Some(ParamValue::Bool(b)) => *b,
            _ => default,
        }
    }

    pub fn choice(&self, key: &str, default: u32) -> u32 {
        match self.params.get(key) {
            Some(ParamValue::Choice(c)) => *c,
            Some(ParamValue::Int(i)) => *i as u32,
            _ => default,
        }
    }

    pub fn color(&self, key: &str, default: [f32; 4]) -> [f32; 4] {
        match self.params.get(key) {
            Some(ParamValue::Color(c)) => *c,
            _ => default,
        }
    }

    pub fn vec2(&self, key: &str, default: [f32; 2]) -> [f32; 2] {
        match self.params.get(key) {
            Some(ParamValue::Vec2(c)) => *c,
            _ => default,
        }
    }

    pub fn vec3(&self, key: &str, default: [f32; 3]) -> [f32; 3] {
        match self.params.get(key) {
            Some(ParamValue::Vec3(c)) => *c,
            _ => default,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Link {
    pub from: NodeId,
    pub from_out: u16,
    pub to: NodeId,
    pub to_in: u16,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: BTreeMap<NodeId, Node>,
    pub links: Vec<Link>,
    pub output: NodeId,
    pub next_id: u64,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new_default()
    }
}

impl Graph {
    /// `Layer Stack -> Output`.
    pub fn new_default() -> Self {
        let mut g = Graph { nodes: BTreeMap::new(), links: vec![], output: NodeId(0), next_id: 1 };
        let out = g.add_node(KIND_OUTPUT, BTreeMap::new(), [420.0, 0.0]);
        let stack = g.add_node(KIND_LAYER_STACK, BTreeMap::new(), [0.0, 0.0]);
        g.output = out;
        g.links.push(Link { from: stack, from_out: 0, to: out, to_in: 0 });
        g
    }

    pub fn add_node(&mut self, kind: &str, params: BTreeMap<String, ParamValue>, pos: [f32; 2]) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        self.nodes.insert(id, Node { id, kind: kind.to_string(), params, pos, bypass: false, collapsed: false });
        id
    }

    /// Remove a node and its links. The output node cannot be removed.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        if id == self.output {
            return false;
        }
        self.links.retain(|l| l.from != id && l.to != id);
        self.nodes.remove(&id).is_some()
    }

    pub fn input(&self, node: NodeId, input: u16) -> Option<(NodeId, u16)> {
        self.links.iter().find(|l| l.to == node && l.to_in == input).map(|l| (l.from, l.from_out))
    }

    pub fn outputs_of(&self, node: NodeId) -> impl Iterator<Item = &Link> {
        self.links.iter().filter(move |l| l.from == node)
    }

    /// True if `a` is reachable upstream from `b` (i.e. b depends on a).
    pub fn depends_on(&self, b: NodeId, a: NodeId) -> bool {
        let mut stack = vec![b];
        let mut seen = std::collections::HashSet::new();
        while let Some(n) = stack.pop() {
            if n == a {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            for l in self.links.iter().filter(|l| l.to == n) {
                stack.push(l.from);
            }
        }
        false
    }

    /// Connect an output to an input, replacing any existing link into that input.
    /// Fails if it would create a cycle.
    pub fn connect(&mut self, from: NodeId, from_out: u16, to: NodeId, to_in: u16) -> bool {
        if from == to || self.depends_on(from, to) {
            return false;
        }
        self.links.retain(|l| !(l.to == to && l.to_in == to_in));
        self.links.push(Link { from, from_out, to, to_in });
        true
    }

    pub fn disconnect_input(&mut self, to: NodeId, to_in: u16) {
        self.links.retain(|l| !(l.to == to && l.to_in == to_in));
    }

    /// Nodes in dependency order reachable from the output.
    pub fn topo_from_output(&self) -> Vec<NodeId> {
        let mut order = vec![];
        let mut state = std::collections::HashMap::new();
        fn visit(g: &Graph, n: NodeId, state: &mut std::collections::HashMap<NodeId, u8>, order: &mut Vec<NodeId>) {
            match state.get(&n) {
                Some(1) | Some(2) => return,
                _ => {}
            }
            state.insert(n, 1);
            let mut ins: Vec<_> = g.links.iter().filter(|l| l.to == n).collect();
            ins.sort_by_key(|l| l.to_in);
            for l in ins {
                visit(g, l.from, state, order);
            }
            state.insert(n, 2);
            order.push(n);
        }
        visit(self, self.output, &mut state, &mut order);
        order
    }

    /// True when the graph is exactly the default `Layer Stack -> Output` with no exclusions,
    /// letting renderers skip graph evaluation.
    pub fn is_trivial(&self) -> bool {
        match self.input(self.output, 0) {
            Some((n, 0)) => {
                let node = &self.nodes[&n];
                node.kind == KIND_LAYER_STACK
                    && !node.bypass
                    && match node.params.get("exclude") {
                        None => true,
                        Some(ParamValue::LayerSet(v)) => v.is_empty(),
                        _ => true,
                    }
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cycles() {
        let mut g = Graph::new_default();
        let a = g.add_node("invert", BTreeMap::new(), [0.0; 2]);
        let b = g.add_node("invert", BTreeMap::new(), [0.0; 2]);
        assert!(g.connect(a, 0, b, 0));
        assert!(!g.connect(b, 0, a, 0));
        assert!(g.connect(b, 0, g.output, 0));
        assert_eq!(g.topo_from_output(), vec![a, b, g.output]);
    }
}
