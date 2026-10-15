// Brush stroke kernels (appended to kernels.wgsl: same bindings and helpers).
//
// A stroke keeps level-0 textures over a growing tile-aligned rect: the target's snapshot
// (`base`, straight, f32), the accumulated coverage (paint) and the live result (`live`).
// Dab maths mirror prisme-brush (`Dab::coverage`, `StrokeCanvas`) and prisme-tools
// (`RetouchCanvas`) exactly; single-channel targets (masks) use `.r`.
// Dab encoding (4 vec4): [cx, cy, radius, angle], [roundness, hardness, alpha, tip],
// [pixel, grain, seed, 0], [x0, y0, x1, y1] integer bounds.

struct Dab {
    c: vec2<f32>,
    r: f32,
    angle: f32,
    roundness: f32,
    hardness: f32,
    alpha: f32,
    tip: i32, // 0 round, 1 square, 2 chalk, 3 charcoal, 4 splatter
    pixel: bool,
    grain: f32,
    seed: u32,
};

fn dab_from(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>) -> Dab {
    return Dab(a.xy, a.z, a.w, b.x, b.y, b.z, i32(b.w), c.x > 0.5, c.y, u32(c.z));
}

fn dab_hash(x: i32, y: i32, seed: u32) -> f32 {
    var h = (bitcast<u32>(x) * 0x8da6b343u) ^ (bitcast<u32>(y) * 0xd8163841u) ^ (seed * 0xcb1ab31fu);
    h = h ^ (h >> 13u);
    h = h * 0x5bd1e995u;
    h = h ^ (h >> 15u);
    return f32(h & 0xffffu) / 65535.0;
}

// Smooth value noise anchored to document pixels (paper texture).
fn dab_noise(x: f32, y: f32, scale: f32, seed: u32) -> f32 {
    let fx = x / scale;
    let fy = y / scale;
    let ix = i32(floor(fx));
    let iy = i32(floor(fy));
    let tx = fx - f32(ix);
    let ty = fy - f32(iy);
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let a = dab_hash(ix, iy, seed);
    let b = dab_hash(ix + 1, iy, seed);
    let c = dab_hash(ix, iy + 1, seed);
    let d = dab_hash(ix + 1, iy + 1, seed);
    return (a + (b - a) * sx) * (1.0 - sy) + (c + (d - c) * sx) * sy;
}

// Coverage in [0,1] of the pixel centred at (px, py) (document px).
fn dab_cov(d: Dab, px: f32, py: f32) -> f32 {
    let dx = px - d.c.x;
    let dy = py - d.c.y;
    let sn = sin(d.angle);
    let cs = cos(d.angle);
    let lx = dx * cs + dy * sn;
    let ly = (-dx * sn + dy * cs) / max(d.roundness, 0.05);
    let r = max(d.r, 0.5);
    if (d.pixel) {
        var hit: bool;
        if (d.tip == 1) { hit = abs(lx) <= r && abs(ly) <= r; } else { hit = lx * lx + ly * ly <= r * r + 0.25; }
        return select(0.0, 1.0, hit);
    }
    var dist: f32;
    if (d.tip == 1) { dist = max(abs(lx), abs(ly)); } else { dist = sqrt(lx * lx + ly * ly); }
    // Soft falloff from `hardness * r` to `r`, at least ~1px wide for anti-aliasing.
    let feather = max((1.0 - d.hardness) * r, 0.75);
    let inner = r - feather;
    var cov: f32;
    if (dist <= inner) {
        cov = 1.0;
    } else if (dist >= r + 0.5) {
        cov = 0.0;
    } else {
        let t = clamp((dist - inner) / (r + 0.5 - inner), 0.0, 1.0);
        if (d.hardness < 0.5) { cov = exp(-(t * t) * 4.0) * (1.0 - t); } else { cov = 1.0 - t * t * (3.0 - 2.0 * t); }
    }
    if (d.tip == 2) {
        let g = dab_noise(px, py, 2.2, 7u) * 0.6 + dab_noise(px, py, 6.0, 3u) * 0.4;
        let thr = d.grain * (1.0 - cov * d.alpha * 0.5);
        cov = cov * clamp((g - thr * 0.8) / 0.25, 0.0, 1.0);
    } else if (d.tip == 3) {
        let streak = dab_noise(lx * 0.25 + px * 0.02, ly * 3.0, 1.5, 11u);
        let g = dab_noise(px, py, 1.6, 5u);
        cov = cov * clamp((streak * 0.5 + g * 0.5 - d.grain * 0.55) / 0.3, 0.0, 1.0);
    } else if (d.tip == 4) {
        // Droplets: cells of a jittered grid inside the dab.
        let cell = max(r * 0.18, 1.5);
        let gx = floor(lx / cell);
        let gy = floor(ly / cell);
        let h = dab_hash(i32(gx), i32(gy), d.seed);
        if (h < 0.72) { return 0.0; }
        let ox = (gx + 0.5 + (dab_hash(i32(gx) + 3, i32(gy), d.seed) - 0.5) * 0.6) * cell;
        let oy = (gy + 0.5 + (dab_hash(i32(gx), i32(gy) + 5, d.seed) - 0.5) * 0.6) * cell;
        let dr = sqrt((lx - ox) * (lx - ox) + (ly - oy) * (ly - oy));
        let rr = cell * 0.5 * (0.4 + h * 0.6);
        cov = select(clamp(rr - dr + 0.5, 0.0, 1.0), 0.0, dist > r);
    }
    return cov;
}

// Accumulate a batch of dabs into the coverage, in order: cov += (1 - cov) * dab.
// in0 = coverage, lut = dabs, P0.x = dab count.
@compute @workgroup_size(8, 8)
fn k_stroke_dabs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    var c = ld0(p).r;
    let n = u32(pf(0u));
    let pc = vec2<f32>(p) + 0.5;
    for (var i = 0u; i < n; i = i + 1u) {
        let b = vec4<i32>(lut[i * 4u + 3u]);
        if (p.x < b.x || p.y < b.y || p.x >= b.z || p.y >= b.w) { continue; }
        let d = dab_from(lut[i * 4u], lut[i * 4u + 1u], lut[i * 4u + 2u]);
        let k = dab_cov(d, pc.x, pc.y) * d.alpha;
        if (k > 0.0) { c = c + (1.0 - c) * k; }
    }
    textureStore(out_tex, out_coord(gid), vec4<f32>(c, 0.0, 0.0, 1.0));
}

// Composite the stroke onto its base snapshot (opacity caps the stroke).
// in0 = base, in1 = coverage, in2 = selection (optional).
// P0 = colour (straight), P1 = [0 colour / 1 erase / 2 value, opacity, lock alpha, channels], P2.x = value.
@compute @workgroup_size(8, 8)
fn k_stroke_paint(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let b = ld0(p);
    let pa = pv(1u);
    var a = ld1(p).r * pa.y;
    if (info.connected.z != 0u) { a = a * ld2(p).r; }
    let one = pa.w < 1.5;
    var o = b;
    if (pa.x < 0.5) {
        let col = pv(0u);
        let ac = a * col.a;
        if (one) {
            o.r = b.r + (1.0 - b.r) * ac;
        } else if (pa.z > 0.5) {
            o = vec4<f32>(b.rgb + (col.rgb - b.rgb) * ac, b.a);
        } else {
            let ao = ac + b.a * (1.0 - ac);
            if (ao > 1e-7) { o = vec4<f32>((col.rgb * ac + b.rgb * b.a * (1.0 - ac)) / ao, ao); } else { o.a = ao; }
        }
    } else if (pa.x < 1.5) {
        if (one) { o.r = b.r * (1.0 - a); } else { o.a = b.a * (1.0 - a); }
    } else {
        o.r = b.r + (pf(8u) - b.r) * a;
    }
    textureStore(out_tex, out_coord(gid), o);
}

// ---- Retouch (one dab per dispatch; P0..P3 = the dab, P4 = [mode 0 smudge / 1 blur /
// 2 clone / 3 heal, strength, source offset], P5 = [blur radius, first dab]).

fn rt_cov(p: vec2<i32>) -> f32 {
    let d = dab_from(pv(0u), pv(1u), pv(2u));
    var c = dab_cov(d, f32(p.x) + 0.5, f32(p.y) + 0.5) * d.alpha;
    if (info.connected.z != 0u) { c = c * ld2(p).r; }
    return c;
}

fn rt_mean(acc: vec4<f32>, w: f32) -> vec4<f32> {
    if (acc.a > 1e-6) { return vec4<f32>(acc.rgb / acc.a, acc.a / max(w, 1e-6)); }
    return vec4<f32>(0.0);
}

var<workgroup> rt_acc0: array<vec4<f32>, 64>;
var<workgroup> rt_acc1: array<vec4<f32>, 64>;
var<workgroup> rt_w: array<f32, 64>;

// Coverage-weighted means under the dab (single workgroup, output 2x1).
// Clone/heal: texel 0 = layer (in0), texel 1 = source (in1 at the offset).
// Smudge: texel 0 = colour carried into this dab, texel 1 = carried to the next (in3 = previous output).
@compute @workgroup_size(8, 8)
fn k_retouch_mean(@builtin(local_invocation_index) li: u32) {
    let b = vec4<i32>(pv(3u));
    let w = max(b.z - b.x, 1);
    let n = max(b.z - b.x, 0) * max(b.w - b.y, 0);
    let off = vec2<i32>(pv(4u).zw);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var ws = 0.0;
    // Uniform trip count so the barriers below stay in uniform control flow (WebGPU).
    for (var j = 0; j < (n + 63) / 64; j = j + 1) {
        let i = j * 64 + i32(li);
        if (i < n) {
            let p = b.xy + vec2<i32>(i % w, i / w);
            let c = rt_cov(p);
            if (c > 0.0) {
                let d = ld0(p);
                a0 = a0 + vec4<f32>(d.rgb * d.a * c, d.a * c);
                let s = ld1(p + off);
                a1 = a1 + vec4<f32>(s.rgb * s.a * c, s.a * c);
                ws = ws + c;
            }
        }
    }
    rt_acc0[li] = a0;
    rt_acc1[li] = a1;
    rt_w[li] = ws;
    workgroupBarrier();
    for (var k = 32u; k > 0u; k = k >> 1u) {
        if (li < k) {
            rt_acc0[li] = rt_acc0[li] + rt_acc0[li + k];
            rt_acc1[li] = rt_acc1[li] + rt_acc1[li + k];
            rt_w[li] = rt_w[li] + rt_w[li + k];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let m0 = rt_mean(rt_acc0[0], rt_w[0]);
        let m1 = rt_mean(rt_acc1[0], rt_w[0]);
        let m = pv(4u);
        if (m.x < 0.5) {
            var used = m0;
            if (pv(5u).y < 0.5) { used = ld3(vec2<i32>(1, 0)); }
            textureStore(out_tex, vec2<i32>(0, 0), used);
            textureStore(out_tex, vec2<i32>(1, 0), used + (m0 - used) * (1.0 - m.y));
        } else {
            textureStore(out_tex, vec2<i32>(0, 0), m0);
            textureStore(out_tex, vec2<i32>(1, 0), m1);
        }
    }
}

// Apply one retouch dab over its bounds. in0 = layer, in1 = clone source, in2 = selection
// (optional), in3 = k_retouch_mean output.
@compute @workgroup_size(8, 8)
fn k_retouch_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    var o = ld0(p);
    let m = pv(4u);
    let c = rt_cov(p);
    if (m.x < 0.5) {
        let k = c * m.y;
        if (k > 0.0) { o = o + (ld3(vec2<i32>(0, 0)) - o) * k; }
    } else if (m.x < 1.5) {
        let k = c * m.y;
        if (k > 0.0) {
            let rad = i32(pv(5u).x);
            var acc = vec4<f32>(0.0);
            var n = 0.0;
            for (var oy = -rad; oy <= rad; oy = oy + 1) {
                for (var ox = -rad; ox <= rad; ox = ox + 1) {
                    let s = ld0(p + vec2<i32>(ox, oy));
                    acc = acc + vec4<f32>(s.rgb * s.a, s.a);
                    n = n + 1.0;
                }
            }
            var bl = vec4<f32>(0.0);
            if (acc.a > 1e-6) { bl = vec4<f32>(acc.rgb / acc.a, acc.a / n); }
            o = o + (bl - o) * k;
        }
    } else if (c > 0.0) {
        let s = ld1(p + vec2<i32>(m.zw));
        var shift = vec3<f32>(0.0);
        // Heal: match low frequencies (mean(dest) - mean(src) inside the dab).
        if (m.x > 2.5) { shift = ld3(vec2<i32>(0, 0)).rgb - ld3(vec2<i32>(1, 0)).rgb; }
        o = o + (vec4<f32>(s.rgb + shift, s.a) - o) * c;
    }
    textureStore(out_tex, out_coord(gid), o);
}

// Layer content with the live stroke spliced in, at this level. in0 = regular upload
// (optional), in1 = live (level 0). Blocks fully inside the live rect are averaged from it
// (alpha weighted like `Tile::downsample`). P0 = [2^mip, channels, default value].
@compute @workgroup_size(8, 8)
fn k_stroke_view(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let s = i32(pf(0u));
    let q0 = p * s - info.in1.xy;
    var o = vec4<f32>(pf(2u));
    if (q0.x >= 0 && q0.y >= 0 && q0.x + s <= info.in1.z && q0.y + s <= info.in1.w) {
        if (s == 1) {
            o = textureLoad(in1, q0, 0);
        } else {
            var acc = vec4<f32>(0.0);
            for (var y = 0; y < s; y = y + 1) {
                for (var x = 0; x < s; x = x + 1) {
                    let c = textureLoad(in1, q0 + vec2<i32>(x, y), 0);
                    acc = acc + vec4<f32>(c.rgb * c.a, c.a);
                }
            }
            let n = f32(s * s);
            if (pf(1u) < 1.5) {
                // Single channel: plain average of .r (rgb * a with a == 1 for r32float loads).
                o = vec4<f32>(acc.r / n, 0.0, 0.0, 1.0);
            } else if (acc.a > 0.0) {
                o = vec4<f32>(acc.rgb / acc.a, acc.a / n);
            } else {
                o = vec4<f32>(0.0);
            }
        }
    } else {
        let q = p - info.in0.xy;
        if (info.connected.x != 0u && inside(q, info.in0.zw)) { o = textureLoad(in0, q, 0); }
    }
    textureStore(out_tex, out_coord(gid), o);
}

// ---- Liquify: the layer snapshot warped through a displacement field (backward map:
// output p samples base at p + d(p)). Mirrors prisme_tools::liquify::warp_rect at level 0.
// in0 = regular upload at this level (optional), in1 = base (level 0, straight, covers the
// working area), in2 = field nodes (dx, dy, freeze, 0), in3 = selection (level 0, optional).
// P0 = [2^mip, node step, tint freeze, bicubic], P1 = working rect (document px), P2.x = default.

fn lq_node(i: vec2<i32>) -> vec4<f32> {
    return textureLoad(in2, clamp(i, vec2<i32>(0), info.in2.zw - 1), 0);
}

// Bilinear field value at a document position (edge nodes extend outwards).
fn lq_field(pc: vec2<f32>) -> vec4<f32> {
    let u = (pc - pv(1u).xy) / pf(1u) - 0.5;
    let f = floor(u);
    let t = u - f;
    let i = vec2<i32>(f);
    let a = mix(lq_node(i), lq_node(i + vec2<i32>(1, 0)), t.x);
    let b = mix(lq_node(i + vec2<i32>(0, 1)), lq_node(i + vec2<i32>(1, 1)), t.x);
    return mix(a, b, t.y);
}

// Premultiplied base pixel, clamped to the working area (edge extend).
fn lq_px(q: vec2<i32>) -> vec4<f32> {
    let fr = vec4<i32>(pv(1u));
    let r = clamp(q, fr.xy, fr.zw - 1) - info.in1.xy;
    if (!inside(r, info.in1.zw)) { return vec4<f32>(0.0); }
    let v = textureLoad(in1, r, 0);
    return vec4<f32>(v.rgb * v.a, v.a);
}

fn lq_cubic(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

// Premultiplied sample (pixel centres at +0.5); Catmull-Rom clamped to the 4 nearest.
fn lq_sample(s: vec2<f32>, bicubic: bool) -> vec4<f32> {
    let f = s - 0.5;
    let fl = floor(f);
    let t = f - fl;
    let i = vec2<i32>(fl);
    let n0 = lq_px(i);
    let n1 = lq_px(i + vec2<i32>(1, 0));
    let n2 = lq_px(i + vec2<i32>(0, 1));
    let n3 = lq_px(i + vec2<i32>(1, 1));
    if (!bicubic) { return mix(mix(n0, n1, t.x), mix(n2, n3, t.x), t.y); }
    let wx = lq_cubic(t.x);
    let wy = lq_cubic(t.y);
    var acc = vec4<f32>(0.0);
    for (var y = 0; y < 4; y = y + 1) {
        for (var x = 0; x < 4; x = x + 1) {
            acc = acc + lq_px(i + vec2<i32>(x - 1, y - 1)) * (wx[x] * wy[y]);
        }
    }
    return clamp(acc, min(min(n0, n1), min(n2, n3)), max(max(n0, n1), max(n2, n3)));
}

// Premultiplied warped value at document position pc.
fn lq_warp(pc: vec2<f32>, bicubic: bool) -> vec4<f32> {
    let fr = pv(1u);
    let d = lq_field(pc);
    let s = clamp(pc + d.xy, fr.xy + 0.5, fr.zw - 0.5);
    var v = lq_sample(s, bicubic);
    if (info.connected.w != 0u) {
        let q = vec2<i32>(floor(pc));
        v = mix(lq_px(q), v, clamp(ld3(q).r, 0.0, 1.0));
    }
    if (pf(2u) > 0.5 && d.z > 0.0) {
        let k = 0.45 * clamp(d.z, 0.0, 1.0);
        v = v * (1.0 - k) + vec4<f32>(0.85, 0.12, 0.15, 1.0) * k;
    }
    return v;
}

// Layer content at this level (straight), warped inside the working area.
@compute @workgroup_size(8, 8)
fn k_liquify(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let k = pf(0u);
    let fr = pv(1u);
    let c = (vec2<f32>(p) + 0.5) * k;
    var o = vec4<f32>(pf(8u));
    if (c.x >= fr.x && c.y >= fr.y && c.x < fr.z && c.y < fr.w) {
        var v: vec4<f32>;
        if (k < 1.5) {
            v = lq_warp(c, pf(3u) > 0.5);
        } else {
            // 2x2 samples over the level pixel's footprint.
            let q = vec2<f32>(p) * k;
            v = (lq_warp(q + vec2<f32>(0.25, 0.25) * k, false) + lq_warp(q + vec2<f32>(0.75, 0.25) * k, false)
                + lq_warp(q + vec2<f32>(0.25, 0.75) * k, false) + lq_warp(q + vec2<f32>(0.75, 0.75) * k, false)) * 0.25;
        }
        if (v.a > 1e-6) { o = vec4<f32>(v.rgb / v.a, min(v.a, 1.0)); } else { o = vec4<f32>(0.0); }
    } else {
        let q = p - info.in0.xy;
        if (info.connected.x != 0u && inside(q, info.in0.zw)) { o = textureLoad(in0, q, 0); }
    }
    textureStore(out_tex, out_coord(gid), o);
}

// ---- Free Transform preview: a snapshot drawn through a projective map (backward: output
// p samples the source at H⁻¹(p)), bilinear like prisme_tools::transform::warp.
// in0 = regular upload at this level (what stays behind), in1 = source (level 0, straight;
// a single-channel mask in mask mode). P0 = [2^mip, mask mode, default, 0],
// P1..P3 = H⁻¹ row-major (document -> source px), P4 = transformed bounds (document px).
// Content mode draws the source over in0; mask mode replaces the mask, `default` outside.

fn xf_px(q: vec2<i32>, mask: bool) -> vec4<f32> {
    let r = q - info.in1.xy;
    if (!inside(r, info.in1.zw)) { return vec4<f32>(select(0.0, pf(2u), mask)); }
    let v = textureLoad(in1, r, 0);
    if (mask) { return vec4<f32>(v.r); }
    return vec4<f32>(v.rgb * v.a, v.a);
}

// Premultiplied source value (mask: value in every channel) at document position c.
fn xf_sample(c: vec2<f32>, mask: bool) -> vec4<f32> {
    let a = pv(1u);
    let b = pv(2u);
    let w = b.z * c.x + b.w * c.y + pf(12u);
    let s = vec2<f32>(a.x * c.x + a.y * c.y + a.z, a.w * c.x + b.x * c.y + b.y) / w;
    if (abs(w) < 1e-12 || abs(s.x) > 1e8 || abs(s.y) > 1e8) { return vec4<f32>(select(0.0, pf(2u), mask)); }
    let f = s - 0.5;
    let fl = floor(f);
    let t = f - fl;
    let i = vec2<i32>(fl);
    let n0 = xf_px(i, mask);
    let n1 = xf_px(i + vec2<i32>(1, 0), mask);
    let n2 = xf_px(i + vec2<i32>(0, 1), mask);
    let n3 = xf_px(i + vec2<i32>(1, 1), mask);
    return mix(mix(n0, n1, t.x), mix(n2, n3, t.x), t.y);
}

@compute @workgroup_size(8, 8)
fn k_xform(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let k = pf(0u);
    let mask = pf(1u) > 0.5;
    let bd = pv(4u);
    let lo = vec2<f32>(p) * k;
    let hi = lo + k;
    let hit = hi.x > bd.x && hi.y > bd.y && lo.x < bd.z && lo.y < bd.w;
    var v = vec4<f32>(select(0.0, pf(2u), mask));
    if (hit) {
        if (k < 1.5) {
            v = xf_sample(lo + 0.5 * k, mask);
        } else {
            // 2x2 samples over the level pixel's footprint.
            v = (xf_sample(lo + vec2<f32>(0.25, 0.25) * k, mask) + xf_sample(lo + vec2<f32>(0.75, 0.25) * k, mask)
                + xf_sample(lo + vec2<f32>(0.25, 0.75) * k, mask) + xf_sample(lo + vec2<f32>(0.75, 0.75) * k, mask)) * 0.25;
        }
    }
    if (mask) {
        textureStore(out_tex, out_coord(gid), vec4<f32>(v.r));
        return;
    }
    var o = vec4<f32>(0.0);
    let q = p - info.in0.xy;
    if (info.connected.x != 0u && inside(q, info.in0.zw)) { o = textureLoad(in0, q, 0); }
    let r = v + vec4<f32>(o.rgb * o.a, o.a) * (1.0 - v.a);
    if (r.a > 1e-6) { o = vec4<f32>(r.rgb / r.a, min(r.a, 1.0)); } else { o = vec4<f32>(0.0); }
    textureStore(out_tex, out_coord(gid), o);
}
