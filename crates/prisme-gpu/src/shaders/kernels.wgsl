// Prisme compute kernels.
//
// Conventions
// - All images are premultiplied, scene-linear, working space (linear Rec.2020).
//   Layer uploads are straight alpha and are converted by the blend/layer kernels.
// - Coordinates are *level* pixels (document pixels scaled by 2^-mip). Each input
//   carries its own origin/size so inputs with larger ROI (blur padding) line up.
// - Masks are read from the `.r` channel.
// - `OUT_FORMAT` and the OKLab matrices are substituted from Rust.

struct Info {
    out_origin: vec2<i32>,
    out_size: vec2<i32>,
    in0: vec4<i32>,
    in1: vec4<i32>,
    in2: vec4<i32>,
    in3: vec4<i32>,
    connected: vec4<u32>,
    level_scale: f32,   // level px per document px
    mip: f32,
    doc_w: f32,         // page (Fixed) / nominal frame (Infinite)
    doc_h: f32,
    infinite: f32,      // 1: no page edge
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<uniform> P: array<vec4<f32>, 16>;
@group(0) @binding(2) var out_tex: texture_storage_2d<OUT_FORMAT, write>;
@group(0) @binding(3) var in0: texture_2d<f32>;
@group(0) @binding(4) var in1: texture_2d<f32>;
@group(0) @binding(5) var in2: texture_2d<f32>;
@group(0) @binding(6) var in3: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> lut: array<vec4<f32>>;

const W2LMS: mat3x3<f32> = MAT_W2LMS;
const LMS2LAB: mat3x3<f32> = MAT_LMS2LAB;
const LAB2LMS: mat3x3<f32> = MAT_LAB2LMS;
const LMS2W: mat3x3<f32> = MAT_LMS2W;
const LUMA: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);
const PI: f32 = 3.14159265358979;

fn pf(i: u32) -> f32 { return P[i / 4u][i % 4u]; }
fn pv(i: u32) -> vec4<f32> { return P[i]; }

fn inside(q: vec2<i32>, size: vec2<i32>) -> bool {
    return q.x >= 0 && q.y >= 0 && q.x < size.x && q.y < size.y;
}
fn ld0(p: vec2<i32>) -> vec4<f32> {
    let q = p - info.in0.xy;
    if (!inside(q, info.in0.zw)) { return vec4<f32>(0.0); }
    return textureLoad(in0, q, 0);
}
fn ld1(p: vec2<i32>) -> vec4<f32> {
    let q = p - info.in1.xy;
    if (!inside(q, info.in1.zw)) { return vec4<f32>(0.0); }
    return textureLoad(in1, q, 0);
}
fn ld2(p: vec2<i32>) -> vec4<f32> {
    let q = p - info.in2.xy;
    if (!inside(q, info.in2.zw)) { return vec4<f32>(0.0); }
    return textureLoad(in2, q, 0);
}
fn ld3(p: vec2<i32>) -> vec4<f32> {
    let q = p - info.in3.xy;
    if (!inside(q, info.in3.zw)) { return vec4<f32>(0.0); }
    return textureLoad(in3, q, 0);
}
// Clamp-to-edge variant for filters (avoids dark borders at ROI edges).
fn ld0c(p: vec2<i32>) -> vec4<f32> {
    let q = clamp(p - info.in0.xy, vec2<i32>(0), max(info.in0.zw - 1, vec2<i32>(0)));
    return textureLoad(in0, q, 0);
}
fn ld1c(p: vec2<i32>) -> vec4<f32> {
    let q = clamp(p - info.in1.xy, vec2<i32>(0), max(info.in1.zw - 1, vec2<i32>(0)));
    return textureLoad(in1, q, 0);
}

// Edge handling for filters: 0 = extend the document's edge pixels (Photoshop-like),
// 1 = treat everything outside the document as transparent. Infinite canvases have no
// edge: always transparent past the content.
fn doc_edge(p: vec2<i32>) -> vec2<i32> {
    let m = vec2<i32>(ceil(vec2<f32>(info.doc_w, info.doc_h) * info.level_scale)) - 1;
    return clamp(p, vec2<i32>(0), max(m, vec2<i32>(0)));
}
fn ld0e(p: vec2<i32>, mode: f32) -> vec4<f32> {
    if (mode < 0.5 && info.infinite < 0.5) { return ld0c(doc_edge(p)); }
    return ld0(p);
}

// Optional mask on input 3 (1.0 when not connected).
fn mask3(p: vec2<i32>) -> f32 {
    if (info.connected.w == 0u) { return 1.0; }
    return clamp(ld3(p).r, 0.0, 1.0);
}

fn unpremul(c: vec4<f32>) -> vec4<f32> {
    if (c.a <= 1e-8) { return vec4<f32>(0.0); }
    return vec4<f32>(c.rgb / c.a, c.a);
}
fn premul(c: vec4<f32>) -> vec4<f32> { return vec4<f32>(c.rgb * c.a, c.a); }
fn luma(c: vec3<f32>) -> f32 { return dot(c, LUMA); }

fn cbrt(x: f32) -> f32 { return sign(x) * pow(abs(x), 1.0 / 3.0); }
fn to_oklab(c: vec3<f32>) -> vec3<f32> {
    let lms = W2LMS * c;
    return LMS2LAB * vec3<f32>(cbrt(lms.x), cbrt(lms.y), cbrt(lms.z));
}
fn from_oklab(lab: vec3<f32>) -> vec3<f32> {
    let l = LAB2LMS * lab;
    return LMS2W * (l * l * l);
}
fn srgb_oetf(x: f32) -> f32 {
    let a = abs(x);
    var e: f32;
    if (a <= 0.0031308) { e = a * 12.92; } else { e = 1.055 * pow(a, 1.0 / 2.4) - 0.055; }
    return sign(x) * e;
}
fn srgb_eotf(x: f32) -> f32 {
    let a = abs(x);
    var l: f32;
    if (a <= 0.04045) { l = a / 12.92; } else { l = pow((a + 0.055) / 1.055, 2.4); }
    return sign(x) * l;
}
fn enc3(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(srgb_oetf(c.r), srgb_oetf(c.g), srgb_oetf(c.b)); }
fn dec3(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(srgb_eotf(c.r), srgb_eotf(c.g), srgb_eotf(c.b)); }

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Document-space centre of a level pixel.
fn doc_pos(p: vec2<i32>) -> vec2<f32> { return (vec2<f32>(p) + 0.5) / info.level_scale; }

fn out_coord(gid: vec3<u32>) -> vec2<i32> { return vec2<i32>(gid.xy); }
fn in_bounds(gid: vec3<u32>) -> bool {
    return i32(gid.x) < info.out_size.x && i32(gid.y) < info.out_size.y;
}
fn gpos(gid: vec3<u32>) -> vec2<i32> { return info.out_origin + vec2<i32>(gid.xy); }

// Finish a pointwise adjustment: mix original and adjusted (straight) by the optional mask.
fn finish_adjust(gid: vec3<u32>, p: vec2<i32>, orig: vec4<f32>, adjusted_straight: vec4<f32>) {
    let m = mask3(p);
    let a = premul(adjusted_straight);
    textureStore(out_tex, out_coord(gid), mix(orig, a, m));
}

// ------------------------------------------------------------------------------ blending

fn blend_sep(mode: u32, b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 1u: { return b * s; }
        case 2u: { return b + s - b * s; }
        case 3u: { return hard_light(s, b); }
        case 4u: { return min(b, s); }
        case 5u: { return max(b, s); }
        case 6u: { return dodge(b, s); }
        case 7u: { return burn(b, s); }
        case 8u: { return hard_light(b, s); }
        case 9u: { return soft_light(b, s); }
        case 10u: { return abs(b - s); }
        case 11u: { return b + s - 2.0 * b * s; }
        case 12u: { return b + s; }
        case 13u: { return max(b - s, vec3<f32>(0.0)); }
        case 14u: { return b / max(s, vec3<f32>(1e-5)); }
        case 15u: { return max(b + s - 1.0, vec3<f32>(0.0)); }
        case 16u: { return vivid(b, s); }
        case 17u: { return clamp(b + 2.0 * s - 1.0, vec3<f32>(0.0), vec3<f32>(max(1.0, max(b.r, max(b.g, b.b))))); }
        case 18u: { return pin(b, s); }
        default: { return s; }
    }
}
fn hard_light(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    let lo = b * 2.0 * s;
    let hi = b + (2.0 * s - 1.0) - b * (2.0 * s - 1.0);
    return select(hi, lo, s <= vec3<f32>(0.5));
}
fn dodge(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    var r: vec3<f32>;
    for (var i = 0; i < 3; i++) {
        if (b[i] <= 0.0) { r[i] = 0.0; } else if (s[i] >= 1.0) { r[i] = max(1.0, b[i]); } else { r[i] = min(b[i] / (1.0 - s[i]), 64.0); }
    }
    return r;
}
fn burn(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    var r: vec3<f32>;
    for (var i = 0; i < 3; i++) {
        if (b[i] >= 1.0) { r[i] = b[i]; } else if (s[i] <= 0.0) { r[i] = 0.0; } else { r[i] = 1.0 - min(1.0, (1.0 - b[i]) / s[i]); }
    }
    return r;
}
fn soft_light(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    var r: vec3<f32>;
    for (var i = 0; i < 3; i++) {
        let bb = clamp(b[i], 0.0, 1.0);
        var d: f32;
        if (bb <= 0.25) { d = ((16.0 * bb - 12.0) * bb + 4.0) * bb; } else { d = sqrt(bb); }
        if (s[i] <= 0.5) { r[i] = bb - (1.0 - 2.0 * s[i]) * bb * (1.0 - bb); } else { r[i] = bb + (2.0 * s[i] - 1.0) * (d - bb); }
    }
    return r;
}
fn vivid(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    let lo = burn(b, 2.0 * s);
    let hi = dodge(b, 2.0 * (s - 0.5));
    return select(hi, lo, s <= vec3<f32>(0.5));
}
fn pin(b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    let lo = min(b, 2.0 * s);
    let hi = max(b, 2.0 * s - 1.0);
    return select(hi, lo, s <= vec3<f32>(0.5));
}
fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    let d = l - luma(c);
    var r = c + d;
    let ll = luma(r);
    let n = min(r.r, min(r.g, r.b));
    let x = max(r.r, max(r.g, r.b));
    if (n < 0.0) { r = ll + (r - ll) * ll / max(ll - n, 1e-6); }
    if (x > 1.0 && ll <= 1.0) { r = ll + (r - ll) * (1.0 - ll) / max(x - ll, 1e-6); }
    return r;
}
fn sat_of(c: vec3<f32>) -> f32 { return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b)); }
fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    if (mx - mn <= 1e-6) { return vec3<f32>(0.0); }
    return (c - mn) * s / (mx - mn);
}
fn blend_color(mode: u32, b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 19u: { return set_lum(set_sat(s, sat_of(b)), luma(b)); }
        case 20u: { return set_lum(set_sat(b, sat_of(s)), luma(b)); }
        case 21u: { return set_lum(s, luma(b)); }
        case 22u: { return set_lum(b, luma(s)); }
        default: { return blend_sep(mode, b, s); }
    }
}

// Composite straight-alpha source `s` (alpha already includes opacity/mask) over premultiplied backdrop.
fn composite(mode: u32, back: vec4<f32>, s: vec4<f32>) -> vec4<f32> {
    if (s.a <= 0.0) { return back; }
    let ab = back.a;
    let as_ = s.a;
    if (mode == 0u || ab <= 0.0) {
        return vec4<f32>(s.rgb * as_ + back.rgb * (1.0 - as_), as_ + ab * (1.0 - as_));
    }
    let cb = back.rgb / ab;
    let bl = blend_color(mode, cb, s.rgb);
    let co = s.rgb * as_ * (1.0 - ab) + back.rgb * (1.0 - as_) + as_ * ab * bl;
    return vec4<f32>(co, as_ + ab * (1.0 - as_));
}

// P0: x=mode, y=opacity, z=top_is_straight, w=mask_default (value outside mask texture)
// P1: x=mask_connected_on_in2(layer mask), y=mask_invert
// in0 = backdrop (premul), in1 = top, in2 = layer mask (r), in3 = extra mask (blend node)
@compute @workgroup_size(8, 8)
fn k_blend(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let back = ld0(p);
    var top = ld1(p);
    let pa = pv(0u);
    if (pa.z < 0.5) { top = unpremul(top); }
    var a = top.a * pa.y;
    if (pv(1u).x > 0.5) {
        let q = p - info.in2.xy;
        var m = pa.w;
        if (inside(q, info.in2.zw)) { m = textureLoad(in2, q, 0).r; }
        if (pv(1u).y > 0.5) { m = 1.0 - m; }
        a = a * clamp(m, 0.0, 1.0);
    }
    a = a * mask3(p);
    textureStore(out_tex, out_coord(gid), composite(u32(pa.x), back, vec4<f32>(top.rgb, a)));
}

// Layer node: straight top -> premultiplied, with optional mask (in2) and opacity.
@compute @workgroup_size(8, 8)
fn k_layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    var c = ld1(p);
    let pa = pv(0u);
    if (pa.z < 0.5) { c = unpremul(c); }
    var a = c.a * pa.y;
    if (pv(1u).x > 0.5) {
        let q = p - info.in2.xy;
        var m = pa.w;
        if (inside(q, info.in2.zw)) { m = textureLoad(in2, q, 0).r; }
        a = a * clamp(m, 0.0, 1.0);
    }
    textureStore(out_tex, out_coord(gid), vec4<f32>(c.rgb * a, a));
}

@compute @workgroup_size(8, 8)
fn k_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    textureStore(out_tex, out_coord(gid), pv(0u));
}

@compute @workgroup_size(8, 8)
fn k_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    textureStore(out_tex, out_coord(gid), ld0(gpos(gid)));
}

// Gradient image. P0.xy start, P0.zw end (document px), P1.x shape, P1.y repeat, P1.z reverse, P1.w dither.
// lut: 256 straight RGBA entries.
fn gradient_t(d: vec2<f32>) -> f32 {
    let s = pv(0u).xy;
    let e = pv(0u).zw;
    let g = e - s;
    let len2 = max(dot(g, g), 1e-6);
    let len = sqrt(len2);
    let v = d - s;
    let shape = u32(pv(1u).x);
    var t: f32;
    switch shape {
        case 1u: { t = length(v) / len; }
        case 2u: {
            let a0 = atan2(g.y, g.x);
            let a = atan2(v.y, v.x);
            t = fract((a - a0) / (2.0 * PI) + 1.0);
        }
        case 3u: { t = abs(dot(v, g) / len2); }
        case 4u: {
            let u = g / len;
            t = (abs(dot(v, u)) + abs(dot(v, vec2<f32>(-u.y, u.x)))) / len;
        }
        default: { t = dot(v, g) / len2; }
    }
    let rep = u32(pv(1u).y);
    if (rep == 1u) { t = fract(t); } else if (rep == 2u) { let m = t - 2.0 * floor(t / 2.0); t = select(m, 2.0 - m, m > 1.0); } else { t = clamp(t, 0.0, 1.0); }
    if (pv(1u).z > 0.5) { t = 1.0 - t; }
    return t;
}
fn lut_sample(t: f32, n: u32, offset: u32) -> vec4<f32> {
    let x = clamp(t, 0.0, 1.0) * f32(n - 1u);
    let i = u32(floor(x));
    let j = min(i + 1u, n - 1u);
    return mix(lut[offset + i], lut[offset + j], x - f32(i));
}
@compute @workgroup_size(8, 8)
fn k_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    var t = gradient_t(doc_pos(p));
    if (pv(1u).w > 0.5) { t = t + (hash12(vec2<f32>(p)) - 0.5) / 255.0; }
    let c = lut_sample(t, 256u, 0u);
    textureStore(out_tex, out_coord(gid), premul(c));
}

// ------------------------------------------------------------------------------ tonal

fn levels_ch(x: f32, base: u32) -> f32 {
    let ib = pf(base);
    let iw = pf(base + 1u);
    let g = pf(base + 2u);
    let ob = pf(base + 3u);
    let ow = pf(base + 4u);
    var t = (x - ib) / max(iw - ib, 1e-5);
    t = sign(t) * pow(abs(t), 1.0 / max(g, 1e-3));
    return ob + t * (ow - ob);
}
// Levels operate in sRGB-encoded space (perceptual handles). Params: 4 channels x 5 floats.
@compute @workgroup_size(8, 8)
fn k_levels(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    var e = enc3(c.rgb);
    e = vec3<f32>(levels_ch(e.r, 0u), levels_ch(e.g, 0u), levels_ch(e.b, 0u));
    e = vec3<f32>(levels_ch(e.r, 5u), levels_ch(e.g, 10u), levels_ch(e.b, 15u));
    finish_adjust(gid, p, o, vec4<f32>(dec3(e), c.a));
}

// Curves: lut[0..1024) = (master, r, g, b), lut[1024..2048).x = OKLab L curve.
fn curve_lookup(x: f32, comp: u32, offset: u32) -> f32 {
    let t = clamp(x, 0.0, 1.0) * 1023.0;
    let i = u32(floor(t));
    let j = min(i + 1u, 1023u);
    let a = lut[offset + i][comp];
    let b = lut[offset + j][comp];
    var y = mix(a, b, t - f32(i));
    // extend linearly beyond [0,1] for HDR values
    if (x > 1.0) { y = y + (x - 1.0) * (lut[offset + 1023u][comp] - lut[offset + 1022u][comp]) * 1023.0; }
    if (x < 0.0) { y = y + x * (lut[offset + 1u][comp] - lut[offset][comp]) * 1023.0; }
    return y;
}
@compute @workgroup_size(8, 8)
fn k_curves(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    var e = enc3(c.rgb);
    e = vec3<f32>(curve_lookup(e.r, 0u, 0u), curve_lookup(e.g, 0u, 0u), curve_lookup(e.b, 0u, 0u));
    e = vec3<f32>(curve_lookup(e.r, 1u, 0u), curve_lookup(e.g, 2u, 0u), curve_lookup(e.b, 3u, 0u));
    var rgb = dec3(e);
    if (pf(0u) > 0.5) {
        var lab = to_oklab(rgb);
        lab.x = curve_lookup(lab.x, 0u, 1024u);
        rgb = from_oklab(lab);
    }
    finish_adjust(gid, p, o, vec4<f32>(rgb, c.a));
}

// P0: exposure, offset, gamma
@compute @workgroup_size(8, 8)
fn k_exposure(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    var rgb = c.rgb * exp2(pf(0u)) + pf(1u);
    let g = max(pf(2u), 1e-3);
    rgb = sign(rgb) * pow(abs(rgb), vec3<f32>(1.0 / g));
    finish_adjust(gid, p, o, vec4<f32>(rgb, c.a));
}

// P0: brightness, contrast, pivot — on OKLab L with chroma scaled to keep saturation.
@compute @workgroup_size(8, 8)
fn k_brightness_contrast(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    var lab = to_oklab(c.rgb);
    let l0 = max(lab.x, 1e-5);
    let k = tan((clamp(pf(1u), -0.99, 0.99) + 1.0) * PI / 4.0);
    var l = (lab.x - pf(2u)) * k + pf(2u);
    l = l + pf(0u) * 0.5 * (1.0 - abs(l - 0.5));
    l = max(l, 0.0);
    lab = vec3<f32>(l, lab.yz * (l / l0));
    finish_adjust(gid, p, o, vec4<f32>(from_oklab(lab), c.a));
}

// Shadows/highlights combine: in0 original, in1 blurred log-luminance mask. P0: shadows, highlights.
@compute @workgroup_size(8, 8)
fn k_sh_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let local_l = clamp(ld1c(p).r, 0.0, 1.0); // perceptual 0..1
    let sh = pf(0u);
    let hi = pf(1u);
    let shadow_w = pow(1.0 - local_l, 2.0);
    let high_w = pow(local_l, 2.0);
    let ev = sh * 2.0 * shadow_w - hi * 2.0 * high_w;
    finish_adjust(gid, p, o, vec4<f32>(c.rgb * exp2(ev), c.a));
}

// Perceptual luminance (OKLab L) as a mask, used before blurring.
@compute @workgroup_size(8, 8)
fn k_oklab_l(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let c = unpremul(ld0(gpos(gid)));
    let l = to_oklab(max(c.rgb, vec3<f32>(0.0))).x;
    textureStore(out_tex, out_coord(gid), vec4<f32>(l, l, l, 1.0));
}

// ------------------------------------------------------------------------------ colour

// Wheels: P0 lift(a,b,L,_) P1 gamma P2 gain P3 offset; P4.x saturation.
fn wheel_rgb(w: vec4<f32>) -> vec3<f32> {
    // wheel stores an OKLab (a,b) direction and a luminance offset; convert to an RGB offset
    let base = from_oklab(vec3<f32>(0.6, w.x * 0.25, w.y * 0.25)) - from_oklab(vec3<f32>(0.6, 0.0, 0.0));
    return base / 0.6 * 0.7 + vec3<f32>(w.z);
}
@compute @workgroup_size(8, 8)
fn k_color_wheels(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let lift = wheel_rgb(pv(0u));
    let gamma = wheel_rgb(pv(1u));
    let gain = wheel_rgb(pv(2u));
    let off = wheel_rgb(pv(3u));
    var e = enc3(c.rgb);
    e = e * (1.0 + gain) + lift * (1.0 - e);
    e = sign(e) * pow(abs(e), 1.0 / max(1.0 + gamma, vec3<f32>(0.05)));
    e = e + off;
    var rgb = dec3(e);
    let l = luma(rgb);
    rgb = l + (rgb - l) * pf(16u);
    finish_adjust(gid, p, o, vec4<f32>(rgb, c.a));
}

// P0: hue (deg), saturation, lightness, colorize; P1: colorize color (linear)
@compute @workgroup_size(8, 8)
fn k_hsl(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    var lab = to_oklab(c.rgb);
    if (pf(3u) > 0.5) {
        let t = to_oklab(pv(1u).rgb);
        let ch = length(t.yz) * (1.0 + pf(1u));
        let h = atan2(t.z, t.y);
        lab = vec3<f32>(lab.x, ch * cos(h), ch * sin(h));
    } else {
        let h = pf(0u) * PI / 180.0;
        let cs = cos(h);
        let sn = sin(h);
        lab = vec3<f32>(lab.x, lab.y * cs - lab.z * sn, lab.y * sn + lab.z * cs);
        lab = vec3<f32>(lab.x, lab.yz * (1.0 + pf(1u)));
    }
    let lt = pf(2u);
    if (lt >= 0.0) { lab.x = lab.x + (1.0 - lab.x) * lt; lab = vec3<f32>(lab.x, lab.yz * (1.0 - lt)); } else { lab.x = lab.x * (1.0 + lt); lab = vec3<f32>(lab.x, lab.yz * (1.0 + lt)); }
    finish_adjust(gid, p, o, vec4<f32>(from_oklab(lab), c.a));
}

// P0: vibrance, saturation
@compute @workgroup_size(8, 8)
fn k_vibrance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    var lab = to_oklab(c.rgb);
    let ch = length(lab.yz);
    let protect = 1.0 - smoothstep(0.0, 0.25, ch);
    let k = 1.0 + pf(0u) * protect + pf(1u);
    lab = vec3<f32>(lab.x, lab.yz * max(k, 0.0));
    finish_adjust(gid, p, o, vec4<f32>(from_oklab(lab), c.a));
}

// P0: temperature, tint. Scale LMS cone responses (von Kries).
@compute @workgroup_size(8, 8)
fn k_white_balance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let t = pf(0u);
    let n = pf(1u);
    let scale = vec3<f32>(exp2(t * 0.6 - n * 0.15), exp2(n * 0.3), exp2(-t * 0.6 - n * 0.15));
    let lms = (W2LMS * c.rgb) * scale;
    finish_adjust(gid, p, o, vec4<f32>(LMS2W * lms, c.a));
}

// P0 shadows wheel, P1 midtones, P2 highlights, P3.x preserve luminosity
@compute @workgroup_size(8, 8)
fn k_color_balance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let e = enc3(c.rgb);
    let l = clamp(luma(e), 0.0, 1.0);
    let ws = smoothstep(0.6, 0.0, l);
    let wh = smoothstep(0.4, 1.0, l);
    let wm = 1.0 - ws - wh;
    var r = e + wheel_rgb(pv(0u)) * ws * 0.5 + wheel_rgb(pv(1u)) * wm * 0.5 + wheel_rgb(pv(2u)) * wh * 0.5;
    var rgb = dec3(r);
    if (pf(12u) > 0.5) { rgb = rgb * (luma(c.rgb) / max(luma(rgb), 1e-6)); }
    finish_adjust(gid, p, o, vec4<f32>(rgb, c.a));
}

// P0..P2 rows (rr rg rb _), (gr gg gb _), (br bg bb _), P3.x monochrome
@compute @workgroup_size(8, 8)
fn k_channel_mixer(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    var rgb = vec3<f32>(dot(pv(0u).xyz, c.rgb), dot(pv(1u).xyz, c.rgb), dot(pv(2u).xyz, c.rgb));
    if (pf(12u) > 0.5) { rgb = vec3<f32>(rgb.r); }
    finish_adjust(gid, p, o, vec4<f32>(rgb, c.a));
}

fn hue_weight(h: f32, centre: f32, width: f32) -> f32 {
    var d = abs(h - centre);
    d = d - 2.0 * PI * floor(d / (2.0 * PI));
    d = min(d, 2.0 * PI - d);
    return smoothstep(width * PI, 0.0, d);
}
// P0: centre hue (rad), width, hue shift (deg), saturation; P1.x lightness
@compute @workgroup_size(8, 8)
fn k_selective_color(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    var lab = to_oklab(c.rgb);
    let ch = length(lab.yz);
    let h = atan2(lab.z, lab.y);
    let w = hue_weight(h, pf(0u), pf(1u)) * smoothstep(0.0, 0.04, ch);
    let nh = h + pf(2u) * PI / 180.0 * w;
    let nc = ch * max(1.0 + pf(3u) * w, 0.0);
    let nl = lab.x * (1.0 + pf(4u) * w * 0.5);
    lab = vec3<f32>(nl, nc * cos(nh), nc * sin(nh));
    finish_adjust(gid, p, o, vec4<f32>(from_oklab(lab), c.a));
}

// P0: reds yellows greens cyans; P1: blues magentas tint _; P2 tint colour
@compute @workgroup_size(8, 8)
fn k_black_white(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let lab = to_oklab(c.rgb);
    let h = atan2(lab.z, lab.y);
    let ch = length(lab.yz);
    // OKLab hue angles of primaries/secondaries (approx).
    var centres = array<f32, 6>(0.51, 1.70, 2.48, 3.38, 4.57, 5.65);
    var w = 0.0;
    var acc = 0.0;
    for (var i = 0u; i < 6u; i++) {
        let k = hue_weight(h, centres[i], 0.33) + 1e-4;
        w = w + k;
        acc = acc + k * pf(i);
    }
    let factor = acc / w;
    var l = lab.x + (factor - 0.5) * ch * 1.5;
    var rgb = from_oklab(vec3<f32>(max(l, 0.0), 0.0, 0.0));
    if (pf(6u) > 0.5) {
        let t = pv(2u).rgb / max(luma(pv(2u).rgb), 1e-5);
        rgb = rgb * t;
    }
    finish_adjust(gid, p, o, vec4<f32>(rgb, c.a));
}

@compute @workgroup_size(8, 8)
fn k_gradient_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let t = to_oklab(max(c.rgb, vec3<f32>(0.0))).x;
    let g = lut_sample(t, 256u, 0u);
    finish_adjust(gid, p, o, vec4<f32>(g.rgb, c.a * g.a));
}

// 3D LUT in sRGB-encoded domain. P0.x size, P0.y strength. lut = size^3 entries r-fastest.
@compute @workgroup_size(8, 8)
fn k_lut3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let n = u32(pf(0u));
    let e = clamp(enc3(c.rgb), vec3<f32>(0.0), vec3<f32>(1.0)) * f32(n - 1u);
    let i0 = min(vec3<u32>(floor(e)), vec3<u32>(n - 2u));
    let t = e - vec3<f32>(i0);
    var acc = vec3<f32>(0.0);
    for (var dz = 0u; dz < 2u; dz++) {
        for (var dy = 0u; dy < 2u; dy++) {
            for (var dx = 0u; dx < 2u; dx++) {
                let w = select(1.0 - t.x, t.x, dx == 1u) * select(1.0 - t.y, t.y, dy == 1u) * select(1.0 - t.z, t.z, dz == 1u);
                let idx = ((i0.z + dz) * n + (i0.y + dy)) * n + (i0.x + dx);
                acc = acc + lut[idx].rgb * w;
            }
        }
    }
    let rgb = mix(c.rgb, dec3(acc), pf(1u));
    finish_adjust(gid, p, o, vec4<f32>(rgb, c.a));
}

@compute @workgroup_size(8, 8)
fn k_invert(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    finish_adjust(gid, p, o, vec4<f32>(dec3(1.0 - clamp(enc3(c.rgb), vec3<f32>(0.0), vec3<f32>(1.0))), c.a));
}

@compute @workgroup_size(8, 8)
fn k_threshold(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let v = select(0.0, 1.0, srgb_oetf(luma(c.rgb)) >= pf(0u));
    finish_adjust(gid, p, o, vec4<f32>(vec3<f32>(v), c.a));
}

@compute @workgroup_size(8, 8)
fn k_posterize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let n = max(pf(0u) - 1.0, 1.0);
    let e = floor(clamp(enc3(c.rgb), vec3<f32>(0.0), vec3<f32>(1.0)) * n + 0.5) / n;
    finish_adjust(gid, p, o, vec4<f32>(dec3(e), c.a));
}

// ------------------------------------------------------------------------------ filters

// Separable gaussian. P0.x sigma (level px), P0.y radius (level px, integer).
@compute @workgroup_size(8, 8)
fn k_blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let sigma = max(pf(0u), 0.01);
    let r = i32(pf(1u));
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = -r; i <= r; i++) {
        let w = exp(-f32(i * i) / (2.0 * sigma * sigma));
        acc = acc + ld0e(p + vec2<i32>(i, 0), pf(2u)) * w;
        wsum = wsum + w;
    }
    textureStore(out_tex, out_coord(gid), acc / wsum);
}
@compute @workgroup_size(8, 8)
fn k_blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let sigma = max(pf(0u), 0.01);
    let r = i32(pf(1u));
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = -r; i <= r; i++) {
        let w = exp(-f32(i * i) / (2.0 * sigma * sigma));
        acc = acc + ld0e(p + vec2<i32>(0, i), pf(2u)) * w;
        wsum = wsum + w;
    }
    textureStore(out_tex, out_coord(gid), acc / wsum);
}
// Box blur, P0.x radius
@compute @workgroup_size(8, 8)
fn k_box_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let r = i32(round(pf(0u)));
    var acc = vec4<f32>(0.0);
    for (var i = -r; i <= r; i++) { acc = acc + ld0e(p + vec2<i32>(i, 0), pf(1u)); }
    textureStore(out_tex, out_coord(gid), acc / f32(2 * r + 1));
}
@compute @workgroup_size(8, 8)
fn k_box_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let r = i32(round(pf(0u)));
    var acc = vec4<f32>(0.0);
    for (var i = -r; i <= r; i++) { acc = acc + ld0e(p + vec2<i32>(0, i), pf(1u)); }
    textureStore(out_tex, out_coord(gid), acc / f32(2 * r + 1));
}
// Motion blur: P0.xy direction (unit), P0.z distance (level px)
@compute @workgroup_size(8, 8)
fn k_motion(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let dir = pv(0u).xy;
    let dist = max(pf(2u), 1.0);
    let n = i32(ceil(dist));
    var acc = vec4<f32>(0.0);
    var cnt = 0.0;
    for (var i = -n; i <= n; i++) {
        let off = dir * (f32(i) * dist / f32(max(n, 1)) * 0.5);
        acc = acc + ld0e(p + vec2<i32>(round(off)), pf(3u));
        cnt = cnt + 1.0;
    }
    let res = acc / cnt;
    let orig = ld0(p);
    textureStore(out_tex, out_coord(gid), mix(orig, res, mask3(p)));
}
// Unsharp: in0 orig, in1 blurred. P0: amount, threshold
@compute @workgroup_size(8, 8)
fn k_unsharp(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let b = ld1c(p);
    let c = unpremul(o);
    let bl = unpremul(b);
    let e = enc3(c.rgb);
    let eb = enc3(bl.rgb);
    let d = e - eb;
    let th = pf(1u);
    let mask = select(vec3<f32>(1.0), smoothstep(vec3<f32>(th), vec3<f32>(th * 2.0 + 1e-4), abs(d)), th > 0.0);
    let r = e + d * pf(0u) * mask;
    finish_adjust(gid, p, o, vec4<f32>(dec3(r), c.a));
}
@compute @workgroup_size(8, 8)
fn k_high_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let bl = unpremul(ld1c(p));
    let r = enc3(c.rgb) - enc3(bl.rgb) + 0.5;
    finish_adjust(gid, p, o, vec4<f32>(dec3(r), c.a));
}
// Noise: P0 amount, mono, seed
@compute @workgroup_size(8, 8)
fn k_noise(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    // noise is anchored to document pixels so it doesn't swim with zoom
    let d = floor(doc_pos(p)) + pf(2u) * 17.0;
    var n: vec3<f32>;
    let g = (hash12(d) + hash12(d + 7.1) + hash12(d + 13.7) - 1.5);
    if (pf(1u) > 0.5) { n = vec3<f32>(g); } else { n = vec3<f32>(g, hash12(d + 3.3) + hash12(d + 5.9) + hash12(d + 9.2) - 1.5, hash12(d + 1.7) + hash12(d + 11.3) + hash12(d + 2.9) - 1.5); }
    let e = enc3(c.rgb) + n * pf(0u) * 0.5;
    finish_adjust(gid, p, o, vec4<f32>(dec3(e), c.a));
}
// Vignette: P0 amount, midpoint, roundness, feather
@compute @workgroup_size(8, 8)
fn k_vignette(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let o = ld0(p);
    let c = unpremul(o);
    let d = doc_pos(p);
    let size = vec2<f32>(info.doc_w, info.doc_h);
    var uv = (d / size - 0.5) * 2.0;
    let aspect = size.x / size.y;
    let r = pf(2u);
    uv = vec2<f32>(uv.x * mix(1.0, aspect, r), uv.y);
    uv = uv / mix(1.0, max(aspect, 1.0), r);
    let dist = length(uv) / 1.4142;
    let v = smoothstep(pf(1u), pf(1u) + pf(3u), dist);
    let k = exp2(pf(0u) * 3.0 * v);
    finish_adjust(gid, p, o, vec4<f32>(c.rgb * k, c.a));
}

// ------------------------------------------------------------------------------ composite

// Mix: in0 A, in1 B, in3 mask. P0.x factor
@compute @workgroup_size(8, 8)
fn k_mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let m = clamp(ld3(p).r, 0.0, 1.0) * pf(0u);
    textureStore(out_tex, out_coord(gid), mix(ld0(p), ld1(p), m));
}

fn bilinear0(pos: vec2<f32>) -> vec4<f32> {
    let f = pos - 0.5;
    let i = vec2<i32>(floor(f));
    let t = f - floor(f);
    let a = ld0(i);
    let b = ld0(i + vec2<i32>(1, 0));
    let c = ld0(i + vec2<i32>(0, 1));
    let d = ld0(i + vec2<i32>(1, 1));
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}
// Transform: P0,P1 = inverse affine (level space) rows (a b tx _), (c d ty _)
@compute @workgroup_size(8, 8)
fn k_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<f32>(gpos(gid)) + 0.5;
    let r0 = pv(0u);
    let r1 = pv(1u);
    let src = vec2<f32>(r0.x * p.x + r0.y * p.y + r0.z, r1.x * p.x + r1.y * p.y + r1.z);
    textureStore(out_tex, out_coord(gid), bilinear0(src));
}
// Crop: P0.xy min, P0.zw max (document px)
@compute @workgroup_size(8, 8)
fn k_crop(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let d = doc_pos(p);
    let r = pv(0u);
    let inside_r = d.x >= r.x && d.y >= r.y && d.x < r.z && d.y < r.w;
    textureStore(out_tex, out_coord(gid), select(vec4<f32>(0.0), ld0(p), inside_r));
}

// ------------------------------------------------------------------------------ utility

// Channel to mask: P0.x channel (0 r,1 g,2 b,3 a,4 luminance)
@compute @workgroup_size(8, 8)
fn k_channel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let c = unpremul(ld0(gpos(gid)));
    let ch = u32(pf(0u));
    var v: f32;
    if (ch == 4u) { v = to_oklab(max(c.rgb, vec3<f32>(0.0))).x * c.a; } else if (ch == 3u) { v = c.a; } else { v = srgb_oetf(c[ch]) * c.a; }
    textureStore(out_tex, out_coord(gid), vec4<f32>(v, v, v, 1.0));
}
// Image -> mask coercion (perceptual luminance * alpha)
@compute @workgroup_size(8, 8)
fn k_to_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let c = unpremul(ld0(gpos(gid)));
    let v = to_oklab(max(c.rgb, vec3<f32>(0.0))).x * c.a;
    textureStore(out_tex, out_coord(gid), vec4<f32>(v, v, v, 1.0));
}
// Mask -> image coercion (grey, opaque). P0.x = default when outside input.
@compute @workgroup_size(8, 8)
fn k_mask_to_image(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let q = p - info.in0.xy;
    var v = pf(0u);
    if (inside(q, info.in0.zw)) { v = textureLoad(in0, q, 0).r; }
    let l = srgb_eotf(clamp(v, 0.0, 1.0));
    textureStore(out_tex, out_coord(gid), vec4<f32>(l, l, l, 1.0));
}
// Raw mask copy with default outside (r-texture upload -> rgba mask image). P0.x default
@compute @workgroup_size(8, 8)
fn k_mask_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let q = p - info.in0.xy;
    var v = pf(0u);
    if (inside(q, info.in0.zw)) { v = textureLoad(in0, q, 0).r; }
    textureStore(out_tex, out_coord(gid), vec4<f32>(v, v, v, 1.0));
}
@compute @workgroup_size(8, 8)
fn k_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let r = select(0.0, srgb_eotf(ld0(p).r), info.connected.x != 0u);
    let g = select(0.0, srgb_eotf(ld1(p).r), info.connected.y != 0u);
    let b = select(0.0, srgb_eotf(ld2(p).r), info.connected.z != 0u);
    let a = select(1.0, ld3(p).r, info.connected.w != 0u);
    textureStore(out_tex, out_coord(gid), vec4<f32>(vec3<f32>(r, g, b) * a, a));
}
@compute @workgroup_size(8, 8)
fn k_mask_invert(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let v = 1.0 - ld0(gpos(gid)).r;
    textureStore(out_tex, out_coord(gid), vec4<f32>(v, v, v, 1.0));
}
@compute @workgroup_size(8, 8)
fn k_mask_levels(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let v = smoothstep(pf(0u), max(pf(1u), pf(0u) + 1e-4), ld0(gpos(gid)).r);
    textureStore(out_tex, out_coord(gid), vec4<f32>(v, v, v, 1.0));
}
// Math: P0.x op, P0.y value
@compute @workgroup_size(8, 8)
fn k_math(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let c = unpremul(ld0(p));
    let v = pf(1u);
    var rgb = c.rgb;
    switch u32(pf(0u)) {
        case 0u: { rgb = rgb + v; }
        case 1u: { rgb = rgb * v; }
        case 2u: { rgb = sign(rgb) * pow(abs(rgb), vec3<f32>(v)); }
        case 3u: { rgb = min(rgb, vec3<f32>(v)); }
        case 4u: { rgb = max(rgb, vec3<f32>(v)); }
        default: { rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)); }
    }
    textureStore(out_tex, out_coord(gid), premul(vec4<f32>(rgb, c.a)));
}
@compute @workgroup_size(8, 8)
fn k_set_alpha(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let c = unpremul(ld0(p));
    textureStore(out_tex, out_coord(gid), premul(vec4<f32>(c.rgb, mask3(p))));
}

// Convert to sRGB8 for thumbnails (over a checkerboard). Output format is rgba8unorm.
@compute @workgroup_size(8, 8)
fn k_thumb(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = gpos(gid);
    let c = ld0(p);
    let chk = select(0.75, 0.55, ((p.x / 8) + (p.y / 8)) % 2 == 0);
    let bg = srgb_eotf(chk);
    let rgb = c.rgb + vec3<f32>(bg) * (1.0 - c.a);
    // working (Rec.2020) -> sRGB primaries
    let m = mat3x3<f32>(1.6605, -0.1246, -0.0182, -0.5876, 1.1329, -0.1006, -0.0728, -0.0083, 1.1187);
    textureStore(out_tex, out_coord(gid), vec4<f32>(clamp(enc3(m * rgb), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
