// Canvas display pass: maps framebuffer pixels to document space, samples the evaluated
// level image, composites over a checkerboard and applies the display transform
// (analytic matrix + transfer, or a baked 3D LUT from the display ICC profile).

struct View {
    fb_to_doc0: vec4<f32>,  // row 0: a b tx
    fb_to_doc1: vec4<f32>,  // row 1: c d ty
    img_origin: vec2<f32>,  // level coords of texel 0
    img_size: vec2<f32>,    // valid size (level px)
    tex_size: vec2<f32>,
    level_scale: f32,
    nearest: f32,
    doc_size: vec2<f32>,
    checker: f32,
    has_lut: f32,
    disp0: vec4<f32>,
    disp1: vec4<f32>,
    disp2: vec4<f32>,
    transfer: f32,          // 0 sRGB, 1 linear, 2 gamma 2.2, 3 BT.1886 (2.4)
    target_srgb: f32,       // surface is *Srgb: write linear and let hardware encode
    exposure: f32,
    dither: f32,
    bg: vec4<f32>,          // display-encoded
    chk_a: vec4<f32>,
    chk_b: vec4<f32>,
};

@group(0) @binding(0) var<uniform> v: View;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var samp_lin: sampler;
@group(0) @binding(3) var lut: texture_3d<f32>;
@group(0) @binding(4) var samp_lut: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn oetf(x: f32) -> f32 {
    let a = max(x, 0.0);
    if (v.transfer < 0.5) {
        if (a <= 0.0031308) { return a * 12.92; }
        return 1.055 * pow(a, 1.0 / 2.4) - 0.055;
    } else if (v.transfer < 1.5) {
        return a;
    } else if (v.transfer < 2.5) {
        return pow(a, 1.0 / 2.2);
    }
    return pow(a, 1.0 / 2.4);
}
fn srgb_eotf(x: f32) -> f32 {
    if (x <= 0.04045) { return x / 12.92; }
    return pow((x + 0.055) / 1.055, 2.4);
}
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn fetch(level: vec2<f32>) -> vec4<f32> {
    let local = level - v.img_origin;
    if (local.x < 0.0 || local.y < 0.0 || local.x >= v.img_size.x || local.y >= v.img_size.y) {
        return vec4<f32>(0.0);
    }
    if (v.nearest > 0.5) {
        return textureLoad(img, vec2<i32>(floor(local)), 0);
    }
    // Clamp the bilinear footprint to valid texels.
    let uv = clamp(local, vec2<f32>(0.5), v.img_size - 0.5) / v.tex_size;
    return textureSampleLevel(img, samp_lin, uv, 0.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let fb = pos.xy;
    let doc = vec2<f32>(dot(v.fb_to_doc0.xyz, vec3<f32>(fb, 1.0)), dot(v.fb_to_doc1.xyz, vec3<f32>(fb, 1.0)));
    var out: vec3<f32>;
    if (doc.x < 0.0 || doc.y < 0.0 || doc.x >= v.doc_size.x || doc.y >= v.doc_size.y) {
        out = v.bg.rgb;
    } else {
        let c = fetch(doc * v.level_scale) * vec4<f32>(vec3<f32>(exp2(v.exposure)), 1.0);
        // working -> display
        var disp: vec3<f32>;
        if (v.has_lut > 0.5) {
            let n = f32(textureDimensions(lut).x);
            let rgb = select(vec3<f32>(0.0), c.rgb / c.a, c.a > 1e-6);
            let e = vec3<f32>(
                select(srgb_enc(rgb.r), 0.0, rgb.r <= 0.0),
                select(srgb_enc(rgb.g), 0.0, rgb.g <= 0.0),
                select(srgb_enc(rgb.b), 0.0, rgb.b <= 0.0));
            let coord = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0)) * ((n - 1.0) / n) + 0.5 / n;
            disp = textureSampleLevel(lut, samp_lut, coord, 0.0).rgb;
        } else {
            let rgb = select(vec3<f32>(0.0), c.rgb / c.a, c.a > 1e-6);
            let m = mat3x3<f32>(v.disp0.xyz, v.disp1.xyz, v.disp2.xyz);
            let l = m * rgb;
            disp = vec3<f32>(oetf(l.r), oetf(l.g), oetf(l.b));
        }
        // checkerboard in screen space
        let k = floor(fb / v.checker);
        let chk = select(v.chk_a.rgb, v.chk_b.rgb, (i32(k.x) + i32(k.y)) % 2 == 0);
        out = mix(chk, clamp(disp, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(c.a, 0.0, 1.0));
    }
    if (v.dither > 0.5) {
        out = out + (hash12(fb) - 0.5) / 255.0;
    }
    if (v.target_srgb > 0.5) {
        out = vec3<f32>(srgb_eotf(max(out.r, 0.0)), srgb_eotf(max(out.g, 0.0)), srgb_eotf(max(out.b, 0.0)));
    }
    return vec4<f32>(out, 1.0);
}

fn srgb_enc(a: f32) -> f32 {
    if (a <= 0.0031308) { return a * 12.92; }
    return 1.055 * pow(a, 1.0 / 2.4) - 0.055;
}
