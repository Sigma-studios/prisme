// Histogram (R,G,B,Luma), RGB waveform and OKLab vectorscope accumulated with atomics.
// Inputs are premultiplied working-space images; scopes show display-encoded sRGB.
// Extended range: `hist[1024..2048]` and the waveform map linear [0, range] (× SDR white)
// through the sRGB curve onto 256 bins, so HDR highlights are visible; `hist[0..1024]` stays
// the SDR [0, 1] histogram (used by Levels/Curves). hist[2048..2052] count samples above SDR
// white per R, G, B, luma; hist[2052] counts all (non-transparent) samples.

struct Params { size: vec2<u32>, origin: vec2<u32>, range: f32, enc_range: f32, _pad: vec2<f32>, };
@group(0) @binding(0) var<uniform> prm: Params;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>, 2056>;
@group(0) @binding(3) var<storage, read_write> wave: array<atomic<u32>, 196608>;
@group(0) @binding(4) var<storage, read_write> vec_scope: array<atomic<u32>, 65536>;

const W2LMS: mat3x3<f32> = MAT_W2LMS;
const LMS2LAB: mat3x3<f32> = MAT_LMS2LAB;
const TO_SRGB: mat3x3<f32> = mat3x3<f32>(1.6605, -0.1246, -0.0182, -0.5876, 1.1329, -0.1006, -0.0728, -0.0083, 1.1187);

fn enc(x: f32) -> f32 {
    let a = clamp(x, 0.0, 1.0);
    if (a <= 0.0031308) { return a * 12.92; }
    return 1.055 * pow(a, 1.0 / 2.4) - 0.055;
}
// Linear [0, range] -> [0, 1] through the (unbounded) sRGB curve.
fn enc_r(x: f32) -> f32 {
    let a = clamp(x, 0.0, prm.range);
    var e: f32;
    if (a <= 0.0031308) { e = a * 12.92; } else { e = 1.055 * pow(a, 1.0 / 2.4) - 0.055; }
    return e / prm.enc_range;
}
fn cbrt(x: f32) -> f32 { return sign(x) * pow(abs(x), 1.0 / 3.0); }

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= prm.size.x || gid.y >= prm.size.y) { return; }
    let c = textureLoad(img, vec2<i32>(gid.xy + prm.origin), 0);
    if (c.a <= 0.001) { return; }
    let rgb = c.rgb / c.a;
    let s = TO_SRGB * rgb;
    let e = vec3<f32>(enc(s.r), enc(s.g), enc(s.b));
    let bins = vec3<u32>(e * 255.0 + 0.5);
    let lin_l = dot(rgb, vec3<f32>(0.2627, 0.6780, 0.0593));
    let l = enc(lin_l);
    atomicAdd(&hist[bins.r], 1u);
    atomicAdd(&hist[256u + bins.g], 1u);
    atomicAdd(&hist[512u + bins.b], 1u);
    atomicAdd(&hist[768u + u32(l * 255.0 + 0.5)], 1u);
    let xb = vec3<u32>(vec3<f32>(enc_r(s.r), enc_r(s.g), enc_r(s.b)) * 255.0 + 0.5);
    atomicAdd(&hist[1024u + xb.r], 1u);
    atomicAdd(&hist[1280u + xb.g], 1u);
    atomicAdd(&hist[1536u + xb.b], 1u);
    atomicAdd(&hist[1792u + u32(enc_r(lin_l) * 255.0 + 0.5)], 1u);
    // Above SDR white (small tolerance for f16 rounding of exact white).
    let over = 1.001;
    if (s.r > over) { atomicAdd(&hist[2048u], 1u); }
    if (s.g > over) { atomicAdd(&hist[2049u], 1u); }
    if (s.b > over) { atomicAdd(&hist[2050u], 1u); }
    if (lin_l > over) { atomicAdd(&hist[2051u], 1u); }
    atomicAdd(&hist[2052u], 1u);
    let col = min(gid.x * 256u / prm.size.x, 255u);
    atomicAdd(&wave[(255u - xb.r) * 256u + col], 1u);
    atomicAdd(&wave[65536u + (255u - xb.g) * 256u + col], 1u);
    atomicAdd(&wave[131072u + (255u - xb.b) * 256u + col], 1u);
    let lms = W2LMS * max(rgb, vec3<f32>(0.0));
    let lab = LMS2LAB * vec3<f32>(cbrt(lms.x), cbrt(lms.y), cbrt(lms.z));
    let uv = clamp(vec2<f32>(lab.y, -lab.z) / 0.4 * 127.5 + 127.5, vec2<f32>(0.0), vec2<f32>(255.0));
    atomicAdd(&vec_scope[u32(uv.y) * 256u + u32(uv.x)], 1u);
}
