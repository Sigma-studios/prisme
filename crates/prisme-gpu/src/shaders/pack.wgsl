// Stroke tiles to f16 for landing: one f32 tile in, its f16 bits out (tile layout: row-major,
// channels interleaved, two values per u32, little end first), exactly as the `half` crate
// converts on the CPU. The conversion is done on the bits so nothing (texture format
// conversion, denormal flushing) can round differently from the CPU.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

const TS: u32 = TILE_SIZE_U;

// half::f16::from_f32 (`f32_to_f16_fallback`): round to nearest, ties to even.
fn f16_bits(v: f32) -> u32 {
    let x = bitcast<u32>(v);
    let sign = x & 0x80000000u;
    let exp = x & 0x7F800000u;
    let man = x & 0x007FFFFFu;
    let half_sign = sign >> 16u;
    if (exp == 0x7F800000u) {
        // Infinity, or NaN with its mantissa's top bits and the quiet bit.
        var nan_bit = 0u;
        if (man != 0u) { nan_bit = 0x0200u; }
        return half_sign | 0x7C00u | nan_bit | (man >> 13u);
    }
    let half_exp = i32(exp >> 23u) - 127 + 15;
    if (half_exp >= 0x1F) {
        return half_sign | 0x7C00u;
    }
    if (half_exp <= 0) {
        if (14 - half_exp > 24) {
            return half_sign;
        }
        let m = man | 0x00800000u;
        var half_man = m >> u32(14 - half_exp);
        let round_bit = 1u << u32(13 - half_exp);
        if ((m & round_bit) != 0u && (m & (3u * round_bit - 1u)) != 0u) {
            half_man = half_man + 1u;
        }
        return half_sign | half_man;
    }
    let h = half_sign | (u32(half_exp) << 10u) | (man >> 13u);
    let round_bit = 0x00001000u;
    if ((man & round_bit) != 0u && (man & (3u * round_bit - 1u)) != 0u) {
        return h + 1u;
    }
    return h;
}

fn value(i: u32, channels: u32) -> f32 {
    let p = i / channels;
    let v = textureLoad(src, vec2<i32>(i32(p % TS), i32(p / TS)), 0);
    switch i % channels {
        case 0u: { return v.r; }
        case 1u: { return v.g; }
        case 2u: { return v.b; }
        default: { return v.a; }
    }
}

fn pack(gid: vec3<u32>, channels: u32) {
    let j = gid.x;
    if (j >= TS * TS * channels / 2u) { return; }
    dst[j] = f16_bits(value(2u * j, channels)) | (f16_bits(value(2u * j + 1u, channels)) << 16u);
}

@compute @workgroup_size(64)
fn k_pack4(@builtin(global_invocation_id) gid: vec3<u32>) {
    pack(gid, 4u);
}

@compute @workgroup_size(64)
fn k_pack1(@builtin(global_invocation_id) gid: vec3<u32>) {
    pack(gid, 1u);
}
