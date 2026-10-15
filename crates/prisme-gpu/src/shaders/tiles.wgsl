// Resident tile pyramid kernels (appended to kernels.wgsl: same bindings and helpers).

// A level tile from its four children: in0..in3 = top-left, top-right, bottom-left,
// bottom-right (one 256² tile each, origin 0; unconnected = missing). Mirrors
// prisme_core::Tile::downsample, except that a missing child reads as `default` (masks
// that reveal by default) instead of zero. P0 = [channels, default].
fn ds_px(q: u32, p: vec2<i32>) -> vec4<f32> {
    let d = vec4<f32>(pf(1u));
    switch q {
        case 0u: { if (info.connected.x == 0u) { return d; } return ld0(p); }
        case 1u: { if (info.connected.y == 0u) { return d; } return ld1(p); }
        case 2u: { if (info.connected.z == 0u) { return d; } return ld2(p); }
        default: { if (info.connected.w == 0u) { return d; } return ld3(p); }
    }
}

@compute @workgroup_size(8, 8)
fn k_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let o = vec2<i32>(gid.xy);
    let half = i32(TILE_HALF);
    let q = u32(o.x / half) + 2u * u32(o.y / half);
    let s = (o % vec2<i32>(half)) * 2;
    let a = ds_px(q, s);
    let b = ds_px(q, s + vec2<i32>(1, 0));
    let c = ds_px(q, s + vec2<i32>(0, 1));
    let d = ds_px(q, s + vec2<i32>(1, 1));
    if (pf(0u) < 1.5) {
        let v = 0.25 * (a.r + b.r + c.r + d.r);
        textureStore(out_tex, out_coord(gid), vec4<f32>(v));
        return;
    }
    var acc = vec4<f32>(0.0);
    for (var i = 0; i < 4; i = i + 1) {
        var p: vec4<f32>;
        switch i {
            case 0: { p = a; }
            case 1: { p = b; }
            case 2: { p = c; }
            default: { p = d; }
        }
        acc = acc + vec4<f32>(p.rgb * p.a, p.a);
    }
    var res = vec4<f32>(0.0);
    if (acc.a > 0.0) { res = vec4<f32>(acc.rgb / acc.a, acc.a * 0.25); }
    textureStore(out_tex, out_coord(gid), res);
}
