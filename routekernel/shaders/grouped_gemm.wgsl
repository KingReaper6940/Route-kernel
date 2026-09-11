// Each workgroup computes one 16x16 output tile belonging to one expert.
// Shared tiles reuse loads across 256 invocations. Partial tiles load zeros.
struct Shape { rows: u32, cols: u32, inner: u32, activation: u32 }
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read> tile_experts: array<u32>;
@group(0) @binding(5) var<storage, read> tile_rows: array<u32>;
@group(0) @binding(6) var<uniform> shape: Shape;
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let expert = tile_experts[group.y];
    let row = tile_rows[group.y] + local.y;
    let row_end = offsets[expert + 1u];
    let col = group.x * 16u + local.x;
    let slot = local.y * 16u + local.x;
    var acc = 0.0;
    for (var base = 0u; base < shape.inner; base += 16u) {
        tile_a[slot] = 0.0;
        tile_b[slot] = 0.0;
        if (row < row_end && base + local.x < shape.inner) {
            tile_a[slot] = inputs[row * shape.inner + base + local.x];
        }
        if (col < shape.cols && base + local.y < shape.inner) {
            tile_b[slot] = weights[(expert * shape.inner + base + local.y) * shape.cols + col];
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k += 1u) {
            acc += tile_a[local.y * 16u + k] * tile_b[k * 16u + local.x];
        }
        workgroupBarrier();
    }
    if (row < row_end && col < shape.cols) {
        if (shape.activation == 1u) { acc = max(acc, 0.0); }
        outputs[row * shape.cols + col] = acc;
    }
}
