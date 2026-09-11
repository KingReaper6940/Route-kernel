// One invocation owns one final output element; no floating-point atomics.
struct Shape { rows: u32, width: u32, top_k: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> expert_outputs: array<f32>;
@group(0) @binding(1) var<storage, read> route_weights: array<f32>;
@group(0) @binding(2) var<storage, read> inverse: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < shape.rows * shape.width) {
        let token = index / shape.width;
        let col = index % shape.width;
        var acc = 0.0;
        for (var slot = 0u; slot < shape.top_k; slot += 1u) {
            let row = inverse[token * shape.top_k + slot];
            acc += route_weights[row] * expert_outputs[row * shape.width + col];
        }
        outputs[index] = acc;
    }
}
