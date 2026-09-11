// Duplicate input rows into contiguous expert order.
struct Shape { rows: u32, width: u32, top_k: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> tokens: array<f32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> grouped: array<f32>;
@group(0) @binding(3) var<uniform> shape: Shape;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < shape.rows * shape.width) {
        let row = index / shape.width;
        let col = index % shape.width;
        grouped[index] = tokens[token_ids[row] * shape.width + col];
    }
}
