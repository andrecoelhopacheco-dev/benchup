//! Binary glTF 2.0 (`.glb`) export for game engines (Godot, three.js, Blender).
//!
//! - Units stay in metres; BenchUp is Z-up and glTF is Y-up, so `(x, y, z)` becomes `(x, z, -y)`.
//! - Every visible face is triangulated with flat normals (vertices are not shared between faces, so hard
//!   edges stay hard).
//! - `TEXCOORD_0` uses a box projection in metres: a texture with `repeat` covers one metre per tile, which
//!   is what a tiling PBR material (Poly Haven, FreeStylized) expects.
//! - One primitive per face colour, each with a PBR metallic-roughness material (double-sided, because
//!   push/pull faces are not guaranteed to be wound outward).
use benchup_core::{DVec3, Model};
use std::collections::BTreeMap;

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"
const DEFAULT_COLOR: [u8; 3] = [200, 200, 200];
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

#[derive(Default)]
struct Primitive {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

/// BenchUp Z-up to glTF Y-up (right-handed in both).
fn to_gltf(p: DVec3) -> [f32; 3] {
    [p.x as f32, p.z as f32, -p.y as f32]
}

/// Box projection: the face uses the plane of its dominant normal axis, measured in metres.
fn box_uv(p: [f32; 3], n: [f32; 3]) -> [f32; 2] {
    let [ax, ay, az] = n.map(f32::abs);
    if ay >= ax && ay >= az {
        [p[0], -p[2]]
    } else if ax >= az {
        [-p[2] * n[0].signum(), -p[1]]
    } else {
        [p[0] * n[2].signum(), -p[1]]
    }
}

fn collect(model: &Model) -> BTreeMap<[u8; 3], Primitive> {
    let mut by_color: BTreeMap<[u8; 3], Primitive> = BTreeMap::new();
    for face in model.faces.values().filter(|f| !model.is_hidden(f.id)) {
        let color = model.face_colors.get(&face.id).copied().unwrap_or(DEFAULT_COLOR);
        let prim = by_color.entry(color).or_default();
        for tri in model.face_triangles(face) {
            let [a, b, c] = tri.map(to_gltf);
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            if !(len > 1e-12) {
                continue; // degenerate sliver: nothing to draw
            }
            let n = cross.map(|v| v / len);
            for p in [a, b, c] {
                prim.indices.push(prim.positions.len() as u32);
                prim.positions.push(p);
                prim.normals.push(n);
                prim.uvs.push(box_uv(p, n));
            }
        }
    }
    by_color.retain(|_, p| !p.indices.is_empty());
    by_color
}

struct Writer {
    bin: Vec<u8>,
    views: Vec<serde_json::Value>,
    accessors: Vec<serde_json::Value>,
}

impl Writer {
    fn push(&mut self, bytes: &[u8], target: u32) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        self.views.push(serde_json::json!({
            "buffer": 0, "byteOffset": offset, "byteLength": bytes.len(), "target": target
        }));
        self.views.len() - 1
    }

    fn vec_accessor<const N: usize>(&mut self, data: &[[f32; N]], with_bounds: bool) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.iter().flat_map(|c| c.to_le_bytes())).collect();
        let view = self.push(&bytes, ARRAY_BUFFER);
        let kind = match N { 2 => "VEC2", 3 => "VEC3", _ => "VEC4" };
        let mut acc = serde_json::json!({
            "bufferView": view, "componentType": FLOAT, "count": data.len(), "type": kind
        });
        if with_bounds {
            let mut min = [f32::INFINITY; N];
            let mut max = [f32::NEG_INFINITY; N];
            for v in data {
                for i in 0..N {
                    min[i] = min[i].min(v[i]);
                    max[i] = max[i].max(v[i]);
                }
            }
            acc["min"] = serde_json::json!(min.to_vec());
            acc["max"] = serde_json::json!(max.to_vec());
        }
        self.accessors.push(acc);
        self.accessors.len() - 1
    }

    fn index_accessor(&mut self, data: &[u32]) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let view = self.push(&bytes, ELEMENT_ARRAY_BUFFER);
        self.accessors.push(serde_json::json!({
            "bufferView": view, "componentType": UNSIGNED_INT, "count": data.len(), "type": "SCALAR"
        }));
        self.accessors.len() - 1
    }
}

fn srgb_to_linear(c: u8) -> f64 {
    let s = f64::from(c) / 255.;
    if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

/// Exports the visible faces as a `.glb`. `name` becomes the mesh and node name.
pub fn export_glb(model: &Model, name: &str) -> Result<Vec<u8>, String> {
    let groups = collect(model);
    if groups.is_empty() {
        return Err(benchup_core::tr!("interchange.there_is_no_visible_geometry_to_export").into());
    }
    let mut w = Writer { bin: Vec::new(), views: Vec::new(), accessors: Vec::new() };
    let mut primitives = Vec::new();
    let mut materials = Vec::new();
    for (color, prim) in &groups {
        let position = w.vec_accessor(&prim.positions, true);
        let normal = w.vec_accessor(&prim.normals, false);
        let uv = w.vec_accessor(&prim.uvs, false);
        let indices = w.index_accessor(&prim.indices);
        materials.push(serde_json::json!({
            "name": format!("cor_{:02x}{:02x}{:02x}", color[0], color[1], color[2]),
            "doubleSided": true,
            "pbrMetallicRoughness": {
                "baseColorFactor": [srgb_to_linear(color[0]), srgb_to_linear(color[1]), srgb_to_linear(color[2]), 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8
            }
        }));
        primitives.push(serde_json::json!({
            "attributes": { "POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv },
            "indices": indices,
            "material": materials.len() - 1,
            "mode": 4
        }));
    }
    while w.bin.len() % 4 != 0 {
        w.bin.push(0);
    }
    let mesh_name = if name.trim().is_empty() { "BenchUp" } else { name.trim() };
    let doc = serde_json::json!({
        "asset": { "version": "2.0", "generator": "BenchUp glTF export" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": mesh_name }],
        "meshes": [{ "name": mesh_name, "primitives": primitives }],
        "materials": materials,
        "accessors": w.accessors,
        "bufferViews": w.views,
        "buffers": [{ "byteLength": w.bin.len() }]
    });
    let mut json = serde_json::to_vec(&doc).map_err(|e| e.to_string())?;
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let total = 12 + 8 + json.len() + 8 + w.bin.len();
    let mut out = Vec::with_capacity(total);
    for word in [GLB_MAGIC, 2, total as u32, json.len() as u32, CHUNK_JSON] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&json);
    out.extend_from_slice(&(w.bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&w.bin);
    Ok(out)
}

#[cfg(test)]
#[path = "gltf_export_tests.rs"]
mod tests;
