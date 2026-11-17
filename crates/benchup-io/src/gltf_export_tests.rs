use super::*;
use benchup_core::DVec3;

/// A 2 m × 1 m × 0.5 m box, red on top, built the way the user does it (rectangle, then push/pull).
fn box_model() -> Model {
    let mut model = Model::default();
    let corners = [DVec3::ZERO, DVec3::X * 2., DVec3::new(2., 1., 0.), DVec3::Y];
    for i in 0..4 {
        model.add_edge(corners[i], corners[(i + 1) % 4]).unwrap();
    }
    let base = *model.faces.keys().next().unwrap();
    let top = model.push_pull(base, 0.5).unwrap();
    model.face_colors.insert(top, [255, 0, 0]);
    model
}

fn parse(glb: &[u8]) -> (serde_json::Value, &[u8]) {
    let word = |i: usize| u32::from_le_bytes(glb[i..i + 4].try_into().unwrap());
    assert_eq!(word(0), GLB_MAGIC);
    assert_eq!(word(4), 2);
    assert_eq!(word(8) as usize, glb.len());
    let json_len = word(12) as usize;
    assert_eq!(word(16), CHUNK_JSON);
    let json = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
    let bin_start = 20 + json_len;
    assert_eq!(word(bin_start + 4), CHUNK_BIN);
    (json, &glb[bin_start + 8..])
}

#[test]
fn box_exports_y_up_metres_with_uvs_and_one_material_per_colour() {
    let glb = export_glb(&box_model(), "Machado").unwrap();
    let (json, bin) = parse(&glb);
    assert_eq!(json["nodes"][0]["name"], "Machado");
    let prims = json["meshes"][0]["primitives"].as_array().unwrap();
    assert_eq!(prims.len(), 2, "grey sides + red top");
    let triangles: u64 = prims
        .iter()
        .map(|p| json["accessors"][p["indices"].as_u64().unwrap() as usize]["count"].as_u64().unwrap() / 3)
        .sum();
    assert_eq!(triangles, 12, "6 quads");
    // bounds: x 0..2, height (y) 0..0.5, depth z -1..0 (BenchUp +Y becomes glTF -Z)
    let (mut min, mut max) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in prims {
        let acc = &json["accessors"][p["attributes"]["POSITION"].as_u64().unwrap() as usize];
        for i in 0..3 {
            min[i] = min[i].min(acc["min"][i].as_f64().unwrap());
            max[i] = max[i].max(acc["max"][i].as_f64().unwrap());
        }
        assert!(p["attributes"]["TEXCOORD_0"].is_u64());
        assert!(p["attributes"]["NORMAL"].is_u64());
    }
    let near = |a: f64, b: f64| (a - b).abs() < 1e-5;
    assert!(near(min[0], 0.) && near(max[0], 2.));
    assert!(near(min[1], 0.) && near(max[1], 0.5));
    assert!(near(min[2], -1.) && near(max[2], 0.));
    assert_eq!(json["buffers"][0]["byteLength"].as_u64().unwrap() as usize, bin.len());
    let red = json["materials"].as_array().unwrap().iter().any(|m| m["pbrMetallicRoughness"]["baseColorFactor"][0] == 1.0);
    assert!(red, "top keeps its colour (linear)");
}

#[test]
fn uvs_are_in_metres_so_textures_tile_per_metre() {
    let glb = export_glb(&box_model(), "Caixa").unwrap();
    let (json, bin) = parse(&glb);
    let mut widest = 0f32;
    for p in json["meshes"][0]["primitives"].as_array().unwrap() {
        let acc = &json["accessors"][p["attributes"]["TEXCOORD_0"].as_u64().unwrap() as usize];
        let view = &json["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let off = view["byteOffset"].as_u64().unwrap() as usize;
        let len = view["byteLength"].as_u64().unwrap() as usize;
        let floats: Vec<f32> = bin[off..off + len].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        for uv in floats.chunks(2) {
            widest = widest.max(uv[0].abs()).max(uv[1].abs());
        }
    }
    assert!((widest - 2.).abs() < 1e-5, "the 2 m side spans 2 texture tiles, got {widest}");
}

#[test]
fn empty_model_is_an_error() {
    assert!(export_glb(&Model::default(), "Nada").is_err());
}
