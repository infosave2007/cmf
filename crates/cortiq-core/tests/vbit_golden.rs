//! vbit golden parity: Rust dequant vs the python encoder's expected
//! reconstruction (fixture: converter encode_vbit, tests/fixtures).

use cortiq_core::quant::dequant_vbit;

#[test]
fn vbit_dequant_matches_python_encoder() {
    let fx: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/vbit_golden.json")).unwrap();
    let rows = fx["rows"].as_u64().unwrap() as usize;
    let cols = fx["cols"].as_u64().unwrap() as usize;
    let bytes = {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(fx["bytes"].as_str().unwrap())
            .unwrap()
    };
    let expect: Vec<f32> = fx["expect"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    let mut got = vec![0f32; rows * cols];
    dequant_vbit(&bytes, rows, cols, &mut got).unwrap();
    for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
        assert!(
            (g - e).abs() <= 1e-6 * e.abs().max(1e-3),
            "elem {i}: {g} vs {e}"
        );
    }

    // Safe-floor: a 2-bit row must be refused loudly (P13 claim 13).
    let mut bad = bytes.clone();
    bad[0] = 2;
    assert!(dequant_vbit(&bad, rows, cols, &mut got).is_err());
}
