//! Two ways to build the same ladder, and how far apart they land.
//!
//!     CMF_Q4TP_PARITY=wtoy.cmf cargo test -p cortiq-engine --test q4tp_ladder_forms -- --nocapture
//!
//! The CPU expands a row's 32 rungs geometrically — `t[c] = t[c-1] * 2^step`
//! — while every GPU kernel evaluates `2^(lo + c*step)` directly, on the
//! stated grounds that the gap is far below a 4-bit grid. This measures the
//! gap instead of assuming it, because a device that disagrees with the CPU
//! by a thousandth is either fine or the first sign of something else, and
//! the difference between those two readings is a number.

#[test]
fn the_two_ladder_forms_agree() {
    let Ok(path) = std::env::var("CMF_Q4TP_PARITY") else {
        return;
    };
    let model = std::sync::Arc::new(cortiq_core::CmfModel::open(&path).expect("open"));
    let mut worst = 0.0f64;
    let mut worst_row = String::new();
    let mut checked = 0;
    for e in model.tensors.iter() {
        if e.dtype != cortiq_core::TensorDtype::Q4TiledP || e.shape.len() != 2 {
            continue;
        }
        let (rows, cols) = (e.shape[0], e.shape[1]);
        let bytes = model.tensor_bytes(&e.name).expect("bytes");
        let (params_off, _, _) = cortiq_core::quant::q4tp_sections(rows, cols);
        let params = &bytes[params_off..params_off + rows * 4];
        for r in 0..rows {
            let tab = cortiq_core::quant::q4tp_ladder(params, r);
            let lo = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([params[r * 4], params[r * 4 + 1]]));
            let st = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([params[r * 4 + 2], params[r * 4 + 3]]));
            for (c, &t) in tab.iter().enumerate() {
                let closed = (lo + c as f32 * st).exp2();
                if t == 0.0 && closed == 0.0 {
                    continue;
                }
                let d = ((closed - t) as f64 / (t as f64).abs().max(1e-30)).abs();
                if d > worst {
                    worst = d;
                    worst_row = format!("{} строка {r} ступень {c}: {t:e} против {closed:e}, lo={lo} шаг={st}", e.name);
                }
            }
        }
        checked += 1;
        if checked >= 8 {
            break;
        }
    }
    println!("худшее относительное расхождение лестниц: {worst:.3e}");
    println!("  {worst_row}");

    // And the other half of the question, which needs no device at all: does
    // the quantized matvec agree with the format's own dequantizer?
    let mut checked = 0;
    for e in model.tensors.iter() {
        if e.dtype != cortiq_core::TensorDtype::Q4TiledP || e.shape.len() != 2 {
            continue;
        }
        let (rows, cols) = (e.shape[0], e.shape[1]);
        if cols % 32 != 0 {
            continue;
        }
        let xs: Vec<f32> = (0..cols).map(|i| ((i * 7) as f32 * 0.013).sin()).collect();
        let t = cortiq_engine::qtensor::QTensor::from_model(&model, &e.name).expect("t");
        let mut mv = vec![0.0f32; rows];
        t.matvec(&xs, &mut mv, None);
        let bytes = model.tensor_bytes(&e.name).expect("bytes");
        let mut w = vec![0.0f32; rows * cols];
        cortiq_core::quant::dequant_q4tp(bytes, rows, cols, &mut w);
        let plain: Vec<f32> = (0..rows)
            .map(|r| {
                w[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(&xs)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        let num: f32 = plain.iter().zip(&mv).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = plain.iter().map(|a| a * a).sum::<f32>().max(1e-20);
        println!("{}: [{rows}x{cols}] матвек↔деквант {:.3e}", e.name, (num / den).sqrt());
        checked += 1;
        if checked >= 6 {
            break;
        }
    }
}
