//! Hold the Metal q4tp matvec against the CPU kernel on a real model tensor.
//!
//!   cargo run --release --example q4tp_gpu_ab -- <model.cmf>

use cortiq_core::CmfModel;
use cortiq_core::types::TensorDtype;
use std::sync::Arc;

fn main() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    let path = std::env::args().nth(1).expect("usage: q4tp_gpu_ab <model.cmf>");
    let model = Arc::new(CmfModel::open_sharded(&path).expect("open"));

    let mut checked = 0usize;
    for (idx, e) in model.tensors.iter().enumerate() {
        if !matches!(e.dtype, TensorDtype::Q4TiledP | TensorDtype::Q4Tiled) || e.shape.len() != 2 {
            continue;
        }
        let tp = e.dtype == TensorDtype::Q4TiledP;
        let (rows, cols) = (e.shape[0] as usize, e.shape[1] as usize);
        if rows < 64 || cols % 32 != 0 || rows * cols > 40_000_000 {
            continue;
        }
        let xs: Vec<f32> = (0..cols)
            .map(|i| ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5)
            .collect();

        // CPU reference straight off the canonical scalar dequant.
        let bytes = model.entry_bytes(e);
        let mut w = vec![0f32; rows * cols];
        if tp {
            cortiq_core::quant::dequant_q4tp(bytes, rows, cols, &mut w);
        } else {
            cortiq_core::quant::dequant_q4_tiled(bytes, &mut w);
        }
        let want: Vec<f32> = (0..rows)
            .map(|r| (0..cols).map(|c| w[r * cols + c] * xs[c]).sum())
            .collect();

        let mut got = vec![0f32; rows];
        let call = |out: &mut [f32]| {
            if tp {
                cortiq_engine::gpu_metal::q4tp_matvec_for_test(&model, idx, &xs, rows, cols, out)
            } else {
                cortiq_engine::gpu_metal::q4t_matvec_for_test(&model, idx, &xs, rows, cols, out)
            }
        };
        if !call(&mut got) {
            eprintln!("{}: GPU refused", e.name);
            continue;
        }
        let best =
            cortiq_engine::gpu_metal::q4_matvec_bench(&model, idx, &xs, rows, cols, 64).unwrap();
        let (mut worst, mut at) = (0f32, 0usize);
        for r in 0..rows {
            let mag: f32 = (0..cols).map(|c| (w[r * cols + c] * xs[c]).abs()).sum();
            let rel = (got[r] - want[r]).abs() / mag.max(1e-9);
            if rel > worst {
                worst = rel;
                at = r;
            }
        }
        println!(
            "{:<48} {:>5} {}x{}  {:.3} мс  расхождение {:.1e}",
            e.name,
            if tp { "q4tp" } else { "q4t" },
            rows,
            cols,
            best * 1e3,
            worst
        );
        let _ = at;
        checked += 1;
        if checked >= 4 {
            break;
        }
    }
    if checked == 0 {
        eprintln!("в модели нет подходящих тензоров");
    }

    // Sweep over every eligible tensor, one dispatch each — the pattern a
    // token actually walks, with nothing left warm between tensors.
    let all: Vec<(usize, usize, usize)> = model
        .tensors
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            matches!(e.dtype, TensorDtype::Q4TiledP | TensorDtype::Q4Tiled)
                && e.shape.len() == 2
                && e.shape[1] as usize % 32 == 0
        })
        .map(|(i, e)| (i, e.shape[0] as usize, e.shape[1] as usize))
        .collect();
    let bytes: usize = all
        .iter()
        .map(|&(i, r, c)| {
            cortiq_core::quant::expected_nbytes(model.tensors[i].dtype, &[r, c]).unwrap_or(0)
        })
        .sum();
    for (serial, lbl) in [(false, "перекрытый"), (true, "сериализованный")] {
        match cortiq_engine::gpu_metal::q4_matvec_sweep(&model, &all, serial) {
            Some(t) => println!(
                "проход {lbl:<16} по {} тензорам: {:.2} мс, {:.2} ГБ, {:.1} ГБ/с",
                all.len(),
                t * 1e3,
                bytes as f64 / 1e9,
                bytes as f64 / 1e9 / t
            ),
            None => eprintln!("проход отклонён"),
        }
    }
}
