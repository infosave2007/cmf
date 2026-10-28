//! End to end: `teacher-data generate` → `compress` → a runnable file with a
//! report, on a real small model.
//!
//! Env-gated — it needs a real model and prompts:
//!   CMF_COMPRESS_TEACHER=teacher.cmf      (a .cmf with a chat template)
//!   CMF_COMPRESS_SOURCE=source            (HF dir or .cmf of the same tokenizer;
//!                                          default: the teacher)
//!   CMF_COMPRESS_PROMPTS=prompts.jsonl    (`calibration_prompts.jsonl` works)
//! e.g. Qwen3.5-0.8B: ~10 min on an M4 (24 prompts, 64-token answers).

use std::path::PathBuf;
use std::process::Command;

fn cortiq() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cortiq"))
}

#[test]
fn generate_compress_evaluate_resume() {
    let (Ok(teacher), Ok(prompts)) = (std::env::var("CMF_COMPRESS_TEACHER"), std::env::var("CMF_COMPRESS_PROMPTS")) else {
        eprintln!("skipped: set CMF_COMPRESS_TEACHER and CMF_COMPRESS_PROMPTS");
        return;
    };
    let source = std::env::var("CMF_COMPRESS_SOURCE").unwrap_or_else(|_| teacher.clone());
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("compress-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data = dir.join("task.tdata");
    let st = cortiq()
        .args(["teacher-data", "generate", &teacher, &prompts])
        .arg(&data)
        .args(["--no-think", "--max-tokens", "64", "--limit", "24", "--split", "60/20/20"])
        .status()
        .unwrap();
    assert!(st.success(), "teacher-data generate failed");
    let st = cortiq().args(["teacher-data", "inspect"]).arg(&data).arg("--verify").status().unwrap();
    assert!(st.success(), "inspect --verify failed");

    let out = dir.join("task.cmf");
    let run = |extra: &[&str]| {
        cortiq()
            .args(["compress", &source])
            .arg(&data)
            .arg(&out)
            .args(["q8_2f", "--ffn", "0.75", "--heal-epochs", "0.5", "--heal-layers", "deep2", "--eval-prompts", "4", "--eval-gen-tokens", "48"])
            .args(extra)
            .status()
            .unwrap()
    };
    let st = run(&[]);
    // a failed gate exits 5 and still writes the report; anything else is a bug
    assert!(st.success() || st.code() == Some(5), "compress failed: {st}");
    let work = PathBuf::from(format!("{}.work", out.display()));
    let report = work.join("S6.eval").join("report.md");
    assert!(report.exists(), "no report");
    let md = std::fs::read_to_string(&report).unwrap();
    for row in ["| source |", "| fold |", "| healed |", "| final |"] {
        assert!(md.contains(row), "report lacks {row}:\n{md}");
    }
    if st.success() {
        assert!(out.exists(), "gates passed but no output");
        let v = cortiq().arg("verify").arg(&out).status().unwrap();
        assert!(v.success(), "cortiq verify failed on the output");
    }
    // a second run resumes: every stage cached, the same output
    let before = std::fs::read(work.join("S5.quant").join("done.json")).unwrap();
    let st2 = run(&[]);
    assert_eq!(st.code(), st2.code());
    assert_eq!(before, std::fs::read(work.join("S5.quant").join("done.json")).unwrap(), "S5 recomputed on resume");
    let journal = std::fs::read_to_string(work.join("JOURNAL.md")).unwrap();
    assert!(journal.matches("cached").count() >= 7, "resume did not reuse the stages:\n{journal}");
    std::fs::remove_dir_all(&dir).ok();
}
