//! Shared test helpers (owned by WP1; other test files use them read-only):
//! a `.npy`/`.npz` reader, the reader of the shipped v3 PH skills
//! (`artifacts/decision-v3-20260926/{ds}-product/cortiq.cmf`), the exact v3
//! signal composition from the stored Python features, JSONL rows, numpy's
//! pairwise summation and the test-split access log.
#![allow(dead_code)]

use cortiq_core::{CmfModel, TensorDtype};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Environment variable naming `artifacts/decision-v3-20260926`.
pub const V3_DIR_ENV: &str = "CORTIQ_DECISION_V3_DIR";
/// Environment variable naming the test-split access log: every open of a
/// test split is appended there (one JSON line: utc, file, purpose).
pub const TEST_ACCESS_LOG_ENV: &str = "CORTIQ_DECISION_TEST_ACCESS_LOG";
/// Without [`TEST_ACCESS_LOG_ENV`]: this file under `$CMFPUBLIC`.
pub const TEST_ACCESS_LOG_DEFAULT: &str = "artifacts/decision-v4-20260926/test-access.log";
/// The three shipped datasets in the spec's order.
pub const DATASETS: [&str; 3] = ["banking77", "clinc150", "massive"];

pub fn sha256_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

pub fn sha256_file(p: &Path) -> String {
    sha256_hex(&std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())))
}

/// `CORTIQ_DECISION_V3_DIR`, when set.
pub fn v3_dir() -> Option<PathBuf> {
    std::env::var_os(V3_DIR_ENV).map(PathBuf::from)
}

// ------------------------------------------------------------------ utc + access log

/// Current UTC time as `YYYY-MM-DDTHH:MM:SS.ffffffZ` (no chrono in this crate's dev deps).
pub fn utc_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after 1970");
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        d.subsec_micros()
    )
}

/// The test-split access log: `$CORTIQ_DECISION_TEST_ACCESS_LOG`, else
/// [`TEST_ACCESS_LOG_DEFAULT`] under `$CMFPUBLIC`; `None` when neither is set.
pub fn test_access_log() -> Option<PathBuf> {
    std::env::var_os(TEST_ACCESS_LOG_ENV)
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("CMFPUBLIC").map(|r| PathBuf::from(r).join(TEST_ACCESS_LOG_DEFAULT))
        })
}

/// Append one access record to [`test_access_log`] (O_APPEND). Call it before
/// opening any test split, test feature file or test ledger. Only the local
/// `#[ignore]` gates call it: it panics when no log is configured, because a
/// test split must never be read unlogged.
pub fn log_test_access(file: &Path, purpose: &str) {
    let log = test_access_log().unwrap_or_else(|| {
        panic!("set {TEST_ACCESS_LOG_ENV} (or CMFPUBLIC) before a test split is read")
    });
    let line = serde_json::json!({"utc": utc_now(), "file": file.display().to_string(), "purpose": purpose});
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .unwrap_or_else(|e| panic!("open {}: {e}", log.display()));
    writeln!(f, "{line}").expect("append test-access.log");
}

// ------------------------------------------------------------------ numpy semantics

/// numpy's pairwise summation of a contiguous f32 array (`np.add.reduce` on the
/// last axis): 8 accumulators up to 128 values, halves at multiples of 8.
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f32;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, rj) in r.iter_mut().enumerate() {
                *rj += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// `evaluate_v3/common.py::l2f32` on one row: `v / f32(np.linalg.norm(v))`
/// (squares in f32, pairwise sum, f32 sqrt; a zero norm becomes 1).
pub fn l2f32_numpy(v: &[f32]) -> Vec<f32> {
    let sq: Vec<f32> = v.iter().map(|x| x * x).collect();
    let mut n = pairwise_sum_f32(&sq).sqrt();
    if n == 0.0 {
        n = 1.0;
    }
    v.iter().map(|x| x / n).collect()
}

// ------------------------------------------------------------------ npy / npz

/// One numpy array: dtype descriptor, shape and raw little-endian C-order bytes.
#[derive(Clone, Debug)]
pub struct NpyArray {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl NpyArray {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    fn typed<const W: usize, T>(&self, descrs: &[&str], f: fn([u8; W]) -> T) -> Vec<T> {
        assert!(
            descrs.contains(&self.descr.as_str()),
            "dtype {} is not one of {descrs:?}",
            self.descr
        );
        assert_eq!(self.data.len(), self.len() * W, "npy payload size");
        self.data
            .chunks_exact(W)
            .map(|c| f(c.try_into().unwrap()))
            .collect()
    }

    pub fn f32(&self) -> Vec<f32> {
        self.typed(&["<f4"], f32::from_le_bytes)
    }

    pub fn f64(&self) -> Vec<f64> {
        self.typed(&["<f8"], f64::from_le_bytes)
    }

    pub fn u32(&self) -> Vec<u32> {
        self.typed(&["<u4"], u32::from_le_bytes)
    }

    pub fn i64(&self) -> Vec<i64> {
        self.typed(&["<i8"], i64::from_le_bytes)
    }

    /// Any integer dtype widened to i64.
    pub fn int(&self) -> Vec<i64> {
        match self.descr.as_str() {
            "<i8" => self.i64(),
            "<u4" => self.u32().into_iter().map(i64::from).collect(),
            "<i4" => self
                .typed(&["<i4"], i32::from_le_bytes)
                .into_iter()
                .map(i64::from)
                .collect(),
            "<u2" => self
                .typed(&["<u2"], u16::from_le_bytes)
                .into_iter()
                .map(i64::from)
                .collect(),
            d => panic!("not an integer dtype: {d}"),
        }
    }
}

/// Parse one `.npy` payload (format 1.0, 2.0 or 3.0; C order only).
pub fn parse_npy(bytes: &[u8]) -> NpyArray {
    assert!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "not an npy file"
    );
    let major = bytes[6];
    let (hlen, start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        ),
        v => panic!("npy version {v} unsupported"),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).expect("npy header utf-8");
    let field = |key: &str| -> &str {
        let k = format!("'{key}':");
        let i = header
            .find(&k)
            .unwrap_or_else(|| panic!("npy header lacks {key}: {header}"));
        header[i + k.len()..].trim_start()
    };
    let descr = field("descr");
    let descr = descr[1..].split('\'').next().unwrap().to_string();
    assert!(
        field("fortran_order").starts_with("False"),
        "Fortran-order npy unsupported"
    );
    let shape_s = field("shape");
    let shape_s = &shape_s[1..shape_s.find(')').unwrap()];
    let shape: Vec<usize> = shape_s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    NpyArray {
        descr,
        shape,
        data: bytes[start + hlen..].to_vec(),
    }
}

pub fn read_npy(path: &Path) -> NpyArray {
    parse_npy(&std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())))
}

fn le16(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes([b[o], b[o + 1]]) as usize
}

fn le32(b: &[u8], o: usize) -> u64 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as u64
}

fn le64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// Read an `.npz` written by `np.savez` (ZIP, stored members, zip64 extras
/// honoured). Keys are the member names without `.npy`.
pub fn read_npz(path: &Path) -> BTreeMap<String, NpyArray> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let eocd = (0..=b.len().saturating_sub(22))
        .rev()
        .find(|&i| b[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .expect("zip end of central directory");
    let mut entries = le16(&b, eocd + 10) as u64;
    let mut cd = le32(&b, eocd + 16);
    if entries == 0xffff || cd == 0xffff_ffff {
        // zip64 end of central directory locator precedes the EOCD.
        let loc = eocd - 20;
        assert_eq!(b[loc..loc + 4], [0x50, 0x4b, 0x06, 0x07], "zip64 locator");
        let z64 = le64(&b, loc + 8) as usize;
        assert_eq!(b[z64..z64 + 4], [0x50, 0x4b, 0x06, 0x06], "zip64 eocd");
        entries = le64(&b, z64 + 32);
        cd = le64(&b, z64 + 48);
    }
    let mut out = BTreeMap::new();
    let mut p = cd as usize;
    for _ in 0..entries {
        assert_eq!(
            b[p..p + 4],
            [0x50, 0x4b, 0x01, 0x02],
            "central directory header"
        );
        let method = le16(&b, p + 10);
        assert_eq!(method, 0, "compressed npz members are not supported");
        let mut csize = le32(&b, p + 20);
        let mut usize_ = le32(&b, p + 24);
        let (nlen, xlen, clen) = (le16(&b, p + 28), le16(&b, p + 30), le16(&b, p + 32));
        let mut off = le32(&b, p + 42);
        let name = String::from_utf8(b[p + 46..p + 46 + nlen].to_vec()).unwrap();
        // zip64 extra field: the 0xFFFFFFFF fields, in order usize, csize, offset.
        let mut x = p + 46 + nlen;
        let xend = x + xlen;
        while x + 4 <= xend {
            let (id, sz) = (le16(&b, x), le16(&b, x + 2));
            if id == 1 {
                let mut q = x + 4;
                if usize_ == 0xffff_ffff {
                    usize_ = le64(&b, q);
                    q += 8;
                }
                if csize == 0xffff_ffff {
                    csize = le64(&b, q);
                    q += 8;
                }
                if off == 0xffff_ffff {
                    off = le64(&b, q);
                }
            }
            x += 4 + sz;
        }
        assert_eq!(csize, usize_, "stored member sizes differ");
        let l = off as usize;
        assert_eq!(b[l..l + 4], [0x50, 0x4b, 0x03, 0x04], "local file header");
        let data = l + 30 + le16(&b, l + 26) + le16(&b, l + 28);
        let arr = parse_npy(&b[data..data + csize as usize]);
        out.insert(name.trim_end_matches(".npy").to_string(), arr);
        p += 46 + nlen + xlen + clen;
    }
    out
}

/// A CSR matrix of hashed features as `research_v3/hashfeat.py` stores it,
/// expanded to dense f32 rows (`dense_from_sparse`).
pub fn read_hash_npz_dense(path: &Path) -> (usize, Vec<f32>) {
    let z = read_npz(path);
    let dim = z["dim"].int()[0] as usize;
    let indptr = z["indptr"].int();
    let indices = z["indices"].int();
    let values = z["values"].f32();
    let n = indptr.len() - 1;
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        for k in indptr[i] as usize..indptr[i + 1] as usize {
            out[i * dim + indices[k] as usize] = values[k];
        }
    }
    (n, out)
}

// ------------------------------------------------------------------ data rows

/// One labelled row of a split file.
#[derive(Clone, Debug)]
pub struct Row {
    pub text: String,
    pub label: String,
}

impl Row {
    pub fn text_sha256(&self) -> String {
        sha256_hex(self.text.as_bytes())
    }
}

/// `{"text","label"}` rows of a JSONL file, in file order.
pub fn read_jsonl_rows(path: &Path) -> Vec<Row> {
    let s =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("jsonl row");
            Row {
                text: v["text"].as_str().expect("text").to_string(),
                label: v["label"].as_str().expect("label").to_string(),
            }
        })
        .collect()
}

// ------------------------------------------------------------------ v3 skills

/// One task of a shipped v3 skill.
#[derive(Clone, Debug)]
pub struct V3Task {
    pub id: usize,
    pub label: String,
    pub mean: Vec<f32>,
    /// `[rank × dim]`.
    pub basis: Vec<f32>,
    pub err_mean: f32,
    pub err_std: f32,
    pub n_train: usize,
    pub active: bool,
}

impl V3Task {
    pub fn rank(&self) -> usize {
        self.basis.len() / self.mean.len()
    }
}

/// A shipped v3 PH skill (`cortiq-decision-affine-core-v1`) and its policy.
#[derive(Clone, Debug)]
pub struct V3Skill {
    pub dir: PathBuf,
    pub dim: usize,
    pub k: usize,
    pub temperature: f32,
    pub novelty_theta: f32,
    /// `policy.json` `min_confidence` (the certified τ), when set.
    pub tau: Option<f32>,
    pub certified: bool,
    pub representation_id: String,
    pub tasks: Vec<V3Task>,
}

impl V3Skill {
    /// The tasks the v3 runtime scores (`active && !basis.is_empty()`), in order.
    pub fn active(&self) -> Vec<&V3Task> {
        self.tasks
            .iter()
            .filter(|t| t.active && !t.basis.is_empty())
            .collect()
    }
}

fn read_f32_tensor(c: &CmfModel, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let e = c.tensor(name)?;
    assert_eq!(e.dtype, TensorDtype::F32, "{name} is not F32");
    let b = c.tensor_bytes(name).expect("tensor bytes");
    Some((
        e.shape.clone(),
        b.chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect(),
    ))
}

/// Read `{v3}/{ds}-product/cortiq.cmf` and its `policy.json`.
pub fn read_v3_skill(v3: &Path, ds: &str) -> V3Skill {
    let dir = v3.join(format!("{ds}-product"));
    let c = CmfModel::open(dir.join("cortiq.cmf")).expect("open the v3 cortiq.cmf");
    assert!(c.verify().is_empty(), "v3 CMF integrity");
    let m: serde_json::Value =
        serde_json::from_slice(c.tensor_bytes("decision.manifest").expect("manifest")).unwrap();
    assert_eq!(m["profile"], "cortiq-decision-affine-core-v1");
    let seed = &m["metadata"];
    let dim = seed["input_dim"].as_u64().unwrap() as usize;
    let mut tasks = Vec::new();
    for (i, t) in seed["tasks"].as_array().unwrap().iter().enumerate() {
        let (ms, mean) = read_f32_tensor(&c, &format!("decision.task.{i}.mean")).expect("mean");
        assert_eq!(ms, vec![dim]);
        let basis = match read_f32_tensor(&c, &format!("decision.task.{i}.basis")) {
            Some((s, b)) => {
                assert_eq!(s[1], dim);
                b
            }
            None => Vec::new(),
        };
        tasks.push(V3Task {
            id: t["id"].as_u64().unwrap() as usize,
            label: t["label"].as_str().unwrap().to_string(),
            mean,
            basis,
            err_mean: t["err_mean"].as_f64().unwrap() as f32,
            err_std: t["err_std"].as_f64().unwrap() as f32,
            n_train: t["n_train"].as_u64().unwrap() as usize,
            active: t["active"].as_bool().unwrap(),
        });
    }
    let policy: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("policy.json")).expect("policy.json"))
            .unwrap();
    V3Skill {
        dim,
        k: seed["k"].as_u64().unwrap() as usize,
        temperature: seed["temperature"].as_f64().unwrap() as f32,
        novelty_theta: seed["novelty_theta"].as_f64().unwrap() as f32,
        tau: policy["min_confidence"].as_f64().map(|v| v as f32),
        certified: !policy["calibration_gate"].is_null(),
        representation_id: m["representation_id"].as_str().unwrap().to_string(),
        tasks,
        dir,
    }
}

/// The v3 PH signal of one selection split (train, dev or calibration),
/// rebuilt exactly as `evaluate_v3/common.py::arm_signal('PH')` +
/// `.astype(float32)` did for the shipped fit: `[l2f32(φ_P) ; 0.5·φ_H]` from
/// the stored ORT features and the stored Python-port hash features.
pub struct V3Split {
    pub rows: Vec<Row>,
    /// `n × (384 + 4096)`, row-major.
    pub x: Vec<f32>,
    pub dim: usize,
}

/// The split's source files are the ones `{ds}-product/build.json` recorded
/// (paths and sha256 are checked). Only train, dev and calibration exist here.
pub fn read_v3_split(v3: &Path, ds: &str, split: &str) -> V3Split {
    assert!(
        matches!(split, "train" | "dev" | "calibration"),
        "v3 stored hash features exist for train/dev/calibration only"
    );
    let build: serde_json::Value = serde_json::from_slice(
        &std::fs::read(v3.join(format!("{ds}-product/build.json"))).expect("build.json"),
    )
    .unwrap();
    let input = &build["inputs"][split];
    let path = PathBuf::from(input["path"].as_str().unwrap());
    assert_eq!(
        sha256_file(&path),
        input["sha256"].as_str().unwrap(),
        "{} changed",
        path.display()
    );
    let rows = read_jsonl_rows(&path);
    assert_eq!(rows.len() as u64, input["rows"].as_u64().unwrap());
    let feat = &build["product_features"][split];
    let npy = v3
        .join("features-product")
        .join(ds)
        .join(format!("{split}.npy"));
    assert_eq!(
        sha256_file(&npy),
        feat["sha256"].as_str().unwrap(),
        "{} changed",
        npy.display()
    );
    let p = read_npy(&npy);
    assert_eq!(p.shape, vec![rows.len(), 384]);
    // Row alignment is the v3 build's own: it checked the features against the
    // split file and recorded both sha256 values, verified above.
    let p = p.f32();
    let hdir = v3.join("hash-features").join(ds);
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(hdir.join(format!("manifest-{split}.json"))).expect("hash manifest"),
    )
    .unwrap();
    assert_eq!(
        manifest["input_sha256"].as_str().unwrap(),
        input["sha256"].as_str().unwrap()
    );
    let npz = hdir.join(format!("{split}.npz"));
    assert_eq!(sha256_file(&npz), manifest["npz_sha256"].as_str().unwrap());
    let (hn, h) = read_hash_npz_dense(&npz);
    assert_eq!(hn, rows.len());
    let hdim = h.len() / hn;
    assert_eq!(hdim, 4096);
    let dim = 384 + hdim;
    let mut x = Vec::with_capacity(rows.len() * dim);
    for i in 0..rows.len() {
        x.extend(l2f32_numpy(&p[i * 384..(i + 1) * 384]));
        // 0.5·φ_H in f64 then f32 is exact: the same bits as the f32 product.
        x.extend(h[i * hdim..(i + 1) * hdim].iter().map(|v| 0.5 * v));
    }
    V3Split { rows, x, dim }
}
