//! `cortiq-decision-rows-v1` blob of training and calibration rows (spec §2.5).
//!
//! The rows a skill was built from, kept as signal vectors (never text or a text
//! hash) for exact refits (old ∪ new rows), the self-learning holdout and
//! recertification. Little-endian, no padding:
//!
//! ```text
//! header  "CDRW" | u32 version = 1 | u32 n_rows | u32 dim_p | u32 dim_h | u32 0
//! row     u32 task | u8 split | u8 flags | u8 source | u8 0 | f32 weight | u32 nnz
//!         | f32 phi_p[dim_p] | u16 h_idx[nnz] (strictly ascending) | f32 h_val[nnz]
//! ```
//!
//! * `split`: 0 train, 1 calibration, 2 learned; the blob lists train rows (per
//!   task, in input-file order), then calibration rows (by sha256 of the text),
//!   then learned rows, so `split` never decreases;
//! * `flags` (calibration rows only): bit 0 odd half, bit 1 holdout;
//! * `source`: 0 data, 1 oracle, 2 client_feedback, 3 agreement (train and
//!   calibration rows are `data`);
//! * `weight`: finite and positive (recorded; the fit is unweighted);
//! * `h_idx`/`h_val`: the non-zero entries of `φ_H = hashfeat::dense(text, dim_h)`
//!   (an entry is kept when its bits are not `+0.0`, so the dense vector is
//!   restored bit for bit).
//!
//! The signal of a row is `[φ_P ; 0.5·φ_H]` (the product by 0.5 is exact).
//! `dim_p` is the encoder dimension (384 for the release encoder) and `dim_h` the
//! hashing dimension (4096); `dim_h ≤ 65,536` so every index fits a `u16`.

use anyhow::{Result, bail, ensure};

/// Layout name recorded in the skill manifest.
pub const LAYOUT: &str = "cortiq-decision-rows-v1";
/// Magic of the blob.
pub const MAGIC: [u8; 4] = *b"CDRW";
/// Format version.
pub const VERSION: u32 = 1;
/// Header length in bytes.
pub const HEADER_LEN: usize = 24;
/// Fixed part of a row before `phi_p` (task, split, flags, source, pad, weight, nnz).
pub const ROW_FIXED_LEN: usize = 16;
/// `flags` bit 0: the calibration row is in the odd half (it tests the gate).
pub const FLAG_ODD_HALF: u8 = 1;
/// `flags` bit 1: the calibration row is in the self-learning holdout.
pub const FLAG_HOLDOUT: u8 = 2;
/// Weight of the φ_H part in the signal `[φ_P ; 0.5·φ_H]`.
pub const PHI_H_WEIGHT: f32 = 0.5;
/// Largest `dim_h` whose indices fit a `u16`.
pub const MAX_DIM_H: usize = 1 << 16;

/// Which set a row belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Split {
    Train = 0,
    Calibration = 1,
    Learned = 2,
}

impl Split {
    pub fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            0 => Self::Train,
            1 => Self::Calibration,
            2 => Self::Learned,
            _ => bail!("unknown split {v}"),
        })
    }
}

/// Where the label of a row comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Source {
    Data = 0,
    Oracle = 1,
    ClientFeedback = 2,
    Agreement = 3,
}

impl Source {
    pub fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            0 => Self::Data,
            1 => Self::Oracle,
            2 => Self::ClientFeedback,
            3 => Self::Agreement,
            _ => bail!("unknown source {v}"),
        })
    }

    /// Recorded weight of an example from this source (spec §5.7).
    pub fn default_weight(self) -> f32 {
        match self {
            Self::Data | Self::Oracle => 1.0,
            Self::ClientFeedback => 3.0,
            Self::Agreement => 0.5,
        }
    }
}

/// One stored row.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// Task index within the skill.
    pub task: u32,
    pub split: Split,
    pub flags: u8,
    pub source: Source,
    pub weight: f32,
    /// `[dim_p]`.
    pub phi_p: Vec<f32>,
    /// Strictly ascending indices of the stored φ_H entries.
    pub h_idx: Vec<u16>,
    /// Values of the stored φ_H entries.
    pub h_val: Vec<f32>,
}

impl Row {
    /// A row from a dense φ_H (`phi_h.len()` = `dim_h`).
    pub fn from_dense(
        task: u32,
        split: Split,
        flags: u8,
        source: Source,
        weight: f32,
        phi_p: Vec<f32>,
        phi_h: &[f32],
    ) -> Self {
        let (h_idx, h_val) = sparse_from_dense(phi_h);
        Self {
            task,
            split,
            flags,
            source,
            weight,
            phi_p,
            h_idx,
            h_val,
        }
    }

    pub fn odd_half(&self) -> bool {
        self.flags & FLAG_ODD_HALF != 0
    }

    pub fn holdout(&self) -> bool {
        self.flags & FLAG_HOLDOUT != 0
    }

    /// Stored φ_H entries.
    pub fn nnz(&self) -> usize {
        self.h_idx.len()
    }

    /// The dense φ_H of `dim_h` values.
    pub fn phi_h_dense(&self, dim_h: usize) -> Vec<f32> {
        let mut h = vec![0.0f32; dim_h];
        for (&i, &v) in self.h_idx.iter().zip(&self.h_val) {
            h[i as usize] = v;
        }
        h
    }

    /// The signal `[φ_P ; 0.5·φ_H]` into `out` (`dim_p + dim_h` values).
    pub fn signal_into(&self, dim_h: usize, out: &mut [f32]) {
        let dp = self.phi_p.len();
        assert_eq!(out.len(), dp + dim_h, "signal buffer length");
        out[..dp].copy_from_slice(&self.phi_p);
        let tail = &mut out[dp..];
        tail.fill(0.0);
        for (&i, &v) in self.h_idx.iter().zip(&self.h_val) {
            tail[i as usize] = PHI_H_WEIGHT * v;
        }
    }

    /// The signal `[φ_P ; 0.5·φ_H]`.
    pub fn signal(&self, dim_h: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.phi_p.len() + dim_h];
        self.signal_into(dim_h, &mut out);
        out
    }

    /// Encoded length of this row.
    pub fn encoded_len(&self) -> usize {
        ROW_FIXED_LEN + 4 * self.phi_p.len() + 6 * self.h_idx.len()
    }

    /// Check the row against the blob dimensions and the row rules above.
    pub fn check(&self, dim_p: usize, dim_h: usize) -> Result<()> {
        ensure!(
            self.phi_p.len() == dim_p,
            "phi_p has {} values, the blob has dim_p {dim_p}",
            self.phi_p.len()
        );
        ensure!(
            self.phi_p.iter().all(|v| v.is_finite()),
            "phi_p must be finite"
        );
        ensure!(
            self.h_idx.len() == self.h_val.len(),
            "h_idx and h_val lengths differ ({} vs {})",
            self.h_idx.len(),
            self.h_val.len()
        );
        ensure!(
            self.h_idx.len() <= dim_h,
            "nnz {} exceeds dim_h {dim_h}",
            self.h_idx.len()
        );
        for w in self.h_idx.windows(2) {
            ensure!(
                w[0] < w[1],
                "h_idx must be strictly ascending ({} then {})",
                w[0],
                w[1]
            );
        }
        if let Some(&last) = self.h_idx.last() {
            ensure!(
                (last as usize) < dim_h,
                "h_idx {last} out of range (dim_h {dim_h})"
            );
        }
        ensure!(
            self.h_val.iter().all(|v| v.is_finite()),
            "h_val must be finite"
        );
        ensure!(
            self.weight.is_finite() && self.weight > 0.0,
            "weight must be finite and positive (got {})",
            self.weight
        );
        ensure!(
            self.flags & !(FLAG_ODD_HALF | FLAG_HOLDOUT) == 0,
            "unknown flag bits {:#04x}",
            self.flags
        );
        if self.split != Split::Calibration {
            ensure!(
                self.flags == 0,
                "flags are defined for calibration rows only"
            );
        }
        if self.split != Split::Learned {
            ensure!(
                self.source == Source::Data,
                "train and calibration rows have source data"
            );
        }
        Ok(())
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.task.to_le_bytes());
        out.push(self.split as u8);
        out.push(self.flags);
        out.push(self.source as u8);
        out.push(0);
        out.extend_from_slice(&self.weight.to_le_bytes());
        out.extend_from_slice(&(self.h_idx.len() as u32).to_le_bytes());
        for v in &self.phi_p {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for i in &self.h_idx {
            out.extend_from_slice(&i.to_le_bytes());
        }
        for v in &self.h_val {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// The stored entries of a dense φ_H: every index whose value is not `+0.0`
/// (bitwise), so [`Row::phi_h_dense`] restores the vector exactly.
pub fn sparse_from_dense(dense: &[f32]) -> (Vec<u16>, Vec<f32>) {
    assert!(dense.len() <= MAX_DIM_H, "dim_h {} > 65536", dense.len());
    let mut idx = Vec::new();
    let mut val = Vec::new();
    for (i, &v) in dense.iter().enumerate() {
        if v.to_bits() != 0 {
            idx.push(i as u16);
            val.push(v);
        }
    }
    (idx, val)
}

fn check_dims(dim_p: usize, dim_h: usize) -> Result<()> {
    ensure!(dim_p >= 1, "dim_p must be positive");
    ensure!(
        (1..=MAX_DIM_H).contains(&dim_h),
        "dim_h {dim_h} outside 1..=65536"
    );
    ensure!(dim_p <= u32::MAX as usize / 4, "dim_p {dim_p} too large");
    Ok(())
}

/// Incremental writer of a rows blob: rows are checked and appended as they come
/// (the order rule `split` non-decreasing is enforced), the header's row count is
/// patched by [`RowsWriter::finish`].
#[derive(Clone, Debug)]
pub struct RowsWriter {
    buf: Vec<u8>,
    n_rows: u32,
    dim_p: usize,
    dim_h: usize,
    last_split: Split,
    counts: [u64; 3],
}

impl RowsWriter {
    pub fn new(dim_p: usize, dim_h: usize) -> Result<Self> {
        check_dims(dim_p, dim_h)?;
        let mut buf = Vec::with_capacity(HEADER_LEN);
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(dim_p as u32).to_le_bytes());
        buf.extend_from_slice(&(dim_h as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        Ok(Self {
            buf,
            n_rows: 0,
            dim_p,
            dim_h,
            last_split: Split::Train,
            counts: [0; 3],
        })
    }

    /// Append one row.
    pub fn push(&mut self, row: &Row) -> Result<()> {
        row.check(self.dim_p, self.dim_h)
            .map_err(|e| anyhow::anyhow!("row {}: {e}", self.n_rows))?;
        ensure!(
            row.split >= self.last_split,
            "row {}: {:?} row after {:?} rows (order: train, calibration, learned)",
            self.n_rows,
            row.split,
            self.last_split
        );
        ensure!(self.n_rows < u32::MAX, "too many rows");
        self.last_split = row.split;
        self.counts[row.split as usize] += 1;
        row.encode_into(&mut self.buf);
        self.n_rows += 1;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.n_rows as usize
    }

    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    /// Rows per split: train, calibration, learned.
    pub fn counts(&self) -> [u64; 3] {
        self.counts
    }

    /// The finished blob.
    pub fn finish(mut self) -> Vec<u8> {
        self.buf[8..12].copy_from_slice(&self.n_rows.to_le_bytes());
        self.buf
    }
}

/// A decoded rows blob.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows {
    pub dim_p: usize,
    pub dim_h: usize,
    pub rows: Vec<Row>,
}

impl Rows {
    pub fn new(dim_p: usize, dim_h: usize) -> Result<Self> {
        check_dims(dim_p, dim_h)?;
        Ok(Self {
            dim_p,
            dim_h,
            rows: Vec::new(),
        })
    }

    /// Rows of one split.
    pub fn count(&self, split: Split) -> usize {
        self.rows.iter().filter(|r| r.split == split).count()
    }

    /// Rows of one split and task.
    pub fn count_task(&self, split: Split, task: u32) -> usize {
        self.rows
            .iter()
            .filter(|r| r.split == split && r.task == task)
            .count()
    }

    /// Encode (every row checked, order enforced).
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut w = RowsWriter::new(self.dim_p, self.dim_h)?;
        for r in &self.rows {
            w.push(r)?;
        }
        Ok(w.finish())
    }

    /// Decode and check a blob: magic, version, reserved fields, every row, the
    /// split order and the exact length.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let header = read_header(bytes)?;
        let mut rows = Vec::with_capacity(header.n_rows.min(1 << 20));
        let mut last = Split::Train;
        let mut at = HEADER_LEN;
        for n in 0..header.n_rows {
            let (row, next) = decode_row(bytes, at, header.dim_p, header.dim_h)
                .map_err(|e| anyhow::anyhow!("row {n}: {e}"))?;
            ensure!(
                row.split >= last,
                "row {n}: {:?} row after {:?} rows (order: train, calibration, learned)",
                row.split,
                last
            );
            last = row.split;
            rows.push(row);
            at = next;
        }
        ensure!(
            at == bytes.len(),
            "{} trailing bytes after {} rows",
            bytes.len() - at,
            header.n_rows
        );
        Ok(Self {
            dim_p: header.dim_p,
            dim_h: header.dim_h,
            rows,
        })
    }
}

/// The fixed header of a blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowsHeader {
    pub n_rows: usize,
    pub dim_p: usize,
    pub dim_h: usize,
}

/// Parse and check the 24-byte header.
pub fn read_header(bytes: &[u8]) -> Result<RowsHeader> {
    ensure!(
        bytes.len() >= HEADER_LEN,
        "rows blob shorter than its {HEADER_LEN}-byte header"
    );
    ensure!(bytes[..4] == MAGIC, "rows blob magic is not CDRW");
    let u = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().expect("4 bytes"));
    ensure!(u(4) == VERSION, "rows blob version {} (expected 1)", u(4));
    ensure!(u(20) == 0, "rows blob reserved header field is not 0");
    let (dim_p, dim_h) = (u(12) as usize, u(16) as usize);
    check_dims(dim_p, dim_h)?;
    Ok(RowsHeader {
        n_rows: u(8) as usize,
        dim_p,
        dim_h,
    })
}

fn decode_row(bytes: &[u8], at: usize, dim_p: usize, dim_h: usize) -> Result<(Row, usize)> {
    let need = |len: usize| -> Result<usize> {
        let end = at
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("length overflow"))?;
        ensure!(end <= bytes.len(), "truncated rows blob");
        Ok(end)
    };
    let fixed_end = need(ROW_FIXED_LEN)?;
    let b = &bytes[at..fixed_end];
    let task = u32::from_le_bytes(b[0..4].try_into().expect("4 bytes"));
    let split = Split::from_u8(b[4])?;
    let flags = b[5];
    let source = Source::from_u8(b[6])?;
    ensure!(b[7] == 0, "reserved row byte is not 0");
    let weight = f32::from_le_bytes(b[8..12].try_into().expect("4 bytes"));
    let nnz = u32::from_le_bytes(b[12..16].try_into().expect("4 bytes")) as usize;
    ensure!(nnz <= dim_h, "nnz {nnz} exceeds dim_h {dim_h}");
    let end = need(ROW_FIXED_LEN + 4 * dim_p + 6 * nnz)?;
    let mut o = fixed_end;
    let f32s = |o: &mut usize, n: usize| -> Vec<f32> {
        let v = bytes[*o..*o + 4 * n]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        *o += 4 * n;
        v
    };
    let phi_p = f32s(&mut o, dim_p);
    let h_idx: Vec<u16> = bytes[o..o + 2 * nnz]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().expect("2 bytes")))
        .collect();
    o += 2 * nnz;
    let h_val = f32s(&mut o, nnz);
    debug_assert_eq!(o, end);
    let row = Row {
        task,
        split,
        flags,
        source,
        weight,
        phi_p,
        h_idx,
        h_val,
    };
    row.check(dim_p, dim_h)?;
    Ok((row, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_is_phi_p_then_half_phi_h() {
        let r = Row::from_dense(
            0,
            Split::Train,
            0,
            Source::Data,
            1.0,
            vec![1.0, 2.0],
            &[0.0, 0.5, 0.0, -0.25],
        );
        assert_eq!(r.h_idx, vec![1, 3]);
        assert_eq!(r.signal(4), vec![1.0, 2.0, 0.0, 0.25, 0.0, -0.125]);
        assert_eq!(r.phi_h_dense(4), vec![0.0, 0.5, 0.0, -0.25]);
    }

    #[test]
    fn minus_zero_survives_the_sparse_form() {
        let dense = [0.0f32, -0.0, 1.0];
        let r = Row::from_dense(0, Split::Train, 0, Source::Data, 1.0, vec![0.0], &dense);
        let back = r.phi_h_dense(3);
        assert_eq!(
            back.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            dense.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
