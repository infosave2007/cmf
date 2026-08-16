//! Corpus readers → text documents. jsonl(.gz) with a "text" field, plain
//! .txt (one document), and — with the `data` feature — parquet with a
//! "text" column (FineWeb / the-stack / fineweb-2).

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Iterate documents of a file, calling `f(text)` for each. Returns the
/// number of documents.
pub fn for_each_doc(path: &Path, mut f: impl FnMut(&str)) -> anyhow::Result<usize> {
    let name = path.to_string_lossy().to_string();
    let mut n = 0usize;
    if name.ends_with(".parquet") {
        #[cfg(feature = "data")]
        {
            use parquet::file::reader::{FileReader, SerializedFileReader};
            let file = std::fs::File::open(path)?;
            let reader = SerializedFileReader::new(file)?;
            let schema = reader.metadata().file_metadata().schema_descr();
            let col = (0..schema.num_columns())
                .find(|&i| schema.column(i).name() == "text" || schema.column(i).name() == "content")
                .ok_or_else(|| anyhow::anyhow!("{name}: no text/content column"))?;
            let proj = parquet::schema::types::Type::group_type_builder("schema")
                .with_fields(vec![std::sync::Arc::new(schema.column(col).self_type().clone())])
                .build()?;
            for row in reader.get_row_iter(Some(proj))? {
                let row = row?;
                if let Some((_, field)) = row.get_column_iter().next() {
                    if let parquet::record::Field::Str(s) = field {
                        f(s);
                        n += 1;
                    }
                }
            }
            return Ok(n);
        }
        #[cfg(not(feature = "data"))]
        anyhow::bail!("{name}: parquet needs `--features data`");
    }
    let file = std::fs::File::open(path)?;
    let reader: Box<dyn Read> = if name.ends_with(".gz") { Box::new(flate2::read::GzDecoder::new(file)) } else { Box::new(file) };
    if name.contains(".json") {
        let br = BufReader::with_capacity(1 << 20, reader);
        for line in br.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(t) = v.get("text").and_then(|t| t.as_str()).or_else(|| v.get("content").and_then(|t| t.as_str())) {
                f(t);
                n += 1;
            }
        }
    } else {
        let mut s = String::new();
        BufReader::new(reader).read_to_string(&mut s)?;
        f(&s);
        n = 1;
    }
    Ok(n)
}
