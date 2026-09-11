use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A single tensor is assembled in memory before the native codec consumes it.
/// The Qwen2.5-VL embedding is about 1.1 GiB, so the per-tensor ceiling must
/// cover that real source. The prefetcher still targets 256 MiB per batch and
/// holds at most one current batch plus one queued batch.
const MAX_RANGE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_BATCH_BYTES: usize = 256 * 1024 * 1024;

fn content_range(value: &str) -> anyhow::Result<(usize, usize, usize)> {
    let (unit, value) = value
        .split_once(' ')
        .ok_or_else(|| anyhow::anyhow!("tensor server returned invalid Content-Range"))?;
    anyhow::ensure!(
        unit == "bytes",
        "tensor server returned invalid Content-Range"
    );
    let (span, total) = value
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("tensor server returned invalid Content-Range"))?;
    let (start, end) = span
        .split_once('-')
        .ok_or_else(|| anyhow::anyhow!("tensor server returned invalid Content-Range"))?;
    let start = start.parse::<usize>()?;
    let end = end.parse::<usize>()?;
    let total = total.parse::<usize>()?;
    anyhow::ensure!(
        total > 0 && start <= end && end < total,
        "tensor server returned an invalid byte span"
    );
    Ok((start, end, total))
}

/// Fetch a bounded interval, rejecting servers which silently return the whole file.
pub(crate) fn read_http_range(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    start: usize,
    end: usize,
    expected_total: Option<usize>,
) -> anyhow::Result<(Vec<u8>, usize)> {
    anyhow::ensure!(end > start, "empty HTTP range");
    anyhow::ensure!(
        end - start <= MAX_RANGE_BYTES,
        "HTTP range exceeds the 2 GiB limit"
    );
    let mut request = agent
        .get(url)
        .set("Range", &format!("bytes={start}-{}", end - 1))
        .set("Accept-Encoding", "identity");
    if url.starts_with("https://huggingface.co/") {
        if let Some(token) = token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
    }
    let response = request
        .call()
        .map_err(|e| anyhow::anyhow!("tensor range request failed: {e}"))?;
    anyhow::ensure!(
        response.status() == 206,
        "tensor server ignored byte range (HTTP {})",
        response.status()
    );
    let (got_start, got_end, total) =
        content_range(response.header("Content-Range").unwrap_or(""))?;
    anyhow::ensure!(
        total > 0 && got_start == start && got_end == (end.min(total) - 1),
        "tensor server returned a different byte range"
    );
    if let Some(expected) = expected_total {
        anyhow::ensure!(
            total == expected,
            "tensor source size changed during import"
        );
    }
    let expected_len = got_end
        .checked_sub(got_start)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("invalid tensor range length"))?;
    let mut bytes = Vec::with_capacity(expected_len.min(MAX_RANGE_BYTES));
    response
        .into_reader()
        .take(expected_len as u64 + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() == expected_len,
        "tensor range body length mismatch"
    );
    Ok((bytes, total))
}

/// Parallel, ordered prefetch bounded by eight requests and one queued batch.
pub(crate) fn with_http_ranges<T>(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    total: usize,
    ranges: Vec<std::ops::Range<usize>>,
    consume: impl FnOnce(&mut dyn FnMut() -> anyhow::Result<Vec<u8>>) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    // Limit each batch by both concurrency and raw size. One queued batch plus
    // one in flight is bounded; the consumer quantizes in original directory order.
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut batch_bytes = 0usize;
    for range in ranges {
        anyhow::ensure!(
            range.start < range.end && range.end <= total,
            "tensor range is outside the pinned source"
        );
        anyhow::ensure!(
            range.len() <= MAX_RANGE_BYTES,
            "tensor exceeds the 2 GiB streaming limit"
        );
        if !batch.is_empty()
            && (batch.len() == 8
                || batch_bytes
                    .checked_add(range.len())
                    .map_or(true, |n| n > MAX_BATCH_BYTES))
        {
            batches.push(std::mem::take(&mut batch));
            batch_bytes = 0;
        }
        batch_bytes += range.len();
        batch.push(range);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let producer_cancelled = cancelled.clone();
        scope.spawn(move || {
            for batch in batches {
                if producer_cancelled.load(Ordering::Acquire) {
                    break;
                }
                let worker_cancelled = producer_cancelled.clone();
                let result: anyhow::Result<Vec<Vec<u8>>> = std::thread::scope(|workers| {
                    let jobs: Vec<_> = batch
                        .into_iter()
                        .map(|r| {
                            let worker_cancelled = worker_cancelled.clone();
                            workers.spawn(move || {
                                if worker_cancelled.load(Ordering::Acquire) {
                                    return Err(anyhow::anyhow!("tensor range producer cancelled"));
                                }
                                read_http_range(agent, url, token, r.start, r.end, Some(total))
                                    .map(|(b, _)| b)
                            })
                        })
                        .collect();
                    jobs.into_iter()
                        .map(|j| {
                            j.join()
                                .map_err(|_| anyhow::anyhow!("tensor range worker panicked"))?
                        })
                        .collect()
                });
                let failed = result.is_err();
                if producer_cancelled.load(Ordering::Acquire) {
                    break;
                }
                if tx.send(result).is_err() || failed {
                    break;
                }
            }
        });
        let mut pending = std::vec::IntoIter::default();
        let result = consume(&mut || {
            if pending.len() == 0 {
                pending = rx
                    .recv()
                    .map_err(|_| anyhow::anyhow!("tensor range reader ended early"))??
                    .into_iter();
            }
            pending
                .next()
                .ok_or_else(|| anyhow::anyhow!("empty tensor range batch"))
        });
        cancelled.store(true, Ordering::Release);
        drop(rx);
        result
    })
}

#[cfg(test)]
mod tests {
    use super::content_range;

    #[test]
    fn content_range_requires_a_single_bounded_bytes_span() {
        assert_eq!(content_range("bytes 4-9/20").unwrap(), (4, 9, 20));
        for value in [
            "",
            "items 4-9/20",
            "bytes 9-4/20",
            "bytes 4-20/20",
            "bytes */20",
            "bytes 4-9/*",
        ] {
            assert!(
                content_range(value).is_err(),
                "{value:?} should be rejected"
            );
        }
    }

    #[test]
    #[ignore = "live HF probe; run explicitly when network access is available"]
    fn pinned_hf_small_ranges_follow_redirects_and_validate_lengths() {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(30))
            .timeout_read(std::time::Duration::from_secs(30))
            .build();
        let url = "https://huggingface.co/Qwen/Qwen-Image-Edit-2509/resolve/983d8d220ec4cf16278ef80bf3f30fe0378c8263/vae/diffusion_pytorch_model.safetensors";
        let (first, total) = super::read_http_range(&agent, url, None, 0, 8, None).unwrap();
        let (second, total_again) =
            super::read_http_range(&agent, url, None, 8, 24, Some(total)).unwrap();
        assert_eq!(first.len(), 8);
        assert_eq!(second.len(), 16);
        assert_eq!(total, 253_806_966);
        assert_eq!(total_again, total);
    }
}
