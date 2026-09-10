#!/usr/bin/env python3
"""Download the pinned Issue #10 GGUF with bounded parallel pwrite.

The file is public, so this script needs no token. It uses HTTP byte ranges,
eight workers by default, a pre-sized ``.partial`` file, and a small receipt
per completed range. A completed range is rehashed before it is reused. The
free-space gate subtracts bytes already allocated by a resumable partial file,
and the final file is renamed into place only after the whole-file SHA-256
matches. The model and partial files should live in a runner temporary
directory; the receipt is the only file intended for artifact upload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import http.client
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request


URL = (
    "https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF/resolve/"
    "84a3006979126011422eeeefe0c9485ddf431ef5/"
    "Qwen-Image-Edit-2509-Q6_K.gguf"
)
EXPECTED_SIZE = 16_824_990_240
EXPECTED_SHA256 = "ec5694f11a2908c10ef5324c50c79b1bb433547a39a211996551417b4b16f0ce"
CHUNK_SIZE = 64 * 1024**2
IO_SIZE = 8 * 1024**2
MIN_FREE_BYTES = 45_000_000_000
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def hash_range(fd: int, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    cursor = offset
    while remaining:
        block = os.pread(fd, min(IO_SIZE, remaining), cursor)
        if not block:
            raise OSError(f"short partial file at offset {cursor}")
        digest.update(block)
        cursor += len(block)
        remaining -= len(block)
    return digest.hexdigest()


def hash_file(fd: int, size: int) -> str:
    return hash_range(fd, 0, size)


def hash_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(IO_SIZE):
            digest.update(block)
    return digest.hexdigest()


def allocated_bytes(path: pathlib.Path) -> int:
    """Return bytes already allocated by a resumable partial file."""
    try:
        info = path.stat()
    except FileNotFoundError:
        return 0
    blocks = getattr(info, "st_blocks", 0)
    if blocks:
        # POSIX st_blocks is measured in 512-byte units. On filesystems that
        # do not expose it (notably Windows), the logical size is conservative.
        return blocks * 512
    return info.st_size


def pwrite_all(fd: int, block: bytes, offset: int) -> None:
    view = memoryview(block)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise OSError("pwrite returned no progress")
        offset += written
        view = view[written:]


def chunk_receipt(path: pathlib.Path, offset: int, size: int, sha256: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "offset": offset,
        "size": size,
        "sha256": sha256,
        "completed_utc": utc_now(),
    }
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reusable_chunk(fd: int, receipt: pathlib.Path, offset: int, size: int) -> bool:
    try:
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved.get("offset") != offset or saved.get("size") != size:
            return False
        saved_sha = saved.get("sha256")
        if not isinstance(saved_sha, str) or len(saved_sha) != 64:
            return False
        return hash_range(fd, offset, size) == saved_sha
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def fetch_chunk(
    fd: int,
    url: str,
    offset: int,
    size: int,
    receipt: pathlib.Path,
    timeout: float,
    retries: int,
) -> dict[str, object]:
    if reusable_chunk(fd, receipt, offset, size):
        print(f"reused range={offset}:{offset + size}", flush=True)
        return {"offset": offset, "size": size, "status": "reused"}

    end = offset + size - 1
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept-Encoding": "identity",
                    "Range": f"bytes={offset}-{end}",
                    "User-Agent": "cmf-issue10-validation/1",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != http.client.PARTIAL_CONTENT:
                    raise OSError(
                        f"range {offset}:{end} returned HTTP {response.status}; expected 206"
                    )
                content_range = response.headers.get("Content-Range", "")
                match = CONTENT_RANGE.fullmatch(content_range)
                if not match or tuple(map(int, match.groups())) != (offset, end, EXPECTED_SIZE):
                    raise OSError(
                        f"bad Content-Range for {offset}:{end}: {content_range!r}"
                    )
                remaining = size
                cursor = offset
                digest = hashlib.sha256()
                while remaining:
                    block = response.read(min(IO_SIZE, remaining))
                    if not block:
                        raise OSError(f"short HTTP range at {cursor}")
                    pwrite_all(fd, block, cursor)
                    digest.update(block)
                    cursor += len(block)
                    remaining -= len(block)
                if response.read(1):
                    raise OSError(f"HTTP range {offset}:{end} returned extra bytes")
            sha256 = digest.hexdigest()
            chunk_receipt(receipt, offset, size, sha256)
            print(f"downloaded range={offset}:{offset + size} attempt={attempt}", flush=True)
            return {"offset": offset, "size": size, "status": "downloaded"}
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(f"range {offset}:{end} failed after {retries} attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_SIZE)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=6)
    return parser.parse_args()


def free_bytes(path: pathlib.Path) -> int:
    usage = os.statvfs(path)
    return usage.f_bavail * usage.f_frsize


def check_disk(path: pathlib.Path, reserved_bytes: int) -> tuple[int, int]:
    available = free_bytes(path)
    required = max(0, MIN_FREE_BYTES - reserved_bytes)
    if available < required:
        raise RuntimeError(
            f"insufficient free space at {path}: {available} bytes; "
            f"need at least {required} with {reserved_bytes} bytes already allocated"
        )
    return available, required


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.retries < 1 or args.timeout <= 0:
        raise SystemExit("--workers/--retries must be positive and --timeout must be > 0")
    if args.chunk_bytes < IO_SIZE:
        raise SystemExit(f"--chunk-bytes must be at least {IO_SIZE}")
    if args.chunk_bytes > EXPECTED_SIZE:
        raise SystemExit("--chunk-bytes cannot exceed the pinned file size")

    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = args.receipt.absolute()
    receipt.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        if output.stat().st_size != EXPECTED_SIZE:
            raise RuntimeError(f"refusing existing output with wrong size: {output}")
        actual = hash_path(output)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(f"refusing existing output with wrong SHA-256: {output}")
        payload = {
            "status": "already_verified",
            "url": URL,
            "revision": URL.split("/resolve/", 1)[1].split("/", 1)[0],
            "size": EXPECTED_SIZE,
            "sha256": EXPECTED_SHA256,
            "output": str(output),
            "free_bytes_at_verification": free_bytes(output.parent),
            "workers": args.workers,
            "chunk_bytes": args.chunk_bytes,
            "verified_utc": utc_now(),
        }
        receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 0

    partial = output.with_name(output.name + ".partial")
    state = partial.with_name(partial.name + ".state.json")
    state_payload = {
        "url": URL,
        "size": EXPECTED_SIZE,
        "sha256": EXPECTED_SHA256,
        "chunk_bytes": args.chunk_bytes,
    }
    if state.exists():
        saved_state = json.loads(state.read_text(encoding="utf-8"))
        if saved_state != state_payload:
            raise RuntimeError(f"partial download belongs to a different pinned input: {state}")
    else:
        state.write_text(json.dumps(state_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parts_dir = partial.with_name(partial.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    reserved_bytes = allocated_bytes(partial)
    available, required_free = check_disk(output.parent, reserved_bytes)
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(fd, EXPECTED_SIZE)
        chunks = [
            (offset, min(args.chunk_bytes, EXPECTED_SIZE - offset))
            for offset in range(0, EXPECTED_SIZE, args.chunk_bytes)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    fetch_chunk,
                    fd,
                    URL,
                    offset,
                    size,
                    parts_dir / f"{offset:012d}.json",
                    args.timeout,
                    args.retries,
                )
                for offset, size in chunks
            ]
            for future in futures:
                future.result()
        actual = hash_file(fd, EXPECTED_SIZE)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(f"assembled file SHA-256 mismatch: {actual}")
        os.fsync(fd)
    finally:
        os.close(fd)

    if output.exists():
        raise RuntimeError(f"output appeared while downloading: {output}")
    os.replace(partial, output)
    payload = {
        "status": "verified_download",
        "url": URL,
        "revision": URL.split("/resolve/", 1)[1].split("/", 1)[0],
        "size": EXPECTED_SIZE,
        "sha256": EXPECTED_SHA256,
        "output": str(output),
        "free_bytes_before": available,
        "required_free_bytes_before": required_free,
        "reserved_bytes_before": reserved_bytes,
        "minimum_effective_bytes": MIN_FREE_BYTES,
        "workers": args.workers,
        "chunk_bytes": args.chunk_bytes,
        "chunks": len(chunks),
        "verified_utc": utc_now(),
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_name(output.name + ".sha256").write_text(
        f"{EXPECTED_SHA256}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
