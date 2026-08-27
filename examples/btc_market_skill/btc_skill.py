#!/usr/bin/env python3
"""One-year crypto experiment for a native Qwen3.5-2B CMF skill.

This is deliberately a next-hour classifier, not a trading oracle.  Every
sample uses 48 closed hourly candles, every next candle is labelled UP/DOWN,
splits are chronological, and the validation/test gaps are purged.  Native
``cortiq skill bake`` learns a DTG-MA mask, runs FCD and physically
defragments the FFN.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


MODEL_NAME = "Qwen/Qwen3.5-2B"
BINANCE_DATA_API = "https://data-api.binance.vision"
INTERVAL = "1h"
HOUR_MS = 3_600_000
CLASS_NAMES = ("DOWN", "UP")
DIRECT_ANSWER_DIRECTIVE = (
    "Answer directly and concisely. Do NOT reason, think step-by-step, or explain your "
    "process. Output ONLY the final answer."
)


def class_codes(class_names: Sequence[str]) -> dict[str, str]:
    # Qwen3.5 tokenizes DOWN and UP as one token each. Keeping the target
    # human-readable also lets native skill bake focus its loss on exactly
    # these answer positions instead of reconstructing ~1000 prompt tokens.
    return {name: name for name in class_names}


def system_prompt(class_names: Sequence[str], symbol: str = "BTCUSDT") -> str:
    del class_names
    return (
        f"Binary {symbol.upper()} market classifier. From causal features of 48 closed "
        "hourly candles, predict whether the next candle close is lower or higher. "
        "Output exactly DOWN or UP."
    )


def rendered_system_prompt(class_names: Sequence[str], symbol: str) -> str:
    """Exact system text used by ``cortiq serve`` with thinking disabled."""
    return f"{DIRECT_ANSWER_DIRECTIVE}\n\n{system_prompt(class_names, symbol)}"


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def parse_utc(value: str | None) -> pd.Timestamp:
    stamp = utc_now() if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")
    os.replace(tmp, path)


def request_json(url: str, payload: dict[str, Any] | None = None, retries: int = 5) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "cmf-btc-skill/1.0", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 == retries:
                detail = ""
                if isinstance(exc, urllib.error.HTTPError):
                    detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Запрос {url} не выполнен: {exc} {detail}") from exc
            time.sleep(min(2**attempt, 16))
    raise AssertionError("unreachable")


def fetch_klines(
    symbol: str, start: pd.Timestamp, end_exclusive: pd.Timestamp, api_base: str
) -> pd.DataFrame:
    """Download a complete interval through Binance's 1000-row pagination."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end_exclusive.timestamp() * 1000) - 1
    cursor, rows = start_ms, []
    while cursor <= end_ms:
        query = urllib.parse.urlencode(
            {
                "symbol": symbol.upper(),
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
        )
        page = request_json(f"{api_base.rstrip('/')}/api/v3/klines?{query}")
        if not isinstance(page, list):
            raise RuntimeError(f"Неожиданный ответ Binance: {page!r}")
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][0]) + HOUR_MS
        if next_cursor <= cursor:
            raise RuntimeError("Binance вернул неперемещающийся курсор")
        cursor = next_cursor
    if not rows:
        raise RuntimeError("Binance не вернул свечи")

    columns = [
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "unused",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.drop_duplicates("open_time_ms").sort_values("open_time_ms")
    frame = frame[(frame.open_time_ms >= start_ms) & (frame.close_time_ms <= end_ms)].copy()
    frame["timestamp"] = pd.to_datetime(frame.open_time_ms, unit="ms", utc=True)
    numeric = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "taker_buy_volume", "taker_buy_quote_volume",
    ]
    frame[numeric] = frame[numeric].astype(float)
    frame["trades"] = frame.trades.astype(int)
    frame = frame[
        ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
         "trades", "taker_buy_volume", "taker_buy_quote_volume"]
    ].reset_index(drop=True)
    validate_hourly(frame)
    return frame


def validate_hourly(frame: pd.DataFrame) -> None:
    if frame.empty or frame.timestamp.duplicated().any():
        raise ValueError("Пустые свечи или дубли времени")
    bad_steps = frame.timestamp.diff().dropna() != pd.Timedelta(hours=1)
    if bad_steps.any():
        raise ValueError(f"В данных есть {int(bad_steps.sum())} часовых разрывов")
    prices = frame[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("Цена должна быть положительной")
    if (frame.high < prices[["open", "close"]].max(axis=1)).any():
        raise ValueError("Некорректный high")
    if (frame.low > prices[["open", "close"]].min(axis=1)).any():
        raise ValueError("Некорректный low")


def command_download(args: argparse.Namespace) -> None:
    output = Path(args.output)
    boundary = parse_utc(args.end).floor("h")
    target_start = boundary - pd.Timedelta(days=args.days)
    fetch_start = target_start - pd.Timedelta(hours=args.warmup_bars)
    print(f"Binance {args.symbol}: {fetch_start.isoformat()} .. {boundary.isoformat()}")
    frame = fetch_klines(args.symbol, fetch_start, boundary, args.api_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, output)
    manifest = {
        "schema": 1,
        "source": "Binance Spot market-data-only /api/v3/klines",
        "api_base": args.api_base,
        "symbol": args.symbol.upper(),
        "interval": INTERVAL,
        "target_start_utc": target_start.isoformat(),
        "end_exclusive_utc": boundary.isoformat(),
        "warmup_bars": args.warmup_bars,
        "rows": len(frame),
        "csv_sha256": sha256_file(output),
        "created_utc": utc_now().isoformat(),
    }
    atomic_json(manifest_path(output), manifest)
    print(f"OK: {len(frame)} строк; {int((frame.timestamp >= target_start).sum())} за год")


def read_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    validate_hourly(frame)
    return frame


def add_features(frame: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """All rolling features are causal: current and earlier rows only."""
    out = frame.copy()
    previous = out.close.shift(1)
    for column in ("open", "high", "low", "close"):
        out[f"d_{column}_bps"] = (out[column] / previous - 1) * 10_000
    log_volume = np.log1p(out.volume)
    mean = log_volume.rolling(168, min_periods=24).mean()
    std = log_volume.rolling(168, min_periods=24).std().replace(0, np.nan)
    out["volume_z"] = ((log_volume - mean) / std).clip(-6, 6)
    out["taker_buy_ratio"] = (out.taker_buy_volume / out.volume.replace(0, np.nan)).clip(0, 1)

    delta = out.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema = {n: out.close.ewm(span=n, adjust=False).mean() for n in (8, 12, 21, 26, 55, 200)}
    out["macd_bps"] = (ema[12] - ema[26]) / out.close * 10_000
    out["ema8_21_bps"] = (ema[8] / ema[21] - 1) * 10_000
    out["ema21_55_bps"] = (ema[21] / ema[55] - 1) * 10_000
    out["close_ema200_bps"] = (out.close / ema[200] - 1) * 10_000
    true_range = pd.concat(
        [out.high - out.low, (out.high - previous).abs(), (out.low - previous).abs()], axis=1
    ).max(axis=1)
    out["atr14_bps"] = (
        true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / out.close * 10_000
    )
    typical = (out.high + out.low + out.close) / 3
    typical_mean = typical.rolling(20).mean()
    mean_deviation = typical.rolling(20).apply(
        lambda values: float(np.mean(np.abs(values - values.mean()))), raw=True
    ).replace(0, np.nan)
    out["cci20"] = ((typical - typical_mean) / (0.015 * mean_deviation)).clip(-400, 400)
    out["roc10_bps"] = out.close.pct_change(10) * 10_000
    log_return = np.log(out.close / previous)
    out["hv20_bps"] = log_return.rolling(20).std() * np.sqrt(24) * 10_000
    close_std20 = out.close.rolling(20).std().replace(0, np.nan)
    out["bb_position"] = ((out.close - out.close.rolling(20).mean()) / (2 * close_std20)).clip(-3, 3)
    support = out.low.rolling(48).min()
    resistance = out.high.rolling(48).max()
    out["sr_position"] = ((out.close - support) / (resistance - support).replace(0, np.nan)).clip(0, 1)
    out["body_bps"] = (out.close - out.open) / out.close * 10_000
    out["upper_wick_bps"] = (out.high - out[["open", "close"]].max(axis=1)) / out.close * 10_000
    out["lower_wick_bps"] = (out[["open", "close"]].min(axis=1) - out.low) / out.close * 10_000
    out["range_bps"] = (out.high - out.low) / out.close * 10_000
    for period in sorted({1, 3, 6, 12, 24, 48, *horizons}):
        out[f"ret_{period}h_bps"] = out.close.pct_change(period) * 10_000
    return out


def make_prompt(
    frame: pd.DataFrame, end: int, window: int, horizon: int, symbol: str = "BTCUSDT"
) -> str:
    if horizon != 1:
        raise ValueError("Этот воспроизводимый опыт фиксирует горизонт ровно 1 час")
    last = frame.iloc[end]
    stamp = pd.Timestamp(last.timestamp)
    return (
        f"{symbol.upper()} 1h causal summary of {window} closed bars. "
        f"r1={last.ret_1h_bps:+.0f} r3={last.ret_3h_bps:+.0f} "
        f"r6={last.ret_6h_bps:+.0f} r12={last.ret_12h_bps:+.0f} "
        f"r24={last.ret_24h_bps:+.0f} r48={last.ret_48h_bps:+.0f} "
        f"ema8_21={last.ema8_21_bps:+.0f} ema21_55={last.ema21_55_bps:+.0f} "
        f"ema200={last.close_ema200_bps:+.0f} macd={last.macd_bps:+.0f} "
        f"roc10={last.roc10_bps:+.0f} rsi14={last.rsi14:.0f} "
        f"cci20={last.cci20:+.0f} bb={last.bb_position:+.1f} sr={last.sr_position:.1f} "
        f"atr14={last.atr14_bps:.0f} hv20={last.hv20_bps:.0f} "
        f"volz={last.volume_z:+.1f} buy={last.taker_buy_ratio:.1f} "
        f"body={last.body_bps:+.0f} upper={last.upper_wick_bps:.0f} "
        f"lower={last.lower_wick_bps:.0f} range={last.range_bps:.0f} "
        f"hour={stamp.hour} weekday={stamp.dayofweek}. Next close direction:"
    )


def classify(move_bps: float) -> str:
    """Strict binary target: zero is deterministically assigned to DOWN."""
    return "UP" if move_bps > 0 else "DOWN"


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def evenly_spaced(records: Sequence[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if maximum <= 0 or len(records) <= maximum:
        return list(records)
    return [records[int(i)] for i in np.linspace(0, len(records) - 1, maximum, dtype=int)]


def balanced_records(
    records: Sequence[dict[str, Any]], maximum: int, class_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Round-robin classes; avoids baking the majority-class prior."""
    per_class = max(1, maximum // len(class_names))
    buckets = {
        name: evenly_spaced([r for r in records if r["label"] == name], per_class)
        for name in class_names
    }
    output: list[dict[str, Any]] = []
    for i in range(max(map(len, buckets.values()))):
        for name in class_names:
            if i < len(buckets[name]):
                output.append(buckets[name][i])
                if len(output) == maximum:
                    return output
    return output


def cmf_chat_record(record: dict[str, Any]) -> str:
    """Exact Qwen3.5 ChatML for ``enable_thinking=false``."""
    class_names = tuple(record["class_names"])
    return (
        f"<|im_start|>system\n{rendered_system_prompt(class_names, record['symbol'])}<|im_end|>\n"
        f"<|im_start|>user\n{record['prompt']}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        f"{record['code']}<|im_end|>\n"
    )


def write_corpus(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(cmf_chat_record(record) for record in records), "utf-8")
    os.replace(tmp, path)


def class_counts(
    records: Sequence[dict[str, Any]], class_names: Sequence[str]
) -> dict[str, int]:
    return {name: sum(r["label"] == name for r in records) for name in class_names}


def command_prepare(args: argparse.Namespace) -> None:
    source, output = Path(args.input), Path(args.output_dir)
    horizons = tuple(sorted(set(args.horizons)))
    if horizons != (1,):
        raise ValueError("Для статьи зафиксирован один горизонт: --horizons 1")
    class_names = CLASS_NAMES
    codes = class_codes(class_names)
    if args.train_ratio + args.validation_ratio >= 0.95:
        raise ValueError("Тестовая часть должна занимать хотя бы 5%")
    raw = read_candles(source)
    raw_manifest = (
        json.loads(manifest_path(source).read_text("utf-8"))
        if manifest_path(source).exists() else {}
    )
    symbol = str(raw_manifest.get("symbol", args.symbol)).upper()
    target_start = parse_utc(raw_manifest.get("target_start_utc", str(raw.timestamp.iloc[0])))
    frame = add_features(raw, horizons)
    required = [
        "d_open_bps", "d_high_bps", "d_low_bps", "d_close_bps", "volume_z",
        "taker_buy_ratio", "rsi14", "macd_bps", "atr14_bps", "ema8_21_bps",
        "ema21_55_bps", "close_ema200_bps", "ret_1h_bps", "ret_3h_bps",
        "ret_6h_bps", "ret_12h_bps", "ret_24h_bps", "ret_48h_bps", "cci20",
        "roc10_bps", "hv20_bps", "bb_position", "sr_position", "body_bps",
        "upper_wick_bps", "lower_wick_bps", "range_bps",
    ]
    max_h = max(horizons)
    anchors = []
    for end in range(args.window - 1, len(frame) - max_h, args.stride):
        if frame.timestamp.iloc[end] < target_start:
            continue
        if frame.iloc[end - args.window + 1 : end + 1][required].isna().any().any():
            continue
        anchors.append(end)
    if len(anchors) < 500:
        raise RuntimeError(f"Слишком мало валидных окон: {len(anchors)}")

    times = [frame.timestamp.iloc[i] for i in anchors]
    cut1 = times[int(len(times) * args.train_ratio)]
    cut2 = times[int(len(times) * (args.train_ratio + args.validation_ratio))]
    purge = pd.Timedelta(hours=args.window + max_h)
    split_indices = {
        "train": [i for i in anchors if frame.timestamp.iloc[i] < cut1],
        "validation": [
            i for i in anchors if cut1 + purge <= frame.timestamp.iloc[i] < cut2
        ],
        "test": [i for i in anchors if frame.timestamp.iloc[i] >= cut2 + purge],
    }
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, indices in split_indices.items():
        records = []
        for end in indices:
            last = frame.iloc[end]
            for horizon in horizons:
                move = (frame.close.iloc[end + horizon] / last.close - 1) * 10_000
                label = classify(move)
                records.append(
                    {
                        "timestamp": pd.Timestamp(last.timestamp).isoformat(),
                        "horizon": horizon,
                        "symbol": symbol,
                        "prompt": make_prompt(frame, end, args.window, horizon, symbol),
                        "label": label,
                        "code": codes[label],
                        "class_names": class_names,
                        "future_return_bps": round(float(move), 6),
                        "threshold_bps": 0.0,
                        "past_return_bps": round(float(last[f"ret_{horizon}h_bps"]), 6),
                        "close": round(float(last.close), 8),
                    }
                )
        records_by_split[split] = records
        write_jsonl(output / f"{split}.jsonl", records)

    bake_records = balanced_records(
        records_by_split["train"], args.cmf_corpus_samples, class_names
    )
    write_corpus(output / "cmf_corpus.txt", bake_records)
    for split in ("validation", "test"):
        write_corpus(
            output / f"{split}_corpus.txt",
            balanced_records(records_by_split[split], args.quality_corpus_samples, class_names),
        )
    metadata = {
        "schema": 4,
        "created_utc": utc_now().isoformat(),
        "source_csv": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "symbol": symbol,
        "window": args.window,
        "horizons": horizons,
        "classes": 2,
        "class_names": class_names,
        "stride": args.stride,
        "purge_hours": args.window + max_h,
        "boundaries": {"train_validation": cut1.isoformat(), "validation_test": cut2.isoformat()},
        "splits": {
            name: {
                "records": len(records), "anchors": len(split_indices[name]),
                "first_utc": records[0]["timestamp"], "last_utc": records[-1]["timestamp"],
                "class_counts": class_counts(records, class_names),
            }
            for name, records in records_by_split.items()
        },
        "cmf_corpus_records": len(bake_records),
        "target": "UP if close[t+1] > close[t], else DOWN",
    }
    atomic_json(output / "dataset_manifest.json", metadata)
    print(f"OK: {output}")
    for name, records in records_by_split.items():
        print(f"  {name:10s} {len(records):5d}  {class_counts(records, class_names)}")
    print(
        f"  purge={args.window + max_h} ч; "
        f"bake corpus balanced={class_counts(bake_records, class_names)}"
    )


def cortiq_binary(value: str) -> str:
    candidate = Path(value)
    if candidate.exists():
        return str(candidate.resolve())
    found = shutil.which(value)
    if not found:
        raise FileNotFoundError(f"Не найден cortiq: {value}. Сначала cargo build --release -p cortiq-cli")
    return found


def run_checked(command: Sequence[str], capture: bool = False) -> str:
    print("$", shlex.join(map(str, command)), flush=True)
    result = subprocess.run(command, check=True, text=True, capture_output=capture)
    if capture:
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result.stdout + result.stderr
    return ""


def run_logged(command: Sequence[str], log_path: Path) -> str:
    """Run a long command, stream progress and preserve an exact text log."""
    print("$", shlex.join(map(str, command)), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            lines.append(line)
        return_code = process.wait()
    output = "".join(lines)
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return output


def command_convert(args: argparse.Namespace) -> None:
    run_checked(
        [cortiq_binary(args.cortiq), "convert", "--model", args.model,
         "--quant", "q4tp", "--output", args.output]
    )
    run_checked([cortiq_binary(args.cortiq), "verify", args.output])
    run_checked([cortiq_binary(args.cortiq), "info", args.output])


def command_bake(args: argparse.Namespace) -> None:
    cortiq = cortiq_binary(args.cortiq)
    command = [
        cortiq, "skill", "bake", args.base, "--files", args.corpus,
        "--held-files", args.held_corpus,
        "--output", args.specialist, "--steps-a", str(args.steps_a),
        "--steps-b", str(args.steps_b), "--lr-a", str(args.lr_a),
        "--lr-b", str(args.lr_b), "--eval-every", str(args.eval_every),
        "--fcd-layers", str(args.fcd_layers),
        "--chunk", str(args.chunk), "--held", str(args.held),
        "--calib-chunks", str(args.calib_chunks), "--target-sparsity",
        str(args.target_sparsity), "--l1-aggression", str(args.l1_aggression),
        "--ffn-align", str(args.ffn_align),
    ]
    if args.focus_tokens:
        command.extend(["--focus-tokens", args.focus_tokens])
    if args.uniform_inter:
        command.append("--uniform-inter")
    bake_output = run_logged(command, Path(args.log))
    run_checked(
        [cortiq, "skill", "export", args.specialist, "--base", args.base,
         "--id", args.skill_id, "--name", args.skill_name, "--output", args.skill]
    )
    run_checked([cortiq, "verify", args.specialist])
    print_sizes({"q4tp base": args.base, "standalone skill": args.skill, "baked specialist": args.specialist})
    summary = re.search(
        r"=== bake: baseline ([0-9.]+) \| mask ([0-9.]+) \| "
        r"mask\+FCD ([0-9.]+) \| pruned ([0-9.]+)% \| ([0-9.]+)s",
        bake_output,
    )
    runtime_gate = re.search(
        r"runtime gate .*?: backbone ([0-9.]+) → specialist ([0-9.]+) \(([+-]?[0-9.]+)%\)",
        bake_output,
    )
    report: dict[str, Any] = {
        "schema": 1,
        "created_utc": utc_now().isoformat(),
        "command": list(map(str, command)),
        "base": {"path": args.base, "bytes": Path(args.base).stat().st_size},
        "specialist": {
            "path": args.specialist,
            "bytes": Path(args.specialist).stat().st_size,
            "sha256": sha256_file(Path(args.specialist)),
        },
        "standalone_skill": {
            "path": args.skill,
            "bytes": Path(args.skill).stat().st_size,
            "sha256": sha256_file(Path(args.skill)),
        },
        "log": str(Path(args.log)),
    }
    if summary:
        report["focused_bake"] = {
            "baseline_ppl": float(summary.group(1)),
            "best_mask_ppl": float(summary.group(2)),
            "mask_fcd_ppl": float(summary.group(3)),
            "pruned_percent": float(summary.group(4)),
            "seconds": float(summary.group(5)),
        }
    if runtime_gate:
        report["all_token_runtime_gate"] = {
            "backbone_ppl": float(runtime_gate.group(1)),
            "specialist_ppl": float(runtime_gate.group(2)),
            "delta_percent": float(runtime_gate.group(3)),
        }
    atomic_json(Path(args.report), report)
    print(f"Bake report: {args.report}")


def command_apply(args: argparse.Namespace) -> None:
    cortiq = cortiq_binary(args.cortiq)
    started = time.time()
    run_checked([cortiq, "skill", "apply", args.base, args.skill, "--output", args.output])
    run_checked([cortiq, "verify", args.output])
    print_sizes({"q4tp base": args.base, "standalone skill": args.skill, "applied specialist": args.output})
    print(f"  apply + verify: {time.time() - started:.1f} s")


def print_sizes(paths: dict[str, str]) -> None:
    for name, value in paths.items():
        path = Path(value)
        if path.exists():
            print(f"  {name:20s}: {path.stat().st_size / 1024**2:9.1f} MiB  {path}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def cmf_server(cortiq: str, model: str, log_path: Path) -> Iterator[str]:
    port = free_port()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [cortiq, "serve", model, "--host", "127.0.0.1", "--port", str(port)],
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 180
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"cortiq serve завершился; см. {log_path}")
                try:
                    request_json(f"{url}/v1/models", retries=1)
                    break
                except RuntimeError:
                    time.sleep(1)
            else:
                raise TimeoutError(f"cortiq serve не стартовал; см. {log_path}")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def predict_prompt_scored(
    base_url: str, prompt: str, class_names: Sequence[str], symbol: str
) -> tuple[str, str, float, dict[str, float]]:
    codes = class_codes(class_names)
    code_classes = {value: key for key, value in codes.items()}
    payload = {
        "model": "qwen35-btc-cmf",
        "messages": [
            {"role": "system", "content": system_prompt(class_names, symbol)},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 0,
        "enable_thinking": False,
        "cortiq": {"class_tokens": list(class_names)},
    }
    response = request_json(f"{base_url}/v1/chat/completions", payload, retries=2)
    text = str(response["choices"][0]["message"].get("content") or "").strip()
    classification = response.get("cortiq", {}).get("classification")
    if classification:
        scores = {
            str(row["token"]): float(row["probability"])
            for row in classification.get("scores", [])
        }
        return (
            str(classification["label"]),
            text,
            float(classification["confidence"]),
            scores,
        )
    allowed = "|".join(map(re.escape, code_classes))
    match = re.search(rf"(?<![A-Z])({allowed})(?![A-Z])", text.upper())
    return (code_classes[match.group(1)] if match else "INVALID", text, float("nan"), {})


def predict_prompt(
    base_url: str, prompt: str, class_names: Sequence[str], symbol: str
) -> tuple[str, str]:
    label, text, _, _ = predict_prompt_scored(base_url, prompt, class_names, symbol)
    return label, text


def select_evaluation(records: Sequence[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    horizons = sorted({int(r["horizon"]) for r in records})
    each = max(1, maximum // len(horizons))
    selected: list[dict[str, Any]] = []
    for horizon in horizons:
        selected.extend(evenly_spaced([r for r in records if r["horizon"] == horizon], each))
    return sorted(selected[:maximum], key=lambda r: (r["timestamp"], r["horizon"]))


def classification_metrics(
    truth: Sequence[str], prediction: Sequence[str], class_names: Sequence[str]
) -> dict[str, Any]:
    n = len(truth)
    confusion = {
        actual: {predicted: 0 for predicted in (*class_names, "INVALID")}
        for actual in class_names
    }
    for actual, predicted in zip(truth, prediction, strict=True):
        confusion[actual][predicted if predicted in class_names else "INVALID"] += 1
    per_class, recalls = {}, []
    for name in class_names:
        tp = confusion[name][name]
        fp = sum(confusion[other][name] for other in class_names if other != name)
        fn = sum(confusion[name][other] for other in (*class_names, "INVALID") if other != name)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
        recalls.append(recall)
    return {
        "n": n,
        "accuracy": sum(a == b for a, b in zip(truth, prediction, strict=True)) / max(1, n),
        "macro_f1": float(np.mean([per_class[name]["f1"] for name in class_names])),
        "balanced_accuracy": float(np.mean(recalls)),
        "invalid": sum(p not in class_names for p in prediction),
        "per_class": per_class,
        "confusion": confusion,
    }


def paired_block_bootstrap(
    records: Sequence[dict[str, Any]], base: Sequence[str], skill: Sequence[str], rounds: int = 2000
) -> dict[str, float]:
    groups: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        groups.setdefault(record["timestamp"], []).append(i)
    blocks = list(groups.values())
    rng, deltas = np.random.default_rng(42), []
    for _ in range(rounds):
        chosen = rng.integers(0, len(blocks), len(blocks))
        indices = [i for block in chosen for i in blocks[int(block)]]
        db = np.mean([skill[i] == records[i]["label"] for i in indices])
        da = np.mean([base[i] == records[i]["label"] for i in indices])
        deltas.append(float(db - da))
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {"delta_accuracy": float(np.mean(deltas)), "ci95_low": float(low), "ci95_high": float(high)}


def score_model(
    cortiq: str,
    model: str,
    records: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    log: Path,
) -> tuple[list[str], list[str], float]:
    predictions, raw, started = [], [], time.time()
    with cmf_server(cortiq, model, log) as url:
        for i, record in enumerate(records, 1):
            predicted, text = predict_prompt(
                url, record["prompt"], class_names, str(record.get("symbol", "BTCUSDT"))
            )
            predictions.append(predicted)
            raw.append(text)
            if i % 20 == 0 or i == len(records):
                print(f"  {Path(model).name}: {i}/{len(records)}", flush=True)
    return predictions, raw, time.time() - started


def ppl_value(cortiq: str, model: str, corpus: str, windows: int, window_len: int) -> float:
    output = run_checked(
        [cortiq, "ppl", model, "--file", corpus, "--windows", str(windows),
         "--window-len", str(window_len)], capture=True
    )
    matches = re.findall(r"PPL\s*=\s*([0-9.]+)", output)
    if not matches:
        raise RuntimeError("Не удалось прочитать PPL из cortiq")
    return float(matches[-1])


def command_evaluate(args: argparse.Namespace) -> None:
    cortiq = cortiq_binary(args.cortiq)
    data_dir, report_dir = Path(args.data_dir), Path(args.report_dir)
    train = read_jsonl(data_dir / "train.jsonl")
    evaluation = read_jsonl(data_dir / f"{args.split}.jsonl")
    metadata = json.loads((data_dir / "dataset_manifest.json").read_text("utf-8"))
    class_names = tuple(metadata.get("class_names", CLASS_NAMES))
    symbol = str(metadata.get("symbol", "BTCUSDT"))
    records = select_evaluation(evaluation, args.samples)
    truth = [r["label"] for r in records]
    majority_by_horizon = {}
    for horizon in sorted({r["horizon"] for r in train}):
        subset = [r for r in train if r["horizon"] == horizon]
        majority_by_horizon[horizon] = max(
            class_names, key=lambda name: sum(r["label"] == name for r in subset)
        )
    majority = [majority_by_horizon[r["horizon"]] for r in records]
    momentum = [classify(r["past_return_bps"]) for r in records]

    report_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = report_dir / "predictions.csv"
    baked_pred: list[str] | None = None
    baked_raw: list[str] | None = None
    baked_indices: list[int] = []
    baked_sec: float | None = None
    if args.reuse_predictions:
        saved = pd.read_csv(predictions_path, keep_default_na=False)
        if len(saved) != len(records) or saved.timestamp.tolist() != [
            record["timestamp"] for record in records
        ]:
            raise ValueError("predictions.csv does not match the selected split/samples")
        base_pred = saved.q4tp_base.astype(str).tolist()
        applied_pred = saved.applied_skill.astype(str).tolist()
        base_raw = saved.base_raw.astype(str).tolist()
        applied_raw = saved.applied_raw.astype(str).tolist()
        baked_indices = [
            int(index) for index, value in enumerate(saved.baked_specialist) if value
        ]
        if baked_indices:
            baked_pred = [str(saved.baked_specialist.iloc[index]) for index in baked_indices]
            baked_raw = [str(saved.baked_raw.iloc[index]) for index in baked_indices]
        base_sec = applied_sec = None
        print(f"Reusing {len(records)} saved predictions from {predictions_path}")
    else:
        base_pred, base_raw, base_sec = score_model(
            cortiq, args.base, records, class_names, report_dir / "serve-base.log"
        )
        applied_pred, applied_raw, applied_sec = score_model(
            cortiq, args.applied, records, class_names,
            report_dir / "serve-applied-skill.log"
        )
        if args.baked and Path(args.baked).exists():
            parity_count = min(args.parity_samples, len(records))
            baked_indices = [
                int(index)
                for index in np.linspace(0, len(records) - 1, parity_count, dtype=int)
            ]
            baked_records = [records[index] for index in baked_indices]
            baked_pred, baked_raw, baked_sec = score_model(
                cortiq, args.baked, baked_records, class_names,
                report_dir / "serve-baked-specialist.log"
            )
    metrics = {
        "majority": classification_metrics(truth, majority, class_names),
        "momentum": classification_metrics(truth, momentum, class_names),
        "q4tp_base": classification_metrics(truth, base_pred, class_names),
        "q4tp_plus_applied_skill": classification_metrics(truth, applied_pred, class_names),
    }
    if baked_pred is not None:
        baked_truth = [truth[index] for index in baked_indices]
        metrics["baked_specialist_parity_subset"] = classification_metrics(
            baked_truth, baked_pred, class_names
        )
    ppl = {}
    if not args.skip_ppl:
        corpus = str(data_dir / f"{args.split}_corpus.txt")
        ppl["q4tp_base"] = ppl_value(cortiq, args.base, corpus, args.ppl_windows, args.ppl_window_len)
        ppl["q4tp_plus_applied_skill"] = ppl_value(
            cortiq, args.applied, corpus, args.ppl_windows, args.ppl_window_len
        )
    bootstrap = paired_block_bootstrap(records, base_pred, applied_pred)
    parity = None
    if baked_pred is not None:
        applied_subset = [applied_pred[index] for index in baked_indices]
        applied_raw_subset = [applied_raw[index] for index in baked_indices]
        parity = {
            "indices": baked_indices,
            "same_predictions": sum(
                a == b for a, b in zip(applied_subset, baked_pred, strict=True)
            ),
            "total": len(baked_indices),
            "exact": applied_subset == baked_pred and applied_raw_subset == baked_raw,
        }
    baked_by_index = {
        index: (baked_pred[offset], baked_raw[offset])
        for offset, index in enumerate(baked_indices)
    } if baked_pred is not None and baked_raw is not None else {}
    rows = []
    for index, (record, pb, ps, rb, rs) in enumerate(
        zip(records, base_pred, applied_pred, base_raw, applied_raw, strict=True)
    ):
        rows.append(
            {
                "timestamp": record["timestamp"], "horizon": record["horizon"],
                "truth": record["label"], "q4tp_base": pb, "applied_skill": ps,
                "baked_specialist": baked_by_index.get(index, (None, None))[0],
                "base_raw": rb, "applied_raw": rs,
                "baked_raw": baked_by_index.get(index, (None, None))[1],
                "future_return_bps": record["future_return_bps"],
            }
        )
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    sizes = {
        "q4tp_base_bytes": Path(args.base).stat().st_size,
        "applied_skill_model_bytes": Path(args.applied).stat().st_size,
        "baked_specialist_bytes": Path(args.baked).stat().st_size
        if args.baked and Path(args.baked).exists() else None,
        "standalone_skill_bytes": Path(args.skill).stat().st_size if Path(args.skill).exists() else None,
    }
    report = {
        "schema": 1, "created_utc": utc_now().isoformat(), "samples": len(records),
        "symbol": symbol, "split": args.split,
        "class_names": class_names,
        "metrics": metrics, "paired_block_bootstrap": bootstrap, "test_ppl": ppl,
        "applied_vs_baked_parity": parity,
        "sizes": sizes, "seconds": {
            "q4tp_base": base_sec,
            "q4tp_plus_applied_skill": applied_sec,
            "baked_specialist": baked_sec,
        },
    }
    atomic_json(report_dir / "report.json", report)
    lines = [
        f"# {symbol} CMF skill — фактический отчёт",
        "",
        f"Split: {args.split}; samples: {len(records)}",
        "",
    ]
    lines += ["| model | accuracy | macro-F1 | balanced accuracy | invalid |", "|---|---:|---:|---:|---:|"]
    for name, row in metrics.items():
        lines.append(
            f"| {name} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | "
            f"{row['balanced_accuracy']:.3f} | {row['invalid']} |"
        )
    delta = metrics["q4tp_plus_applied_skill"]["accuracy"] - metrics["q4tp_base"]["accuracy"]
    lines += [
        "", f"Accuracy delta: {delta * 100:+.2f} п.п.",
        f"Paired block-bootstrap 95% CI: [{bootstrap['ci95_low'] * 100:+.2f}; "
        f"{bootstrap['ci95_high'] * 100:+.2f}] п.п.",
    ]
    if parity is not None:
        lines += [
            "",
            f"Applied vs baked parity: {parity['same_predictions']}/{parity['total']} "
            f"predictions; exact raw output = {parity['exact']}.",
        ]
    if ppl:
        ppl_delta = (ppl["q4tp_plus_applied_skill"] / ppl["q4tp_base"] - 1) * 100
        lines += ["", f"Test PPL: {ppl['q4tp_base']:.3f} → "
                  f"{ppl['q4tp_plus_applied_skill']:.3f} ({ppl_delta:+.1f}%)."]
    size_delta = (sizes["applied_skill_model_bytes"] / sizes["q4tp_base_bytes"] - 1) * 100
    lines += ["", f"CMF size: {sizes['q4tp_base_bytes']/2**20:.1f} → "
              f"{sizes['applied_skill_model_bytes']/2**20:.1f} MiB ({size_delta:+.1f}%)."]
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    print(f"\nОтчёт: {report_dir / 'report.json'}")


def _hf_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hf_prompt(tokenizer: Any, record: dict[str, Any]) -> str:
    class_names = tuple(record.get("class_names", CLASS_NAMES))
    messages = [
        {
            "role": "system",
            "content": rendered_system_prompt(class_names, str(record["symbol"])),
        },
        {"role": "user", "content": record["prompt"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _load_or_extract_last_ffn_cache(
    model: Any,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    path: Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    """Cache the frozen input/residual around the final FFN.

    Only the last FFN is trained. Its input cannot change while every earlier
    parameter is frozen, so this cache is mathematically identical to running
    the whole 2B backbone during every optimizer step, but is far faster and
    needs much less memory.
    """
    if path.exists():
        loaded = np.load(path)
        cache = {name: loaded[name] for name in loaded.files}
        if len(cache["labels"]) != len(records):
            raise ValueError(f"{path}: cache has {len(cache['labels'])}, expected {len(records)}")
        print(f"  cache hit: {path} ({len(records)} samples)")
        return cache

    import torch
    from torch import nn

    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    last_layer = model.model.layers[-1]
    original_mlp = last_layer.mlp
    captured: dict[str, torch.Tensor] = {}

    class ZeroMlp(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(value)

    def capture_norm(_module: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["residual"] = inputs[0][:, -1, :].detach()
        captured["normed"] = output[:, -1, :].detach()

    hook = last_layer.post_attention_layernorm.register_forward_hook(capture_norm)
    last_layer.mlp = ZeroMlp()
    normed_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    labels: list[int] = []
    try:
        model.eval()
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            texts = [_hf_prompt(tokenizer, record) for record in batch_records]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            captured.clear()
            with torch.inference_mode():
                model.model(**encoded, use_cache=False, return_dict=True)
            if "normed" not in captured:
                raise RuntimeError("final post-attention norm hook did not fire")
            normed_parts.append(captured["normed"].float().cpu().numpy().astype(np.float16))
            residual_parts.append(captured["residual"].float().cpu().numpy().astype(np.float16))
            labels.extend(0 if record["label"] == "DOWN" else 1 for record in batch_records)
            done = min(start + batch_size, len(records))
            if done % max(batch_size * 10, 1) == 0 or done == len(records):
                print(f"  extracting {path.stem}: {done}/{len(records)}", flush=True)
    finally:
        hook.remove()
        last_layer.mlp = original_mlp

    cache = {
        "normed": np.concatenate(normed_parts),
        "residual": np.concatenate(residual_parts),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **cache)
    os.replace(tmp, path)
    print(f"  cache wrote: {path} ({path.stat().st_size / 2**20:.1f} MiB)")
    return cache


def _cached_logits(
    mlp: Any,
    final_norm: Any,
    head_rows: Any,
    cache: dict[str, np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    rows: list[np.ndarray] = []
    mlp.eval()
    with torch.inference_mode():
        for start in range(0, len(cache["labels"]), batch_size):
            end = min(start + batch_size, len(cache["labels"]))
            normed = torch.from_numpy(cache["normed"][start:end]).to(device=device, dtype=torch.bfloat16)
            residual = torch.from_numpy(cache["residual"][start:end]).to(device=device, dtype=torch.bfloat16)
            hidden = final_norm(residual + mlp(normed))
            logits = functional.linear(hidden, head_rows).float()
            rows.append(logits.cpu().numpy())
    return np.concatenate(rows)


def _softmax2(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def _probability_report(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probabilities = _softmax2(logits)
    prediction_ids = probabilities.argmax(axis=1)
    predictions = [CLASS_NAMES[int(value)] for value in prediction_ids]
    truth = [CLASS_NAMES[int(value)] for value in labels]
    confidence = probabilities.max(axis=1)
    correct = prediction_ids == labels
    selected_probability = probabilities[np.arange(len(labels)), labels]
    nll = float(-np.log(np.clip(selected_probability, 1e-9, 1.0)).mean())
    brier = float(np.mean((probabilities[:, 1] - (labels == 1)) ** 2))
    ece = 0.0
    reliability = []
    for low in np.linspace(0.5, 0.95, 10):
        high = low + 0.05
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= 1.0)
        if not mask.any():
            continue
        bin_conf = float(confidence[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)
        reliability.append(
            {"low": float(low), "high": float(min(high, 1.0)), "n": int(mask.sum()),
             "accuracy": bin_acc, "mean_confidence": bin_conf}
        )
    slices = {}
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        mask = confidence >= threshold
        slices[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "n": int(mask.sum()),
            "coverage": float(mask.mean()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        }
    report = classification_metrics(truth, predictions, CLASS_NAMES)
    report.update(
        {
            "nll": nll,
            "brier": brier,
            "ece10": float(ece),
            "confidence_mean": float(confidence.mean()),
            "confidence_std": float(confidence.std()),
            "confidence_correct_mean": float(confidence[correct].mean()) if correct.any() else None,
            "confidence_wrong_mean": float(confidence[~correct].mean()) if (~correct).any() else None,
            "confidence_slices": slices,
            "reliability": reliability,
        }
    )
    return report


def _pick_validation_confidence_threshold(report: dict[str, Any], min_coverage: float) -> float:
    candidates = [
        row for row in report["confidence_slices"].values()
        if row["threshold"] >= 0.55 and row["coverage"] >= min_coverage
        and row["accuracy"] is not None
    ]
    if not candidates:
        return 0.50
    # Threshold selection is validation-only. Accuracy wins; coverage breaks ties.
    return float(max(candidates, key=lambda row: (row["accuracy"], row["coverage"]))["threshold"])


def _save_partial_ffn_donor(model: Any, output: Path, layer_index: int) -> None:
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=True)
    prefix = f"model.layers.{layer_index}.mlp"
    mlp = model.model.layers[layer_index].mlp
    tensors = {
        f"{prefix}.gate_proj.weight": mlp.gate_proj.weight.detach().cpu().contiguous(),
        f"{prefix}.up_proj.weight": mlp.up_proj.weight.detach().cpu().contiguous(),
        f"{prefix}.down_proj.weight": mlp.down_proj.weight.detach().cpu().contiguous(),
    }
    save_file(tensors, output / "model.safetensors")


def command_train_fcd(args: argparse.Namespace) -> None:
    """Train a real BF16 FCD skill: one full FFN, no LoRA/adapters."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _hf_device(args.device)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output = Path(args.output_dir)
    records = {split: read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}
    records["train"] = balanced_records(records["train"], args.train_samples, CLASS_NAMES)
    records["validation"] = evenly_spaced(records["validation"], args.validation_samples)
    records["test"] = evenly_spaced(records["test"], args.test_samples)
    metadata = json.loads((data_dir / "dataset_manifest.json").read_text("utf-8"))
    print(f"Loading {args.model} as {dtype} on {device}; LoRA is not used")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        dtype=dtype,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer_index = len(model.model.layers) - 1
    mlp = model.model.layers[layer_index].mlp
    for parameter in mlp.parameters():
        parameter.requires_grad_(True)

    token_ids = {
        name: tokenizer.encode(name, add_special_tokens=False) for name in CLASS_NAMES
    }
    if any(len(ids) != 1 for ids in token_ids.values()):
        raise ValueError(f"UP/DOWN must each be one token, got {token_ids}")
    class_token_ids = torch.tensor(
        [token_ids[name][0] for name in CLASS_NAMES], device=device, dtype=torch.long
    )
    head_rows = model.lm_head.weight[class_token_ids].detach()
    print(f"Class tokens: {token_ids}; FCD layer: {layer_index}; trainable FFN params: "
          f"{sum(p.numel() for p in mlp.parameters())/1e6:.1f}M")

    caches = {
        split: _load_or_extract_last_ffn_cache(
            model, tokenizer, split_records,
            cache_dir / f"{metadata['symbol'].lower()}_{split}_lastffn.npz",
            device, args.extract_batch_size, args.max_length,
        )
        for split, split_records in records.items()
    }
    base_logits = {
        split: _cached_logits(
            mlp, model.model.norm, head_rows, cache, device, args.eval_batch_size
        )
        for split, cache in caches.items()
    }
    base_reports = {
        split: _probability_report(caches[split]["labels"], logits)
        for split, logits in base_logits.items()
    }
    print(
        f"BF16 base: validation={base_reports['validation']['accuracy']*100:.2f}% "
        f"test={base_reports['test']['accuracy']*100:.2f}%"
    )

    train_cache = caches["train"]
    train_count = len(train_cache["labels"])
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(train_count / args.batch_size) * args.epochs
    step = 0
    best_score = (
        base_reports["validation"]["balanced_accuracy"],
        base_reports["validation"]["accuracy"],
    )
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = {
        name: value.detach().cpu().clone() for name, value in mlp.state_dict().items()
    }
    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "lr": 0.0,
            "validation_accuracy": base_reports["validation"]["accuracy"],
            "validation_balanced_accuracy": base_reports["validation"]["balanced_accuracy"],
            "validation_macro_f1": base_reports["validation"]["macro_f1"],
            "validation_confidence_mean": base_reports["validation"]["confidence_mean"],
        }
    ]
    stale = 0
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_count)
        mlp.train()
        loss_sum = 0.0
        seen = 0
        for start in range(0, train_count, args.batch_size):
            index = order[start : start + args.batch_size]
            normed = torch.from_numpy(train_cache["normed"][index]).to(
                device=device, dtype=torch.bfloat16
            )
            residual = torch.from_numpy(train_cache["residual"][index]).to(
                device=device, dtype=torch.bfloat16
            )
            labels = torch.from_numpy(train_cache["labels"][index]).to(device=device)
            hidden = model.model.norm(residual + mlp(normed))
            logits = functional.linear(hidden, head_rows).float()
            ce = functional.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            if args.kd_weight > 0:
                teacher = torch.from_numpy(base_logits["train"][index]).to(device=device)
                kd = functional.kl_div(
                    functional.log_softmax(logits, dim=-1),
                    functional.softmax(teacher, dim=-1),
                    reduction="batchmean",
                )
                loss = ce + args.kd_weight * kd
            else:
                loss = ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), args.grad_clip)
            progress = step / max(total_steps - 1, 1)
            lr = args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) *
                            0.5 * (1.0 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            loss_sum += float(loss.item()) * len(index)
            seen += len(index)

        validation_logits = _cached_logits(
            mlp, model.model.norm, head_rows, caches["validation"], device, args.eval_batch_size
        )
        validation_report = _probability_report(caches["validation"]["labels"], validation_logits)
        score = (validation_report["balanced_accuracy"], validation_report["accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(seen, 1),
                "lr": optimizer.param_groups[0]["lr"],
                "validation_accuracy": validation_report["accuracy"],
                "validation_balanced_accuracy": validation_report["balanced_accuracy"],
                "validation_macro_f1": validation_report["macro_f1"],
                "validation_confidence_mean": validation_report["confidence_mean"],
            }
        )
        print(
            f"epoch {epoch:02d}: loss={loss_sum/max(seen,1):.4f} "
            f"val_acc={score[1]*100:.2f}% val_bal={score[0]*100:.2f}% "
            f"conf={validation_report['confidence_mean']:.3f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in mlp.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop after {stale} stale epoch(s)")
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    mlp.load_state_dict(best_state)

    skill_logits = {
        split: _cached_logits(
            mlp, model.model.norm, head_rows, cache, device, args.eval_batch_size
        )
        for split, cache in caches.items() if split != "train"
    }
    skill_reports = {
        split: _probability_report(caches[split]["labels"], logits)
        for split, logits in skill_logits.items()
    }
    selected_threshold = _pick_validation_confidence_threshold(
        skill_reports["validation"], args.min_confidence_coverage
    )
    threshold_key = f"{selected_threshold:.2f}"
    test_slice = skill_reports["test"]["confidence_slices"][threshold_key]
    base_predictions = [CLASS_NAMES[i] for i in base_logits["test"].argmax(axis=1)]
    skill_predictions = [CLASS_NAMES[i] for i in skill_logits["test"].argmax(axis=1)]
    bootstrap = paired_block_bootstrap(records["test"], base_predictions, skill_predictions)

    _save_partial_ffn_donor(model, output, layer_index)
    report = {
        "schema": 1,
        "created_utc": utc_now().isoformat(),
        "method": "BF16 full-FFN FCD; no LoRA; frozen backbone and attention",
        "model": args.model,
        "device": device,
        "symbol": metadata["symbol"],
        "window": metadata["window"],
        "target": metadata["target"],
        "fcd_layer": layer_index,
        "trainable_parameters": sum(parameter.numel() for parameter in mlp.parameters()),
        "class_token_ids": token_ids,
        "split_boundaries": metadata["boundaries"],
        "sample_counts": {split: len(value) for split, value in records.items()},
        "purge_hours": metadata["purge_hours"],
        "best_epoch": best_epoch,
        "history": history,
        "base": {"validation": base_reports["validation"], "test": base_reports["test"]},
        "skill": skill_reports,
        "selected_confidence_threshold_on_validation": selected_threshold,
        "test_at_selected_confidence": test_slice,
        "paired_block_bootstrap": bootstrap,
        "hyperparameters": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "kd_weight": args.kd_weight,
            "label_smoothing": args.label_smoothing, "seed": args.seed,
        },
    }
    atomic_json(output / "skill_manifest.json", report)
    print(
        f"\nCLOSED TEST: {base_reports['test']['accuracy']*100:.2f}% → "
        f"{skill_reports['test']['accuracy']*100:.2f}% "
        f"({(skill_reports['test']['accuracy']-base_reports['test']['accuracy'])*100:+.2f} pp)"
    )
    print(
        f"confidence ≥ {selected_threshold:.2f}: accuracy="
        f"{(test_slice['accuracy'] or 0)*100:.2f}%, coverage={test_slice['coverage']*100:.1f}%"
    )
    print(f"Partial BF16 donor: {output / 'model.safetensors'}")
    print(f"Report: {output / 'skill_manifest.json'}")


def command_predict(args: argparse.Namespace) -> None:
    boundary = parse_utc(args.end).floor("h")
    raw = fetch_klines(args.symbol, boundary - pd.Timedelta(hours=360), boundary, args.api_base)
    frame = add_features(raw, args.horizons)
    end = len(frame) - 1
    if frame.iloc[end - args.window + 1 : end + 1].isna().any().any():
        raise RuntimeError("Недостаточно истории для индикаторов")
    cortiq = cortiq_binary(args.cortiq)
    with cmf_server(cortiq, args.model, Path(args.log)) as url:
        output = []
        for horizon in args.horizons:
            prompt = make_prompt(frame, end, args.window, horizon, args.symbol)
            label, raw_text, confidence, probabilities = predict_prompt_scored(
                url, prompt, CLASS_NAMES, args.symbol
            )
            output.append(
                {
                    "horizon_hours": horizon,
                    "scenario": label,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "raw": raw_text,
                }
            )
    result = {
        "symbol": args.symbol.upper(),
        "last_closed_candle_utc": pd.Timestamp(frame.timestamp.iloc[end]).isoformat(),
        "close": float(frame.close.iloc[end]),
        "model": str(Path(args.model).resolve()),
        "predictions": output,
        "warning": "Исследовательский сценарий, не инвестиционная рекомендация.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download one year of closed 1h candles")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--warmup-bars", type=int, default=240)
    p.add_argument("--end", help="exclusive UTC boundary; default last closed hour")
    p.add_argument("--api-base", default=BINANCE_DATA_API)
    p.add_argument("--output", default="data/btcusdt_1h.csv")
    p.set_defaults(func=command_download)

    p = sub.add_parser("prepare", help="make chronological purged data and CMF corpora")
    p.add_argument("--input", default="data/btcusdt_1h.csv")
    p.add_argument("--output-dir", default="data/market_skill_binary")
    p.add_argument("--symbol", default="BTCUSDT", help="fallback if download manifest is absent")
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--horizons", type=int, nargs="+", default=[1])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--validation-ratio", type=float, default=0.15)
    p.add_argument("--cmf-corpus-samples", type=int, default=2500)
    p.add_argument("--quality-corpus-samples", type=int, default=400)
    p.set_defaults(func=command_prepare)

    p = sub.add_parser(
        "train-fcd",
        help="train one complete BF16 FFN as a binary skill (no LoRA)",
    )
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--data-dir", default="data/eth_market_skill_binary")
    p.add_argument("--cache-dir", default="artifacts/fcd-cache")
    p.add_argument("--output-dir", default="artifacts/eth-binary-fcd-donor")
    p.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--validation-samples", type=int, default=500)
    p.add_argument("--test-samples", type=int, default=800)
    p.add_argument("--extract-batch-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--min-lr-ratio", type=float, default=0.15)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--kd-weight", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-confidence-coverage", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_train_fcd)

    p = sub.add_parser("convert", help="BF16 Qwen -> q4tp CMF base")
    p.add_argument("--cortiq", default="../../target/release/cortiq")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--output", default="artifacts/qwen35-2b-q4tp.cmf")
    p.set_defaults(func=command_convert)

    p = sub.add_parser("bake", help="DTG-MA + FCD + defrag, then export skill")
    p.add_argument("--cortiq", default="../../target/release/cortiq")
    p.add_argument("--base", default="artifacts/qwen35-2b-q4tp.cmf")
    p.add_argument("--corpus", default="data/market_skill_binary/cmf_corpus.txt")
    p.add_argument("--held-corpus", default="data/market_skill_binary/validation_corpus.txt")
    p.add_argument("--specialist", default="artifacts/qwen35-btc-binary-v2-specialist.cmf")
    p.add_argument("--skill", default="artifacts/btc-binary-v2.skill.cmf")
    p.add_argument("--skill-id", default="btc-binary-v2")
    p.add_argument("--skill-name", default="BTC hourly UP DOWN classifier")
    p.add_argument("--log", default="artifacts/bake-binary-v2.log")
    p.add_argument("--report", default="artifacts/bake-binary-v2.json")
    p.add_argument("--steps-a", type=int, default=180)
    p.add_argument(
        "--steps-b", type=int, default=0,
        help="0 keeps the better mask-only validation checkpoint; raise to enable optional FCD",
    )
    p.add_argument("--lr-a", type=float, default=0.1)
    p.add_argument("--lr-b", type=float, default=0.0001)
    p.add_argument("--eval-every", type=int, default=30)
    p.add_argument("--fcd-layers", type=int, default=4)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--held", type=int, default=24)
    p.add_argument("--calib-chunks", type=int, default=1200)
    p.add_argument(
        "--focus-tokens", default="DOWN,UP",
        help="single-token labels for answer-focused native LM loss",
    )
    p.add_argument("--target-sparsity", type=float, default=0.0)
    p.add_argument("--l1-aggression", type=float, default=1.0)
    p.add_argument("--ffn-align", type=int, default=32)
    p.add_argument("--uniform-inter", action="store_true")
    p.set_defaults(func=command_bake)

    p = sub.add_parser("apply", help="attach standalone skill to exact base")
    p.add_argument("--cortiq", default="../../target/release/cortiq")
    p.add_argument("--base", default="artifacts/qwen35-2b-q4tp.cmf")
    p.add_argument("--skill", default="artifacts/btc-binary-v2.skill.cmf")
    p.add_argument("--output", default="artifacts/qwen35-btc-binary-v2-applied.cmf")
    p.set_defaults(func=command_apply)

    p = sub.add_parser("evaluate", help="strict test: q4tp base versus applied CMF skill")
    p.add_argument("--cortiq", default="../../target/release/cortiq")
    p.add_argument("--data-dir", default="data/market_skill_binary")
    p.add_argument("--base", default="artifacts/qwen35-2b-q4tp.cmf")
    p.add_argument("--applied", default="artifacts/qwen35-btc-binary-v2-applied.cmf")
    p.add_argument("--baked", default="artifacts/qwen35-btc-binary-v2-specialist.cmf")
    p.add_argument("--skill", default="artifacts/btc-binary-v2.skill.cmf")
    p.add_argument("--report-dir", default="artifacts/evaluation-binary-v2")
    p.add_argument(
        "--split", choices=("validation", "test"), default="test",
        help="use validation for model/asset selection; open test only for the final choice",
    )
    p.add_argument("--samples", type=int, default=180)
    p.add_argument(
        "--parity-samples", type=int, default=30,
        help="evenly-spaced baked checks; primary base/applied metrics still use --samples",
    )
    p.add_argument("--ppl-windows", type=int, default=12)
    p.add_argument("--ppl-window-len", type=int, default=512)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument(
        "--reuse-predictions", action="store_true",
        help="rebuild metrics/report from an already completed predictions.csv",
    )
    p.set_defaults(func=command_evaluate)

    p = sub.add_parser("predict", help="fetch latest closed candles and predict")
    p.add_argument("--cortiq", default="../../target/release/cortiq")
    p.add_argument("--model", default="artifacts/qwen35-btc-binary-v2-applied.cmf")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--horizons", type=int, nargs="+", default=[1])
    p.add_argument("--end")
    p.add_argument("--api-base", default=BINANCE_DATA_API)
    p.add_argument("--log", default="artifacts/predict-server.log")
    p.set_defaults(func=command_predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
