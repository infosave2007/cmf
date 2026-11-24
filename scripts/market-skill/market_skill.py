#!/usr/bin/env python3
"""Market-skill experiment for `cortiq skill bake`, standard library only.

The protocol of examples/btc_market_skill (causal features of 48 closed hourly candles, chronological
70/15/15 splits purged by 49 hours, one-token answers, the test period opened once), extended to
several markets and two targets:

  dir    UP if close[t+1] > close[t], else DOWN                         (the article's target)
  range  HIGH if the next candle's high-low range (bps) exceeds the market's median next-candle range
         over its own train period, else LOW

Commands:
  fetch-binance SYMBOL START END OUT.csv      1h klines (open time UTC, OHLC, volume, taker buy volume)
  fetch-yahoo TICKER OUT.csv                  1h FX bars for the last 730 days (no volume)
  prepare OUT_DIR CSV:SYMBOL...               features, splits, corpora, eval sets for both targets
  baselines DATA_DIR TARGET SPLIT             majority / rule / hour prior / logistic on SPLIT
  evaluate DATA_DIR TARGET SPLIT MODEL OUT    class probabilities through `cortiq serve` (CMF_GPU as set)
  report DATA_DIR TARGET SPLIT BASE.jsonl SKILL.jsonl   per-market metrics + paired block-bootstrap CI

This measures classification accuracy only; it is not a trading strategy.
"""
import csv
import datetime as dt
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

WINDOW = 48
PURGE_H = WINDOW + 1
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
DIRECTIVE = ("Answer directly and concisely. Do NOT reason, think step-by-step, or explain your "
             "process. Output ONLY the final answer.")
CLASSES = {"dir": ("DOWN", "UP"), "range": ("LOW", "HIGH")}
TRAIN_PER_MARKET = 4000      # balanced bake corpus records per market
HELD_PER_MARKET = 160        # balanced held (validation) corpus records per market, the bake's gate
EVAL_PER_MARKET = {"validation": 400, "test": 1000}   # evenly spaced, natural class mix


def utc(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)


def get_json(url, retries=5):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 + 2 * k)
            last = e  # noqa: F841


# ───────────────────────── data ─────────────────────────

def fetch_binance(symbol, start, end, out):
    start_ms = int(utc(start).timestamp() * 1000)
    end_ms = int(utc(end).timestamp() * 1000) - 1
    rows, cur = [], start_ms
    bases = ["https://api.binance.com", "https://data-api.binance.vision"]
    while cur <= end_ms:
        q = urllib.parse.urlencode({"symbol": symbol, "interval": "1h", "startTime": cur, "endTime": end_ms, "limit": 1000})
        page = None
        for b in bases:
            try:
                page = get_json(f"{b}/api/v3/klines?{q}", retries=3)
                break
            except Exception:  # noqa: BLE001
                continue
        if page is None:
            raise RuntimeError("Binance unreachable")
        if not page:
            break
        rows.extend(page)
        cur = int(page[-1][0]) + 3_600_000
    seen, out_rows = set(), []
    for r in rows:
        t = int(r[0])
        if t in seen or int(r[6]) > end_ms:
            continue
        seen.add(t)
        out_rows.append((dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                         r[1], r[2], r[3], r[4], r[5], r[9]))
    out_rows.sort()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume", "taker_buy_volume"])
        w.writerows(out_rows)
    print(symbol, len(out_rows), out_rows[0][0], out_rows[-1][0])


def fetch_yahoo(ticker, out):
    d = get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1h&range=730d")
    r = d["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    rows = []
    for i, s in enumerate(r["timestamp"]):
        vals = (q["open"][i], q["high"][i], q["low"][i], q["close"][i])
        t = dt.datetime.fromtimestamp(s, dt.timezone.utc)
        if None in vals or t.minute != 0:
            continue
        rows.append((t.strftime("%Y-%m-%d %H:%M"), *vals))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close"])
        w.writerows(rows)
    print(ticker, len(rows), rows[0][0], rows[-1][0])


def read_bars(path, end_exclusive):
    bars = []
    with open(path) as f:
        for row in csv.DictReader(f):
            t = utc(row["time"])
            if t >= end_exclusive:
                continue
            b = {"t": t, "o": float(row["open"]), "h": float(row["high"]), "l": float(row["low"]), "c": float(row["close"])}
            if "volume" in row and row["volume"] not in ("", None):
                b["v"] = float(row["volume"])
                b["tb"] = float(row["taker_buy_volume"])
            bars.append(b)
    bars.sort(key=lambda b: b["t"])
    return bars


# ───────────────────────── features (pandas semantics of btc_skill.add_features) ─────────────────────────

NAN = float("nan")


def isnan(x):
    return x != x


def ewm(xs, alpha, min_periods=0):
    """pandas ewm(adjust=False): the recursion starts at the first valid value; NaN before min_periods."""
    out, y, n = [], None, 0
    for x in xs:
        if isnan(x):
            out.append(NAN if y is None else (y if n >= min_periods else NAN))
            continue
        y = x if y is None else (1 - alpha) * y + alpha * x
        n += 1
        out.append(y if n >= max(min_periods, 1) else NAN)
    return out


def rolling(xs, n, fn, min_periods=None):
    mp = n if min_periods is None else min_periods
    out = []
    for i in range(len(xs)):
        w = [x for x in xs[max(0, i - n + 1): i + 1]]
        if len(w) < n and min_periods is None:
            out.append(NAN)
            continue
        valid = [x for x in w if not isnan(x)]
        if (min_periods is None and len(valid) < n) or len(valid) < mp:
            out.append(NAN)
            continue
        out.append(fn(valid))
    return out


def mean(v):
    return sum(v) / len(v)


def std(v):
    if len(v) < 2:
        return NAN
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def clip(x, lo, hi):
    return x if isnan(x) else max(lo, min(hi, x))


def div(a, b):
    return NAN if isnan(a) or isnan(b) or b == 0 else a / b


def features(bars):
    n = len(bars)
    c = [b["c"] for b in bars]
    o = [b["o"] for b in bars]
    h = [b["h"] for b in bars]
    lo = [b["l"] for b in bars]
    prev = [NAN] + c[:-1]
    F = [dict() for _ in range(n)]
    for i in range(n):
        for k, col in (("open", o), ("high", h), ("low", lo), ("close", c)):
            F[i][f"d_{k}_bps"] = (col[i] / prev[i] - 1) * 1e4 if i else NAN
    has_vol = "v" in bars[0]
    if has_vol:
        lv = [math.log1p(b["v"]) for b in bars]
        m168 = rolling(lv, 168, mean, min_periods=24)
        s168 = rolling(lv, 168, std, min_periods=24)
        for i in range(n):
            F[i]["volume_z"] = clip(div(lv[i] - m168[i], s168[i] if s168[i] != 0 else NAN), -6, 6)
            F[i]["taker_buy_ratio"] = clip(div(bars[i]["tb"], bars[i]["v"]), 0, 1)
    delta = [NAN] + [c[i] - c[i - 1] for i in range(1, n)]
    gain = ewm([x if isnan(x) else max(x, 0.0) for x in delta], 1 / 14, 14)
    loss = ewm([x if isnan(x) else max(-x, 0.0) for x in delta], 1 / 14, 14)
    ema = {k: ewm(c, 2 / (k + 1)) for k in (8, 12, 21, 26, 55, 200)}
    tr = [h[i] - lo[i] if i == 0 else max(h[i] - lo[i], abs(h[i] - prev[i]), abs(lo[i] - prev[i])) for i in range(n)]
    atr = ewm(tr, 1 / 14, 14)
    tp = [(h[i] + lo[i] + c[i]) / 3 for i in range(n)]
    tp_mean = rolling(tp, 20, mean)
    tp_md = rolling(tp, 20, lambda v: mean([abs(x - mean(v)) for x in v]))
    logret = [NAN] + [math.log(c[i] / c[i - 1]) for i in range(1, n)]
    hv = rolling(logret, 20, std)
    c_mean20 = rolling(c, 20, mean)
    c_std20 = rolling(c, 20, std)
    sup = rolling(lo, 48, min)
    res = rolling(h, 48, max)
    for i in range(n):
        f = F[i]
        rs = div(gain[i], loss[i] if loss[i] != 0 else NAN)
        f["rsi14"] = NAN if isnan(rs) else 100 - 100 / (1 + rs)
        f["macd_bps"] = (ema[12][i] - ema[26][i]) / c[i] * 1e4
        f["ema8_21_bps"] = (ema[8][i] / ema[21][i] - 1) * 1e4
        f["ema21_55_bps"] = (ema[21][i] / ema[55][i] - 1) * 1e4
        f["close_ema200_bps"] = (c[i] / ema[200][i] - 1) * 1e4
        f["atr14_bps"] = NAN if isnan(atr[i]) else atr[i] / c[i] * 1e4
        f["cci20"] = clip(div(tp[i] - tp_mean[i], 0.015 * tp_md[i] if not isnan(tp_md[i]) and tp_md[i] != 0 else NAN), -400, 400)
        f["roc10_bps"] = (c[i] / c[i - 10] - 1) * 1e4 if i >= 10 else NAN
        f["hv20_bps"] = NAN if isnan(hv[i]) else hv[i] * math.sqrt(24) * 1e4
        f["bb_position"] = clip(div(c[i] - c_mean20[i], 2 * c_std20[i] if not isnan(c_std20[i]) else NAN), -3, 3)
        f["sr_position"] = clip(div(c[i] - sup[i], res[i] - sup[i]), 0, 1)
        f["body_bps"] = (c[i] - o[i]) / c[i] * 1e4
        f["upper_wick_bps"] = (h[i] - max(o[i], c[i])) / c[i] * 1e4
        f["lower_wick_bps"] = (min(o[i], c[i]) - lo[i]) / c[i] * 1e4
        f["range_bps"] = (h[i] - lo[i]) / c[i] * 1e4
        for p in (1, 3, 6, 12, 24, 48):
            f[f"ret_{p}h_bps"] = (c[i] / c[i - p] - 1) * 1e4 if i >= p else NAN
    return F, has_vol


REQUIRED = ["d_open_bps", "d_high_bps", "d_low_bps", "d_close_bps", "rsi14", "macd_bps", "atr14_bps",
            "ema8_21_bps", "ema21_55_bps", "close_ema200_bps", "ret_1h_bps", "ret_3h_bps", "ret_6h_bps",
            "ret_12h_bps", "ret_24h_bps", "ret_48h_bps", "cci20", "roc10_bps", "hv20_bps", "bb_position",
            "sr_position", "body_bps", "upper_wick_bps", "lower_wick_bps", "range_bps"]
NUMERIC = [k for k in REQUIRED if not k.startswith("d_")]


def make_prompt(sym, f, t, has_vol):
    vol = f"volz={f['volume_z']:+.1f} buy={f['taker_buy_ratio']:.1f} " if has_vol else ""
    return (
        f"{sym} 1h causal summary of {WINDOW} closed bars. "
        f"r1={f['ret_1h_bps']:+.0f} r3={f['ret_3h_bps']:+.0f} "
        f"r6={f['ret_6h_bps']:+.0f} r12={f['ret_12h_bps']:+.0f} "
        f"r24={f['ret_24h_bps']:+.0f} r48={f['ret_48h_bps']:+.0f} "
        f"ema8_21={f['ema8_21_bps']:+.0f} ema21_55={f['ema21_55_bps']:+.0f} "
        f"ema200={f['close_ema200_bps']:+.0f} macd={f['macd_bps']:+.0f} "
        f"roc10={f['roc10_bps']:+.0f} rsi14={f['rsi14']:.0f} "
        f"cci20={f['cci20']:+.0f} bb={f['bb_position']:+.1f} sr={f['sr_position']:.1f} "
        f"atr14={f['atr14_bps']:.0f} hv20={f['hv20_bps']:.0f} "
        f"{vol}"
        f"body={f['body_bps']:+.0f} upper={f['upper_wick_bps']:.0f} "
        f"lower={f['lower_wick_bps']:.0f} range={f['range_bps']:.0f} "
        f"hour={t.hour} weekday={t.weekday()}. "
    )


def system_prompt(target, sym):
    if target == "dir":
        return (f"Binary {sym} market classifier. From causal features of {WINDOW} closed hourly candles, "
                "predict whether the next candle close is lower or higher. Output exactly DOWN or UP.")
    return (f"Binary {sym} volatility classifier. From causal features of {WINDOW} closed hourly candles, "
            "predict whether the next candle's high-low range is narrower or wider than this market's "
            "typical hourly range. Output exactly LOW or HIGH.")


QUESTION = {"dir": "Next close direction:", "range": "Next candle range:"}


def chat_record(rec):
    sp = f"{DIRECTIVE}\n\n{system_prompt(rec['target'], rec['symbol'])}"
    return (f"<|im_start|>system\n{sp}<|im_end|>\n<|im_start|>user\n{rec['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{rec['label']}<|im_end|>\n")


def evenly(recs, k):
    if len(recs) <= k:
        return list(recs)
    return [recs[int(i * len(recs) / k)] for i in range(k)]


def balanced(recs, k, classes):
    per = max(1, k // len(classes))
    buckets = [evenly([r for r in recs if r["label"] == c], per) for c in classes]
    out = []
    for i in range(max(map(len, buckets))):
        for b in buckets:
            if i < len(b):
                out.append(b[i])
    return out[:k]


def interleave(lists):
    out = []
    for i in range(max(map(len, lists))):
        for l in lists:
            if i < len(l):
                out.append(l[i])
    return out


def prepare(out_dir, specs, end_exclusive):
    os.makedirs(out_dir, exist_ok=True)
    per_target = {t: {s: [] for s in ("train", "validation", "test")} for t in CLASSES}
    meta = {"window": WINDOW, "purge_hours": PURGE_H, "ratios": [TRAIN_RATIO, VAL_RATIO, 1 - TRAIN_RATIO - VAL_RATIO],
            "end_exclusive_utc": end_exclusive.isoformat(), "markets": {}}
    for spec in specs:
        path, sym = spec.split(":")
        bars = read_bars(path, end_exclusive)
        F, has_vol = features(bars)
        req = REQUIRED + (["volume_z", "taker_buy_ratio"] if has_vol else [])
        bad = [any(isnan(F[i][k]) for k in req) for i in range(len(bars))]
        # a bad row poisons every window that contains it
        last_bad, anchors = -10 ** 9, []
        for i in range(len(bars) - 1):
            if bad[i]:
                last_bad = i
            if i - last_bad < WINDOW or i < WINDOW - 1:
                continue
            if bars[i + 1]["t"] - bars[i]["t"] != dt.timedelta(hours=1):
                continue          # the next bar must be the next hour (FX weekends, gaps)
            anchors.append(i)
        times = [bars[i]["t"] for i in anchors]
        cut1 = times[int(len(times) * TRAIN_RATIO)]
        cut2 = times[int(len(times) * (TRAIN_RATIO + VAL_RATIO))]
        purge = dt.timedelta(hours=PURGE_H)
        split_of = lambda t: "train" if t < cut1 else ("validation" if cut1 + purge <= t < cut2 else ("test" if t >= cut2 + purge else None))  # noqa: E731
        train_next_range = sorted(F[i + 1]["range_bps"] for i in anchors if split_of(bars[i]["t"]) == "train")
        med = train_next_range[len(train_next_range) // 2]
        for i in anchors:
            t = bars[i]["t"]
            sp = split_of(t)
            if sp is None:
                continue
            move = (bars[i + 1]["c"] / bars[i]["c"] - 1) * 1e4
            nrange = F[i + 1]["range_bps"]
            base = make_prompt(sym, F[i], t, has_vol)
            feats = {k: round(F[i][k], 6) for k in NUMERIC + (["volume_z", "taker_buy_ratio"] if has_vol else [])}
            for target, label in (("dir", "UP" if move > 0 else "DOWN"), ("range", "HIGH" if nrange > med else "LOW")):
                per_target[target][sp].append({
                    "timestamp": t.isoformat(), "symbol": sym, "target": target,
                    "prompt": base + QUESTION[target], "label": label,
                    "future_return_bps": round(move, 4), "next_range_bps": round(nrange, 4),
                    "range_threshold_bps": round(med, 4), "hour": t.hour, "weekday": t.weekday(), "feats": feats,
                })
        meta["markets"][sym] = {"source": path, "bars": len(bars), "anchors": len(anchors), "has_volume": has_vol,
                                "first_bar": bars[0]["t"].isoformat(), "last_bar": bars[-1]["t"].isoformat(),
                                "train_validation": cut1.isoformat(), "validation_test": cut2.isoformat(),
                                "range_median_bps": round(med, 4)}
    for target, classes in CLASSES.items():
        d = os.path.join(out_dir, target)
        os.makedirs(d, exist_ok=True)
        markets = list(meta["markets"])
        stats = {}
        for sp, recs in per_target[target].items():
            with open(os.path.join(d, f"{sp}.jsonl"), "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            stats[sp] = {m: {c: sum(1 for r in recs if r["symbol"] == m and r["label"] == c) for c in classes} for m in markets}
        train = interleave([balanced([r for r in per_target[target]["train"] if r["symbol"] == m], TRAIN_PER_MARKET, classes) for m in markets])
        held = interleave([balanced([r for r in per_target[target]["validation"] if r["symbol"] == m], HELD_PER_MARKET, classes) for m in markets])
        with open(os.path.join(d, "train_corpus.txt"), "w") as f:
            f.write("".join(chat_record(r) for r in train))
        with open(os.path.join(d, "held_corpus.txt"), "w") as f:
            f.write("".join(chat_record(r) for r in held))
        with open(os.path.join(d, "held_tags.jsonl"), "w") as f:
            for r in held:
                f.write(json.dumps({"family": r["symbol"]}) + "\n")
        for sp, k in EVAL_PER_MARKET.items():
            ev = interleave([evenly([r for r in per_target[target][sp] if r["symbol"] == m], k) for m in markets])
            with open(os.path.join(d, f"eval_{sp}.jsonl"), "w") as f:
                for r in ev:
                    f.write(json.dumps(r) + "\n")
        meta[target] = {"classes": classes, "split_counts": stats, "train_corpus": len(train), "held_corpus": len(held),
                        "eval": {sp: k for sp, k in EVAL_PER_MARKET.items()}}
    with open(os.path.join(out_dir, "dataset_manifest.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k == "markets"}, indent=1))
    for t in CLASSES:
        print(t, json.dumps(meta[t]["split_counts"]), "train corpus", meta[t]["train_corpus"])


# ───────────────────────── baselines ─────────────────────────

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def metrics(gold, pred, classes):
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    recalls, f1s = [], []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"n": n, "acc": acc, "bal_acc": sum(recalls) / len(recalls), "macro_f1": sum(f1s) / len(f1s)}


def logistic(train, test, keys, classes, iters=300, lr=0.5, l2=1e-3):
    X = [[r["feats"][k] for k in keys] + [math.sin(2 * math.pi * r["hour"] / 24), math.cos(2 * math.pi * r["hour"] / 24)] for r in train]
    y = [1.0 if r["label"] == classes[1] else 0.0 for r in train]
    d = len(X[0])
    mu = [mean([x[j] for x in X]) for j in range(d)]
    sd = [std([x[j] for x in X]) or 1.0 for j in range(d)]
    Z = [[(x[j] - mu[j]) / sd[j] for j in range(d)] for x in X]
    w, b = [0.0] * d, 0.0
    n = len(Z)
    for _ in range(iters):
        gw, gb = [0.0] * d, 0.0
        for z, t in zip(Z, y):
            s = b + sum(wj * zj for wj, zj in zip(w, z))
            p = 1 / (1 + math.exp(-max(-30, min(30, s))))
            e = p - t
            gb += e
            for j in range(d):
                gw[j] += e * z[j]
        b -= lr * gb / n
        w = [wj - lr * (gj / n + l2 * wj) for wj, gj in zip(w, gw)]
    out = []
    for r in test:
        x = [r["feats"][k] for k in keys] + [math.sin(2 * math.pi * r["hour"] / 24), math.cos(2 * math.pi * r["hour"] / 24)]
        s = b + sum(wj * (xj - m) / sj for wj, xj, m, sj in zip(w, x, mu, sd))
        out.append(classes[1] if s > 0 else classes[0])
    return out


def baselines(data_dir, target, split):
    classes = CLASSES[target]
    d = os.path.join(data_dir, target)
    train = load(os.path.join(d, "train.jsonl"))
    ev = load(os.path.join(d, f"eval_{split}.jsonl"))
    res = {}
    for m in sorted({r["symbol"] for r in ev}):
        tr = [r for r in train if r["symbol"] == m]
        te = [r for r in ev if r["symbol"] == m]
        gold = [r["label"] for r in te]
        maj = max(classes, key=lambda c: sum(r["label"] == c for r in tr))
        rows = {"majority": [maj] * len(te)}
        if target == "dir":
            rows["momentum r1"] = ["UP" if r["feats"]["ret_1h_bps"] > 0 else "DOWN" for r in te]
            rows["reversal r1"] = ["DOWN" if r["feats"]["ret_1h_bps"] > 0 else "UP" for r in te]
        else:
            thr = te[0]["range_threshold_bps"]
            rows["persistence range"] = ["HIGH" if r["feats"]["range_bps"] > thr else "LOW" for r in te]
            ph = {}
            for hh in range(24):
                sub = [r for r in tr if r["hour"] == hh]
                ph[hh] = sum(r["label"] == "HIGH" for r in sub) / max(1, len(sub))
            rows["hour prior"] = ["HIGH" if ph[r["hour"]] > 0.5 else "LOW" for r in te]
        keys = [k for k in te[0]["feats"]]
        sub = evenly(tr, 6000)
        rows["logistic (features + hour)"] = logistic(sub, te, keys, classes)
        res[m] = {name: metrics(gold, p, classes) for name, p in rows.items()}
    print(json.dumps(res, indent=1))
    return res


# ───────────────────────── model evaluation through cortiq serve ─────────────────────────

def held_records(data_dir, target):
    """The bake's held corpus as records, in its order (prepare's balanced interleave of validation)."""
    classes = CLASSES[target]
    val = load(os.path.join(data_dir, target, "validation.jsonl"))
    markets = list(dict.fromkeys(r["symbol"] for r in val))
    return interleave([balanced([r for r in val if r["symbol"] == m], HELD_PER_MARKET, classes) for m in markets])


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def evaluate(data_dir, target, split, model, out, cortiq=os.path.expanduser("~/cmf-skillode/target/release/cortiq")):
    classes = CLASSES[target]
    ev = held_records(data_dir, target) if split == "held" else load(os.path.join(data_dir, target, f"eval_{split}.jsonl"))
    if os.environ.get("EVAL_LIMIT"):
        ev = ev[: int(os.environ["EVAL_LIMIT"])]
    port = free_port()
    log = open(out + ".serve.log", "w")
    extra = os.environ.get("SERVE_ARGS", "").split()
    proc = subprocess.Popen([cortiq, "serve", model, *extra, "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 600
        while True:
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=5).read()
                break
            except Exception:  # noqa: BLE001
                if proc.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f"serve did not start; see {out}.serve.log")
                time.sleep(2)
        t0 = time.time()
        with open(out, "w") as f:
            for k, r in enumerate(ev):
                resp = post(base + "/v1/chat/completions", {
                    "model": "m", "temperature": 0.0, "max_tokens": 0, "enable_thinking": False,
                    "messages": [{"role": "system", "content": system_prompt(target, r["symbol"])},
                                 {"role": "user", "content": r["prompt"]}],
                    "cortiq": {"class_tokens": list(classes)}})
                cl = resp.get("cortiq", {}).get("classification")
                if not cl:
                    raise RuntimeError(f"no classification in the response: {json.dumps(resp)[:300]}")
                scores = {str(s["token"]): float(s["probability"]) for s in cl.get("scores", [])}
                f.write(json.dumps({"timestamp": r["timestamp"], "symbol": r["symbol"], "label": r["label"],
                                    "pred": str(cl["label"]), "p": scores}) + "\n")
                if k % 200 == 0:
                    print(f"{k}/{len(ev)} {time.time() - t0:.0f}s", flush=True)
        print(f"{len(ev)} requests in {time.time() - t0:.1f}s ({len(ev) / max(1e-9, time.time() - t0):.1f}/s)")
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("wrote", out)


def block_bootstrap(diffs, block=20, reps=2000, seed=7):
    rnd = random.Random(seed)
    n = len(diffs)
    nb = max(1, n // block)
    blocks = [diffs[i * block:(i + 1) * block] for i in range(nb)]
    stats = []
    for _ in range(reps):
        s = []
        for _ in range(nb):
            s.extend(blocks[rnd.randrange(nb)])
        stats.append(sum(s) / len(s))
    stats.sort()
    return stats[int(0.025 * reps)], stats[int(0.975 * reps)]


def report(data_dir, target, split, base_p, skill_p):
    classes = CLASSES[target]
    A, B = load(base_p), load(skill_p)
    assert [a["timestamp"] + a["symbol"] for a in A] == [b["timestamp"] + b["symbol"] for b in B]
    out = {}
    for m in sorted({a["symbol"] for a in A}) + ["ALL"]:
        ia = [i for i, a in enumerate(A) if m == "ALL" or a["symbol"] == m]
        gold = [A[i]["label"] for i in ia]
        ma = metrics(gold, [A[i]["pred"] for i in ia], classes)
        mb = metrics(gold, [B[i]["pred"] for i in ia], classes)
        diffs = [(B[i]["pred"] == B[i]["label"]) - (A[i]["pred"] == A[i]["label"]) for i in ia]
        lo, hi = block_bootstrap(diffs)
        # selective: the skill's most confident 20 % (threshold = the 80th percentile of this split's confidence)
        conf = sorted(max(B[i]["p"].values()) for i in ia)
        thr = conf[int(0.8 * len(conf))]
        sel = [i for i in ia if max(B[i]["p"].values()) >= thr]
        sel_acc = sum(B[i]["pred"] == B[i]["label"] for i in sel) / max(1, len(sel))
        pred_mix = {c: sum(B[i]["pred"] == c for i in ia) for c in classes}
        out[m] = {"base": ma, "skill": mb, "delta_acc": mb["acc"] - ma["acc"], "delta_ci95": [lo, hi],
                  "skill_top20_acc": sel_acc, "skill_top20_n": len(sel), "skill_pred_mix": pred_mix}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "fetch-binance":
        fetch_binance(a[1], a[2], a[3], a[4])
    elif a[0] == "fetch-yahoo":
        fetch_yahoo(a[1], a[2])
    elif a[0] == "prepare":
        prepare(a[1], a[2:], dt.datetime(2026, 11, 24, tzinfo=dt.timezone.utc))
    elif a[0] == "baselines":
        baselines(a[1], a[2], a[3])
    elif a[0] == "evaluate":
        evaluate(a[1], a[2], a[3], a[4], a[5])
    elif a[0] == "report":
        report(a[1], a[2], a[3], a[4], a[5])
    else:
        print(__doc__)
