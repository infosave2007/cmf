"""Per-stage report and backup of a `birth` run to a private Hub repo.

    IFS= read -r HF_TOKEN; export HF_TOKEN       # the token comes on stdin, never from a file
    python3 stage_sync.py --run /root/hc/run --name pass1 --repo infosave/cmf-experiments \
        --prefix runs/hermes_core_a40_20261114 --steps 60200 --batch 8 --seq 1024 --ckpt-every 1 --keep 3 \
        [--subdir base1 --start-step 60200 --title "..." --about about.md]

`--subdir` puts the report files of a later run of the same campaign into
their own folder; the checkpoints of all its runs share `<prefix>/ckpt/`,
so `--keep` counts them together. `--start-step` is the step the run began
at (its schedule and its stages count from there).

Watches `<run>/<name>.log`. A stage is complete when the trainer has printed
its held-out losses and `saved … (step N)`. For every complete stage the
script rewrites `REPORT.md` and `metrics.jsonl` and uploads them with the
log; for every `--ckpt-every`-th stage and for the last one it also uploads
a copy of the checkpoint as `ckpt/<name>-step<N>.ckpt` (the trainer replaces
its file by rename, so the copy is whole). Every upload is checked against
the Hub: the size, and the sha256 of a large file. Only the last `--keep`
checkpoints stay: an older one is removed from the tree and its LFS object
is deleted for good, by the sha256 this script uploaded it with and only
under `<prefix>/ckpt/`. Nothing is uploaded unless the repo is private.
Ends when the run is done or has failed and that state is on the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi

STEP = re.compile(r"^step\s+(\d+) loss ([\d.]+) \(ema ([\d.]+)\) \|g\| ([\d.]+) lr ([\d.e+-]+) (\d+) ms (\d+) tok/s")
VAL = re.compile(r"^\s+val loss ([\d.]+)\s+ppl ([\d.]+)(.*)$")
EXTRA = re.compile(r"(\w+) ([\d.]+) \(ppl ([\d.]+), (\d+) batches\)")
SAVED = re.compile(r"^\s+saved (\S+) \(step (\d+)\)")


def parse(text: str) -> tuple[list[dict], dict | None]:
    """Complete stages of the log, and the last step line seen."""
    stages: list[dict] = []
    last = None
    speeds: list[int] = []
    pending = None
    for line in text.splitlines():
        m = STEP.match(line)
        if m:
            last = {"step": int(m.group(1)), "loss": float(m.group(2)), "ema": float(m.group(3)),
                    "gnorm": float(m.group(4)), "lr": float(m.group(5)), "tok_s": int(m.group(7))}
            speeds.append(last["tok_s"])
            continue
        m = VAL.match(line)
        if m:
            pending = {"val": float(m.group(1)), "extra": {n: float(v) for n, v, _, _ in EXTRA.findall(m.group(3))}}
            continue
        m = SAVED.match(line)
        if m and pending is not None:
            step = int(m.group(2))
            if stages and stages[-1]["step"] >= step:
                pending = None  # a resumed attempt repeats nothing
                continue
            stages.append({"step": step, "ema": last["ema"] if last else None, "lr": last["lr"] if last else None,
                           "tok_s": round(sum(speeds) / len(speeds)) if speeds else None, **pending})
            speeds, pending = [], None
    return stages, last


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


class Hub:
    def __init__(self, repo: str, prefix: str, log: Path):
        self.api = HfApi(token=os.environ["HF_TOKEN"])
        self.repo, self.prefix, self.log = repo, prefix.strip("/"), log
        if not self.api.model_info(repo).private:
            raise SystemExit(f"{repo} is not private: nothing is uploaded")

    def say(self, msg: str) -> None:
        with open(self.log, "a") as f:
            f.write(f"{time.strftime('%FT%TZ', time.gmtime())} {msg}\n")

    def put(self, local: Path, name: str, message: str, digest: str | None = None) -> bool:
        """Upload and verify; False (and a log line) on any failure."""
        remote = f"{self.prefix}/{name}"
        for attempt in (1, 2, 3):
            try:
                self.api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=self.repo,
                                     repo_type="model", commit_message=message)
                info = self.api.get_paths_info(self.repo, [remote], repo_type="model")[0]
                oid = lfs_oid(info)
                ok = info.size == local.stat().st_size and (digest is None or oid == digest)
                self.say(f"{'ok' if ok else 'MISMATCH'} {remote} size {info.size} sha256 {oid or '-'}")
                if ok:
                    return True
            except Exception as e:  # the network, the Hub
                self.say(f"upload of {remote} failed (attempt {attempt}): {type(e).__name__}: {str(e)[:200]}")
            time.sleep(30 * attempt)
        return False


def lfs_oid(info) -> str | None:
    lfs = getattr(info, "lfs", None)
    if not lfs:
        return None
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def hub_ckpts(hub: "Hub", name: str) -> list[dict]:
    """The checkpoints of this run already on the Hub, oldest first."""
    out = []
    try:
        for f in hub.api.list_repo_tree(hub.repo, path_in_repo=f"{hub.prefix}/ckpt", repo_type="model", expand=True):
            m = re.search(r"/ckpt/[\w.\-]+-step(\d+)\.ckpt$", f.path)
            if m and lfs_oid(f):
                out.append({"step": int(m.group(1)), "file": f.path.rsplit("/", 1)[1], "size": f.size, "sha256": lfs_oid(f)})
    except Exception:  # no folder yet
        pass
    return sorted(out, key=lambda c: c["step"])


def drop_ckpt(hub: "Hub", c: dict) -> bool:
    """Remove a superseded checkpoint: from the tree, then its LFS object."""
    remote = f"{hub.prefix}/ckpt/{c['file']}"
    try:
        hub.api.delete_file(remote, hub.repo, repo_type="model", commit_message=f"drop superseded checkpoint of step {c['step']}")
        gone = [f for f in hub.api.list_lfs_files(hub.repo, repo_type="model")
                if f.file_oid == c["sha256"] and f.filename.startswith(f"{hub.prefix}/ckpt/")]
        if gone:
            hub.api.permanently_delete_lfs_files(hub.repo, gone, rewrite_history=False, repo_type="model")
        hub.say(f"dropped {remote} (LFS objects deleted: {len(gone)})")
        return True
    except Exception as e:
        hub.say(f"drop of {remote} failed: {type(e).__name__}: {str(e)[:200]}")
        return False


def report(args, stages: list[dict], last: dict | None, base: dict | None, state: dict, status: str, started: float) -> str:
    per_step = args.batch * args.seq
    now = time.time()
    first, total = args.start_step, args.steps - args.start_step
    done = max(0, (last["step"] + 1 if last else first) - first)
    elapsed = now - started
    speed = stages[-1]["tok_s"] if stages else (last or {}).get("tok_s")
    # the wall clock of the stages done so far: the printed tok/s leaves out the subspace refreshes, evals and saves
    if stages and stages[-1]["step"] > first and elapsed > 0:
        eta = elapsed * (total - done) / max(done, 1)
    else:
        eta = (total - done) * per_step / speed if speed else None
    names = sorted({n for s in stages for n in s["extra"]} | set((base or {}).get("extra", {})))
    title = {"hermes": "Hermes", "docs": "Документация ОС", "se": "Stack Exchange", "general": "Общий текст",
             "chat": "Диалоги", "multilingual": "Многоязычные", "think": "Рассуждения", "tools": "Инструменты",
             "lookup": "Справочник", "harness": "Код харнессов"}
    about = Path(args.about).read_text().strip() if args.about and Path(args.about).exists() else (
        "Модель: выращенный из Embryo-56M ствол 340M (16 слоёв × 768). Рецепт: `birth --carry --dropless`, "
        f"B{args.batch}×T{args.seq}, lr 1e-4 с разогревом 300 шагов и косинусом, зерно 17. Корпус: Stack Exchange "
        "435,9 млн токенов (вес 0,884), документация ОС 37,3 млн (0,076), исходники Hermes 20,1 млн (0,040). "
        "Отложенные потери считаются на 8 фиксированных пакетах каждого набора.")
    out = [
        "# " + (args.title or "Ядро Hermes: проход 340M по корпусу на A40"),
        "",
        f"Обновлено {time.strftime('%d.%m.%Y %H:%M UTC', time.gmtime(now))}. Состояние: **{status}**.",
        "",
        f"- Шаг {done} из {total} ({100 * done / total:.1f} %), прочитано {done * per_step / 1e6:.1f} млн токенов "
        f"из {total * per_step / 1e6:.1f} млн." + (f" Счёт шагов в журнале тренера идёт с {first}." if first else ""),
        f"- Идёт {elapsed / 3600:.2f} ч" + (f", осталось около {eta / 3600:.1f} ч." if eta is not None and status == "идёт" else "."),
        f"- Скорость последнего этапа: {speed or '—'} ток/с.",
        "",
        about,
        "",
        "## Этапы",
        "",
        "| Этап | Шаг | Токенов, млн | Потеря обучения (EMA) | Отложенная общая | "
        + " | ".join(title.get(n, n) for n in names) + " | ток/с |",
        "|---|---|---|---|---|" + "---|" * (len(names) + 1),
    ]
    if base:
        out.append("| 0, до обучения | 0 | 0 | — | " + f"{base['val']:.4f} | "
                   + " | ".join(f"{base['extra'].get(n, float('nan')):.4f}" for n in names) + " | — |")
    for i, s in enumerate(stages, 1):
        out.append(f"| {i} | {s['step'] - first} | {(s['step'] - first) * per_step / 1e6:.1f} | {s['ema']:.4f} | {s['val']:.4f} | "
                   + " | ".join(f"{s['extra'].get(n, float('nan')):.4f}" for n in names) + f" | {s['tok_s']} |")
    out += ["", "## Резервная копия", ""]
    if state.get("ckpts"):
        out += [f"В `{args.prefix}/ckpt/` хранятся {args.keep} последних чекпойнта кампании (с моментами оптимизатора); "
                "размер и sha256 каждого сверены с Hub после загрузки, более старые удаляются.", "",
                "| Шаг | Файл | Размер, ГБ | sha256 |", "|---|---|---|---|"]
        out += [f"| {c['step']} | `{c['file']}` | {c['size'] / 1e9:.2f} | `{c['sha256']}` |" for c in reversed(state["ckpts"])]
        out += [
            "",
            "Продолжить с любого из них на том же расписании:",
            "",
            "```",
            f"cortiq-embryo birth --resume <файл> --schedule-from 0 --steps {args.steps} --lr-restart …остальные флаги из train.sh",
            "```",
        ]
    else:
        out.append("Чекпойнт ещё не загружался.")
    if state.get("problems"):
        out += ["", "## Сбои", ""] + [f"- {p}" for p in state["problems"][-10:]]
    out += ["", "Файлы рядом: `train.log` — полный журнал тренера, `metrics.jsonl` — этапы числами, "
            "`sync.log` — журнал загрузок, `train.sh`, `stage_sync.py`, `JOURNAL.ru.md` — как ускорялся тренер.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--ckpt-every", type=int, default=1)
    ap.add_argument("--keep", type=int, default=3, help="checkpoints kept on the Hub")
    ap.add_argument("--subdir", default="", help="folder of this run's report files under the prefix")
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--title")
    ap.add_argument("--about", help="a text file: the paragraph of the report that describes the run")
    ap.add_argument("--base", help="log of a one-step run: the losses before training")
    ap.add_argument("--also", nargs="*", default=[], help="files uploaded once, as they are")
    args = ap.parse_args()
    run = Path(args.run)
    log, ckpt = run / f"{args.name}.log", run / f"{args.name}.ckpt"
    hub = Hub(args.repo, args.prefix, run / f"{args.name}.sync.log")
    base = None
    if args.base and Path(args.base).exists():
        m = [VAL.match(l) for l in Path(args.base).read_text(errors="replace").splitlines()]
        m = [x for x in m if x]
        if m:
            base = {"val": float(m[-1].group(1)), "extra": {n: float(v) for n, v, _, _ in EXTRA.findall(m[-1].group(3))}}
    state: dict = {"problems": [], "ckpts": hub_ckpts(hub, args.name)}
    sub = (args.subdir.strip("/") + "/") if args.subdir else ""
    for extra in args.also:
        if not hub.put(Path(extra), sub + Path(extra).name, f"{args.name}: {Path(extra).name}"):
            state["problems"].append(f"не загружен {Path(extra).name}")
    seen, first, final_sent = -1, True, False
    while True:
        started = float((run / f"{args.name}.start").read_text()) if (run / f"{args.name}.start").exists() else time.time()
        text = log.read_text(errors="replace") if log.exists() else ""
        stages, last = parse(text)
        finished = (run / f"{args.name}.done").exists()
        failed = (run / f"{args.name}.failed").exists()
        status = "завершено" if finished else "остановлено с ошибкой" if failed else "идёт"
        if len(stages) > seen or first or ((finished or failed) and not final_sent):
            n = len(stages)
            on_hub = {c["step"] for c in state["ckpts"]}
            want_ckpt = n > 0 and n > seen and (n % args.ckpt_every == 0 or finished) and ckpt.exists()
            if (finished or failed) and not final_sent and ckpt.exists() and n > 0:
                want_ckpt = True
            if want_ckpt and stages[-1]["step"] not in on_hub:
                step = stages[-1]["step"]
                fname = f"{args.name}-step{step:06d}.ckpt"
                copy = run / f"{args.name}.upload.ckpt"
                shutil.copyfile(ckpt, copy)
                digest = sha256(copy)
                if hub.put(copy, f"ckpt/{fname}", f"{args.name}: checkpoint of step {step}", digest):
                    state["ckpts"].append({"step": step, "file": fname, "size": copy.stat().st_size, "sha256": digest})
                    state["ckpts"].sort(key=lambda c: c["step"])
                    while len(state["ckpts"]) > args.keep:
                        if not drop_ckpt(hub, state["ckpts"][0]):
                            state["problems"].append(f"старый чекпойнт шага {state['ckpts'][0]['step']} не удалён")
                        state["ckpts"].pop(0)
                else:
                    state["problems"].append(f"чекпойнт шага {step} не загружен")
                copy.unlink(missing_ok=True)
            (run / f"{args.name}.metrics.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in stages))
            (run / f"{args.name}.REPORT.md").write_text(report(args, stages, last, base, state, status, started))
            label = f"{args.name}: stage {n}" + (" (final)" if finished else " (failed)" if failed else "")
            snap = run / f"{args.name}.upload.log"  # the trainer keeps appending to its log
            if log.exists():
                shutil.copyfile(log, snap)
            ok = all([hub.put(run / f"{args.name}.REPORT.md", sub + "REPORT.md", label),
                      hub.put(run / f"{args.name}.metrics.jsonl", sub + "metrics.jsonl", label),
                      hub.put(snap, sub + "train.log", label) if log.exists() else True])
            hub.put(hub.log, sub + "sync.log", label)
            if ok:
                seen, first = n, False
                final_sent = finished or failed
        if final_sent:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
