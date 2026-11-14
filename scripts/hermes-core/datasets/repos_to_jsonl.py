"""Sources of agent harnesses as a text corpus: one document per file.

    python3 repos_to_jsonl.py --repos repos.txt --clone /root/hc/repos --out /root/hc/prep/harness_code \
        [--skip REPO ...] [--only REPO ...]

Clones each repository of the list (depth 1, into its own directory) unless
it is there already, and writes `<out>.jsonl.gz` and `<out>.val.jsonl.gz`
(one file in 200 by the hash of repo and path, held out), each line
`{"text": "# File: <repo>/<path>\\n<contents>"}` — the form of the Hermes
corpus of 03.11, with the repository in the path.

Kept: source, shell, configuration and documentation files up to 200 KB
(JSON up to 20 KB). Dropped: dependencies and build output, lock files,
minified and generated files (a mean line over 300 characters), snapshots
and recorded fixtures, binary files, translations of the documentation.
The licence of every repository is checked before it goes into the list
(SURVEY.ru.md); the per-repository statistics go to `<out>.stats.json`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess

EXT = {
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".rs", ".go", ".java", ".kt", ".swift", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".cs", ".rb", ".php", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql", ".toml", ".yaml",
    ".yml", ".json", ".md", ".mdx", ".rst", ".txt", ".html", ".css", ".scss", ".vue", ".svelte", ".lua", ".ex", ".exs",
    ".dockerfile", ".tf", ".proto", ".graphql", ".ini", ".cfg", ".nix", ".j2", ".jinja",
}
NAMES = {"Dockerfile", "Makefile", "Justfile", "Procfile"}
DROP_DIRS = {
    ".git", "node_modules", "vendor", "third_party", "dist", "build", "out", "target", ".next", "__snapshots__",
    "cassettes", "fixtures", "__fixtures__", "testdata", "snapshots", "coverage", ".venv", "venv", "locales", "i18n",
    "translations", "site-packages", "generated", "__generated__", ".yarn",
}
DROP_FILES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "Cargo.lock", "uv.lock", "poetry.lock", "bun.lock",
              "bun.lockb", "go.sum", "composer.lock", "Gemfile.lock", "flake.lock"}
MAX_BYTES = 200_000
MAX_JSON = 20_000


def keep(path: str, name: str) -> bool:
    if name in DROP_FILES or ".min." in name or name.endswith((".lock", ".map", ".snap")):
        return False
    ext = os.path.splitext(name)[1].lower()
    if name not in NAMES and ext not in EXT:
        return False
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    return 0 < size <= (MAX_JSON if ext == ".json" else MAX_BYTES)


def docs(root: str, repo: str):
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in DROP_DIRS and not os.path.islink(os.path.join(d, x)))
        for name in sorted(files):
            path = os.path.join(d, name)
            if os.path.islink(path) or not keep(path, name):
                continue
            try:
                text = open(path, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            lines = text.count("\n") + 1
            if "\x00" in text or len(text) / lines > 300 or not text.strip():
                continue
            rel = os.path.relpath(path, root)
            yield rel, f"# File: {repo}/{rel}\n{text}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", required=True)
    ap.add_argument("--clone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--held-paths", help="relative paths (one per line) that stay held out: an earlier held-out set")
    args = ap.parse_args()
    held_paths = set(l.strip() for l in open(args.held_paths)) if args.held_paths else set()
    os.makedirs(args.clone, exist_ok=True)
    repos = [l.strip() for l in open(args.repos) if l.strip() and not l.startswith("#")]
    repos = [r for r in repos if r not in args.skip and (not args.only or r in args.only)]
    stats = {}
    with gzip.open(args.out + ".jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as tr, \
            gzip.open(args.out + ".val.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as va:
        for repo in repos:
            root = os.path.join(args.clone, repo.replace("/", "__"))
            if not os.path.isdir(os.path.join(root, ".git")):
                try:  # a stalled transfer must not hold the rest
                    r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", f"https://github.com/{repo}.git", root],
                                       capture_output=True, text=True, timeout=900)
                    err = r.stderr.strip() if r.returncode != 0 else ""
                except subprocess.TimeoutExpired:
                    err = "clone timed out"
                if err:
                    stats[repo] = {"error": err[:200]}
                    print(f"{repo}: clone failed: {err[:120]}", flush=True)
                    continue
            head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            n = nbytes = held = 0
            for rel, text in docs(root, repo):
                line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                if rel in held_paths or int(hashlib.md5(f"{repo}/{rel}".encode()).hexdigest()[:8], 16) % 200 == 0:
                    va.write(line)
                    held += 1
                else:
                    tr.write(line)
                    n += 1
                    nbytes += len(text)
            stats[repo] = {"commit": head, "files": n, "bytes": nbytes, "held_out_files": held}
            print(f"{repo:<40} {head[:10]} {n:>6} files {nbytes / 1e6:8.1f} MB", flush=True)
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"total {sum(s.get('bytes', 0) for s in stats.values()) / 1e6:.1f} MB in {sum(s.get('files', 0) for s in stats.values())} files")


if __name__ == "__main__":
    main()
