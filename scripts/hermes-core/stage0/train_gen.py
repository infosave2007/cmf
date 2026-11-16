"""Training tasks of "reasoning over a card", disjoint from the R set.

    python -I train_gen.py --tldr <dir>/tldr-main --out train.messages.jsonl [--n 90000] [--seed 7] [--contrast N] [--pipes N]

Output: one `{"messages": [user, assistant], "group": ..., "cat": ...}` per
line, the `messages` format of `cortiq-embryo sft-prepare` (`group` drives
its disjoint train / dev / final split).

Held-out rule: every command name that appears in R (families.py: cards,
distractors, OS sections, NONE cards) is excluded here, as are R's paths
(/data, /tmp/work), its rule set and its labels. The task FORMATS are R's:
that is what the fine-tune is meant to teach.

  slot    a tldr example with typed placeholders, values sampled, the request
          built from the example's description and the values
  direct  an example without placeholders
  os      a card with a section per OS (same-name pages of two or three
          platforms, or unrelated pages), the request names the OS
  choose  the right card among three; none: no card fits -> NONE
  react   a random rule set over a made-up output -> the label
  fix     a failed command and its error -> the corrected one (path typos
          with a directory listing; a few flag and sudo patterns)
  two     a pipeline of a listing command and a filter, from two cards
  arith   a number read or computed from a small table

--contrast N appends N tasks, drawn from their own generator (the tasks before
them stay byte-identical), that take the wording of `two` away as a cue:
  direct  the listing command of a pipeline alone ("Print the CPU information.")
  slot    a filter of a pipeline applied to a file, one command ("Show
          /srv/a.txt encoded as Base64.")
  two     pipelines over more filters, in the frames of the other requests
          ("How do I ...?", "I need to ...", "Please ...") and with more verbs
Without them a model learns that "Print/Show/List/Display <text operation>"
means two commands: at half of the base it opened 39 of R's 128 one-command
slot tasks as a pipeline (journal, the 50 % milestone).

--pipes N appends N more tasks from a third generator: pipelines over many
commands. `two` above pairs 15 listings with 8 filters, a set small enough to
know by heart, and a model that knows it by heart writes those names for any
pipeline instead of reading the cards. Here the listing is an example of any
page whose caption says what it prints ("List all partitions" -> "all
partitions"), the filter one of PIPE_FILTERS, an example of a card in the form
that reads `stdin` (the card's command without its file, or the part after the
pipe), and the request names the result ("Show the lines of all partitions
that match cache."):
  two     7 in 10: listing | filter, from two cards
  slot    2 in 10: the same filter over a file, one command, in the same words
  slot, direct  1 in 10: the listing alone

The generator is a program over tldr-pages (CC-BY 4.0) text; no model wrote
any of the data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import families as F  # noqa: E402
from build_r import INSTR  # noqa: E402
from cards import parse_page  # noqa: E402

PLATS = {"common": "Linux", "linux": "Linux", "osx": "macOS", "windows": "Windows"}

WORDS = """alpha amber anchor apex atlas aurora basalt beacon birch bravo cable cedar cinder cobalt comet copper coral
crane delta drift ember falcon fennel flint forge garnet glacier harbor hazel heron indigo iris jasper juniper kestrel
lagoon larch lotus maple marble meadow mesa mistral nectar nimbus oak onyx orchid osprey pebble pine plume quartz
quill raven reef ridge saffron sage sierra slate sparrow spruce summit talon thistle tide topaz tundra umber valley
vertex violet willow zephyr zinc""".split()
EXTS = ["txt", "log", "csv", "json", "md", "conf", "yaml", "xml", "sh", "py", "dat", "bak", "tsv", "ini"]
ROOTS = ["/srv", "/opt", "/var/lib", "/var/log", "/home/mira", "/home/tomas", "/mnt/store", "/etc", "/usr/local/share"]
WROOTS = [r"C:\Projects", r"C:\Users\mira\Documents", r"D:\Archive", r"C:\Temp", r"D:\Work"]
USERS = ["mira", "tomas", "ines", "pavel", "yuki", "omar", "greta", "devops", "deploy", "backup"]
GROUPS = ["staff", "admins", "www-data", "developers", "ops", "audio", "docker"]
PKGS = ["curl", "tmux", "ncdu", "fzf", "bat", "tree", "nmap", "vim", "rsync", "zsh", "jq", "wget", "git", "make", "ffmpeg", "sqlite3"]
HOSTS = ["web1", "db2", "cache3", "build-01", "node7", "gateway", "storage2", "mail1"]
DOMAINS = ["example.org", "intranet.test", "files.example.net", "api.example.io", "mirror.example.edu"]
SERVICES = ["sshd", "cron", "docker", "redis", "mysql", "apache2", "cups", "bluetooth", "ufw", "rsyslog"]
PATTERNS = ["timeout", "failed", "warning", "user", "retry", "payload", "session", "cache", "deploy", "invoice"]
BRANCHES = ["feature/login", "hotfix-42", "release-2.1", "dev", "bugfix/cache", "experiment"]
NAMES = ["orion", "lyra", "vega", "atlas", "nova", "pluto", "sirius", "rigel"]
IFACES = ["eth0", "eth1", "wlan0", "enp3s0", "en0"]

VERBS = {
    "Show": ["Show", "Display", "Print"], "Display": ["Display", "Show", "Print"], "Print": ["Print", "Show", "Output", "Display"],
    "List": ["List", "Show"], "Remove": ["Remove", "Delete"], "Delete": ["Delete", "Remove"], "Create": ["Create", "Make"],
    "Find": ["Find", "Locate"], "Extract": ["Extract", "Unpack"], "Get": ["Get", "Fetch"], "View": ["View", "Show"],
    "Run": ["Run", "Execute"], "Execute": ["Execute", "Run"], "Start": ["Start", "Launch"], "Check": ["Check", "Verify"],
    "Search": ["Search", "Look"], "Output": ["Output", "Print"],
}
SKIP_DESC = re.compile(r"documentation|display help|display version|show version|show help|print help|interactive", re.I)
KEYS = re.compile(r"<[A-Za-z][^>]*>")


class Skip(Exception):
    pass


def word(rng):
    return rng.choice(WORDS)


def rfile(rng, ext=None):
    ext = ext or rng.choice(EXTS)
    depth = rng.choice([1, 1, 2])
    return "/".join([rng.choice(ROOTS)] + [word(rng) for _ in range(depth - 1)] + [f"{word(rng)}.{ext}"])


def rdir(rng):
    return "/".join([rng.choice(ROOTS)] + [word(rng) for _ in range(rng.choice([1, 2]))])


def wfile(rng, ext=None):
    return rng.choice(WROOTS) + "\\" + word(rng) + "." + (ext or rng.choice(EXTS))


def wdir(rng):
    return rng.choice(WROOTS) + "\\" + word(rng)


def sample(ph: str, rng) -> tuple[str, str]:
    """(value, hint) for one placeholder, or Skip."""
    p = ph.strip()
    low = p.lower()
    if "|" in p or p.startswith("[") or not p:
        raise Skip
    if "..." in p:
        base = p.split(" ")[0]
        if "..." in base or "," in base:
            raise Skip
        k = rng.choice([1, 2])
        vals = [sample(re.sub(r"\d+$", "", base), rng)[0] for _ in range(k)]
        hint = re.sub(r"\d+$", "", base.replace("path/to/", "").replace("path\\to\\", "")).replace("_", " ")
        return " ".join(vals), (hint + ("s" if k > 1 and not hint.endswith("s") else ""))
    if re.fullmatch(r"-?\d+", p):
        n = int(p)
        v = rng.randint(2, 12) if n <= 10 else rng.randint(11, 200) if n <= 100 else rng.randint(101, 5000)
        if v == n:
            v += 1
        return str(v), "#" + p
    win = "\\" in p
    if low.startswith(("path/to", "/path/to", "path\\to")):
        tail = re.split(r"[\\/]", p)[-1] if len(re.split(r"[\\/]", p)) > 2 else ""
        hint = (tail or "path").replace("_", " ")
        m = re.search(r"\.([A-Za-z0-9.]{1,7})$", tail)
        ext = m.group(1) if m and not tail.startswith(".") else None
        if ext:
            hint = re.sub(r"\.[A-Za-z0-9.]+$", "", hint) or "file"
            return (wfile(rng, ext) if win else rfile(rng, ext)), hint
        if not tail or re.search(r"director|folder|dir$|mount|destination|repository|repo$", tail):
            return (wdir(rng) if win else rdir(rng)), hint or "directory"
        if "file_or_directory" in tail:
            return (rdir(rng) if rng.random() < 0.4 else rfile(rng)), "path"
        return (wfile(rng) if win else rfile(rng)), hint
    table = [
        (r"^(user|username|owner|login|remote_username)$", USERS, "user"),
        (r"^(group|group_name)$", GROUPS, "group"),
        (r"^(package|package_name|formula|module_name|gem_name|crate)$", PKGS, "package"),
        (r"^(host|hostname|remote_host|server|host1|host2)$", HOSTS, "host"),
        (r"^(domain|example\.com|example\.org)$", DOMAINS, "domain"),
        (r"^(unit|service|service_name)$", SERVICES, "service"),
        (r"^(pattern|search_pattern|keyword|string|search_string|text|regex|query|term)$", PATTERNS, "pattern"),
        (r"^(branch|branch_name|local_branch|remote_branch|new_branch)$", BRANCHES, "branch"),
        (r"^(name|container_name|image|image_name|vm_name|project_name|app_name|profile_name|database_name|"
         r"bucket_name|pod_name|session_name|alias|word|label|tag|volume_name|network_name)$", NAMES, None),
        (r"^(interface|interface_name|eth0|ethx)$", IFACES, "interface"),
        (r"^(remote|remote_name)$", ["origin", "upstream", "backup", "fork"], "remote"),
    ]
    for rx, pool, hint in table:
        if re.fullmatch(rx, low):
            return rng.choice(pool), hint or low.replace("_", " ")
    if re.fullmatch(r"(https?://)?example\.com(/.*)?|url", low):
        return f"https://{rng.choice(DOMAINS)}/{word(rng)}", "URL"
    if re.fullmatch(r"ip|ip_address|address", low):
        return f"10.{rng.randint(0, 9)}.{rng.randint(0, 99)}.{rng.randint(2, 250)}", "address"
    if re.fullmatch(r"port|port_number|local_port|remote_port", low):
        return str(rng.choice([2222, 3306, 5000, 6379, 8000, 8443, 9000, 9090, 27017])), "port"
    if re.fullmatch(r"pid|process_id|job_id|task_id", low):
        return str(rng.randint(300, 32000)), low.replace("_", " ")
    if re.fullmatch(r"n|number|count|lines|num|limit|depth|level|width|height|size|seconds|minutes|timeout|"
                    r"interval|days|hours|columns|rows|length|index|times", low):
        return str(rng.randint(2, 60)), low if len(low) > 1 else "number"
    if re.fullmatch(r"filename|file|input_file|output_file", low):
        return f"{word(rng)}.{rng.choice(EXTS)}", "file"
    raise Skip


def fill_example(desc: str, cmd: str, rng) -> tuple[str, str, list[tuple[str, str]]]:
    """(description with numeric literals replaced, command, [(hint, value)])."""
    if KEYS.search(cmd) or SKIP_DESC.search(desc) or cmd.startswith("tldr ") or len(cmd) > 110:
        raise Skip
    pairs = []

    def rep(m):
        nonlocal desc
        v, h = sample(m.group(1), rng)
        if h.startswith("#"):
            lit = h[1:]
            if re.search(rf"(?<![\w.]){re.escape(lit)}(?![\w.])", desc):
                desc = re.sub(rf"(?<![\w.]){re.escape(lit)}(?![\w.])", v, desc, count=1)
            else:
                pairs.append(("value", v))
        else:
            pairs.append((h, v))
        return v

    out = re.sub(r"\{\{(.*?)\}\}", rep, cmd)
    if "{{" in out or "}}" in out:
        raise Skip
    return desc, out, pairs


def clean_desc(d: str) -> str:
    d = re.sub(r"\[([A-Za-z])\]", r"\1", d)
    d = d.replace("`", "")
    d = re.sub(r"\s*\((Note|note)[^)]*\)", "", d)
    return d.strip().rstrip(".")


NOUN = re.compile(r"\b(a|an|the)( specific| given| specified| particular| single)? "
                  r"(file|directory|folder|package|user|host|service|unit|branch|port|URL|domain|interface|pattern|string)\b")


def request(desc: str, pairs: list[tuple[str, str]], rng) -> str:
    d = clean_desc(desc)
    first = d.split(" ")[0]
    if first in VERBS and rng.random() < 0.5:
        d = rng.choice(VERBS[first]) + d[len(first):]
        if d.startswith("Look "):
            d = "Look for" + d[4:] if not d.startswith("Look for") else d
    vals = ""
    if pairs:
        if len(pairs) == 1 and NOUN.search(d) and rng.random() < 0.6:
            d = NOUN.sub(lambda m: pairs[0][1], d, count=1)
        else:
            style = rng.randrange(4)
            if len(pairs) == 1 and style < 2:
                vals = rng.choice([": {v}", " — {v}", ", namely {v}", " for {v}"]).format(v=pairs[0][1])
            elif style == 2:
                vals = " (" + ", ".join(f"{h} = {v}" for h, v in pairs) + ")"
            elif style == 3:
                vals = ". " + "; ".join(f"The {h} is {v}" for h, v in pairs)
            else:
                vals = ": " + ", ".join(f"{h} {v}" for h, v in pairs)
    lead = rng.random()
    tail = ""
    if vals.startswith(". "):
        vals, tail = "", " " + vals[2:] + "."
    body = d + vals
    if lead < 0.12:
        return "How do I " + body[0].lower() + body[1:] + "?" + tail
    if lead < 0.24:
        return "I need to " + body[0].lower() + body[1:] + "." + tail
    if lead < 0.32:
        return "Please " + body[0].lower() + body[1:] + "." + tail
    return body + "." + tail


def card_text(name, desc, ex, label=None):
    head = f"{name}: {desc}" if label is None else f"[{label}] — {name}: {desc}"
    return "\n".join([head] + [f"- {w}: {c}" for w, c in ex])


def trim(ex, must: int, k: int, rng):
    idx = list(range(len(ex)))
    keep = {must} | set(rng.sample([i for i in idx if i != must], min(k - 1, len(ex) - 1)))
    return [ex[i] for i in sorted(keep)]


def msg(prompt, answer, group, cat):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            "group": group, "cat": cat}


def framed(body: str, rng) -> str:
    """A request in the frames of request(): bare, "How do I", "I need to", "Please"."""
    lead = rng.random()
    low = body[0].lower() + body[1:]
    if lead < 0.12:
        return f"How do I {low}?"
    if lead < 0.24:
        return f"I need to {low}."
    if lead < 0.32:
        return f"Please {low}."
    return body + "."


LISTS = ("List", "Show", "Display", "Print", "Get", "Dump", "Output")
ARTICLES = ("the", "all", "every", "information", "details", "detailed", "available", "installed", "running", "current", "active")
FILE_END = re.compile(r" \{\{[^{}]*(?:file|path)[^{}]*\}\}$")
SUMS = [("sha1sum", "SHA1"), ("sha224sum", "SHA224"), ("sha384sum", "SHA384"), ("sha512sum", "SHA512"), ("b2sum", "BLAKE2"),
        ("b3sum", "BLAKE3"), ("shasum", "SHA1")]
PIPE_FILTERS = [  # (page, the card's example, requests over {s}: what is filtered, {v}: the example's first value)
    ("common/base64", "base64 {{path/to/file}}", ["{s} encoded as Base64", "the Base64 encoding of {s}"]),
    ("common/base64", "base64 -d {{path/to/file}}", ["{s} decoded from Base64"]),
    ("common/base32", "base32 {{path/to/file}}", ["{s} encoded as Base32", "the Base32 encoding of {s}"]),
    ("common/base32", "base32 -d {{path/to/file}}", ["{s} decoded from Base32"]),
    *[(f"common/{c}", c + " {{path/to/file1 path/to/file2 ...}}", [f"the {n} checksum of {{s}}", f"{{s}} and compute the {n} checksum of that output"])
      for c, n in SUMS],
    ("common/cksum", "cksum {{path/to/file}}", ["the CRC checksum and byte count of {s}"]),
    ("common/sum", "sum {{path/to/file}}", ["the BSD checksum and block count of {s}"]),
    ("common/rhash", "rhash {{path/to/file}}", ["the CRC32 digest of {s}"]),
    ("common/expand", "expand {{path/to/file}}", ["{s} with tabs converted to spaces"]),
    ("common/expand", "expand -t {{number}} {{path/to/file}}", ["{s} with tabs converted to spaces, tab stops {v} characters apart"]),
    ("common/unexpand", "unexpand {{path/to/file}}", ["{s} with leading blanks converted to tabs"]),
    ("common/unexpand", "unexpand -a {{path/to/file}}", ["{s} with all blanks converted to tabs"]),
    ("common/fmt", "fmt {{path/to/file}}", ["{s} reformatted into paragraphs"]),
    ("common/fmt", "fmt -w {{n}} {{path/to/file}}", ["{s} reformatted to lines of at most {v} characters"]),
    ("common/fmt", "fmt -u {{path/to/file}}", ["{s} reformatted with uniform spacing"]),
    ("common/fold", "fold {{path/to/file}}", ["{s} wrapped to the default width"]),
    ("linux/fold", "fold -w {{width}} {{path/to/file}}", ["{s} wrapped to a width of {v} characters", "{s} folded to lines {v} characters wide"]),
    ("linux/fold", "fold -s -w {{width}} {{path/to/file}}", ["{s} wrapped at blanks to a width of {v} characters"]),
    ("linux/rev", "rev {{path/to/file}}", ["{s} with every line reversed"]),
    ("common/od", "od {{path/to/file}}", ["{s} as an octal dump", "an octal dump of {s}"]),
    ("common/hexdump", "hexdump {{path/to/file}}", ["{s} as a hexadecimal dump", "a hexadecimal dump of {s}"]),
    ("common/hexdump", "hexdump -C {{path/to/file}}", ["a hexadecimal dump of {s} with an ASCII column"]),
    ("common/hexyl", "hexyl {{path/to/file}}", ["the hexadecimal representation of {s}"]),
    ("common/hexyl", "hexyl -n {{n}} {{path/to/file}}", ["the hexadecimal representation of the first {v} bytes of {s}"]),
    ("common/strings", "strings {{path/to/file}}", ["the printable strings of {s}", "only the printable strings in {s}"]),
    ("common/strings", "strings -n {{n}} {{path/to/file}}", ["the printable strings of {s} that are at least {v} characters long"]),
    ("common/rg", "rg {{pattern}} {{path/to/file_or_directory}}",
     ["the lines of {s} that match {v}", "only the lines of {s} that contain {v}", "{s}, keeping the lines that match {v}"]),
    ("common/ugrep", 'ugrep -n "{{search_pattern}}" {{path/to/file_or_directory}}', ["the lines of {s} that match {v}, with their line numbers"]),
    ("common/gawk", "gawk '{print $5}' {{path/to/file}}", ["the fifth column of {s}", "only the fifth column of {s}"]),
    ("common/gawk", "gawk '/{{pattern}}/ {print $2}' {{path/to/file}}", ["the second column of the lines of {s} that match {v}"]),
    ("common/gawk", "gawk -F ',' '{print $NF}' {{path/to/file}}", ["the last comma-separated column of {s}"]),
    ("common/gawk", "gawk '{s+=$1} END {print s}' {{path/to/file}}", ["the sum of the first column of {s}", "the total of the values in the first column of {s}"]),
    ("common/gawk", "gawk 'NR%3==1' {{path/to/file}}", ["every third line of {s}, starting from the first"]),
    ("common/bat", "bat -n {{path/to/file}}", ["{s} with line numbers"]),
    ("common/bat", "bat -A {{path/to/file}}", ["{s} with the non-printable characters made visible"]),
    ("common/lolcat", "lolcat {{path/to/file}}", ["{s} in rainbow colors"]),
    ("common/more", "more {{path/to/file}}", ["{s} one screen at a time"]),
    ("common/lpr", "lpr {{path/to/file}}", ["Send {s} to the default printer"]),
    ("common/pr", "pr {{path/to/file1 path/to/file2 ...}}", ["{s} paginated with a header and a footer"]),
    ("common/hunspell", "hunspell -l {{path/to/file}}", ["the misspelled words in {s}"]),
    ("common/aspell", "cat {{path/to/file}} | aspell list", ["the misspelled words in {s}"]),
    ("common/tee", 'echo "example" | tee {{path/to/file}}', ["{s} and also save that output to {v}", "Save {s} to {v} and still show it"]),
    ("common/tee", 'echo "example" | tee -a {{path/to/file}}', ["{s} and also append that output to {v}"]),
    ("common/ts", "{{command}} | ts", ["{s} with a timestamp at the beginning of each line", "Put a timestamp before each line of {s}"]),
    ("linux/column", 'printf "header1 header2\\nbar foo\\n" | column -t', ["{s} aligned as a table", "{s} with the columns aligned"]),
    ("linux/col", "{{command}} | col -b", ["{s} with the backspaces removed"]),
    ("linux/wl-copy", "{{ls}} | wl-copy", ["Copy {s} to the clipboard", "Put {s} on the clipboard"]),
    ("linux/xsel", "echo 123 | xsel -ib", ["Copy {s} to the clipboard"]),
    ("common/figlet", "{{command}} | figlet", ["{s} as a large ASCII banner"]),
]


def generate(tldr: Path, n: int, seed: int, contrast: int = 0, pipes: int = 0) -> list[dict]:
    rng = random.Random(seed)
    used = set()
    for f in F.FAMILIES:
        used.update(pg.split("/")[1] for _, pg, _ in f["cards"])
    for o in F.OS:
        used.update(spec[1].split("/")[1] for spec in o["pages"].values())
    for pages, _ in F.NONE_TASKS:
        used.update(pg.split("/")[1] for pg in pages)
    used.update(pg.split("/")[1] for pg, _, _ in F.DIRECT)

    pages = {}
    for plat in PLATS:
        for f in sorted((tldr / "pages" / plat).glob("*.md")):
            name, desc, ex = parse_page(f.read_text())
            if f.stem in used or name.split(" ")[0] in used or not ex or len(desc) > 260:
                continue
            ex = [(w, c) for w, c in ex if not w.startswith("View documentation")]
            if ex:
                pages[f"{plat}/{f.stem}"] = (name, desc, ex)
    keys = sorted(pages)
    byname = defaultdict(list)
    for k in keys:
        byname[k.split("/")[1]].append(k)
    multi = [v for v in byname.values() if len({PLATS[k.split("/")[0]] for k in v}) >= 2]
    by_plat = {p: [k for k in keys if k.startswith(p + "/")] for p in PLATS}

    def pick_filled(page, want_values=None):
        name, desc, ex = pages[page]
        order = list(range(len(ex)))
        rng.shuffle(order)
        for i in order:
            try:
                d, cmd, pairs = fill_example(ex[i][0], ex[i][1], rng)
            except Skip:
                continue
            if want_values is not None and bool(pairs) != want_values:
                if not (want_values and d != ex[i][0]):
                    continue
            return i, d, cmd, pairs
        raise Skip

    out: list[dict] = []
    quota = {"slot": 0.44, "direct": 0.07, "os": 0.09, "choose": 0.09, "none": 0.045, "react": 0.07,
             "fix": 0.055, "two": 0.035, "arith": 0.105}

    def many(cat, make):
        want, tries = int(n * quota[cat]), 0
        got = 0
        while got < want and tries < want * 40:
            tries += 1
            try:
                out.append(make())
                got += 1
            except Skip:
                pass

    one, orn = INSTR["command"], INSTR["command_or_none"]

    def slot():
        page = rng.choice(keys)
        name, desc, ex = pages[page]
        i, d, cmd, pairs = pick_filled(page, True)
        prompt = f"Reference card:\n{card_text(name, desc, ex)}\n\nTask: {request(d, pairs, rng)}\n\n{one}"
        return msg(prompt, cmd, page, "slot")

    def direct():
        page = rng.choice(keys)
        name, desc, ex = pages[page]
        i, d, cmd, pairs = pick_filled(page, False)
        if pairs:
            raise Skip
        prompt = f"Reference card:\n{card_text(name, desc, ex)}\n\nTask: {request(d, [], rng)}\n\n{one}"
        return msg(prompt, cmd, page, "direct")

    def os_task():
        if rng.random() < 0.6:
            group = rng.choice(multi)
            chosen = {}
            for k in group:
                chosen.setdefault(PLATS[k.split("/")[0]], k)
            if len(chosen) == 2 and "Linux" in chosen and rng.random() < 0.5:
                pass
        else:
            chosen = {"Linux": rng.choice(by_plat["linux"] + by_plat["common"]), "macOS": rng.choice(by_plat["osx"]),
                      "Windows": rng.choice(by_plat["windows"])}
            if rng.random() < 0.4:
                chosen.pop(rng.choice(sorted(chosen)))
        target = rng.choice(sorted(chosen))
        i, d, cmd, pairs = pick_filled(chosen[target])
        secs = []
        for osn in [o for o in ("Linux", "macOS", "Windows") if o in chosen]:
            name, desc, ex = pages[chosen[osn]]
            ex2 = trim(ex, i, 3, rng) if osn == target else rng.sample(ex, min(3, len(ex)))
            secs.append(card_text(name, desc, ex2, osn))
        req = request(d, pairs, rng)
        req = req[0].lower() + req[1:] if not req.startswith(("How do I", "I need", "Please")) else req
        if req.startswith(("How do I", "I need", "Please")):
            task = f"On {target}: {req}"
        else:
            task = f"On {target}, {req}"
        prompt = "Reference card:\n" + "\n\n".join(secs) + f"\n\nTask: {task}\n\n{one}"
        return msg(prompt, cmd, chosen[target], "os")

    def three(target_page, i):
        others = rng.sample([k for k in keys if k.split("/")[1] != target_page.split("/")[1]], 2)
        cards = []
        for k in others:
            name, desc, ex = pages[k]
            cards.append(card_text(name, desc, rng.sample(ex, min(rng.choice([2, 3, 4]), len(ex)))))
        if i is not None:
            name, desc, ex = pages[target_page]
            cards.insert(rng.randrange(3), card_text(name, desc, trim(ex, i, rng.choice([2, 3, 4]), rng)))
        else:
            k = rng.choice([x for x in keys if x not in others and x.split("/")[1] != target_page.split("/")[1]])
            name, desc, ex = pages[k]
            cards.insert(rng.randrange(3), card_text(name, desc, rng.sample(ex, min(3, len(ex)))))
        return "Reference cards:\n" + "\n\n".join(cards)

    def choose():
        page = rng.choice(keys)
        i, d, cmd, pairs = pick_filled(page)
        return msg(f"{three(page, i)}\n\nTask: {request(d, pairs, rng)}\n\n{orn}", cmd, page, "choose")

    def none():
        page = rng.choice(keys)
        i, d, cmd, pairs = pick_filled(page)
        return msg(f"{three(page, None)}\n\nTask: {request(d, pairs, rng)}\n\n{orn}", "NONE", page, "none")

    triggers = ["No space left on device", "Operation not permitted", "Read-only file system", "Device or resource busy",
                "Name or service not known", "Network is unreachable", "Too many open files", "File exists",
                "Invalid argument", "Broken pipe", "Out of memory", "Address already in use", "certificate has expired",
                "Host key verification failed", "Authentication failed", "Segmentation fault", "syntax error",
                "Disk quota exceeded", "No route to host", "Text file busy", "Argument list too long", "unknown option",
                "checksum mismatch", "lock is held", "already running", "not a git repository", "merge conflict"]
    labels = ["CLEAN", "WAIT", "STOP", "SKIP", "REPORT", "REMOUNT", "RESTART", "FREE", "RENAME", "ABORT", "CONTINUE",
              "OK", "FIX", "CLOSE", "RENEW", "RELOGIN", "DEBUG", "IGNORE", "PAUSE", "NOTIFY", "ROLLBACK", "UNLOCK"]
    progs = ["cp", "mv", "rsync", "tar", "git", "make", "curl", "scp", "dd", "mount", "apt", "pip", "docker", "ssh", "nginx"]
    benign = ["done", "3 files changed", "ok", "Already up to date.", "12 packages can be upgraded.", "(no output)",
              "active (running)", "total 48", "Build succeeded", "0 errors, 0 warnings"]

    def react():
        k = rng.choice([2, 3, 4])
        trig = rng.sample(triggers, k)
        lab = rng.sample(labels, k + 1)
        rules = "Rules for the next action:\n" + "\n".join(f'- If the output contains "{t}", answer {l}.' for t, l in zip(trig, lab))
        rules += f"\n- Otherwise answer {lab[-1]}."
        prog = rng.choice(progs)
        cmdline = f"{prog} {rfile(rng)}" if rng.random() < 0.7 else f"{prog} {word(rng)}"
        j = rng.randrange(k + 1)
        if j < k:
            outp = rng.choice([f"{prog}: {rfile(rng)}: {trig[j]}", f"{prog}: error: {trig[j]}", f"fatal: {trig[j]}",
                               f"{prog}: {trig[j].lower()}" if rng.random() < 0.0 else f"E: {trig[j]}"])
        else:
            outp = rng.choice(benign)
        prompt = f"{rules}\n\nPrevious command: {cmdline}\nIts output:\n{outp}\n\nTask: Decide the next action.\n\n{INSTR['label']}"
        return msg(prompt, lab[j], "react", "react")

    def fix_typo():
        page = rng.choice(keys)
        name, desc, ex = pages[page]
        i, d, cmd, pairs = pick_filled(page, True)
        paths = [v for h, v in pairs if v.startswith("/") and " " not in v]
        if len(paths) != 1:
            raise Skip
        good = paths[0]
        parent, base = good.rsplit("/", 1)
        stem = base.split(".")[0]
        if len(stem) < 4:
            raise Skip
        j = rng.randrange(1, len(stem) - 1)
        bad_base = (base[:j] + base[j + 1:]) if rng.random() < 0.6 else (base[:j] + base[j + 1] + base[j] + base[j + 2:])
        bad = parent + "/" + bad_base
        listing = sorted({base} | {f"{word(rng)}.{rng.choice(EXTS)}" for _ in range(rng.randint(4, 8))})
        prog = cmd.split(" ")[0] if not cmd.startswith("sudo ") else cmd.split(" ")[1]
        ctx = (f"The command `{cmd.replace(good, bad)}` failed with:\n{prog}: {bad}: No such file or directory\n\n"
               f"Output of `ls {parent}`:\n{'  '.join(listing)}")
        prompt = f"Reference card:\n{card_text(name, desc, ex)}\n\n{ctx}\n\nTask: Give the corrected command.\n\n{one}"
        return msg(prompt, cmd, page, "fix")

    flagfix = [
        ("common/scp", "scp {h}:{d} {l}", "scp: {d}: not a regular file", "scp -r {h}:{d} {l}", "Give the corrected command."),
        ("common/chown", "chown {u} {f}", "chown: changing ownership of '{f}': Operation not permitted", "sudo chown {u} {f}", "Give the corrected command."),
        ("linux/umount", "sudo umount {d}", "umount: {d}: target is busy.", "sudo umount -l {d}",
         "The filesystem may be detached as soon as it is no longer used. Give the corrected command."),
        ("common/git-checkout", "git checkout {b}", "error: pathspec '{b}' did not match any file(s) known to git",
         "git checkout -b {b}", "The branch does not exist yet: create it and switch to it."),
        ("common/pip-install", "pip install {f}", "ERROR: Could not find a version that satisfies the requirement {f}",
         "pip install -r {f}", "That path is a requirements file. Give the corrected command."),
        ("linux/dpkg", "dpkg -i {f}", "dpkg: error: requested operation requires superuser privilege", "sudo dpkg -i {f}", "Give the corrected command."),
        ("common/crontab", "crontab -e -u {u}", "must be privileged to use -u", "sudo crontab -e -u {u}", "Give the corrected command."),
        ("linux/lsusb", "lsusb -v", "Couldn't open device, some information will be missing", "sudo lsusb -v", "Give the corrected command."),
        ("linux/dmesg", "dmesg", "dmesg: read kernel buffer failed: Operation not permitted", "sudo dmesg", "Give the corrected command."),
        ("common/chgrp", "chgrp {g} {d}", "The group changed only on the directory itself, not on the files inside it.",
         "chgrp -R {g} {d}", "Everything inside must get the group too. Give the corrected command."),
    ]
    flagfix = [p for p in flagfix if p[0] in pages]

    def fix_flag():
        page, bad, err, good, ask = rng.choice(flagfix)
        name, desc, ex = pages[page]
        v = {"h": rng.choice(HOSTS), "d": rdir(rng), "l": rdir(rng), "u": rng.choice(USERS), "g": rng.choice(GROUPS),
             "b": rng.choice(BRANCHES), "f": rfile(rng, {"common/pip-install": "txt", "linux/dpkg": "deb"}.get(page))}
        sub = lambda s: re.sub(r"\{([a-z])\}", lambda m: v[m.group(1)], s)
        if err.startswith("The group"):
            ctx = f"The command `{sub(bad)}` ran, but: {sub(err)}"
        else:
            ctx = f"The command `{sub(bad)}` failed with:\n{sub(err)}"
        prompt = f"Reference card:\n{card_text(name, desc, ex)}\n\n{ctx}\n\nTask: {ask}\n\n{one}"
        return msg(prompt, sub(good), page, "fix")

    def fix():
        return fix_typo() if rng.random() < 0.7 or not flagfix else fix_flag()

    sources = [
        ("linux/dmesg", "sudo dmesg", "the kernel messages"), ("common/printenv", "printenv", "all environment variables"),
        ("common/who", "who", "the logged-in sessions"), ("linux/lscpu", "lscpu", "the CPU information"),
        ("linux/lsusb", "lsusb", "the USB devices"), ("linux/lspci", "lspci", "the PCI devices"),
        ("linux/lsmod", "lsmod", "the loaded kernel modules"), ("common/pip-list", "pip list", "the installed Python packages"),
        ("common/pip-freeze", "pip freeze", "the installed packages in requirements format"),
        ("common/git-branch", "git branch -a", "all Git branches"), ("common/crontab", "crontab -l", "the cron jobs of the current user"),
        ("common/w", "w", "who is logged in and what they are doing"), ("common/groups", "groups", "the groups of the current user"),
        ("common/alias", "alias", "all shell aliases"), ("common/gem", "gem list", "the local Ruby gems"),
    ]
    filters = [
        ("common/fold", "fold -w{n}", "wrapped to a width of {n} characters"), ("common/base64", "base64", "encoded as Base64"),
        ("common/sha1sum", "sha1sum", "and compute the SHA1 checksum of that output"),
        ("common/tee", "tee {f}", "and also save the output to {f}"),
        ("common/fmt", "fmt -w {n}", "reformatted to lines of at most {n} characters"),
        ("linux/column", "column -t", "aligned as a table"), ("common/expand", "expand", "with tabs converted to spaces"),
        ("common/cksum", "cksum", "and print the CRC checksum of that output"),
    ]
    sources = [s for s in sources if s[0] in pages and any(c == s[1] for _, c in pages[s[0]][2])]
    filters = [f for f in filters if f[0] in pages]

    def two():
        sp, scmd, sphr = rng.choice(sources)
        fp, fcmd, fphr = rng.choice(filters)
        v = {"n": str(rng.randint(20, 120)), "f": rfile(rng, "txt")}
        sub = lambda s: s.replace("{n}", v["n"]).replace("{f}", v["f"])
        cards = []
        for pg, must in ((sp, scmd), (fp, None)):
            name, desc, ex = pages[pg]
            i = next((k for k, (_, c) in enumerate(ex) if c == must), 0)
            cards.append(card_text(name, desc, trim(ex, i, min(4, len(ex)), rng) if must else ex[:4]))
        if rng.random() < 0.5:
            cards.reverse()
        verb = rng.choice(["Show", "Print", "List", "Display"])
        prompt = "Reference cards:\n" + "\n\n".join(cards) + f"\n\nTask: {verb} {sphr} {sub(fphr)}.\n\n{one}"
        return msg(prompt, f"{scmd} | {sub(fcmd)}", sp, "two")

    lognames = [f"{w}.log" for w in WORDS[:40]] + [f"{w}.dat" for w in WORDS[40:60]]

    def arith():
        kind = rng.randrange(8)
        if kind in (0, 1):
            fs = rng.sample(lognames, rng.randint(5, 7))
            sizes = [rng.randrange(150, 9900) for _ in fs]
            table = "\n".join(f"-rw-r--r-- 1 {rng.choice(USERS[:4])} staff {s:>5} Mar  3 09:12 {nm}" for nm, s in zip(fs, sizes))
            d = rdir(rng)
            if kind == 0:
                thr = sorted(sizes)[rng.randrange(1, len(sizes) - 1)] - rng.randrange(1, 60)
                return msg(f"Output of `ls -l {d}`:\n{table}\n\nQuestion: How many of these files are larger than {thr} bytes?\n{INSTR['number']}",
                           str(sum(s > thr for s in sizes)), "arith.ls_count", "arith")
            a, b = rng.sample(range(len(fs)), 2)
            return msg(f"Output of `ls -l {d}`:\n{table}\n\nQuestion: What is the combined size in bytes of {fs[a]} and {fs[b]}?\n{INSTR['number']}",
                       str(sizes[a] + sizes[b]), "arith.ls_sum", "arith")
        if kind in (2, 3):
            fs = rng.sample(lognames, rng.randint(3, 5))
            ls = [rng.randrange(8, 950) for _ in fs]
            table = "\n".join(f"{l:>5} {nm}" for nm, l in zip(fs, ls)) + f"\n{sum(ls):>5} total"
            if kind == 2:
                a, b = rng.sample(range(len(fs)), 2)
                return msg(f"Output of `wc -l *`:\n{table}\n\nQuestion: How many lines do {fs[a]} and {fs[b]} have together?\n{INSTR['number']}",
                           str(ls[a] + ls[b]), "arith.wc_sum", "arith")
            if ls.count(max(ls)) > 1:
                raise Skip
            return msg(f"Output of `wc -l *`:\n{table}\n\nQuestion: Which file has the most lines?\n{INSTR['name']}",
                       fs[ls.index(max(ls))], "arith.wc_max", "arith")
        if kind in (4, 5):
            ms = rng.sample(["/", "/home", "/var", "/srv", "/opt", "/mnt/store", "/boot"], 4)
            rows = []
            for j, m in enumerate(ms):
                size = rng.choice([20, 40, 80, 120, 160, 320, 640])
                used = rng.randrange(3, size - 2)
                rows.append((f"/dev/nvme0n1p{j + 1}", size, used, size - used, m))
            table = "Filesystem      Size  Used Avail Use% Mounted on\n" + "\n".join(
                f"{dv:<14} {s:>4}G {u:>4}G {a:>4}G {100 * u // s:>3}% {m}" for dv, s, u, a, m in rows)
            if kind == 4:
                k = rng.randrange(4)
                return msg(f"Output of `df -h`:\n{table}\n\nQuestion: How many GB are available on {rows[k][4]}?\n{INSTR['number']}",
                           str(rows[k][3]), "arith.df_avail", "arith")
            a, b = rng.sample(range(4), 2)
            return msg(f"Output of `df -h`:\n{table}\n\nQuestion: How many GB are used on {rows[a][4]} and {rows[b][4]} together?\n{INSTR['number']}",
                       str(rows[a][2] + rows[b][2]), "arith.df_used_sum", "arith")
        if kind == 6:
            ds = [rdir(rng) for _ in range(3)]
            ks = rng.sample(range(90, 99000), 3)
            a, b = (0, 1) if ks[0] > ks[1] else (1, 0)
            table = "\n".join(f"{k}\t{d}" for k, d in zip(ks, ds))
            return msg(f"Output of `du -sk {' '.join(ds)}`:\n{table}\n\nQuestion: How many KB larger is {ds[a]} than {ds[b]}?\n{INSTR['number']}",
                       str(ks[a] - ks[b]), "arith.du_diff", "arith")
        us = rng.sample(USERS, 4)
        rows = [(rng.choice(us), rng.randrange(300, 32000), rng.choice(SERVICES + ["bash", "python3", "node"])) for _ in range(rng.randint(7, 10))]
        table = "USER       PID COMMAND\n" + "\n".join(f"{u:<9} {p:>5} {c}" for u, p, c in rows)
        u = rng.choice(sorted({r[0] for r in rows}))
        return msg(f"Output of `ps -eo user,pid,comm`:\n{table}\n\nQuestion: How many of these processes belong to the user {u}?\n{INSTR['number']}",
                   str(sum(r[0] == u for r in rows)), "arith.ps_count", "arith")

    for cat, make in (("slot", slot), ("direct", direct), ("os", os_task), ("choose", choose), ("none", none),
                      ("react", react), ("fix", fix), ("two", two), ("arith", arith)):
        many(cat, make)
    rng.shuffle(out)
    if not contrast and not pipes:
        return out

    # One command in the wording of `two`, and `two` in more wordings: its own generator, appended.
    rng = random.Random(seed + 4409)
    more_filters = filters + [f for f in [
        ("common/base32", "base32", "encoded as Base32"),
        ("common/sha512sum", "sha512sum", "and compute the SHA512 checksum of that output"),
        ("common/sha224sum", "sha224sum", "and compute the SHA224 checksum of that output"),
        ("common/sha384sum", "sha384sum", "and compute the SHA384 checksum of that output"),
        ("common/b2sum", "b2sum", "and compute the BLAKE2 checksum of that output"),
        ("common/unexpand", "unexpand", "with spaces converted to tabs"),
        ("linux/rev", "rev", "with every line reversed"),
        ("common/od", "od", "as an octal dump"),
        ("common/hexdump", "hexdump", "as a hexadecimal dump"),
    ] if f[0] in pages]
    on_file = [f for f in [  # (page, the example's template, phrases over {f}, {n}: the values in the template's order)
        ("common/base64", "base64 {{path/to/file}}", ["{f} encoded as Base64"]),
        ("common/base32", "base32 {{path/to/file}}", ["{f} encoded as Base32"]),
        ("common/sha1sum", "sha1sum {{path/to/file1 path/to/file2 ...}}", ["the SHA1 checksum of {f}"]),
        ("common/sha224sum", "sha224sum {{path/to/file1 path/to/file2 ...}}", ["the SHA224 checksum of {f}"]),
        ("common/sha384sum", "sha384sum {{path/to/file1 path/to/file2 ...}}", ["the SHA384 checksum of {f}"]),
        ("common/sha512sum", "sha512sum {{path/to/file1 path/to/file2 ...}}", ["the SHA512 checksum of {f}"]),
        ("common/b2sum", "b2sum {{path/to/file1 path/to/file2 ...}}", ["the BLAKE2 checksum of {f}"]),
        ("common/cksum", "cksum {{path/to/file}}", ["the CRC checksum of {f}"]),
        ("common/expand", "expand {{path/to/file}}", ["{f} with tabs converted to spaces"]),
        ("common/unexpand", "unexpand {{path/to/file}}", ["{f} with spaces converted to tabs"]),
        ("common/fmt", "fmt {{path/to/file}}", ["{f} reformatted into paragraphs"]),
        ("common/fmt", "fmt -w {{n}} {{path/to/file}}", ["{f} reformatted to lines of at most {n} characters"]),
        ("common/fold", "fold {{path/to/file}}", ["{f} wrapped to the default width"]),
        ("linux/rev", "rev {{path/to/file}}", ["{f} with every line reversed"]),
        ("common/od", "od {{path/to/file}}", ["{f} as an octal dump"]),
        ("common/hexdump", "hexdump {{path/to/file}}", ["{f} as a hexadecimal dump"]),
        ("common/strings", "strings {{path/to/file}}", ["the printable strings of {f}"]),
    ] if f[0] in pages and any(c == f[1] for _, c in pages[f[0]][2])]
    src_verbs = ["Show", "Print", "List", "Display", "Output", "Get"]

    def source_alone():
        sp, scmd, sphr = rng.choice(sources)
        name, desc, ex = pages[sp]
        i = next(k for k, (_, c) in enumerate(ex) if c == scmd)
        card = card_text(name, desc, trim(ex, i, min(4, len(ex)), rng))
        return msg(f"Reference card:\n{card}\n\nTask: {framed(rng.choice(src_verbs) + ' ' + sphr, rng)}\n\n{one}", scmd, sp, "direct")

    def filter_on_file():
        page, tpl, phrases = rng.choice(on_file)
        name, desc, ex = pages[page]
        i = next(k for k, (_, c) in enumerate(ex) if c == tpl)
        _, cmd, pairs = fill_example(ex[i][0], tpl, rng)
        v = {"f": " and ".join(pairs[-1][1].split(" ")), "n": pairs[0][1]}
        phrase = rng.choice(phrases).format(**v)
        verbs = ["Show", "Print", "Display", "Output"] + (["Compute", "Get"] if "checksum" in phrase else [])
        card = card_text(name, desc, trim(ex, i, min(4, len(ex)), rng))
        return msg(f"Reference card:\n{card}\n\nTask: {framed(rng.choice(verbs) + ' ' + phrase, rng)}\n\n{one}", cmd, page, "slot")

    def two_more():
        sp, scmd, sphr = rng.choice(sources)
        fp, fcmd, fphr = rng.choice(more_filters)
        v = {"n": str(rng.randint(20, 120)), "f": rfile(rng, "txt")}
        sub = lambda t: t.replace("{n}", v["n"]).replace("{f}", v["f"])
        cards = []
        for pg, must in ((sp, scmd), (fp, None)):
            name, desc, ex = pages[pg]
            i = next((k for k, (_, c) in enumerate(ex) if c == must), 0)
            cards.append(card_text(name, desc, trim(ex, i, min(4, len(ex)), rng) if must else ex[:4]))
        if rng.random() < 0.5:
            cards.reverse()
        task = framed(f"{rng.choice(src_verbs)} {sphr} {sub(fphr)}", rng)
        return msg("Reference cards:\n" + "\n\n".join(cards) + f"\n\nTask: {task}\n\n{one}", f"{scmd} | {sub(fcmd)}", sp, "two")

    extra = []
    for share, make in ((0.2, source_alone), (0.5, filter_on_file), (0.3, two_more)):
        extra += [make() for _ in range(int(contrast * share))]
    rng.shuffle(extra)
    out = out + extra
    if not pipes:
        return out

    # Pipelines over many commands: a listing of any page and a filter that reads it. Its own generator, appended.
    rng = random.Random(seed + 7717)
    flt = []  # (page, the example's index, its form that reads `stdin`, requests)
    for page, tpl, phrases in PIPE_FILTERS:
        i = next((k for k, (_, c) in enumerate(pages[page][2]) if c == tpl), None) if page in pages else None
        if i is not None:
            flt.append((page, i, tpl.rsplit(" | ", 1)[1] if " | " in tpl else FILE_END.sub("", tpl), phrases))
    on_files = [f for f in flt if " | " not in pages[f[0]][2][f[1]][1]]
    plain = [k for k in keys if k.split("/")[0] in ("common", "linux")]

    def listing():
        """(page, the example's index, the command, what it prints as a noun phrase, whether values were filled in)."""
        if rng.random() < 0.12:
            sp, scmd, sphr = rng.choice(sources)
            return sp, next(k for k, (_, c) in enumerate(pages[sp][2]) if c == scmd), scmd, sphr, False
        for _ in range(60):  # most pages have no such example: draw pages until one has
            page = rng.choice(plain)
            ex = pages[page][2]
            order = list(range(len(ex)))
            rng.shuffle(order)
            for i in order:
                w, c = ex[i]
                first = w.split(" ")[0]
                if first not in LISTS or re.search(r"[|<>;&]", c):
                    continue
                try:
                    d, cmd, pairs = fill_example(w, c, rng)
                except Skip:
                    continue
                filled = bool(pairs) or d != w
                d = clean_desc(d)
                if pairs:
                    if len(pairs) != 1 or not NOUN.search(d):
                        continue
                    d = NOUN.sub(lambda m: pairs[0][1], d, count=1)
                noun = d[len(first) + 1:]
                if len(cmd) <= 70 and len(noun) <= 80 and noun.split(" ")[0] in ARTICLES:
                    return page, i, cmd, noun, filled
        raise Skip

    def worded(phrase):
        return framed(phrase if phrase[0].isupper() else f"{rng.choice(src_verbs)} {phrase}", rng)

    def shown(page, i):
        name, desc, ex = pages[page]
        return card_text(name, desc, trim(ex, i, min(4, len(ex)), rng))

    def pipeline():
        sp, si, scmd, noun, _ = listing()
        fp, fi, form, phrases = rng.choice(flt)
        if pages[fp][0] == pages[sp][0]:
            raise Skip
        _, fcmd, pairs = fill_example(pages[fp][2][fi][0], form, rng)
        task = worded(rng.choice(phrases).format(s=noun, v=pairs[0][1] if pairs else ""))
        cards = [shown(sp, si), shown(fp, fi)]
        if rng.random() < 0.5:
            cards.reverse()
        return msg("Reference cards:\n" + "\n\n".join(cards) + f"\n\nTask: {task}\n\n{one}", f"{scmd} | {fcmd}", sp, "two")

    def on_a_file():
        fp, fi, _, phrases = rng.choice(on_files)
        w, tpl = pages[fp][2][fi]
        _, cmd, pairs = fill_example(w, tpl, rng)
        phrases = [p for p in phrases if "that output" not in p] or phrases
        task = worded(rng.choice(phrases).format(s=" and ".join(pairs[-1][1].split(" ")), v=pairs[0][1]))
        return msg(f"Reference card:\n{shown(fp, fi)}\n\nTask: {task}\n\n{one}", cmd, fp, "slot")

    def listing_alone():
        sp, si, scmd, noun, filled = listing()
        task = framed(f"{rng.choice(src_verbs)} {noun}", rng)
        return msg(f"Reference card:\n{shown(sp, si)}\n\nTask: {task}\n\n{one}", scmd, sp, "slot" if filled else "direct")

    more = []
    for share, make in ((0.7, pipeline), (0.2, on_a_file), (0.1, listing_alone)):
        want, got, tries = int(pipes * share), 0, 0
        while got < want and tries < want * 60:
            tries += 1
            try:
                more.append(make())
                got += 1
            except Skip:
                pass
    rng.shuffle(more)
    return out + more


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tldr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=90000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--contrast", type=int, default=0,
                    help="append N tasks that take the wording of `two` away as a cue: its listing alone, "
                         "its filters applied to a file, pipelines in more wordings")
    ap.add_argument("--pipes", type=int, default=0,
                    help="append N tasks of pipelines over many commands: a listing of any page and a filter that reads "
                         "it (7 in 10), the same filter over a file (2 in 10), the listing alone (1 in 10)")
    args = ap.parse_args()
    rows = generate(Path(args.tldr), args.n, args.seed, args.contrast, args.pipes)
    seen, uniq = set(), []
    for r in rows:
        key = (r["messages"][0]["content"], r["messages"][1]["content"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in uniq))
    by = defaultdict(int)
    for r in uniq:
        by[r["cat"]] += 1
    chars = sum(len(r["messages"][0]["content"]) + len(r["messages"][1]["content"]) for r in uniq)
    print(f"{len(uniq)} tasks ({len(rows) - len(uniq)} duplicates dropped): " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    print(f"groups {len({r['group'] for r in uniq})}, {chars / 1e6:.1f} M chars")


if __name__ == "__main__":
    main()
