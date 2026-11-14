"""Task families of the R set ("reasoning over a card").

EVALUATION ONLY. Nothing here, and nothing generated from it, may enter a
training corpus: the cards, requests and parameters below are the held-out
side of the Hermes-core gates.

Every family is one procedure a controller has to carry out with a command
card in front of it. A family expands to a few instances (parameter sets x
phrasings). Categories:

  slot    one card: find the matching example, fill the values       (exec)
  os      one card with a section per OS: take the right section     (match, Linux exec where deterministic)
  choose  two or three cards: take the right one                     (exec)
  none    two or three cards, none fits: say NONE                    (label)
  react   rules + the previous command and its output: next action   (label)
  fix     a failed command and its error: the corrected command      (exec)
  two     two cards (or two examples of one): a two-step command     (exec)
  arith   a command's output: a number read or computed from it      (value)
  direct  one card, no values to fill: the right example verbatim    (match)

Cards are rendered from tldr-pages (CC-BY 4.0) by cards.py; `(page, keep)`
selects examples by index so that a three-card prompt stays short.
"""

from __future__ import annotations

FAMILIES: list[dict] = []


def fam(cat, name, cards, asks, ref=None, check="out", params=None, setup="", probe="",
        accept=None, limit=3, kind="command", context="", fold_case=False):
    FAMILIES.append(dict(cat=cat, name=name, cards=cards, asks=asks, ref=ref, check=check,
                         params=params or [{}], setup=setup, probe=probe, accept=accept,
                         limit=limit, kind=kind, context=context, fold_case=fold_case))


def P(page, keep=None, label=None):
    return (label, page, keep)


W = "/tmp/work"

# ───────────────────────────── slot ─────────────────────────────

fam("slot", "wc_lines", [P("common/wc")],
    ["How many lines does {f} have?", "Count the lines in {f}."],
    "wc -l {f}", "num", [{"f": "/data/app.log"}, {"f": "/data/users.csv"}])
fam("slot", "wc_words", [P("common/wc")],
    ["How many words are in {f}?", "Count the words of {f}."],
    "wc -w {f}", "num", [{"f": "/data/notes.txt"}, {"f": "/data/docs/report.txt"}])
fam("slot", "wc_bytes", [P("common/wc")],
    ["How many bytes does {f} contain?", "Count the bytes in {f}."],
    "wc -c {f}", "num", [{"f": "/data/config.ini"}, {"f": "/data/big/c.bin"}])
fam("slot", "head_n", [P("common/head")],
    ["Show the first {n} lines of {f}.", "Print only the top {n} lines of {f}."],
    "head -n {n} {f}", "out", [{"n": 3, "f": "/data/users.csv"}, {"n": 7, "f": "/data/app.log"}])
fam("slot", "tail_n", [P("common/tail")],
    ["Show the last {n} lines of {f}.", "Print the final {n} lines of {f}."],
    "tail -n {n} {f}", "out", [{"n": 4, "f": "/data/notes.txt"}, {"n": 12, "f": "/data/access.log"}])
fam("slot", "tail_from", [P("common/tail")],
    ["Print {f} starting from line {n}.", "Show everything in {f} from line {n} onward."],
    "tail -n +{n} {f}", "out", [{"n": 190, "f": "/data/app.log"}, {"n": 38, "f": "/data/users.csv"}])
fam("slot", "grep_pat", [P("common/grep")],
    ["Show the lines of {f} that contain {p}.", "Which lines in {f} mention {p}?"],
    'grep "{p}" {f}', "out", [{"p": "ERROR", "f": "/data/app.log"}, {"p": "Oslo", "f": "/data/users.csv"}])
fam("slot", "grep_ci", [P("common/grep")],
    ["Find {p} in {f}, ignoring upper and lower case.", "Search {f} for {p} case-insensitively."],
    'grep -Ei "{p}" {f}', "out", [{"p": "todo", "f": "/data/notes.txt"}, {"p": "error", "f": "/data/app.log"}])
fam("slot", "grep_rec", [P("common/grep")],
    ["Search all files under {d} for {p}.", "Look for {p} in every file below the directory {d}."],
    'grep -rI "{p}" {d}', "outset", [{"p": "TODO", "d": "/data/src"}, {"p": "revenue", "d": "/data/docs"}])
fam("slot", "grep_v", [P("common/grep")],
    ["Show the lines of {f} that do not contain {p}.", "Print {f} without the lines that contain {p}."],
    'grep -v "{p}" {f}', "out", [{"p": "INFO", "f": "/data/app.log"}, {"p": "ok", "f": "/data/logs/app.5.log"}])
fam("slot", "find_ext", [P("common/find")],
    ["Find all {e} files under {d}.", "List every file below {d} whose name ends in {e}."],
    "find {d} -name '*{e}'", "outset", [{"e": ".py", "d": "/data/src"}, {"e": ".log", "d": "/data/logs"}])
fam("slot", "find_size", [P("common/find")],
    ["Find the files directly inside {d} that are larger than {s}.",
     "Which files in {d} itself, not in subdirectories, exceed {s}?"],
    "find {d} -maxdepth 1 -size +{s}", "outset", [{"s": "500k", "d": "/data/big"}, {"s": "3k", "d": "/data/logs"}])
fam("slot", "sort_num", [P("common/sort")],
    ["Sort {f} numerically.", "Print the lines of {f} in numeric order."],
    "sort -n {f}", "out", [{"f": "/data/numbers.txt"}], limit=2)
fam("slot", "sort_rev", [P("common/sort")],
    ["Sort {f} in descending order.", "Print the lines of {f} from Z to A."],
    "sort -r {f}", "out", [{"f": "/data/names.txt"}], limit=2)
fam("slot", "sort_uniq", [P("common/sort")],
    ["Sort {f} and keep each line only once.", "Print the distinct lines of {f} in sorted order."],
    "sort -u {f}", "out", [{"f": "/data/names.txt"}], limit=2)
fam("slot", "uniq_count", [P("common/uniq")],
    ["Show how many times each line occurs in {f}.", "Count the occurrences of every distinct line in {f}."],
    "sort {f} | uniq -c", "out", [{"f": "/data/names.txt"}], limit=2)
fam("slot", "cut_field", [P("common/cut")],
    ["Print the {n} column of the comma-separated file {f}.",
     "{f} is comma-separated. Print only its {n} field."],
    'cut -d "," -f {k} {f}', "out",
    [{"n": "second", "k": 2, "f": "/data/users.csv"}, {"n": "third", "k": 3, "f": "/data/users.csv"}])
fam("slot", "sed_replace", [P("common/sed")],
    ["Print {f} with every {a} replaced by {b}.", "Output the contents of {f}, changing each {a} to {b}."],
    "sed 's/{a}/{b}/g' {f}", "out",
    [{"a": "pear", "b": "grape", "f": "/data/names.txt"}, {"a": "ok", "b": "pass", "f": "/data/logs/app.5.log"}])
fam("slot", "tr_upper", [P("common/tr")],
    ["Print {f} in upper case.", "Show the contents of {f} converted to capital letters."],
    'tr <{f} "[:lower:]" "[:upper:]"', "out", [{"f": "/data/notes.txt"}, {"f": "/data/docs/report.txt"}])
fam("slot", "cp_file", [P("common/cp")],
    ["Copy {s} to {t}.", "Make a copy of {s} named {t}."],
    "cp {s} {t}", "post", [{"s": "/data/notes.txt", "t": W + "/notes.bak"}, {"s": "/data/config.ini", "t": W + "/config.old"}],
    probe="md5sum {t}")
fam("slot", "cp_dir", [P("common/cp")],
    ["Copy the directory {s} with everything in it to {t}.", "Duplicate the whole directory {s} as {t}."],
    "cp -r {s} {t}", "post", [{"s": "/data/docs", "t": W + "/docs"}, {"s": "/data/site", "t": W + "/site2"}],
    probe="cd {t} && find . | sort")
fam("slot", "mv_rename", [P("common/mv")],
    ["Rename {s} to {t}.", "Give {s} the new name {t}."],
    "mv {s} {t}", "post", [{"s": W + "/a.txt", "t": W + "/b.txt"}, {"s": W + "/draft.md", "t": W + "/final.md"}],
    setup="cp /data/notes.txt {s}", probe="ls -A " + W)
fam("slot", "rm_dir", [P("common/rm")],
    ["Delete the directory {d} and everything inside it.", "Remove {d} together with its contents."],
    "rm -r {d}", "post", [{"d": W + "/cache"}, {"d": W + "/build"}],
    setup="mkdir -p {d}/x && touch {d}/x/f {d}/g " + W + "/keep", probe="ls -A " + W)
fam("slot", "mkdir_p", [P("common/mkdir")],
    ["Create the directory {d}, including any missing parent directories.",
     "Make the nested directory {d}; its parents do not exist yet."],
    "mkdir -p {d}", "post", [{"d": W + "/a/b/c"}, {"d": W + "/reports/2026/q1"}], probe="find " + W + " | sort")
fam("slot", "chmod_octal", [P("common/chmod")],
    ["Set the permissions of {f} to {m}.", "Change the mode of {f} to {m}."],
    "chmod {m} {f}", "post", [{"m": 750, "f": W + "/run.sh"}, {"m": 644, "f": W + "/key.pem"}],
    setup="cp /data/notes.txt {f}; chmod 600 {f}", probe="stat -c %a {f}")
fam("slot", "chmod_ux", [P("common/chmod")],
    ["Make {f} executable for its owner.", "Give the owner of {f} permission to execute it."],
    "chmod u+x {f}", "post", [{"f": W + "/run.sh"}], setup="cp /data/notes.txt {f}; chmod 600 {f}",
    probe="stat -c %A {f}", limit=2)
fam("slot", "ln_s", [P("common/ln")],
    ["Create a symbolic link {l} that points to {t}.", "Make {l} a symlink to {t}."],
    "ln -s {t} {l}", "post", [{"l": W + "/latest", "t": "/data/logs/app.5.log"}, {"l": W + "/cfg", "t": "/data/config.ini"}],
    probe="readlink {l}")
fam("slot", "tar_create", [P("common/tar")],
    ["Create a gzipped tar archive {a} of {d}.", "Pack {d} into the gzip-compressed tarball {a}."],
    "tar czf {a} {d}", "post", [{"a": W + "/docs.tar.gz", "d": "/data/docs"}, {"a": W + "/src.tgz", "d": "/data/src"}],
    probe="tar tzf {a} | awk -F/ '{print $NF}' | grep -v '^$' | sort")
fam("slot", "tar_extract", [P("common/tar")],
    ["Extract {a} into the directory {d}.", "Unpack the archive {a} to {d}."],
    "tar xf {a} -C {d}", "post", [{"a": "/data/backup.tar.gz", "d": W + "/out"}],
    setup="mkdir -p {d}", probe="cd {d} && find . -type f | sort", limit=2)
fam("slot", "tar_list", [P("common/tar")],
    ["List the contents of the archive {a}.", "Show which files are inside {a} without extracting it."],
    "tar tvf {a}", {"type": "has", "tokens": ["docs/report.txt", "docs/summary.md", "docs/sub/old.txt"]},
    [{"a": "/data/backup.tar.gz"}], limit=2)
fam("slot", "gzip_keep", [P("common/gzip")],
    ["Compress {f} with gzip but keep the original file.", "Gzip {f} and leave the uncompressed file in place."],
    "gzip -k {f}", "post", [{"f": W + "/app.log"}], setup="cp /data/app.log {f}", probe="ls -A " + W, limit=2)
fam("slot", "gzip_d", [P("common/gzip")],
    ["Decompress {f}.", "Unpack the gzip file {f}."],
    "gzip -d {f}", "post", [{"f": W + "/old.log.gz"}], setup="cp /data/logs/old.log.gz {f}",
    probe="ls -A " + W + "; wc -l < " + W + "/old.log", limit=2)
fam("slot", "unzip_to", [P("common/unzip")],
    ["Extract {a} into {d}.", "Unpack the zip archive {a} to the directory {d}."],
    "unzip {a} -d {d}", "post", [{"a": "/data/site.zip", "d": W + "/site_out"}],
    probe="cd {d} && find . -type f | sort", limit=2)
fam("slot", "du_sh", [P("common/du")],
    ["Show the total size of the directory {d} in human-readable units.",
     "How much disk space does {d} take in total? Use human-readable units."],
    "du -sh {d}", "out", [{"d": "/data/big"}, {"d": "/data/logs"}])
fam("slot", "cat_concat", [P("common/cat")],
    ["Concatenate {a} and {b} into the new file {o}.", "Join {a} and {b}, in that order, and save the result as {o}."],
    "cat {a} {b} >{o}", "post",
    [{"a": "/data/docs/report.txt", "b": "/data/docs/summary.md", "o": W + "/all.txt"},
     {"a": "/data/logs/app.5.log", "b": "/data/logs/app.1.log", "o": W + "/joined.log"}],
    probe="md5sum {o}")
fam("slot", "awk_col", [P("common/awk", [0, 1, 2, 3])],
    ["Print the {n} space-separated column of {f}.", "From {f}, show only field number {k}; fields are separated by spaces."],
    "awk '{print ${k}}' {f}", "out",
    [{"n": "third", "k": 3, "f": "/data/app.log"}, {"n": "first", "k": 1, "f": "/data/access.log"}])
fam("slot", "awk_sum", [P("common/awk", [0, 1, 2, 3])],
    ["Add up the numbers in {f} and print the total.", "{f} has one number per line. What is their sum?"],
    "awk '{s+=$1} END {print s}' {f}", "num", [{"f": "/data/numbers.txt"}], limit=2)
fam("slot", "awk_last_csv", [P("common/awk", [0, 1, 2, 3])],
    ["Print the last column of the comma-separated file {f}.", "{f} uses commas between fields. Show the final field of each line."],
    "awk -F ',' '{print $NF}' {f}", "out", [{"f": "/data/users.csv"}], limit=2)
fam("slot", "basename", [P("common/basename")],
    ["Print only the file name part of the path {p}.", "Strip the directories from {p}."],
    "basename {p}", "out", [{"p": "/data/docs/sub/old.txt"}], limit=2)
fam("slot", "dirname", [P("common/dirname")],
    ["Print the parent directory of {p}.", "Which directory contains {p}? Print its path."],
    "dirname {p}", "out", [{"p": "/data/src/main.py"}], limit=2)
fam("slot", "seq_range", [P("common/seq")],
    ["Print the numbers from {a} to {b}.", "Output the sequence {a} through {b}, one number per line."],
    "seq {a} {b}", "out", [{"a": 3, "b": 9}, {"a": 15, "b": 22}])
fam("slot", "nl", [P("common/nl")],
    ["Print {f} with line numbers.", "Number the lines of {f}."],
    "nl {f}", "out", [{"f": "/data/notes.txt"}], limit=2)
fam("slot", "sha256", [P("common/sha256sum")],
    ["Compute the SHA256 checksum of {f}.", "What is the SHA-256 hash of {f}?"],
    "sha256sum {f}", "hash", [{"f": "/data/users.csv"}, {"f": "/data/config.ini"}])
fam("slot", "touch", [P("common/touch")],
    ["Create an empty file {f}.", "Make a new empty file at {f}."],
    "touch {f}", "post", [{"f": W + "/done.flag"}], probe="ls -A " + W + "; wc -c < {f}", limit=2)
fam("slot", "echo_append", [P("common/echo")],
    ['Append the line "{t}" to {f}.', 'Add "{t}" as a new last line of {f}.'],
    'echo "{t}" >>{f}', "post", [{"t": "backup ok", "f": W + "/status.txt"}, {"t": "step 2 done", "f": W + "/progress.log"}],
    setup="printf 'first line\\n' > {f}", probe="cat {f}")
fam("slot", "bc", [P("common/bc")],
    ["Use bc to calculate {e}.", "Compute {e} with the bc calculator."],
    "echo '{e}' | bc", "num", [{"e": "17 * 23"}, {"e": "1024 - 389"}])
fam("slot", "tac", [P("common/tac")],
    ["Print {f} with its lines in reverse order.", "Show {f} from the last line to the first."],
    "tac {f}", "out", [{"f": "/data/notes.txt"}], limit=2)
fam("slot", "paste_join", [P("common/paste")],
    ["Join all lines of {f} into a single line, separated by commas.", "Put every line of {f} on one line with commas between them."],
    "paste -s -d , {f}", "out", [{"f": "/data/numbers.txt"}], limit=2)
fam("slot", "stat_size", [P("common/stat")],
    ["Show the size of {f} in bytes.", "How many bytes is {f}?"],
    'stat -c "%s %n" {f}', "num", [{"f": "/data/big/b.bin"}], limit=2)

# ───────────────────────────── os ─────────────────────────────
# One card with a section per OS. Each entry of `tasks`: (os, ask, accepted
# commands, params). macOS and Windows are match-checked.

OS: list[dict] = []


def osfam(name, pages, tasks):
    OS.append(dict(name=name, pages=pages, tasks=tasks))


L, M, WIN = "Linux", "macOS", "Windows"

osfam("clipboard_file",
      {L: P("linux/xclip", [1, 3]), M: P("osx/pbcopy"), WIN: P("windows/clip", [0, 1])},
      [(L, "copy the contents of the file {f} to the system clipboard",
        ["xclip -se c {f}", "xclip -selection clipboard {f}", "xclip -sel c {f}", "xclip -se clipboard {f}",
         "xclip -sel clipboard {f}", "xclip -selection c {f}"],
        [{"f": "/home/anna/todo.txt"}]),
       (M, "copy the contents of the file {f} to the clipboard", ["pbcopy <{f}", "cat {f} | pbcopy"],
        [{"f": "/Users/anna/todo.txt"}]),
       (WIN, "copy the contents of the file {f} to the clipboard", ["clip <{f}", "type {f} | clip"],
        [{"f": r"C:\Users\anna\todo.txt"}])])
osfam("open_here",
      {L: P("linux/xdg-open", [0, 1]), M: P("osx/open", [0, 3]), WIN: P("windows/explorer")},
      [(L, "open the current directory in the graphical file manager", ["xdg-open ."], [{}]),
       (M, "open the current directory in the graphical file manager", ["open ."], [{}]),
       (WIN, "open the current directory in the graphical file manager", ["explorer ."], [{}])])
osfam("install_pkg",
      {L: P("linux/apt", [1, 4, 5]), M: P("common/brew", [0, 2, 5]), WIN: P("windows/winget", [0, 1, 3])},
      [(L, "install the package {p}", ["sudo apt install {p}", "sudo apt install -y {p}"], [{"p": "htop"}, {"p": "ripgrep"}]),
       (M, "install the package {p}", ["brew install {p}"], [{"p": "htop"}, {"p": "ripgrep"}]),
       (WIN, "install the package {p}", ["winget add {p}", "winget install {p}"], [{"p": "htop"}, {"p": "ripgrep"}])])
osfam("kill_name",
      {L: P("common/pkill", [0, 1, 2]), WIN: P("windows/taskkill", [0, 1, 2])},
      [(L, "terminate the process named {n}", ["pkill {n}"], [{"n": "nginx"}, {"n": "firefox"}]),
       (WIN, "terminate the process named {n}", ["taskkill /im {n}"], [{"n": "notepad.exe"}, {"n": "chrome.exe"}])])
osfam("list_procs",
      {L: P("common/ps", [0, 1]), WIN: P("windows/tasklist", [0, 1, 4])},
      [(L, "list all running processes", ["ps aux"], [{}]),
       (WIN, "list the currently running processes", ["tasklist"], [{}])])
osfam("net_detail",
      {L: P("linux/ip", [0, 1, 3]), M: P("common/ifconfig", [0, 1]), WIN: P("windows/ipconfig", [0, 1, 5])},
      [(L, "list the network interfaces with detailed information", ["ip a", "ip addr", "ip address"], [{}]),
       (M, "show the details of all network interfaces, disabled ones included", ["ifconfig -a"], [{}]),
       (WIN, "show the detailed list of network adapters", ["ipconfig /all"], [{}])])
osfam("flush_dns",
      {L: P("linux/resolvectl", [0, 3, 4]), WIN: P("windows/ipconfig", [0, 4, 5])},
      [(L, "clear the local DNS cache", ["resolvectl flush-caches"], [{}]),
       (WIN, "clear the local DNS cache", ["ipconfig /flushdns"], [{}])])
osfam("start_service",
      {L: P("linux/systemctl", [0, 2, 3]), M: P("common/brew-services", [0, 2, 3, 4]), WIN: P("windows/sc", [0, 1, 2])},
      [(L, "start the service {s}", ["systemctl start {s}"], [{"s": "nginx"}, {"s": "postgresql"}]),
       (L, "restart the service {s}", ["systemctl restart {s}"], [{"s": "nginx"}]),
       (M, "start the service {s}", ["brew services start {s}"], [{"s": "nginx"}, {"s": "postgresql"}]),
       (M, "restart the service {s}", ["brew services restart {s}"], [{"s": "redis"}]),
       (WIN, "start the service {s}", ["sc start {s}"], [{"s": "Spooler"}, {"s": "wuauserv"}]),
       (WIN, "stop the service {s}", ["sc stop {s}"], [{"s": "Spooler"}])])
osfam("os_version",
      {L: P("common/uname", [0, 1, 3]), M: P("osx/sw_vers"), WIN: P("windows/ver")},
      [(L, "print all available system information", ["uname -a"], [{}]),
       (M, "print only the version number of the operating system", ["sw_vers -productVersion"], [{}]),
       (WIN, "show the version number of the operating system", ["ver"], [{}])])
osfam("port_listen",
      {L: P("linux/ss", [2, 3, 4]), M: P("common/lsof", [0, 1, 2])},
      [(L, "show the TCP sockets listening on local port {p}", ["ss -lt src :{p}", "ss -tl src :{p}"], [{"p": 5432}, {"p": 3000}]),
       (M, "find the process that opened the local port {p}", ["lsof -i :{p}"], [{"p": 5432}, {"p": 3000}])])
osfam("reboot",
      {L: P("linux/shutdown", [0, 1, 4]), M: P("osx/shutdown", [0, 1, 2]), WIN: P("windows/shutdown", [0, 2, 4])},
      [(L, "reboot the machine right now", ["shutdown -r now"], [{}]),
       (L, "power the machine off right now", ["shutdown -h now"], [{}]),
       (M, "put the machine to sleep right now", ["shutdown -s now"], [{}]),
       (WIN, "restart the machine immediately", ["shutdown /r /t 0"], [{}]),
       (WIN, "shut the machine down", ["shutdown /s"], [{}])])
osfam("copy_file",
      {L: P("common/cp", [0, 1, 2]), WIN: P("windows/copy", [0, 1, 2])},
      [(L, "copy the file {s} to {t}", ["cp {s} {t}"], [{"s": "/home/anna/a.txt", "t": "/home/anna/b.txt"}]),
       (WIN, "copy the file {s} to {t}", ["copy {s} {t}"], [{"s": r"C:\work\a.txt", "t": r"C:\work\b.txt"}])])
osfam("show_file",
      {L: P("common/cat", [0, 1]), WIN: P("windows/type")},
      [(L, "print the contents of the file {f}", ["cat {f}"], [{"f": "/etc/hosts"}]),
       (WIN, "print the contents of the file {f}", ["type {f}"], [{"f": r"C:\work\readme.txt"}])])
osfam("delete_dir",
      {L: P("common/rm", [0, 4, 5]), WIN: P("windows/rmdir")},
      [(L, "delete the directory {d} and all its contents", ["rm -r {d}", "rm -rf {d}"], [{"d": "/home/anna/old"}]),
       (WIN, "delete the directory {d} and all its contents", ["rmdir {d} /s", "rmdir {d} /s /q", "rmdir /s {d}", "rmdir /s /q {d}"],
        [{"d": r"C:\work\old"}])])
osfam("env_vars",
      {L: P("common/env", [0, 1]), WIN: P("windows/set", [0, 1, 2])},
      [(L, "list all environment variables", ["env"], [{}]),
       (WIN, "list all environment variables", ["set"], [{}])])
osfam("disks",
      {L: P("linux/lsblk", [0, 1, 3]), M: P("osx/diskutil", [0, 1, 2])},
      [(L, "list all storage devices", ["lsblk"], [{}]),
       (M, "list all disks, partitions and mounted volumes", ["diskutil list"], [{}])])
osfam("memory",
      {L: P("linux/free", [0, 2, 4]), M: P("osx/vm_stat")},
      [(L, "show memory usage in human-readable units", ["free -h"], [{}]),
       (M, "show virtual memory statistics", ["vm_stat"], [{}])])
osfam("updates",
      {L: P("linux/apt", [0, 4, 6]), M: P("osx/softwareupdate", [0, 1, 2]), WIN: P("windows/winget", [0, 4, 5])},
      [(L, "upgrade all installed packages", ["sudo apt upgrade", "sudo apt upgrade -y"], [{}]),
       (M, "download and install all available updates", ["softwareupdate -i -a", "softwareupdate -ia"], [{}]),
       (WIN, "upgrade all packages to the latest versions", ["winget upgrade -r", "winget upgrade --all"], [{}])])

# ───────────────────────────── choose / none ─────────────────────────────

K4 = [0, 1, 2, 3]
fam("choose", "c_wc", [P("common/du", [1, 2]), P("common/wc", K4), P("common/df", [0, 1])],
    ["How many lines are in {f}?"], "wc -l {f}", "num", [{"f": "/data/access.log"}, {"f": "/data/names.txt"}])
fam("choose", "c_sort_rev", [P("common/uniq", [0, 1, 2]), P("common/cut", [1, 2]), P("common/sort", K4)],
    ["Print {f} sorted in descending order."], "sort -r {f}", "out", [{"f": "/data/names.txt"}, {"f": "/data/notes.txt"}])
fam("choose", "c_head", [P("common/tail", [0, 2, 3]), P("common/head"), P("common/cat", [0, 1])],
    ["Show the first {n} lines of {f}."], "head -n {n} {f}", "out", [{"n": 5, "f": "/data/notes.txt"}, {"n": 2, "f": "/data/config.ini"}])
fam("choose", "c_find", [P("common/grep", [0, 1, 2]), P("common/find", [0, 2, 4]), P("common/sed")],
    ["Find all {e} files under {d}."], "find {d} -name '*{e}'", "outset", [{"e": ".md", "d": "/data"}, {"e": ".txt", "d": "/data/docs"}])
fam("choose", "c_mv_into", [P("common/cp", [0, 1, 2]), P("common/ln", [0, 1]), P("common/mv", [0, 1, 2])],
    ["Move {s} into the existing directory {d}."], "mv {s} {d}", "post", [{"s": W + "/a.txt", "d": W + "/arch"}, {"s": W + "/x.log", "d": W + "/logs"}],
    setup="mkdir -p {d}; cp /data/notes.txt {s}", probe="find " + W + " | sort")
fam("choose", "c_unzip_list", [P("common/tar", [1, 4, 6]), P("common/gzip", [0, 1, 3]), P("common/unzip", [0, 1, 4])],
    ["Show what is inside {a} without extracting it."], "unzip -l {a}",
    {"type": "has", "tokens": ["site/index.html", "site/css/main.css"]}, [{"a": "/data/site.zip"}], limit=1)
fam("choose", "c_touch", [P("common/mkdir", [0, 1]), P("common/rmdir", [0, 1]), P("common/touch", [0, 4])],
    ["Create an empty file {f}."], "touch {f}", "post", [{"f": W + "/lock"}, {"f": W + "/ready.txt"}], probe="ls -A " + W + "; wc -c < {f}")
fam("choose", "c_stat_perm", [P("common/chmod", [0, 6]), P("common/stat", [0, 3, 4, 5]), P("common/ls", [3, 4])],
    ["Show the octal permissions of {f}."], 'stat -c "%a %n" {f}', "num", [{"f": "/data/config.ini"}], limit=1)
fam("choose", "c_dirname", [P("common/basename"), P("common/realpath", [0, 1]), P("common/dirname", [0, 1])],
    ["Print the directory part of the path {p}."], "dirname {p}", "out", [{"p": "/data/src/util.py"}, {"p": "/data/docs/sub/old.txt"}])
fam("choose", "c_seq_step", [P("common/nl", [0, 1]), P("common/seq", K4), P("common/shuf", [0, 1])],
    ["Print every {s} number from {a} to {b}, starting at {a}."], "seq {a} {k} {b}", "out",
    [{"s": "2nd", "k": 2, "a": 4, "b": 12}, {"s": "5th", "k": 5, "a": 10, "b": 40}])
fam("choose", "c_md5", [P("common/sha256sum", [0, 1]), P("common/wc", [0, 2]), P("common/md5sum", [0, 1])],
    ["Compute the MD5 checksum of {f}."], "md5sum {f}", "hash", [{"f": "/data/numbers.txt"}, {"f": "/data/names.txt"}])
fam("choose", "c_tac", [P("common/sort", [0, 1]), P("common/head"), P("common/tac", [0, 1])],
    ["Print {f} bottom-up: the last line first."], "tac {f}", "out", [{"f": "/data/numbers.txt"}, {"f": "/data/config.ini"}])
fam("choose", "c_grep", [P("common/wc", [0, 1]), P("common/sort", [0, 1]), P("common/grep", [0, 1, 5])],
    ["Show the lines of {f} that contain {p}."], 'grep "{p}" {f}', "out", [{"p": "404", "f": "/data/access.log"}, {"p": "Vienna", "f": "/data/users.csv"}])
fam("choose", "c_du", [P("common/ls", [3, 4]), P("common/find", [0, 4]), P("common/du", [1, 2, 3])],
    ["Show the total size of the directory {d} in human-readable units."], "du -sh {d}", "out", [{"d": "/data/src"}, {"d": "/data/docs"}])
fam("choose", "c_gzip", [P("common/tar", [1, 4]), P("common/zip", [0, 1]), P("common/gzip", [0, 1, 3])],
    ["Compress {f} with gzip, replacing the file with the archive."], "gzip {f}", "post", [{"f": W + "/notes.txt"}],
    setup="cp /data/notes.txt {f}", probe="ls -A " + W, limit=1)
fam("choose", "c_rmdir", [P("common/mv", [0, 1]), P("common/rmdir", [0, 1]), P("common/mkdir", [0, 1])],
    ["Remove the empty directory {d}."], "rmdir {d}", "post", [{"d": W + "/tmpdir"}], setup="mkdir -p {d}; touch " + W + "/keep",
    probe="ls -A " + W, limit=1)

NONE_TASKS = [
    (["common/wc", "common/sort", "common/head"], "Compress /data/app.log with gzip."),
    (["common/cp", "common/mv", "common/rm"], "Show how much memory is free."),
    (["common/grep", "common/sed", "common/tr"], "Create a symbolic link /tmp/work/x that points to /data/notes.txt."),
    (["common/tar", "common/gzip", "common/unzip"], "Print the SHA256 checksum of /data/users.csv."),
    (["common/ls", "common/du", "common/find"], "Change the permissions of /tmp/work/run.sh to 755."),
    (["common/mkdir", "common/touch", "common/rmdir"], "Count the words in /data/notes.txt."),
    (["common/seq", "common/nl", "common/tac"], "Terminate the process named nginx."),
    (["common/chmod", "common/ln", "common/stat"], "Sort /data/numbers.txt numerically."),
    (["common/head", "common/tail", "common/cat"], "Install the package htop."),
    (["common/sha256sum", "common/md5sum", "common/basename"], "Show the last 3 lines of /data/app.log."),
    (["common/uniq", "common/cut", "common/paste"], "Extract /data/backup.tar.gz into /tmp/work."),
    (["common/date", "common/seq", "common/bc"], "List the files in /data/logs."),
    (["common/pkill", "common/kill", "common/ps"], "Copy /data/notes.txt to /tmp/work/n.txt."),
    (["common/find", "common/grep", "common/wc"], "Reboot the machine."),
    (["common/rm", "common/mkdir", "common/cp"], "Print the current date as a Unix timestamp."),
    (["common/zip", "common/unzip", "common/tar"], "Print /data/names.txt with every pear replaced by plum."),
]

# ───────────────────────────── react ─────────────────────────────

RULES = """Rules for the next action:
- If the output contains "Permission denied", answer SUDO.
- If the output contains "No such file or directory", answer ASK.
- If the output contains "command not found", answer INSTALL.
- If the output contains "Connection refused" or "timed out", answer RETRY.
- Otherwise answer DONE."""

REACT = [
    ("SUDO", "cat /etc/shadow", "cat: /etc/shadow: Permission denied"),
    ("SUDO", "mkdir /opt/app", "mkdir: cannot create directory '/opt/app': Permission denied"),
    ("SUDO", "touch /var/log/app.log", "touch: cannot touch '/var/log/app.log': Permission denied"),
    ("SUDO", "cp nginx.conf /etc/nginx/nginx.conf", "cp: cannot create regular file '/etc/nginx/nginx.conf': Permission denied"),
    ("SUDO", "ls /root", "ls: cannot open directory '/root': Permission denied"),
    ("ASK", "cat /data/report.pdf", "cat: /data/report.pdf: No such file or directory"),
    ("ASK", "cd /srv/www", "bash: cd: /srv/www: No such file or directory"),
    ("ASK", "ls /mnt/backup", "ls: cannot access '/mnt/backup': No such file or directory"),
    ("ASK", "wc -l notes.md", "wc: notes.md: No such file or directory"),
    ("ASK", "tar xf site.tar", "tar: site.tar: Cannot open: No such file or directory\ntar: Error is not recoverable: exiting now"),
    ("INSTALL", "htop", "bash: htop: command not found"),
    ("INSTALL", "jq . data.json", "bash: jq: command not found"),
    ("INSTALL", "rg TODO src", "bash: rg: command not found"),
    ("INSTALL", "docker ps", "zsh: command not found: docker"),
    ("INSTALL", "tree /data", "bash: tree: command not found"),
    ("RETRY", "curl http://localhost:8080/health", "curl: (7) Failed to connect to localhost port 8080 after 0 ms: Connection refused"),
    ("RETRY", "ssh db1 uptime", "ssh: connect to host db1 port 22: Connection timed out"),
    ("RETRY", "psql -h db.internal -c 'select 1'", 'psql: error: connection to server at "db.internal", port 5432 failed: Connection refused'),
    ("RETRY", "wget http://mirror.local/pkg.tar", "Connecting to mirror.local:80... failed: Connection timed out."),
    ("RETRY", "nc -z cache 6379", "nc: connect to cache port 6379 (tcp) failed: Connection refused"),
    ("DONE", "ls /data/docs", "report.txt  sub  summary.md"),
    ("DONE", "wc -l /data/app.log", "200 /data/app.log"),
    ("DONE", "grep -c ERROR /data/app.log", "20"),
    ("DONE", "systemctl is-active nginx", "active"),
    ("DONE", "df -h /", "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1        50G   21G   27G  44% /"),
]

# ───────────────────────────── fix ─────────────────────────────

FAILED = "The command `{bad}` failed with:\n{err}"

fam("fix", "f_mkdir_p", [P("common/mkdir")], ["Give the corrected command."], "mkdir -p {d}", "post",
    [{"d": W + "/a/b/c"}, {"d": W + "/var/cache/app"}], probe="find " + W + " | sort", limit=2,
    context=FAILED.replace("{bad}", "mkdir {d}").replace("{err}", "mkdir: cannot create directory '{d}': No such file or directory"))
fam("fix", "f_rm_r", [P("common/rm")], ["Give the corrected command."], "rm -r {d}", "post",
    [{"d": W + "/cache"}, {"d": W + "/old_build"}], setup="mkdir -p {d}/x && touch {d}/x/f " + W + "/keep", probe="ls -A " + W, limit=2,
    context=FAILED.replace("{bad}", "rm {d}").replace("{err}", "rm: cannot remove '{d}': Is a directory"))
fam("fix", "f_cp_r", [P("common/cp")], ["Give the corrected command."], "cp -r {s} {t}", "post",
    [{"s": "/data/docs", "t": W + "/docs"}, {"s": "/data/site", "t": W + "/www"}], probe="cd {t} && find . | sort", limit=2,
    context=FAILED.replace("{bad}", "cp {s} {t}").replace("{err}", "cp: -r not specified; omitting directory '{s}'"))
fam("fix", "f_grep_r", [P("common/grep")], ["Give the corrected command."], 'grep -rI "{p}" {d}', "outset",
    [{"p": "TODO", "d": "/data/src"}, {"p": "costs", "d": "/data/docs"}], limit=2,
    context=FAILED.replace("{bad}", "grep {p} {d}").replace("{err}", "grep: {d}: Is a directory"))
fam("fix", "f_rmdir_full", [P("common/rmdir"), P("common/rm")],
    ["The directory and its contents are not needed. Give a command that removes it."], "rm -r {d}", "post",
    [{"d": W + "/tmpdata"}, {"d": W + "/session"}], setup="mkdir -p {d}/x && touch {d}/x/f " + W + "/keep", probe="ls -A " + W, limit=2,
    context=FAILED.replace("{bad}", "rmdir {d}").replace("{err}", "rmdir: failed to remove '{d}': Directory not empty"))
fam("fix", "f_chmod", [P("common/chmod")],
    ["The script must be runnable by its owner. Give the command that fixes this."], "chmod u+x {f}", "post",
    [{"f": W + "/run.sh"}, {"f": W + "/deploy.sh"}], setup="printf '#!/bin/sh\\necho hi\\n' > {f}; chmod 600 {f}",
    probe="stat -c %A {f} | cut -c1-4", limit=2,
    context=FAILED.replace("{bad}", "{f}").replace("{err}", "bash: {f}: Permission denied"))
fam("fix", "f_sort_n", [P("common/sort")],
    ["The numbers must come out in numeric order. Give the corrected command."], "sort -n {f}", "out",
    [{"f": "/data/numbers.txt"}], limit=1,
    context="The command `sort {f}` printed the numbers in the wrong order: 100 came before 23.")
fam("fix", "f_tar_dir", [P("common/tar", [1, 3, 4, 6]), P("common/mkdir", [0, 1])],
    ["Give one command line that makes the extraction succeed."], "mkdir -p {d} && tar xf {a} -C {d}", "post",
    [{"a": "/data/backup.tar.gz", "d": W + "/out"}, {"a": "/data/backup.tar.gz", "d": W + "/restore/docs"}],
    probe="cd {d} && find . -type f | sort", limit=2,
    context=FAILED.replace("{bad}", "tar xf {a} -C {d}").replace("{err}", "tar: {d}: Cannot open: No such file or directory\ntar: Error is not recoverable: exiting now"))

LS_DATA = "access.log  app.log  backup.tar.gz  big  config.ini  docs  empty  logs  names.txt  notes.txt  numbers.txt  site  site.zip  src  users.csv"
LS_LOGS = "app.1.log  app.2.log  app.3.log  app.4.log  app.5.log  old.log.gz"
TYPO = "The command `{bad}` failed with:\n{err}\n\nOutput of `ls {dir}`:\n{ls}"
fam("fix", "f_typo_cat", [P("common/cat", [0, 1])], ["Give the corrected command."], "cat /data/notes.txt", "out", limit=1,
    context=TYPO.replace("{bad}", "cat /data/note.txt").replace("{err}", "cat: /data/note.txt: No such file or directory").replace("{dir}", "/data").replace("{ls}", LS_DATA))
fam("fix", "f_typo_wc", [P("common/wc", K4)], ["Give the corrected command."], "wc -l /data/users.csv", "num", limit=1,
    context=TYPO.replace("{bad}", "wc -l /data/user.csv").replace("{err}", "wc: /data/user.csv: No such file or directory").replace("{dir}", "/data").replace("{ls}", LS_DATA))
fam("fix", "f_typo_head", [P("common/head")], ["Give the corrected command."], "head -n 3 /data/logs/app.1.log", "out", limit=1,
    context=TYPO.replace("{bad}", "head -n 3 /data/logs/app1.log").replace("{err}", "head: cannot open '/data/logs/app1.log' for reading: No such file or directory").replace("{dir}", "/data/logs").replace("{ls}", LS_LOGS))
fam("fix", "f_typo_tail", [P("common/tail", [0, 2, 3])], ["Give the corrected command."], "tail -n 2 /data/docs/report.txt", "out", limit=1,
    context=TYPO.replace("{bad}", "tail -n 2 /data/doc/report.txt").replace("{err}", "tail: cannot open '/data/doc/report.txt' for reading: No such file or directory").replace("{dir}", "/data").replace("{ls}", LS_DATA))

# ───────────────────────────── two ─────────────────────────────

fam("two", "t_grep_wc", [P("common/grep", [0, 1, 5]), P("common/wc", K4)],
    ["How many lines of {f} contain {p}?"], 'grep "{p}" {f} | wc -l', "num",
    [{"p": "ERROR", "f": "/data/app.log"}, {"p": "404", "f": "/data/access.log"}, {"p": "TODO", "f": "/data/notes.txt"}])
fam("two", "t_sort_wc", [P("common/sort", [0, 1, 3, 6]), P("common/wc", K4)],
    ["How many different lines does {f} contain?"], "sort -u {f} | wc -l", "num",
    [{"f": "/data/names.txt"}, {"f": "/data/numbers.txt"}])
fam("two", "t_top_freq", [P("common/uniq", [0, 2, 4, 5]), P("common/head")],
    ["Which line occurs most often in {f}? Show it with its count."], "sort {f} | uniq -c | sort -nr | head -n 1", "out",
    [{"f": "/data/names.txt"}], limit=1)
fam("two", "t_largest", [P("common/sort", [0, 1, 3, 6]), P("common/head")],
    ["Print the {n} largest numbers in {f}."], "sort -nr {f} | head -n {n}", "outset",
    [{"n": 5, "f": "/data/numbers.txt"}, {"n": 3, "f": "/data/numbers.txt"}])
fam("two", "t_find_wc", [P("common/find", [0, 2, 4]), P("common/wc", K4)],
    ["How many {e} files are there under {d}?"], "find {d} -name '*{e}' | wc -l", "num",
    [{"e": ".py", "d": "/data/src"}, {"e": ".log", "d": "/data/logs"}, {"e": ".txt", "d": "/data"}])
fam("two", "t_cut_sort", [P("common/cut", [1, 2, 3]), P("common/sort", [0, 1, 3, 6])],
    ["List the distinct values of the {n} column of the comma-separated file {f}, sorted."],
    'cut -d "," -f {k} {f} | sort -u', "out",
    [{"n": "third", "k": 3, "f": "/data/users.csv"}, {"n": "second", "k": 2, "f": "/data/users.csv"}])
fam("two", "t_grep_tail", [P("common/grep", [0, 1, 5]), P("common/tail", [0, 2, 3])],
    ["Show the last {n} lines of {f} that contain {p}."], 'grep "{p}" {f} | tail -n {n}', "out",
    [{"n": 3, "p": "WARN", "f": "/data/app.log"}, {"n": 2, "p": "500", "f": "/data/access.log"}])
fam("two", "t_ls_wc", [P("common/ls", [0, 1, 3]), P("common/wc", K4)],
    ["How many non-hidden entries are in the directory {d}?"], "ls {d} | wc -l", "num",
    [{"d": "/data/logs"}, {"d": "/data/src"}])
fam("two", "t_gzip_wc", [P("common/gzip", [1, 3, 4, 5]), P("common/wc", K4)],
    ["How many lines does the compressed file {f} contain? Do not write any file."], "gzip -cd {f} | wc -l", "num",
    [{"f": "/data/logs/old.log.gz"}], limit=1)
fam("two", "t_awk_sum_csv", [P("common/awk", K4)],
    ["Add up the {n} column of the comma-separated file {f}."], "awk -F ',' '{s+=${k}} END {print s}' {f}", "num",
    [{"n": "fourth", "k": 4, "f": "/data/users.csv"}, {"n": "first", "k": 1, "f": "/data/users.csv"}])
fam("two", "t_tr_head", [P("common/tr", [0, 3, 5]), P("common/head")],
    ["Show the first {n} lines of {f} in upper case."], 'head -n {n} {f} | tr "[:lower:]" "[:upper:]"', "out",
    [{"n": 2, "f": "/data/notes.txt"}, {"n": 4, "f": "/data/names.txt"}])
fam("two", "t_find_exec", [P("common/find", [0, 2, 5])],
    ["Count the lines of every {e} file under {d}, one count per file."], "find {d} -name '*{e}' -exec wc -l {} \\;", "outset",
    [{"e": ".py", "d": "/data/src"}, {"e": ".log", "d": "/data/logs"}])
fam("two", "t_grep_cut", [P("common/grep", [0, 1, 5]), P("common/cut", [1, 2, 3])],
    ["{f} is comma-separated. Print the second field of the lines that contain {p}."], 'grep "{p}" {f} | cut -d "," -f 2', "out",
    [{"p": "Oslo", "f": "/data/users.csv"}, {"p": "Rome", "f": "/data/users.csv"}])
fam("two", "t_du_sort", [P("common/du", [0, 1, 2]), P("common/sort", [0, 1, 3, 6])],
    ["List the sizes of {d} and its subdirectories in kilobytes, largest first."], "du -k {d} | sort -nr", "out",
    [{"d": "/data"}], limit=1)

# ───────────────────────────── direct ─────────────────────────────
# (page, ask, accepted commands, fold_case)

DIRECT = [
    ("common/uname", "Print all available system information.", ["uname -a"]),
    ("common/hostname", "Show the current host name.", ["hostname"]),
    ("common/id", "Show the name of the current user.", ["id -un", "whoami"]),
    ("common/date", "Print the current date as a Unix timestamp.", ["date +%s"]),
    ("common/env", "Show the environment.", ["env"]),
    ("common/ps", "List all running processes with the full command string.", ["ps auxww"]),
    ("linux/free", "Show memory in the most appropriate human-readable units.", ["free -h"]),
    ("linux/ip", "Display the routing table.", ["ip r", "ip route"]),
    ("linux/lsblk", "Show information about the filesystems on the block devices.", ["lsblk -f"]),
    ("common/uptime", "Show only how long the system has been up.", ["uptime -p"]),
    ("linux/systemctl", "List the units that have failed.", ["systemctl --failed"]),
    ("linux/apt", "Refresh the list of available packages and versions.", ["sudo apt update"]),
    ("common/brew", "List all installed formulae and casks.", ["brew list"]),
    ("osx/diskutil", "List all disks, partitions and mounted volumes.", ["diskutil list"]),
    ("osx/sw_vers", "Print only the build identifier of macOS.", ["sw_vers -buildVersion"]),
    ("windows/ipconfig", "Show the local DNS cache.", ["ipconfig /displaydns"]),
    ("windows/tasklist", "Show which services run inside each process.", ["tasklist /svc"]),
    ("common/kill", "List the available signal names.", ["kill -l"]),
    ("linux/resolvectl", "Display the DNS statistics.", ["resolvectl statistics"]),
    ("common/pwd", "Print the current directory.", ["pwd"]),
]
