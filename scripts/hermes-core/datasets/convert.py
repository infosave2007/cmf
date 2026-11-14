"""Open chat and tool-use datasets as text in one contract, ready for `cortiq-embryo shard --specials`.

    python3 convert.py --raw /root/hc/raw --out /root/hc/prep [--only NAME ...] [--workers 4]

The contract ("cmf-im-v1" frame, Hermes function-calling tags as plain text):

    <|im_start|>system
    {system text}

    {TOOLS_HEAD}
    <tools>
    {one JSON function description per line}
    </tools><|im_end|>
    <|im_start|>user
    {text}<|im_end|>
    <|im_start|>assistant
    <think>
    {reasoning}
    </think>
    {text}
    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call><|im_end|>
    <|im_start|>tool
    <tool_response>
    {result}
    </tool_response><|im_end|>

A conversation is one document (`{"text": ...}` per line, gzip); the shard
builder adds the end-of-text token after it. The model's own reference
lookup is a tool of this contract (`lookup`), see `lookup_traces.py`.

Every source goes to `<out>/<name>.jsonl.gz`, with a held-out
`<name>.val.jsonl.gz` (1 conversation in 200 by the hash of its text, up to
`VAL_BYTES`), and a line in `<out>/convert.stats.jsonl`: conversations kept
and dropped, bytes. Tool results are cut at `TOOL_CUT` characters, a
conversation longer than `DOC_MAX` characters is dropped, so is one with an
unparseable tool call or without an assistant turn.
"""

from __future__ import annotations

import argparse
import ast
import glob
import gzip
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

TOOLS_HEAD = ("You have access to the following tools. To call a tool, reply with a JSON object inside "
              '<tool_call></tool_call> tags: {"name": <tool name>, "arguments": <arguments object>}.')
TOOL_CUT = 2500
DOC_MAX = 60_000
VAL_BYTES = 800_000

THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


class Skip(Exception):
    pass


def loose(s):
    """A JSON value that may be written as a Python literal."""
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        raise Skip("unparseable structure")


def fn_desc(t):
    """A tool description as {name, description, parameters}."""
    t = loose(t) if isinstance(t, str) else t
    if isinstance(t, dict) and isinstance(t.get("function"), dict):
        t = t["function"]
    if not isinstance(t, dict) or "name" not in t:
        raise Skip("tool without a name")
    out = {"name": t["name"]}
    if t.get("description"):
        out["description"] = t["description"]
    if t.get("parameters") is not None:
        out["parameters"] = t["parameters"]
    return out


def call_obj(c):
    """A tool call as {name, arguments: object}."""
    c = loose(c) if isinstance(c, str) else c
    if isinstance(c, dict) and isinstance(c.get("function"), dict):
        c = c["function"]
    if not isinstance(c, dict) or not c.get("name"):
        raise Skip("tool call without a name")
    args = c.get("arguments", c.get("parameters", {}))
    if isinstance(args, str):
        args = loose(args) if args.strip() else {}
    if not isinstance(args, dict):
        raise Skip("tool call arguments are not an object")
    return {"name": c["name"], "arguments": args}


def split_assistant(content):
    """Reasoning, text and tool calls written inside an assistant message."""
    content = content or ""
    think = None
    m = THINK.match(content)
    if m:
        think, content = m.group(1).strip(), content[m.end():]
    calls = [call_obj(x) for x in CALL.findall(content)]
    text = CALL.sub("", content).strip()
    return think, text, calls


def render(system, tools, turns):
    """turns: (role, text) for system/user/tool, (assistant, think, text, calls)."""
    out = []
    sys_text = (system or "").strip()
    if tools:
        block = TOOLS_HEAD + "\n<tools>\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>"
        sys_text = (sys_text + "\n\n" + block) if sys_text else block
    if sys_text:
        out.append("<|im_start|>system\n" + sys_text + "<|im_end|>\n")
    seen_assistant = False
    merged = None
    for t in turns:
        role = t[0]
        if role != "assistant":
            merged = None
        if role == "user":
            if not (t[1] or "").strip():
                raise Skip("empty user turn")
            out.append("<|im_start|>user\n" + t[1].strip() + "<|im_end|>\n")
        elif role == "tool":
            body = (t[1] or "").strip()
            if len(body) > TOOL_CUT:
                body = body[:TOOL_CUT] + " …"
            out.append("<|im_start|>tool\n<tool_response>\n" + body + "\n</tool_response><|im_end|>\n")
        elif role == "assistant":
            _, think, text, calls = t
            if out and merged is not None:
                # a second assistant message in a row continues the turn (text, then calls)
                out.pop()
                think = merged[0] or think
                text = "\n".join(x for x in (merged[1], text) if x and x.strip())
                calls = merged[2] + calls
            merged = (think, text, calls)
            parts = []
            if think:
                parts.append("<think>\n" + think.strip() + "\n</think>")
            if text and text.strip():
                parts.append(text.strip())
            for c in calls:
                parts.append("<tool_call>\n" + json.dumps(c, ensure_ascii=False) + "\n</tool_call>")
            if not parts:
                raise Skip("empty assistant turn")
            out.append("<|im_start|>assistant\n" + "\n".join(parts) + "<|im_end|>\n")
            seen_assistant = True
        else:
            raise Skip(f"role {role}")
    if not seen_assistant:
        raise Skip("no assistant turn")
    doc = "".join(out)
    if len(doc) > DOC_MAX:
        raise Skip("too long")
    if "<|endoftext|>" in doc or doc.count("<|im_start|>") != len(out):
        raise Skip("special token inside the text")
    return doc


def from_messages(msgs, system=None, tools=None):
    """OpenAI-style messages (tool calls as a field or written in the text)."""
    turns = []
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") if m.get("content") is not None else m.get("value")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = content or ""
        if role in ("system", "developer"):
            system = ((system + "\n\n") if system else "") + content if not turns else system
        elif role in ("user", "human"):
            if turns and turns[-1][0] == "assistant" and turns[-1][3] and content.lstrip().startswith("Observation:"):
                turns.append(("tool", content.lstrip()[len("Observation:"):]))
            else:
                turns.append(("user", content))
        elif role in ("assistant", "gpt"):
            think, text, calls = split_assistant(content)
            for key in ("reasoning_content", "thinking", "reasoning"):
                if m.get(key) and not think:
                    think = str(m[key]).strip()
            for c in m.get("tool_calls") or []:
                calls.append(call_obj(c))
            if m.get("function_call"):
                calls.append(call_obj(m["function_call"]))
            turns.append(("assistant", think, text, calls))
        elif role == "tool_call":  # a call as a message of its own (Toucan SFT)
            turns.append(("assistant", None, "", [call_obj(content)]))
        elif role in ("tool", "function", "observation", "ipython", "tool_response"):
            body = re.sub(r"^\s*<tool_response>\s*|\s*</tool_response>\s*$", "", content)
            turns.append(("tool", re.sub(r"^\s*Observation:\s*", "", body)))
        else:
            raise Skip(f"role {role}")
    return render(system, tools, turns)


# ---------------------------------------------------------------- sources


def rows_parquet(path):
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    for b in f.iter_batches(batch_size=2000):
        yield from b.to_pylist()


def rows_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def smoltalk(row):
    kw = row.get("chat_template_kwargs") or {}
    tools = []
    system = kw.get("custom_instructions") or None
    for x in kw.get("xml_tools") or []:
        blocks = re.findall(r"<tools>\s*(.*?)\s*</tools>", x, re.S)
        if blocks:
            # a whole system prompt with the list inside: the task-specific prose stays as the system text
            prose = x.split("<tools>")[0].strip()
            if prose and "function calling AI model" not in prose:
                system = ((system + "\n\n") if system else "") + prose
            x = max(blocks, key=len)
        try:
            v = loose(x)
            tools += [fn_desc(t) for t in (v if isinstance(v, list) else [v])]
        except Skip:
            tools += [fn_desc(l) for l in x.splitlines() if l.strip()]
    return from_messages(row["messages"], system=system, tools=tools)


def ultrachat(row):
    return from_messages(row["messages"])


def toolmind(row):
    return from_messages(row["conversations"], tools=[fn_desc(t) for t in row.get("tools") or []])


def when2call(row):
    return from_messages(row["messages"], tools=[fn_desc(t) for t in row.get("tools") or []])


GL_ROLE = re.compile(r"(?:^|\n)\s*(USER|ASSISTANT|FUNCTION RESPONSE):\s*")


def json_objects(text):
    """Top-level {...} objects of a text, by brace matching."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def glaive(row):
    tools = [fn_desc(o) for o in json_objects(row.get("system") or "")]
    parts = GL_ROLE.split(row["chat"])
    msgs = []
    for role, body in zip(parts[1::2], parts[2::2]):
        body = body.replace("<|endoftext|>", "").strip()
        if role == "USER":
            msgs.append({"role": "user", "content": body})
        elif role == "FUNCTION RESPONSE":
            msgs.append({"role": "tool", "content": body})
        else:
            m = re.search(r"<functioncall>\s*(\{.*\})\s*$", body, re.S)
            if m:
                raw = m.group(1)
                a = re.search(r'"arguments":\s*\'(.*)\'\s*\}\s*$', raw, re.S)
                if a:
                    name = re.search(r'"name":\s*"([^"]+)"', raw)
                    if not name:
                        raise Skip("glaive call without a name")
                    call = {"name": name.group(1), "arguments": loose(a.group(1))}
                else:
                    call = loose(raw)
                msgs.append({"role": "assistant", "content": body[:m.start()].strip(), "tool_calls": [call]})
            else:
                msgs.append({"role": "assistant", "content": body})
    return from_messages(msgs, tools=tools)


def tau2_rows(path):
    """The last recorded turn of every dialog: its context is the whole dialog."""
    last = {}
    with open(path, encoding="utf-8") as f:
        pos = 0
        for line in f:
            if line.strip():
                md = json.loads(line).get("metadata") or {}
                key, ti = md.get("source_dialog_id"), md.get("turn_index", 0)
                if key is not None and (key not in last or ti >= last[key][0]):
                    last[key] = (ti, pos)
            pos += len(line.encode("utf-8"))
    keep = sorted(p for _, p in last.values())
    with open(path, "rb") as f:
        for p in keep:
            f.seek(p)
            yield json.loads(f.readline())


def tau2(row):
    msgs = list(row["messages"]) + [row["answer"]]
    return from_messages(msgs, tools=[fn_desc(t) for t in row.get("tools") or []])


def nemotron(row):
    return from_messages(row["messages"], tools=[fn_desc(t) for t in row.get("tools") or []])


def toucan(row):
    msgs = [m for m in loose(row["messages"]) if m.get("role") != "system"]
    tools = row.get("available_tools") or row.get("tools") or "[]"
    return from_messages(msgs, tools=[fn_desc(t) for t in loose(tools)])


RESP = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.S)


def envfactory_rows(path):
    """One record per dialog: the one with the longest history."""
    best = {}
    for r in rows_json_list(path):
        hist = r.get("history") or []
        key = (hist[0][0] if hist else r.get("instruction") or "").strip()
        if key not in best or len(hist) > len(best[key].get("history") or []):
            best[key] = r
    yield from best.values()


def envfactory(row):
    system = row.get("system") or ""
    tools = []
    blocks = re.findall(r"<tools>\s*(.*?)\s*</tools>", system, re.S)
    if blocks:  # the prose above the list also says "<tools></tools>"
        tools = [fn_desc(l) for l in max(blocks, key=len).splitlines() if l.strip()]
    system = system.split("\n# Tools")[0].strip()
    msgs = []
    for inp, outp in list(row.get("history") or []) + [[row.get("instruction") or "", row.get("output") or ""]]:
        results = RESP.findall(inp)
        if results:
            msgs += [{"role": "tool", "content": r} for r in results]
        else:
            msgs.append({"role": "user", "content": inp})
        msgs.append({"role": "assistant", "content": outp})
    return from_messages(msgs, system=system, tools=tools)


SOURCES = {
    # name: (glob under --raw, row reader, adapter)
    "ultrachat": ("ultrachat/data/train_sft-*.parquet", rows_parquet, ultrachat),
    "magpie": ("smoltalk2/SFT/smoltalk_smollm3_smol_magpie_ultra_no_think-*", rows_parquet, smoltalk),
    "systemchats": ("smoltalk2/SFT/smoltalk_smollm3_systemchats_30k_no_think-*", rows_parquet, smoltalk),
    "everyday": ("smoltalk2/SFT/smoltalk_smollm3_everyday_conversations_no_think-*", rows_parquet, smoltalk),
    "rewrite": ("smoltalk2/SFT/smoltalk_smollm3_smol_rewrite_no_think-*", rows_parquet, smoltalk),
    "summarize": ("smoltalk2/SFT/smoltalk_smollm3_smol_summarize_no_think-*", rows_parquet, smoltalk),
    "explore": ("smoltalk2/SFT/smoltalk_smollm3_explore_instruct_rewriting_no_think-*", rows_parquet, smoltalk),
    "mot_science": ("smoltalk2/SFT/Mixture_of_Thoughts_science_no_think-*", rows_parquet, smoltalk),
    "openthoughts_answers": ("smoltalk2/SFT/OpenThoughts3_1.2M_no_think_no_think-*", rows_parquet, smoltalk),
    "multilingual": ("smoltalk2/SFT/smoltalk_multilingual_8languages_lang_5_no_think-*", rows_parquet, smoltalk),
    "multilingual_think": ("smoltalk2/SFT/smoltalk_multilingual8_Qwen3_32B_think-*", rows_parquet, smoltalk),
    "aya_think": ("smoltalk2/SFT/aya_dataset_Qwen3_32B_think-*", rows_parquet, smoltalk),
    "tablegpt_think": ("smoltalk2/SFT/table_gpt_Qwen3_32B_think-*", rows_parquet, smoltalk),
    "systemchats_think": ("smoltalk2/SFT/smoltalk_systemchats_Qwen3_32B_think-*", rows_parquet, smoltalk),
    "everyday_think": ("smoltalk2/SFT/smoltalk_everyday_convs_reasoning_Qwen3_32B_think-*", rows_parquet, smoltalk),
    "multiturn_if_think": ("smoltalk2/SFT/multi_turn_reasoning_if_think-*", rows_parquet, smoltalk),
    "openthoughts_think": ("smoltalk2/SFT/OpenThoughts3_1.2M_think-*", rows_parquet, smoltalk),
    "hermes_fc": ("smoltalk2/SFT/hermes_function_calling_v1_no_think-*", rows_parquet, smoltalk),
    "smolagents": ("smoltalk2/SFT/smolagents_toolcalling_traces_think-*", rows_parquet, smoltalk),
    "toolmind": ("toolmind/*_datasets/*.jsonl", rows_jsonl, toolmind),
    "when2call": ("when2call/train/when2call_train_sft.jsonl", rows_jsonl, when2call),
    "glaive": ("glaive/glaive-function-calling-v2.json", None, glaive),
    "tau2": ("tau2/tau2_sft_train.jsonl", tau2_rows, tau2),
    "envfactory": ("envfactory/mcp_factory_sft_nips.json", envfactory_rows, envfactory),
    "toucan_kimi": ("toucan/Kimi-K2/train-000[0-1]?-of-*.parquet", rows_parquet, toucan),
    "toucan_sft": ("toucan/SFT/*.parquet", rows_parquet, toucan),
    "nemotron_search": ("nemotron-agentic/data/search.jsonl", rows_jsonl, nemotron),
    "nemotron_agent": ("nemotron-agentic/data/interactive_agent.jsonl", rows_jsonl, nemotron),
}


def rows_json_list(path):
    with open(path, encoding="utf-8") as f:
        yield from json.load(f)


def work(job):
    name, path, part, out_dir = job
    _, reader, adapter = SOURCES[name]
    reader = reader or rows_json_list
    kept = dropped = nbytes = vbytes = 0
    why: dict[str, int] = {}
    tr = gzip.open(os.path.join(out_dir, f"{name}.part{part:04d}.jsonl.gz"), "wt", encoding="utf-8", compresslevel=3)
    va = gzip.open(os.path.join(out_dir, f"{name}.part{part:04d}.val.jsonl.gz"), "wt", encoding="utf-8", compresslevel=3)
    for row in reader(path):
        try:
            doc = adapter(row)
        except Skip as e:
            dropped += 1
            why[str(e)] = why.get(str(e), 0) + 1
            continue
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            dropped += 1
            k = f"{type(e).__name__}"
            why[k] = why.get(k, 0) + 1
            continue
        line = json.dumps({"text": doc}, ensure_ascii=False) + "\n"
        held = int(hashlib.md5(doc.encode("utf-8")).hexdigest()[:8], 16) % 200 == 0 and vbytes < VAL_BYTES
        if held:
            va.write(line)
            vbytes += len(doc)
        else:
            tr.write(line)
            kept += 1
            nbytes += len(doc)
    tr.close()
    va.close()
    return name, kept, dropped, nbytes, vbytes, why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample", action="store_true", help="print one converted conversation per source and stop")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = args.only or list(SOURCES)
    if args.sample:
        for name in names:
            pat, reader, adapter = SOURCES[name]
            files = sorted(glob.glob(os.path.join(args.raw, pat)))
            print(f"===== {name}: {len(files)} files")
            shown = bad = 0
            for row in (reader or rows_json_list)(files[0]) if files else []:
                try:
                    doc = adapter(row)
                except Exception as e:
                    bad += 1
                    if bad <= 2:
                        print(f"  [skip: {type(e).__name__}: {e}]")
                    continue
                shown += 1
                if shown == 3:
                    print(doc[:1800] + ("\n…" if len(doc) > 1800 else ""))
                    break
        return
    jobs = []
    for name in names:
        files = sorted(glob.glob(os.path.join(args.raw, SOURCES[name][0])))
        if not files:
            print(f"{name}: no files", file=sys.stderr)
        jobs += [(name, f, i, args.out) for i, f in enumerate(files)]
    total: dict[str, list] = {}
    with Pool(args.workers) as pool:
        for name, kept, dropped, nbytes, vbytes, why in pool.imap_unordered(work, jobs):
            t = total.setdefault(name, [0, 0, 0, 0, {}])
            t[0] += kept
            t[1] += dropped
            t[2] += nbytes
            t[3] += vbytes
            for k, v in why.items():
                t[4][k] = t[4].get(k, 0) + v
    with open(os.path.join(args.out, "convert.stats.jsonl"), "a") as st:
        for name in names:
            if name not in total:
                continue
            kept, dropped, nbytes, vbytes, why = total[name]
            for suffix in ("", ".val"):
                parts = sorted(glob.glob(os.path.join(args.out, f"{name}.part????{suffix}.jsonl.gz")))
                with open(os.path.join(args.out, f"{name}{suffix}.jsonl.gz"), "wb") as dst:
                    for p in parts:
                        with open(p, "rb") as src:
                            while chunk := src.read(1 << 24):
                                dst.write(chunk)
                        os.remove(p)
            rec = {"name": name, "kept": kept, "dropped": dropped, "bytes": nbytes, "val_bytes": vbytes, "dropped_why": why}
            st.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{name:<22} kept {kept:>8} dropped {dropped:>7} {nbytes / 1e6:9.1f} MB  val {vbytes / 1e3:6.0f} KB  {why}")


if __name__ == "__main__":
    main()
