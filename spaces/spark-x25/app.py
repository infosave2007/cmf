"""Spark Lab: a calm, honest public playground for Spark-X2.5 CMF.

The numerical path is the released Cortiq binary. Python only manages the
Space UI and forwards OpenAI-compatible requests to a loopback Cortiq server.
No prompt is sent to a third-party inference API.
"""
from __future__ import annotations

import atexit
import html
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from huggingface_hub import hf_hub_download
from logic import MODEL_VARIANTS, PRESETS, build_payload, response_parts, split_thinking

MODEL_REPO = os.environ.get("SPARK_CMF_REPO", "infosave/Spark-X2.5-cmf")
PORT = int(os.environ.get("SPARK_PORT", "8080"))
GRADIO_HOST = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
BASE_URL = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_S = 300
REQUEST_TIMEOUT_S = 600
PROGRESS = gr.Progress()

TOOL_DEMO = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. The caller executes this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Preferred temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

EXAMPLE_PROMPTS = {
    "Ask": "Explain why a sliding-window attention layer can make long-context inference cheaper.",
    "Rewrite": "Rewrite this so it is direct but kind: “Your report is late and incomplete.”",
    "Translate": "Переведи на английский: «Нам нужно согласовать план до пятницы и назначить ответственного.»",
    "Plan": "Create a five-step launch plan for a small open-source developer tool.",
}


class CortiqService:
    """One active model server per Space process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._variant: str | None = None
        self._path: str | None = None
        self._log_path: Path | None = None
        self._log_file: Any | None = None

    @staticmethod
    def _download(variant: str, progress: gr.Progress | None = None) -> str:
        filename = MODEL_VARIANTS[variant]["file"]
        if progress is not None:
            progress(0.04, desc=f"Preparing {filename}; the first load may download {MODEL_VARIANTS[variant]['size']}")
        try:
            return hf_hub_download(repo_id=MODEL_REPO, filename=filename)
        except Exception as exc:
            raise RuntimeError(f"Could not download {filename} from the public model repository: {exc}") from exc

    def _tail_log_unlocked(self) -> str:
        if self._log_file is not None:
            self._log_file.flush()
        if self._log_path is None or not self._log_path.exists():
            return ""
        text = self._log_path.read_text(encoding="utf-8", errors="replace")[-1_200:]
        return " ".join(text.split())

    def _close_log_unlocked(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._log_path = None

    def _stop_unlocked(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=6)
        self._process = None
        self._variant = None
        self._path = None
        self._close_log_unlocked()

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    def ensure(self, variant: str, progress: gr.Progress | None = None) -> str:
        if not shutil.which("cortiq"):
            raise RuntimeError("The Cortiq runtime is missing from this Space image.")
        if variant not in MODEL_VARIANTS:
            raise ValueError("Choose a listed Spark model.")

        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is None
                and self._variant == variant
                and self._path is not None
            ):
                return self._path

            self._stop_unlocked()
            path = self._download(variant, progress)
            if progress is not None:
                progress(0.35, desc="Starting the local CMF runtime")
            command = [
                "cortiq",
                "serve",
                path,
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
            ]
            log_dir = Path(tempfile.gettempdir()) / "spark-lab"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = log_dir / "cortiq-runtime.log"
            self._log_file = self._log_path.open("w", encoding="utf-8")
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "RUST_BACKTRACE": "0"},
                )
            except OSError as exc:
                self._close_log_unlocked()
                raise RuntimeError(f"Could not start the Cortiq runtime: {exc}") from exc
            deadline = time.monotonic() + STARTUP_TIMEOUT_S
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    diagnostics = self._tail_log_unlocked()
                    self._stop_unlocked()
                    suffix = f" Runtime detail: {diagnostics}" if diagnostics else ""
                    raise RuntimeError(
                        "Cortiq stopped while loading this model. Try the smaller 1.7B variant." + suffix
                    )
                try:
                    response = requests.get(f"{BASE_URL}/v1/models", timeout=2)
                    if response.ok:
                        self._variant, self._path = variant, path
                        return path
                except requests.RequestException:
                    pass
                time.sleep(0.5)

            diagnostics = self._tail_log_unlocked()
            self._stop_unlocked()
            suffix = f" Runtime detail: {diagnostics}" if diagnostics else ""
            raise RuntimeError("The local model did not become ready in time. Try the 1.7B Fast variant." + suffix)

    def complete(self, payload: dict[str, Any], variant: str, progress: gr.Progress | None) -> dict[str, Any]:
        self.ensure(variant, progress)
        if progress is not None:
            progress(0.55, desc="Spark is generating locally")
        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=REQUEST_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise RuntimeError("The local runtime could not be reached. Please retry.") from exc
        if not response.ok:
            detail = response.text.strip().replace("\n", " ")[:500]
            raise RuntimeError(f"The local runtime returned HTTP {response.status_code}: {detail or 'no details'}")
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("The local runtime returned an invalid response.") from exc


SERVICE = CortiqService()
atexit.register(SERVICE.stop)


def mode_to_thinking(mode: str) -> bool:
    return mode == "Think it through"


def receipt_markdown(
    *,
    variant: str,
    preset: str,
    mode: str,
    elapsed_s: float,
    usage: dict[str, Any],
    tool_calls: int = 0,
) -> str:
    details = MODEL_VARIANTS[variant]
    tokens = usage.get("total_tokens")
    token_line = f"{tokens} total tokens" if tokens is not None else "token count unavailable"
    tool_line = f" · {tool_calls} tool call(s) proposed" if tool_calls else ""
    return (
        '<div class="receipt">'
        '<div class="receipt-kicker">RUN RECEIPT</div>'
        f'<strong>{html.escape(variant)}</strong><br>'
        f'<span>{html.escape(details["file"])} · {html.escape(details["size"])}</span>'
        '<div class="receipt-grid">'
        f'<span>Mode<strong>{html.escape(mode)}</strong></span>'
        f'<span>Preset<strong>{html.escape(preset)}</strong></span>'
        f'<span>Runtime<strong>{elapsed_s:.1f}s</strong></span>'
        f'<span>Usage<strong>{html.escape(token_line)}</strong></span>'
        '</div>'
        f'<div class="receipt-note">Local Cortiq runtime{tool_line}. Re-run with the same controls for the same setup.</div>'
        '</div>'
    )


def send_message(
    history: list[dict[str, Any]] | None,
    prompt: str,
    variant: str,
    preset: str,
    mode: str,
    progress: gr.Progress = PROGRESS,
):
    try:
        payload = build_payload(
            variant=variant,
            preset=preset,
            thinking=mode_to_thinking(mode),
            prompt=prompt,
            history=history,
        )
        started = time.monotonic()
        result = SERVICE.complete(payload, variant, progress)
        elapsed = time.monotonic() - started
        content, calls, usage = response_parts(result)
        reasoning, answer = split_thinking(content)
        if calls:
            answer = answer or "Spark proposed a tool call. Open Tool-call lab to inspect the structured request."
        if not answer:
            answer = "Spark returned no visible text. Try Direct answer or a clearer request."
        conversation = list(history or [])
        conversation.extend(
            [
                {"role": "user", "content": prompt.strip()},
                {"role": "assistant", "content": answer},
            ]
        )
        receipt = receipt_markdown(
            variant=variant,
            preset=preset,
            mode=mode,
            elapsed_s=elapsed,
            usage=usage,
            tool_calls=len(calls),
        )
        raw = json.dumps(result, ensure_ascii=False, indent=2)
        progress(1, desc="Ready")
        return conversation, "", receipt, reasoning or "No reasoning trace was returned for this run.", raw
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc


def run_tool_lab(
    prompt: str,
    variant: str,
    preset: str,
    progress: gr.Progress = PROGRESS,
):
    try:
        payload = build_payload(
            variant=variant,
            preset=preset,
            thinking=True,
            prompt=prompt,
            tools=TOOL_DEMO,
        )
        started = time.monotonic()
        result = SERVICE.complete(payload, variant, progress)
        elapsed = time.monotonic() - started
        content, calls, usage = response_parts(result)
        _, answer = split_thinking(content)
        receipt = receipt_markdown(
            variant=variant,
            preset=preset,
            mode="Tool-call lab",
            elapsed_s=elapsed,
            usage=usage,
            tool_calls=len(calls),
        )
        if calls:
            call_json = json.dumps(calls, ensure_ascii=False, indent=2)
            status = "Spark proposed this tool call. The Space intentionally does not execute it."
        else:
            call_json = "[]"
            status = "Spark answered without a tool call. Try a request that needs current weather."
        return status, answer or "—", call_json, receipt
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc


def set_example(name: str) -> str:
    return EXAMPLE_PROMPTS[name]


CSS = """
:root {
  --spark-ink: #0b1120;
  --spark-paper: #f7f6f1;
}
.gradio-container {
  background: radial-gradient(circle at 8% -12%, #ffedc7 0, transparent 30rem),
              radial-gradient(circle at 98% 0%, #dcd9ff 0, transparent 29rem),
              var(--spark-paper) !important;
  color: var(--spark-ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
#spark-hero {
  border: 1px solid rgba(11, 17, 32, .12);
  border-radius: 28px;
  padding: 28px 30px;
  margin: 6px 0 18px;
  background: linear-gradient(122deg, rgba(255,255,255,.9), rgba(255,250,239,.72));
  box-shadow: 0 18px 50px rgba(39, 30, 79, .08);
}
#spark-hero .eyebrow { color: #7b4c00; font-size: .74rem; font-weight: 800; letter-spacing: .14em; }
#spark-hero h1 { font-size: clamp(2rem, 5vw, 4.4rem); line-height: .95; letter-spacing: -.055em; margin: .48rem 0 .8rem; }
#spark-hero p { max-width: 53rem; font-size: 1.06rem; line-height: 1.55; margin: 0; color: #384053; }
#spark-hero .spark-chip { display: inline-block; margin: 1rem .45rem 0 0; border-radius: 999px; padding: .32rem .66rem; background: #101828; color: white; font-size: .78rem; }
.spark-card { border: 1px solid rgba(11,17,32,.1); border-radius: 20px; background: rgba(255,255,255,.72); padding: 16px; }
.receipt { border: 1px solid rgba(107,85,217,.23); border-radius: 18px; background: linear-gradient(145deg, #f6f3ff, #fff); padding: 15px 16px; min-height: 145px; }
.receipt-kicker { color: #6754bd; font-weight: 800; font-size: .69rem; letter-spacing: .12em; margin-bottom: 4px; }
.receipt span { color: #596176; font-size: .82rem; }
.receipt-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px; margin-top: 12px; }
.receipt-grid span { display: flex; flex-direction: column; padding: 8px; border-radius: 10px; background: rgba(107,85,217,.06); }
.receipt-grid strong { color: #1c2435; font-size: .88rem; margin-top: 2px; }
.receipt-note { color: #596176; font-size: .76rem; margin-top: 10px; line-height: 1.35; }
#send-button button { background: linear-gradient(115deg, #f2a33c, #de6b35) !important; border: 0 !important; color: #18120a !important; font-weight: 800 !important; }
#task-row button { border-radius: 999px !important; }
footer { opacity: .68; }
"""


with gr.Blocks(
    title="Spark Lab · Spark-X2.5 CMF",
    theme=gr.themes.Soft(primary_hue="orange", secondary_hue="violet"),
    css=CSS,
) as demo:
    gr.HTML(
        """
        <section id="spark-hero">
          <div class="eyebrow">SPARK-X2.5 · CMF PLAYGROUND</div>
          <h1>One quiet place<br>to try a fast local model.</h1>
          <p>Choose a task, write naturally in English, Russian, Chinese, or another language,
          and Spark answers through a local Cortiq runtime. Start simple; the controls and
          developer details appear only when you ask for them.</p>
          <span class="spark-chip">1.7B → 4B</span>
          <span class="spark-chip">direct or reasoning</span>
          <span class="spark-chip">tool-call inspection</span>
          <span class="spark-chip">no third-party inference API</span>
        </section>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=7, min_width=460):
            gr.Markdown("### Start with a real task")
            with gr.Row(elem_id="task-row"):
                ask = gr.Button("✦ Ask", size="sm")
                rewrite = gr.Button("↻ Rewrite", size="sm")
                translate = gr.Button("⇄ Translate", size="sm")
                plan = gr.Button("☷ Plan", size="sm")

            chat = gr.Chatbot(
                type="messages",
                label="Conversation",
                placeholder="Your conversation will appear here.",
                height=425,
                show_copy_button=True,
            )
            prompt = gr.Textbox(
                label="What should Spark help with?",
                placeholder="Try a question, a draft to improve, or a planning problem…",
                lines=3,
                max_lines=8,
            )
            with gr.Row():
                send = gr.Button("Generate", variant="primary", elem_id="send-button")
                clear = gr.ClearButton([chat, prompt], value="Clear conversation")
            gr.Examples(
                examples=[[value] for value in EXAMPLE_PROMPTS.values()],
                inputs=[prompt],
                label="Try an example",
                examples_per_page=4,
            )

        with gr.Column(scale=5, min_width=320):
            with gr.Group(elem_classes="spark-card"):
                gr.Markdown("### Your setup")
                variant = gr.Dropdown(
                    choices=list(MODEL_VARIANTS),
                    value="Spark 1.7B · fastest",
                    label="Model",
                    info="The shared demo keeps one chosen model active at a time.",
                )
                preset = gr.Radio(
                    choices=list(PRESETS),
                    value="Balanced",
                    label="Response style",
                    info="Fast is shorter; Deep allows a longer answer.",
                )
                mode = gr.Radio(
                    choices=["Direct answer", "Think it through"],
                    value="Direct answer",
                    label="How should it answer?",
                    info="Thinking may take longer and exposes a reasoning trace below.",
                )
                with gr.Accordion("What am I choosing?", open=False):
                    gr.Markdown(
                        """**1.7B Fast** is the default because it is the smallest 1.36 GB file.
                        **4B Quality** is the model-card default (4.14 GB) for a more capable
                        local run. Spark supports up to 1M tokens, but this public demo keeps
                        conversations deliberately short and responsive."""
                    )

            gr.Markdown("### Run receipt")
            receipt = gr.HTML(
                '<div class="receipt"><div class="receipt-kicker">READY</div>'
                "<strong>No run yet</strong><br><span>Every response will show the model, mode, "
                "sampling preset, runtime and total token count.</span></div>"
            )
            with gr.Accordion("Reasoning trace", open=False):
                reasoning = gr.Textbox(label="Only shown when returned", lines=9, interactive=False)
            with gr.Accordion("Raw OpenAI-compatible response", open=False):
                raw = gr.Code(label="Response JSON", language="json", value="{}", interactive=False)

    gr.Markdown("---")
    with gr.Tabs():
        with gr.Tab("Tool-call lab"):
            gr.Markdown(
                """### Inspect a tool call without granting the Space any power

Spark can return the OpenAI-compatible structured request for a tool. This lab
offers a weather schema, shows what Spark proposes, and **never executes the call**.
That keeps the demo useful for developers without pretending it accessed live data."""
            )
            with gr.Row():
                tool_prompt = gr.Textbox(
                    label="Request that needs weather",
                    value="What is the weather in Paris today? Use Celsius.",
                    lines=3,
                )
                with gr.Column():
                    tool_go = gr.Button("Ask Spark to choose a tool", variant="primary")
                    tool_status = gr.Markdown("No tool request yet.")
            with gr.Row():
                tool_answer = gr.Textbox(label="Visible model reply", lines=8, interactive=False)
                tool_json = gr.Code(label="Proposed tool_calls", language="json", value="[]", interactive=False)
            tool_receipt = gr.HTML()

        with gr.Tab("Run it yourself"):
            gr.Markdown(
                """### Same model, your hardware

~~~bash
cargo install cortiq-cli
hf download infosave/Spark-X2.5-cmf Spark-X2.5-4B-q8_2f.cmf --local-dir .
cortiq run Spark-X2.5-4B-q8_2f.cmf --prompt "Explain quicksort in three sentences." --no-think
~~~

Cortiq uses Metal on Apple silicon, Vulkan/DX12 where available, or CPU. See the
[model card](https://huggingface.co/infosave/Spark-X2.5-cmf) for exact file
sizes, measured speed, long-context memory, API use and checksums."""
            )

        with gr.Tab("What this demo does — and does not do"):
            gr.Markdown(
                """- **Does:** runs a selected public Spark-X2.5 CMF file locally inside this Space.
- **Does:** demonstrates direct answers, reasoning traces when returned, and tool-call formatting.
- **Does not:** send your prompt to a third-party inference API or execute a proposed tool.
- **Does not:** promise that a demo run is deterministic; use the same model and settings for a comparable rerun.
- **Need production deployment?** Use the documented local Cortiq serve command and the
  OpenAI-compatible API from the model card."""
            )

    ask.click(lambda: set_example("Ask"), outputs=prompt)
    rewrite.click(lambda: set_example("Rewrite"), outputs=prompt)
    translate.click(lambda: set_example("Translate"), outputs=prompt)
    plan.click(lambda: set_example("Plan"), outputs=prompt)

    send.click(
        send_message,
        inputs=[chat, prompt, variant, preset, mode],
        outputs=[chat, prompt, receipt, reasoning, raw],
    )
    prompt.submit(
        send_message,
        inputs=[chat, prompt, variant, preset, mode],
        outputs=[chat, prompt, receipt, reasoning, raw],
    )
    tool_go.click(
        run_tool_lab,
        inputs=[tool_prompt, variant, preset],
        outputs=[tool_status, tool_answer, tool_json, tool_receipt],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=12).launch(
        server_name=GRADIO_HOST,
        server_port=7860,
        show_error=True,
    )
