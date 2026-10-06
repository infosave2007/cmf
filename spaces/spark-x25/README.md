---
title: Spark Lab · CMF
emoji: ✨
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: A clear local Spark-X2.5 CMF playground.
models:
  - infosave/Spark-X2.5-cmf
---

# Spark Lab · CMF

A calm, truthful playground for [Spark-X2.5 — CMF](https://huggingface.co/infosave/Spark-X2.5-cmf).

- Start with a task, not a wall of parameters.
- Choose a 1.7B or 4B quantized model deliberately.
- Switch between a direct answer and a visible reasoning trace.
- Inspect an OpenAI-compatible tool call without allowing the demo to execute it.
- Keep a run receipt with model, mode, preset, latency and usage.

The Space starts a local Cortiq process and sends it requests over loopback.
It does **not** forward prompts to a third-party inference API, and it does not
execute a tool call. The first use of each model may download its public CMF
file, so it can take longer than a warm run.

For production deployment, exact hardware guidance, long-context memory, API
examples and checksums, visit the [model card](https://huggingface.co/infosave/Spark-X2.5-cmf).
