# Model cards, kept in the repository

The card that ships on the Hub is generated from nowhere and edited by hand,
which means the version on the Hub and the truth in this tree drift apart
silently. These files are the source: edit here, then upload.

```bash
hf auth login                       # or: export HF_TOKEN=...
hf upload infosave/DeepSeek-V4-Flash-0731-cmf \
    docs/hf/DeepSeek-V4-Flash-0731-cmf.md README.md
hf upload infosave/MiniMax-H3-Turbo-cmf \
    docs/hf/MiniMax-H3-Turbo-cmf.md README.md
```

A card claims measured numbers. When a release changes them — as 0.5.44 did,
moving DeepSeek-V4-Flash from "CPU only, 0.1–3 tok/s" to 12.7 on one card —
the card is part of the release, not a follow-up.
