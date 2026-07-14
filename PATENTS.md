# Patents

CMF (Cortiq Model Format) implements methods that are the subject of the
following United States patent applications filed by the author, Oleg
Urevich Kirichenko:

| # | Application Serial No. | Filed | Title |
|---|------------------------|-------|-------|
| 1 | 19/452,440 | January 19, 2026 | Resonance Routing (unsupervised task selection by reconstruction-error minimization) |
| 2 | 19/452,464 | January 19, 2026 | Dynamic Task-Guided Mask Activation (DTG-MA) Compression |
| 3 | 19/731,402 | July 6, 2026 | Unified Execution Architecture for Serving a Plurality of Specialized Language Models from a Single Shared Backbone via Dynamically Overlaid Compressed Delta Representations Without Materializing Separate Models |
| 4 | 19/738,763 | July 13, 2026 | Training-Free Constant-Memory Streaming Attention Conversion |

Where these methods appear in this repository:

- **Resonance Routing (App. 19/452,440)** — the reconstruction-error skill
  selector used by the runtime and the `selection descriptor` records in the
  container.
- **DTG-MA Compression (App. 19/452,464)** — the task-guided mask/skill
  production path used by the converter to derive per-skill delta records.
- **Unified Execution Architecture (App. 19/731,402)** — the CMF container
  itself (shared backbone stored once + per-skill full-shape replacement
  tensors + byte-offset delta index) and the dependency-free overlay runtime
  that reads replacement tensors *in place of* the backbone at forward time
  without materializing a per-skill model.
- **Streaming Attention Conversion (App. 19/738,763)** — the training-free
  O(1) attention path: `cortiq convert --o1` (weights byte-identical), the
  streaming sink/window/landmark-skeleton kernel with delayed insertion and
  a single joint denominator (`nystrom.rs`, runtime seal that releases the
  KV cache), and the generation-gated `cortiq fcd` restoration stage.

## Patent grant

This project is licensed under the **Apache License, Version 2.0**. Section 3
of that license ("Grant of Patent License") gives every user a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license
to make, have made, use, offer to sell, sell, import, and otherwise transfer
the Work, for those patent claims of the above applications that are
necessarily infringed by this software as distributed. That patent license
terminates for any party that initiates patent litigation alleging that this
software infringes a patent (Apache-2.0 §3).

In plain terms: **you may use, modify, and redistribute this software —
including for commercial purposes — under Apache-2.0, and you receive a
patent license to the claims practiced by the code as shipped.** The
underlying inventions remain patented; this grant is scoped to this software,
consistent with Apache-2.0.

This file is informational and does not modify the Apache License, Version 2.0,
which is the sole governing license (see `LICENSE`).
