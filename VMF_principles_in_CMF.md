Русский: [VMF_principles_in_CMF.ru.md](VMF_principles_in_CMF.ru.md) · 中文: [VMF_principles_in_CMF.zh.md](VMF_principles_in_CMF.zh.md)

# The VMF/NVG principles behind the CMF format

> **Author of the theory and the format:** Oleg Kirichenko
>
> **Theory:** [Null-Vector Gravity / Vacuum Mass Fraction (NVG/VMF)](https://github.com/infosave2007/vmf)
>
> **Format:** [CMF — Cortiq Model Format](https://github.com/infosave2007/cmf)

---

## Introduction

CMF (Cortiq Model Format) — a binary container for storing and serving large language models — is not a purely engineering artifact. Its architecture is deliberately built on analogies and structural principles borrowed from the author's original physical theory, the **Vacuum Mass Fraction (VMF)**, which sits inside the broader framework of **Null-Vector Gravity (NVG)**.

The CMF v2 specification states this openly in its preamble:

> *"Physical basis (VMF): the model is a vacuum condensate 𝒲; a skill is its regular core above the critical density; a task mask selects an active subset without changing the weights. The format carries the consequences of this physics (two-field 𝒲×θ quantization, Born importance, the critical mask threshold) — but **only those confirmed by measurement**."*
>
> — [docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md), specification preamble

What follows traces exactly which NVG/VMF principles were carried into the format, how a physical metaphor became an engineering architecture, and where the author drew the line between "confirmed by measurement" and "declarative metaphor".

---

## 1. The vacuum condensate 𝒲 → the model as one shared backbone

### In physics (VMF)
The core of NVG is the complex order parameter of the vacuum condensate:

$$\Phi(x) = \mathcal{W}(x)\,e^{i\theta(x)}$$

where $\mathcal{W}$ is the condensate amplitude that accounts for ~91% of the nucleon mass. The vacuum is a single, continuous, global object; every observable particle and field is an excitation of it.

### In the format (CMF)
**The model = one shared backbone**, single and indivisible, stored as a monolithic weight block in a single file. Every "specialist" (skill) is not a separate model but a *perturbation* of this one backbone. Storage scales as `|backbone| + Σ|deltas|`, not `N × |model|`.

> **Principle:** Uniqueness of the vacuum → uniqueness of the model backbone. Not N copies of the world, but one world and N local perturbations.

---

## 2. The regular black-hole core → the skill as a local perturbation

### In physics (VMF)
NVG replaces the black-hole singularity with a **regular de Sitter core** (the Hayward metric). As $\rho \to \rho_c$ the vacuum condensate "melts" ($\mathcal{W} \to 0$) but is never destroyed to infinity — a region of finite curvature and finite radius $r_0 = (3M/4\pi\rho_c)^{1/3}$ appears, fully determined by the QCD anchor.

### In the format (CMF)
**A skill is a regular core above the critical density.** A skill stores only the tensors it *replaces* (full-shape replacement tensors) — not low-rank, not diff lists. This is a local region where the "density of deviation from the backbone" exceeds a threshold. The runtime reads either a backbone tensor or a skill tensor, by the principle of **indirect tensor-source addressing** (Patent 19/731,402, claims 1/3/18).

> **Principle:** A regular de Sitter core → a skill. A local region of deviation that does not destroy the overall structure. There are no "singularities" (full copies).

---

## 3. The two-field structure Φ = 𝒲 × e^{iθ} → two-field quantization q8_2f

### In physics (VMF)
The condensate is described by two fields: the amplitude $\mathcal{W}$ (radial mode) and the Goldstone phase $\theta$ (angular mode). The split into amplitude and phase is fundamental — each field has its own dynamics: $\mathcal{W}$ governs mass and energy density, $\theta$ governs time, topology and coherence.

This is the same **Madelung decomposition** of the wavefunction:

$$\psi = \sqrt{\rho}\,e^{iS/\hbar}$$

which NVG derives as a physical hydrodynamics of the vacuum.

### In the format (CMF)
The **`q8_2f`** codec (two-field quantization) stores a tensor as

```
[int8 quantized values][f16 row scale][f16 column field]
```

Reconstruction: `w = q · scale[o] · col[i]` — a product of two fields. One factor (the row scale) is the analog of the amplitude $\mathcal{W}$ (the row scale, a neuron's "density"); the other (the column field) is the analog of the phase $\theta$ (the column structure, the "direction" of an input channel).

Result: +37% accuracy at equal file size versus plain q8; it recovers ~75% of the int8→fp16 quality gap on outlier channels — exactly where scalar quantization makes its largest error.

> **Principle:** The Madelung split Φ = 𝒲 × θ → two-field quantization q8_2f. Amplitude × phase is not an abstraction but a real gain in accuracy.

---

## 4. The VMF principle: a skill selects a subset of the condensate → task masks

### In physics (VMF)
In the physics of dense nuclear matter, the key VMF principle is that the vacuum condensate is *shared*, and a concrete physical situation (density, temperature) *selects an active subset* of its excitations. The condensate is not "created anew" — the needed configuration is *carved out* of it.

### In the format (CMF)
A **task mask** is a bitfield of "what is active" over the shared weights. The weights do not change — the mask selects the subset of neurons and attention heads that participate in the computation. The specification says it directly:

> *"The weights do not change — the VMF principle: a skill selects a subset of the condensate."*

Masks are bit-packed (1 bit per neuron), which gives minimal overhead. Unused neurons cost no RAM thanks to memory mapping (mmap).

> **Principle:** One vacuum, the observable is its subset → one backbone, a task is its mask. Zero copying and zero memory cost for "cold" regions.

---

## 5. The critical density ρ_c → the critical mask threshold

### In physics (VMF)
NVG has a **critical density** $\rho_c = M_{\Omega,0}^4 / (\hbar c)^3 \approx 7.09 \times 10^4$ MeV/fm³, below which the vacuum behaves like ordinary matter and above which a phase transition occurs (condensate melting, violation of the strong energy condition, a cosmological bounce).

### In the format (CMF)
The specification mentions a **critical mask threshold** as a consequence of VMF physics. It is a sparsity threshold above which a mask becomes meaningful — not every set of "switched-off" neurons is a meaningful skill. Below a certain sparsity, a mask is indistinguishable from noise.

> **Principle:** Critical density ρ_c → the critical mask threshold. A phase transition: below it, the shared backbone; above it, a specialized skill.

---

## 6. Holographic information preservation → integrity hashing (hash64)

### In physics (VMF)
NVG postulates that information is **never destroyed**: holographic entropy is compressed by a factor of $\sim 10^{32}$ inside the regular core, but not lost. Unitary transfer: $\mathcal{I}_{n+1} = \mathcal{U}_b\,\mathcal{I}_n$ — information is carried across every bounce.

### In the format (CMF)
A **per-tensor hash64** — a 64-bit hash of every tensor, recorded in the directory. A `.cmf` file is either valid or `open()` returns an error; there is no third state. `cortiq verify` checks the whole chain: envelope → header → directory → every tensor.

> *"A file is either valid or open() returns an error — there is no third state."*

> **Principle:** Information is not destroyed → integrity is guaranteed. No "silent corruption", no "read best-effort".

---

## 7. The single action S → the single canon of the format

### In physics (VMF)
The whole of NVG is derived from a **single action functional**:

$$S = \int d^4x\,\sqrt{-g}\left[\frac{R}{16\pi G} - g^{\mu\nu}\partial_\mu\Phi^*\partial_\nu\Phi - V(|\Phi|) - \frac{1}{4}Z_{\rm EM}(\mathcal{W})F_{\mu\nu}F^{\mu\nu} + \ldots\right]$$

One formula — three pillars: dense matter, cyclic cosmology, black holes.

### In the format (CMF)
The architectural motto is **"a single canon"** — one layout at each level (envelope, directory, quant block, mask), **never two definitions of the same thing**. The tensor-directory and quantization formats are byte-for-byte identical between `.cmf` and `.vmfc` (the internal vmfcore format):

> *"Harmony is achieved not by a count of features but by a single canon: one layout per level, byte-compatible with the validated `.vmfc` v2 format."*

> **Principle:** A single action → a single canon. A minimal set of primitives from which everything is derived.

---

## 8. Cyclic cosmology (the Tolman chain) → append-only growth + compaction

### In physics (VMF)
The NVG universe is cyclic: each cycle produces irreversible entropy that survives the bounce. Each new cycle is larger than the last: $M \times 2$ per cycle, the "Tolman snowball".

### In the format (CMF)
**Append-only growth** (Patent 19/731,402, claim 11): adding a new skill = appending tensors to the end of the file + rewriting the directory/header at the tail + updating the envelope offsets. The bytes and offsets of previously written tensors *never change*. Old directories become "dead tails" — an analog of the relic radiation of previous cycles.

**Compaction** = a full rewrite — an analog of a new cycle starting from a clean state but preserving all accumulated information.

> **Principle:** The Tolman chain → append-only growth. Old data is not destroyed, new data is accreted. Compaction = a new cycle.

---

## 9. The Born rule and the importance of observation → the quality contract and honest measurement

### In physics (VMF)
The Born rule in NVG's reading: $P \propto e^{-V(\theta)/T}$ — the probability of observation is set by the condensate potential, not by "declaration". "Measurement" is the physical process of thermalizing the phase $\theta$ with the apparatus.

### In the format (CMF)
CMF draws a hard line between **measured** and **declarative**:

- The `quality` field in a mask's metadata is a **held-out-data contract**, not a declaration. A converter without a measured metric writes `null`; the runtime prints a warning when switching to an unmeasured mask.
- A default `quality_score: 1.0` is **forbidden**. The list of "anti-features" says it plainly:

> *"Declarative fields — a default `quality_score: 1.0`, area-law 'capacities', Born multipliers in the dynamics: a metaphor does not become a format field until it is measured."*

> **Principle:** The Born rule — observation defines reality → only measured metrics enter the format. A metaphor does not become a field until it is confirmed by measurement.

---

## 10. Violation of the strong energy condition → soft superposition of skills

### In physics (VMF)
When the condensate "melts" ($\mathcal{W} \to 0$), the strong energy condition is violated ($\varepsilon + 3P < 0$) — matter enters a state that "repels" rather than attracts. This is not a collapse but a *mixing* of phases.

### In the format (CMF)
**Soft superposition** (Patent 19/731,402, claim 14): instead of a hard choice of one skill, the runtime can *blend* several skills:

$$T_{\rm blend} = \sum_i w_i \cdot T_i, \qquad w_i = \mathrm{softmax}(-E_i / T)$$

where $E_i$ is the reconstruction error (routing by minimal reconstruction) and $T$ is the "temperature" of the soft choice. This is a direct analog of a superposition of quantum condensate states weighted by energy.

> **Principle:** Superposition of condensate states → soft blending of skills with softmax-of-energy weights.

---

## 11. Resonance and routing → resonance routing (Patent 19/452,440)

### In physics (VMF)
Resonant excitations of the vacuum condensate determine which particles "exist" under given conditions: the $\rho$ meson is a resonance in the $\pi\pi$ channel, its parameters set not by "assignment" but by minimizing the excitation energy.

### In the format (CMF)
**Resonance routing** — unsupervised skill selection by minimizing the reconstruction error:

$$E = \frac{\|r - BB^T r\|^2}{\|\phi\|^2}$$

A skill is chosen as the one whose subspace best "resonates" with the input hidden representation. The file is self-contained for routing — the affine-subspace parameters (mean, basis) live in each skill's selection record.

> **Principle:** Resonance in the condensate → resonance routing. The best response, not a manual assignment.

---

## 12. The router's B-field and variable-bit quantization → "amplitude × frequency"

### In physics (VMF)
In the NVG framework, physical observables are set by the product of the condensate amplitude and the frequency of its excitation. For a mixture of experts, the router decides how often each expert is "excited" (the B-field).

### In the format (CMF)
The variable-bit codec **`vbit`** allocates bits across a tensor's rows in proportion to `log₂(A · B)`, where:
- **A** is the row's logarithmic amplitude (how much information the weights hold);
- **B** is the router's selection frequency (how often the expert is used).

A "loud" expert gets more bits, a "quiet" one is pressed to the minimum floor (lower bound = 3 bits).

> **Principle:** Amplitude × frequency → bit allocation `b ∝ log₂(A·B)`. A physical analogy became a resource-allocation algorithm.

---

## 13. From far away, an object is just a few numbers → constant-memory attention (Patent 19/738,763)

### In physics (VMF)
The field of a collapsed object does not, from outside, remember what it was made of. An arbitrarily complex history — the composition of the matter, the order in which it fell, every detail — is described for a distant observer by just a few quantities: mass, spin, charge. Two completely different objects that agree in those quantities produce the **same** field from far away, and telling them apart from outside is impossible in principle. This is the no-hair theorem. The detail does not vanish into nothing — it stops being distinguishable at a distance.

Close in, everything is different: short of the horizon the field remains exact, there the detail is still distinguishable, and no short list of numbers replaces it.

In the NVG framework the regular core (principle 2) removes the singularity inside, but changes nothing outside: the field is still exhausted by a finite number of quantities.

### In the format (CMF)
**Distant context is stored as a summary, not as tokens.** The `--o1` flag splits a layer's attention along exactly that boundary:

- **anchors** — the first S=4 keys, exact forever: a boundary condition that never "melts" and never passes into the far field;
- **the exact window** — the last W=128 tokens: this is "close in", where the detail is still distinguishable and is kept as-is;
- **the far field** — everything older: compressed into m=32 landmarks. This is a short summary of the same kind (the landmarks themselves are not numbers but vectors of the model dimension), and how many of them there are **does not depend** on how many tokens went into it — a hundred or a hundred thousand.

All three zones live under **one shared softmax denominator**: one field, not a sum of three different fields. A token enters the far field no earlier than it leaves the exact window — exactly once, with no gap and no double counting.

Hence the main consequence: the attention state **stops growing with context length** — 124.1 MB at 543 tokens and at 4127 alike. The weights are byte-for-byte the same: conversion only records a hint in the header, retraining nothing.

> **Principle:** From far away, an object is just a few numbers → distant context is just a few landmarks. A short summary instead of an unbounded history.

**The boundary of transfer** (by the rule of principle 9): the loss of detail is paid for in quality, and the price is **measured, not declared** — perplexity rises by **1.13×** on Qwen3.5-4B and by **1.30×** on Qwen3-0.6B, measured through the real streaming kernel on the region least favorable to it. The more of the model is softmax attention, the more it costs: a hybrid has recurrent layers that carry the distant context themselves, while a pure-attention model has to rely on the summary alone. Checked with a single command: `cortiq ppl model.cmf --file wiki.txt --o1 all`.

---

## Summary table

| NVG/VMF principle | CMF element | Status |
|---|---|---|
| Single vacuum condensate 𝒲 | One shared backbone in a single file | ✅ Implemented |
| Regular de Sitter BH core | Skill as replacement tensors | ✅ Implemented (Patent 19/731,402) |
| Two-field split Φ = 𝒲 × e^{iθ} | Two-field quantization q8_2f (scale × column field) | ✅ Implemented, +37% accuracy |
| VMF: a skill selects a subset | Bit-packed task masks | ✅ Implemented |
| Critical density ρ_c | Critical mask threshold | 📐 Design principle |
| Holographic information preservation | Per-tensor hash64, integrity check | ✅ Implemented |
| Single action S | The format's "single canon" | ✅ Architectural principle |
| Tolman cyclic chain | Append-only growth + compaction | ✅ Implemented (Patent 19/731,402, claim 11) |
| Born rule = measurement | Held-out quality contract | ✅ Implemented, declarative fields forbidden |
| Superposition of condensate states | Soft superposition of skills | ✅ Implemented (Patent 19/731,402, claim 14) |
| Resonant vacuum excitations | Resonance routing (Patent 19/452,440) | ✅ Implemented |
| Amplitude × frequency (B-field) | Variable-bit codec: bit allocation `b ∝ log₂(A·B)` | ✅ Implemented |
| From far away, an object is just a few numbers | Constant-memory attention (`--o1`): distant context as a few landmarks | ✅ Implemented (Patent 19/738,763), cost measured: 1.13× / 1.30× |

---

## The boundary of transfer: metaphor vs. measurement

The author deliberately draws a hard line: **only those consequences of the physics that are confirmed by measurement enter the format.** Three physical metaphors were considered but **not included** in the specification as format fields:

| Metaphor | Why it was not included |
|---|---|
| A default `quality_score: 1.0` | A declaration without measurement — forbidden |
| Area-law "capacities" | The holographic analog had no experimental equivalent |
| Born multipliers in the dynamics | Dynamic Born weights as fields — a metaphor with no metric |

This approach reflects the central scientific principle of NVG/VMF:

> *"A metaphor does not become a format field until it is measured."*

---

## Conclusion

CMF is a rare case where an author's fundamental physical theory was not merely an inspiration but a **constructive foundation** for an engineering format. Thirteen NVG/VMF principles were carried from cosmology and nuclear physics into the architecture of a binary container for language models:

- **Vacuum = the model backbone** (uniqueness)
- **Core = a skill** (a local perturbation without a singularity)
- **Φ = 𝒲 × θ** → **q8_2f** (amplitude × phase)
- **Mask = a subset of the condensate** (zero copying)
- **Critical density = the mask threshold** (phase transition)
- **Information preservation = hash64** (integrity)
- **Single action = a single canon** (harmony)
- **Tolman cycles = append-only growth** (accretion without destruction)
- **Born rule = the quality contract** (measured only)
- **Superposition = soft blending** (of skills)
- **Resonance = routing** (error minimization)
- **A × B = the variable-bit codec** (amplitude × frequency)
- **Far field = landmarks** (a short summary instead of the whole history)

Each of these principles yielded a concrete, measurable engineering gain — from +37% quantization accuracy and constant attention memory at any context length to zero memory cost for unused skills — confirming that the physics of the vacuum condensate turned out to be a fruitful foundation for designing data formats for artificial intelligence.
