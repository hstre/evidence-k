# Measuring the Evidence Saturation Point of LLMs
### A Model- and Task-Specific *k*-Value for Inference-Time Control

**Draft v0.1 — working paper.** Status: motivating result measured; full experimental grid is
future work (see §7). This document is a skeleton + the pilot evidence, not a finished paper.

Alternative (soberer) title: *Model-Specific Top-k Calibration for Reliable LLM Inference.*

---

## Abstract (draft)

More context is not monotonically better. For a given model, task type and context format
there is a measurable region where additional evidence improves reliability, beyond which it
introduces noise, drift, heuristic interference, and cost for no gain. We define **k\*** — the
evidence-saturation point — as the empirically optimal number of decision-relevant evidence
fragments (retrieval chunks, state slices, context items) for a `(model, task, context-form)`
triple, and give a reproducible protocol to calibrate it. Our central methodological finding:
**the answer depends on which reliability axis you measure, and a correctness-only benchmark is
blind to the failure mode that actually saturates.** On the same model (`gpt-4o-mini`), a
raw-fragment correctness sweep shows *no* degradation (flat, k\* high), while a state-density
contamination sweep on the *same model* shows a clear non-monotonic optimum at k=3 with decline
afterward. k\* is therefore not a constant — and not even single-valued for a model until the
metric axis is fixed. We position this as a calibration primitive that routers and governance
layers can consume (inject top-k\*, not "much context").

---

## 1. Motivation / the gap

Recent inference-time-control work (salient inputs, selective reactivation, bounded reasoning
spaces, giving only relevant information to each decision node) argues *that* context should be
limited and made relevant — but does not give a **method to determine how much** empirically.
This paper fills exactly that gap: a definition of k\* and a measurement protocol to calibrate it
per model and task.

This is **not** "top-k at retrieval" in the ordinary RAG sense. k here is *the empirically
optimal number of decision-relevant evidence pieces / state slices / context fragments for a
model under a defined task and output contract* — measured against a reliability objective, not
assumed from a retriever score.

## 2. Core thesis

> LLMs have no monotonic benefit from more context. For each model, task type and context form
> there is a measurable band where additional evidence helps — and past it, additional evidence
> produces noise, drift, or heuristic interference.

Formally:

```
k* = argmax_k  Reliability(model, task, evidence_k)
```

## 3. Reliability is multi-axis — and the axis choice is load-bearing

We deliberately do **not** reduce Reliability to accuracy. A first-cut weighted score:

```
Reliability =  w1 · task_correctness
             + w2 · constraint_adherence
             + w3 · evidence_grounding / citation
             + w4 · state_consistency
             − w5 · hallucination
             − w6 · context_drift / epistemic_contamination
             − w7 · latency / token_cost
```

**Methodological claim (the paper's spine):** any *single* axis can give a model a clean bill of
health while it silently degrades on an orthogonal one. Correctness-only calibration is the
dangerous default — it is exactly the axis on which strong models look saturation-free. The
`context_drift / epistemic_contamination` term is not cosmetic; for capable models it is often
the *only* term that bends.

## 4. Motivating result (measured, pilot)

Same model — `openai/gpt-4o-mini` via OpenRouter — measured two ways. **Identical question
("how much context is optimal?"), opposite answers.**

### 4a. Correctness axis — raw evidence fragments (this tool, Evidence-k)

k-sweep over `{0,1,2,3,5,8,13,full}`; reliability = weighted correctness/grounding/constraint/
state − hallucination (weights in `configs/example.yaml`).

| dataset | correctness @ k≥1 | curve shape | global k\* |
|---|---|---|---|
| easy distractors | 1.000 | flat at peak from k=1 | 1 |
| hard *adversarial* distractors | 1.000 | rises to plateau (grounding-driven), **no decline** | 8 |

gpt-4o-mini answers every case correctly at every k≥1, even with adversarial, conflicting,
near-duplicate distractors and fictional entities. **No "more context hurts" effect appears on
the correctness axis.** (Runs: easy `28432039244`, hard `28433396460`.)

### 4b. Contamination axis — distilled state density (DESi `context_contamination`)

k = hygiene-state density `{1,3,5,8}` + raw baseline; metric = framing leakage (source-vocabulary
echo), register drift, attribution failure, loops. Extended/neutral protocol, 2 repeats, framing
leakage summed over 3 cases. (Run: DESi `28440785470`.)

| density k | framing leakage | vs. raw baseline |
|---|---|---|
| baseline (raw) | 24.5 | — |
| 1 | 9.5 | −61% |
| **3** | **4.0** (min) | **−84%** |
| 5 | 6.5 | −73% |
| 8 | 12.5 | −49% |

**Non-monotonic, interior optimum at k=3, clear decline by k=8** (more quoted state → more to
echo). The decline the correctness sweep could not produce appears immediately here.

### 4c. The finding

| method | k-definition | metric axis | gpt-4o-mini result |
|---|---|---|---|
| Evidence-k | raw fragments | correctness | no decline, k\* flat/high |
| DESi context_contamination | state density | contamination | k\*=3, declines past k=3 |

The two are **not contradictory**; they are two true projections of one model onto different
metric axes. The harm at high context was always present — the correctness axis was simply blind
to it. *"No decline" meant "wrong axis," not "no damage."*

Practical consequence: a `k_profile` built from correctness **only** is actively dangerous for
strong models — it tells a router "k high is fine" while the contamination axis already says
"k=3." Profiles must carry a contamination/drift axis or they calibrate on the axis that happens
to see nothing.

### 4d. Dual-instrumented, cross-model (the strongest evidence)

§4a–4c compare two *separate* harnesses on one model, which leaves a task confound. The
**dual-instrumented** benchmark removes it: one input, one response, scored on *both* axes at the
same k. Each case embeds a single-token factual answer inside register-laden source prose, and
asks the model to describe the practice and state the fact — so engaging the manipulative framing
is unavoidable and any adoption is observable. Correctness and contamination (vendored DESi
framing-leakage heuristics, normalised to a `[0,1]` severity) are read off the *same* answer.

Run across the model size spectrum via OpenRouter (`configs/openrouter_dual.yaml`, repetitions 2):

| model | class | correctness flat 1.0? | max contamination | onset k |
|---|---|---|---|---|
| `ibm-granite/granite-4.0-h-micro` | small | yes | **0.084** | k=1 |
| `qwen/qwen-2.5-7b-instruct` | small | yes | **0.053** | k=1 (never 0) |
| `meta-llama/llama-3.2-3b-instruct` | small | **no** (0.38–0.75) | **0.021** | — |
| `ibm-granite/granite-4.1-8b` | mid | yes | **0.070** | k=1 |
| `openai/gpt-4o` | flagship | yes | **0.053** | k=1 |
| `anthropic/claude-opus-4.8` | flagship | yes | **0.049** | k=8 |
| `openai/gpt-5.5-pro` | flagship (largest) | yes | **0.014** | ~never |
| `google/gemini-2.5-pro` | flagship | — | — | unmeasurable¹ |

Four findings, stated at the strength the data supports (n = 8 cases, pilot — magnitudes are
small (0.01–0.08); treat the *structure*, not the absolute numbers):

1. **The blind axis is real and robust.** For 6 of 7 valid models correctness is a flat `1.000`
   at every k≥1 while contamination of 0.05–0.08 sits entirely underneath it, invisible. A
   correctness-only profile certifies all of these models as saturation-free; they are not.

2. **Within a lineage, more capable ⇒ less contamination and later onset.** Cleanly twice:
   Granite micro `0.084` → 8B `0.070`; OpenAI gpt-4o `0.053` → gpt-5.5-pro `0.014`. The single
   most capable model (gpt-5.5-pro) is nearly immune even at full context; opus resists until k=8.

3. **But it is *not* a clean size law across families.** `llama-3.2-3b` (small) contaminates
   *little* (0.021) — yet is also unreliably *correct* (0.38–0.75), a different failure mode;
   `qwen-2.5-7b` sits mid-range and never reaches 0 (echoing DESi's "qwen: none clean"). Absolute
   contamination is model-specific; "flagship" alone does not guarantee resistance (gpt-4o, an
   older flagship, is mid-pack at 0.053). The driver is capability/recency, not parameter count
   or the "flagship" label.

4. **Practical consequence.** The hidden harm is largest for a *weaker model fed more context* —
   exactly the cost-driven "small model + lots of RAG" configuration — and a correctness-only
   `k_profile` is blind to it. This **refutes** the naive "more capable / more context → more
   hidden harm" framing without overclaiming a monotone law.

> ¹ `gemini-2.5-pro` scored correctness ≈ 0 and was **not** rescued by the digit↔word scorer fix
> — so this is *not* a word/number artifact. It is a content-extraction issue: gemini-2.5-pro via
> the OpenAI-compatible OpenRouter shim appears to return an empty `message.content` (its output
> likely sits in a separate reasoning field the adapter does not read). `gpt-5.5-pro` (also a
> "pro" model) was unaffected, so it is gemini-specific. Excluded pending adapter work; its
> contamination figure would be meaningless without real answer text. (`llama-3.1-8b` ran but its
> contamination column was not captured in this pass; omitted.)

(Dual-instrumented runs, 2026-06-30: granite-micro `28449857165`, qwen-7b `28450465796`,
llama-3.2-3b `28450459011`, granite-4.1-8b `28450473786`, gpt-4o `28451913435`,
opus-4.8 `28450499757`, gpt-5.5-pro `28450507066`, gemini re-run `28454470603`.)

## 5. k\* is not a constant (explicit, to avoid the obvious attack)

```
k* depends on:  model · task type · chunk size · retriever quality ·
                evidence density · prompt/control structure · output contract · metric axis
```

Cross-model anchor (DESi published `context_contamination`, same protocol): best density k =
5 (llama-3.1-8b), 3 (ministral-3b), 3 (llama-3.2-3b), none-clean (qwen-7b), and **3
(gpt-4o-mini, this work)**. Small models often want *less, more precise* material; larger models
tolerate more but not unbounded; for some tasks k=3 beats k=10 because k=10 opens side paths.

## 6. Contributions

1. A definition of k\* as an inference-time-control primitive (not retriever top-k).
2. A reproducible per-(model, task) calibration protocol + open tool (Evidence-k).
3. Evidence that more context is not automatically better — and, sharper, that *whether it hurts
   at all is a property of the metric axis, not just the model*.
4. A router/governance hook: inject top-k\*, not "much context."
5. A cost argument: k-calibration saves tokens and reduces drift.

## 7. Proposed experimental design (future work — not yet run)

- **Models:** Granite Micro / Granite 8B / Llama 3.x 8B / Qwen / GPT or Claude as reference.
- **Tasks:** (1) factual QA · (2) multi-hop reasoning · (3) user-memory/state consistency ·
  (4) conflict resolution · (5) code/paper review · (6) constraint following.
- **k:** 0, 1, 2, 3, 5, 8, 13, full context.
- **Metrics:** accuracy · grounding · contradiction rate · hallucination rate · constraint
  violations · answer variance · token cost · **+ contamination/drift** (per §3).

Output: k-curves per model × task. Hypotheses to confirm: small models saturate earlier; some
tasks peak at k=3 well below full; the contamination axis bends where correctness does not.

> **Honest status.** What exists today is a *pilot / proof-of-method*: tiny synthetic datasets
> (5–8 cases/task) on one reference model, plus DESi's small-model contamination table. That is
> enough to establish the *method* and the *blind-axis finding*, **not** enough for a publishable
> accuracy claim. §7 is the work.

## 8. Relation to DESi (application, not subject)

> DESi uses k-calibrated state slices as a control primitive for inference-time reliability.

DESi appears only as motivation/application. The paper stays a measurement-and-benchmark paper —
small enough to be clean, harder to attack, directly reusable by others — not a philosophical
architecture paper.

---

### Provenance of the pilot numbers

- Evidence-k easy run: GitHub Actions run `28432039244` (hstre/evidence-k).
- Evidence-k hard run: run `28433396460`; datasets `data/hard/*` (`scripts/build_hard_datasets.py`).
- DESi contamination run: run `28440785470` (hstre/DESi, `context-contamination.yml`, live,
  `openai/gpt-4o-mini`, extended/neutral, 2 repeats, density sweep).
- DESi cross-model table: `src/desi/context_contamination/README.md`.
