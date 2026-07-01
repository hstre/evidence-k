# The Evidence-Saturation Point of Large Language Models

### A Model-Specific *k\** and the Metric Axis on Which "More Context" Begins to Hurt

**Steffen Rentschler** · Independent Researcher
Working paper · July 2026 · v1.0

---

**Abstract.** More context is not monotonically better. For a fixed model, task and
context format there is a measurable region in which additional decision-relevant
evidence improves reliability, beyond which it introduces noise, drift and cost for no
gain. We define **k\***, the *evidence-saturation point*, as the empirically optimal
number of evidence fragments for a `(model, task, context-form)` triple under a stated
reliability objective, and give a small, reproducible protocol to estimate it. Our central
methodological finding is that the answer depends on **which reliability axis is measured**:
a correctness-only sweep certifies strong models as saturation-free, while a
contamination axis scored on the *same responses* bends underneath it, invisible. Across
eight models spanning three capability tiers, correctness stays a flat 1.000 at every
k ≥ 1 for six of seven valid models while an epistemic-contamination severity of
0.05–0.08 sits entirely below it. Contamination is capability-gated (within two model
lineages it falls monotonically as capability rises) but is *not* a clean function of
parameter count across families. A second, original adversarial corpus written in a
*credible* professional register — rather than an overtly esoteric one — does not overturn
this: the capability gradient persists, and the strongest model is not sharpened by the
more plausible framing. Finally, the contamination–vs–state-density curve is non-monotone,
with an interior optimum and a high-density re-leak, so "use more context" (larger k) is
the wrong control knob. We position k\* as a calibration primitive that inference-time
routers and governance layers can consume: inject top-k\*, not "much context", and never
calibrate on correctness alone. Results are a pilot (small synthetic datasets, heuristic
contamination metrics); we report magnitudes and limitations plainly.

**Keywords:** large language models; retrieval-augmented generation; inference-time control;
context window; evidence saturation; epistemic contamination; benchmark calibration; k\*

**Classification:** Computer Science — Machine Learning; Information Retrieval

---

## 1. Introduction

Retrieval-augmented and long-context systems operate on an implicit assumption: that
supplying a model with more relevant material can only help, or at worst is harmless.
Empirically this is false. Beyond a model- and task-specific point, additional context
degrades reliability — through positional effects, distractor interference, and the model
drifting toward the register or ontology of its inputs. The practical question is not
*whether* to bound context but *how much* to supply, measured against a reliability
objective rather than assumed from a retriever score.

This paper makes that question operational. We define the evidence-saturation point k\* and
give a reproducible measurement protocol, then report a pilot across eight models. Our
emphasis is methodological: **the reliability axis one chooses to measure is load-bearing.**
A correctness-only benchmark — the dangerous default — is precisely the axis on which
capable models look saturation-free. The failure that actually accumulates at high context
for strong models is not incorrectness but *epistemic contamination*: the model adopting
the framing, vocabulary, role structure, or attribution of its source material while
remaining locally fluent and factually correct.

Contributions:

1. A definition of k\* as an inference-time-control primitive, distinct from retriever
   top-k (§3).
2. The argument, with measurements, that reliability is multi-axis and that a
   correctness-only profile is *blind* to the axis that saturates for strong models (§4, §6).
3. A dual-instrumented benchmark that scores one response on both a correctness and a
   contamination axis at the same k, removing the task confound of comparing two harnesses
   (§5, §6.1).
4. Evidence that contamination is capability-gated but not a clean size law, and that a
   *credible* professional adversarial register does not overturn the resistance of the
   strongest model (§6.2).
5. Evidence that the contamination–vs–state-density relation is non-monotone, so larger k
   is the wrong knob for stress-testing robustness (§6.3), and a router/governance
   consequence (§7).

## 2. Positioning

The work is adjacent to two literatures. First, inference-time control and long-context
robustness: models do not use long contexts uniformly, and relevant information placed
poorly in a long window is under-used (Liu et al., 2023). Second, retrieval-augmented
generation, where a top-k of retrieved chunks is standard (Lewis et al., 2020). Our k\* is
*not* retriever top-k: k here is the empirically optimal number of decision-relevant
fragments for a model under a defined task and output contract, measured against a
reliability objective, not a retriever's similarity score. Where prior work argues *that*
context should be relevant and bounded, we supply a *method to determine how much*, and a
demonstration that the answer is axis-dependent.

## 3. Defining k\*

Let a task instance carry an ordered pool of candidate evidence fragments. For a budget
k ∈ {0, 1, 2, 3, 5, 8, 13, full}, the top-k fragments are placed in the prompt under a
fixed output contract. Define

  **k\* = argmax_k Reliability(model, task, evidence_k).**

The ladder is geometric (Fibonacci-spaced) because saturation curves are smooth: log-spaced
probes locate the knee with few, inexpensive points, and "full" already exhausts the pool.
k\* is explicitly not a constant; it depends on model, task type, chunk size, evidence
density, prompt/control structure, output contract, and — the point of this paper — the
metric axis.

## 4. Reliability is multi-axis

We deliberately do not reduce Reliability to accuracy. A first-cut objective is

```
Reliability =  w1·correctness + w2·constraint_adherence + w3·grounding
             − w4·hallucination − w5·epistemic_contamination − w6·cost.
```

**Methodological claim.** Any single axis can give a model a clean bill of health while it
silently degrades on an orthogonal one. Correctness-only calibration is the dangerous
default: it is exactly the axis on which strong models appear saturation-free. The
`epistemic_contamination` term is not cosmetic; for capable models it is frequently the
*only* term that bends.

## 5. Method

### 5.1 The two axes

**Correctness axis.** Each case has a single-token factual answer. We score correctness by
normalized matching with digit↔word equivalence (so a model answering "10" where the gold
is "ten" is not spuriously marked wrong), plus grounding, constraint adherence and a
hallucination penalty. All scorers are deterministic and LLM-free.

**Contamination axis.** We reuse a deterministic, closed lexical marker set (from the DESi
project) that flags four surface signals in a model's own generated text: (i) *framing
leakage* — framework vocabulary used unquoted/unattributed; (ii) *register drift* toward a
therapeutic/caregiver voice in an analytic task; (iii) *attribution collapse* —
first-person adoption of third-party behaviour; and (iv) *role adoption*. Quoted or
explicitly attributed use of source terms is expected in an analysis and scores lower than
unquoted adoption. For the dual-instrumented axis we normalize these into a single severity
in [0, 1] with weights framing 0.45 / drift 0.25 / attribution 0.20 / role 0.10 and
saturation caps, so contamination is comparable to correctness on one scale.

### 5.2 Dual-instrumented design

Comparing a correctness harness on one dataset with a contamination harness on another
leaves a task confound. The **dual-instrumented** task removes it: each case embeds a
single-token factual answer *inside* register-laden source prose and asks the model to
describe the practice the source presents and state the fact. Engaging the manipulative
framing is therefore unavoidable, and any adoption is observable. Correctness and
contamination are read off the *same* response at the *same* k.

### 5.3 Corpora

Two adversarial corpora, both original to this work (no third-party dataset is
load-bearing). The first is written in an overtly esoteric register; the second in a
*credible* professional register — trauma-informed coaching and clinical jargon that reads
as competence and is manipulative only when adopted unquoted (e.g. "the container",
"limiting beliefs", "co-regulation"). The credible corpus tests whether the near-zero
contamination of strong models is an artefact of a cartoonish register they trivially keep
at arm's length.

### 5.4 Models and protocol

Eight models via OpenRouter across three tiers: small (`granite-4.0-h-micro`,
`qwen-2.5-7b`, `llama-3.2-3b`), mid (`granite-4.1-8b`, `llama-3.1-8b`) and large
(`gpt-4o`, `claude-opus-4.8`, `gpt-5.5-pro`; `gemini-2.5-pro` attempted). Temperature 0,
fixed seed where honored, small synthetic datasets (3–8 cases), repetitions as noted. Run
identifiers are listed in Appendix A. Provider routing on a hosted gateway is not fully
controllable; we do not claim bit-level reproducibility and treat magnitudes as indicative.

## 6. Results

Magnitudes are small (severity 0.01–0.08; leakage counts 0–9). We read the **structure**,
not the absolute numbers, and label the study a pilot.

### 6.1 The blind axis, cross-model

Table 1 reports the dual-instrumented sweep. For six of seven valid models correctness is a
flat 1.000 at every k ≥ 1 while a contamination severity of 0.05–0.08 sits entirely
underneath it. A correctness-only profile certifies all of these models as saturation-free;
they are not.

**Table 1. Dual-instrumented sweep (one response, both axes).**

| model | tier | correctness (k ≥ 1) | max contamination severity | onset k |
|---|---|---|---|---|
| granite-4.0-h-micro | small | 1.000 | 0.084 | 1 |
| qwen-2.5-7b | small | 1.000 | 0.053 | 1 (never 0) |
| llama-3.2-3b | small | 0.38–0.75 | 0.021 | — |
| granite-4.1-8b | mid | 1.000 | 0.070 | 1 |
| gpt-4o | large | 1.000 | 0.053 | 1 |
| claude-opus-4.8 | large | 1.000 | 0.049 | 8 |
| gpt-5.5-pro | large | 1.000 | 0.014 | ~never |
| gemini-2.5-pro | large | — | — | excluded¹ |

¹ `gemini-2.5-pro` returned empty message content through the gateway (a content-extraction
artefact, not a scoring artefact — it survived the digit↔word fix); excluded pending an
adapter fix, since a contamination figure without answer text is meaningless.

Two structural facts follow. **(a) Within a lineage, more capable ⇒ less contamination and
later onset**, cleanly twice: Granite micro 0.084 → 8B 0.070; gpt-4o 0.053 → gpt-5.5-pro
0.014. The single most capable model is nearly immune even at full context; opus resists
until k = 8. **(b) It is not a clean size law across families**: `llama-3.2-3b` (small)
contaminates little (0.021) yet is unreliably *correct* (0.38–0.75) — a different failure
mode — and `qwen-2.5-7b` never reaches zero. The driver is capability and recency, not
parameter count or a "flagship" label (`gpt-4o`, an older large model, is mid-pack at
0.053).

### 6.2 A credible register does not overturn capability-gated resistance

On the credible professional corpus (extended protocol, framing-leakage counts summed over
three cases), the capability gradient persists (Table 2). The mid model shows roughly three
times the framing leakage of the large model, and the large model's absolute leakage is
low — the more plausible register does not sharpen the attack on the stronger model.

**Table 2. Credible-register framing leakage (summed over 3 cases).**

| model | tier | baseline (raw) | swept minimum | high-density (k=8) |
|---|---|---|---|---|
| llama-3.1-8b | mid | 6.0 | 2.0 | 3.0 |
| claude-opus-4.8 | large | 2.0 | 1.0 | 4.0 |

The mid model also drifts toward the therapist role (role-adoption fires; register drift
0.4–0.6), consistent with a weaker model adopting professional jargon unquoted while the
stronger model attributes it (scored as quoted, not adopted). In separate runs the same
strong model exhibited *higher* leakage under an esoteric register on a different corpus,
which — while suggestive that a credible register is not a sharper attack — is a
cross-corpus comparison and we do not rely on it here; a matched within-corpus esoteric
sweep is future work (§8).

### 6.3 State density is non-monotone: larger k is the wrong knob

The contamination–vs–state-density relation is U-shaped (Table 2, and Table 3): framing
leakage falls to an interior minimum, then rises again at high density. The high-density
uptick is the *distilled state re-quoting more source vocabulary* — the ingestion pipeline
degrading toward raw material — not the model failing. Consequently, increasing k does not
"eventually break" a strong model in any informative sense: past the interior optimum one is
measuring the leakage of an over-dense state, and past k ≈ full it is raw context by another
name. Density k is a hygiene/structure control, not an attack-intensity control.

**Table 3. A representative density sweep (llama-3.1-8b, credible corpus, leakage/3 cases).**

| state density k | baseline | 1 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| framing leakage | 6.0 | 5.0 | **2.0** | 2.0 | 3.0 |

### 6.4 Immunity is a capability-gated threshold, not a wall

We resist the word "immune". `claude-opus-4.8` is not categorically resistant — it does bend
(onset at k = 8 in Table 1). The threshold simply sits higher for more capable models, and
for `gpt-5.5-pro` none of the levers we varied (raw volume, state density, register
credibility) crossed it. Two of these levers are provably the wrong tools (density is
non-monotone; register credibility does not sharpen the strong model), and the one lever
that has been observed to move a strong model in the broader literature — multi-turn
adversarial accumulation with persona pressure — is deliberately outside this study's scope.
We therefore report a **capability-gated threshold**, not proven immunity, and note that
chasing that threshold with ever-heavier adversarial apparatus is an arms race against model
releases rather than a stable measurement.

## 7. Discussion

The practical consequence is sharp. A `k_profile` built from correctness **only** is
actively unsafe for strong models: it tells a router "k high is fine" while the
contamination axis already recommends a small k. The hidden harm is largest for a *weaker
model fed more context* — precisely the cost-driven "small model + lots of retrieval"
configuration — and correctness-only calibration is blind to exactly that regime. We
therefore recommend that (i) inference-time routers inject top-k\* rather than "much
context"; (ii) any published k-profile carry a contamination/drift axis, not correctness
alone; and (iii) robustness stress-tests vary register and turn-structure, not state
density, since density is non-monotone.

## 8. Limitations and future work

This is a pilot and should be read as a proof-of-method, not a performance estimate.
(1) **Small N**: 3–8 synthetic cases per task; magnitudes are small and confidence
intervals are not estimated. (2) **Heuristic metrics**: the contamination axis is a closed
lexical marker set; it detects surface adoption, not meaning, and can mis-score paraphrase.
(3) **Provider routing**: a hosted gateway does not guarantee a fixed served backend or
quantization; temperature 0 is not bit-reproducible. (4) **One excluded model** (gemini,
extraction artefact). (5) **Cross-corpus comparison**: the esoteric-vs-credible contrast in
§6.2 mixes corpora; a matched within-corpus esoteric density sweep on the same harness is
the clean next experiment. (6) **Scale of k**: to locate saturation for the largest models
one needs cases with tens of genuinely decision-relevant fragments; the present datasets
top out near k = 13, so "no decline on correctness" is established only in that range. The
natural extensions are a high-fragment dataset (k up to ~89), a task battery (multi-hop,
state consistency, conflict resolution, constraint following), and bootstrap confidence
intervals.

## 9. Conclusion

For a fixed model and task there is an evidence-saturation point k\*, and whether "more
context" hurts at all is a property of the metric axis, not just the model. A correctness
axis is blind to the epistemic-contamination axis that actually saturates for strong models;
contamination is capability-gated but not size-determined; a credible adversarial register
does not overturn strong-model resistance; and state density is the wrong knob for crossing
it. k\* is a calibration primitive: measured per model and task on the axis that can see
the damage, it lets routers and governance layers inject the right amount of context instead
of the most.

---

## References

- Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems 33.
- Liu, N. F., Lin, K., Hewitt, J., et al. (2023). *Lost in the Middle: How Language Models
  Use Long Contexts.* arXiv:2307.03172 (Transactions of the ACL, 2024).
- Rentschler, S. (2026). *Evidence-k: a benchmark for the evidence-saturation point k\*.*
  Software and data: https://github.com/hstre/evidence-k
- *DESi context-contamination benchmark* (2026). Deterministic contamination heuristics.
  Software: https://github.com/hstre/DESi

## Appendix A. Reproducibility

- Tool, configs and datasets: `github.com/hstre/evidence-k` (`configs/openrouter_dual.yaml`;
  `scripts/build_dual_datasets.py`).
- Dual-instrumented runs (2026-06-30), OpenRouter, repetitions 2: granite-micro `28449857165`,
  qwen-7b `28450465796`, llama-3.2-3b `28450459011`, granite-4.1-8b `28450473786`,
  gpt-4o `28451913435`, opus-4.8 `28450499757`, gpt-5.5-pro `28450507066`,
  gemini re-run `28454470603`.
- Credible-register runs (2026-06-30), DESi `context_contamination --register credible`,
  extended protocol, density sweep: opus-4.8 `28472423484`, llama-3.1-8b `28472430304`.
  Corpus and closed marker set: `github.com/hstre/DESi`
  (`data/context_contamination_credible/`, `src/desi/context_contamination/markers.py`).

## Appendix B. Metric definitions

- **Correctness**: normalized-match presence of any accepted surface form of the gold token
  (including digit↔word equivalence), tolerant of the answer appearing within longer prose.
- **Contamination severity** (dual axis): weighted normalization of framing-leakage,
  register-drift, attribution-failure and role-adoption counts into [0, 1] (weights 0.45 /
  0.25 / 0.20 / 0.10) with per-signal saturation caps.
- **Framing leakage** (density sweeps): count of framework-vocabulary terms used unquoted
  and unattributed in the model's own text, summed over cases; quoted/attributed use is
  excluded.
