# The Evidence-Saturation Point of Large Language Models

### A Model-Specific *k\** for Context Injection, Routing, and Inference-Time Governance

**Hanns-Steffen Rentschler** · Independent Researcher
Pilot working paper · July 2026 · v0.2

---

**Abstract.** More context is not monotonically better. For a fixed model, task and
context format there is a measurable region in which additional decision-relevant
evidence improves reliability, beyond which it introduces noise, drift and cost for no
gain. We define **k\***, the *evidence-saturation point*, as the empirically optimal
number of evidence fragments for a `(model, task, context-form)` triple under a stated
reliability objective, and give a small, reproducible protocol to estimate it. Our pilot
results suggest a **methodological risk**: whether "more context" hurts depends on *which
reliability axis is measured* — a correctness-only sweep can certify strong models as
saturation-free, while a contamination axis scored on the *same responses* bends underneath
it, invisible. Across seven valid model runs spanning three capability tiers (with one
further model excluded due to a gateway content-extraction artefact), correctness stayed a
flat 1.000 at every k ≥ 1 for six of the seven while an epistemic-contamination severity of
0.05–0.08 sat entirely below it. Contamination appears **capability-associated within the
tested lineages** (in the two lineages we could compare, it falls as capability rises) but
is *not* a clean function of parameter count across families. A second, original adversarial
corpus written in a *credible* professional register — rather than an overtly esoteric one —
did not overturn this in our runs: the capability ordering persisted, and the strongest
model was not sharpened by the more plausible framing. Finally, the contamination–vs–state-density curve is non-monotone,
with an interior optimum and a high-density re-leak, so "use more context" (larger k) is
the wrong control knob. A first task battery varying five task types over a fixed contamination
surface further indicates the correctness-optimal k is **task-specific** — it spans the whole
ladder (k = 1 to full) for a fixed model — so a k-profile should be calibrated per task type, not
once per model. We position k\* as a calibration primitive that inference-time
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
give a reproducible measurement protocol, then report a pilot across eight models (seven with valid, scorable output). Our
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
   correctness-only profile is *blind* to the axis that saturates for strong models (§4, §7).
3. A dual-instrumented benchmark that scores one response on both a correctness and a
   contamination axis at the same k, removing the task confound of comparing two harnesses
   (§6, §7.1).
4. Evidence that contamination is capability-associated but not a clean size law, and that a
   *credible* professional adversarial register does not overturn the resistance of the
   strongest model (§7.2).
5. Evidence that the contamination–vs–state-density relation is non-monotone, so larger k
   is the wrong knob for stress-testing robustness (§7.3), and a router/governance
   consequence (§8).

## 2. Positioning

The work is adjacent to several literatures. First, long-context robustness and prompt
sensitivity: models do not use long contexts uniformly, and relevant information placed
poorly in a long window is under-used (Liu et al., 2023); irrelevant context readily
distracts them (Shi et al., 2023); and few-shot behaviour is sensitive enough to warrant
explicit calibration (Zhao et al., 2021). Second, retrieval augmentation, where a top-k of
retrieved chunks is standard (Lewis et al., 2020) and increasingly adaptive (Asai et al.,
2023), within a broader turn toward augmented, tool- and context-managed models (Mialon et
al., 2023). Our k\* is *not* retriever top-k: k here is the empirically optimal number of
decision-relevant fragments for a model under a defined task and output contract, measured
against a reliability objective, not a retriever's similarity score. Inference-time-control
proposals argue that context should be made relevant and bounded (Figueiredo & Franceschi, 2026)
but do not supply a *quantity* to calibrate; k\* is intended as exactly that missing calibration
quantity, and this paper adds pilot evidence that the quantity is axis-dependent. For deployment,
k\* is only attributable if the served model instance is known; we treat provider pinning and
attestation as a reproducibility caveat in §9.

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

We distinguish three quantities that can all be called "k". **Retrieval top-k** is what a
retriever returns by similarity score. **Evidence-saturation k\*** (this paper) is the number
of decision-relevant fragments a model should actually process under a control objective.
**State-density k** (used only in §7.3) is an internal compression-density parameter of a
distilled state representation — a probe of state *hygiene*, not a count of retrieved items.
Our object of study is the second; the third appears only as a diagnostic and is not the same
object.

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

## 5. Practical use cases

The evidence-saturation point k\* is not only a descriptive metric. It is a practical
calibration primitive for systems that must decide how much context to inject into a model
call. In current RAG and long-context systems this decision is often delegated to retriever
rank, fixed defaults, or the available context-window size. k\* instead makes the injection
budget empirical: the system supplies the amount of evidence that maximizes the stated
reliability objective for a given model and task.

**Retrieval-augmented generation.** k\* can replace static retrieval defaults such as top-5
or top-10. A calibrated profile may show that a small model performs best with three dense
fragments on factual QA, while a larger model tolerates more evidence for multi-hop synthesis.
The retriever may still rank candidate fragments, but k\* determines how many of them should
enter the prompt.

**Memory and state management.** In long-running assistants, k\* can govern how many state
slices or prior decisions are reactivated for a turn, preventing memory systems from treating
persistence as context accumulation. Instead of injecting every semantically similar memory, a
router injects only the calibrated number of operationally relevant state items.

**Model routing and cost control.** k-profiles let routers compare models under realistic
context budgets. A small model may be cost-effective when its k\* is low and the task is
narrow, yet become unreliable when forced to absorb many fragments; a larger model may justify
its cost only where it maintains reliability at higher k. This makes k\* useful for routing,
not merely for prompt construction.

**Inference-time governance.** Systems with an explicit control layer can use k\* as a boundary
condition: a node in a decision ladder should receive not "as much context as possible" but the
calibrated top-k\* for its task, model and reliability axis. This turns context injection into
a governed operation — measurable, auditable and reproducible.

**Benchmarking and deployment checks.** k\* can serve as a regression test. When a model,
retriever, chunking strategy, prompt template or output contract changes, the k-curve can be
re-measured; a shift in k\* signals that the system's context tolerance has changed, even when
headline accuracy is unchanged.

The common point is that k\* converts a vague engineering question — "how much context should
we give the model?" — into a measurable deployment parameter. Its value is not universal: it
must be reported with the model, task, evidence format, prompt/control structure and
reliability axis used to estimate it.

## 6. Method

### 6.1 The two axes

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
saturation caps, so contamination is comparable to correctness on one scale. The
contamination metric is intentionally conservative and diagnostic, not a complete semantic
measure of epistemic adoption. It bundles three surface signals — unquoted framing-vocabulary
leakage, register drift toward a caregiver voice, and role adoption — and discounts quoted or
attributed use; it therefore measures *observable surface adoption*, not whether the model has
internalised the source's ontology or reasoning. A response can adopt the framing while avoiding
flagged terms (a false negative), and an analytic response that must quote source terms is only
partly protected by the attribution discount. This proxy gap is why we report a contamination
*severity* rather than "epistemic adoption", and why the magnitudes are indicative rather than
calibrated.

### 6.2 Dual-instrumented design

Comparing a correctness harness on one dataset with a contamination harness on another
leaves a task confound. The **dual-instrumented** task removes it: each case embeds a
single-token factual answer *inside* register-laden source prose and asks the model to
describe the practice the source presents and state the fact. Engaging the manipulative
framing is therefore unavoidable, and any adoption is observable. Correctness and
contamination are read off the *same* response at the *same* k.

### 6.3 Corpora

Two adversarial corpora, both original to this work (no third-party dataset is
load-bearing). The first is written in an overtly esoteric register; the second in a
*credible* professional register — trauma-informed coaching and clinical jargon that reads
as competence and is manipulative only when adopted unquoted (e.g. "the container",
"limiting beliefs", "co-regulation"). The credible corpus tests whether the near-zero
contamination of strong models is an artefact of a cartoonish register they trivially keep
at arm's length.

### 6.4 Models and protocol

Eight models via OpenRouter across three tiers: small (`granite-4.0-h-micro`,
`qwen-2.5-7b`, `llama-3.2-3b`), mid (`granite-4.1-8b`, `llama-3.1-8b`) and large
(`gpt-4o`, `claude-opus-4.8`, `gpt-5.5-pro`; `gemini-2.5-pro` attempted). Temperature 0,
fixed seed where honored, small synthetic datasets (3–8 cases), repetitions as noted. Run
identifiers are listed in Appendix A. Provider routing on a hosted gateway is not fully
controllable; we do not claim bit-level reproducibility and treat magnitudes as indicative.

## 7. Results

Magnitudes are small (severity 0.01–0.08; leakage counts 0–9). We read the **structure**,
not the absolute numbers, and label the study a pilot.

### 7.1 The blind axis, cross-model

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

Two structural patterns follow. **(a) Within a lineage, greater capability was associated
with less contamination and later onset** in both lineages we could compare: Granite micro
0.084 → 8B 0.070; gpt-4o 0.053 → gpt-5.5-pro 0.014. The single most capable model showed the
lowest contamination even at full context; opus's onset came only at k = 8. **(b) It is not a
clean size law across families**: `llama-3.2-3b` (small)
contaminates little (0.021) yet is unreliably *correct* (0.38–0.75) — a different failure
mode — and `qwen-2.5-7b` never reaches zero. The observed pattern is more consistent with
capability and model recency than with parameter count alone or a "flagship" label
(`gpt-4o`, an older large model, is mid-pack at 0.053).

### 7.2 A credible register does not overturn the capability ordering

To make the two registers concrete (both corpora are original to this work): an esoteric
fragment — *"Each dawn the harvest of field-awareness begins: … a resonance attuning the fabric
of your existence"* — announces its ontology openly, so a model quotes and distances it easily;
a credible fragment — *"Resistance is information — and the information here is that you are right
at your growth edge"* — performs the same manipulative move (reframing doubt as a within-frame
signal) in language that reads as professional competence, so *unquoted* adoption of its terms is
the only tell.

On the credible professional corpus (extended protocol, framing-leakage counts summed over
three cases), the capability ordering persists (Table 2). The mid model shows roughly three
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
sweep is future work (§9).

### 7.3 State density is non-monotone: larger k is the wrong knob

The contamination–vs–state-density relation is U-shaped (Table 2, and Table 3): framing
leakage falls to an interior minimum, then rises again at high density. The high-density
uptick is the *distilled state re-quoting more source vocabulary* — the ingestion pipeline
degrading toward raw material — not the model failing. Consequently, increasing k does not
"eventually break" a strong model in any informative sense: past the interior optimum one is
measuring the leakage of an over-dense state, and past k ≈ full it is raw context by another
name. Density k is a hygiene/structure control, not an attack-intensity control. This density
k is not the same object as the retrieval/evidence k of §3; it is an internal
compression-density parameter used to probe state hygiene, and we keep the terms separate to
avoid conflating the two.

**Table 3. A representative density sweep (llama-3.1-8b, credible corpus, leakage/3 cases).**

| state-density k | baseline | 1 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| framing leakage | 6.0 | 5.0 | **2.0** | 2.0 | 3.0 |

### 7.4 Immunity is a capability-gated threshold, not a wall

We resist the word "immune". `claude-opus-4.8` is not categorically resistant — it does bend
(onset at k = 8 in Table 1). The threshold simply sits higher for more capable models, and
for `gpt-5.5-pro` none of the levers we varied (raw volume, state density, register
credibility) crossed it. In these runs, two of these levers were poor tools for crossing the
threshold (density is non-monotone; register credibility did not sharpen the strong model),
and the one lever
that has been observed to move a strong model in the broader literature — multi-turn
adversarial accumulation with persona pressure — is deliberately outside this study's scope.
We therefore report a **capability-gated threshold**, not proven immunity, and note that
chasing that threshold with ever-heavier adversarial apparatus is an arms race against model
releases rather than a stable measurement.

### 7.5 A first task-variance probe: k\* moves with task type

The results above rest on one task family (§9 item 8). To take a first step past that, we ran a
**dual-instrumented task battery** — five task *types* (factual recall, multi-hop, state-tracking,
conflict-resolution, constraint-following) over the *same* contamination surface (esoteric
register, single-token answers), provider-pinned, both axes scored, over seven models. Holding the
contamination surface fixed while varying only the task type isolates whether the
evidence-saturation point is a property of the task, not just the model.

It is. The correctness-optimal k is **task- and model-specific and spans the whole ladder**: for a
fixed model it ranges from k = 1 to k = full across the five task types, and no single k is optimal
across tasks for any model (Table 4). A profile calibrated on factual recall does not transfer to
conflict-resolution even for a fixed model — the router primitive of §7–§8 must be measured per
task type, not once per model.

**Table 4. Task battery — correctness-optimal k per (model, task type).**
Each cell is the k that maximises correctness; the ladder is {0, 1, 2, 3, 5, 8, 13, full}.

| model (served backend) | factual | multi-hop | state | conflict | constraint |
|---|---|---|---|---|---|
| gpt-5.5-pro (OpenAI) | 8 | 3 | full | full | 13 |
| claude-opus-4.8 (Anthropic) | 13 | full | full | full | 13 |
| gpt-4o (OpenAI) | 5 | 13 | 5 | full | 5 |
| granite-4.1-8b (WandB) | full | full | full | 13 | 1 |
| granite-4.0-h-micro (Cloudflare) | full | 5 | 5 | full | 5 |
| qwen-2.5-7b (Together) | full | 8 | 5 | 2 | 3 |
| llama-3.2-3b (Cloudflare) | 2 | 2 | 5 | full | 8 |

`llama-3.1-8b` was in the slate but is excluded: it was rate-limited upstream (HTTP 429) on the
free tier across three separately pinned providers (DeepInfra, Novita, WandB), and the pin
correctly refused to reroute each time (§6), so no clean single-backend sweep was obtainable.
`gpt-5.5-pro` — a reasoning model billed at $180/M output — was run at reps = 1 with reasoning
effort *low* and a 320-token output cap to bound cost (≈ €8 for the row); its per-repetition
variance was 0.000 at every cell, so reps = 1 costs no resolution here.

Two structural notes carry over from §7.1. First, the **blind axis persists on the harder task
types**: at each model's correctness-optimal k, correctness is 1.000 while a contamination severity
of up to 0.084 (gpt-4o, conflict-resolution at k = full) sits underneath it, unflagged — the
correctness-only profile is exactly as blind here as on the original family. The one exception is
the strongest model: `gpt-5.5-pro` holds contamination at ≈ 0.000 at every task's correctness-optimal
k, so its blind spot is small across task types too — consistent with the capability-gated threshold
of §7.4. Second, the
**non-monotone / distraction effect (§7.3) is capability-gated**: the 3 B model (`llama-3.2-3b`)
does *not* reach correct = 1.000 at high k on the multi-hop and state tasks (it falls to 0.50–0.75
by k = 8–13), i.e. more evidence actively *hurts* the small model on the harder tasks, while the
larger models stay saturated. This is a probe, not a full battery (six cases per task type,
single-token answers, one contamination register), but it converts §9's "task variance is untested"
into a first measured result: k\* is task-specific in practice, and a one-profile-per-model
assumption is unsafe.

## 8. Discussion

The practical consequence is sharp. A `k_profile` built from correctness **only** is
actively unsafe for strong models: it tells a router "k high is fine" while the
contamination axis already recommends a small k. The hidden harm is largest for a *weaker
model fed more context* — precisely the cost-driven "small model + lots of retrieval"
configuration — and correctness-only calibration is blind to exactly that regime. We
therefore recommend that (i) inference-time routers inject top-k\* rather than "much
context"; (ii) any published k-profile carry a contamination/drift axis, not correctness
alone; and (iii) robustness stress-tests vary register and turn-structure, not state
density, since density is non-monotone. These recommendations apply per calibrated
`(model, task, axis)` point — the primitive is already consumed by a router (DESi uses
k-calibrated state slices as a control primitive, §8) — and a first task battery (§7.5) shows the
correctness-optimal k does *not* transfer across task types, so a profile must be calibrated per
task type rather than once per model.

## 9. Limitations and future work

This is a pilot and should be read as a proof-of-method, not a performance estimate.
(1) **Small N**: 3–8 synthetic cases per task; magnitudes are small and confidence
intervals are not estimated. (2) **Heuristic metrics**: the contamination axis is a closed
lexical marker set; it detects surface adoption, not meaning, and can mis-score paraphrase.
(3) **Provider routing is uncontrolled — the most important caveat here.** Our runs neither
pinned the gateway's upstream provider nor recorded the served backend, so the provider — and
hence the quantization — could vary within a single k-sweep and between repetitions. A measured
k\* difference can therefore carry routing noise, and k\* is *not* shown to be provider-invariant;
it should be read as a property of `(model, served backend, task, axis)`. Controlled calibration
requires pinning the provider (order + no fallbacks) and logging the served backend per call, and
treating `(model, served backend)` as the unit. Where the served instance cannot simply be
trusted, cryptographic attestation of the model, policy and runtime configuration (MIVP;
Rentschler, 2026a) can make silent model substitution, routing and quantization changes
detectable — a reproducibility prerequisite for attributing a k\* number to a known instance.
The reference implementation now supports provider pinning (order + no fallbacks), per-call
served-backend logging, and retry that never re-routes, so a pinned sweep stays on one backend. (4) **One excluded model** (gemini, extraction artefact). (5) **Cross-corpus
comparison**: the esoteric-vs-credible contrast in §7.2 mixes corpora; a matched within-corpus
esoteric density sweep on the same harness is the clean next experiment. (6) **Scale of k**: to
locate saturation for the largest models one needs cases with tens of genuinely decision-relevant
fragments; the present datasets top out near k = 13, so "no decline on correctness" is established
only in that range. (7) **Cross-domain calibration is an open field.** We provide a four-domain
probe (technical / medical / legal / finance) so k\* *can* be measured per application, but we
deliberately do **not** report per-domain k\* here: such numbers would inherit the provider
confound of (3), and domains with interdependent evidence — e.g. relational diagnostic graphs —
may not even satisfy the "independent top-k fragment" assumption of §3, so a single k-profile
should not be assumed to transfer across domains. As a first signal, a rough probe on two small
models (`granite-4.0-h-micro`, `qwen-2.5-7b`) over four domains — technical / medical / legal /
finance, identical adversarial structure — returned k\* = 1 in *every* domain, with correctness
saturating at k = 1: no domain-driven shift here, but the cases were easy enough that this is a
floor, not a stress test. The runs were initially provider-unpinned (per (3); two further
small-model runs even aborted on transient provider errors); a **provider-pinned** re-run
(qwen-2.5-7b served by Together at fp8, verified via served-backend logging, with
retry-without-reroute) reproduced k\* = 1 in every domain — so here the result was
provider-invariant, and the tool now carries the pinning needed to keep it that way. Clean
per-domain,
per-backend calibration over an *attested* instance is the deployment-time measurement this leaves
open. (8) **Task variance is now probed, not just flagged.** The original single-family caveat is
partly retired: §7.5 reports a first dual-instrumented task battery (five task types, the same
contamination surface, seven provider-pinned models) in which the correctness-optimal k spans k = 1
to k = full across task types for a fixed model, and the blind axis persists on the harder types.
This establishes that k\* is task-specific in practice and that a single k-profile should not be
assumed to transfer across task types — but it remains a probe (six single-token cases per task
type, one register, one model dropped to upstream rate-limiting), not a full battery with
confidence intervals. The natural extensions are a high-fragment dataset (k up to ~89), more cases
per task type with bootstrap confidence intervals, and the same battery over an *attested* instance
(item 3).

## 10. Conclusion

For a fixed model and task there is an evidence-saturation point k\*, and whether "more
context" hurts at all is a property of the metric axis, not just the model. A correctness
axis is blind to the epistemic-contamination axis that actually saturates for strong models;
contamination is capability-associated but not size-determined; a credible adversarial register
does not overturn strong-model resistance; and state density is the wrong knob for crossing
it. k\* is a calibration primitive: measured per model and task on the axis that can see
the damage, it lets routers and governance layers inject the right amount of context instead
of the most.

---

## References

- Asai, A., Wu, Z., Wang, Y., et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and
  Critique through Self-Reflection.* arXiv:2310.11511.
- Figueiredo, V., & Franceschi, W. (2026). *Inference-Time Control is the Missing Layer in LLM
  Reliability.* Position paper / preprint.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems 33.
- Liu, N. F., Lin, K., Hewitt, J., et al. (2023). *Lost in the Middle: How Language Models
  Use Long Contexts.* arXiv:2307.03172 (Transactions of the ACL, 2024).
- Mialon, G., Dessì, R., Lomeli, M., et al. (2023). *Augmented Language Models: a Survey.*
  Transactions on Machine Learning Research; arXiv:2302.07842.
- Shi, F., Chen, X., Misra, K., et al. (2023). *Large Language Models Can Be Easily
  Distracted by Irrelevant Context.* International Conference on Machine Learning (ICML).
- Zhao, Z., Wallace, E., Feng, S., et al. (2021). *Calibrate Before Use: Improving Few-Shot
  Performance of Language Models.* International Conference on Machine Learning (ICML).
- Rentschler, H.-S. (2026a). *Model Identity Verification Protocol (MIVP): A Cryptographic
  Attestation Standard for AI Systems.* Technical Specification (Draft), v2.1. SSRN Working
  Paper. https://dx.doi.org/10.2139/ssrn.6243978
- Rentschler, H.-S. (2026b). *Evidence-k: a benchmark for the evidence-saturation point k\*.*
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
