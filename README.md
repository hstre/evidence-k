# Evidence-k

**Measure how much evidence an LLM can use before context turns into noise.**

Evidence-k is a small, standalone Python tool that empirically finds, for a given model
and task, the number of evidence fragments (retrieval chunks, state slices, context
snippets) that maximises answer **reliability** — and the point beyond which *more context
makes things worse*.

It produces a portable `k_profile.json` that routers, RAG systems, memory layers and
governance / control architectures (such as DESi) can load to decide how many evidence
fragments to inject per call. **Evidence-k has no dependency on DESi or any other system** —
it only *emits* a profile they may consume.

---

## Why more context is not always better

Adding evidence helps — until it doesn't. Past a certain amount, extra fragments are mostly
distractors: they dilute the signal, pull the model toward confidently wrong but
*grounded* answers, increase output variance, and cost more tokens and latency for no gain.

There is usually a measurable saturation point. Evidence-k calls it **`k*`**:

```
k* = argmax_k  Reliability(model, task, evidence_k)
```

`Reliability` is a configurable weighted score over several dimensions (see below). The
curve typically rises, plateaus, and then declines as distractors accumulate — so the goal
is to find the peak, not to maximise context.

> **`k*` is model- and task-specific. It is not a universal constant.** A profile measured
> for one model/task/evidence-format does not transfer unchanged to another.

## What "evidence saturation `k*`" means

For each `k` in a configured sweep (`0, 1, 2, 3, 5, 8, 13, full`), Evidence-k feeds the
model the **first `k` evidence fragments** (evidence is ordered; `k=0` gets none, `full`
gets all) across a dataset of cases, repeats it for stability, and measures:

| Dimension | What it captures |
| --- | --- |
| `correctness` | task accuracy (normalized / exact match) |
| `grounding` | is the answer supported by the provided evidence? |
| `constraint_adherence` | were the stated output constraints obeyed? |
| `state_consistency` | did the answer track the current state (no stale regressions)? |
| `hallucination` (penalty) | content in neither the evidence nor the expected answer |
| `answer_variance` | disagreement across repetitions |
| `token cost` / `latency` | reported for every `k` |

`k*` is the `k` with the highest aggregate reliability.

## Install

```bash
pip install -e .
```

Python 3.11+. The only runtime dependency is **PyYAML**; the OpenAI-compatible adapter uses
the standard library (`urllib`), so the footprint stays tiny.

## Quickstart with the MockModel (offline, deterministic)

The bundled `mock` provider is a fully seeded simulator — no network, no API key — that
reproduces a realistic evidence-saturation curve so you can try the whole pipeline:

```bash
evidence-k init                                   # scaffold config + example datasets
evidence-k run --config configs/example.yaml      # run the k-sweep
evidence-k summarize --run-dir runs/<run_id>      # print the summary table
evidence-k export-profile --run-dir runs/<run_id> --out k_profile.json
```

Or just run `examples/run_mock_benchmark.sh`.

Typical output (peaks in the middle, then declines):

```
  factual_qa  → recommended k = 5 (reliability 0.861)
        k |  reliab | correct |  ground |  halluc |    var
        0 |   0.450 |   0.444 |   0.000 |   0.556 |  0.222
        1 |   0.719 |   0.722 |   0.722 |   0.278 |  0.167
        3 |   0.822 |   0.778 |   1.000 |   0.000 |  0.222
        5 |   0.861 |   0.889 |   1.000 |   0.000 |  0.111
       13 |   0.589 |   0.111 |   1.000 |   0.000 |  0.111
     full |   0.608 |   0.167 |   1.000 |   0.000 |  0.167
```

## Using a real OpenAI-compatible API

The `openai_compatible` provider talks to any OpenAI-style `/chat/completions` endpoint
(OpenAI, DeepSeek, OpenRouter, Together, local vLLM, …). **API keys are read from an
environment variable only — never hardcoded and never stored in the config.** If the key is
missing, the tool stops with a clear message.

```yaml
model:
  provider: openai_compatible
  name: gpt-4o-mini
  temperature: 0
  max_tokens: 512
  base_url: https://api.openai.com/v1   # optional; or set OPENAI_BASE_URL
  api_key_env: OPENAI_API_KEY           # which env var to read the key from
```

```bash
export OPENAI_API_KEY=sk-...
evidence-k run --config configs/my_model.yaml
```

For DeepSeek, for example, set `base_url: https://api.deepseek.com/v1`,
`api_key_env: DEEPSEEK_API_KEY` and `name: deepseek-chat`.

## Dataset format

One JSON object per line. Evidence is **ordered**; distractors are explicitly allowed so
you can see when extra context starts to hurt.

```json
{
  "id": "case_001",
  "task": "factual_qa",
  "question": "Which city is the Eiffel Tower in?",
  "evidence": [
    {"rank": 1, "text": "The Eiffel Tower is located in Paris, France.", "label": "supporting"},
    {"rank": 2, "text": "Berlin is the capital of Germany.", "label": "distractor"}
  ],
  "expected_answer": "Paris",
  "constraints": ["Answer with only the city name."]
}
```

- `k=0` → no evidence; `k=3` → the first three fragments by rank; `k=full` → all fragments.
- `label` is `supporting` or `distractor` (any non-`supporting` label is treated as a
  distractor). The mock simulator uses labels; real models never see them.
- Task-specific optional fields: `state_consistency` cases may add
  `"stale_values": ["..."]` — values the answer must not regress to.

Four task types ship with example datasets: `factual_qa`, `state_consistency`,
`conflict_resolution`, `constraint_following`.

## Scoring

Scoring for the example datasets is **LLM-free by default** — deterministic and cheap:
normalized/exact match, rule-based constraint checks, token-overlap grounding, a
hallucination proxy (answer content found in neither evidence nor the expected answer), and
state-consistency tracking. An LLM judge can be added later, but it is **not** a required
component.

Weights are configured in YAML and combined into a single reliability score, clamped to
`[0, 1]`:

```yaml
scoring:
  weights:
    correctness: 0.35
    grounding: 0.20
    constraint_adherence: 0.20
    state_consistency: 0.15
    hallucination_penalty: 0.10   # subtracted
    cost_penalty: 0.00            # subtracted (token cost), off by default
```

```
reliability = Σ wᵢ·dimensionᵢ  −  w_hallucination·halluc_rate  −  w_cost·normalized_cost
```

## Run outputs

Each run is written under `runs/<run_id>/`:

```
runs/<run_id>/
  config.resolved.yaml   # the exact validated config used
  raw_outputs.jsonl      # one row per (task, k, case, repetition) model call
  scores.jsonl           # per-sample dimension scores + reliability
  summary.json           # aggregated per-(task, k) metrics + recommendations
  summary.csv            # same, flat/tabular
  k_profile.json         # the portable profile (see below)
  README_run.md          # human-readable run report
```

## The `k_profile.json`

The deliverable other systems load. Self-describing and dependency-free to consume:

```json
{
  "profile_version": "0.1",
  "model": "mock-model",
  "provider": "mock",
  "created_at": "...",
  "tested_k_values": [0, 1, 2, 3, 5, 8, 13, "full"],
  "tasks": {
    "factual_qa": {
      "recommended_k": 3,
      "best_score": 0.91,
      "score_curve": {"0": 0.42, "1": 0.71, "2": 0.86, "3": 0.91, "5": 0.89, "8": 0.82, "13": 0.74, "full": 0.68}
    }
  },
  "global_recommendation": {
    "default_k": 3,
    "notes": "Use task-specific k where available."
  }
}
```

A consumer (e.g. a DESi router) reads `tasks[<task>].recommended_k` for a per-task budget,
falling back to `global_recommendation.default_k`. See
`examples/desi_k_profile_example.json` for a worked example.

## CLI reference

| Command | Purpose |
| --- | --- |
| `evidence-k init [--dir .] [--force]` | scaffold `configs/example.yaml` + example datasets |
| `evidence-k run --config <yaml> [--runs-dir runs] [--base-dir <path>]` | run the k-sweep |
| `evidence-k summarize --run-dir runs/<id>` | print a saved run's summary |
| `evidence-k export-profile --run-dir runs/<id> --out k_profile.json` | (re)export the profile |

## Development

```bash
pip install -e ".[dev]"
pytest          # test suite (offline, deterministic)
ruff check .    # lint
```

## Scope / non-goals

No web frontend, no dashboard, no DESi core code, and no mandatory LLM judge. Evidence-k is
a focused measurement tool: it estimates `k*` and writes a profile. Treat its scores as a
calibration aid, not ground truth — and remember that `k*` is model- and task-specific.

## License

MIT — see [LICENSE](LICENSE).
