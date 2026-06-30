"""Typed configuration for an Evidence-k run.

The YAML config is parsed into frozen dataclasses with explicit validation. Anything
unexpected raises ``ConfigError`` — there are no silent defaults that hide a typo.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# A k-value is either a non-negative integer or the sentinel string "full".
KValue = int | str
FULL = "full"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    name: str
    temperature: float = 0.0
    max_tokens: int = 512
    # Optional knobs for OpenAI-compatible providers. Keys/secrets are NEVER stored here;
    # they are read from environment variables by the adapter.
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    k_values: tuple[KValue, ...]
    repetitions: int = 1
    random_seed: int = 0


@dataclass(frozen=True)
class TaskConfig:
    name: str
    dataset: str


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, float]

    DEFAULT_WEIGHTS = {
        "correctness": 0.35,
        "grounding": 0.20,
        "constraint_adherence": 0.20,
        "state_consistency": 0.15,
        "hallucination_penalty": 0.10,
        "cost_penalty": 0.00,
    }

    # Dimensions that contribute positively vs. those subtracted as penalties.
    POSITIVE = ("correctness", "grounding", "constraint_adherence", "state_consistency")
    PENALTY = ("hallucination_penalty", "cost_penalty")


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    benchmark: BenchmarkConfig
    tasks: tuple[TaskConfig, ...]
    scoring: ScoringConfig
    source_path: str | None = None


_VALID_PROVIDERS = {"mock", "openai_compatible", "openai"}


def _normalise_k_values(raw: Any) -> tuple[KValue, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ConfigError("benchmark.k_values must be a non-empty list")
    out: list[KValue] = []
    seen: set[KValue] = set()
    for item in raw:
        if isinstance(item, bool):  # bool is a subclass of int; reject it explicitly
            raise ConfigError(f"benchmark.k_values has an invalid entry: {item!r}")
        if isinstance(item, int):
            if item < 0:
                raise ConfigError(f"benchmark.k_values has a negative entry: {item}")
            value: KValue = item
        elif isinstance(item, str) and item.strip().lower() == FULL:
            value = FULL
        else:
            raise ConfigError(
                f"benchmark.k_values entries must be non-negative ints or 'full', got {item!r}"
            )
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return d[key]


def parse_config(data: dict[str, Any], source_path: str | None = None) -> Config:
    """Validate a parsed YAML/JSON mapping into a :class:`Config`."""
    if not isinstance(data, dict):
        raise ConfigError("top-level config must be a mapping")

    model_raw = _require(data, "model", "config")
    benchmark_raw = _require(data, "benchmark", "config")
    tasks_raw = _require(data, "tasks", "config")
    scoring_raw = data.get("scoring", {}) or {}

    provider = str(_require(model_raw, "provider", "model"))
    if provider not in _VALID_PROVIDERS:
        raise ConfigError(
            f"unknown model.provider {provider!r}; expected one of {sorted(_VALID_PROVIDERS)}"
        )

    model = ModelConfig(
        provider=provider,
        name=str(_require(model_raw, "name", "model")),
        temperature=float(model_raw.get("temperature", 0.0)),
        max_tokens=int(model_raw.get("max_tokens", 512)),
        base_url=model_raw.get("base_url"),
        api_key_env=model_raw.get("api_key_env"),
        extra={
            k: v
            for k, v in model_raw.items()
            if k
            not in {"provider", "name", "temperature", "max_tokens", "base_url", "api_key_env"}
        },
    )

    repetitions = int(benchmark_raw.get("repetitions", 1))
    if repetitions < 1:
        raise ConfigError("benchmark.repetitions must be >= 1")
    benchmark = BenchmarkConfig(
        k_values=_normalise_k_values(_require(benchmark_raw, "k_values", "benchmark")),
        repetitions=repetitions,
        random_seed=int(benchmark_raw.get("random_seed", 0)),
    )

    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ConfigError("tasks must be a non-empty list")
    tasks: list[TaskConfig] = []
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            raise ConfigError(f"tasks[{i}] must be a mapping")
        tasks.append(
            TaskConfig(
                name=str(_require(t, "name", f"tasks[{i}]")),
                dataset=str(_require(t, "dataset", f"tasks[{i}]")),
            )
        )

    weights = dict(ScoringConfig.DEFAULT_WEIGHTS)
    user_weights = scoring_raw.get("weights", {}) or {}
    if not isinstance(user_weights, dict):
        raise ConfigError("scoring.weights must be a mapping")
    for key, val in user_weights.items():
        if key not in weights:
            raise ConfigError(
                f"unknown scoring weight {key!r}; valid keys: {sorted(weights)}"
            )
        weights[key] = float(val)
    scoring = ScoringConfig(weights=weights)

    return Config(
        model=model,
        benchmark=benchmark,
        tasks=tuple(tasks),
        scoring=scoring,
        source_path=source_path,
    )


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return parse_config(data, source_path=str(p))


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Render a :class:`Config` back to a plain, round-trippable dict."""
    model: dict[str, Any] = {
        "provider": cfg.model.provider,
        "name": cfg.model.name,
        "temperature": cfg.model.temperature,
        "max_tokens": cfg.model.max_tokens,
    }
    if cfg.model.base_url is not None:
        model["base_url"] = cfg.model.base_url
    if cfg.model.api_key_env is not None:
        model["api_key_env"] = cfg.model.api_key_env
    model.update(cfg.model.extra)

    return {
        "model": model,
        "benchmark": {
            "k_values": list(cfg.benchmark.k_values),
            "repetitions": cfg.benchmark.repetitions,
            "random_seed": cfg.benchmark.random_seed,
        },
        "tasks": [{"name": t.name, "dataset": t.dataset} for t in cfg.tasks],
        "scoring": {"weights": dict(cfg.scoring.weights)},
    }


def dump_resolved_yaml(cfg: Config) -> str:
    """Serialise the resolved config to YAML (used for ``config.resolved.yaml``)."""
    buf = io.StringIO()
    yaml.safe_dump(config_to_dict(cfg), buf, sort_keys=False, allow_unicode=True)
    return buf.getvalue()
