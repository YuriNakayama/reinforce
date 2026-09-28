"""Training case registry and path helpers for GPU pod launches.

This is the single registry shared by every GPU provider CLI (RunPod, Vast.ai,
Kaggle Notebook). Each entry names one trainable case and tells the pod how to
run it: which module to invoke, which extra CLI args to pass, which optional
uv dependency groups it needs, and where the adopted best weights are copied
back to. Registry keys do not have to equal the on-disk directory — see
:func:`case_subdir` for how variants of a case share one output tree.

Add an entry here when a case becomes worth training on a GPU. ``case1`` of
every family is a CPU smoke baseline and is registered only so the pod path
has a cheap end-to-end target.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

#: Registry of GPU-trainable cases.
#:
#: Keys:
#:   family            - pipeline family directory ("ppo", "dqn", ...)
#:   train_module      - python -m target, run from backend/
#:   config_arg        - extra args appended to the train invocation
#:                       ('' / '--smoke' / '--config <path relative to backend/>')
#:   preprocess_cmd    - python -m target run once before training; empty means
#:                       nothing to preprocess (RL cases generate their own data
#:                       by interacting with the environment)
#:   uv_groups         - extra uv dependency groups for the pod, space separated
#:                       (e.g. "nn" for torch). The `cuda` group (JAX CUDA 12
#:                       plugin) is always installed on GPU pods.
#:   canonical_weights - repo path the adopted best weights are copied to by
#:                       `promote`
CASE_DEFAULTS: dict[str, dict[str, str]] = {
    # ppo/case1 is the plumbing smoke test (CartPole, a few updates). It runs
    # in seconds on a laptop CPU; registered so `dev/runpod train --case
    # ppo_case1_smoke` can prove the pod path without a real GPU workload.
    "ppo_case1_smoke": {
        "family": "ppo",
        "train_module": "pipeline.ppo.case1.train",
        "config_arg": "--smoke",
        "preprocess_cmd": "",
        "uv_groups": "",
        "canonical_weights": "backend/pipeline/ppo/case1/weights/best.eqx",
    },
}

DEFAULT_CASE = "ppo_case1_smoke"

_CONFIG_FLAG = re.compile(r"--config\s+(\S+)")


def case_subdir(case: str) -> str:
    """Map a registry key to its on-disk case subdirectory.

    Keys are ``<family>_case<N>[_<variant>]``; the family prefix and any
    variant suffix are stripped so all variants of one case share a single
    output tree (``data/output/models/<family>/case<N>/runs/``).
    """
    parts = case.split("_")
    for part in parts:
        if part.startswith("case") and part[4:].isdigit():
            return part
    return case


def case_family(case: str) -> str:
    """Return the top-level pipeline family directory for a registry key."""
    if case in CASE_DEFAULTS:
        return CASE_DEFAULTS[case]["family"]
    return case.split("_", 1)[0]


def runs_root_for(case: str) -> Path:
    return Path(f"data/output/models/{case_family(case)}/{case_subdir(case)}/runs")


def case_defaults(case: str) -> dict[str, str]:
    if case not in CASE_DEFAULTS:
        raise typer.BadParameter(
            f"unknown case={case!r}; supported: {sorted(CASE_DEFAULTS)}"
        )
    return CASE_DEFAULTS[case]


def case_config_path(case: str) -> Path:
    """Repo-relative path of the config file a case trains with.

    An explicit ``--config <path>`` in ``config_arg`` wins (that path is
    relative to ``backend/``); otherwise the case directory's
    ``config.yaml`` is used (run-dir contract, .claude/rules/backend/
    pipeline.md). The file is not required to exist.
    """
    config_arg = CASE_DEFAULTS.get(case, {}).get("config_arg", "")
    match = _CONFIG_FLAG.search(config_arg)
    if match:
        return Path("backend") / match.group(1)
    return (
        Path("backend/pipeline") / case_family(case) / case_subdir(case) / "config.yaml"
    )


def find_best_weights(run_dir: Path, prefer_suffix: str = "") -> Path | None:
    """Return the ``best.*`` file of a run directory, or None.

    The run-dir contract allows any framework-specific suffix (``best.eqx``,
    ``best.pt``, ...). When several exist, the one matching ``prefer_suffix``
    (typically the canonical weights' suffix) wins, then the first by name.
    """
    candidates = sorted(p for p in run_dir.glob("best.*") if p.is_file())
    if not candidates:
        return None
    for path in candidates:
        if prefer_suffix and path.suffix == prefer_suffix:
            return path
    return candidates[0]


__all__ = [
    "CASE_DEFAULTS",
    "DEFAULT_CASE",
    "case_config_path",
    "case_defaults",
    "case_family",
    "case_subdir",
    "find_best_weights",
    "runs_root_for",
]
