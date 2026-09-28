# reinforce

強化学習(RL)の調査・実験リポジトリ。アルゴリズムを小規模環境で再現し、サンプル効率・学習時間・seed 間の安定性を計測して `docs/experiment/` に記録する。

構成・規約は [`.claude/CLAUDE.md`](.claude/CLAUDE.md) を参照。

## Setup

```bash
dev/setup            # uv sync (JAX CPU)
dev/setup --nn       # + torch
dev/setup --gpu      # + RunPod / Vast.ai / Kaggle SDK
dev/lint             # ruff + mypy
dev/test             # pytest
```

## Run an experiment

```bash
dev/train --case ppo/case1 --smoke          # local smoke (CPU, seconds)
dev/train --case ppo/case1                  # full run
dev/evaluate --case ppo/case1 --run <run_id>
dev/runpod train <commit-sha> --case ppo_case1_smoke   # on a pod
```

## Layout

```
backend/   src/ (simulate, gpu) + pipeline/<family>/case<N>/ + tests/
data/      lake / processed / mart / output (DVC)
dev/       command wrappers
docs/      develop/ experiment/ plans/
infra/     Terraform + GPU runtime image
```
