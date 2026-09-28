# pipeline — experiment cases

```
pipeline/<family>/case<N>/
  README.md        hypothesis, what changed vs the previous case, result link
  config.yaml      all hyper-parameters, env id, seed list
  env.py           make_env()
  policy.py        networks
  train.py         python -m pipeline.<family>.case<N>.train [--config PATH] [--smoke]
  evaluate.py      python -m pipeline.<family>.case<N>.evaluate --run <run_id>
```

Family = algorithm family (`ppo`, `dqn`, `sac`, …). Cases are append-only. The
run-directory contract every `train.py` must honour is in
`.claude/rules/backend/pipeline.md`.

## Status table

| Family | Case | Env | Framework | Role / main change |
|---|---|---|---|---|
| (none yet) | | | | |
