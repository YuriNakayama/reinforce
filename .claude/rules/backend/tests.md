---
paths:
  - "backend/tests/**"
---

# Backend Test Rules (`backend/tests/**`)

Pytest conventions for the backend test suite. General Python rules (type hints, logging, naming) live in `.claude/rules/python.md`.

## Frameworks

- **Unit/Integration**: Pytest + pytest-asyncio
- Tests mirror `src/` structure in `tests/`

## Test Guidelines

- Write in AAA pattern (Arrange, Act, Assert)
- Use Fixtures for common setup
- Minimize use of mock and patch — keep close to actual behavior
- Each test should be executable independently
- Tests live under `backend/tests/`, mirroring the `backend/src/` layout

## Test-Driven Development

1. Write test first (RED) — test should FAIL
2. Write minimal implementation (GREEN) — test should PASS
3. Refactor (IMPROVE)
4. Verify coverage (80%+)

## Running tests

```bash
# All tests with coverage
dev/test

# Specific module
uv run --directory backend pytest tests/<path> -x --no-header -q
```

## Test Classification Layout

When reorganizing `backend/tests`, classify tests by execution scope first, then keep only `src` or `pipeline` directly under each class directory:

```text
backend/tests/
  unit/
    src/
    pipeline/
  integration/
    src/
    pipeline/
  e2e/
    pipeline/
```

Prefer placing as many tests as practical under `unit`.

- **unit**: ordinary unit tests. Use this for pure logic and small APIs, including geometry, physics, decoder, featurizer, model forward, metrics, schema/dataclass conversion, parser/validation, small `tmp_path` IO checks, small mocks, dataset `__len__`/`__getitem__`, and DataLoader shape checks.
- **integration**: smoke tests that treat internal logic as a black box. Use this for training smoke tests, CLI happy paths, packaging/archive smoke tests, external-service mock flows, launch/watch flows, and other "does this workflow minimally run" checks.
- **e2e**: full training-loop or real-environment execution. Any test that creates a Gymnasium / JAX environment and steps it through whole episodes, runs `train.py` for more than a smoke update, or evaluates a checkpoint against a fixed seed set belongs under `e2e/pipeline`.

Additional placement rules:

- Tests that run a pipeline case through real episodes belong under `e2e/pipeline/...`.
- Prefer `conftest.py` for shared fixtures and pytest-only setup. Use a `utils` package only for tiny importable helpers that cannot be represented as fixtures.
- `utils` packages must not contain `test_*.py` files or `test_*` test functions. Test bodies must live outside `utils`.
- Tests may depend on `tests/pipeline/utils`-style helpers only when the helper is tiny (for example short assertion helpers). If helper logic becomes worth testing, move it into a `backend/src` module.
- Avoid importing test implementation from another test module. Shared evaluation and aggregation logic belongs in `backend/src/simulate`.
- Pytest scope markers are assigned from the top-level directory by `backend/tests/conftest.py` only for `integration` and `e2e`; `unit` tests intentionally have no scope marker. Keep the directory classification as the source of truth and reserve per-test markers for orthogonal concerns such as `slow` or `timeout`.

