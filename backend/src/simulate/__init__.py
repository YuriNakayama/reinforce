"""Environment tooling.

``simulate.jaxenv`` holds batched JAX environments: every rule is a masked
array update so one ``jit``ted step drives ``B`` environments at once, and
``benchmark.py`` measures env-steps/s per backend and batch size.

The current ``jaxenv`` implementation is the farming-sim port carried over
from the kaggriculture project. It is kept as a **worked template** for the
state-layout / masked-update / benchmark pattern, not as an environment of
interest in itself — see ``jaxenv/README.md`` before building a new one.
"""

from __future__ import annotations

__all__: list[str] = []
