# Python → JAX Conversion Patterns

Concrete before/after translations, grouped by the construct you're replacing.
Examples use generic RL structures. The in-repo reference for env-level
structure is `backend/src/simulate/jaxenv/` (`state.py`, `env.py`, `town.py`).
These are distillations — read the source you are porting for the full story.

## Table of contents

1. [Dynamic length → fixed shape + mask](#1-dynamic-length--fixed-shape--mask)
2. [Python loops → lax.scan / vmap](#2-python-loops--laxscan--vmap)
3. [Conditionals → lax.cond / jnp.where](#3-conditionals--laxcond--jnpwhere)
4. [Mutation → immutable updates](#4-mutation--immutable-updates)
5. [PyTorch nn.Module → equinox.Module](#5-pytorch-nnmodule--equinoxmodule)
6. [PyTorch optimizer/loss → Optax](#6-pytorch-optimizerloss--optax)
7. [Weight loading by name](#7-weight-loading-by-name)
8. [Avoiding recompilation (one trace)](#8-avoiding-recompilation-one-trace)
9. [Host-side vs device-side split](#9-host-side-vs-device-side-split)
10. [Randomness: PRNGKey threading](#10-randomness-prngkey-threading)
11. [State containers: pytrees](#11-state-containers-pytrees)
12. [The hard hot-path constructs (early-exit loops, variable-length sets, ragged data)](#12-the-hard-hot-path-constructs)

---

## 1. Dynamic length → fixed shape + mask

JAX compiles for a fixed shape. Anything whose size depends on data (a list that
grows, `arr[mask]` boolean indexing, a variable loop count) cannot be traced.
The fix: pad to a `MAX_*` constant, carry a boolean validity mask, compute over
the full array, and neutralize padding with `jnp.where`.

```python
# Python — variable number of entities, filtered sum
def team_health(entities, team):
    return sum(e.health for e in entities if e.team == team)
```
```python
# JAX — fixed MAX_ENTITIES, masked reduce
def team_health(state: EnvState, team: int) -> jax.Array:
    is_team = state.entity_valid & (state.entity_team == team)
    return jnp.sum(jnp.where(is_team, state.entity_health, 0.0))
```

Key idea: `entity_valid` distinguishes real slots from padding; you always
operate on the full `(MAX_ENTITIES,)` array (or `(B, MAX_ENTITIES)` batched).
`backend/src/simulate/jaxenv/state.py` uses the same idea for its fixed board.

## 2. Python loops → lax.scan / vmap

A sequential loop with a carried state → `lax.scan`. Independent parallel
repeats → `vmap`.

```python
# Python — sequential rollout
state = reset(seed)
transitions = []
for t in range(horizon):
    action = policy(state)
    state, reward = env.step(state, action)
    transitions.append((action, reward))
```
```python
# JAX — lax.scan: carry threads state, per-step outputs stack into (T, ...)
def step_fn(carry, _t):
    state, key = carry
    key, k = jax.random.split(key)
    action = policy(state, k)
    new_state, reward = env_step(state, action)
    return (new_state, key), (action, reward)

(final_state, _), (actions_t, rewards_t) = jax.lax.scan(
    step_fn, (init_state, key), jnp.arange(horizon)
)
```

```python
# N parallel episodes — vmap over the per-episode args, broadcast the model.
vmapped = jax.vmap(_rollout_one_env, in_axes=(None, 0, 0, None))
#                                              ^model ^key ^state ^scalar
batch = vmapped(model, keys, batched_init_states, horizon)
```

`scan` requires the carry pytree to keep identical shape/dtype every iteration —
this is *why* env state must be fixed-shape (pattern 1). To "freeze" a finished
episode while peers keep running, don't break — gate with a `done` flag:
`state_next = jax.tree.map(lambda new, old: jnp.where(done, old, new), new, old)`.
For training rollouts the usual alternative is **auto-reset**: when `done`,
replace the state with a freshly reset one via the same `jnp.where` pattern, so
every env keeps producing transitions.

Nested loops (PPO epochs × minibatches) → nested `scan`: scan epochs over
scanned minibatches, and implement `target_kl` early-stop as a bool flag in the
carry (no data-dependent break).

Backward recursions (GAE, discounted returns) → `scan(..., reverse=True)`:

```python
def gae_step(last_adv, x):
    reward, value, next_value, done = x
    nonterminal = 1.0 - done
    delta = reward + gamma * next_value * nonterminal - value
    adv = delta + gamma * lam * nonterminal * last_adv
    return adv, adv

_, advantages = jax.lax.scan(
    gae_step, jnp.zeros_like(rewards[0]),
    (rewards, values[:-1], values[1:], dones), reverse=True,
)  # (T, B)
```

## 3. Conditionals → lax.cond / jnp.where

`if` on a *static* Python value is fine and stays in Python. `if` on a *traced*
value must become `lax.cond` (when the branches are expensive / have side
structure) or `jnp.where` (cheap, elementwise — usually preferred).

```python
if cnt > 0: adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # Python (cnt static? keep it)
```
```python
# traced predicate, two whole-array branches → lax.cond
bonus = jax.lax.cond(use_intrinsic, lambda: rnd_bonus(obs), lambda: jnp.zeros(obs.shape[:-1]))
# elementwise select → jnp.where
reward = jnp.where(done_already, 0.0, shaping + terminal)
```

Prefer `jnp.where` inside `vmap`: under `vmap`, `lax.cond` with a batched
predicate executes both branches anyway.

## 4. Mutation → immutable updates

JAX arrays are immutable. In-place ops become functional updates.

| Python | JAX |
|---|---|
| `arr[i] = v` | `arr = arr.at[i].set(v)` |
| `arr[i] += v` | `arr = arr.at[i].add(v)` |
| `state.x = v` (dataclass) | `state = state._replace(x=v)` (NamedTuple) or `eqx.tree_at` |
| `list.append(x)` in loop | collect via `scan` stacked outputs |
| `dict[k] = v` accumulator | return an aux dict from the scan body |

Never write `NEVER mutate` violations — this repo's Python rules already ban
mutation, and JAX makes it a hard error.

## 5. PyTorch nn.Module → equinox.Module

```python
# PyTorch
class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.value_head = nn.Linear(cfg.hidden, 1)
    def forward(self, x):
        return self.value_head(x)
```
```python
# Equinox — fields are the pytree leaves; __call__ is the forward
class ActorCriticJax(eqx.Module):
    value_w: jax.Array
    value_b: jax.Array
    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.value_w.T + self.value_b
    @classmethod
    def from_init(cls, key: jax.Array) -> "ActorCriticJax":
        ...  # initialize leaves with jax.random
```

Because the module is a registered pytree, `model(batch)` is jit/vmap-able and
`eqx.filter_value_and_grad(loss_fn)(model, ...)` differentiates w.r.t. the array
leaves only (filtering out static config). Output containers are also
`eqx.Module`s or `NamedTuple`s. Mirror the PyTorch architecture *exactly* —
same layer order, dims, presence/absence of bias — or weight loading
(pattern 7) silently mismatches.

## 6. PyTorch optimizer/loss → Optax

```python
# PyTorch
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
opt.step(); opt.zero_grad()
```
```python
# Optax — clip then adamw, threaded functionally
def make_optimizer(cfg) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),   # == clip_grad_norm_
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
    )

grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
(loss, aux), grads = grad_fn(model, ...)
updates, opt_state = optimizer.update(grads, opt_state, model)
model = eqx.apply_updates(model, updates)
```

`opt_state` is threaded explicitly (no hidden mutable optimizer object). For an
lr schedule, pass an `optax.Schedule` as `learning_rate`. Loss math translates
directly: `nn.functional.log_softmax`→`jax.nn.log_softmax`,
`F.mse_loss(a,b)`→`jnp.mean((a-b)**2)`, `.mean()`→`jnp.mean(...)`,
`torch.clamp`→`jnp.clip`, `torch.exp`→`jnp.exp`.

## 7. Weight loading by name

A ported model is useless if it can't ingest the trained `.pt`. Read PyTorch
weights and copy leaf-by-leaf by parameter name, then **assert nothing was left
unmapped**.

```python
def load_torch_weights_jax(model, path):
    sd = torch.load(path, weights_only=True)          # never unpickle arbitrary code
    loaded, missing = 0, 0
    # map each torch key -> the corresponding eqx leaf, copy via numpy, count.
    ...
    return model, loaded, missing
# in the parity test:
model, loaded, missing = load_torch_weights_jax(model, weights_path)
assert missing == 0, f"{missing} torch keys unmapped"
```

`weights_only=True` is mandatory (no arbitrary code execution from a pickle).
Keep the loader next to the ported module and cover it in the parity test.

## 8. Avoiding recompilation (one trace)

jit recompiles for every distinct *static* argument and every new *shape*. A
Python `int seed` threaded into a jitted body recompiles per seed — the
earlier project this setup came from blew its compile cache past 3 GB and hit
SIGABRT exactly this way.

```python
# BAD — recompiles per seed value, one trace per episode
def rollout(model, seed: int, horizon): state = reset(seed); ...

# GOOD — build states on host, stack into a batched pytree, pass the array
init_states = [reset(seed=seed + i) for i in range(n)]          # host loop
batched = jax.tree.map(lambda *xs: jnp.stack(xs), *init_states)  # (n, ...)
vmapped = jax.vmap(rollout_one, in_axes=(None, 0, 0, None))      # one trace
```

Rules of thumb: keep varying scalars as traced arrays (or `jnp.int32(x)`), keep
truly-constant config as `static`/closure, and keep all batch elements the same
shape.

## 9. Host-side vs device-side split

Not everything should be jitted. Rejection sampling, RNG that must bit-match a
reference simulator, and variable-count generation belong on the **host** (plain
NumPy/Python), run once at `reset`, then `jnp.array`-ified into the state. The
per-step `step()` is the pure-jax, jitted hot path. Example: procedurally
generated levels whose generator uses rejection sampling are generated on the
host at reset (or pre-generated into a fixed pool) so `step` never samples.
State the split explicitly in the file docstring.

## 10. Randomness: PRNGKey threading

PyTorch/NumPy global RNG → explicit JAX keys. Split before every consuming call;
never reuse a key.

```python
key, k_sample = jax.random.split(key)   # thread `key` forward in the carry
action = sample_action_jax(output, mask, k_sample)
```
Per-episode keys come from `jax.random.split(key, n)`. A reused key gives
identical "random" draws — a subtle parity bug.

## 11. State containers: pytrees

Group related arrays into a `NamedTuple` (transitions, configs, rollout batches)
or a registered pytree class (`EnvState`). Pytrees flow through `scan`/`vmap`/
`tree.map` as one object. Use `_replace(field=...)` for functional updates and
`jax.tree.map(fn, a, b)` for elementwise ops across two same-structured trees
(e.g. the done-gating freeze in pattern 2). See
`backend/src/simulate/jaxenv/state.py::EnvState`; a rollout batch is typically
a `NamedTuple(obs, action, logp, reward, done, value)` of `(T, B, …)` arrays.

## 12. The hard hot-path constructs

These are the patterns that make env / rollout hot paths "hard to JAX" — and
exactly the ones worth porting, because a hot path is hard *because* it is
doing the per-step work. The framing that makes them tractable: don't port the
function to run once — port it so it can be **`vmap`'d over the whole batch**
(every env, every candidate) in one call.

### 12a. Loops with early exit → fixed-iteration scan + `done` freeze

Python: `for _ in range(max_iters):` that refines an estimate and `break`s once
it converges (e.g. a CEM planner, a line search, a physics sub-step loop that
stops on contact). You can't `break` in a traced loop. Use `lax.scan` over the
fixed maximum, carry a `done` flag, and **freeze updates once done**.

```python
def refine_step(carry, _):
    estimate, done = carry
    proposal = improve(estimate)
    converged = jnp.max(jnp.abs(proposal - estimate)) < EPS
    estimate = jnp.where(done, estimate, proposal)   # frozen once done
    done = done | converged
    return (estimate, done), None

(estimate, done), _ = jax.lax.scan(
    refine_step, (init, jnp.array(False)), None, length=MAX_ITERS
)
```

Use `lax.while_loop` only when you do not need reverse-mode gradients through
the loop and the iteration count really varies a lot; `scan` is differentiable
and has a predictable cost.

### 12b. Variable-size action / entity sets → padded arrays + masked argmax

Python: iterate over the currently legal actions and pick the best, or sample
among them. Replace the variable list with a fixed `(N_ACTIONS,)` logits vector
and a legality mask; masked-out entries get `-inf` before `argmax` /
`categorical`.

```python
def masked_logits(logits: jax.Array, legal: jax.Array) -> jax.Array:
    return jnp.where(legal, logits, -jnp.inf)

def select_action(key, logits, legal):
    ml = masked_logits(logits, legal)
    greedy = jnp.argmax(ml, axis=-1)
    sampled = jax.random.categorical(key, ml, axis=-1)
    any_legal = jnp.any(legal, axis=-1)            # all-illegal -> fallback
    return jnp.where(any_legal, sampled, NOOP_ACTION), greedy
```

Entropy and log-prob must also respect the mask
(`jnp.where(legal, p * logp, 0.0)`), or padded actions leak into the loss.

### 12c. Ragged data → resolve on host, compute in JAX

Offline datasets and replay buffers often hold episodes of different lengths.
Resolve the ragged structure once on the host into fixed arrays — pad each
episode to `MAX_T` and keep `lengths` — then do every numeric step in JAX with
a `t < length` mask.

```python
# host: list[np.ndarray] (variable T_i) -> (N, MAX_T, ...) + lengths
def pad_episodes(episodes, max_t):
    n = len(episodes)
    out = np.zeros((n, max_t, *episodes[0].shape[1:]), np.float32)
    lengths = np.array([min(len(e), max_t) for e in episodes], np.int32)
    for i, e in enumerate(episodes):
        out[i, : lengths[i]] = e[: lengths[i]]
    return out, lengths

# JAX: mask by length inside the jitted loss
valid = jnp.arange(MAX_T)[None, :] < lengths[:, None]      # (N, MAX_T)
loss = jnp.sum(jnp.where(valid, per_step_loss, 0.0)) / jnp.sum(valid)
```

When is host-only (no port at all) right? When the code is *not* on the hot
path — logging, checkpoint I/O, config parsing, RNG that must bit-match an
external simulator. The litmus test: would vectorising it speed up training?
If yes, host-resolve + JAX-compute it. If no, leave it on the host and say so.
