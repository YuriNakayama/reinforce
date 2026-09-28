mark "20_clone_done"
# uv は冒頭で前倒し install 済みのはず。万一 PATH が落ちていれば再 export。
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[onstart] step=install_uv (early install missed; fallback)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mark "30_before_uv_sync"
echo "[onstart] step=uv_sync"
# 2026-05-20: venv on /persist (MFS network volume) repeatedly produced
# ImportError (libgomp .so unloadable; packages missing despite the install
# log showing them installed). Build venv on container-local disk instead.
# Cost: ~30-45s `uv sync` per pod (vs ~5s SKIP), but eliminates the dominant
# failure class.
#
# Remove any stale /persist symlink left by 30_persist_and_clone.sh so uv
# creates a real container-local directory.
if [ -L backend/.venv ]; then
  echo "[onstart] removing stale /persist symlink backend/.venv -> $(readlink backend/.venv)"
  rm -f backend/.venv
fi

# Dependency groups (see backend/pyproject.toml):
# - `cuda` is always installed on a GPU pod: JAX is the default framework and
#   without the CUDA 12 plugin jaxlib silently falls back to CPU (10x+ slower,
#   long enough for the stall watcher to kill the pod).
# - <UV_GROUPS> comes from the case registry (gpu/runpod/config/cases.py
#   `uv_groups`), e.g. "nn" for torch-based cases.
# Everything is installed in ONE `uv sync`: sync is declarative, so a later
# `uv sync --group X` would prune packages that an earlier call installed.
UV_GROUP_ARGS="--group cuda"
for g in <UV_GROUPS>; do
  if [ "${g}" != "cuda" ]; then
    UV_GROUP_ARGS="${UV_GROUP_ARGS} --group ${g}"
  fi
done
UV_SYNC_ARGS="--frozen --no-dev ${UV_GROUP_ARGS} --directory backend"
echo "[onstart] uv sync args: ${UV_SYNC_ARGS}"
UV_SYNC_LOG="/tmp/uv_sync.log"
UV_SYNC_START=$(date +%s)
for attempt in 1 2 3; do
  if uv sync ${UV_SYNC_ARGS} 2>&1 | tee "${UV_SYNC_LOG}"; then
    break
  fi
  echo "[onstart] uv_sync attempt=${attempt} failed; retrying in 30s"
  sleep 30
  if [ "${attempt}" -eq 3 ]; then
    echo "[onstart] uv_sync exhausted retries"
    if command -v aws >/dev/null 2>&1; then
      aws s3 cp "${UV_SYNC_LOG}" "${S3_ARTIFACT_PREFIX}/uv_sync.log" 2>&1 | head -3 || true
    fi
    mark "35_uv_sync_failed"
    exit 1
  fi
done
UV_SYNC_ELAPSED=$(( $(date +%s) - UV_SYNC_START ))
echo "[onstart] uv sync elapsed=${UV_SYNC_ELAPSED}s (container-local)"

# Sanity check: site-packages actually usable. Catches partial installs or
# future MFS-style breakage early so later steps don't fail obscurely.
IMPORT_SMOKE="import numpy, jax, equinox, optax, gymnasium, dvc"
case " <UV_GROUPS> " in
  *" nn "*) IMPORT_SMOKE="${IMPORT_SMOKE}, torch" ;;
esac
if ! backend/.venv/bin/python -c "${IMPORT_SMOKE}" 2>&1; then
  echo "[onstart] FATAL: venv import smoke failed post uv sync (${IMPORT_SMOKE})" >&2
  mark "35_uv_import_smoke_failed"
  exit 1
fi

# JAX device probe (informational). A CPU-only result on a GPU pod means the
# cuda group did not load — the run will still start but be very slow.
JAX_PROBE=$(backend/.venv/bin/python -c "
import json, jax
devs = jax.devices()
print(json.dumps({'backend': jax.default_backend(), 'devices': [str(d) for d in devs]}))
" 2>&1 | tail -1)
echo "[onstart] jax probe: ${JAX_PROBE}"
if ! echo "${JAX_PROBE}" | grep -q '"backend": "gpu"'; then
  echo "[onstart] WARNING: JAX is not using the GPU (see probe above)" >&2
  mark "36_jax_cpu_fallback"
fi

# torch GPU smoke + cu118 fallback, only for cases that install the `nn`
# group. 2026-05-22 observation: the torch wheel resolved by uv can target
# CUDA 13.x, which the RunPod cu1241 image driver (12.4) cannot run, so torch
# silently falls back to CPU. On failure, reinstall torch from the cu118 index
# (covers sm_50-sm_90 and runs on the 12.4 driver).
case " <UV_GROUPS> " in
  *" nn "*)
    CUDA_SMOKE_OUT=$(backend/.venv/bin/python -c "
import json, torch
out = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
       'available': torch.cuda.is_available(), 'smoke_ok': False, 'err': None}
if torch.cuda.is_available():
    try:
        (torch.zeros(2, 2).cuda() + 1).sum().item()
        out['smoke_ok'] = True
    except Exception as e:
        out['err'] = repr(e)
print(json.dumps(out))
" 2>&1 | tail -1)
    echo "[onstart] torch cuda probe: ${CUDA_SMOKE_OUT}"
    if ! echo "${CUDA_SMOKE_OUT}" | grep -q '"smoke_ok": true'; then
      echo "[onstart] torch CUDA smoke failed; force-reinstalling torch with cu118 wheel"
      # uv-managed venvs have no pip binary; use `uv pip install --python`.
      uv pip install --python backend/.venv/bin/python --reinstall-package torch \
        --index-url https://download.pytorch.org/whl/cu118 \
        torch || {
          echo "[onstart] FATAL: torch cu118 reinstall failed" >&2
          exit 1
      }
      CUDA_RECHECK=$(backend/.venv/bin/python -c "
import json, torch
out = {'available': torch.cuda.is_available(), 'smoke_ok': False, 'err': None}
if torch.cuda.is_available():
    try:
        (torch.zeros(2, 2).cuda() + 1).sum().item()
        out['smoke_ok'] = True
    except Exception as e:
        out['err'] = repr(e)
print(json.dumps(out))
" 2>&1 | tail -1)
      echo "[onstart] torch cuda recheck: ${CUDA_RECHECK}"
      if ! echo "${CUDA_RECHECK}" | grep -q '"smoke_ok": true'; then
        echo "[onstart] FATAL: GPU unusable for torch after cu118 reinstall" >&2
        exit 1
      fi
    fi
    ;;
  *)
    echo "[onstart] torch CUDA smoke SKIP (case does not install the nn group)"
    ;;
esac

mark "40_uv_sync_done"

# iter11 fix: never use `uv run` from here on. `uv run` re-checks
# pyproject.toml / uv.lock on every call and may trigger a background sync;
# two concurrent `uv run`s on one venv then see each other's half-synced
# state (`ModuleNotFoundError` for dvc / s3fs / rich ...). Calling the venv's
# python directly keeps the environment fixed.
PY_BIN="$(pwd)/backend/.venv/bin/python"
# iter12 fix: the backend/.venv/bin/dvc shim's shebang can point to a stale
# path ("bad interpreter"). `python -m dvc` bypasses the shim.
DVC_BIN="${PY_BIN} -m dvc"
echo "[onstart] pinned PY_BIN=${PY_BIN} DVC_BIN=${DVC_BIN}"

echo "[onstart] step=mkdir_run"
# Absolute path: train.py resolves REINFORCE_RUN_DIR, and a relative path would
# be interpreted from backend/ (the train cwd). The later dvc add / S3 upload
# use the same absolute path.
RUNS_PARENT_REL="data/output/models/<CASE_FAMILY>/<CASE_SUBDIR>/runs"
RUN_DIR_ABS="$(pwd)/${RUNS_PARENT_REL}/<RUN_ID>"
mkdir -p "${RUN_DIR_ABS}"

# ----------------------------------------------------------------- resume
# RL cases generate their training data online, so the only thing a pod ever
# pulls from DVC is a previous run directory to resume from:
#   - <RESUME_FROM> set  -> pull runs/<RESUME_FROM>.dvc and seed this run's
#                           directory with its checkpoint_latest.* files.
#   - otherwise          -> pull runs/<RUN_ID>.dvc if it already exists (a
#                           relaunch of the same run id after preemption).
#   - neither exists     -> fresh run, no dvc pull at all.
# train.py resumes automatically when checkpoint_latest.* is present in
# REINFORCE_RUN_DIR (run-dir contract, .claude/rules/backend/pipeline.md).
echo "[onstart] step=dvc_pull cwd=$(pwd) case=<CASE> family=<CASE_FAMILY>"
RESUME_FROM="<RESUME_FROM>"
if [ -z "${RESUME_FROM}" ] || [ "${RESUME_FROM}" = "none" ]; then
  RESUME_FROM="<RUN_ID>"
fi
RESUME_DVC="${RUNS_PARENT_REL}/${RESUME_FROM}.dvc"
if [ -f "${RESUME_DVC}" ]; then
  echo "[onstart] dvc pull (resume) ${RESUME_DVC}"
  pull_ok=0
  for attempt in 1 2 3; do
    if ${DVC_BIN} pull -j 4 "${RESUME_DVC}" 2>&1 | tail -20; then
      pull_ok=1
      break
    fi
    echo "[onstart] dvc pull (resume) attempt=${attempt} failed; retrying in 30s" >&2
    sleep 30
  done
  if [ "${pull_ok}" -ne 1 ]; then
    echo "[onstart] dvc pull (resume) FAILED" >&2
    mark "45_dvc_pull_resume_failed"
    exit 1
  fi
  if [ "${RESUME_FROM}" != "<RUN_ID>" ]; then
    SRC_RUN_DIR="${RUNS_PARENT_REL}/${RESUME_FROM}"
    shopt -s nullglob
    ckpts=("${SRC_RUN_DIR}"/checkpoint_latest.*)
    shopt -u nullglob
    if [ "${#ckpts[@]}" -eq 0 ]; then
      echo "[onstart] resume: no checkpoint_latest.* in ${SRC_RUN_DIR}" >&2
      mark "45_resume_checkpoint_missing"
      exit 1
    fi
    cp -a "${ckpts[@]}" "${RUN_DIR_ABS}/"
    echo "[onstart] resume: seeded ${RUN_DIR_ABS} with ${#ckpts[@]} checkpoint file(s) from ${RESUME_FROM}"
  fi
  mark "45_dvc_pull_resume_done"
else
  if [ "${RESUME_FROM}" != "<RUN_ID>" ]; then
    echo "[onstart] resume requested but ${RESUME_DVC} not found (commit/push the .dvc stub first)" >&2
    mark "45_resume_dvc_missing"
    exit 1
  fi
  echo "[onstart] dvc pull SKIP (fresh run; RL cases generate data online)"
fi

mark "50_dvc_pull_done"
