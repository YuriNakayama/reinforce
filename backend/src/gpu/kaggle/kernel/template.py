"""Kaggle Notebook (.ipynb) を render する純関数。

学習 run 1 回ごとに生成する notebook。cell 構成:
    A: env setup (os.environ.update + run_dir 作成)
    B: dataset mount を REPO_ROOT に mirror、flatten された data/ ファイルを復元、
       pre-built wheels を install
    C: backend/src と backend/ を sys.path に通す + 不足依存の install
       (case が `nn` group を使うときだけ torch の GPU probe)
    C2: JAX CUDA12 plugin + equinox/optax (JAX が既定 framework なので常に)
    D: train 実行 (registry の train_module + config_arg、subprocess → train.log)
    E: artifact 収集 (run_dir に集約)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:=@]+$")
_VALID_RUN_ID = re.compile(r"^\d{8}-\d{6}__[A-Za-z0-9_-]+__[0-9a-f]{7}__seed\d+$")


@dataclass(frozen=True)
class RenderContext:
    """Notebook render 用パラメータ。"""

    run_id: str
    commit_sha: str
    branch: str
    case: str
    train_module: str
    config_arg: str
    uv_groups: str
    dataset_slug: str
    accelerator: str
    kaggle_kernel_meta_initial: dict[str, Any]
    seed: int = 0


def render_notebook(ctx: RenderContext) -> dict[str, Any]:
    """nbformat 4 の notebook dict を返す。"""
    _validate(ctx)
    dataset_dirname = ctx.dataset_slug.split("/")[-1]
    dataset_mount = f"/kaggle/input/{dataset_dirname}"
    run_dir = f"/kaggle/working/runs/{ctx.run_id}"
    meta_json = json.dumps(ctx.kaggle_kernel_meta_initial, ensure_ascii=False)

    cells: list[dict[str, Any]] = [
        _code_cell(_cell_env_setup(ctx, run_dir, meta_json)),
        _code_cell(_cell_install_wheels(dataset_mount)),
        _code_cell(_cell_install_backend(ctx)),
        _code_cell(_cell_install_jax_cuda(ctx)),
        _code_cell(_cell_train(ctx, run_dir, dataset_mount)),
        _code_cell(_cell_collect_artifacts(ctx, run_dir)),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "name": "python3",
                "language": "python",
                "display_name": "Python 3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _validate(ctx: RenderContext) -> None:
    if not _VALID_RUN_ID.match(ctx.run_id):
        raise ValueError(f"invalid run_id format: {ctx.run_id!r}")
    for name, value in [
        ("commit_sha", ctx.commit_sha),
        ("branch", ctx.branch),
        ("case", ctx.case),
        ("train_module", ctx.train_module),
        ("dataset_slug", ctx.dataset_slug),
        ("accelerator", ctx.accelerator),
    ]:
        if not _VALID_VALUE.match(value):
            raise ValueError(
                f"invalid {name} value (only A-Za-z0-9._-/:=@ allowed): {value!r}"
            )
    if "/" not in ctx.dataset_slug:
        raise ValueError(f"dataset_slug must be <owner>/<name>: {ctx.dataset_slug!r}")
    if ctx.config_arg and not re.match(r"^[A-Za-z0-9._\-/=,\s]+$", ctx.config_arg):
        raise ValueError(f"invalid config_arg: {ctx.config_arg!r}")
    if ctx.uv_groups and not re.match(r"^[a-z0-9\- ]+$", ctx.uv_groups):
        raise ValueError(f"invalid uv_groups: {ctx.uv_groups!r}")


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _cell_env_setup(ctx: RenderContext, run_dir: str, meta_json: str) -> str:
    payload = {
        "REINFORCE_RUN_ID": ctx.run_id,
        "REINFORCE_GIT_SHA": ctx.commit_sha,
        "REINFORCE_GIT_BRANCH": ctx.branch,
        "REINFORCE_CASE": ctx.case,
        "REINFORCE_KAGGLE_KERNEL_SLUG": ctx.kaggle_kernel_meta_initial.get(
            "kernel_slug", ""
        ),
        "REINFORCE_KAGGLE_KERNEL_VERSION": str(
            ctx.kaggle_kernel_meta_initial.get("kernel_version", "")
        ),
        "REINFORCE_KAGGLE_ACCELERATOR": ctx.accelerator,
        "REINFORCE_KAGGLE_KERNEL_META": meta_json,
        "REINFORCE_RUN_DIR": run_dir,
    }
    return (
        "# cell A: env setup\n"
        "import os\n"
        f"_env = {json.dumps(payload, ensure_ascii=False)}\n"
        "os.environ.update({k: str(v) for k, v in _env.items()})\n"
        f"os.makedirs({run_dir!r}, exist_ok=True)\n"
    )


def _cell_install_wheels(dataset_mount: str) -> str:
    """Discover the actual dataset mount path, copy / symlink it into a writable
    REPO_ROOT (Kaggle datasets normally arrive at /kaggle/input/<slug>), then
    install any pre-built wheels (if any).

    Kaggle ``dir_mode='zip'`` uploads compress folders during upload, but the
    files are **unpacked on the dataset side**: at runtime the dataset mount
    contains a normal directory tree (``backend/`` with ``pyproject.toml``).
    We mirror that into ``/tmp/reinforce-repo/`` so the repo-root sentinel
    pair (``backend/pyproject.toml`` + ``.git``) is satisfied.

    Extra files under ``data/`` are shipped flattened as top-level
    ``kdata__<a>__<b>__<file>`` entries (Kaggle drops top-level ``data/`` zips,
    see ``dataset/builder.py``) and restored to ``data/<a>/<b>/<file>`` here.
    """
    return (
        "# cell B: discover dataset mount, mirror to writable REPO_ROOT\n"
        "import os, glob, shutil, subprocess\n"
        "REPO_ROOT = '/tmp/reinforce-repo'\n"
        f"EXPECTED = {dataset_mount!r}\n"
        "def _find_mount():\n"
        "    # Kaggle mounts datasets in a few different layouts depending on\n"
        "    # public/private/version. Walk /kaggle/input up to 3 levels and pick\n"
        "    # the first dir that contains ``backend/pyproject.toml``.\n"
        "    for root, dirs, files in os.walk('/kaggle/input'):\n"
        "        depth = root[len('/kaggle/input'):].count('/')\n"
        "        if depth > 3:\n"
        "            dirs[:] = []\n"
        "            continue\n"
        "        if 'backend' in dirs and os.path.isfile(\n"
        "            os.path.join(root, 'backend', 'pyproject.toml')\n"
        "        ):\n"
        "            return root\n"
        "    return None\n"
        "print('## /kaggle/input listing (debug):')\n"
        "_all_entries = []\n"
        "for root, dirs, files in os.walk('/kaggle/input'):\n"
        "    depth = root[len('/kaggle/input'):].count('/')\n"
        "    if depth > 6:\n"
        "        dirs[:] = []\n"
        "        continue\n"
        "    for f in files:\n"
        "        _all_entries.append(os.path.join(root, f))\n"
        "    if depth <= 4:\n"
        "        print(' ', root, 'dirs=', dirs[:8], 'files=', files[:8])\n"
        "print(f'## total files seen: {len(_all_entries)}')\n"
        "_data_files = [p for p in _all_entries if 'kdata__' in p or 'data.zip' in p]\n"
        "print(f'## flattened data files: {len(_data_files)}')\n"
        "for p in _data_files[:15]:\n"
        "    print('  ', p)\n"
        "DATASET_MOUNT = EXPECTED if os.path.isdir(EXPECTED) else _find_mount()\n"
        "if DATASET_MOUNT is None:\n"
        "    raise RuntimeError(f'dataset mount not found; expected {EXPECTED}')\n"
        "print('using DATASET_MOUNT =', DATASET_MOUNT)\n"
        "os.makedirs(REPO_ROOT, exist_ok=True)\n"
        "for entry in os.listdir(DATASET_MOUNT):\n"
        "    src = os.path.join(DATASET_MOUNT, entry)\n"
        "    dst = os.path.join(REPO_ROOT, entry)\n"
        "    if os.path.isdir(src):\n"
        "        if os.path.exists(dst):\n"
        "            shutil.rmtree(dst)\n"
        "        shutil.copytree(src, dst)\n"
        "    else:\n"
        "        shutil.copy2(src, dst)\n"
        "# Sentinel so find_repo_root() recognizes REPO_ROOT.\n"
        "os.makedirs(os.path.join(REPO_ROOT, '.git'), exist_ok=True)\n"
        "with open(os.path.join(REPO_ROOT, '.git', 'HEAD'), 'w') as f:\n"
        "    f.write('ref: refs/heads/kaggle-snapshot\\n')\n"
        "# Builder flattens extra data files to top-level kdata__<a>__<b>__<file>\n"
        "# (Kaggle drops top-level data/ zips). Restore them to data/<a>/<b>/<file>.\n"
        "for entry in os.listdir(REPO_ROOT):\n"
        "    if not entry.startswith('kdata__'):\n"
        "        continue\n"
        "    rel = entry[len('kdata__'):].split('__')\n"
        "    dst_path = os.path.join(REPO_ROOT, 'data', *rel)\n"
        "    os.makedirs(os.path.dirname(dst_path), exist_ok=True)\n"
        "    shutil.move(os.path.join(REPO_ROOT, entry), dst_path)\n"
        "    print(f'data restored: {entry} -> {dst_path}')\n"
        "# Install any pre-built wheels\n"
        "wheels = glob.glob(os.path.join(REPO_ROOT, 'wheels', '*.whl'))\n"
        "if wheels:\n"
        "    subprocess.check_call(['pip', 'install', '-q'] + wheels)\n"
        "    print('installed wheels:', wheels)\n"
        "else:\n"
        "    print('no wheels dir, skipping')\n"
    )


_TORCH_PROBE_BLOCK = (
    "\n"
    "_PROBE = (\n"
    "    'import json, sys\\n'\n"
    "    'try:\\n'\n"
    "    '    import torch\\n'\n"
    "    '    info = {\"version\": torch.__version__,\\n'\n"
    "    '            \"cuda\": torch.version.cuda,\\n'\n"
    "    '            \"available\": torch.cuda.is_available(),\\n'\n"
    '    \'            "smoke_ok": False, "err": None}\\n\'\n'
    "    '    if torch.cuda.is_available():\\n'\n"
    "    '        try:\\n'\n"
    "    '            (torch.zeros(2, 2).cuda() + 1).sum().item()\\n'\n"
    "    '            info[\"smoke_ok\"] = True\\n'\n"
    "    '        except Exception as e:\\n'\n"
    "    '            info[\"err\"] = repr(e)\\n'\n"
    "    '    else:\\n'\n"
    "    '        info[\"smoke_ok\"] = True\\n'\n"
    "    'except Exception as e:\\n'\n"
    '    \'    info = {"err": repr(e), "smoke_ok": False}\\n\'\n'
    "    'sys.stdout.write(json.dumps(info))\\n'\n"
    ")\n"
    "\n"
    "def _probe():\n"
    "    r = subprocess.run(\n"
    "        [sys.executable, '-c', _PROBE],\n"
    "        capture_output=True, text=True, check=False,\n"
    "    )\n"
    "    try:\n"
    "        info = json.loads(r.stdout.strip().splitlines()[-1])\n"
    "    except Exception:\n"
    "        info = {'smoke_ok': False, 'err': r.stderr[:500]}\n"
    "    print('torch probe:', info)\n"
    "    return info\n"
    "\n"
    "info = _probe()\n"
    "if not info.get('smoke_ok'):\n"
    "    # Kaggle's bundled torch wheel doesn't cover this GPU's CC.\n"
    "    # cu118 wheel carries sm_50/sm_60/sm_70/sm_75/sm_80/sm_86/sm_90\n"
    "    # so it covers every Kaggle-allocated GPU (P100=sm_60, T4=sm_75).\n"
    "    print('reinstalling torch with cu118 wheel (sm_50-sm_90)...')\n"
    "    subprocess.check_call([\n"
    "        'pip', 'install', '--quiet', '--force-reinstall',\n"
    "        '--index-url', 'https://download.pytorch.org/whl/cu118',\n"
    "        'torch',\n"
    "    ])\n"
    "    info = _probe()\n"
    "    if not info.get('smoke_ok'):\n"
    "        # Fail loudly rather than silently fall back to CPU.\n"
    "        raise RuntimeError(\n"
    "            f'GPU unusable for torch after cu118 reinstall; info={info}'\n"
    "        )\n"
)


def _cell_install_backend(ctx: RenderContext) -> str:
    """Make the backend packages importable by adding ``backend/src`` and
    ``backend/`` to ``sys.path`` and install third-party deps the Kaggle image
    lacks.

    Editable `pip install -e .` is fragile on Kaggle because pyproject.toml's
    ``[tool.uv]`` tables are uv-specific. Direct sys.path injection avoids the
    metadata-generation step entirely.

    When the case uses the ``nn`` dependency group (torch), the GPU's
    torch+CUDA pairing is probed; if the smoke fails (CC mismatch on T4 /
    P100), torch is reinstalled from the cu118 index.
    """
    uses_torch = "nn" in ctx.uv_groups.split()
    torch_block = (
        _TORCH_PROBE_BLOCK
        if uses_torch
        else "\n# torch probe skipped: case does not use the `nn` dependency group.\n"
    )
    import_probe = (
        "_IMPORT_PROBE = (\n"
        "    'import importlib, sys\\n'\n"
        "    f'sys.path[:0] = {_paths!r}\\n'\n"
        f"    'importlib.import_module({ctx.train_module!r})\\n'\n"
        "    'print(\"train module import OK\")\\n'\n"
        ")\n"
        "subprocess.run([sys.executable, '-c', _IMPORT_PROBE], check=False)\n"
    )
    return (
        "# cell C: make backend importable + install missing third-party deps\n"
        "# Probes run in subprocesses so the live kernel process never imports\n"
        "# a library that a later reinstall would have to replace.\n"
        "import sys, subprocess, json, os\n"
        "REPO_ROOT = '/tmp/reinforce-repo'\n"
        "_paths = [\n"
        "    REPO_ROOT + '/backend/src',\n"
        "    REPO_ROOT + '/backend',\n"
        "    REPO_ROOT,\n"
        "]\n"
        "for p in _paths:\n"
        "    if p not in sys.path:\n"
        "        sys.path.insert(0, p)\n"
        "# Install third-party deps that the Kaggle image may lack.\n"
        "deps = ['polars', 'gymnasium', 'farama-notifications']\n"
        "subprocess.run(['pip', 'install', '--no-deps'] + deps, check=False)\n"
        + torch_block
        + "\n"
        "# Verify the train module is importable in a subprocess too (matches\n"
        "# the env that cell D's training subprocess will see).\n" + import_probe
    )


def _cell_install_jax_cuda(ctx: RenderContext) -> str:
    """Install jax-cuda12 plugin + matching cuDNN so JAX picks up the GPU.

    JAX is the repo's default RL framework (the RunPod onstart installs the
    `cuda` group for every case too). Kaggle base image ships CPU JAX only;
    without the plugin XLA never sees the GPU and training silently runs on
    CPU. On a GPU accelerator the cell fails loudly if JAX still does not bind
    to the GPU; with ``accelerator=cpu`` it only reports the probe.

    重要な制約:
    - `pip install --reinstall jax[cuda12]` は **避ける** (既ロード jax の
      共有ライブラリハンドルが invalid 化して SIGABRT)。plugin + pjrt + cudnn
      を素直に追加 install する。
    - jax-cuda12-plugin 0.10 は compile-time CuDNN 9.8 を期待する。Kaggle
      の base image は CuDNN 9.x 系だが version が plugin と合わないと
      `RET_CHECK failure dnn_support != nullptr` で GPU 演算が abort する。
      `nvidia-cudnn-cu12>=9.8,<9.9` を pin して plugin に合わせる。
    """
    require_gpu = ctx.accelerator.startswith("gpu")
    return (
        "# cell C2: install jax-cuda12 plugin + matching cuDNN + equinox/optax\n"
        f"REQUIRE_GPU = {require_gpu!r}\n"
        "import subprocess, sys, json, time\n"
        "pkgs = [\n"
        "    'jax-cuda12-plugin',\n"
        "    'jax-cuda12-pjrt',\n"
        "    'nvidia-cudnn-cu12>=9.8,<9.9',\n"
        "    # JAX cases depend on equinox (models) and optax (optimizers).\n"
        "    # Kaggle base image does not ship these.\n"
        "    'equinox>=0.11.0',\n"
        "    'optax>=0.2.0',\n"
        "]\n"
        "print('installing JAX CUDA12 plugin + equinox/optax:', pkgs)\n"
        "# Retry up to 3 times because files.pythonhosted.org occasionally\n"
        "# returns ReadTimeoutError from the Kaggle outbound NAT (observed\n"
        "# in run 20260522-030616). --retries lets pip retry within a\n"
        "# single subprocess invocation; the outer for-loop covers full\n"
        "# subprocess failures (e.g. SIGKILL on first DNS attempt).\n"
        "last_err = ''\n"
        "for attempt in range(3):\n"
        "    r = subprocess.run(\n"
        "        [\n"
        "            'pip', 'install', '-q',\n"
        "            '--retries', '5',\n"
        "            '--timeout', '60',\n"
        "        ] + pkgs,\n"
        "        capture_output=True, text=True, check=False,\n"
        "    )\n"
        "    if r.returncode == 0:\n"
        "        break\n"
        "    last_err = r.stderr[-2000:]\n"
        "    print(f'pip install attempt {attempt + 1} failed:', last_err)\n"
        "    time.sleep(5)\n"
        "else:\n"
        "    msg = (\n"
        "        'jax cuda plugin install failed after 3 attempts; '\n"
        "        f'last err: {last_err}'\n"
        "    )\n"
        "    raise RuntimeError(msg)\n"
        "print('pip install OK')\n"
        "# Probe JAX devices in a subprocess so the live kernel doesn't\n"
        "# import jax (would lock in the CPU backend before plugin is wired).\n"
        "_PROBE = (\n"
        "    'import json, jax\\n'\n"
        "    'devs = jax.devices()\\n'\n"
        "    'print(json.dumps({\"backend\": jax.default_backend(),'\n"
        "    ' \"devices\": [str(d) for d in devs]}))\\n'\n"
        ")\n"
        "r = subprocess.run(\n"
        "    [sys.executable, '-c', _PROBE],\n"
        "    capture_output=True, text=True, check=False,\n"
        ")\n"
        "out = r.stdout.strip().splitlines()[-1] if r.stdout else '{}'\n"
        "try:\n"
        "    info = json.loads(out)\n"
        "except Exception:\n"
        "    info = {'raw': out, 'err': r.stderr[-500:]}\n"
        "print('jax device probe:', info)\n"
        "if REQUIRE_GPU and info.get('backend') != 'gpu':\n"
        "    # Fail loudly — JAX training on CPU defeats the purpose of using\n"
        "    # Kaggle's GPU quota.\n"
        "    raise RuntimeError(\n"
        "        f'JAX did not bind to GPU; backend={info.get(\"backend\")!r}'\n"
        "    )\n"
    )


def _cell_train(ctx: RenderContext, run_dir: str, dataset_mount: str) -> str:
    args = ctx.config_arg.strip().split() if ctx.config_arg.strip() else []
    return (
        "# cell D: train\n"
        "import subprocess, sys, os\n"
        f"run_dir = {run_dir!r}\n"
        "log_path = os.path.join(run_dir, 'train.log')\n"
        f"cmd = [sys.executable, '-m', {ctx.train_module!r}] + {args!r}\n"
        "cwd = '/tmp/reinforce-repo/backend'\n"
        "env = os.environ.copy()\n"
        "env['PYTHONPATH'] = (\n"
        "    '/tmp/reinforce-repo/backend/src:/tmp/reinforce-repo/backend:'\n"
        "    '/tmp/reinforce-repo'\n"
        ")\n"
        "# (GPU enabled by default. Set REINFORCE_KAGGLE_FORCE_CPU=1 in the kernel\n"
        "#  env to hide the GPU from the training process.)\n"
        "if os.environ.get('REINFORCE_KAGGLE_FORCE_CPU') == '1':\n"
        "    env['CUDA_VISIBLE_DEVICES'] = ''\n"
        "    print('CUDA disabled via REINFORCE_KAGGLE_FORCE_CPU')\n"
        "print('running:', ' '.join(cmd), 'cwd=', cwd)\n"
        "with open(log_path, 'w') as logf:\n"
        "    proc = subprocess.run(\n"
        "        cmd, cwd=cwd, env=env,\n"
        "        stdout=logf, stderr=subprocess.STDOUT, check=False,\n"
        "    )\n"
        "print('exit_code=', proc.returncode)\n"
        "with open(log_path) as f:\n"
        "    tail = f.readlines()[-80:]\n"
        "print(''.join(tail))\n"
        "if proc.returncode != 0:\n"
        "    raise SystemExit(proc.returncode)\n"
    )


def _cell_collect_artifacts(ctx: RenderContext, run_dir: str) -> str:
    # The env cell sets REINFORCE_RUN_DIR=run_dir, so train.py writes directly
    # into `run_dir` (= /kaggle/working/runs/<id>). This cell is a
    # belt-and-suspenders fallback that also pulls anything a case wrote under
    # its config-default tree (data/output/models/<family>/<subdir>/runs/<id>).
    # case_family/case_subdir are pure lookups over the shared CASE_DEFAULTS
    # registry. eval/ is copied as a directory (eval/seed<K>.json).
    from gpu.runpod.config.cases import case_family, case_subdir  # noqa: PLC0415

    family = case_family(ctx.case)
    subdir = case_subdir(ctx.case)
    src = f"/tmp/reinforce-repo/data/output/models/{family}/{subdir}/runs/{ctx.run_id}/"
    return (
        "# cell E: collect artifacts into run_dir\n"
        "import os, shutil, glob\n"
        f"src = {src!r}\n"
        f"dst = {run_dir!r}\n"
        "os.makedirs(dst, exist_ok=True)\n"
        "if os.path.isdir(src):\n"
        "    for f in glob.glob(os.path.join(src, '*')):\n"
        "        if os.path.isfile(f):\n"
        "            shutil.copy2(f, dst)\n"
        "        elif os.path.isdir(f):\n"
        "            shutil.copytree(f, os.path.join(dst, os.path.basename(f)),\n"
        "                            dirs_exist_ok=True)\n"
        "for root, _dirs, files in os.walk(dst):\n"
        "    for f in sorted(files):\n"
        "        path = os.path.join(root, f)\n"
        "        print(os.path.relpath(path, dst), os.path.getsize(path))\n"
    )
