"""Interactive notebook (.ipynb) template.

cell A-C: 既存 train notebook と同じ (env / mount / install)
cell D:   S3 command channel loop (training 実行のかわり)
cell E:   sleep guard + final cleanup

kernel が ERROR / OOM で死亡しても、S3 にはそれまでの heartbeat + outbox 内容が
残るため、Claude はローカルから ``dev/kaggle logs <run_id>`` で死亡直前
までの状態を取得できる (これが SSH なしで RunPod ``dev`` 相当を実現する核心)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from gpu.kaggle.interactive.channel import DEFAULT_BUCKET, PREFIX_ROOT

_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:=@]+$")
_VALID_RUN_ID = re.compile(r"^\d{8}-\d{6}__[A-Za-z0-9_-]+__[0-9a-f]{7}__seed\d+$")


@dataclass(frozen=True)
class InteractiveContext:
    """Interactive notebook render 用パラメータ。"""

    run_id: str
    commit_sha: str
    branch: str
    case: str
    dataset_slug: str
    accelerator: str
    kaggle_kernel_meta_initial: dict[str, Any]
    s3_bucket: str = DEFAULT_BUCKET
    heartbeat_interval: float = 10.0
    poll_interval: float = 2.0
    max_idle_minutes: float = 480.0  # 8h before voluntary exit (Kaggle 9h hard limit)
    seed: int = 0


def render_interactive_notebook(ctx: InteractiveContext) -> dict[str, Any]:
    """nbformat 4 の interactive notebook dict を返す。"""
    _validate(ctx)
    dataset_dirname = ctx.dataset_slug.split("/")[-1]
    dataset_mount = f"/kaggle/input/{dataset_dirname}"
    meta_json = json.dumps(ctx.kaggle_kernel_meta_initial, ensure_ascii=False)

    cells: list[dict[str, Any]] = [
        _code_cell(_cell_env_setup(ctx, meta_json)),
        _code_cell(_cell_install_wheels(dataset_mount)),
        _code_cell(_cell_install_backend(dataset_mount)),
        _code_cell(_cell_interactive_loop(ctx)),
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


def _validate(ctx: InteractiveContext) -> None:
    if not _VALID_RUN_ID.match(ctx.run_id):
        raise ValueError(f"invalid run_id format: {ctx.run_id!r}")
    for name, value in [
        ("commit_sha", ctx.commit_sha),
        ("branch", ctx.branch),
        ("case", ctx.case),
        ("dataset_slug", ctx.dataset_slug),
        ("accelerator", ctx.accelerator),
        ("s3_bucket", ctx.s3_bucket),
    ]:
        if not _VALID_VALUE.match(value):
            raise ValueError(
                f"invalid {name} value (only A-Za-z0-9._-/:=@ allowed): {value!r}"
            )
    if "/" not in ctx.dataset_slug:
        raise ValueError(f"dataset_slug must be <owner>/<name>: {ctx.dataset_slug!r}")
    if ctx.heartbeat_interval <= 0 or ctx.poll_interval <= 0:
        raise ValueError("intervals must be positive")
    if ctx.max_idle_minutes <= 0:
        raise ValueError("max_idle_minutes must be positive")


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _cell_env_setup(ctx: InteractiveContext, meta_json: str) -> str:
    """env setup + AWS creds を取得する。

    取得経路 (priority 順):

    1. Kaggle UserSecretsClient (Kaggle Web UI で attach した場合のみ)
    2. dataset 同梱の ``backend/.env`` (dev_cmd は ``include_dotenv=True`` で push)
    3. どちらも無ければ ``RuntimeError``

    後者の経路があるので、毎 kernel push ごとに Web UI で Secrets を attach する
    必要はなく、private dataset 同梱で配送できる。
    """
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
        "REINFORCE_KAGGLE_INTERACTIVE": "1",
    }
    return (
        "# cell A: env setup (interactive) + load AWS creds\n"
        "import os\n"
        f"_env = {json.dumps(payload, ensure_ascii=False)}\n"
        "os.environ.update({k: str(v) for k, v in _env.items()})\n"
        "_REQUIRED = ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')\n"
        "_OPTIONAL = ('AWS_DEFAULT_REGION', 'AWS_SESSION_TOKEN')\n"
        "def _find_dotenv():\n"
        "    # os.walk で /kaggle/input 配下を歩き、backend/.env を探す。glob の hidden\n"
        "    # file 扱いに依存しない。\n"
        "    for root, dirs, files in os.walk('/kaggle/input'):\n"
        "        depth = root[len('/kaggle/input'):].count('/')\n"
        "        if depth > 5:\n"
        "            dirs[:] = []\n"
        "            continue\n"
        "        if os.path.basename(root) == 'backend' and '.env' in files:\n"
        "            return os.path.join(root, '.env')\n"
        "    return None\n"
        "def _load_aws_creds():\n"
        "    # Priority 1: Kaggle UserSecretsClient (notebook attach 経由)\n"
        "    try:\n"
        "        from kaggle_secrets import UserSecretsClient\n"
        "        secrets = UserSecretsClient()\n"
        "        for k in _REQUIRED:\n"
        "            os.environ[k] = secrets.get_secret(k)\n"
        "        for k in _OPTIONAL:\n"
        "            try:\n"
        "                os.environ[k] = secrets.get_secret(k)\n"
        "            except Exception:\n"
        "                pass\n"
        "        return 'kaggle-secrets'\n"
        "    except Exception:\n"
        "        pass\n"
        "    # Priority 2: dataset 同梱の backend/.env (include_dotenv=True)\n"
        "    env_path = _find_dotenv()\n"
        "    if env_path:\n"
        "        with open(env_path) as fh:\n"
        "            for line in fh:\n"
        "                line = line.strip()\n"
        "                if not line or line.startswith('#') or '=' not in line:\n"
        "                    continue\n"
        "                k, v = line.split('=', 1)\n"
        "                k = k.strip(); v = v.strip()\n"
        "                if k in _REQUIRED or k in _OPTIONAL:\n"
        "                    os.environ[k] = v\n"
        "        return f'dotenv ({env_path})'\n"
        "    return None\n"
        "# Debug: dump /kaggle/input layout so failures are diagnosable.\n"
        "print('## /kaggle/input listing:')\n"
        "for root, dirs, files in os.walk('/kaggle/input'):\n"
        "    depth = root[len('/kaggle/input'):].count('/')\n"
        "    if depth > 4:\n"
        "        dirs[:] = []\n"
        "        continue\n"
        "    print(' ', root, 'dirs=', dirs[:5], 'files=', files[:8])\n"
        "_source = _load_aws_creds()\n"
        "os.environ.setdefault('AWS_DEFAULT_REGION', 'ap-northeast-1')\n"
        "_missing = [k for k in _REQUIRED if not os.environ.get(k)]\n"
        "if _missing:\n"
        "    raise RuntimeError(\n"
        "        f'AWS creds missing: {_missing} (source tried: {_source}). '\n"
        "        'Run `dev/kaggle dev` (includes backend/.env in dataset) or '\n"
        "        'attach Kaggle Secrets manually.'\n"
        "    )\n"
        "print(f'AWS creds loaded from {_source} '\n"
        "      f\"(region={os.environ['AWS_DEFAULT_REGION']})\")\n"
    )


def _cell_install_wheels(dataset_mount: str) -> str:
    """train notebook と同じ mount / mirror ロジック (REPO_ROOT=/tmp)。"""
    return (
        "# cell B: discover dataset mount, mirror to writable REPO_ROOT\n"
        "import os, glob, shutil, subprocess\n"
        "REPO_ROOT = '/tmp/reinforce-repo'\n"
        f"EXPECTED = {dataset_mount!r}\n"
        "def _find_mount():\n"
        "    # First dir under /kaggle/input that contains backend/pyproject.toml.\n"
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
        "print('## /kaggle/input listing:')\n"
        "for root, dirs, files in os.walk('/kaggle/input'):\n"
        "    depth = root[len('/kaggle/input'):].count('/')\n"
        "    if depth > 3:\n"
        "        dirs[:] = []\n"
        "        continue\n"
        "    print(' ', root, 'dirs=', dirs[:5], 'files=', files[:5])\n"
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
        "os.makedirs(os.path.join(REPO_ROOT, '.git'), exist_ok=True)\n"
        "with open(os.path.join(REPO_ROOT, '.git', 'HEAD'), 'w') as f:\n"
        "    f.write('ref: refs/heads/kaggle-snapshot\\n')\n"
        "# Restore flattened data files (kdata__<a>__<b>__<file> -> data/<a>/<b>/<file>).\n"
        "for entry in os.listdir(REPO_ROOT):\n"
        "    if not entry.startswith('kdata__'):\n"
        "        continue\n"
        "    rel = entry[len('kdata__'):].split('__')\n"
        "    dst_path = os.path.join(REPO_ROOT, 'data', *rel)\n"
        "    os.makedirs(os.path.dirname(dst_path), exist_ok=True)\n"
        "    shutil.move(os.path.join(REPO_ROOT, entry), dst_path)\n"
        "wheels = glob.glob(os.path.join(REPO_ROOT, 'wheels', '*.whl'))\n"
        "if wheels:\n"
        "    subprocess.check_call(['pip', 'install', '-q'] + wheels)\n"
        "    print('installed wheels:', wheels)\n"
        "else:\n"
        "    print('no wheels dir, skipping')\n"
    )


def _cell_install_backend(dataset_mount: str) -> str:
    return (
        "# cell C: make backend packages importable + install boto3 for S3 channel\n"
        "import sys, subprocess\n"
        "REPO_ROOT = '/tmp/reinforce-repo'\n"
        "for p in [REPO_ROOT + '/backend/src', REPO_ROOT + '/backend', REPO_ROOT]:\n"
        "    if p not in sys.path:\n"
        "        sys.path.insert(0, p)\n"
        "deps = ['polars', 'gymnasium', 'farama-notifications', 'boto3']\n"
        "subprocess.run(['pip', 'install', '--no-deps'] + deps, check=False)\n"
        "import importlib\n"
        "for mod in ('boto3', 'pipeline', 'gpu', 'simulate'):\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "        print('import OK:', mod)\n"
        "    except Exception as e:\n"
        "        print('import FAIL:', mod, '->', e)\n"
    )


def _cell_interactive_loop(ctx: InteractiveContext) -> str:
    """S3 command channel polling loop.

    Claude が put したコマンドを inbox から拾い、subprocess で実行し、結果を
    outbox に書く。heartbeat を 10s 毎に更新。max_idle_minutes 経過後 / shutdown
    コマンド受領で voluntarily 終了する。
    """
    return (
        "# cell D: S3 command channel loop (interactive mode)\n"
        "import os, json, time, subprocess, traceback\n"
        "from datetime import datetime, timezone\n"
        "try:\n"
        "    import boto3\n"
        "except ImportError as e:\n"
        "    raise RuntimeError(\n"
        "        'boto3 is required for interactive mode but failed to import: '\n"
        "        + str(e)\n"
        "    ) from e\n"
        f"BUCKET = {ctx.s3_bucket!r}\n"
        f"PREFIX = '{PREFIX_ROOT}/{ctx.run_id}'\n"
        f"HEARTBEAT_INTERVAL = {ctx.heartbeat_interval}\n"
        f"POLL_INTERVAL = {ctx.poll_interval}\n"
        f"MAX_IDLE_SEC = {ctx.max_idle_minutes * 60.0}\n"
        "STDOUT_CAP = 1_000_000\n"
        "STDERR_CAP = 1_000_000\n"
        "s3 = boto3.client('s3')\n"
        "def _iso():\n"
        "    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')\n"
        "def _heartbeat(state, **extra):\n"
        "    payload = {'state': state, 'iso': _iso(),\n"
        "               'run_id': os.environ.get('REINFORCE_RUN_ID',''), **extra}\n"
        "    s3.put_object(Bucket=BUCKET, Key=f'{PREFIX}/state/heartbeat.json',\n"
        "                  Body=json.dumps(payload).encode('utf-8'))\n"
        "def _run_cmd(cmd):\n"
        "    started = _iso(); start = time.monotonic()\n"
        "    env = os.environ.copy(); env.update(cmd.get('env') or {})\n"
        "    try:\n"
        "        proc = subprocess.run(cmd['argv'], cwd=cmd.get('cwd'),\n"
        "                              env=env, capture_output=True, text=True,\n"
        "                              timeout=float(cmd.get('timeout', 300)))\n"
        "        out = {'returncode': proc.returncode,\n"
        "               'stdout': (proc.stdout or '')[-STDOUT_CAP:],\n"
        "               'stderr': (proc.stderr or '')[-STDERR_CAP:],\n"
        "               'error': None}\n"
        "    except subprocess.TimeoutExpired as e:\n"
        "        out = {'returncode': -9,\n"
        "               'stdout': (e.stdout or '')[-STDOUT_CAP:] if isinstance(e.stdout, str) else '',\n"
        "               'stderr': (e.stderr or '')[-STDERR_CAP:] if isinstance(e.stderr, str) else '',\n"
        "               'error': f'timeout after {cmd.get(\"timeout\", 300)}s'}\n"
        "    except Exception as e:\n"
        "        out = {'returncode': -1, 'stdout': '', 'stderr': '',\n"
        "               'error': f'{type(e).__name__}: {e}\\n' + traceback.format_exc()}\n"
        "    out.update({'command_id': cmd.get('command_id',''),\n"
        "                'started_at': started, 'finished_at': _iso(),\n"
        "                'elapsed_seconds': round(time.monotonic()-start, 3)})\n"
        "    return out\n"
        "print('## interactive loop start; bucket=' + BUCKET + ' prefix=' + PREFIX)\n"
        "_heartbeat('ready')\n"
        "last_hb = time.monotonic(); session_start = time.monotonic()\n"
        "while True:\n"
        "    if time.monotonic() - session_start > MAX_IDLE_SEC:\n"
        "        _heartbeat('voluntary_exit', reason='max_idle_minutes')\n"
        "        print('voluntary exit (max_idle)'); break\n"
        "    try:\n"
        "        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f'{PREFIX}/inbox/', MaxKeys=10)\n"
        "    except Exception as e:\n"
        "        print('list_objects_v2 error:', e); time.sleep(POLL_INTERVAL); continue\n"
        "    for obj in resp.get('Contents') or []:\n"
        "        key = obj['Key']\n"
        "        try:\n"
        "            body = s3.get_object(Bucket=BUCKET, Key=key)['Body'].read()\n"
        "            cmd = json.loads(body)\n"
        "        except Exception as e:\n"
        "            print('failed to read', key, e); continue\n"
        "        if cmd.get('argv') == ['__shutdown__']:\n"
        "            _heartbeat('shutdown'); s3.delete_object(Bucket=BUCKET, Key=key)\n"
        "            print('shutdown command received'); raise SystemExit(0)\n"
        "        _heartbeat('running', command_id=cmd.get('command_id',''))\n"
        "        result = _run_cmd(cmd)\n"
        "        out_key = f\"{PREFIX}/outbox/{cmd.get('command_id','unknown')}.json\"\n"
        "        s3.put_object(Bucket=BUCKET, Key=out_key,\n"
        "                      Body=json.dumps(result).encode('utf-8'))\n"
        "        s3.delete_object(Bucket=BUCKET, Key=key)\n"
        "        _heartbeat('ready')\n"
        "        last_hb = time.monotonic()\n"
        "    if time.monotonic() - last_hb >= HEARTBEAT_INTERVAL:\n"
        "        _heartbeat('idle'); last_hb = time.monotonic()\n"
        "    time.sleep(POLL_INTERVAL)\n"
    )
