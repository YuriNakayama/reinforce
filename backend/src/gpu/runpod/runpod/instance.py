"""onstart テンプレ render と RunPod create_pod ラッパ。

placeholder 置換時に shell injection を防ぐため、値は厳格な regex でバリデート
してから単純な文字列置換を行う (vast.instance と同設計)。

RunPod の `docker_args` は GraphQL 文字列にそのまま埋め込まれる
(`f'dockerArgs: "{docker_args}"'`) ため、改行や `"` を含むスクリプトを直接渡すと
GraphQL Syntax Error になる。本基盤では onstart スクリプトを base64 エンコードして
1 行の bootstrap (`bash -c "echo <b64> | base64 -d | bash"`) として渡すことで
回避する。
"""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 置換可能な値の許容文字種。shell injection を防ぐため厳しめに固定。
_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:]+$")
# CONFIG_ARG は空文字、または `--flag` / `--flag <value>` の並び
# (例: `--smoke`, `--config pipeline/ppo/case2/config.yaml`) のみ許可。
_FLAG = r"--[a-z][a-z0-9-]*( [A-Za-z0-9._\-/]+)?"
_VALID_CONFIG_ARG = re.compile(rf"^(|{_FLAG}( {_FLAG})*)$")
# UV_GROUPS は空文字または space 区切りの group 名 (例: "nn", "nn cuda")。
_VALID_UV_GROUPS = re.compile(r"^(|[a-z][a-z0-9-]*( [a-z][a-z0-9-]*)*)$")
# PREPROCESS_CMD は空文字または `module.path[ --config <path>]` 形式のみ許可。
_VALID_PREPROCESS_CMD = re.compile(
    r"^(|[A-Za-z0-9._\-/]+( --config [A-Za-z0-9._\-/]+)?)$"
)
_TEMPLATE_PLACEHOLDERS = (
    "<COMMIT_SHA>",
    "<RUN_ID>",
    "<BRANCH>",
    "<REPO_URL>",
    "<CASE>",
    # `<CASE>` is the registry key (e.g. "ppo_case1_smoke"). `<CASE_FAMILY>`
    # is the top-level family directory ("ppo" / "dqn" / ...) and
    # `<CASE_SUBDIR>` is the on-disk case dir ("case1", ...). Onstart shell
    # scripts persist / DVC-pull / push under
    # `data/output/models/<CASE_FAMILY>/<CASE_SUBDIR>/runs/<RUN_ID>`.
    "<CASE_FAMILY>",
    "<CASE_SUBDIR>",
    "<TRAIN_MODULE>",
    "<CONFIG_ARG>",
    "<PREPROCESS_CMD>",
    "<UV_GROUPS>",
    "<RESUME_FROM>",
    "<RUNPOD_MODE>",
)

# pod の運用モード:
#   - "oneshot":     train → DVC push → 自動 remove (従来動作、CI / 学習ジョブ向け)
#   - "interactive": uv sync (+ preprocess) 後に sleep infinity で待機。
#                    ユーザーが `dev/runpod ssh` で接続して対話的に作業し、
#                    `dev/runpod destroy` で明示的に削除する。
SUPPORTED_RUNPOD_MODES = ("oneshot", "interactive")

_ONSTART_LIBS: dict[str, str] = {
    "__ONSTART_LIB_00_PRELUDE__": "00_prelude.sh",
    "__ONSTART_LIB_10_TOOLS__": "10_tools.sh",
    "__ONSTART_LIB_20_CLEANUP__": "20_cleanup.sh",
    "__ONSTART_LIB_30_PERSIST_AND_CLONE__": "30_persist_and_clone.sh",
    "__ONSTART_LIB_40_UV_AND_DVC_PULL__": "40_uv_and_dvc_pull.sh",
    "__ONSTART_LIB_50_PREPROCESS__": "50_preprocess.sh",
    "__ONSTART_LIB_60_TRAIN__": "60_train.sh",
    "__ONSTART_LIB_70_ARTIFACTS_AND_DVC__": "70_artifacts_and_dvc.sh",
}

# RunPod 公式 PyTorch image。SSH/Jupyter/runpodctl が pre-install されている。
# CUDA 12.4 系を default とする: CUDA 13 系 (cu1300-torch291-ubuntu2404) は
# host NVIDIA driver 580+ 必須で、RunPod の SECURE 4090 fallback ノードに
# 古い driver (CUDA 12.x までしか提供しない) が混在しており、運悪く当たると
# OCI prestart hook で `nvidia-container-cli: requirement error: cuda>=13.0`
# で起動失敗する。CUDA 12.4 (driver 550+ 要件) はほぼ全 host driver で動く。
# 観測 (2026-05-04, EU-RO-1 SECURE 4090): cu1300 image で `pretart hook #0:
# exit status 1` を返したノードあり → cu1241 にダウングレードで回避。
# 古いタグ (pytorch/pytorch:2.4.0-cuda12.4.1-devel-ubuntu22.04) は pull 中
# に止まる挙動を観測したため runpod/pytorch 系を採用。
DEFAULT_IMAGE = os.environ.get(
    "REINFORCE_RUNPOD_IMAGE",
    "runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204",
)
DEFAULT_DISK_GB = 40
DEFAULT_PORTS = "22/tcp,8888/http"


class TemplateError(ValueError):
    """テンプレート置換で不正値を検出したときに投げる。"""


def _validate(name: str, value: str) -> None:
    if not value:
        raise TemplateError(f"empty value for {name!r}")
    if not _VALID_VALUE.match(value):
        raise TemplateError(
            f"invalid characters in {name!r}={value!r}; "
            "shell injection prevention rejected the value"
        )


def _validate_config_arg(value: str) -> None:
    if not _VALID_CONFIG_ARG.match(value):
        raise TemplateError(
            f"invalid config_arg={value!r}; expected '' or '--flag [value]' "
            "tokens (e.g. '--smoke', '--config <path>') with safe characters only"
        )


def _validate_uv_groups(value: str) -> None:
    if not _VALID_UV_GROUPS.match(value):
        raise TemplateError(
            f"invalid uv_groups={value!r}; expected '' or space-separated "
            "dependency group names"
        )


def _validate_preprocess_cmd(value: str) -> None:
    if not _VALID_PREPROCESS_CMD.match(value):
        raise TemplateError(
            f"invalid preprocess_cmd={value!r}; expected '' or "
            "'module.path [--config <path>]' with safe characters only"
        )


def _inline_onstart_libs(template_path: Path, text: str) -> str:
    """Expand onstart_lib placeholders into a single self-contained script."""
    lib_dir = template_path.parent / "onstart_lib"
    for placeholder, filename in _ONSTART_LIBS.items():
        if placeholder not in text:
            continue
        lib_path = lib_dir / filename
        if not lib_path.is_file():
            raise TemplateError(f"onstart library not found: {lib_path}")
        text = text.replace(placeholder, lib_path.read_text(encoding="utf-8"))
    leftovers = [p for p in _ONSTART_LIBS if p in text]
    if leftovers:
        raise TemplateError(f"unsubstituted onstart libraries remain: {leftovers}")
    return text


def render_onstart(
    template_path: Path,
    *,
    commit_sha: str,
    run_id: str,
    branch: str,
    repo_url: str,
    case: str,
    case_family: str,
    case_subdir: str,
    train_module: str,
    config_arg: str = "",
    preprocess_cmd: str = "",
    uv_groups: str = "",
    resume_from: str = "",
    mode: str = "oneshot",
) -> str:
    """テンプレを読み placeholder を置換した script 文字列を返す。

    値は事前に regex でバリデーション。違反したら TemplateError。

    Args:
        uv_groups: 追加の uv dependency group (space 区切り、例 "nn")。
            `cuda` group は onstart 側で常に追加される。
        resume_from: 空文字、または checkpoint_latest.* を引き継ぐ過去 run_id。
        mode: "oneshot" (既定) で train → 自動 remove。"interactive" で sleep
            infinity 保持し SSH 接続で対話操作する。詳細は SUPPORTED_RUNPOD_MODES。
    """
    _validate("commit_sha", commit_sha)
    _validate("run_id", run_id)
    _validate("branch", branch)
    _validate("repo_url", repo_url)
    _validate("case", case)
    _validate("case_family", case_family)
    _validate("case_subdir", case_subdir)
    # train_module は preprocess-only 経路では空文字。値があれば形式チェック。
    if train_module and not _VALID_VALUE.match(train_module):
        raise TemplateError(
            f"invalid characters in 'train_module'={train_module!r}; "
            "shell injection prevention rejected the value"
        )
    _validate_config_arg(config_arg)
    _validate_preprocess_cmd(preprocess_cmd)
    _validate_uv_groups(uv_groups)
    if resume_from:
        _validate("resume_from", resume_from)
    if mode not in SUPPORTED_RUNPOD_MODES:
        raise TemplateError(
            f"invalid mode={mode!r}; must be one of {SUPPORTED_RUNPOD_MODES}"
        )
    if not template_path.is_file() and template_path.name == "onstart.sh.tmpl":
        template_path = (
            Path(__file__).resolve().parent.parent / "execution" / "onstart.sh.tmpl"
        )
    text = template_path.read_text(encoding="utf-8")
    text = _inline_onstart_libs(template_path, text)
    substitutions = {
        "<COMMIT_SHA>": commit_sha,
        "<RUN_ID>": run_id,
        "<BRANCH>": branch,
        "<REPO_URL>": repo_url,
        "<CASE>": case,
        "<CASE_FAMILY>": case_family,
        "<CASE_SUBDIR>": case_subdir,
        "<TRAIN_MODULE>": train_module,
        "<CONFIG_ARG>": config_arg,
        "<PREPROCESS_CMD>": preprocess_cmd,
        "<UV_GROUPS>": uv_groups,
        "<RESUME_FROM>": resume_from,
        "<RUNPOD_MODE>": mode,
    }
    for placeholder, value in substitutions.items():
        text = text.replace(placeholder, value)
    leftover = [p for p in _TEMPLATE_PLACEHOLDERS if p in text]
    if leftover:
        raise TemplateError(f"unsubstituted placeholders remain: {leftover}")
    return text


def build_env_dict(env: Mapping[str, str]) -> dict[str, str]:
    """`runpod.create_pod(env=...)` に渡す dict を組み立てる。

    変数名のバリデーションだけ行う。値の shell 注入経路は onstart 側で別途検証。
    """
    result: dict[str, str] = {}
    for key, value in env.items():
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", key):
            raise TemplateError(f"invalid env var name: {key!r}")
        result[key] = value
    return result


def create_pod(
    sdk: Any,
    *,
    name: str,
    gpu_type_id: str,
    cloud_type: str,
    onstart_script: str,
    env: Mapping[str, str],
    image: str = DEFAULT_IMAGE,
    container_disk_gb: int = DEFAULT_DISK_GB,
    network_volume_id: str | None = None,
    volume_mount_path: str = "/persist",
    ports: str = DEFAULT_PORTS,
    data_center_id: str | None = None,
) -> str:
    """`runpod.create_pod(...)` を呼び、生成された pod id を返す。

    `docker_args` には `bash -c "<onstart_script>"` を渡す (RunPod は Vast の
    `onstart_cmd` 直接相当がなく、コンテナの ENTRYPOINT/CMD を上書きする経路)。
    onstart 側で末尾に `runpodctl stop pod $RUNPOD_POD_ID` を仕込んでおけば
    docker_args の exit による pod 再起動ループには入らない。
    """
    if cloud_type not in ("SECURE", "COMMUNITY", "ALL"):
        raise ValueError(f"cloud_type must be SECURE/COMMUNITY/ALL, got {cloud_type!r}")
    # RunPod の dockerArgs は GraphQL 文字列にそのまま挿入されるため、改行や引用符
    # を含む bash script を直接渡すと parse error になる。さらに RunPod 側の
    # mutation には実質的な payload サイズ制限があり、生 base64 (~68KB) を送ると
    # `Something went wrong` で全 attempt が失敗する。gzip + base64 にすると
    # 22KB 程度まで縮み、安定して起動できる (2026-05-11 ハマり踏みで確定)。
    #
    # ENTRYPOINT 上書き問題への対処: 公式 runpod/* image は /start.sh で sshd /
    # jupyter を起動する。docker_args で ENTRYPOINT を上書きするとそれが走らず
    # SSH 接続不可になる (runpodctl#170, runpod-python answeroverflow より)。
    # よって以下の順で実行する:
    #   1. /start.sh が存在すれば background で実行 (sshd 起動を維持)
    #   2. その後 onstart を foreground で実行
    # この順なら SSH は早期に開通し、onstart の任意の段階で `tail -F` 接続可能。
    import gzip

    compressed = gzip.compress(onstart_script.encode("utf-8"), compresslevel=9)
    encoded = base64.b64encode(compressed).decode("ascii")
    docker_args = (
        "bash -c '"
        "[ -x /start.sh ] && (/start.sh > /var/log/start.log 2>&1 &); "
        f"echo {encoded} | base64 -d | gunzip | bash"
        "'"
    )
    kwargs: dict[str, Any] = {
        "name": name,
        "image_name": image,
        "gpu_type_id": gpu_type_id,
        "cloud_type": cloud_type,
        "gpu_count": 1,
        "container_disk_in_gb": container_disk_gb,
        "volume_in_gb": 0,
        "volume_mount_path": volume_mount_path,
        "docker_args": docker_args,
        "env": dict(env),
        "support_public_ip": True,
        "start_ssh": True,
        "ports": ports,
    }
    if network_volume_id is not None:
        kwargs["network_volume_id"] = network_volume_id
    if data_center_id is not None:
        kwargs["data_center_id"] = data_center_id
    response = sdk.create_pod(**kwargs)
    if not isinstance(response, Mapping):
        raise RuntimeError(f"unexpected create_pod response: {type(response).__name__}")
    pod_id = response.get("id") or response.get("podId")
    if pod_id is None:
        raise RuntimeError(f"create_pod response missing pod id: keys={list(response)}")
    return str(pod_id)
