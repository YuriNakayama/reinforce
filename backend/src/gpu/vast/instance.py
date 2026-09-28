"""onstart テンプレ render と Vast.ai create_instance ラッパ。

placeholder 置換時に shell injection を防ぐため、値は厳格な regex でバリデーション
してから単純な文字列置換を行う。
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 置換可能な値の許容文字種。shell injection を防ぐため厳しめに固定。
# 英数 + . _ - / : の組合せのみ ( `:` は URL の scheme separator、`/` は path separator)
# branch slug 内に `/` は出ないが repo URL のために : と / は許容する。
_VALID_VALUE = re.compile(r"^[A-Za-z0-9._\-/:]+$")
# CONFIG_ARG は空文字、または `--flag` / `--flag <value>` の並び
# (例: `--smoke`, `--config pipeline/ppo/case2/config.yaml`) のみ許可。
# 空白を含むため _VALID_VALUE とは別の regex でバリデーションする。
_FLAG = r"--[a-z][a-z0-9-]*( [A-Za-z0-9._\-/]+)?"
_VALID_CONFIG_ARG = re.compile(rf"^(|{_FLAG}( {_FLAG})*)$")
# UV_GROUPS は空文字または space 区切りの dependency group 名。
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
    "<CASE_FAMILY>",
    "<CASE_SUBDIR>",
    "<TRAIN_MODULE>",
    "<CONFIG_ARG>",
    "<PREPROCESS_CMD>",
    "<UV_GROUPS>",
)

# ECR の依存焼込み image (infra/runtime/build_and_push.sh で build) を使う場合は
# env で上書きする。
# 例: export REINFORCE_RUNTIME_IMAGE=<acct>.dkr.ecr.ap-northeast-1.amazonaws.com/reinforce-runtime:latest  # noqa: E501
# 未設定時は upstream の pytorch image にフォールバック (毎回 uv sync 必要).
DEFAULT_IMAGE = os.environ.get(
    "REINFORCE_RUNTIME_IMAGE",
    "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
)
DEFAULT_DISK_GB = 40


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
    """空文字 or `--flag [value]` の並びのみ許容。"""
    if not _VALID_CONFIG_ARG.match(value):
        raise TemplateError(
            f"invalid config_arg={value!r}; expected '' or '--flag [value]' "
            "tokens (e.g. '--smoke', '--config <path>') with safe characters only"
        )


def _validate_uv_groups(value: str) -> None:
    """空文字 or space 区切りの group 名のみ許容。"""
    if not _VALID_UV_GROUPS.match(value):
        raise TemplateError(
            f"invalid uv_groups={value!r}; expected '' or space-separated "
            "dependency group names"
        )


def _validate_preprocess_cmd(value: str) -> None:
    """空文字 or `module[ --config <path>]` のみ許容。"""
    if not _VALID_PREPROCESS_CMD.match(value):
        raise TemplateError(
            f"invalid preprocess_cmd={value!r}; expected '' or "
            "'module.path [--config <path>]' with safe characters only"
        )


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
) -> str:
    """テンプレを読み placeholder を置換した script 文字列を返す。

    値は事前に regex でバリデーション。違反したら TemplateError。
    case 関連の値は共有 registry (`gpu.runpod.config.cases`) から来る前提。
    """
    _validate("commit_sha", commit_sha)
    _validate("run_id", run_id)
    _validate("branch", branch)
    _validate("repo_url", repo_url)
    _validate("case", case)
    _validate("case_family", case_family)
    _validate("case_subdir", case_subdir)
    _validate("train_module", train_module)
    _validate_config_arg(config_arg)
    _validate_preprocess_cmd(preprocess_cmd)
    _validate_uv_groups(uv_groups)
    text = template_path.read_text(encoding="utf-8")
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
    }
    for placeholder, value in substitutions.items():
        text = text.replace(placeholder, value)
    # 念のため未置換の placeholder が残っていないか確認
    leftover = [p for p in _TEMPLATE_PLACEHOLDERS if p in text]
    if leftover:
        raise TemplateError(f"unsubstituted placeholders remain: {leftover}")
    return text


def build_env_dict(env: Mapping[str, str]) -> dict[str, str]:
    """`vastai SDK create_instance(env=...)` に渡す dict を組み立てる。

    SDK の低レベル API は `env` を dict として受け取り、JSON に直接埋める。
    変数名のバリデーションだけ行う (shell 注入経路は onstart 側で別途検証)。
    """
    result: dict[str, str] = {}
    for key, value in env.items():
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", key):
            raise TemplateError(f"invalid env var name: {key!r}")
        result[key] = value
    return result


def create_instance(
    sdk: Any,
    offer_id: int,
    *,
    onstart_cmd: str,
    env: Mapping[str, str],
    image: str = DEFAULT_IMAGE,
    disk_gb: int = DEFAULT_DISK_GB,
    label: str,
    runtype: str = "ssh_direc ssh_proxy",
    volume_info: Mapping[str, Any] | None = None,
) -> int:
    """`VastAI().create_instance(...)` を呼び、生成された instance id を返す。

    SSH + direct connection 用の `runtype="ssh_direc ssh_proxy"` をデフォルトとする
    (vastai CLI の `--ssh --direct` と等価)。SDK の応答 dict から `new_contract`
    または `id` を抽出する。
    `volume_info` を渡すと既存 volume の link or 新規作成で /persist mount される.
    """
    kwargs: dict[str, Any] = {
        "image": image,
        "disk": float(disk_gb),
        "env": dict(env),
        "label": label,
        "onstart_cmd": onstart_cmd,
        "runtype": runtype,
    }
    if volume_info is not None:
        kwargs["volume_info"] = dict(volume_info)
    response = sdk.create_instance(offer_id, **kwargs)
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"unexpected create_instance response: {type(response).__name__}"
        )
    instance_id = response.get("new_contract") or response.get("id")
    if instance_id is None:
        raise RuntimeError(
            f"create_instance response missing instance id: keys={list(response)}"
        )
    return int(instance_id)
