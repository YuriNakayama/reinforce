"""RunPod 操作の Typer CLI: train / pull / promote / cost-report / volume。

`dev/runpod` thin wrapper から `python -m gpu.runpod <cmd>` で呼ばれる前提。
SDK との衝突回避のため、SDK は `import runpod as runpod_sdk` の alias 経由
で読む規約 (パッケージ名は `runpod_io`)。
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import runpod as runpod_sdk
import typer
from rich.console import Console

from gpu.runpod.artifacts import cost as cost_mod
from gpu.runpod.artifacts.launch import (
    find_run_dir,
    read_launch_json,
    write_launch_json,
)
from gpu.runpod.artifacts.run_meta import generate_run_id, update_run_json
from gpu.runpod.auth import (
    DEFAULT_AWS_PROFILE,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
    load_dvc_bucket,
    load_git_pat,
    load_runpod_api_key,
)
from gpu.runpod.config.cases import (
    CASE_DEFAULTS,
    DEFAULT_CASE,
    find_best_weights,
)
from gpu.runpod.config.cases import (
    case_defaults as _case_defaults,
)
from gpu.runpod.config.cases import (
    case_family as _case_family,
)
from gpu.runpod.config.cases import (
    case_subdir as _case_subdir,
)
from gpu.runpod.config.cases import (
    runs_root_for as _runs_root_for,
)
from gpu.runpod.execution.watcher import (
    DEFAULT_MAX_WAIT,
    DEFAULT_POLL_INTERVAL,
    watch_pod,
)
from gpu.runpod.git import (
    git_current_branch as _git_current_branch,
)
from gpu.runpod.git import (
    git_remote_url as _git_remote_url,
)
from gpu.runpod.git import (
    repo_root as _repo_root,
)
from gpu.runpod.git import (
    verify_commit_pushed as _verify_commit_pushed,
)
from gpu.runpod.runpod.instance import (
    DEFAULT_DISK_GB,
    DEFAULT_IMAGE,
    build_env_dict,
    create_pod,
    render_onstart,
)
from gpu.runpod.runpod.offers import (
    DEFAULT_GPU_NAMES,
    Offer,
    pick_offer,
    search_offers,
)
from gpu.runpod.runpod.volumes import (
    KNOWN_DATA_CENTERS,
    Volume,
    find_volume_by_name,
    list_volumes,
    render_volume_offers,
    search_volume_offers,
)
from gpu.runpod.runpod.volumes import (
    create_volume as create_volume_fn,
)

app = typer.Typer(
    add_completion=False,
    help="RunPod-driven GPU training basis for reinforce.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_CLOUD_TYPE = "SECURE"
DEFAULT_COST_LIMIT_USD = 1.5
ESTIMATED_RUNTIME_HOURS = 0.5
DEFAULT_VOLUME_NAME = "reinforce"
DEFAULT_MOUNT_PATH = "/persist"
DEFAULT_COST_REPORT_DIR = Path("docs/experiment")

# gpu.runpod.config.cases.CASE_DEFAULTS を参照。
# 全基盤 (RunPod / Vast.ai / Kaggle) で同じ train_module / config_arg /
# preprocess_cmd / uv_groups / canonical_weights を共有する。
DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "execution" / "onstart.sh.tmpl"
)


def _run_preflight_smoke(case: str) -> None:
    """Run the case's CPU smoke train + import sanity before renting a pod.

    Why: every RunPod attempt costs $0.20–$1.50 (varies by case) and ties up
    a GPU node. A single broken import or YAML typo can burn the whole run
    in onstart. Catching these locally — fast, free — is the cheapest
    insurance against onstart failures that would otherwise surface only on the pod.

    Behavior:
      * Assert ``CASE_DEFAULTS[case].train_module`` is importable from the
        backend venv (catches typos and import-time errors without paying
        RunPod boot time).
      * Invoke that module with ``--smoke`` on CPU and assert it succeeds
        within 60s (every case's train.py must honour ``--smoke`` — see
        ``.claude/rules/backend/pipeline.md``).

    Failure raises ``typer.Exit(1)`` so the gate blocks ``runpod_sdk`` calls.
    """
    repo_root = _repo_root()
    backend_dir = repo_root / "backend"
    case_default = _case_defaults(case)
    train_module = case_default["train_module"]

    console.print(f"[cyan]preflight:[/] importing {train_module} ...")
    import_check = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(backend_dir),
            "python",
            "-c",
            f"import {train_module}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if import_check.returncode != 0:
        console.print(
            f"[red]preflight:[/] cannot import {train_module}\n"
            f"{import_check.stderr.strip()}"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[cyan]preflight:[/] running {train_module} --smoke on CPU (timeout 60s)..."
    )
    smoke = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(backend_dir),
            "python",
            "-m",
            train_module,
            "--smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if smoke.returncode != 0:
        tail = "\n".join(
            smoke.stdout.splitlines()[-15:] + smoke.stderr.splitlines()[-15:]
        )
        console.print(
            f"[red]preflight:[/] smoke train failed; refusing to launch.\n{tail}"
        )
        raise typer.Exit(code=1)
    console.print("[green]preflight:[/] smoke train OK.")


def _build_sdk(api_key: str) -> Any:
    """RunPod SDK を初期化して返す (module-level state なので module 自体を返す)。"""
    runpod_sdk.api_key = api_key
    return runpod_sdk


def _pull_from_s3(
    repo_root: Path, relative: Path, run_id: str, aws_profile: str
) -> None:
    """`s3://.../runpod_artifacts/<run_id>/` から成果物を `repo_root/relative/` に
    download する。pull の DVC 経路 fallback。
    """
    from gpu.runpod.execution import progress as progress_mod

    target_dir = repo_root / relative
    target_dir.mkdir(parents=True, exist_ok=True)
    files = progress_mod.list_artifacts(run_id, profile=aws_profile)
    if not files:
        console.print(
            f"[red]no artifacts in s3://.../runpod_artifacts/{run_id}/[/] — "
            "onstart did not reach early_artifact_upload step"
        )
        raise typer.Exit(code=1)
    downloaded: list[str] = []
    for fname in files:
        dest = target_dir / fname
        ok = progress_mod.download_artifact(run_id, fname, dest, profile=aws_profile)
        if ok:
            downloaded.append(fname)
            console.print(f"[green]pulled[/] {fname} -> {dest}")
        else:
            console.print(f"[yellow]skip[/] {fname} (download failed)")
    if not downloaded:
        raise typer.Exit(code=1)
    console.print(
        "\n[cyan]NOTE:[/] artifacts pulled from S3 fallback (not DVC). "
        "Re-run `dev/runpod pull --from dvc` once the run's .dvc stub is on origin."
    )


@app.command()
def train(
    commit_sha: str = typer.Argument(..., help="must be pushed to origin"),
    case: str = typer.Option(
        DEFAULT_CASE,
        "--case",
        help=f"case to train (one of: {sorted(CASE_DEFAULTS)})",
    ),
    cloud_type: str = typer.Option(
        DEFAULT_CLOUD_TYPE,
        "--cloud-type",
        help="SECURE / COMMUNITY / ALL",
    ),
    gpu_names: list[str] = typer.Option(
        list(DEFAULT_GPU_NAMES),
        "--gpu-name",
        help="GPU type ids to consider (repeatable)",
    ),
    max_dph: float = typer.Option(2.0, "--max-dph"),
    seed: int = typer.Option(0, "--seed"),
    label: str | None = typer.Option(None, "--label"),
    cost_limit_usd: float = typer.Option(
        DEFAULT_COST_LIMIT_USD,
        "--cost-limit",
        help="confirm before launching if estimated cost exceeds this",
    ),
    aws_profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--aws-profile"),
    image: str = typer.Option(DEFAULT_IMAGE, "--image"),
    disk_gb: int = typer.Option(DEFAULT_DISK_GB, "--disk-gb"),
    volume_id: str | None = typer.Option(
        None,
        "--volume-id",
        help="既存 network volume id (Secure Cloud のみ attach 可)",
    ),
    volume_name: str = typer.Option(
        DEFAULT_VOLUME_NAME,
        "--volume-name",
        help="自動再利用用の volume name",
    ),
    mount_path: str = typer.Option(DEFAULT_MOUNT_PATH, "--mount-path"),
    auto_create_volume: bool = typer.Option(
        False,
        "--auto-create-volume",
        help="一致 volume が無ければ --data-center-id で新規作成",
    ),
    volume_size_gb: int = typer.Option(300, "--volume-size"),
    data_center_id: str | None = typer.Option(
        None,
        "--data-center-id",
        help="pod / 新規 volume の DC (例: US-KS-2)",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        help="pod 起動後に終了まで poll してデスクトップ通知を発火する。"
        " 完了/失敗のいずれでも 1 度だけ通知。",
    ),
    poll_interval: float = typer.Option(
        DEFAULT_POLL_INTERVAL,
        "--poll-interval",
        help="--watch 時の S3 marker ポーリング間隔 (秒)",
    ),
    max_wait_seconds: float = typer.Option(
        DEFAULT_MAX_WAIT,
        "--max-wait",
        help="--watch のタイムアウト (秒)。デフォルトは onstart の 2h ガードと一致",
    ),
    preprocess_only: bool = typer.Option(
        False,
        "--preprocess-only",
        help="train を実行せず、case に登録された preprocess_cmd だけ走らせて終了する。",
    ),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="過去 run_id。その run の .dvc を pull し、checkpoint_latest.* を"
        " 新 run dir に置いて train.py に resume させる (.dvc が origin に必要)。",
    ),
    auto_pick_first_offer: bool = typer.Option(
        False,
        "--auto-pick-first-offer",
        help="対話 prompt をスキップし最初の offer を自動選択 / cost limit 内なら "
        "確認なしで起動。loop / cron 経由の自動運転用。",
    ),
    skip_smoke: bool = typer.Option(
        False,
        "--skip-smoke",
        help="ローカル smoke test (train_module の import + `--smoke` CPU 実行) を"
        " スキップする。本来は失敗時に RunPod 起動を block する gate。",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="smoke test と git/aws 検証だけ走らせて pod は起動しない。"
        " loop / CI から起動前検証のみしたいとき。",
    ),
) -> None:
    """Search RunPod GPU offers, pick one, and launch training for a given commit."""
    defaults = _case_defaults(case)

    if not skip_smoke:
        _run_preflight_smoke(case)

    try:
        repo_url = _git_remote_url()
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            "could not resolve `git remote get-url origin`"
        ) from exc
    _verify_commit_pushed(commit_sha)
    branch = _git_current_branch()

    if dry_run:
        console.print(
            "[green]dry-run OK:[/] smoke + git checks passed; skipping pod launch."
        )
        raise typer.Exit(code=0)

    try:
        aws_creds = load_aws_creds(profile=aws_profile)
        runpod_api_key = load_runpod_api_key()
        dvc_bucket = load_dvc_bucket()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc

    sdk = _build_sdk(runpod_api_key)

    # Volume 解決 (Secure Cloud + Pod 作成時 attach のみ).
    # 1) --volume-id 明示 → そのまま attach
    # 2) --volume-name 一致 → 既存 network volume を再利用
    # 3) --auto-create-volume + --data-center-id → 新規作成
    # Community Cloud のときは attach 不可。
    volume_id_resolved: str | None = None
    if cloud_type == "COMMUNITY" and (volume_id or auto_create_volume):
        console.print(
            "[yellow]volume:[/] Community Cloud は network volume 不可。"
            " volume 設定は無視されます。"
        )
    elif volume_id is not None:
        volume_id_resolved = volume_id
        console.print(
            f"[cyan]volume:[/] linking id={volume_id_resolved} at {mount_path}"
        )
    else:
        existing = find_volume_by_name(list_volumes(_volume_sdk()), volume_name)
        if existing is not None:
            volume_id_resolved = existing.id
            console.print(
                f"[cyan]volume:[/] reusing {existing.name!r} id={existing.id} "
                f"size={existing.size_gb}GB at {mount_path}"
            )
            if data_center_id is None:
                data_center_id = existing.data_center_id
        elif auto_create_volume:
            if data_center_id is None:
                console.print(
                    "[red]--auto-create-volume requires --data-center-id "
                    f"(known: {KNOWN_DATA_CENTERS})[/]"
                )
                raise typer.Exit(code=1)
            new_id = create_volume_fn(
                _volume_sdk(),
                name=volume_name,
                size_gb=volume_size_gb,
                data_center_id=data_center_id,
            )
            volume_id_resolved = new_id
            console.print(
                f"[green]created network volume:[/] id={new_id} "
                f"name={volume_name!r} dc={data_center_id}"
            )
        else:
            console.print(
                "[yellow]volume: not configured (uv cache 等は永続化されない)。[/]\n"
                "  --auto-create-volume + --data-center-id で初回作成、"
                "以降は --volume-name で自動再利用。"
            )

    # `*_smoke` case は RunPod 基盤の E2E smoke 専用なので、安価な GPU に絞り
    # cost_limit / max_dph も縮める。ユーザーが明示的に上書きしている場合は
    # その値を尊重する (デフォルト値とのみ比較)。
    if case.endswith("_smoke"):
        if list(gpu_names) == list(DEFAULT_GPU_NAMES):
            # SECURE 候補。`pick_offer` は dph 昇順で選ぶので
            # A4000 ($0.25) → 4000 Ada ($0.26) → A5000 ($0.27) → 3090 ($0.46)
            # → A6000 ($0.49) の順で fallback する。
            # 4090 は「pod RUNNING だが bash 開始すら届かない (markers ゼロ)」
            # 失敗を再現したので除外。A6000 は 4090 ガチャ回避策として実績あり。
            gpu_names = [
                "NVIDIA GeForce RTX 3090",
                "NVIDIA RTX A4000",
                "NVIDIA RTX A5000",
                "NVIDIA RTX A6000",
                "NVIDIA RTX 4000 Ada Generation",
            ]
            console.print(
                "[cyan]smoke case:[/] GPU を A4000/4000Ada/A5000/3090/A6000 に絞り込み"
                " (4090 はノードガチャ trap を引いたため除外)"
            )
        if max_dph == 2.0:
            max_dph = 0.6
            console.print("[cyan]smoke case:[/] --max-dph を 0.6 に縮小")
        if cost_limit_usd == DEFAULT_COST_LIMIT_USD:
            cost_limit_usd = 0.40
            console.print("[cyan]smoke case:[/] --cost-limit を $0.40 に縮小")

    offers = search_offers(
        sdk,
        gpu_names=gpu_names,
        cloud_type=cloud_type,
        max_dph=max_dph,
    )
    if not offers:
        console.print(
            "[red]No offers matched.[/] Loosen --max-dph, broaden --gpu-name, or "
            "switch --cloud-type=ALL."
        )
        raise typer.Exit(code=1)
    if auto_pick_first_offer:
        chosen = offers[0]
        console.print(
            f"[cyan]auto-pick-first-offer:[/] {chosen.gpu_type_id} "
            f"({chosen.cloud_type}) @ ${chosen.dph_total:.3f}/h"
        )
    else:
        chosen = pick_offer(offers, console=console)
    estimated = chosen.dph_total * ESTIMATED_RUNTIME_HOURS
    console.print(
        f"\nSelected {chosen.gpu_type_id} ({chosen.cloud_type}) "
        f"@ ${chosen.dph_total:.3f}/h "
        f"(est. ${estimated:.3f} for {ESTIMATED_RUNTIME_HOURS}h)"
    )
    if estimated > cost_limit_usd:
        if auto_pick_first_offer:
            console.print(
                f"[red]auto-pick-first-offer:[/] estimated ${estimated:.3f} "
                f"exceeds limit ${cost_limit_usd:.2f}. Aborting (raise --cost-limit)."
            )
            raise typer.Exit(code=1)
        if not typer.confirm(
            f"Estimated cost ${estimated:.3f} exceeds limit ${cost_limit_usd:.2f}. "
            "Proceed anyway?",
            default=False,
        ):
            raise typer.Exit(code=1)

    run_id = generate_run_id(branch, commit_sha, seed)
    if preprocess_only and not defaults["preprocess_cmd"]:
        raise typer.BadParameter(
            f"case={case!r} には preprocess_cmd が定義されていないため "
            "--preprocess-only は使用できません。"
        )
    onstart_cmd = render_onstart(
        DEFAULT_TEMPLATE_PATH,
        commit_sha=commit_sha,
        run_id=run_id,
        branch=branch,
        repo_url=repo_url,
        case=case,
        case_family=_case_family(case),
        case_subdir=_case_subdir(case),
        train_module="" if preprocess_only else defaults["train_module"],
        config_arg=defaults["config_arg"],
        preprocess_cmd=defaults["preprocess_cmd"],
        uv_groups=defaults["uv_groups"],
        resume_from=resume_from or "",
        mode="oneshot",
    )
    # snapshot は JSON 文字列 (引用符を含む) なので、RunPod の GraphQL env エンコード
    # (`f'value: "{v}"'`) で壊れる。base64 化して安全な ASCII のみにする。
    # train.py 側 (`pipeline/<family>/case<N>/train.py`) は env から
    # JSON 文字列を期待するので、onstart で base64 -d して再注入する。
    snapshot_json = json.dumps(chosen.to_snapshot())
    snapshot_b64 = base64.b64encode(snapshot_json.encode("utf-8")).decode("ascii")
    env_dict: dict[str, str] = {
        "AWS_ACCESS_KEY_ID": aws_creds.access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_creds.secret_access_key,
        "AWS_DEFAULT_REGION": aws_creds.region,
        "RUNPOD_API_KEY": runpod_api_key,
        # onstart の S3 marker / artifacts / DVC push 先。未設定だと `s3:///...`
        # になり全 upload が無言で失敗するので launch 前に必須化している。
        "REINFORCE_DVC_BUCKET": dvc_bucket,
        "REINFORCE_RUN_ID": run_id,
        "REINFORCE_GIT_SHA": commit_sha,
        "REINFORCE_GIT_BRANCH": branch,
        "REINFORCE_CASE": case,
        # case_family / case_subdir を onstart shell に渡す。registry key
        # `<family>_case<N>[_variant]` から family / subdir を解決し、variant も
        # 同じ data/output/models/<family>/<caseN>/runs/ に persist / DVC push する。
        "REINFORCE_CASE_FAMILY": _case_family(case),
        "REINFORCE_CASE_SUBDIR": _case_subdir(case),
        "REINFORCE_RUNPOD_OFFER_SNAPSHOT_B64": snapshot_b64,
    }
    # GIT_PAT は onstart の git push (.dvc メタコミット) で必要。.env か shell env
    # から取得し、無ければ警告して continue (push はスキップされる)。
    git_pat = load_git_pat()
    if git_pat:
        env_dict["GIT_PAT"] = git_pat
    else:
        console.print(
            "[yellow]WARNING:[/] GIT_PAT not set — onstart の git push (.dvc 永続化) "
            "は skip されます。`backend/.env` に GIT_PAT を追加してください。"
        )
    env = build_env_dict(env_dict)
    # 在庫切れ (QueryError "no longer any instances available") に当たることが
    # あるため、search_offers の cheapest-first 順で fallback を試す。chosen は
    # pick_offer の戻り (= 通常 offers[0])、それ以外を 2 番手以降として順に試す。
    from runpod.error import QueryError as _RunPodQueryError

    fallback_chain: list[Any] = [chosen]
    for o in offers:
        if o is not chosen:
            fallback_chain.append(o)
    pod_id: str | None = None
    last_err: Exception | None = None
    used_offer = chosen
    for attempt_idx, offer_try in enumerate(fallback_chain, start=1):
        try:
            pod_id = create_pod(
                sdk,
                name=label or run_id,
                gpu_type_id=offer_try.gpu_type_id,
                cloud_type=offer_try.cloud_type,
                onstart_script=onstart_cmd,
                env=env,
                image=image,
                container_disk_gb=disk_gb,
                network_volume_id=volume_id_resolved,
                volume_mount_path=mount_path,
                data_center_id=data_center_id,
            )
            used_offer = offer_try
            if attempt_idx > 1:
                console.print(
                    f"[green]launched on fallback #{attempt_idx}:[/] "
                    f"{offer_try.gpu_type_id} ({offer_try.cloud_type})"
                )
            break
        except _RunPodQueryError as exc:
            last_err = exc
            console.print(
                f"[yellow]offer {offer_try.gpu_type_id} unavailable "
                f"(attempt {attempt_idx}/{len(fallback_chain)}):[/] {exc}"
            )
            continue
    if pod_id is None:
        console.print(
            "[red]all offers exhausted; no pod was created.[/] "
            "Try later, broaden --gpu-name, or switch --cloud-type=ALL."
        )
        if last_err is not None:
            raise last_err
        raise typer.Exit(code=1)
    chosen = used_offer
    run_dir_local = find_run_dir(_repo_root(), run_id, case)
    write_launch_json(
        run_dir_local,
        run_id=run_id,
        pod_id=pod_id,
        commit_sha=commit_sha,
        branch=branch,
        case=case,
        cloud_type=chosen.cloud_type,
        gpu_type_id=chosen.gpu_type_id,
        dph_total=chosen.dph_total,
        data_center_id=data_center_id,
        mode="oneshot",
    )
    console.print(
        f"\n[green]Pod launched![/] id=[bold]{pod_id}[/] run_id={run_id} case={case}"
    )
    console.print(
        f"  Inspect: [cyan]dev/runpod status {run_id} --case {case}[/]  "
        f"Logs: [cyan]dev/runpod logs {run_id} --case {case}[/]"
    )
    console.print(
        f"  After completion: [cyan]dev/runpod pull {run_id} --case {case}[/] then "
        f"[cyan]dev/runpod promote {run_id} --case {case}[/] to adopt"
    )

    if watch:
        console.print(
            f"\n[bold]watching pod[/] {pod_id} until completion "
            f"(poll={poll_interval:.0f}s, max_wait={max_wait_seconds:.0f}s)"
        )
        result = watch_pod(
            run_id=run_id,
            pod_id=pod_id,
            aws_profile=aws_profile,
            poll_interval=poll_interval,
            max_wait_seconds=max_wait_seconds,
        )
        color = "green" if result.is_success else "red"
        console.print(
            f"\n[{color}]watch result:[/] outcome={result.outcome} "
            f"last_step={result.last_step!r} "
            f"elapsed={result.elapsed_seconds:.0f}s"
        )
        console.print(f"  detail: {result.detail}")
        console.print(
            f"\n  Run [cyan]dev/runpod logs {run_id} --case {case}[/] "
            "to see the full marker timeline."
        )
        if not result.is_success:
            # If we hit stalled / failure while the pod is still RUNNING, try
            # to grab /var/log/onstart.log over SSH BEFORE terminating: the
            # bash cleanup_destroy trap may not run when the pod is killed
            # externally, leaving only the marker timeline behind. Best-effort.
            if result.outcome in {"stalled", "failure"}:
                from gpu.runpod.execution.rescue import rescue_onstart_log

                rescue = rescue_onstart_log(
                    sdk, pod_id, run_id, aws_profile=aws_profile
                )
                if rescue.uploaded:
                    console.print(
                        f"[green]rescue:[/] onstart log saved to {rescue.s3_uri} "
                        f"({rescue.bytes_uploaded} bytes)"
                    )
                else:
                    console.print(f"[dim]rescue skipped:[/] {rescue.error}")

            # Defense in depth: ensure the pod is terminated even if onstart
            # trap missed it. Idempotent — already-removed pods are accepted.
            from gpu.runpod.runpod.cleanup import terminate_pod

            cleanup_result = terminate_pod(sdk, pod_id)
            if cleanup_result.terminated:
                console.print(f"[dim]cleanup: pod {pod_id} terminated[/]")
            elif cleanup_result.error:
                console.print(f"[yellow]cleanup warning:[/] {cleanup_result.error}")
            raise typer.Exit(code=1)


@app.command("dev")
def dev_cmd(
    commit_sha: str = typer.Argument(..., help="must be pushed to origin"),
    case: str = typer.Option(
        DEFAULT_CASE,
        "--case",
        help=f"case to base the dev pod on (one of: {sorted(CASE_DEFAULTS)})",
    ),
    cloud_type: str = typer.Option(
        DEFAULT_CLOUD_TYPE,
        "--cloud-type",
        help="SECURE / COMMUNITY / ALL",
    ),
    gpu_names: list[str] = typer.Option(
        list(DEFAULT_GPU_NAMES),
        "--gpu-name",
        help="GPU type ids to consider (repeatable)",
    ),
    max_dph: float = typer.Option(2.0, "--max-dph"),
    label: str | None = typer.Option(None, "--label"),
    cost_limit_usd: float = typer.Option(
        DEFAULT_COST_LIMIT_USD,
        "--cost-limit",
        help="confirm before launching if estimated cost exceeds this",
    ),
    aws_profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--aws-profile"),
    image: str = typer.Option(DEFAULT_IMAGE, "--image"),
    disk_gb: int = typer.Option(DEFAULT_DISK_GB, "--disk-gb"),
    volume_id: str | None = typer.Option(
        None,
        "--volume-id",
        help="既存 network volume id (Secure Cloud のみ attach 可)",
    ),
    volume_name: str = typer.Option(DEFAULT_VOLUME_NAME, "--volume-name"),
    mount_path: str = typer.Option(DEFAULT_MOUNT_PATH, "--mount-path"),
    data_center_id: str | None = typer.Option(None, "--data-center-id"),
    skip_smoke: bool = typer.Option(
        False,
        "--skip-smoke",
        help=(
            "ローカル smoke test をスキップ。"
            "interactive 用途は import 検査だけで十分なことが多い"
        ),
    ),
) -> None:
    """Launch an interactive RunPod pod for development / debugging.

    train モードと異なり:
      * onstart は uv sync (+ resume 用 DVC pull / preprocess) まで実行して sleep infinity で待機
      * 自動 terminate / 8h timeout guard / EXIT trap を全てスキップ
      * 終了は `dev/runpod destroy <run_id>` で明示的に行う
      * SSH 接続コマンドを launch 直後に表示する (`dev/runpod ssh <run_id>` も使える)
    """
    defaults = _case_defaults(case)

    if not skip_smoke:
        _run_preflight_smoke(case)

    try:
        repo_url = _git_remote_url()
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            "could not resolve `git remote get-url origin`"
        ) from exc
    _verify_commit_pushed(commit_sha)
    branch = _git_current_branch()

    try:
        aws_creds = load_aws_creds(profile=aws_profile)
        runpod_api_key = load_runpod_api_key()
        dvc_bucket = load_dvc_bucket()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc

    sdk = _build_sdk(runpod_api_key)

    # Volume 解決: train とほぼ同じだが auto-create は対話運用にそぐわないので未対応。
    volume_id_resolved: str | None = None
    if cloud_type == "COMMUNITY" and volume_id:
        console.print(
            "[yellow]volume:[/] Community Cloud は network volume 不可。"
            "設定は無視されます。"
        )
    elif volume_id is not None:
        volume_id_resolved = volume_id
        console.print(
            f"[cyan]volume:[/] linking id={volume_id_resolved} at {mount_path}"
        )
    else:
        existing = find_volume_by_name(list_volumes(_volume_sdk()), volume_name)
        if existing is not None:
            volume_id_resolved = existing.id
            console.print(
                f"[cyan]volume:[/] reusing {existing.name!r} id={existing.id}"
                f" size={existing.size_gb}GB at {mount_path}"
            )
            if data_center_id is None:
                data_center_id = existing.data_center_id

    offers = search_offers(
        sdk,
        gpu_names=gpu_names,
        cloud_type=cloud_type,
        max_dph=max_dph,
    )
    if not offers:
        console.print("[red]No offers matched.[/]")
        raise typer.Exit(code=1)
    chosen = pick_offer(offers, console=console)
    # interactive は長時間保持しがちなので 1h ベースで見積もり警告。
    estimated_1h = chosen.dph_total
    console.print(
        f"\nSelected {chosen.gpu_type_id} ({chosen.cloud_type}) "
        f"@ ${chosen.dph_total:.3f}/h (est. ${estimated_1h:.3f} per hour held)"
    )
    if estimated_1h > cost_limit_usd:
        if not typer.confirm(
            f"Per-hour cost ${estimated_1h:.3f} exceeds limit ${cost_limit_usd:.2f}. "
            "Proceed anyway? (interactive pods are billed until destroy)",
            default=False,
        ):
            raise typer.Exit(code=1)

    run_id = generate_run_id(branch, commit_sha, seed=0)
    onstart_cmd = render_onstart(
        DEFAULT_TEMPLATE_PATH,
        commit_sha=commit_sha,
        run_id=run_id,
        branch=branch,
        repo_url=repo_url,
        case=case,
        case_family=_case_family(case),
        case_subdir=_case_subdir(case),
        train_module=defaults["train_module"],
        config_arg=defaults["config_arg"],
        preprocess_cmd=defaults["preprocess_cmd"],
        uv_groups=defaults["uv_groups"],
        mode="interactive",
    )
    snapshot_json = json.dumps(chosen.to_snapshot())
    snapshot_b64 = base64.b64encode(snapshot_json.encode("utf-8")).decode("ascii")
    env_dict: dict[str, str] = {
        "AWS_ACCESS_KEY_ID": aws_creds.access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_creds.secret_access_key,
        "AWS_DEFAULT_REGION": aws_creds.region,
        "RUNPOD_API_KEY": runpod_api_key,
        # onstart の S3 marker / artifacts / DVC push 先。未設定だと `s3:///...`
        # になり全 upload が無言で失敗するので launch 前に必須化している。
        "REINFORCE_DVC_BUCKET": dvc_bucket,
        "REINFORCE_RUN_ID": run_id,
        "REINFORCE_GIT_SHA": commit_sha,
        "REINFORCE_GIT_BRANCH": branch,
        "REINFORCE_CASE": case,
        "REINFORCE_RUNPOD_OFFER_SNAPSHOT_B64": snapshot_b64,
        "REINFORCE_RUNPOD_MODE": "interactive",
    }
    git_pat = load_git_pat()
    if git_pat:
        env_dict["GIT_PAT"] = git_pat
    env = build_env_dict(env_dict)

    # train_cmd と同じ fallback chain: 在庫切れ (QueryError "no longer any
    # instances available") に当たったら cheapest-first 順で次の offer を試す。
    from runpod.error import QueryError as _RunPodQueryError

    fallback_chain: list[Any] = [chosen]
    for o in offers:
        if o is not chosen:
            fallback_chain.append(o)
    pod_id: str | None = None
    last_err: Exception | None = None
    used_offer = chosen
    for attempt_idx, offer_try in enumerate(fallback_chain, start=1):
        try:
            pod_id = create_pod(
                sdk,
                name=label or f"dev-{run_id}",
                gpu_type_id=offer_try.gpu_type_id,
                cloud_type=offer_try.cloud_type,
                onstart_script=onstart_cmd,
                env=env,
                image=image,
                container_disk_gb=disk_gb,
                network_volume_id=volume_id_resolved,
                volume_mount_path=mount_path,
                data_center_id=data_center_id,
            )
            used_offer = offer_try
            if attempt_idx > 1:
                console.print(
                    f"[green]launched on fallback #{attempt_idx}:[/] "
                    f"{offer_try.gpu_type_id} ({offer_try.cloud_type})"
                )
            break
        except _RunPodQueryError as exc:
            last_err = exc
            console.print(
                f"[yellow]offer {offer_try.gpu_type_id} unavailable "
                f"(attempt {attempt_idx}/{len(fallback_chain)}):[/] {exc}"
            )
            continue
    if pod_id is None:
        console.print(
            "[red]all offers exhausted; no pod was created.[/] "
            "Try later, broaden --gpu-name, or switch --cloud-type=ALL."
        )
        if last_err is not None:
            raise last_err
        raise typer.Exit(code=1)
    chosen = used_offer
    run_dir_local = find_run_dir(_repo_root(), run_id, case)
    write_launch_json(
        run_dir_local,
        run_id=run_id,
        pod_id=pod_id,
        commit_sha=commit_sha,
        branch=branch,
        case=case,
        cloud_type=chosen.cloud_type,
        gpu_type_id=chosen.gpu_type_id,
        dph_total=chosen.dph_total,
        data_center_id=data_center_id,
        mode="interactive",
    )
    console.print(
        f"\n[green]Interactive pod launched![/] id=[bold]{pod_id}[/] "
        f"run_id={run_id} case={case}"
    )
    console.print(
        "[yellow]Billed continuously until destroyed.[/]"
        " Pod は明示的に止めるまで課金され続けます。"
    )
    console.print(
        "\n[bold]Next steps[/]:\n"
        f"  Wait for setup:  [cyan]dev/runpod status {run_id} --case {case}[/]"
        f" (look for 50_interactive_ready)\n"
        f"  Open SSH:        [cyan]dev/runpod ssh {run_id} --case {case}[/]\n"
        f"  Sync code:       [cyan]dev/runpod sync {run_id} --case {case} --push[/]\n"
        f"  Destroy when done: [cyan]dev/runpod destroy {run_id} --case {case}[/]"
    )


@app.command()
def pull(
    run_id: str = typer.Argument(..., help="run_id from runpod train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    source: str = typer.Option(
        "auto",
        "--from",
        help="auto / dvc / s3 — auto は dvc を試して失敗時 s3 にフォールバック",
    ),
    aws_profile: str = typer.Option(
        DEFAULT_AWS_PROFILE, "--aws-profile", help="S3 fallback 用 AWS profile"
    ),
) -> None:
    """`dvc pull` で run dir を取得し run.json を表示する。

    `--from s3` または DVC 経路失敗時に S3 artifacts prefix
    (`s3://.../runpod_artifacts/<run_id>/`) から best.* / checkpoint_latest.* 等を
    直接 download する。
    onstart の dvc push 完了前に pod kill された run でも、artifacts prefix に
    ある成果物は救出可能。
    """
    if source not in ("auto", "dvc", "s3"):
        raise typer.BadParameter(f"--from must be auto/dvc/s3, got {source!r}")
    if runs_root is None:
        runs_root = _runs_root_for(case)
    relative = runs_root / run_id
    repo_root = _repo_root()
    dvc_meta = repo_root / f"{relative}.dvc"

    if source == "s3":
        _pull_from_s3(repo_root, relative, run_id, aws_profile)
        return

    if not dvc_meta.is_file():
        if source == "dvc":
            console.print(
                f"[red]missing:[/] {dvc_meta.relative_to(repo_root)} "
                "— DVC meta not committed to origin"
            )
            raise typer.Exit(code=1)
        # auto: fall back to s3
        console.print(
            f"[yellow]missing:[/] {dvc_meta.relative_to(repo_root)} "
            "— falling back to s3 artifacts prefix"
        )
        _pull_from_s3(repo_root, relative, run_id, aws_profile)
        return

    cmd = ["uv", "run", "--project", "backend", "dvc", "pull", str(relative)]
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    result = subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        if source == "dvc":
            raise typer.Exit(code=result.returncode)
        console.print("[yellow]dvc pull failed; falling back to s3 artifacts prefix[/]")
        _pull_from_s3(repo_root, relative, run_id, aws_profile)
        return
    run_json_path = _repo_root() / relative / "run.json"
    if not run_json_path.is_file():
        console.print(
            f"[yellow]warning:[/] {run_json_path} not found after pull "
            "(check run_id and DVC remote)"
        )
        raise typer.Exit(code=1)
    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    console.print_json(data=payload)
    if payload.get("status") != "pushed":
        console.print(
            f"[yellow]warning:[/] status={payload.get('status')!r}; "
            "expected 'pushed'. Verify the run completed."
        )


@app.command()
def promote(
    run_id: str = typer.Argument(..., help="run_id to promote"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    eval_results_path: Path | None = typer.Option(
        None, "--eval-results", help="optional JSON file with local eval"
    ),
) -> None:
    """Run dir の best.* を canonical weights にコピーし status=adopted。"""
    defaults = _case_defaults(case)
    if runs_root is None:
        runs_root = _runs_root_for(case)
    if canonical is None:
        canonical = Path(defaults["canonical_weights"])
    repo_root = _repo_root()
    run_dir = repo_root / runs_root / run_id
    src = find_best_weights(run_dir, prefer_suffix=canonical.suffix)
    if src is None:
        console.print(f"[red]not found:[/] {run_dir}/best.*")
        raise typer.Exit(code=1)
    if src.suffix != canonical.suffix:
        console.print(
            f"[yellow]warning:[/] {src.name} suffix differs from canonical "
            f"{canonical.name}; copying anyway"
        )
    dst = repo_root / canonical
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    console.print(f"[green]copied[/] {src} -> {dst}")

    # canonical weights are tracked with a standalone `.dvc` stub (there is no
    # dvc.yaml pipeline in this repo), so `dvc add` both creates and updates it.
    dvc_track_cmd = [
        "uv",
        "run",
        "--project",
        "backend",
        "dvc",
        "add",
        str(canonical),
    ]
    console.print(f"[dim]$ {' '.join(dvc_track_cmd)}[/]")
    subprocess.run(dvc_track_cmd, cwd=str(repo_root), check=False)

    patch: dict[str, Any] = {"status": "adopted"}
    if eval_results_path is not None:
        patch["local_eval_results"] = json.loads(
            eval_results_path.read_text(encoding="utf-8")
        )
    update_run_json(run_dir, **patch)
    console.print(f"[green]run.json status -> adopted[/] in {run_dir}")

    dvc_add_cmd = [
        "uv",
        "run",
        "--project",
        "backend",
        "dvc",
        "add",
        str(runs_root / run_id),
    ]
    console.print(f"[dim]$ {' '.join(dvc_add_cmd)}[/]")
    subprocess.run(dvc_add_cmd, cwd=str(repo_root), check=False)

    status_result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    console.print("\n[bold]git status:[/]")
    console.print(status_result.stdout or "(clean)")
    console.print("\nNext: review the diff, then `git commit` and open a PR to main.")


@app.command("cost-report")
def cost_report_cmd(
    month: str | None = typer.Option(
        None, "--month", help="filter by YYYY-MM (default: current month)"
    ),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    output_dir: Path = typer.Option(DEFAULT_COST_REPORT_DIR, "--output-dir"),
) -> None:
    """run.json を集計して RunPod 月次コストレポートを生成する。"""
    if runs_root is None:
        runs_root = _runs_root_for(case)
    if month is None:
        month = datetime.now(UTC).strftime("%Y-%m")
    cost_mod.parse_month(month)
    repo_root = _repo_root()
    report = cost_mod.aggregate_runs(repo_root / runs_root, month=month)
    md = cost_mod.render_markdown(report)
    out_dir = repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"runpod_cost_report_{month}.md"
    out_path.write_text(md, encoding="utf-8")
    console.print(md)
    console.print(f"\n[green]written:[/] {out_path}")


def _ensure_runpod_key() -> None:
    """RUNPOD_API_KEY を SDK module-level state に読み込む。"""
    try:
        runpod_sdk.api_key = load_runpod_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc


@app.command("stock")
def stock_cmd(
    min_memory_gb: int = typer.Option(16, "--min-memory-gb"),
    max_dph: float = typer.Option(
        10.0, "--max-dph", help="cap on uninterruptable price"
    ),
    gpu_count: int = typer.Option(
        1, "--gpu-count", help="quote price/stock for this count"
    ),
    include_no_stock: bool = typer.Option(
        False, "--include-no-stock", help="show GPUs with stockStatus=null too"
    ),
) -> None:
    """RunPod の GPU 型別 stock 状況 (High/Medium/Low/-) を一覧表示する。

    Train pod が \"no instances available\" で繰り返し失敗するときに、どの GPU
    型を `--gpu-name` に渡せば確保見込みが高いかを判断するための補助情報。
    SDK の GraphQL `gpuTypes.lowestPrice.stockStatus` を直接叩く。
    """
    from rich.table import Table
    from runpod.api.graphql import run_graphql_query

    _ensure_runpod_key()
    all_gpus = runpod_sdk.get_gpus()
    rows: list[tuple[str, str, int, str, str, str]] = []
    for entry in all_gpus:
        mem = int(entry.get("memoryInGb") or 0)
        if mem < min_memory_gb:
            continue
        gid = str(entry.get("id", ""))
        query = (
            '{ gpuTypes(input: {id: "' + gid + '"}) { '
            "memoryInGb securePrice communityPrice "
            "lowestPrice(input: {gpuCount: " + str(gpu_count) + "}) { "
            "uninterruptablePrice stockStatus } } }"
        )
        try:
            resp = run_graphql_query(query)
        except Exception as exc:  # noqa: BLE001
            rows.append(("ERR", gid, mem, "-", "-", str(exc)[:30]))
            continue
        entries = (resp.get("data") or {}).get("gpuTypes") or []
        for e in entries:
            lp = e.get("lowestPrice") or {}
            stock = lp.get("stockStatus") or "-"
            uninterruptable = lp.get("uninterruptablePrice")
            if uninterruptable is not None and uninterruptable > max_dph:
                continue
            sec = e.get("securePrice")
            comm = e.get("communityPrice")
            rows.append(
                (
                    stock,
                    gid,
                    mem,
                    f"{sec:.3f}" if sec is not None else "-",
                    f"{comm:.3f}" if comm is not None else "-",
                    f"{uninterruptable:.3f}" if uninterruptable is not None else "-",
                )
            )

    if not include_no_stock:
        rows = [r for r in rows if r[0] != "-"]

    stock_order = {"High": 0, "Medium": 1, "Low": 2, "-": 3, "ERR": 4}
    rows.sort(key=lambda r: (stock_order.get(r[0], 9), r[5] if r[5] != "-" else "z"))

    table = Table(
        title=f"RunPod GPU stock (min_mem={min_memory_gb}GB, gpu_count={gpu_count})"
    )
    table.add_column("stock")
    table.add_column("gpu")
    table.add_column("mem", justify="right")
    table.add_column("sec$/h", justify="right")
    table.add_column("comm$/h", justify="right")
    table.add_column("lowest$/h", justify="right")
    for stock, gid, mem, sec, comm, low in rows:
        color = {
            "High": "green",
            "Medium": "cyan",
            "Low": "yellow",
            "-": "dim",
            "ERR": "red",
        }.get(stock, "")
        prefix = f"[{color}]" if color else ""
        suffix = "[/]" if color else ""
        table.add_row(f"{prefix}{stock}{suffix}", gid, str(mem), sec, comm, low)
    console.print(table)


@app.command("ps")
def ps_cmd(
    case: str | None = typer.Option(
        None, "--case", help="特定 case の launched run のみ突き合わせて表示"
    ),
    show_all: bool = typer.Option(
        False, "--all", help="EXITED / TERMINATED の pod も含める"
    ),
) -> None:
    """所有する RunPod pod を一覧表示し、ローカル launch.json と突き合わせる。"""
    from rich.table import Table

    from gpu.runpod.runpod import pod_state as pod_state_mod

    _ensure_runpod_key()
    pods = pod_state_mod.list_pods()
    if not show_all:
        pods = [p for p in pods if p.is_active]

    pod_id_to_run: dict[str, str] = {}
    pod_id_to_mode: dict[str, str] = {}
    if case is not None:
        runs_root = _repo_root() / _runs_root_for(case)
        if runs_root.is_dir():
            for run_dir in runs_root.iterdir():
                if not run_dir.is_dir():
                    continue
                try:
                    payload = read_launch_json(run_dir)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                pid = payload.get("pod_id")
                rid = payload.get("run_id")
                if pid and rid:
                    pod_id_to_run[str(pid)] = str(rid)
                    pod_id_to_mode[str(pid)] = str(payload.get("mode", "oneshot"))

    if not pods:
        msg = "[yellow]No active pods.[/]" if not show_all else "[yellow]No pods.[/]"
        console.print(msg)
        return

    table = Table(title="RunPod pods")
    table.add_column("pod_id")
    table.add_column("name / run_id")
    table.add_column("mode")
    table.add_column("status")
    table.add_column("gpu")
    table.add_column("uptime", justify="right")
    table.add_column("$/h", justify="right")
    table.add_column("est. $", justify="right")
    for p in pods:
        run_id_label = pod_id_to_run.get(p.pod_id, p.name)
        mode_label = pod_id_to_mode.get(p.pod_id, "?")
        # interactive は課金停止し忘れリスクが高いので黄色強調。
        if mode_label == "interactive":
            mode_display = f"[yellow]{mode_label}[/]"
        else:
            mode_display = mode_label
        table.add_row(
            p.pod_id,
            run_id_label,
            mode_display,
            p.desired_status,
            p.gpu_display_name,
            pod_state_mod.format_uptime(p.uptime_seconds),
            f"{p.cost_per_hr:.3f}",
            f"{p.estimated_cost_usd:.3f}",
        )
    console.print(table)
    # interactive pod が混じっている場合、destroy リマインダを表示。
    interactive_runs = [
        pod_id_to_run.get(p.pod_id)
        for p in pods
        if pod_id_to_mode.get(p.pod_id) == "interactive"
    ]
    interactive_runs = [r for r in interactive_runs if r]
    if interactive_runs:
        console.print(
            "\n[yellow]Reminder:[/] interactive pods are billed until destroyed."
            " Stop them with: [cyan]dev/runpod destroy <run_id>[/]"
        )
        for r in interactive_runs:
            console.print(f"  - dev/runpod destroy {r}")


@app.command("status")
def status_cmd(
    run_id: str = typer.Argument(..., help="run_id from runpod train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    aws_profile: str = typer.Option(
        DEFAULT_AWS_PROFILE, "--aws-profile", help="S3 marker 取得用"
    ),
) -> None:
    """run_id 単位で pod state + onstart 進捗 + DVC push 状況を表示。"""
    from gpu.runpod.execution import progress as progress_mod
    from gpu.runpod.runpod import pod_state as pod_state_mod

    repo_root = _repo_root()
    run_dir = find_run_dir(repo_root, run_id, case)
    if not run_dir.is_dir():
        console.print(
            f"[red]run dir not found:[/] {run_dir}. "
            "Did you launch this run from a different worktree?"
        )
        raise typer.Exit(code=1)
    try:
        launch = read_launch_json(run_dir)
    except FileNotFoundError as exc:
        console.print(
            f"[red]launch.json not found:[/] {run_dir}. "
            "This run was likely launched before launch.json was wired in."
        )
        raise typer.Exit(code=1) from exc
    pod_id = str(launch["pod_id"])

    console.print(f"[bold]run_id:[/] {run_id}  [bold]pod_id:[/] {pod_id}")
    console.print(
        f"  branch={launch.get('branch')!r} commit={launch.get('commit_sha')!r} "
        f"gpu={launch.get('gpu_type_id')!r} cloud={launch.get('cloud_type')!r} "
        f"launched_at={launch.get('launched_at')!r}"
    )

    _ensure_runpod_key()
    pod = pod_state_mod.get_pod(pod_id)
    if pod is None:
        console.print(
            f"[yellow]Pod {pod_id} not found via SDK.[/] "
            "RunPod 側で既に削除済みの可能性。"
        )
    else:
        console.print(
            f"[bold]pod:[/] status={pod.desired_status} "
            f"uptime={pod_state_mod.format_uptime(pod.uptime_seconds)} "
            f"cost/h=${pod.cost_per_hr:.3f} "
            f"est_total=${pod.estimated_cost_usd:.3f}"
        )

    try:
        markers = progress_mod.list_markers(run_id, profile=aws_profile)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]progress markers unavailable:[/] {exc}")
        markers = []
    if markers:
        latest = progress_mod.latest_step(markers)
        assert latest is not None
        console.print(
            f"[bold]progress:[/] latest=[cyan]{latest.step}[/] at {latest.timestamp} "
            f"(total {len(markers)} markers)"
        )
        for m in markers:
            console.print(f"  - {m.timestamp}  {m.step}")
    else:
        console.print(
            "[bold]progress:[/] [yellow]no S3 markers (yet).[/] "
            "Pod might still be in container_started phase or AWS creds missing."
        )

    dvc_meta = repo_root / f"{_runs_root_for(case)}/{run_id}.dvc"
    if dvc_meta.is_file():
        console.print(
            f"[bold]artifact:[/] [green]{dvc_meta.relative_to(repo_root)} present[/]"
            " — `dev/runpod pull` ready"
        )
    else:
        console.print(
            f"[bold]artifact:[/] [yellow]{dvc_meta.relative_to(repo_root)} not yet "
            "committed[/] — onstart はまだ git push 段階に到達していない"
        )


@app.command("logs")
def logs_cmd(
    run_id: str = typer.Argument(..., help="run_id from runpod train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    source: str = typer.Option(
        "markers",
        "--source",
        help=(
            "markers (S3 marker 列挙、デフォルト) / "
            "onstart (run_dir or S3 の onstart.log 全文)"
        ),
    ),
    tail: int | None = typer.Option(None, "--tail", help="末尾 N 行のみ"),
    grep: str | None = typer.Option(None, "--grep", help="正規表現で行フィルタ"),
    aws_profile: str = typer.Option(
        DEFAULT_AWS_PROFILE, "--aws-profile", help="S3 アクセス用 AWS profile"
    ),
) -> None:
    """onstart 進捗ログを表示する。

    RunPod 公式 CLI には `pod logs` サブコマンドが無いため、本実装は
    onstart.sh.tmpl が S3 / run_dir に書き出した記録を読む 2 経路を持つ:

    --source markers (default): S3 progress marker
        (s3://.../runpod_progress/<RUN_ID>/<TS>_<STEP>) を timestamp 順に列挙。
        どこまで進んだかの粗い把握用。
    --source onstart: onstart の全文 stdout/stderr (`/var/log/onstart.log`) を
        表示。run_dir/onstart.log を優先 (DVC pull 後)、無ければ S3 fallback
        (s3://.../runpod_progress/<RUN_ID>/onstart.log、cleanup_destroy が
        失敗時に直接 upload した snapshot)。
    """
    from gpu.runpod.execution import progress as progress_mod

    if source not in ("markers", "onstart"):
        raise typer.BadParameter(
            f"--source must be 'markers' or 'onstart', got {source!r}"
        )

    repo_root = _repo_root()
    run_dir = find_run_dir(repo_root, run_id, case)
    if run_dir.is_dir():
        try:
            launch = read_launch_json(run_dir)
            console.print(
                f"[dim]launch:[/] pod_id={launch.get('pod_id')!r} "
                f"gpu={launch.get('gpu_type_id')!r} "
                f"cloud={launch.get('cloud_type')!r} "
                f"launched_at={launch.get('launched_at')!r}"
            )
        except (FileNotFoundError, json.JSONDecodeError):
            console.print(
                f"[yellow]launch.json missing in {run_dir}[/] "
                "(run launched before launch.json wiring; logs still readable)"
            )

    if source == "markers":
        try:
            markers = progress_mod.list_markers(run_id, profile=aws_profile)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]failed to list S3 markers:[/] {exc}")
            raise typer.Exit(code=1) from exc
        if not markers:
            console.print(
                f"[yellow]no S3 markers for run_id={run_id!r}.[/] "
                "Either onstart never wrote any (image pull failed before bash) "
                "or AWS credentials don't grant ListBucket on runpod_progress/."
            )
            raise typer.Exit(code=1)
        lines = [f"{m.timestamp}  {m.step}" for m in markers]
    else:
        # source == "onstart"
        try:
            log = progress_mod.fetch_onstart_log(
                run_id, run_dir=run_dir, profile=aws_profile
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]failed to fetch onstart log:[/] {exc}")
            raise typer.Exit(code=1) from exc
        if log is None:
            console.print(
                f"[yellow]no onstart log for run_id={run_id!r}.[/] "
                "Neither run_dir/onstart.log (try `dev/runpod pull`) nor S3 fallback "
                "(s3://.../runpod_progress/<run_id>/onstart.log) found."
            )
            raise typer.Exit(code=1)
        console.print(f"[dim]onstart log source: {log.source}[/]")
        lines = log.text.splitlines()

    if grep is not None:
        pattern = re.compile(grep)
        lines = [line for line in lines if pattern.search(line)]
    if tail is not None and tail > 0:
        lines = lines[-tail:]
    for line in lines:
        console.print(line, highlight=False)


@app.command("watch")
def watch_cmd(
    run_id: str = typer.Argument(..., help="run_id from runpod train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    poll_interval: float = typer.Option(
        DEFAULT_POLL_INTERVAL, "--poll-interval", help="ポーリング間隔 (秒)"
    ),
    max_wait_seconds: float = typer.Option(
        DEFAULT_MAX_WAIT, "--max-wait", help="タイムアウト (秒)"
    ),
    aws_profile: str = typer.Option(
        DEFAULT_AWS_PROFILE, "--aws-profile", help="S3 marker 取得用 AWS profile"
    ),
) -> None:
    """既存 pod の終了を待ち、完了/失敗時にデスクトップ通知を発火する。

    `dev/runpod train --watch` を起動し忘れた / 別ターミナルで監視したい場合に
    使う。launch.json から pod_id を解決する。
    """
    repo_root = _repo_root()
    run_dir = find_run_dir(repo_root, run_id, case)
    try:
        launch = read_launch_json(run_dir)
    except FileNotFoundError as exc:
        console.print(
            f"[red]launch.json not found:[/] {run_dir}. "
            "Provide --case or check the run was launched from this worktree."
        )
        raise typer.Exit(code=1) from exc
    pod_id = str(launch["pod_id"])
    console.print(
        f"[bold]watching[/] pod={pod_id} run_id={run_id} "
        f"(poll={poll_interval:.0f}s, max_wait={max_wait_seconds:.0f}s)"
    )
    _ensure_runpod_key()
    result = watch_pod(
        run_id=run_id,
        pod_id=pod_id,
        aws_profile=aws_profile,
        poll_interval=poll_interval,
        max_wait_seconds=max_wait_seconds,
    )
    color = "green" if result.is_success else "red"
    console.print(
        f"\n[{color}]watch result:[/] outcome={result.outcome} "
        f"last_step={result.last_step!r} "
        f"elapsed={result.elapsed_seconds:.0f}s"
    )
    console.print(f"  detail: {result.detail}")
    console.print(
        f"\n  Run [cyan]dev/runpod logs {run_id} --case {case}[/] "
        "to see the full marker timeline."
    )
    if not result.is_success:
        # Pre-terminate log rescue (same as train --watch path). The pod may
        # already be EXITED — rescue_onstart_log returns ssh-unavailable
        # quietly in that case, which is fine.
        if result.outcome in {"stalled", "failure"}:
            from gpu.runpod.execution.rescue import rescue_onstart_log

            rescue = rescue_onstart_log(
                runpod_sdk, pod_id, run_id, aws_profile=aws_profile
            )
            if rescue.uploaded:
                console.print(
                    f"[green]rescue:[/] onstart log saved to {rescue.s3_uri} "
                    f"({rescue.bytes_uploaded} bytes)"
                )
            else:
                console.print(f"[dim]rescue skipped:[/] {rescue.error}")
        raise typer.Exit(code=1)


# ===== TAIL_SOURCES (PR2: live tail via SSH) =====
# `dev/runpod tail <run_id> --source X` で SSH 経由 tail する対象ファイル定義。
# pod が RUNNING 中のみ機能する (永続化不要、生中継のみ)。pod terminate 後は
# `dev/runpod logs --source onstart` (S3 fallback) を使う。
TAIL_SOURCES: dict[str, str] = {
    "onstart": "tail -F /var/log/onstart.log",
    "train": "tail -F data/output/models/{family}/{case}/runs/{run_id}/train.log",
    "gpu": "tail -F data/output/models/{family}/{case}/runs/{run_id}/gpu.log",
    "system": "tail -F data/output/models/{family}/{case}/runs/{run_id}/system.log",
}


@app.command("tail")
def tail_cmd(
    run_id: str = typer.Argument(..., help="run_id from runpod train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    source: str = typer.Option(
        "onstart",
        "--source",
        help="onstart / train / gpu / system — どのログを SSH 経由 tail するか",
    ),
    follow: bool = typer.Option(
        True,
        "--follow/--no-follow",
        help="tail -F で続行 (default) か、現状の末尾だけ表示して exit するか",
    ),
) -> None:
    """SSH 経由で pod 内のログをリアルタイム tail する。

    `--source onstart`: /var/log/onstart.log (立ち上げ + 全体)
    `--source train`:   run_dir/train.log (学習プロセスの stdout のみ)
    `--source gpu`:     run_dir/gpu.log (nvidia-smi 10s 周期サンプル)
    `--source system`:  run_dir/system.log (psutil で CPU%/RAM 10s 周期)

    pod が RUNNING 中のみ。terminate 後は `dev/runpod logs --source onstart`
    で S3 fallback を使う。Community Cloud は port 公開仕様が異なる場合あり。
    """
    if source not in TAIL_SOURCES:
        raise typer.BadParameter(
            f"--source must be one of {sorted(TAIL_SOURCES)}, got {source!r}"
        )
    repo_root = _repo_root()
    run_dir = find_run_dir(repo_root, run_id, case)
    try:
        launch = read_launch_json(run_dir)
    except FileNotFoundError as exc:
        console.print(
            f"[red]launch.json not found:[/] {run_dir}. "
            "tail はローカルから launch した run のみ対象です。"
        )
        raise typer.Exit(code=1) from exc
    pod_id = str(launch["pod_id"])

    from gpu.runpod.runpod.ssh import SshUnavailable, get_pod_ssh_endpoint

    _ensure_runpod_key()
    try:
        endpoint = get_pod_ssh_endpoint(runpod_sdk, pod_id)
    except SshUnavailable as exc:
        console.print(f"[red]ssh unavailable:[/] {exc}")
        raise typer.Exit(code=1) from exc

    remote_cmd = TAIL_SOURCES[source].format(
        family=_case_family(case), case=_case_subdir(case), run_id=run_id
    )
    if not follow:
        # `tail -F` を `tail -n 200` に置き換える
        remote_cmd = remote_cmd.replace("tail -F", "tail -n 200")

    cmd = endpoint.to_command(remote_cmd)
    console.print(
        f"[dim]ssh -> {endpoint.host}:{endpoint.port} run={run_id} "
        f"source={source} follow={follow}[/]"
    )
    # フォアグラウンド実行: ユーザの Ctrl-C で止める想定
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 and proc.returncode != 130:  # 130 = SIGINT
        raise typer.Exit(code=proc.returncode)


@app.command("ssh")
def ssh_cmd(
    run_id: str = typer.Argument(..., help="run_id from runpod dev / train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    via: str = typer.Option(
        "proxy",
        "--via",
        help="proxy (ssh.runpod.io 経由、既定) / direct (TCP/22 public port)",
    ),
    key: Path | None = typer.Option(
        None,
        "--key",
        help=(
            "SSH private key path。既定: proxy→~/.ssh/id_ed25519, "
            "direct→~/.runpod/ssh/RunPod-Key-Go"
        ),
    ),
    exec_cmd: str | None = typer.Option(
        None,
        "--exec",
        help="1 回だけ実行するコマンド。省略時は対話 shell を開く。",
    ),
) -> None:
    """Open an interactive SSH session to the pod for the given run_id.

    proxy 経路: `ssh <pod_id>@ssh.runpod.io -i ~/.ssh/id_ed25519` 相当。
    direct 経路: `ssh root@<ip> -p <port> -i ~/.runpod/ssh/RunPod-Key-Go` 相当。
    """
    if via not in ("proxy", "direct"):
        raise typer.BadParameter(f"--via must be proxy/direct, got {via!r}")

    repo_root = _repo_root()
    run_dir = find_run_dir(repo_root, run_id, case)
    try:
        launch = read_launch_json(run_dir)
    except FileNotFoundError as exc:
        console.print(
            f"[red]launch.json not found:[/] {run_dir}. "
            "ssh はローカルから launch した run のみ対象です。"
        )
        raise typer.Exit(code=1) from exc
    pod_id = str(launch["pod_id"])

    from gpu.runpod.runpod.ssh import (
        SshUnavailable,
        get_pod_ssh_endpoint,
    )

    if via == "direct":
        # direct は pod が RUNNING + sshd up でないと port 解決できないので SDK を引く。
        _ensure_runpod_key()
        try:
            endpoint = get_pod_ssh_endpoint(
                runpod_sdk, pod_id, via="direct", key_path=key
            )
        except SshUnavailable as exc:
            console.print(f"[red]ssh unavailable:[/] {exc}")
            console.print(
                "[yellow]hint:[/] pod がまだ RUNNING/SSHD up に達していない可能性。"
                f" `dev/runpod status {run_id} --case {case}` で進捗確認してください。"
            )
            raise typer.Exit(code=1) from exc
    else:
        # proxy は SDK 呼び出し不要。pod_id だけで endpoint を組める。
        endpoint = get_pod_ssh_endpoint(None, pod_id, via="proxy", key_path=key)

    cmd = endpoint.to_command(exec_cmd)
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    proc = subprocess.run(cmd, check=False)
    # 130 = SIGINT (Ctrl-C で終了), 255 = ssh connection lost も exit 0 扱いにはしない
    if proc.returncode not in (0, 130):
        raise typer.Exit(code=proc.returncode)


@app.command("destroy")
def destroy_cmd(
    run_id: str = typer.Argument(..., help="run_id from runpod dev / train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    yes: bool = typer.Option(False, "--yes", "-y", help="確認 prompt を省略する"),
) -> None:
    """Terminate the pod backing a run_id (interactive モードでは必須)。

    interactive mode の pod は auto-cleanup が無いので、ユーザーが明示的に
    terminate する必要がある。oneshot mode の pod に対しても呼べる (no-op
    or idempotent terminate)。
    """
    from gpu.runpod.runpod.cleanup import terminate_pod

    repo_root = _repo_root()
    run_dir = find_run_dir(repo_root, run_id, case)
    try:
        launch = read_launch_json(run_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]launch.json not found:[/] {run_dir}")
        raise typer.Exit(code=1) from exc
    pod_id = str(launch["pod_id"])
    mode = str(launch.get("mode", "oneshot"))
    gpu = launch.get("gpu_type_id")
    dph = launch.get("dph_total")

    console.print(
        f"[bold]target:[/] run_id={run_id} pod_id={pod_id} mode={mode} "
        f"gpu={gpu!r} ${dph}/h"
    )
    if not yes:
        if not typer.confirm("Terminate this pod?", default=False):
            console.print("[yellow]aborted[/]")
            raise typer.Exit(code=1)

    _ensure_runpod_key()
    result = terminate_pod(runpod_sdk, pod_id)
    if result.terminated:
        console.print(f"[green]terminated[/] pod={pod_id}")
        # interactive mode では train を実行しないため run.json はローカルに
        # 存在しないことが多い。oneshot mode でも DVC pull 前に terminate した
        # 場合は同様。書き込めなければ status 記録は諦め、terminate は成功扱い。
        run_json_path = run_dir / "run.json"
        if run_json_path.is_file():
            update_run_json(run_dir, status="destroyed_manual")
            console.print("[dim]run.json status -> destroyed_manual[/]")
        else:
            console.print(
                "[dim]run.json not present (interactive mode); "
                "skipping status update[/]"
            )
    else:
        console.print(
            f"[red]failed to terminate[/] pod={pod_id} error={result.error!r}"
        )
        raise typer.Exit(code=1)


@app.command("sync")
def sync_cmd(
    run_id: str = typer.Argument(..., help="run_id from runpod dev / train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    direction: str = typer.Option(
        "push",
        "--direction",
        help="push (local → pod, 既定) / pull (pod → local)",
    ),
    via: str = typer.Option(
        "proxy",
        "--via",
        help="proxy (ssh.runpod.io) / direct (TCP/22 public port)",
    ),
    key: Path | None = typer.Option(None, "--key"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="rsync --dry-run で差分のみ表示する"
    ),
    delete: bool = typer.Option(
        False, "--delete", help="rsync --delete で送信先の余分なファイルを消す (危険)"
    ),
    push: bool = typer.Option(False, "--push", help="--direction push のエイリアス"),
    pull: bool = typer.Option(False, "--pull", help="--direction pull のエイリアス"),
) -> None:
    """Sync the backend/ directory between local and pod via rsync.

    `dev/runpod dev` で起動した interactive pod 上でコード試行錯誤するための補助。
    ローカルで編集 → `dev/runpod sync <run_id> --push` → 再実行、を 1 サイクルとする。
    """
    if push and pull:
        raise typer.BadParameter("--push and --pull are mutually exclusive")
    if push:
        direction = "push"
    if pull:
        direction = "pull"
    if direction not in ("push", "pull"):
        raise typer.BadParameter(f"--direction must be push/pull, got {direction!r}")
    if via not in ("proxy", "direct"):
        raise typer.BadParameter(f"--via must be proxy/direct, got {via!r}")

    repo_root = _repo_root()
    run_dir = find_run_dir(repo_root, run_id, case)
    try:
        launch = read_launch_json(run_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]launch.json not found:[/] {run_dir}")
        raise typer.Exit(code=1) from exc
    pod_id = str(launch["pod_id"])

    from gpu.runpod.runpod.ssh import (
        SshUnavailable,
        get_pod_ssh_endpoint,
    )
    from gpu.runpod.runpod.sync import build_sync_plan

    if via == "direct":
        _ensure_runpod_key()
        try:
            endpoint = get_pod_ssh_endpoint(
                runpod_sdk, pod_id, via="direct", key_path=key
            )
        except SshUnavailable as exc:
            console.print(f"[red]ssh unavailable:[/] {exc}")
            raise typer.Exit(code=1) from exc
    else:
        endpoint = get_pod_ssh_endpoint(None, pod_id, via="proxy", key_path=key)

    local_backend_dir = repo_root / "backend"
    plan = build_sync_plan(
        endpoint=endpoint,
        local_backend_dir=local_backend_dir,
        direction=direction,
        dry_run=dry_run,
        delete=delete,
    )
    console.print(f"[cyan]{plan.describe()}[/]")
    console.print(f"[dim]$ {' '.join(plan.cmd)}[/]")
    if delete and not dry_run:
        if not typer.confirm(
            "--delete will remove files at the destination not present "
            "at source. Proceed?",
            default=False,
        ):
            raise typer.Exit(code=1)
    proc = subprocess.run(plan.cmd, check=False)
    if proc.returncode != 0:
        raise typer.Exit(code=proc.returncode)


@app.command("summary")
def summary_cmd(
    run_id: str = typer.Argument(..., help="run_id from runpod train"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    aws_profile: str = typer.Option(
        DEFAULT_AWS_PROFILE, "--aws-profile", help="S3 marker / artifacts 用"
    ),
) -> None:
    """1 つの run の集約サマリ (status / cost / metrics / artifacts) を表示。

    複数 source (launch.json / run.json / S3 markers / S3 artifacts /
    DVC meta) を merge する。terminate 後でも S3 経由で参照可能。
    """
    from rich.table import Table

    from gpu.runpod.execution.summary import build_summary

    repo_root = _repo_root()
    s = build_summary(run_id, repo_root=repo_root, case=case, aws_profile=aws_profile)

    color_by_status = {
        "succeeded": "green",
        "failed": "red",
        "stalled": "yellow",
        "running": "cyan",
        "unknown": "white",
    }
    color = color_by_status.get(s.status, "white")
    console.print(
        f"[bold]run_id:[/] {s.run_id}  "
        f"[bold]pod_id:[/] {s.pod_id or '(unknown)'}  "
        f"[{color}]status={s.status}[/]"
    )
    if s.last_step:
        age = (
            f" ({s.last_step_age_seconds}s ago)"
            if s.last_step_age_seconds is not None
            else ""
        )
        console.print(f"  last_step: [cyan]{s.last_step}[/]{age}")
    if s.elapsed_seconds is not None:
        console.print(f"  elapsed: {s.elapsed_seconds}s")
    if s.estimated_cost_usd is not None:
        console.print(f"  estimated_cost: [bold]${s.estimated_cost_usd:.4f}[/]")
    if s.final_step is not None:
        console.print(
            f"  step: {s.final_step}  "
            f"train_return: {s.final_train_return}  "
            f"eval_return: {s.final_eval_return}"
        )

    table = Table(title="artifacts")
    table.add_column("name")
    table.add_column("local")
    table.add_column("S3 artifacts")
    for a in s.artifacts:
        table.add_row(
            a.name,
            "[green]yes[/]" if a.in_run_dir else "[dim]no[/]",
            "[green]yes[/]" if a.in_s3_artifacts else "[dim]no[/]",
        )
    console.print(table)
    if s.dvc_meta_committed:
        console.print("[green]DVC meta committed to origin[/]")
    else:
        console.print("[yellow]DVC meta NOT committed (use --from s3 for pull)[/]")


def _volume_sdk() -> Any:
    """volume 操作は graphql 経由なので runpod.api.graphql を返す。"""
    from runpod.api import graphql as runpod_graphql

    return runpod_graphql


volume_app = typer.Typer(
    add_completion=False,
    help="Manage RunPod network volumes (uv / DVC cache の永続化用).",
    no_args_is_help=True,
)
app.add_typer(volume_app, name="volume")


@volume_app.command("list")
def volume_list_cmd() -> None:
    """所有 network volumes を一覧。"""
    try:
        load_runpod_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    runpod_sdk.api_key = load_runpod_api_key()
    volumes = list_volumes(_volume_sdk())
    if not volumes:
        console.print("[yellow]No volumes found.[/]")
        return
    from rich.table import Table

    table = Table(title="RunPod network volumes")
    table.add_column("id")
    table.add_column("name")
    table.add_column("size_gb", justify="right")
    table.add_column("data_center")
    for v in volumes:
        table.add_row(v.id, v.name, str(v.size_gb), v.data_center_id)
    console.print(table)


@volume_app.command("search")
def volume_search_cmd(
    min_size_gb: int = typer.Option(300, "--min-size"),
    data_center_id: str | None = typer.Option(None, "--data-center-id"),
) -> None:
    """利用可能な data center 一覧を表示。"""
    offers = search_volume_offers(
        min_size_gb=min_size_gb, data_center_id=data_center_id
    )
    if not offers:
        console.print(f"[red]No matching DCs.[/] Known: {KNOWN_DATA_CENTERS}")
        raise typer.Exit(code=1)
    render_volume_offers(offers, console=console)


@volume_app.command("create")
def volume_create_cmd(
    name: str = typer.Argument(..., help="volume name"),
    size_gb: int = typer.Option(300, "--size"),
    data_center_id: str = typer.Option(..., "--data-center-id"),
) -> None:
    """新規 volume を作成し id を表示。"""
    try:
        runpod_sdk.api_key = load_runpod_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    new_id = create_volume_fn(
        _volume_sdk(),
        name=name,
        size_gb=size_gb,
        data_center_id=data_center_id,
    )
    console.print(
        f"[green]created volume:[/] id={new_id} name={name!r} "
        f"size={size_gb}GB dc={data_center_id}"
    )


__all__ = [
    "AwsCreds",
    "CASE_DEFAULTS",
    "DEFAULT_CASE",
    "DEFAULT_CLOUD_TYPE",
    "DEFAULT_TEMPLATE_PATH",
    "Offer",
    "Volume",
    "app",
]
