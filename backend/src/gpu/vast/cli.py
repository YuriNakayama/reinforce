"""Vast.ai 操作の Typer CLI: train / pull / promote / cost-report。

`dev/vast` thin wrapper から `python -m gpu.vast <cmd>` で呼ばれる前提。
case の train_module / config_arg / uv_groups などは RunPod と共有の registry
(`gpu.runpod.config.cases`) から解決する。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from gpu.common.auth import load_dvc_bucket
from gpu.runpod.config.cases import (
    CASE_DEFAULTS,
    DEFAULT_CASE,
    case_defaults,
    case_family,
    case_subdir,
    find_best_weights,
    runs_root_for,
)
from gpu.vast import cost as cost_mod
from gpu.vast.auth import (
    DEFAULT_AWS_PROFILE,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
    load_vast_api_key,
)
from gpu.vast.instance import (
    DEFAULT_DISK_GB,
    DEFAULT_IMAGE,
    build_env_dict,
    create_instance,
    render_onstart,
)
from gpu.vast.offers import (
    Offer,
    pick_offer,
    search_offers,
)
from gpu.vast.run_meta import generate_run_id
from gpu.vast.volumes import (
    create_volume as create_volume_fn,
)
from gpu.vast.volumes import (
    find_volume_by_name,
    list_volumes,
    pick_volume_offer,
    render_volume_offers,
    search_volume_offers,
)

app = typer.Typer(
    add_completion=False,
    help="Vast.ai-driven GPU training basis for reinforce.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "onstart.sh.tmpl"
DEFAULT_COST_REPORT_DIR = Path("docs/experiment")
DEFAULT_COST_LIMIT_USD = 1.0
ESTIMATED_RUNTIME_HOURS = 0.5

# Single case registry shared with the RunPod CLI: gpu/runpod/config/cases.py.
# Adding a new GPU-trainable case = adding an entry there.
DEFAULT_RUNS_ROOT = runs_root_for(DEFAULT_CASE)
DEFAULT_CANONICAL_WEIGHTS = Path(CASE_DEFAULTS[DEFAULT_CASE]["canonical_weights"])


def _runs_root_for(case: str) -> Path:
    return runs_root_for(case)


def _case_defaults(case: str) -> dict[str, str]:
    return case_defaults(case)


def _repo_root() -> Path:
    """backend ディレクトリの親 (リポジトリ root) を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        has_pkg = (parent / "backend" / "pyproject.toml").is_file()
        if has_pkg and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def _git(*args: str, cwd: Path | None = None) -> str:
    """git サブコマンドを実行して stdout を返す (失敗時は CalledProcessError)。"""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or _repo_root()),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_remote_url() -> str:
    return _git("remote", "get-url", "origin")


def _git_current_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def _verify_commit_pushed(commit_sha: str) -> None:
    """commit が origin に push 済みか確認。なければ BadParameter。"""
    try:
        _git("cat-file", "-e", commit_sha)
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            f"commit {commit_sha!r} does not exist locally"
        ) from exc
    try:
        remote_branches = _git("branch", "-r", "--contains", commit_sha)
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            f"failed to check remote branches for {commit_sha!r}"
        ) from exc
    if not remote_branches.strip():
        raise typer.BadParameter(
            f"commit {commit_sha!r} is not pushed to origin. Run `git push` first."
        )


@app.command()
def train(
    commit_sha: str = typer.Argument(..., help="must be pushed to origin"),
    case: str = typer.Option(
        DEFAULT_CASE,
        "--case",
        help=f"case to train (one of: {sorted(CASE_DEFAULTS)})",
    ),
    train_module: str | None = typer.Option(
        None,
        "--train-module",
        help="python -m target (default: derived from --case)",
    ),
    config_arg: str | None = typer.Option(
        None,
        "--config-arg",
        help="extra flags for the train CLI, e.g. '--smoke' or '--config <path>' "
        "(default: derived from --case; pass '' to force empty)",
    ),
    preprocess_cmd: str | None = typer.Option(
        None,
        "--preprocess-cmd",
        help="optional 'module.path [--config <path>]' to run before training "
        "(default: derived from --case; pass '' to skip)",
    ),
    seed: int = typer.Option(0, "--seed", help="seed override (recorded in run.json)"),
    label: str | None = typer.Option(None, "--label", help="custom Vast label"),
    cost_limit_usd: float = typer.Option(
        DEFAULT_COST_LIMIT_USD,
        "--cost-limit",
        help="confirm before launching if estimated cost exceeds this",
    ),
    aws_profile: str = typer.Option(
        DEFAULT_AWS_PROFILE, "--aws-profile", help="AWS profile name"
    ),
    image: str = typer.Option(DEFAULT_IMAGE, "--image"),
    disk_gb: int = typer.Option(DEFAULT_DISK_GB, "--disk-gb"),
    volume_id: int | None = typer.Option(
        None,
        "--volume-id",
        help="既存 volume を /persist に link. 省略 + --volume-name 一致なら自動再利用",
    ),
    volume_name: str = typer.Option(
        "reinforce_cache",
        "--volume-name",
        help=(
            "自動再利用 / 新規作成時の volume name. "
            "Vast.ai の制約で alphanumeric + underscore のみ許可 (max 64 chars)"
        ),
    ),
    mount_path: str = typer.Option(
        "/persist",
        "--mount-path",
        help="volume のマウント先 (container 内 path)",
    ),
    auto_create_volume: bool = typer.Option(
        False,
        "--auto-create-volume",
        help="一致 volume が無ければ最安 offer から新規作成 (size=--volume-size)",
    ),
    volume_size_gb: float = typer.Option(
        15.0, "--volume-size", help="新規作成時の volume size (GB)"
    ),
) -> None:
    """Search Vast offers, pick one, and launch GPU training for a given commit."""
    defaults = _case_defaults(case)
    resolved_train_module = (
        train_module if train_module is not None else defaults["train_module"]
    )
    resolved_config_arg = (
        config_arg if config_arg is not None else defaults["config_arg"]
    )
    resolved_preprocess_cmd = (
        preprocess_cmd if preprocess_cmd is not None else defaults["preprocess_cmd"]
    )

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
        vast_api_key = load_vast_api_key()
        dvc_bucket = load_dvc_bucket()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc

    sdk = _build_sdk(vast_api_key)

    # Volume 解決 (network volume 前提):
    # Network volume は machine 非依存で attach 可能なので GPU offer の machine
    # 制約はかからない. 解決順:
    # 1) --volume-id 明示
    # 2) --volume-name 一致の既存 network volume を再利用
    # 3) --auto-create-volume があれば network volume offer から新規作成
    # 4) どれでもなければ永続化なし (volume_info=None)
    # Vast.ai は customer 側で post-attach する API を持たないため、`volume_info`
    # を create_instance に渡して同時 mount するのが正攻法.
    volume_info: dict[str, Any] | None = None
    volume_id_resolved: int | None = None
    if volume_id is not None:
        volume_id_resolved = int(volume_id)
        volume_info = {
            "mount_path": mount_path,
            "create_new": False,
            "volume_id": volume_id_resolved,
        }
        console.print(
            f"[cyan]volume:[/] linking id={volume_id_resolved} at {mount_path}"
        )
    else:
        existing = find_volume_by_name(list_volumes(sdk), volume_name)
        if existing is not None:
            volume_id_resolved = existing.id
            volume_info = {
                "mount_path": mount_path,
                "create_new": False,
                "volume_id": existing.id,
            }
            console.print(
                f"[cyan]volume:[/] reusing {existing.name!r} id={existing.id} "
                f"size={existing.size_gb:.0f}GB at {mount_path}"
            )
        elif auto_create_volume:
            v_offers = search_volume_offers(
                sdk, min_size_gb=volume_size_gb, network=True
            )
            if not v_offers:
                console.print(
                    "[red]No network volume offers matched. Aborting.[/]\n"
                    "  Try `dev/vast volume search` to inspect offers."
                )
                raise typer.Exit(code=1)
            v_chosen = pick_volume_offer(v_offers, console=console)
            new_id = create_volume_fn(
                sdk,
                offer_id=v_chosen.id,
                size_gb=volume_size_gb,
                name=volume_name,
                network=True,
            )
            volume_id_resolved = new_id
            volume_info = {
                "mount_path": mount_path,
                "create_new": False,
                "volume_id": new_id,
            }
            console.print(
                f"[green]created network volume:[/] id={new_id} name={volume_name!r}"
            )
        else:
            console.print(
                "[yellow]volume: not configured (uv cache 等は永続化されない).[/]\n"
                "  --auto-create-volume で初回作成、以降は --volume-name で自動再利用."
            )

    offers = search_offers(sdk)
    if not offers:
        console.print("[red]No offers matched the filter. Try again later.[/]")
        raise typer.Exit(code=1)
    chosen = pick_offer(offers, console=console)
    estimated = chosen.dph_total * ESTIMATED_RUNTIME_HOURS
    console.print(
        f"\nSelected offer #{chosen.id} {chosen.gpu_name} @ ${chosen.dph_total:.3f}/h "
        f"(est. ${estimated:.3f} for {ESTIMATED_RUNTIME_HOURS}h)"
    )
    if estimated > cost_limit_usd and not typer.confirm(
        f"Estimated cost ${estimated:.3f} exceeds limit ${cost_limit_usd:.2f}. "
        "Proceed anyway?",
        default=False,
    ):
        raise typer.Exit(code=1)

    run_id = generate_run_id(branch, commit_sha, seed)
    onstart_cmd = render_onstart(
        DEFAULT_TEMPLATE_PATH,
        commit_sha=commit_sha,
        run_id=run_id,
        branch=branch,
        repo_url=repo_url,
        case=case,
        case_family=case_family(case),
        case_subdir=case_subdir(case),
        train_module=resolved_train_module,
        config_arg=resolved_config_arg,
        preprocess_cmd=resolved_preprocess_cmd,
        uv_groups=defaults["uv_groups"],
    )
    env = build_env_dict(
        {
            "AWS_ACCESS_KEY_ID": aws_creds.access_key_id,
            "AWS_SECRET_ACCESS_KEY": aws_creds.secret_access_key,
            "AWS_DEFAULT_REGION": aws_creds.region,
            "VAST_API_KEY": vast_api_key,
            "REINFORCE_DVC_BUCKET": dvc_bucket,
            "REINFORCE_RUN_ID": run_id,
            "REINFORCE_GIT_SHA": commit_sha,
            "REINFORCE_GIT_BRANCH": branch,
            "REINFORCE_CASE": case,
        }
    )
    instance_id = create_instance(
        sdk,
        chosen.id,
        onstart_cmd=onstart_cmd,
        env=env,
        image=image,
        disk_gb=disk_gb,
        label=label or run_id,
        volume_info=volume_info,
    )
    console.print(
        f"\n[green]Instance launched![/] id=[bold]{instance_id}[/] run_id={run_id} "
        f"case={case}"
    )
    console.print(f"  Monitor logs: [cyan]vastai logs {instance_id}[/]")
    console.print(f"  Stop manually: [cyan]vastai destroy instance {instance_id}[/]")
    console.print(
        f"  After completion: [cyan]dev/vast pull {run_id} --case {case}[/] then "
        f"[cyan]dev/vast promote {run_id} --case {case}[/] to adopt"
    )


@app.command()
def pull(
    run_id: str = typer.Argument(..., help="run_id from gpu.vast-train"),
    case: str = typer.Option(
        DEFAULT_CASE, "--case", help="case (used to resolve runs_root)"
    ),
    runs_root: Path | None = typer.Option(
        None, "--runs-root", help="override runs root (default: derived from --case)"
    ),
) -> None:
    """`dvc pull` で run dir をローカルに取得し、run.json を表示する。"""
    if runs_root is None:
        runs_root = _runs_root_for(case)
    relative = runs_root / run_id
    repo_root = _repo_root()
    dvc_meta = repo_root / f"{relative}.dvc"
    if not dvc_meta.is_file():
        console.print(
            f"[yellow]missing:[/] {dvc_meta.relative_to(repo_root)} — "
            "Vast onstart should have committed it back to origin. Try "
            "`git fetch origin && git pull --rebase` on the training branch, "
            "then retry `dev/vast pull`."
        )
        raise typer.Exit(code=1)
    cmd = [
        "uv",
        "run",
        "--project",
        "backend",
        "dvc",
        "pull",
        str(relative),
    ]
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    result = subprocess.run(cmd, cwd=str(repo_root))
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
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
    case: str = typer.Option(
        DEFAULT_CASE, "--case", help="case (resolves runs_root + canonical)"
    ),
    runs_root: Path | None = typer.Option(
        None, "--runs-root", help="override runs root (default: derived from --case)"
    ),
    canonical: Path | None = typer.Option(
        None,
        "--canonical",
        help="canonical weights path (default: derived from --case)",
    ),
    eval_results_path: Path | None = typer.Option(
        None,
        "--eval-results",
        help="optional JSON file with local eval to merge into run.json",
    ),
) -> None:
    """Run dir の best.* を canonical weights にコピーし、run.json を adopted に更新する。"""
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
    dst = repo_root / canonical
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    console.print(f"[green]copied[/] {src} -> {dst}")

    # canonical weights は単独の `.dvc` stub で管理する (dvc.yaml pipeline は
    # 無い) ので、作成・更新とも `dvc add`。
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

    # run.json の status 更新
    from gpu.vast.run_meta import update_run_json

    patch: dict[str, Any] = {"status": "adopted"}
    if eval_results_path is not None:
        patch["local_eval_results"] = json.loads(
            eval_results_path.read_text(encoding="utf-8")
        )
    update_run_json(run_dir, **patch)
    console.print(f"[green]run.json status -> adopted[/] in {run_dir}")

    # 採用後の run dir 自体も DVC で push 対象に
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
    case: str = typer.Option(DEFAULT_CASE, "--case", help="case (resolves runs_root)"),
    runs_root: Path | None = typer.Option(
        None, "--runs-root", help="override runs root (default: derived from --case)"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_COST_REPORT_DIR, "--output-dir", help="markdown output dir"
    ),
) -> None:
    """run.json を集計して markdown のコストレポートを生成する。"""
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
    out_path = out_dir / f"vast_cost_report_{month}.md"
    out_path.write_text(md, encoding="utf-8")
    console.print(md)
    console.print(f"\n[green]written:[/] {out_path}")


def _build_sdk(api_key: str) -> Any:
    """vastai SDK の VastAI を遅延 import する。"""
    from vastai import VastAI

    return VastAI(api_key=api_key)


volume_app = typer.Typer(
    add_completion=False,
    help="Manage Vast.ai persistent volumes (uv cache / DVC cache 等の永続化).",
    no_args_is_help=True,
)
app.add_typer(volume_app, name="volume")


@volume_app.command("list")
def volume_list_cmd() -> None:
    """所有する volumes を一覧."""
    try:
        api_key = load_vast_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    sdk = _build_sdk(api_key)
    volumes = list_volumes(sdk)
    if not volumes:
        console.print("[yellow]No volumes found.[/]")
        return
    from rich.table import Table

    table = Table(title="Vast.ai volumes")
    table.add_column("id", justify="right")
    table.add_column("name")
    table.add_column("size_gb", justify="right")
    table.add_column("mount_path")
    table.add_column("machine_id", justify="right")
    table.add_column("$/h", justify="right")
    for v in volumes:
        table.add_row(
            str(v.id),
            v.name or "-",
            f"{v.size_gb:.0f}",
            v.mount_path or "-",
            str(v.machine_id) if v.machine_id else "-",
            f"${v.storage_per_hour:.5f}",
        )
    console.print(table)


@volume_app.command("search")
def volume_search_cmd(
    min_size_gb: float = typer.Option(15.0, "--min-size"),
    max_dph: float = typer.Option(1.00, "--max-dph", help="$/TB/month の上限"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    """購入可能な volume offer を検索."""
    try:
        api_key = load_vast_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    sdk = _build_sdk(api_key)
    offers = search_volume_offers(
        sdk, min_size_gb=min_size_gb, max_storage_per_month_per_tb=max_dph, limit=limit
    )
    if not offers:
        console.print("[yellow]No offers matched.[/]")
        return
    render_volume_offers(offers, console=console)


@volume_app.command("create")
def volume_create_cmd(
    name: str = typer.Argument(..., help="volume name (再利用キー)"),
    size_gb: float = typer.Option(15.0, "--size"),
    max_dph: float = typer.Option(1.00, "--max-dph", help="$/TB/month の上限"),
) -> None:
    """最安 offer から volume を新規作成し id を表示."""
    try:
        api_key = load_vast_api_key()
    except CredentialsError as exc:
        console.print(f"[red]credentials error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    sdk = _build_sdk(api_key)
    offers = search_volume_offers(
        sdk, min_size_gb=size_gb, max_storage_per_month_per_tb=max_dph
    )
    if not offers:
        console.print("[red]No offers matched.[/]")
        raise typer.Exit(code=1)
    chosen = pick_volume_offer(offers, console=console)
    new_id = create_volume_fn(sdk, offer_id=chosen.id, size_gb=size_gb, name=name)
    console.print(f"[green]created volume:[/] id={new_id} name={name!r}")


# Re-exports kept for downstream wrapper usage.
__all__ = [
    "Offer",
    "AwsCreds",
    "DEFAULT_RUNS_ROOT",
    "DEFAULT_TEMPLATE_PATH",
    "app",
]
