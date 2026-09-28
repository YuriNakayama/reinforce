"""Typer CLI: `python -m gpu.kaggle <sub> ...`

サブコマンド:
    train          dataset push + kernel push + (optional) watch
    dataset push   backend/ snapshot を Kaggle Dataset として upload
    dataset status dataset の processing 状態を表示
    pull           kernels_output → run_dir → dvc add
    promote        run の best.* を canonical weights に昇格 (RunPod と同等)
    status         単一 run の status を表示
    ps             active kernel 一覧
    logs           完了済 kernel の train.log を表示
    watch          run の終了まで polling
    cost-report    月次 free-hour 集計 markdown
    dev            interactive kernel を push (S3 command channel + sleep loop)
    exec           interactive kernel に S3 経由で bash コマンドを投入し結果取得
    sync           ローカル file を S3 経由で interactive kernel に転送
    info           interactive kernel の heartbeat / kernel status を表示
    destroy        interactive session を終了 (shutdown cmd + S3 cleanup)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from gpu.kaggle.artifacts.cost import (
    aggregate_runs,
    default_report_path,
    render_markdown,
)
from gpu.kaggle.artifacts.launch import (
    LaunchMeta,
    read_launch_json,
    write_launch_json,
)
from gpu.kaggle.artifacts.output import (
    dvc_add,
    place_into_run_dir,
    pull_kernel_output,
)
from gpu.kaggle.auth import KaggleCreds, load_kaggle_creds
from gpu.kaggle.dataset.api import (
    build_api,
    create_new_dataset,
    dataset_status,
    latest_version_commit,
    push_dataset_version,
)
from gpu.kaggle.dataset.builder import build_snapshot
from gpu.kaggle.dataset.metadata import write_dataset_metadata
from gpu.kaggle.interactive.channel import (
    DEFAULT_BUCKET as INTERACTIVE_BUCKET,
)
from gpu.kaggle.interactive.channel import (
    ChannelTimeoutError,
    Command,
    CommandResult,
    cleanup_session,
    file_key,
    get_heartbeat,
    submit_command,
)
from gpu.kaggle.interactive.template import (
    InteractiveContext,
    render_interactive_notebook,
)
from gpu.kaggle.kernel.runner import (
    KernelPollTimeout,
    poll_status,
    push_kernel,
)
from gpu.kaggle.kernel.state import KernelStatus
from gpu.kaggle.kernel.template import RenderContext, render_notebook
from gpu.runpod.artifacts.run_meta import generate_run_id, update_run_json
from gpu.runpod.config.cases import (
    CASE_DEFAULTS,
    DEFAULT_CASE,
    case_defaults,
    find_best_weights,
    runs_root_for,
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    add_completion=False,
    help="Kaggle Kernel GPU training basis (third provider alongside vast/runpod).",
)

dataset_app = typer.Typer(
    add_completion=False, help="Manage the backend/ Kaggle Dataset snapshot."
)
app.add_typer(dataset_app, name="dataset")

DEFAULT_DATASET_NAME = "reinforce-backend"
DEFAULT_ACCELERATOR = "gpu-t4x2"
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_POLL_TIMEOUT = 36000.0


def _prefix(msg: str) -> str:
    return f"[bold cyan][kaggle-kernel][/bold cyan] {msg}"


def _repo_root() -> Path:
    """Repo root = ``backend/pyproject.toml`` と ``.git`` (dir or file = worktree)
    を両方持つ最初の祖先ディレクトリ。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        has_backend = (parent / "backend" / "pyproject.toml").is_file()
        if has_backend and (parent / ".git").exists():
            return parent
    raise RuntimeError("repo root (containing backend/pyproject.toml + .git) not found")


def _git_rev_parse(rev: str, *, repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", rev],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _git_branch(*, repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _verify_pushed(commit_sha: str, *, repo_root: Path) -> None:
    """commit が origin に push されているか確認 (vast/runpod と同方針)。"""
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "branch", "-r", "--contains", commit_sha],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            console.print(
                _prefix(
                    f"[yellow]warning:[/] commit {commit_sha[:7]} does not appear "
                    "to be on any remote branch. Kernel will still run from the "
                    "Dataset snapshot, but kaggle_kernel_meta.commit_sha may not "
                    "be reproducible from git later."
                )
            )
    except subprocess.CalledProcessError:
        console.print(_prefix("[yellow]git rev verification skipped (offline)[/]"))


def _ensure_creds() -> KaggleCreds:
    creds = load_kaggle_creds()
    import os as _os

    _os.environ.setdefault("KAGGLE_USERNAME", creds.username)
    _os.environ.setdefault("KAGGLE_KEY", creds.key)
    return creds


def _dataset_slug(creds: KaggleCreds, name: str = DEFAULT_DATASET_NAME) -> str:
    return f"{creds.username}/{name}"


@dataset_app.command("push")
def dataset_push(
    commit_sha: str | None = typer.Option(
        None, "--commit-sha", help="commit SHA (HEAD if omitted)"
    ),
    label: str | None = typer.Option(
        None, "--label", help="optional label appended to version_notes"
    ),
    force_new: bool = typer.Option(
        False, "--force-new", help="create a brand new dataset"
    ),
    wheel: list[Path] = typer.Option(
        [], "--wheel", help="path to additional .whl files to include"
    ),
    data_file: list[Path] = typer.Option(
        [],
        "--data-file",
        help="file under data/ to ship with the snapshot (restored to the same "
        "relative path inside the kernel)",
    ),
    include_dotenv: bool = typer.Option(
        False,
        "--include-dotenv",
        help="include backend/.env (AWS creds for interactive mode); private dataset only",
    ),
) -> None:
    """backend/ を Kaggle Dataset として upload する。"""
    creds = _ensure_creds()
    repo_root = _repo_root()
    if commit_sha is None:
        commit_sha = _git_rev_parse("HEAD", repo_root=repo_root)
    console.print(_prefix(f"building snapshot for commit={commit_sha[:7]}"))
    slug = _dataset_slug(creds)
    work_dir = repo_root / "data" / "tmp" / f"kaggle_dataset_{commit_sha[:7]}"
    build_snapshot(
        repo_root,
        work_dir,
        include_wheels=wheel,
        include_data_files=data_file,
        include_dotenv=include_dotenv,
    )
    write_dataset_metadata(
        work_dir,
        slug=slug,
        title=f"reinforce backend snapshot ({commit_sha[:7]})",
        commit_sha=commit_sha,
    )
    api = build_api()
    if force_new:
        result = create_new_dataset(api, work_dir, commit_sha=commit_sha)
    else:
        try:
            result = push_dataset_version(
                api, work_dir, commit_sha=commit_sha, label=label
            )
        except Exception as e:
            console.print(
                _prefix(
                    f"[yellow]dataset_create_version failed ({e}); trying create_new[/]"
                )
            )
            result = create_new_dataset(api, work_dir, commit_sha=commit_sha)
    console.print(
        _prefix(
            f"[green]dataset version pushed[/] slug={result.slug} "
            f"notes={result.version_notes!r}"
        )
    )


@dataset_app.command("status")
def dataset_status_cmd(
    slug: str | None = typer.Option(None, "--slug"),
) -> None:
    """Kaggle Dataset の processing 状態を表示。"""
    creds = _ensure_creds()
    if slug is None:
        slug = _dataset_slug(creds)
    api = build_api()
    info = dataset_status(api, slug)
    latest = latest_version_commit(api, slug)
    console.print(_prefix(f"slug={slug}"))
    console.print(json.dumps(info, indent=2, ensure_ascii=False))
    console.print(_prefix(f"latest commit (parsed): {latest}"))


@app.command()
def train(
    commit_sha: str = typer.Argument(..., help="commit SHA to train on"),
    case: str = typer.Option(
        DEFAULT_CASE,
        "--case",
        help=f"case to train (one of: {sorted(CASE_DEFAULTS)})",
    ),
    accelerator: str = typer.Option(DEFAULT_ACCELERATOR, "--accelerator"),
    seed: int = typer.Option(0, "--seed"),
    label: str | None = typer.Option(
        None, "--label", help="optional label appended to run_id / version_notes"
    ),
    no_internet: bool = typer.Option(
        False, "--no-internet", help="disable internet access in the kernel"
    ),
    dataset_bump_only: bool = typer.Option(
        False,
        "--dataset-bump-only",
        help="reuse latest dataset version (no new version)",
    ),
    watch: bool = typer.Option(False, "--watch"),
    poll_interval: float = typer.Option(DEFAULT_POLL_INTERVAL, "--poll-interval"),
    poll_timeout: float = typer.Option(DEFAULT_POLL_TIMEOUT, "--poll-timeout"),
    wheel: list[Path] = typer.Option([], "--wheel"),
    data_file: list[Path] = typer.Option([], "--data-file"),
) -> None:
    """dataset push + notebook 生成 + kernel push + (--watch なら status polling)。"""
    creds = _ensure_creds()
    repo_root = _repo_root()
    case_default = case_defaults(case)
    branch = _git_branch(repo_root=repo_root)
    _verify_pushed(commit_sha, repo_root=repo_root)

    run_id = generate_run_id(branch, commit_sha, seed)
    if label:
        run_id = f"{run_id}__{label}"
    runs_rel = runs_root_for(case)
    run_dir = repo_root / runs_rel / run_id

    dataset_slug_str = _dataset_slug(creds)
    dataset_version: str | None = None
    if dataset_bump_only:
        api = build_api()
        existing = latest_version_commit(api, dataset_slug_str)
        dataset_version = existing or "latest"
        console.print(
            _prefix(f"reusing existing dataset version (commit={existing or '?'})")
        )
    else:
        work_dir = repo_root / "data" / "tmp" / f"kaggle_dataset_{commit_sha[:7]}"
        build_snapshot(
            repo_root, work_dir, include_wheels=wheel, include_data_files=data_file
        )
        write_dataset_metadata(
            work_dir,
            slug=dataset_slug_str,
            title=f"reinforce backend snapshot ({commit_sha[:7]})",
            commit_sha=commit_sha,
        )
        api = build_api()
        try:
            result = push_dataset_version(
                api, work_dir, commit_sha=commit_sha, label=label
            )
        except Exception as e:
            console.print(
                _prefix(
                    f"[yellow]dataset_create_version failed ({e}); trying create_new[/]"
                )
            )
            result = create_new_dataset(api, work_dir, commit_sha=commit_sha)
        dataset_version = result.version_notes

    # Kaggle resolves the actual kernel slug from the *title* (lowercase + hyphenate),
    # ignoring the `id` field for slug derivation. To keep slug and title aligned,
    # we build both from the same base string.
    # ts_part first to ensure slug uniqueness across same-case runs:
    # the 50-char cap truncates the end, so when ts is at the start every
    # run gets a unique slug even if case names are long (39 chars).
    ts_part = run_id.split("__")[0]
    kernel_basename = f"rl-{ts_part}-{commit_sha[:7]}-{case}"
    kernel_basename = "".join(
        c if c.isalnum() or c == "-" else "-" for c in kernel_basename
    ).lower()[:50]
    # Kaggle drops the trailing '-' when materializing slugs from titles. So
    # our launch.json must store the normalized (rstrip-) form to keep
    # `dev/kaggle status/pull` lookups correct.
    kernel_basename = kernel_basename.rstrip("-")
    kernel_slug = f"{creds.username}/{kernel_basename}"
    kernel_title = kernel_basename
    kernel_slug = _normalize_kernel_slug(kernel_slug)
    started_at = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    kk_meta_initial: dict[str, Any] = {
        "kernel_slug": kernel_slug,
        "dataset_slug": dataset_slug_str,
        "dataset_version": dataset_version,
        "accelerator": accelerator,
        "internet_enabled": not no_internet,
        "started_at": started_at,
    }

    ctx = RenderContext(
        run_id=run_id,
        commit_sha=commit_sha,
        branch=branch,
        case=case,
        train_module=case_default["train_module"],
        config_arg=case_default["config_arg"],
        uv_groups=case_default["uv_groups"],
        dataset_slug=dataset_slug_str,
        accelerator=accelerator,
        kaggle_kernel_meta_initial=kk_meta_initial,
        seed=seed,
    )
    nb = render_notebook(ctx)
    kernel_dir = repo_root / "data" / "tmp" / "gpu.kaggle" / run_id
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "main.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _write_kernel_metadata(
        kernel_dir,
        slug=kernel_slug,
        title=kernel_title,
        accelerator=accelerator,
        dataset_sources=[dataset_slug_str],
        enable_internet=not no_internet,
    )

    api2 = build_api()
    push_result = push_kernel(api2, kernel_dir)
    kk_meta_initial["kernel_version"] = push_result.version

    launch = LaunchMeta(
        run_id=run_id,
        case=case,
        commit_sha=commit_sha,
        branch=branch,
        kernel_slug=push_result.slug,
        kernel_version=push_result.version,
        dataset_slug=dataset_slug_str,
        dataset_version=dataset_version,
        accelerator=accelerator,
        enable_internet=not no_internet,
        started_at=started_at,
        seed=seed,
    )
    write_launch_json(run_dir, launch)
    console.print(
        _prefix(
            f"[green]kernel pushed[/] slug={push_result.slug} "
            f"version={push_result.version} url={push_result.url}"
        )
    )
    console.print(_prefix(f"launch.json written to {run_dir / 'launch.json'}"))

    if watch:
        _watch_loop(
            api2,
            push_result.slug,
            interval=poll_interval,
            timeout=poll_timeout,
        )


def _watch_loop(
    api: Any, slug: str, *, interval: float, timeout: float
) -> KernelStatus:
    try:
        status, msg = poll_status(api, slug, interval=interval, timeout=timeout)
    except KernelPollTimeout as e:
        console.print(_prefix(f"[red]poll timeout[/] {e}"))
        raise typer.Exit(code=2) from e
    if status == KernelStatus.COMPLETE:
        console.print(_prefix(f"[green]kernel complete[/] {slug}"))
    else:
        console.print(_prefix(f"[red]kernel {status.value}[/] {slug} msg={msg}"))
    return status


def _normalize_kernel_slug(slug: str) -> str:
    parts = slug.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid slug (expected owner/name): {slug!r}")
    owner, name = parts
    safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
    return f"{owner}/{safe_name[:50]}"


def _write_kernel_metadata(
    kernel_dir: Path,
    *,
    slug: str,
    title: str,
    accelerator: str,
    dataset_sources: list[str],
    enable_internet: bool,
) -> Path:
    payload = {
        "id": slug,
        "title": title,
        "code_file": "main.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if accelerator.startswith("gpu") else "false",
        "enable_tpu": "true" if accelerator.startswith("tpu") else "false",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
    }
    out = kernel_dir / "kernel-metadata.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


@app.command()
def pull(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    file_pattern: str | None = typer.Option(
        None,
        "--file-pattern",
        help=(
            "regex filter passed to kaggle kernels_output (re.search). "
            "Default: '^runs/' to limit to /kaggle/working/runs/* artifacts "
            "(Kaggle SDK does not paginate; filtering avoids dropping files)."
        ),
    ),
) -> None:
    """Kaggle Kernel の output を run_dir に取り込み、dvc add する。"""
    _ensure_creds()
    repo_root = _repo_root()
    runs_rel = runs_root_for(case)
    run_dir = repo_root / runs_rel / run_id
    launch = read_launch_json(run_dir)
    console.print(_prefix(f"pulling output for slug={launch.kernel_slug}"))
    tmp_dir = repo_root / "data" / "tmp" / f"kaggle_output_{run_id}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    api = build_api()
    if file_pattern is None:
        file_pattern = "^runs/"
    pull_kernel_output(api, launch.kernel_slug, tmp_dir, file_pattern=file_pattern)
    place_into_run_dir(tmp_dir, run_dir, run_id=run_id)
    console.print(_prefix(f"placed artifacts into {run_dir}"))
    dvc_add(run_dir, repo_root)
    console.print(_prefix("[green]dvc add complete[/]"))
    run_json_path = run_dir / "run.json"
    if run_json_path.is_file():
        console.print(_prefix("run.json:"))
        console.print(run_json_path.read_text(encoding="utf-8"))


@app.command()
def promote(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    eval_results: Path | None = typer.Option(None, "--eval-results"),
) -> None:
    """run の best.* を canonical weights に昇格し、run.json status=adopted。"""
    defaults = case_defaults(case)
    repo_root = _repo_root()
    runs_rel = runs_root_for(case)
    run_dir = repo_root / runs_rel / run_id
    canonical = Path(defaults["canonical_weights"])
    src = find_best_weights(run_dir, prefer_suffix=canonical.suffix)
    if src is None:
        console.print(_prefix(f"[red]not found:[/] {run_dir}/best.*"))
        raise typer.Exit(code=1)
    dst = repo_root / canonical
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    console.print(_prefix(f"[green]copied[/] {src} -> {dst}"))

    # canonical weights are tracked by a standalone `.dvc` stub (no dvc.yaml
    # pipeline in this repo), so `dvc add` both creates and updates it.
    dvc_track_cmd = [
        "uv",
        "run",
        "--project",
        "backend",
        "dvc",
        "add",
        str(canonical),
    ]
    console.print(_prefix(f"$ {' '.join(dvc_track_cmd)}"))
    subprocess.run(dvc_track_cmd, cwd=str(repo_root), check=False)

    patch: dict[str, Any] = {"status": "adopted"}
    if eval_results is not None:
        patch["local_eval_results"] = json.loads(
            eval_results.read_text(encoding="utf-8")
        )
    update_run_json(run_dir, **patch)
    console.print(_prefix(f"[green]run.json status -> adopted[/] in {run_dir}"))
    status_result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    console.print("\n[bold]git status:[/]")
    console.print(status_result.stdout or "(clean)")


@app.command()
def status(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
) -> None:
    """単一 run の launch.json / kernel status / run.json を表示。"""
    _ensure_creds()
    repo_root = _repo_root()
    run_dir = repo_root / runs_root_for(case) / run_id
    launch = read_launch_json(run_dir)
    api = build_api()
    raw = api.kernels_status(launch.kernel_slug)
    console.print(_prefix(f"launch: {launch}"))
    console.print(_prefix("kernel status:"))
    console.print(raw if isinstance(raw, (str, dict)) else str(raw))
    run_json = run_dir / "run.json"
    if run_json.is_file():
        console.print(_prefix("run.json:"))
        console.print(run_json.read_text(encoding="utf-8"))


@app.command()
def watch(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    poll_interval: float = typer.Option(DEFAULT_POLL_INTERVAL, "--poll-interval"),
    poll_timeout: float = typer.Option(DEFAULT_POLL_TIMEOUT, "--poll-timeout"),
) -> None:
    """既存 kernel の終了まで poll し、終了で desktop 通知する。"""
    _ensure_creds()
    repo_root = _repo_root()
    run_dir = repo_root / runs_root_for(case) / run_id
    launch = read_launch_json(run_dir)
    api = build_api()
    _watch_loop(api, launch.kernel_slug, interval=poll_interval, timeout=poll_timeout)


@app.command()
def logs(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    tail: int | None = typer.Option(None, "--tail"),
    grep: str | None = typer.Option(None, "--grep"),
) -> None:
    """完了 kernel の train.log を表示 (要事前 ``pull``)。"""
    repo_root = _repo_root()
    run_dir = repo_root / runs_root_for(case) / run_id
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        console.print(
            _prefix(
                f"[red]{log_path} not found[/]. Run `dev/kaggle pull <run_id>` first."
            )
        )
        raise typer.Exit(code=1)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if grep:
        lines = [ln for ln in lines if grep in ln]
    if tail is not None and tail > 0:
        lines = lines[-tail:]
    sys.stdout.write("\n".join(lines))
    sys.stdout.write("\n")


@app.command()
def ps(
    case: str | None = typer.Option(None, "--case"),
    all_: bool = typer.Option(False, "--all", "-a"),
) -> None:
    """``kernels_list`` で起動中の kernel を表示。"""
    creds = _ensure_creds()
    api = build_api()
    listed = api.kernels_list(user=creds.username, page_size=50)
    if not isinstance(listed, list):
        listed = list(listed)
    filtered = []
    for k in listed:
        if hasattr(k, "ref"):
            ref = k.ref
        elif isinstance(k, dict):
            ref = k.get("ref", "")
        else:
            ref = str(k)
        if case and case not in str(ref):
            continue
        filtered.append(ref)
    if not filtered:
        console.print(_prefix("(no kernels found)"))
        return
    console.print(_prefix("kernels:"))
    for ref in filtered[: None if all_ else 20]:
        console.print(f"  {ref}")


@app.command("cost-report")
def cost_report_cmd(
    month: str | None = typer.Option(
        None, "--month", help="YYYY-MM (defaults to current month)"
    ),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    runs_root: Path | None = typer.Option(None, "--runs-root"),
    out_path: Path | None = typer.Option(None, "--out"),
) -> None:
    """月次 free GPU hour 集計を markdown に出力。"""
    repo_root = _repo_root()
    if month is None:
        month = datetime.now(UTC).strftime("%Y-%m")
    if runs_root is None:
        runs_root = repo_root / runs_root_for(case)
    report = aggregate_runs(runs_root, month)
    md = render_markdown(report)
    if out_path is None:
        out_path = repo_root / default_report_path(Path("docs/experiment"), month)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    console.print(
        _prefix(
            f"wrote {out_path} — {len(report.rows)} runs, "
            f"{report.total_gpu_hours:.2f} GPU hours"
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interactive mode subcommands (equivalent of `dev/runpod dev` / ssh / sync / destroy).
# Kaggle Notebook には SSH / live tail がないため、S3 を command channel として
# 使う sleep-loop notebook を push して双方向操作を実現する。
# ─────────────────────────────────────────────────────────────────────────────


def _build_s3_client() -> Any:
    """boto3 S3 client を作成。AWS_PROFILE=reinforce を尊重。"""
    import os as _os

    import boto3

    profile = _os.environ.get("AWS_PROFILE", "reinforce")
    try:
        session = boto3.Session(profile_name=profile)
        return session.client("s3")
    except Exception:
        # profile 未設定の場合 default credential chain
        return boto3.client("s3")


@app.command("dev")
def dev_cmd(
    commit_sha: str = typer.Argument(..., help="commit SHA to bind the session to"),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    accelerator: str = typer.Option(DEFAULT_ACCELERATOR, "--accelerator"),
    seed: int = typer.Option(0, "--seed"),
    no_internet: bool = typer.Option(
        False, "--no-internet", help="Internet OFF is incompatible with S3 channel"
    ),
    dataset_bump_only: bool = typer.Option(
        False,
        "--dataset-bump-only",
        help="reuse latest dataset version (no new version)",
    ),
    s3_bucket: str = typer.Option(INTERACTIVE_BUCKET, "--s3-bucket"),
    max_idle_minutes: float = typer.Option(
        480.0, "--max-idle-minutes", help="voluntary exit threshold (Kaggle 9h hard)"
    ),
    wheel: list[Path] = typer.Option([], "--wheel"),
    data_file: list[Path] = typer.Option([], "--data-file"),
) -> None:
    """interactive kernel を push する (RunPod dev 相当)。

    push 後、kernel が S3 ``kaggle_interactive/<run_id>/`` を polling し続ける
    sleep loop に入る。``dev/kaggle exec`` でコマンド投入可能。
    """
    if no_internet:
        raise typer.BadParameter("--no-internet is incompatible with interactive mode")
    creds = _ensure_creds()
    repo_root = _repo_root()
    case_defaults(case)
    branch = _git_branch(repo_root=repo_root)
    _verify_pushed(commit_sha, repo_root=repo_root)

    run_id = generate_run_id(branch, commit_sha, seed)
    runs_rel = runs_root_for(case)
    run_dir = repo_root / runs_rel / run_id

    dataset_slug_str = _dataset_slug(creds)
    dataset_version: str | None = None
    if dataset_bump_only:
        api = build_api()
        existing = latest_version_commit(api, dataset_slug_str)
        dataset_version = existing or "latest"
    else:
        work_dir = repo_root / "data" / "tmp" / f"kaggle_dataset_{commit_sha[:7]}"
        # interactive mode は AWS creds が必要なため backend/.env を同梱する。
        # private dataset (isPrivate=true) 前提なので、所有者以外には見えない。
        build_snapshot(
            repo_root,
            work_dir,
            include_wheels=wheel,
            include_data_files=data_file,
            include_dotenv=True,
        )
        write_dataset_metadata(
            work_dir,
            slug=dataset_slug_str,
            title=f"reinforce backend snapshot ({commit_sha[:7]})",
            commit_sha=commit_sha,
        )
        api = build_api()
        try:
            result = push_dataset_version(
                api, work_dir, commit_sha=commit_sha, label="interactive"
            )
        except Exception as e:
            console.print(_prefix(f"[yellow]dataset push fallback: {e}[/]"))
            result = create_new_dataset(api, work_dir, commit_sha=commit_sha)
        dataset_version = result.version_notes

    ts_part = run_id.split("__")[0]
    kernel_basename = f"reinforce-dev-{case}-{ts_part}-{commit_sha[:7]}"
    kernel_basename = "".join(
        c if c.isalnum() or c == "-" else "-" for c in kernel_basename
    ).lower()[:50]
    kernel_slug = f"{creds.username}/{kernel_basename}"
    kernel_slug = _normalize_kernel_slug(kernel_slug)
    started_at = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    kk_meta_initial: dict[str, Any] = {
        "kernel_slug": kernel_slug,
        "dataset_slug": dataset_slug_str,
        "dataset_version": dataset_version,
        "accelerator": accelerator,
        "internet_enabled": True,
        "started_at": started_at,
        "mode": "interactive",
        "s3_bucket": s3_bucket,
    }

    ctx = InteractiveContext(
        run_id=run_id,
        commit_sha=commit_sha,
        branch=branch,
        case=case,
        dataset_slug=dataset_slug_str,
        accelerator=accelerator,
        kaggle_kernel_meta_initial=kk_meta_initial,
        s3_bucket=s3_bucket,
        max_idle_minutes=max_idle_minutes,
        seed=seed,
    )
    nb = render_interactive_notebook(ctx)
    kernel_dir = repo_root / "data" / "tmp" / "gpu.kaggle" / run_id
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "main.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _write_kernel_metadata(
        kernel_dir,
        slug=kernel_slug,
        title=kernel_basename,
        accelerator=accelerator,
        dataset_sources=[dataset_slug_str],
        enable_internet=True,
    )

    api2 = build_api()
    push_result = push_kernel(api2, kernel_dir)
    kk_meta_initial["kernel_version"] = push_result.version

    launch = LaunchMeta(
        run_id=run_id,
        case=case,
        commit_sha=commit_sha,
        branch=branch,
        kernel_slug=push_result.slug,
        kernel_version=push_result.version,
        dataset_slug=dataset_slug_str,
        dataset_version=dataset_version,
        accelerator=accelerator,
        enable_internet=True,
        started_at=started_at,
        seed=seed,
        extra={"mode": "interactive", "s3_bucket": s3_bucket},
    )
    write_launch_json(run_dir, launch)
    console.print(
        _prefix(
            f"[green]interactive kernel pushed[/] slug={push_result.slug} "
            f"s3=s3://{s3_bucket}/kaggle_interactive/{run_id}/"
        )
    )
    console.print(_prefix(f"run_id: {run_id}"))
    console.print(
        _prefix(
            "Next: `dev/kaggle info <run_id>` to confirm heartbeat, "
            "then `dev/kaggle exec <run_id> -- <cmd>`."
        )
    )


@app.command("exec")
def exec_cmd(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    cwd: str | None = typer.Option(
        None,
        "--cwd",
        help="working dir on kernel side (default: /tmp/reinforce-repo/backend)",
    ),
    timeout: float = typer.Option(300.0, "--timeout", help="per-cmd timeout (sec)"),
    await_timeout: float = typer.Option(
        600.0, "--await-timeout", help="max time to wait for result (sec)"
    ),
    poll_interval: float = typer.Option(2.0, "--poll-interval"),
    argv: list[str] = typer.Argument(
        ..., help="command tokens (use `--` to separate from options)"
    ),
) -> None:
    """interactive kernel に S3 経由でコマンドを投入し、結果を表示する。"""
    if not argv:
        raise typer.BadParameter("argv is empty")
    _ensure_creds()
    repo_root = _repo_root()
    run_dir = repo_root / runs_root_for(case) / run_id
    launch = read_launch_json(run_dir)
    bucket = (launch.extra or {}).get("s3_bucket") or INTERACTIVE_BUCKET
    s3 = _build_s3_client()
    cmd = Command(
        argv=list(argv),
        cwd=cwd or "/tmp/reinforce-repo/backend",
        timeout=timeout,
    )
    console.print(_prefix(f"submitting argv={argv} cwd={cmd.cwd}"))
    try:
        result = submit_command(
            s3,
            run_id,
            cmd,
            bucket=bucket,
            poll_interval=poll_interval,
            await_timeout=await_timeout,
        )
    except ChannelTimeoutError as e:
        console.print(_prefix(f"[red]timeout[/] {e}"))
        raise typer.Exit(code=2) from e
    _render_result(result)
    if result.returncode != 0:
        raise typer.Exit(code=max(result.returncode, 1))


def _render_result(result: CommandResult) -> None:
    console.print(
        _prefix(
            f"returncode={result.returncode} elapsed={result.elapsed_seconds}s "
            f"started={result.started_at} finished={result.finished_at}"
        )
    )
    if result.error:
        console.print(_prefix(f"[red]error:[/] {result.error}"))
    if result.stdout:
        console.print("\n[bold]stdout:[/]")
        sys.stdout.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if result.stderr:
        console.print("\n[bold]stderr:[/]")
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")


@app.command("sync")
def sync_cmd(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    file: list[Path] = typer.Option(
        ..., "--file", help="local file(s) to upload (relative to repo root)"
    ),
    remote_root: str = typer.Option(
        "/tmp/reinforce-repo",
        "--remote-root",
        help="kernel-side root that staged files will be copied into",
    ),
    await_timeout: float = typer.Option(300.0, "--await-timeout"),
) -> None:
    """ローカル file を S3 経由で kernel に転送する。

    Flow:
        1. file ごとに S3 state/files/<relpath> に put
        2. kernel に `cp -r /staged ...` を exec で実行させる
    """
    _ensure_creds()
    repo_root = _repo_root()
    run_dir = repo_root / runs_root_for(case) / run_id
    launch = read_launch_json(run_dir)
    bucket = (launch.extra or {}).get("s3_bucket") or INTERACTIVE_BUCKET
    s3 = _build_s3_client()
    uploaded: list[str] = []
    for path in file:
        path = path.resolve()
        if not path.is_file():
            raise typer.BadParameter(f"not a file: {path}")
        try:
            rel = str(path.relative_to(repo_root))
        except ValueError as e:
            raise typer.BadParameter(
                f"file {path} must be under repo_root {repo_root}"
            ) from e
        key = file_key(run_id, rel)
        s3.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())
        uploaded.append(rel)
        console.print(_prefix(f"staged {rel} → s3://{bucket}/{key}"))
    if not uploaded:
        return
    fetch_script = (
        "import os, boto3\n"
        f"BUCKET = {bucket!r}\n"
        f"PREFIX = 'kaggle_interactive/{run_id}/state/files/'\n"
        f"DEST_ROOT = {remote_root!r}\n"
        "s3 = boto3.client('s3')\n"
        "paginator = s3.get_paginator('list_objects_v2')\n"
        "for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):\n"
        "    for obj in page.get('Contents') or []:\n"
        "        key = obj['Key']\n"
        "        rel = key[len(PREFIX):]\n"
        "        if not rel: continue\n"
        "        dest = os.path.join(DEST_ROOT, rel)\n"
        "        os.makedirs(os.path.dirname(dest), exist_ok=True)\n"
        "        s3.download_file(BUCKET, key, dest)\n"
        "        print('wrote', dest)\n"
    )
    cmd = Command(
        argv=["python3", "-c", fetch_script],
        cwd=remote_root,
        timeout=await_timeout,
    )
    result = submit_command(s3, run_id, cmd, bucket=bucket, await_timeout=await_timeout)
    _render_result(result)
    if result.returncode != 0:
        raise typer.Exit(code=max(result.returncode, 1))


@app.command("info")
def info_cmd(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
) -> None:
    """interactive session の heartbeat + kernel status を表示。"""
    _ensure_creds()
    repo_root = _repo_root()
    run_dir = repo_root / runs_root_for(case) / run_id
    launch = read_launch_json(run_dir)
    bucket = (launch.extra or {}).get("s3_bucket") or INTERACTIVE_BUCKET
    s3 = _build_s3_client()
    api = build_api()
    raw = api.kernels_status(launch.kernel_slug)
    kernel_status = getattr(getattr(raw, "status", None), "name", str(raw))
    console.print(_prefix(f"kernel status: {kernel_status}"))
    hb = get_heartbeat(s3, run_id, bucket=bucket)
    if hb is None:
        console.print(_prefix("[yellow]no heartbeat yet[/]"))
    else:
        console.print(_prefix("heartbeat:"))
        console.print(json.dumps(hb, indent=2, ensure_ascii=False))
    console.print(_prefix(f"launch.json: {launch}"))


@app.command("destroy")
def destroy_cmd(
    run_id: str = typer.Argument(...),
    case: str = typer.Option(DEFAULT_CASE, "--case"),
    yes: bool = typer.Option(False, "-y", "--yes"),
    skip_shutdown_cmd: bool = typer.Option(
        False, "--skip-shutdown-cmd", help="skip the in-kernel shutdown command"
    ),
) -> None:
    """interactive session を終了し S3 prefix を cleanup。

    Kaggle 側 kernel は次の Quota cycle で自動停止する。即時停止が必要な場合は
    Kaggle Web UI から手動で Stop を押す (API に kernel cancel 機能なし)。
    """
    _ensure_creds()
    repo_root = _repo_root()
    run_dir = repo_root / runs_root_for(case) / run_id
    launch = read_launch_json(run_dir)
    bucket = (launch.extra or {}).get("s3_bucket") or INTERACTIVE_BUCKET
    if not yes:
        ok = typer.confirm(
            f"destroy interactive session {run_id} "
            f"(s3://{bucket}/kaggle_interactive/{run_id}/)?"
        )
        if not ok:
            raise typer.Abort()
    s3 = _build_s3_client()
    if not skip_shutdown_cmd:
        try:
            submit_command(
                s3,
                run_id,
                Command(argv=["__shutdown__"], timeout=10.0),
                bucket=bucket,
                await_timeout=30.0,
                poll_interval=2.0,
            )
        except ChannelTimeoutError:
            console.print(
                _prefix("[yellow]shutdown command did not return; proceeding[/]")
            )
        except Exception as e:
            console.print(_prefix(f"[yellow]shutdown command failed: {e}[/]"))
    deleted = cleanup_session(s3, run_id, bucket=bucket)
    console.print(_prefix(f"[green]S3 cleanup done[/] (deleted {deleted} objects)"))


if __name__ == "__main__":
    app()
