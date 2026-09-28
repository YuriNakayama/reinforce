"""Git helpers shared by RunPod CLI commands."""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer


def repo_root() -> Path:
    """Return the repository root containing the backend dir and ``.git``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        has_pkg = (parent / "backend").is_dir() or (parent / "backend").is_dir()
        if has_pkg and (parent / ".git").exists():
            return parent
    raise RuntimeError(f"repo root not found from {here}")


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or repo_root()),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_remote_url() -> str:
    return git("remote", "get-url", "origin")


def git_current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD")


def verify_commit_pushed(commit_sha: str) -> None:
    """Validate that ``commit_sha`` exists locally and is pushed to origin."""
    try:
        git("cat-file", "-e", commit_sha)
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            f"commit {commit_sha!r} does not exist locally"
        ) from exc
    try:
        remote_branches = git("branch", "-r", "--contains", commit_sha)
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(
            f"failed to check remote branches for {commit_sha!r}"
        ) from exc
    if not remote_branches.strip():
        raise typer.BadParameter(
            f"commit {commit_sha!r} is not pushed to origin. Run `git push` first."
        )


__all__ = [
    "git",
    "git_current_branch",
    "git_remote_url",
    "repo_root",
    "verify_commit_pushed",
]
