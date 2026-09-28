"""学習完了をデスクトップ通知する薄いヘルパ。

- macOS: `osascript -e 'display notification ...'`
- Linux: `notify-send`
- それ以外 / どちらも見つからない場合: stdout に `[notify]` プレフィックスで echo

通知に依存する CLI 経路は `dev/runpod train --watch` と `dev/runpod watch <run_id>`。
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def _quote_for_osascript(text: str) -> str:
    """osascript の AppleScript 文字列リテラル用に `"` と `\\` をエスケープ。"""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, message: str) -> bool:
    """デスクトップ通知を発火。失敗しても例外は出さず False を返す。"""
    if sys.platform == "darwin":
        bin_path = shutil.which("osascript")
        if bin_path is not None:
            script = (
                f'display notification "{_quote_for_osascript(message)}" '
                f'with title "{_quote_for_osascript(title)}"'
            )
            try:
                subprocess.run(
                    [bin_path, "-e", script],
                    check=False,
                    capture_output=True,
                    timeout=5,
                )
                return True
            except (OSError, subprocess.SubprocessError):
                pass
    elif sys.platform.startswith("linux"):
        bin_path = shutil.which("notify-send")
        if bin_path is not None:
            try:
                subprocess.run(
                    [bin_path, title, message],
                    check=False,
                    capture_output=True,
                    timeout=5,
                )
                return True
            except (OSError, subprocess.SubprocessError):
                pass

    # Fallback: stdout
    print(f"[notify] {title}: {message}", flush=True)
    return False


__all__ = ["notify"]


# 薄いラッパなので doctest 対象は最小限。実 OS への副作用を避けるため
# `subprocess.run` を mock するテストは tests/src/runpod_io/test_notify.py。
def _osascript_command(title: str, message: str) -> list[str]:
    """テストから osascript 引数を確認するための internal helper。"""
    script = (
        f'display notification "{_quote_for_osascript(message)}" '
        f'with title "{_quote_for_osascript(title)}"'
    )
    return ["osascript", "-e", script]
