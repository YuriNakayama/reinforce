"""RunPod pod の SSH endpoint 解決と ssh コマンド構築。

2 経路をサポートする:

- **direct**: `runpod_sdk.get_pod()` の `runtime.ports` から TCP/22 の public
  host/port を抽出し、`ssh -i KEY -p PORT root@IP` で繋ぐ。`dev/runpod tail`
  既存実装。pod が RUNNING で sshd 起動完了している必要がある。
- **proxy**: RunPod 公式の proxy SSH (`ssh <pod_id>@ssh.runpod.io`)。pod 内の
  `~/.ssh/authorized_keys` に登録された SSH key で接続できる。Community
  Cloud でも安定し、port 公開仕様の違いを意識せず使える。

既定 key path:
- direct: `~/.runpod/ssh/RunPod-Key-Go` (runpodctl 生成、アカウント側に登録済み)
- proxy:  `~/.ssh/id_ed25519` (ユーザーアカウントの SSH key を RunPod に登録)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_DIRECT_KEY = Path.home() / ".runpod/ssh/RunPod-Key-Go"
DEFAULT_PROXY_KEY = Path.home() / ".ssh/id_ed25519"
PROXY_SSH_HOST = "ssh.runpod.io"

SshKind = Literal["direct", "proxy"]


class SshUnavailable(RuntimeError):
    """sshd 起動前 / pod 消失 / port 未公開 等で SSH できない場合に投げる。"""


@dataclass(frozen=True)
class SshEndpoint:
    """1 つの pod に対する SSH 接続情報。

    kind="direct" のとき: host=public IP, port=mapped TCP port
    kind="proxy"  のとき: host="<pod_id>@ssh.runpod.io", port は無視 (22 固定)
    """

    host: str
    port: int
    user: str = "root"
    key_path: Path = DEFAULT_DIRECT_KEY
    kind: SshKind = "direct"

    def to_command(self, remote_cmd: str | None = None) -> list[str]:
        """`ssh ...` の引数リストを返す。"""
        cmd: list[str] = [
            "ssh",
            "-i",
            str(self.key_path),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile=/tmp/runpod_known_hosts",
            "-o",
            "ConnectTimeout=10",
        ]
        if self.kind == "direct":
            cmd.extend(["-p", str(self.port)])
            cmd.append(f"{self.user}@{self.host}")
        else:
            # proxy: host already encodes `<pod_id>@ssh.runpod.io` form
            cmd.append(self.host)
        if remote_cmd is not None:
            cmd.append(remote_cmd)
        return cmd

    def display(self) -> str:
        """人間可読な接続コマンド文字列 (コピペ可能)。"""
        return " ".join(self.to_command())


def parse_ssh_endpoint(
    pod: Mapping[str, Any],
    *,
    key_path: Path = DEFAULT_DIRECT_KEY,
) -> SshEndpoint:
    """`runpod.get_pod()` 応答から direct SshEndpoint を抽出する。

    `runtime.ports` の中で privatePort=22 かつ isIpPublic=True のエントリを
    探す。見つからなければ SshUnavailable。
    """
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    for entry in ports:
        if not isinstance(entry, Mapping):
            continue
        if int(entry.get("privatePort") or 0) != 22:
            continue
        if not entry.get("isIpPublic", False):
            continue
        host = str(entry.get("ip") or "")
        port = int(entry.get("publicPort") or 0)
        if host and port:
            return SshEndpoint(host=host, port=port, key_path=key_path, kind="direct")
    raise SshUnavailable(
        f"no public TCP/22 port for pod={pod.get('id')!r}; "
        "sshd may still be starting up or pod is not RUNNING"
    )


def build_proxy_endpoint(
    pod_id: str,
    *,
    key_path: Path = DEFAULT_PROXY_KEY,
) -> SshEndpoint:
    """`<pod_id>@ssh.runpod.io` 形式の proxy endpoint を組み立てる。

    proxy SSH は RunPod 側で pod_id をルーティングキーとして使うため、pod が
    RUNNING でなくとも構築自体は可能 (接続時に失敗する)。
    認証は RunPod アカウントに登録済みの公開鍵で行う (`~/.ssh/id_ed25519.pub`
    などを https://runpod.io/console/user/settings に追加しておく)。
    """
    if not pod_id:
        raise SshUnavailable("empty pod_id for proxy SSH")
    return SshEndpoint(
        host=f"{pod_id}@{PROXY_SSH_HOST}",
        port=22,
        user="root",
        key_path=key_path,
        kind="proxy",
    )


def get_pod_ssh_endpoint(
    sdk: Any,
    pod_id: str,
    *,
    via: SshKind = "direct",
    key_path: Path | None = None,
) -> SshEndpoint:
    """pod_id を SDK で引いて SshEndpoint に変換する。

    Args:
        via: "direct" は `runtime.ports` から TCP/22 を引く (要 RUNNING + sshd up)。
             "proxy" は SDK 呼び出し不要で `<pod_id>@ssh.runpod.io` を組む。
        key_path: 既定は direct→DEFAULT_DIRECT_KEY / proxy→DEFAULT_PROXY_KEY。
    """
    if via == "proxy":
        key = key_path or DEFAULT_PROXY_KEY
        return build_proxy_endpoint(pod_id, key_path=key)

    pod = sdk.get_pod(pod_id)
    if pod is None:
        raise SshUnavailable(f"pod {pod_id!r} not found via SDK")
    if not isinstance(pod, Mapping):
        raise SshUnavailable(f"unexpected get_pod response type: {type(pod).__name__}")
    return parse_ssh_endpoint(pod, key_path=key_path or DEFAULT_DIRECT_KEY)


def grab_onstart_log(
    endpoint: SshEndpoint,
    *,
    remote_path: str = "/var/log/onstart.log",
    timeout_seconds: float = 30.0,
    runner: Any = None,
) -> str:
    """SSH 経由で pod から `onstart.log` を取得する。

    watcher が stalled を検出したとき、`terminate_pod` を呼ぶ前にこの関数で
    ログを救出して S3 / ローカルに永久化する。`subprocess.run` を直接呼ぶと
    test しづらいので `runner` で差し替え可能 (defaults to subprocess.run)。

    `SshUnavailable` を投げる条件: SSH command が非ゼロ exit code を返した、
    または stdout が空。timeout や ssh エラーも同じ例外で包む。
    """
    import subprocess

    run = runner if runner is not None else subprocess.run
    cmd = endpoint.to_command(f"cat {remote_path}")
    try:
        result = run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SshUnavailable(
            f"ssh cat {remote_path} timed out after {timeout_seconds}s"
        ) from exc
    except OSError as exc:
        raise SshUnavailable(f"ssh invocation failed: {exc}") from exc
    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip().splitlines()[-3:]
        raise SshUnavailable(
            f"ssh cat {remote_path} exited {result.returncode}: "
            + " | ".join(stderr_tail)
        )
    if not result.stdout:
        raise SshUnavailable(f"ssh cat {remote_path} returned empty output")
    return result.stdout


__all__ = [
    "DEFAULT_DIRECT_KEY",
    "DEFAULT_PROXY_KEY",
    "PROXY_SSH_HOST",
    "SshEndpoint",
    "SshKind",
    "SshUnavailable",
    "build_proxy_endpoint",
    "get_pod_ssh_endpoint",
    "grab_onstart_log",
    "parse_ssh_endpoint",
]
