# Install minimal tools we depend on (uv for python pkg mgmt, aws cli via uv,
# runpodctl for self-destroy, git/curl for clone). pytorch image は小さく
# uv も awscli も持たないので、まず apt で git/curl/ca-certs を入れて
# 続けて uv -> uv pip install awscli の順に走らせる。
install_prereqs() {
  export DEBIAN_FRONTEND=noninteractive
  for attempt in 1 2 3; do
    if apt-get update -qq && \
       apt-get install -y -qq --no-install-recommends \
         git curl ca-certificates >/dev/null; then
      return 0
    fi
    echo "[onstart] install_prereqs apt attempt=${attempt} failed; retrying" >&2
    sleep 10
  done
  return 1
}
echo "[onstart] running install_prereqs (apt-get install git curl)"
install_prereqs || echo "[onstart] install_prereqs apt failed (continuing)"
echo "[onstart] install_prereqs done"

# uv を最初に install する (awscli を uv pip 経由で入れたいため、後段の
# `step=install_uv` よりここで前倒し)。pytorch image に uv はバンドルされて
# いないのが普通なので、astral.sh の install script で /root/.local/bin に置く。
if ! command -v uv >/dev/null 2>&1; then
  echo "[onstart] installing uv (early — awscli を uv pip で入れるため)"
  curl -LsSf https://astral.sh/uv/install.sh | sh \
    || echo "[onstart] uv install script failed" >&2
fi
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  echo "[onstart] uv ready: $(uv --version 2>&1 | head -1)"
else
  echo "[onstart] WARNING: uv unavailable" >&2
fi

# aws cli を uv tool install で入れる。Ubuntu 24.04 は PEP 668
# (externally-managed-environment) で system python への pip install を拒否する
# ため uv pip --system は失敗する。uv tool install は隔離 venv を作って bin を
# ~/.local/bin/aws に配置する (PATH に含めれば command -v aws で検出可能)。
if ! command -v aws >/dev/null 2>&1; then
  echo "[onstart] installing awscli via uv tool install"
  if command -v uv >/dev/null 2>&1; then
    UV_TOOL_OUT=$(uv tool install awscli 2>&1)
    UV_TOOL_EXIT=$?
    echo "[onstart] uv tool install output (last 5 lines):"
    echo "${UV_TOOL_OUT}" | tail -5
    if [ "${UV_TOOL_EXIT}" -eq 0 ]; then
      echo "[onstart] uv tool install awscli ok (exit=0)"
    else
      echo "[onstart] uv tool install awscli FAILED exit=${UV_TOOL_EXIT}" >&2
    fi
    # uv tool は ~/.local/bin に置く。PATH に含める。
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "[onstart] uv unavailable; cannot install awscli" >&2
  fi
fi
# install 完了後の sanity check: aws version と AWS env の有無を log する
if command -v aws >/dev/null 2>&1; then
  echo "[onstart] aws cli ready: $(aws --version 2>&1 | head -1)"
  # NOTE: never echo the key value itself. `${VAR:-no}` would expand to the
  # key when set, leaking it to the S3-persisted onstart log. Use a guard.
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "[onstart] AWS_ACCESS_KEY_ID set=yes"
  else
    echo "[onstart] AWS_ACCESS_KEY_ID set=no"
  fi
  echo "[onstart] AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-unset}"
else
  echo "[onstart] WARNING: aws cli unavailable; S3 markers/artifacts will be skipped" >&2
fi

# runpodctl is the official RunPod cli, used by self-destroy. The pytorch
# image does not ship it, so we fetch the static binary on-demand.
if ! command -v runpodctl >/dev/null 2>&1; then
  curl -sSL -o /usr/local/bin/runpodctl \
    "https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64" \
    && chmod +x /usr/local/bin/runpodctl \
    || echo "[onstart] runpodctl install failed"
fi

