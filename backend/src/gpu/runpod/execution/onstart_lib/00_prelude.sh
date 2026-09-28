# RunPod onstart for reinforce GPU training.
# Placeholders are substituted by gpu.runpod.runpod.instance.render_onstart before upload:
#   <COMMIT_SHA>, <RUN_ID>, <BRANCH>, <REPO_URL>,
#   <CASE> (registry key, e.g. ppo_case1_smoke), <CASE_FAMILY> (e.g. ppo),
#   <CASE_SUBDIR> (e.g. case1), <TRAIN_MODULE> (python -m target),
#   <CONFIG_ARG> ('' or flags such as '--smoke' / '--config <path>'),
#   <PREPROCESS_CMD> ('' or 'module.path [--config <path>]'),
#   <UV_GROUPS> (extra uv dependency groups, e.g. 'nn'; cuda is always added),
#   <RESUME_FROM> ('' or a previous run_id whose checkpoint_latest.* seeds this run),
#   <RUNPOD_MODE> (oneshot / interactive)
set -uo pipefail  # 注意: -e は外す。install フェーズで疑似的な失敗 (apt の hash mismatch 等)
                  # が頻発し、過去 4 連続で onstart 序盤に消滅していた。`set -e` を入れない
                  # 方が早期段階で markers を残せる。trap EXIT は引き続き動く。
mkdir -p /var/log
exec > >(tee -a /var/log/onstart.log) 2>&1

INSTANCE_ID="${RUNPOD_POD_ID:-unknown}"

echo "[onstart] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ) pod=${INSTANCE_ID}"

# 早期 sshd 起動: docker_args で ENTRYPOINT を上書きしているので image の
# /start.sh は走らない → デフォルトでは sshd が立ち上がらない。bash の冒頭で
# 明示起動して、onstart 中の任意のタイミングで `tail -f /var/log/onstart.log`
# を SSH 経由で読めるようにする。AUTHORIZED_KEYS は pod の env PUBLIC_KEY か
# RunPod が予め配置してくれるはず。失敗しても onstart 本体は続行。
start_sshd_early() {
  # AUTHORIZED_KEYS を env から書き戻す (RunPod が PUBLIC_KEY を渡してくる)
  if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    echo "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
  fi
  # openssh-server は image によって既に入っている / いない両方ある。
  if ! command -v sshd >/dev/null 2>&1; then
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq --no-install-recommends openssh-server 2>/dev/null || true
  fi
  if command -v sshd >/dev/null 2>&1; then
    mkdir -p /var/run/sshd
    /usr/sbin/sshd 2>/dev/null && \
      echo "[onstart] sshd started (port 22 open)" || \
      echo "[onstart] sshd start failed (continuing without ssh)"
  else
    echo "[onstart] sshd not available; skipping"
  fi
}
start_sshd_early

