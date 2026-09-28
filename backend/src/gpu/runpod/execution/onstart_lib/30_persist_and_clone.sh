echo "[onstart] step=env_persist"
env >> /etc/environment

# Persistent network volume (mount path is supplied by RunPod create_pod).
# When a network volume is attached at /persist we symlink:
#   uv cache, DVC cache, backend/.venv, and data/output/models/<CASE_FAMILY>
#   (run directories: checkpoints, metrics.json, eval/). RL cases generate their
#   training data online, so there is no lake/mart to persist.
# When the volume is absent we fall back to ephemeral disk. mkdir/ln are idempotent.
if [ -d /persist ]; then
  echo "[onstart] step=persist_setup mount=/persist"
  mkdir -p /persist/uv-cache /persist/dvc-cache
  mkdir -p "$HOME/.cache"
  rm -rf "$HOME/.cache/uv" 2>/dev/null || true
  ln -sfn /persist/uv-cache "$HOME/.cache/uv"
  export UV_CACHE_DIR="$HOME/.cache/uv"
  echo "[onstart] persist: uv cache -> /persist/uv-cache"
else
  echo "[onstart] persist: /persist not mounted (no volume) — using ephemeral disk"
fi

mark "10_before_clone"
echo "[onstart] step=clone repo=<REPO_URL> sha=<COMMIT_SHA>"
mkdir -p /workspace
cd /workspace
if [ ! -d reinforce ]; then
  if [ -n "${GIT_PAT:-}" ]; then
    git clone "https://x-access-token:${GIT_PAT}@$(echo "<REPO_URL>" | sed 's|https://||')" reinforce
  else
    git clone "<REPO_URL>" reinforce
  fi
fi
cd reinforce
git fetch origin "<COMMIT_SHA>" || true
git checkout "<COMMIT_SHA>"

if [ -d /persist ]; then
  echo "[onstart] step=persist_link_repo"
  mkdir -p .dvc
  rm -rf .dvc/cache 2>/dev/null || true
  ln -sfn /persist/dvc-cache .dvc/cache
  # backend/.venv を /persist/uv-venv に symlink。uv は venv の
  # python 絶対 path を pyvenv.cfg に焼き込むので、persist 側で venv を作る
  # ことで再起動後も再利用できる。lock 不変なら uv sync は no-op に近い。
  # DVC では管理しない (依存情報は uv.lock が source of truth)。
  mkdir -p /persist/uv-venv
  # attempt 6 (commit 9b49d74) で `uv sync` が以下のエラーで失敗した:
  #   error: failed to remove directory `/workspace/reinforce/backend/.venv/lib`:
  #   Directory not empty (os error 39)
  # 原因: lock hash mismatch を検出した uv が venv を rebuild しようとして
  # rmdir したが、network volume 上で rmdir が空 dir を空でないと報告した。
  # 対策: lock hash mismatch を検出したら、symlink 化の前に persisted venv
  # の中身を完全クリアしてしまう (uv の rmdir に頼らない)。
  PERSIST_LOCK_HASH_FILE="/persist/uv-venv/.uv_lock_sha256"
  CURRENT_LOCK_HASH=""
  if [ -f backend/uv.lock ] && command -v sha256sum >/dev/null 2>&1; then
    CURRENT_LOCK_HASH=$(sha256sum backend/uv.lock | awk '{print $1}')
  fi
  if [ -n "${CURRENT_LOCK_HASH}" ] \
     && [ -d /persist/uv-venv ] \
     && { [ ! -f "${PERSIST_LOCK_HASH_FILE}" ] \
          || [ "$(cat "${PERSIST_LOCK_HASH_FILE}" 2>/dev/null)" != "${CURRENT_LOCK_HASH}" ]; }; then
    # backend/.venv/bin/python が現存していれば中身は意味がある可能性があるが、
    # uv が rebuild すると判断するので予防的に消す。
    if [ -e /persist/uv-venv/bin/python ] \
       || [ -d /persist/uv-venv/lib ]; then
      echo "[onstart] persisted venv lock hash mismatch — clearing /persist/uv-venv/*"
      # find -delete で network volume の rmdir 不整合を回避
      find /persist/uv-venv -mindepth 1 -delete 2>/dev/null \
        || rm -rf /persist/uv-venv/* /persist/uv-venv/.[!.]* 2>/dev/null \
        || true
    fi
  fi
  if [ -L backend/.venv ]; then
    : # already linked
  elif [ -d backend/.venv ]; then
    # 既存 venv (古い ephemeral 由来) は破棄して symlink に置換
    rm -rf backend/.venv
    ln -sfn /persist/uv-venv backend/.venv
  else
    ln -sfn /persist/uv-venv backend/.venv
  fi
  # data/output/models/<CASE_FAMILY> も /persist に持続化。
  # train artifact の正本は DVC + S3 (`step=dvc_add_run` で push 済み) だが、
  # 学習中に pod が落ちた場合の checkpoint_latest.* からの resume や、複数 run の
  # train.log / gpu.log を残しておきたいケースに有用。サイズ管理は cost-report で観測。
  mkdir -p data/output/models "/persist/data-output-models-<CASE_FAMILY>"
  if [ -L "data/output/models/<CASE_FAMILY>" ]; then
    : # already linked
  elif [ -d "data/output/models/<CASE_FAMILY>" ]; then
    cp -an "data/output/models/<CASE_FAMILY>/." "/persist/data-output-models-<CASE_FAMILY>/" \
      2>/dev/null || true
    rm -rf "data/output/models/<CASE_FAMILY>"
    ln -sfn "/persist/data-output-models-<CASE_FAMILY>" "data/output/models/<CASE_FAMILY>"
  else
    ln -sfn "/persist/data-output-models-<CASE_FAMILY>" "data/output/models/<CASE_FAMILY>"
  fi
fi

