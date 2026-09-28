# Progress marker — write a tiny file to S3 at each step so we can debug
# without SSH access. AWS env vars are injected by create_pod.
S3_MARKER_PREFIX="s3://${REINFORCE_DVC_BUCKET}/remote/runpod_progress/<RUN_ID>"
S3_ARTIFACT_PREFIX="s3://${REINFORCE_DVC_BUCKET}/remote/runpod_artifacts/<RUN_ID>"
mark() {
  local step="$1"
  local ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local body="ts=${ts} step=${step} pod=${INSTANCE_ID}"
  echo "[onstart] mark step=${step}"
  if command -v aws >/dev/null 2>&1; then
    # 初回 mark のみ stderr を表示する (silent failure を見逃さない)。
    # 2 回目以降は log を汚さないために suppress。
    if [ -z "${MARK_DEBUG_DONE:-}" ]; then
      echo "${body}" | aws s3 cp - "${S3_MARKER_PREFIX}/${ts}_${step}" \
        || echo "[onstart] mark first-call s3 cp failed (subsequent silent)" >&2
      export MARK_DEBUG_DONE=1
    else
      echo "${body}" | aws s3 cp - "${S3_MARKER_PREFIX}/${ts}_${step}" 2>/dev/null || true
    fi
  else
    echo "[onstart] mark skipped (aws cli not available) step=${step}" >&2
  fi
}
mark "00_container_started"

# Upload the run directory's contract files to the S3 artifacts prefix
# (.claude/rules/backend/pipeline.md: metrics.json / checkpoint_latest.* /
# best.* / eval/seed<K>.json, plus logs). Used by the train heartbeat, the
# post-train final upload and the early artifact upload. RUN_DIR_ABS is set in
# 40_uv_and_dvc_pull.sh; before that this is a no-op. Never fatal.
#   $1 = "verbose" to log each file, anything else stays quiet.
upload_run_artifacts() {
  local verbose="${1:-quiet}"
  command -v aws >/dev/null 2>&1 || return 0
  if [ -z "${RUN_DIR_ABS:-}" ] || [ ! -d "${RUN_DIR_ABS}" ]; then
    return 0
  fi
  local f name
  shopt -s nullglob
  for f in "${RUN_DIR_ABS}"/{train.log,gpu.log,system.log,metrics.json,run.json,config.resolved.yaml} \
           "${RUN_DIR_ABS}"/best.* "${RUN_DIR_ABS}"/checkpoint_latest.*; do
    [ -f "${f}" ] || continue
    name="$(basename "${f}")"
    if [ "${verbose}" = "verbose" ]; then
      aws s3 cp "${f}" "${S3_ARTIFACT_PREFIX}/${name}" --only-show-errors \
        && echo "[onstart] uploaded ${name}" \
        || echo "[onstart] upload ${name} failed (non-fatal)" >&2
    else
      aws s3 cp "${f}" "${S3_ARTIFACT_PREFIX}/${name}" --only-show-errors >/dev/null 2>&1 || true
    fi
  done
  shopt -u nullglob
  if [ -d "${RUN_DIR_ABS}/eval" ]; then
    aws s3 cp "${RUN_DIR_ABS}/eval" "${S3_ARTIFACT_PREFIX}/eval/" --recursive \
      --only-show-errors >/dev/null 2>&1 || true
  fi
  if [ -f /var/log/onstart.log ]; then
    aws s3 cp /var/log/onstart.log "${S3_ARTIFACT_PREFIX}/onstart.log" \
      --only-show-errors >/dev/null 2>&1 || true
  fi
  return 0
}

# RunPod mode: "oneshot" (default, train→DVC push→自動 remove) or "interactive"
# (sleep infinity で pod を維持、auto remove なし、明示的 `dev/runpod destroy`
# で削除する運用)。<RUNPOD_MODE> は render_onstart で展開される。
RUNPOD_MODE="<RUNPOD_MODE>"
echo "[onstart] runpod_mode=${RUNPOD_MODE}"
mark "01_mode_${RUNPOD_MODE}"

# Periodic log flush: 30 秒間隔で /var/log/onstart.log を S3 に push する。
# stall 検出 → 外部から runpodctl remove pod / SDK terminate された場合に
# bash trap (cleanup_destroy) が動かず onstart.log が失われる問題への対策。
# `onstart.live.log` は本セッション中の暫定 path で、cleanup_destroy 成功時
# は最終 `onstart.log` (同 prefix) で上書きされる。aws cli が無ければ noop。
LOG_FLUSHER_PID=""
if command -v aws >/dev/null 2>&1; then
  (
    while true; do
      sleep 30
      if [ -f /var/log/onstart.log ]; then
        aws s3 cp /var/log/onstart.log \
          "${S3_MARKER_PREFIX}/onstart.live.log" \
          --no-progress --only-show-errors >/dev/null 2>&1 || true
      fi
    done
  ) &
  LOG_FLUSHER_PID=$!
  echo "[onstart] log flusher started pid=${LOG_FLUSHER_PID} interval=30s"
fi

# Hard timeout safety net: trap が壊れても 8h で pod を強制 remove する。
# 長い RL run は train + 最終 dvc push で数時間かかるので 8h に余裕を持たせる。
# これを超える run は checkpoint_latest.* から `--resume-from` で継続する。
# remove 前に log を S3 へ最終 snapshot として書き出す (cleanup_destroy が
# 呼ばれない経路の救済 — 例えば `runpodctl remove pod` を外部から叩かれて
# bash に SIGKILL が届いた場合や、kernel panic 等)。
# interactive モードでは timeout guard を張らない (ユーザーが destroy するまで保持)。
TIMEOUT_GUARD_PID=""
if [ "${RUNPOD_MODE}" = "oneshot" ]; then
  (
    sleep 28800
    if [ -f /var/log/onstart.log ] && command -v aws >/dev/null 2>&1; then
      aws s3 cp /var/log/onstart.log "${S3_MARKER_PREFIX}/onstart.log" 2>/dev/null || true
    fi
    runpodctl remove pod "$INSTANCE_ID" 2>/dev/null || true
  ) &
  TIMEOUT_GUARD_PID=$!
else
  echo "[onstart] interactive mode: skipping 8h timeout guard"
fi

cleanup_destroy() {
  local exit_code=$?
  # Stop the timeout guard so it doesn't fire later in the success path.
  if [ -n "${TIMEOUT_GUARD_PID}" ]; then
    kill "$TIMEOUT_GUARD_PID" 2>/dev/null || true
  fi
  # Stop the periodic log flusher: final snapshot is taken below at the
  # canonical `onstart.log` path, so the `onstart.live.log` daemon is no
  # longer needed and would otherwise race with the final upload.
  if [ -n "${LOG_FLUSHER_PID}" ]; then
    kill "${LOG_FLUSHER_PID}" 2>/dev/null || true
  fi
  echo "[onstart] cleanup status=${exit_code}"
  mark "90_cleanup_exit_${exit_code}"
  # 失敗時 (or 成功時の保険) に onstart log 全体を S3 に直接アップロード。
  # 成功 path では後段で run_dir/onstart.log として DVC 永久化されるが、
  # 序盤で失敗すると DVC 経路に到達しないので、必ずここで S3 へ最終 snapshot を残す。
  if [ -f /var/log/onstart.log ] && command -v aws >/dev/null 2>&1; then
    echo "[onstart] uploading log snapshot to s3://${S3_MARKER_PREFIX#s3://}/onstart.log"
    aws s3 cp /var/log/onstart.log \
      "${S3_MARKER_PREFIX}/onstart.log" 2>/dev/null || \
      echo "[onstart] log upload failed (non-fatal)"
  fi
  # Preserve whatever the run directory holds (checkpoint_latest.* in
  # particular) even when training failed or the pod is being torn down before
  # the DVC push, so the run can be resumed with `dev/runpod train --resume-from`.
  upload_run_artifacts quiet
  # RunPod は docker_args の bash プロセスが exit するとコンテナを再起動して
  # docker_args を再実行する (実 e2e で確認済み: stop pod 後も restart loop)。
  # `runpodctl stop pod` は storage を残して停止するだけで、RunPod の自動
  # restart 機構が発動して再実行ループに入る。`runpodctl remove pod` (alias:
  # delete) なら pod 自体が削除されて確実に止まる。
  if command -v runpodctl >/dev/null 2>&1; then
    if [ "${exit_code}" -ne 0 ]; then
      echo "[onstart] failure path: removing pod (exit=${exit_code})"
    else
      echo "[onstart] success path: removing pod=${INSTANCE_ID}"
    fi
    runpodctl remove pod "${INSTANCE_ID}" || echo "[onstart] remove pod failed"
  else
    echo "[onstart] runpodctl missing; cannot self-destroy."
  fi
  # Hang while RunPod tears the pod down. exec で bash 自体を sleep に置換し、
  # trap 連鎖や set -e による即 exit を抑止する。
  exec sleep infinity
}
# interactive モードでは EXIT trap を張らない。pod は sleep infinity で保持され、
# ユーザーが `dev/runpod destroy <run_id>` で明示的に terminate するまで残る。
if [ "${RUNPOD_MODE}" = "oneshot" ]; then
  trap cleanup_destroy EXIT
else
  echo "[onstart] interactive mode: skipping cleanup_destroy EXIT trap"
fi

