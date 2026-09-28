# ===== EARLY ARTIFACT UPLOAD (PR1: 成果物消失の止血) =====
# dvc push まで到達せずに pod が外部 kill されるケースを想定し、run dir の成果物
# (best.* / checkpoint_latest.* / metrics.json / eval/ / run.json / logs) を
# s3://.../runpod_artifacts/ 配下に直接 upload する。これは DVC とは独立した保険経路。
# dvc push が成功すればそちらが正本だが、失敗しても artifacts prefix から
# `dev/runpod pull --from s3` で取れる。
echo "[onstart] step=early_artifact_upload prefix=${S3_ARTIFACT_PREFIX}"
echo "[onstart] artifact dir listing:"
ls -la "${RUN_DIR_ABS}" 2>&1 | head -20
upload_run_artifacts verbose
mark "75_artifacts_uploaded"

echo "[onstart] step=dvc_add_run"
# dvc add は cwd 基準 path を取る。RUN_DIR_ABS の絶対 path から repo root を
# 落として相対 path にする (cwd=/workspace/reinforce 前提)。
#
# DVC は symlinked dir 内のファイルに `dvc add` できないため、persist_setup で
# 張った data/output/models/<CASE_FAMILY> -> /persist/data-output-models-<CASE_FAMILY>
# の symlink を一時的に解除し、中身を実 dir にコピーで戻してから dvc add する。
# 完了後 (push 成否に関わらず) symlink を再構築する。観測: 2026-05-05 case7
# Step B 試行 #2 で `dvc_add_run` が `Cannot add files inside symlinked
# directories` で fatal exit。
RUN_DIR_REL="data/output/models/<CASE_FAMILY>/<CASE_SUBDIR>/runs/<RUN_ID>"
OUTPUT_LINK_TARGET=""
if [ -L "data/output/models/<CASE_FAMILY>" ]; then
  OUTPUT_LINK_TARGET=$(readlink "data/output/models/<CASE_FAMILY>")
  echo "[onstart] dvc_add_run: unlinking data/output/models/<CASE_FAMILY> -> ${OUTPUT_LINK_TARGET}"
  rm "data/output/models/<CASE_FAMILY>"
  mkdir -p "data/output/models/<CASE_FAMILY>"
  cp -a "${OUTPUT_LINK_TARGET}/." "data/output/models/<CASE_FAMILY>/" 2>&1 | tail -3 || true
fi

_relink_output() {
  if [ -n "${OUTPUT_LINK_TARGET}" ]; then
    echo "[onstart] dvc_add_run: re-linking data/output/models/<CASE_FAMILY> -> ${OUTPUT_LINK_TARGET}"
    rsync -a --delete "data/output/models/<CASE_FAMILY>/" "${OUTPUT_LINK_TARGET}/" 2>&1 | tail -3 || true
    rm -rf "data/output/models/<CASE_FAMILY>"
    ln -sfn "${OUTPUT_LINK_TARGET}" "data/output/models/<CASE_FAMILY>"
  fi
}

# iter11: uv run を排除して .venv/bin/dvc 直接呼び。
if ! ${DVC_BIN} add "${RUN_DIR_REL}" 2>&1; then
  echo "[onstart] dvc_add_run FATAL" >&2
  mark "75_dvc_add_run_failed"
  _relink_output
  exit 1
fi

echo "[onstart] step=dvc_push_runs"
# dvc push を fatal 化。DVC remote (S3) に本体が無いまま .dvc stub だけ
# git push されると次回 run (resume) の `dvc pull` が `FileNotFoundError` で死ぬ。
# 3 回までリトライ後 exit 1。
#
# dvc push は引数に `*.dvc` ファイル path を取る (directory path は
# "Stage X not found" エラーになる)。
# `dvc add ${RUN_DIR_REL}` で `${RUN_DIR_REL}.dvc` stub が生成されるので
# それを push 引数に使う。
RUN_DVC_FILE="${RUN_DIR_REL}.dvc"
if [ ! -f "${RUN_DVC_FILE}" ]; then
  echo "[onstart] dvc_push_runs: ${RUN_DVC_FILE} not found (dvc add failed?)" >&2
  mark "75_dvc_push_runs_failed"
  _relink_output
  exit 1
fi
# `-j 1 -v`: s3fs futures hang 回避 (treeverse/dvc#10374)
push_runs_ok=0
for attempt in 1 2 3; do
  echo "[onstart] dvc push -j 1 -v ${RUN_DVC_FILE} (attempt ${attempt}/3)"
  if ${DVC_BIN} push -j 1 -v "${RUN_DVC_FILE}" 2>&1; then
    push_runs_ok=1
    break
  fi
  echo "[onstart] dvc_push_runs attempt=${attempt} FAILED; retrying in 30s" >&2
  sleep 30
done
if [ "${push_runs_ok}" -ne 1 ]; then
  echo "[onstart] dvc_push_runs FATAL — exiting" >&2
  mark "75_dvc_push_runs_failed"
  _relink_output
  exit 1
fi

echo "[onstart] step=git_push_dvc_meta"
RUN_DVC_FILE="${RUN_DIR_REL}.dvc"
if [ -f "${RUN_DVC_FILE}" ]; then
  git config user.email "runpod-bot@reinforce.local"
  git config user.name "runpod-bot"
  TARGET_BRANCH="<BRANCH>"
  # GIT_PAT が pod env にあれば https URL に埋め込んで auth (push 用)。clone 経路は
  # PUBLIC repo なので URL に PAT 不要、ただし push は auth 必須。
  if [ -n "${GIT_PAT:-}" ]; then
    PUSH_URL="https://x-access-token:${GIT_PAT}@$(echo "<REPO_URL>" | sed 's|https://||')"
    git remote set-url origin "${PUSH_URL}" 2>&1 | tail -2
  else
    echo "[onstart] WARNING: GIT_PAT not set; git push 経路は失敗する見込み" >&2
  fi
  git fetch origin "${TARGET_BRANCH}" || true
  git checkout -B "${TARGET_BRANCH}" "origin/${TARGET_BRANCH}" || git checkout "${TARGET_BRANCH}" || true
  # data/ は .gitignore で除外されているが whitelist (`!/data/**/*.dvc`) で
  # .dvc ファイルだけは戻る。が、新規追加 path が whitelist を完全には
  # 通らない (parent dir も track されてないため hint が出る) ので明示 -f。
  git add -f -- "${RUN_DVC_FILE}"
  git commit --no-verify -m ":robot: runpod: add <CASE>/runs/<RUN_ID>.dvc" || {
    echo "[onstart] git_push_dvc_meta: nothing to commit (already tracked?)"
  }
  for attempt in 1 2 3; do
    if git push origin "HEAD:${TARGET_BRANCH}"; then
      echo "[onstart] git_push_dvc_meta: pushed to ${TARGET_BRANCH}"
      break
    fi
    echo "[onstart] git_push_dvc_meta: attempt=${attempt} rejected; rebasing"
    git pull --rebase origin "${TARGET_BRANCH}" || true
    if [ "${attempt}" -eq 3 ]; then
      echo "[onstart] git_push_dvc_meta: exhausted retries; .dvc stays in S3 only"
    fi
  done
else
  echo "[onstart] git_push_dvc_meta: ${RUN_DVC_FILE} not found (dvc add failed?)"
fi

# 全段階完了後に symlink を再構築 (next run の cache 利用は維持)。
_relink_output

mark "99_done"
echo "[onstart] step=done run_id=<RUN_ID>"
