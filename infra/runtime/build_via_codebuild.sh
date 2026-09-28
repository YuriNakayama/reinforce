#!/usr/bin/env bash
# Vast.ai runtime image を AWS CodeBuild で build & push する.
# ローカル Docker に依存しないクラウドビルド経路.
#
# 流れ:
#  1) repo を zip 化
#  2) S3 source bucket にアップロード
#  3) `aws codebuild start-build` で project を起動
#  4) build-id を表示 (進捗は `aws codebuild batch-get-builds` で確認)
set -euo pipefail

PROFILE="${AWS_PROFILE:-reinforce-admin}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

PROJECT_NAME=$(cd infra/environment/dev && terraform output -raw runtime_codebuild_project_name)
SOURCE_BUCKET=$(cd infra/environment/dev && terraform output -raw runtime_codebuild_source_bucket)
SOURCE_KEY="source/reinforce-source.zip"
TMP_ZIP="$(mktemp -t reinforce-source.XXXXXX).zip"
trap 'rm -f "$TMP_ZIP"' EXIT

echo "[codebuild] project=${PROJECT_NAME} source=s3://${SOURCE_BUCKET}/${SOURCE_KEY}"

echo "[codebuild] step=zip_source"
# .git は不要 (CodeBuild が source 取り込み後に build するだけ).
# tracked + untracked DVC artifacts のうち、Dockerfile が必要とする
# *.dvc メタファイルだけ含める.
git ls-files | zip -q "$TMP_ZIP" -@
echo "  zip size: $(du -h "$TMP_ZIP" | cut -f1)"

echo "[codebuild] step=upload_source"
aws s3 cp "$TMP_ZIP" "s3://${SOURCE_BUCKET}/${SOURCE_KEY}" --profile "$PROFILE"

echo "[codebuild] step=start_build"
BUILD_ID=$(aws codebuild start-build \
  --project-name "$PROJECT_NAME" \
  --profile "$PROFILE" \
  --query 'build.id' \
  --output text)

echo "[codebuild] started build_id=${BUILD_ID}"
echo "  follow logs: aws codebuild batch-get-builds --ids ${BUILD_ID} --profile ${PROFILE}"
echo "  console: https://ap-northeast-1.console.aws.amazon.com/codesuite/codebuild/projects/${PROJECT_NAME}/build/${BUILD_ID//:/%3A}/log"
