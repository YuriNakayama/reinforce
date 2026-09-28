#!/usr/bin/env bash
# Build the Vast.ai runtime image and push to ECR.
#
# Prerequisites:
#   - Terraform applied (ECR repo + IAM user で credentials 取得済み;
#     infra/environment/dev の outputs を参照する)
#   - REINFORCE_DVC_BUCKET (backend/.env と同じ値) が export 済み
#   - AWS credentials available via env / `--profile <name>`
#   - Docker daemon running (Docker Desktop or colima)
#
# Usage:
#   infra/runtime/build_and_push.sh [--tag <tag>] [--profile <aws-profile>]
#
# Tag policy:
#   - 'latest' tag は毎回上書き
#   - <commit-sha> tag を併存させて roll back 可能にする
set -euo pipefail

PROFILE="${AWS_PROFILE:-reinforce}"
TAG="latest"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

# Terraform output から private + public ECR URL を取得.
ECR_URL=$(cd infra/environment/dev && terraform output -raw ecr_runtime_repository_url)
PUBLIC_URI=$(cd infra/environment/dev && terraform output -raw ecr_runtime_public_uri)
REGION="ap-northeast-1"
REGISTRY="${ECR_URL%%/*}"
PUBLIC_REGISTRY="${PUBLIC_URI%%/*}"
SHA=$(git rev-parse --short=7 HEAD)

echo "[runtime] private=${ECR_URL} public=${PUBLIC_URI} tag=${TAG} sha=${SHA}"

echo "[runtime] step=ecr_login (private + public)"
aws ecr get-login-password --region "${REGION}" --profile "${PROFILE}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"
# Public ECR は us-east-1 限定
aws ecr-public get-login-password --region us-east-1 --profile "${PROFILE}" \
  | docker login --username AWS --password-stdin "${PUBLIC_REGISTRY}"

echo "[runtime] step=prepare_dvc_secret"
# DVC が image build 時に S3 から fetch するための credentials を BuildKit
# secret に渡す. Terraform 管理下の dvc_user (S3 RW) を使う.
DVC_PROFILE="${REINFORCE_DVC_PROFILE:-reinforce}"
DVC_CRED_FILE="$(mktemp)"
trap 'rm -f "$DVC_CRED_FILE"' EXIT
{
  echo "[default]"
  aws configure get aws_access_key_id --profile "$DVC_PROFILE" \
    | sed 's/^/aws_access_key_id = /'
  aws configure get aws_secret_access_key --profile "$DVC_PROFILE" \
    | sed 's/^/aws_secret_access_key = /'
  echo "region = ${REGION}"
} > "$DVC_CRED_FILE"

DVC_S3_BUCKET="${REINFORCE_DVC_BUCKET:?REINFORCE_DVC_BUCKET must be set (terraform output -raw dvc_bucket_name)}"

echo "[runtime] step=docker_build"
# linux/amd64 を明示 (Apple Silicon でも Vast 側 amd64 が必要)
DOCKER_BUILDKIT=1 docker buildx build \
  --platform linux/amd64 \
  --file infra/runtime/Dockerfile \
  --secret "id=aws_creds,src=${DVC_CRED_FILE}" \
  --build-arg "DVC_S3_BUCKET=${DVC_S3_BUCKET}" \
  --tag "${ECR_URL}:${TAG}" \
  --tag "${ECR_URL}:${SHA}" \
  --tag "${PUBLIC_URI}:${TAG}" \
  --tag "${PUBLIC_URI}:${SHA}" \
  --load \
  .

echo "[runtime] step=docker_push (private)"
docker push "${ECR_URL}:${TAG}"
docker push "${ECR_URL}:${SHA}"

echo "[runtime] step=docker_push (public)"
docker push "${PUBLIC_URI}:${TAG}"
docker push "${PUBLIC_URI}:${SHA}"

echo "[runtime] done. Vast.ai が pull するのは public:"
echo "  ${PUBLIC_URI}:${TAG}"
