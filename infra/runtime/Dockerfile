# Vast.ai runtime image for reinforce GPU training.
# 目的: uv sync (JAX CUDA12 plugin / cuDNN wheels 数 GB) を image layer に焼き込み、
# Vast 起動時の uv sync を cache hit (~30秒) に短縮する.
#
# Build context は repo root を想定. DVC_S3_BUCKET は必須 build-arg
# (bucket 名は AWS account id を含むので default を持たせない):
#   docker build --build-arg DVC_S3_BUCKET=<bucket> \
#     -t <ecr-url>:latest -f infra/runtime/Dockerfile .

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# DVC remote (S3). build_and_push.sh / buildspec.yml が --build-arg で渡す.
ARG DVC_S3_BUCKET
ARG DVC_S3_PREFIX=remote/files/md5
RUN test -n "${DVC_S3_BUCKET}" \
    || { echo "DVC_S3_BUCKET build-arg is required" >&2; exit 1; }

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# git は Vast onstart の git clone で必須. build-essential は一部の wheel が
# native extension の build に cc/ld を必要とするため.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv インストール (system path に配置)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Python 依存を先に layer に焼く. pyproject.toml / uv.lock の変更時のみ rebuild.
WORKDIR /opt/reinforce/backend
COPY backend/pyproject.toml backend/uv.lock ./

# uv sync は project 自身も install しようとするので --no-install-project で
# 依存のみ install. project 本体は onstart で git clone 後に install される.
# `cuda` group (JAX CUDA12 plugin) は Vast onstart が常に sync するので焼き込む.
# case 固有の追加 group (例: nn) は onstart 側で差分 install される.
RUN uv sync --locked --no-install-project --no-dev --group cuda

# Vast 起動時の uv cache hit を最大化するため、cache をユーザ可視な場所に維持.
ENV UV_CACHE_DIR=/root/.cache/uv

# DVC tracked データ (resume 用の run dir 等) を image に焼き込み、Vast 起動時の
# dvc pull を cache hit にする. remote 全体を sync するので、run が増えたら
# image が肥大化する点に注意 (必要なら DVC_S3_PREFIX を絞る).
# DVC remote の S3 prefix から直接 `aws s3 sync` でコンテンツアドレス cache を
# 取得することで、git/dvc init が不要になり build を簡素化する.
# build 時に AWS credentials を BuildKit secret 経由で受け取り、layer には残さない.
WORKDIR /opt/reinforce
# BuildKit secret 経由 (ローカル dev) または env 経由 (CodeBuild) の AWS auth に
# 両対応する. CodeBuild の場合は task role が credentials を提供するので
# secret mount は no-op (空ファイル) でも問題なく動く.
RUN --mount=type=secret,id=aws_creds,target=/root/.aws/credentials,required=false \
    pip install --no-cache-dir awscli==1.40.* >/dev/null && \
    mkdir -p .dvc/cache/files/md5 && \
    aws s3 sync \
        "s3://${DVC_S3_BUCKET}/${DVC_S3_PREFIX}" \
        .dvc/cache/files/md5 \
        --no-progress \
    && pip uninstall -y awscli >/dev/null

# Vast onstart は clone 先 (/workspace/reinforce) で `.dvc/cache` を
# /opt/reinforce/.dvc/cache に symlink して焼込み済 cache を流用する.

# clone 先 (onstart で /workspace/reinforce に clone される)
WORKDIR /workspace
