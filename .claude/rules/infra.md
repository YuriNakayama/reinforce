---
paths:
  - "infra/**"
---

# Infrastructure Rules (`infra/**`)

Terraform-managed AWS resources. Kept deliberately small: an S3 bucket for the DVC remote (checkpoints, metrics), an OIDC role so GitHub Actions can read it without long-lived credentials, and optionally an ECR repository for the Vast.ai runtime image built by `infra/runtime/`.

```
infra/
  runtime/                  Dockerfile + build scripts for the Vast.ai GPU image (present)
  environment/dev/          Root module — the only place you run terraform   (NOT YET CREATED)
  module/application/                                                         (NOT YET CREATED)
    dvc_remote/             S3 bucket (versioning + SSE + public-access block)
    github_actions_oidc/    OIDC provider + IAM role for GitHub Actions
    ecr_runtime/            (optional) ECR repo for the GPU runtime image
```

> **Status:** only `infra/runtime/` exists. The Terraform tree above is the intended layout; port it from the kaggriculture repository (`infra/environment/dev` + `infra/module/application/{dvc_remote,github_actions_oidc}`) and ptcg (`ecr_runtime`) when the DVC remote is first needed, renaming resources to `reinforce-*`. Until then `dev/dvc`, any pod launch that needs `REINFORCE_DVC_BUCKET`, and `infra/runtime/build_and_push.sh` (which reads ECR URLs from `terraform output` in `infra/environment/dev`) will fail, which is expected.

## Hard rules

> **`terraform apply` and `terraform destroy` are denied in `.claude/settings.json`.**
> Claude runs `fmt`, `validate`, and `plan` only. Applying infrastructure is a human decision.

> Never commit `terraform.tfstate` (it contains resource attributes and sometimes secrets), `terraform.tfvars` (account IDs, bucket names), or `.terraform/`. All three are gitignored — verify before every commit.

## Conventions

- **No hardcoded account IDs, ARNs, or bucket names in `.tf` bodies.** Everything parameterised through `variables.tf`; real values live in `terraform.tfvars`, which is gitignored. `terraform.tfvars.example` documents the shape with placeholders.
- Region is `ap-northeast-1`.
- Run terraform from `infra/environment/dev/` — that is the root module.
- After editing, always `terraform fmt` then `terraform validate`.

## OIDC, not access keys

GitHub Actions authenticates by assuming an IAM role via OIDC. The workflow declares:

```yaml
permissions:
  contents: read
  id-token: write        # required for OIDC
```

and uses `aws-actions/configure-aws-credentials@v4` with `role-to-assume: ${{ secrets.AWS_ROLE_ARN }}`. There are **no** `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` secrets in this repo, and none should be added — the OIDC trust policy scopes access to this specific repository.

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `REINFORCE_DVC_BUCKET` | `gpu/runpod`, `gpu/vast` | DVC/S3 bucket name. **Never hardcoded** — the bucket name embeds the AWS account id. Set it from `terraform output -raw dvc_bucket_name`. |
| `REINFORCE_RUNPOD_IMAGE` | `gpu/runpod` | Override the pod base image (default: a CUDA 12.4 PyTorch image) |
| `REINFORCE_RUNTIME_IMAGE` | `gpu/vast` | ECR image built by `infra/runtime/build_and_push.sh` |

## Repository secrets required

| Secret | Used by | Purpose |
|---|---|---|
| `AWS_ROLE_ARN` | future workflows that touch the DVC remote | Role to assume for DVC S3 access |

Provider API keys (`RUNPOD_KEY`, `VAST_API_KEY`, `KAGGLE_USERNAME` / `KAGGLE_KEY`) live in `backend/.env` on the developer machine only; no workflow needs them.

## Workflow-side conventions

Any scheduled workflow that writes to the DVC remote must serialise:

```yaml
concurrency:
  group: dvc-remote-write
  cancel-in-progress: false   # never cancel mid-write; queue instead
```

`cancel-in-progress: false` is deliberate — cancelling a job partway through a DVC push can leave the remote and the `.dvc` files inconsistent.
