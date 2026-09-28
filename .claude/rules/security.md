# Security Guidelines

## Mandatory Security Checks

Before ANY commit:
- [ ] No hardcoded secrets (API keys, tokens, `GIT_PAT`) and no hardcoded S3 bucket names / AWS account IDs (use `REINFORCE_DVC_BUCKET`)
- [ ] Nothing under `data/`, no `*.tfstate` / `terraform.tfvars`, no `.env` staged
- [ ] Shell commands built from user / CLI input are passed as argument lists, not interpolated strings (`dev/*`, `backend/src/gpu/**`)
- [ ] Checkpoints are loaded only from our own runs (`torch.load(..., weights_only=True)`; never unpickle files from unknown sources)
- [ ] Logs and S3-uploaded onstart output don't echo credentials

## Confidential File Protection

NEVER read or modify the following files:

- `.env`, `.env.*`
- `/config/secrets.*`
- `**/*.pem`, `**/*.key`
- Files containing API keys, certificates, or credentials

## Secret Management

```python
# NEVER: Hardcoded secrets
api_key = "sk-ant-xxxxx"

# ALWAYS: Environment variables (loaded from backend/.env, which is gitignored)
import os
api_key = os.environ["RUNPOD_KEY"]
```

**Security Principles:**

- Manage sensitive information using environment variables
- Prohibit output of confidential information to logs and console
- Prohibit hardcoding of sensitive data

## Security Response Protocol

If security issue found:
1. STOP immediately
2. Use **security-reviewer** agent
3. Fix CRITICAL issues before continuing
4. Rotate any exposed secrets
5. Review entire codebase for similar issues

## CI/CD Pipeline Security

- GitHub Actions authenticates to AWS via OIDC, never long-lived access keys (see `.claude/rules/infra.md`)
- Provider API keys (RunPod / Vast.ai / Kaggle) never go into workflow secrets; pods receive them only through the launch CLI at runtime
- `uv.lock` is committed and CI installs with `--locked`, so dependency changes are always reviewed in a diff
