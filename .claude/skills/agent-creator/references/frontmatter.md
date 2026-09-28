# Agent Frontmatter Reference

Every agent file under `.claude/agents/` starts with a YAML frontmatter block. Only `name` and `description` are strictly required; `tools` and `model` are commonly used to scope behavior.

```yaml
---
name: <kebab-case-name>
description: <single-line trigger + role summary>
tools: ["Read", "Grep", "Glob"]
model: sonnet
---
```

---

## `name` (required)

- Kebab-case, lowercase, no spaces.
- Must equal the filename without extension (`code-reviewer.md` → `name: code-reviewer`).
- Used as the `subagent_type` argument when the main session invokes the `Task` tool. Typos here mean the agent never spawns.
- Avoid generic names (`helper`, `tool`) — they cause description-level confusion when the project grows multiple agents.

---

## `description` (required)

This is the **single most important field**. It is the only thing the main session sees about the agent in its tool catalog, and it is the sole signal that determines whether the agent is invoked.

### Anatomy

```
<trigger style prefix>. <what it does>. <when to use it>. <when NOT to use it>.
```

### Trigger style prefixes

Pick one based on how aggressively the agent should fire:

| Prefix | Effect | Use when |
|--------|--------|----------|
| `Use when the user asks to ...` | Fires only when the user explicitly invokes | One-shot user-driven tools |
| `Use PROACTIVELY when ...` | Main agent considers spawning on relevant cues | Reviewers, validators, helpful side-tasks |
| `MUST BE USED for ...` | Main agent should always spawn for the matching context | Gatekeepers (security, mandatory review) |

`MUST BE USED` is expensive — every matching turn will spawn the agent, costing tokens and adding latency. Reserve it for genuinely mandatory checks. For everything else, `Use PROACTIVELY` is the right default.

### What to include

- **What the agent does** — one sentence, action verbs.
- **Concrete trigger phrases** — what the user might say or what state of the codebase should cue invocation. Include synonyms and casual phrasings so the trigger isn't brittle to wording.
- **Negative scope** — when *not* to use it, especially when it competes with another agent. Example: "for SQL or schema review, prefer database-reviewer".

### Examples

Good:

> Expert code review specialist. Proactively reviews code for quality, security, and maintainability. Use immediately after writing or modifying code. MUST BE USED for all code changes.

> Build and TypeScript error resolution specialist. Use PROACTIVELY when build fails or type errors occur. Fixes build/type errors only with minimal diffs, no architectural edits. Focuses on getting the build green quickly.

Bad (too narrow / implementation-focused):

> An agent that runs `tsc --noEmit` and parses the output.

Bad (too vague):

> Helps with code.

### Length

A few sentences is fine. Multi-paragraph descriptions are fine if the trigger conditions are genuinely complex (see `e2e-runner.md` for a long, well-structured example). Don't pad — every word should sharpen *when* to invoke.

---

## `tools` (optional)

JSON array of Claude Code tool names. If omitted, the agent inherits all tools available to the parent session (rarely what you want — almost always specify a subset).

See `tools.md` in this directory for the full tool list and recommended allow-lists by agent type.

```yaml
tools: ["Read", "Grep", "Glob"]                       # read-only reviewer
tools: ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]  # code-modifying agent
```

Notes:

- Tool names are case-sensitive and must match Claude Code's tool catalog exactly.
- Granting `Bash` enlarges blast radius — only include it if the agent actually needs to run scripts.
- `Task` is **not** typically granted to agents (agents spawning sub-agents leads to deep call stacks; let the main session orchestrate).

---

## `model` (optional)

Pick one to override the parent session's model:

- `haiku` — Claude Haiku, cheapest and fastest. Good for narrow mechanical agents.
- `sonnet` — Claude Sonnet, balanced default for most agents.
- `opus` — Claude Opus, deepest reasoning, most expensive. Use for architecture, security, complex planning.

Omit the `model` field to inherit from the parent. This is fine for agents that don't have strong cost/quality preferences, but explicit is usually better — it makes the agent's intended profile visible in review.

### Choosing a model

| Agent type | Suggested model |
|-----------|----------------|
| Lint fixer, formatter, simple file finder | `haiku` |
| Code reviewer, test writer, doc updater | `sonnet` |
| Architect, security reviewer, complex planner | `opus` |
| Build/test fixer (mostly mechanical, occasional reasoning) | `sonnet` |

If unsure, default to `sonnet`. It's the right answer ~70% of the time.

---

## Complete examples

### Read-only proactive reviewer

```yaml
---
name: terraform-reviewer
description: Terraform / IaC review specialist. Use PROACTIVELY when reviewing changes under `infra/`, `*.tf`, or `*.tfvars`. Flags missing tags, IAM over-permissions, and state-file leakage. For Python or app-code review, prefer code-reviewer instead.
tools: ["Read", "Grep", "Glob", "Bash"]
model: opus
---
```

### Mechanical fixer with write access

```yaml
---
name: lint-fixer
description: Auto-fixes ruff and mypy errors with minimal surgical edits. Use when lint or type-check fails after a code change. Does not refactor, only fixes the specific reported issues.
tools: ["Read", "Edit", "Bash", "Grep"]
model: haiku
---
```

### Explicit-only one-shot

```yaml
---
name: pr-summarizer
description: Use when the user asks to summarize a pull request or generate a PR description. Reads the diff and recent commits, then produces a 3-section markdown summary (Why / What / Test plan). Read-only.
tools: ["Read", "Grep", "Glob", "Bash"]
model: sonnet
---
```
