# Claude Code Tools — Allow-list Reference

This is the catalog of tool names you can put in an agent's `tools:` frontmatter array, plus recommended allow-list patterns by agent role.

> Tool names are case-sensitive and must match exactly. Misspelled names silently fail (the tool just isn't granted).

---

## Core tools

| Tool | What it does | Risk |
|------|-------------|------|
| `Read` | Read any file the user can read | Low |
| `Grep` | Search file contents (ripgrep) | Low |
| `Glob` | Find files by pattern | Low |
| `Write` | Create or overwrite files | High (overwrites without diff) |
| `Edit` | Targeted in-place edits | Medium |
| `Bash` | Run shell commands | High (arbitrary execution) |
| `WebFetch` | Fetch a single URL | Medium (network egress) |
| `WebSearch` | Web search | Low (read-only network) |

## Specialized tools

| Tool | What it does | Notes |
|------|-------------|-------|
| `Task` | Spawn a subagent | Rarely granted to agents themselves — let the main session orchestrate |
| `TaskCreate` / `TaskUpdate` / `TaskList` / `TaskGet` / `TaskOutput` / `TaskStop` | Background task management | Niche, only if the agent runs long-lived work |
| `NotebookEdit` | Edit Jupyter notebooks | Only for notebook-aware agents |
| `Monitor` | Watch a streaming process | Niche, for long-running observers |
| `Skill` | Invoke another skill | Reserved for the main session; rarely useful inside an agent |
| `AskUserQuestion` | Pop a structured question UI | Some agent contexts strip this; prefer free-form questions in the agent's output instead |
| `EnterPlanMode` / `ExitPlanMode` / `EnterWorktree` / `ExitWorktree` | Mode switches | Rarely needed in agents |

## MCP / integration tools

These appear with prefixes like `mcp__claude_ai_<service>__<action>` (Gmail, Slack, Google Calendar, Google Drive, etc.). Grant only when the agent's job specifically requires that integration — they each carry external-side-effect risk.

---

## Recommended allow-list patterns

### Read-only reviewer

For agents that analyze code or configs and emit a report, never modifying state.

```yaml
tools: ["Read", "Grep", "Glob"]
```

Examples: `code-reviewer`, `architect`, `python-reviewer`.

### Read-only with shell access

When the agent needs to run scripts (tests, linters, CLIs) but should not edit files. Common for validators that check `tsc --noEmit`, `pytest`, etc., and produce a report.

```yaml
tools: ["Read", "Grep", "Glob", "Bash"]
```

Examples: `e2e-runner`, `terraform-reviewer`.

### Code-modifying agent

For agents that fix issues by editing files. Include `Bash` only if they need to verify their fix by running tests/builds.

```yaml
tools: ["Read", "Write", "Edit", "Grep", "Glob"]
```

Add `Bash` if needed:

```yaml
tools: ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
```

Examples: `build-error-resolver`, `python-build-resolver`, `lint-fixer`.

### Documentation / doc-sync

Reads source-of-truth files and writes derived docs.

```yaml
tools: ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
```

Examples: `doc-updater`.

### Full access

For agents that genuinely need every tool — usually broad investigators or generalists. Prefer scoping if you can.

Omit `tools:` entirely to inherit all tools, or list explicitly:

```yaml
tools: ["Read", "Write", "Edit", "Bash", "Grep", "Glob", "WebFetch", "WebSearch"]
```

Examples: `general-purpose`.

---

## Picking the right allow-list

Walk through these questions in order:

1. **Does the agent emit a report only, with no file changes?** → Read-only.
2. **Does it need to run scripts (tests, builds, CLIs) to verify or gather info?** → Add `Bash`.
3. **Does it modify code or write new files?** → Add `Write` and `Edit`.
4. **Does it need network access (web docs, external APIs)?** → Add `WebFetch` / `WebSearch`.
5. **Does it interact with Slack, Gmail, etc.?** → Add the specific MCP tools, nothing else.

When in doubt, grant less. It's easy to widen the allow-list later when a missing tool surfaces; it's harder to claw back access after the agent develops habits that depend on it.

---

## Anti-patterns

- **Granting `Bash` "just in case".** A read-only reviewer with `Bash` access can drift into running scripts the user didn't expect.
- **Granting `Write` without `Edit`.** The agent will overwrite files instead of patching them, losing surrounding context.
- **Omitting `tools:` for any non-trivial agent.** The default (all tools) is rarely what you want.
- **Granting `Task` to an agent.** Agents-spawning-agents creates deep call stacks that are hard to debug. Let the main session do the orchestration.
