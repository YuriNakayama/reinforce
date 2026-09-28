#!/usr/bin/env node
/**
 * PostCompact Hook — Re-inject CLAUDE.md after context compaction
 *
 * Compaction summarizes prior turns; project rules from CLAUDE.md (and the
 * rules/*.md files it references) tend to get washed out of the summary. This
 * hook reads CLAUDE.md (plus any .claude/rules/*.md it links to) and pushes
 * the full text back into model context via hookSpecificOutput.additionalContext
 * so the post-compact session keeps following the same rules.
 */

const fs = require('fs');
const path = require('path');

function readIfExists(p) {
  try {
    if (fs.existsSync(p) && fs.statSync(p).isFile()) {
      return fs.readFileSync(p, 'utf8');
    }
  } catch (_) {}
  return null;
}

async function main() {
  // Drain stdin (Claude Code passes hook input as JSON; we don't need it here
  // but must consume it so the parent process can exit cleanly).
  let _stdin = '';
  process.stdin.on('data', c => (_stdin += c));
  await new Promise(resolve => process.stdin.on('end', resolve));

  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const claudeMdPath = path.join(projectDir, '.claude', 'CLAUDE.md');
  const rulesDir = path.join(projectDir, '.claude', 'rules');

  const claudeMd = readIfExists(claudeMdPath);
  if (!claudeMd) {
    // Nothing to inject; exit silently so we don't pollute the post-compact view.
    process.exit(0);
  }

  const sections = [
    `# Re-injected CLAUDE.md (post-compact)\n\nThe following project instructions were re-loaded automatically after context compaction. They override default behavior — follow them.\n`,
    `## .claude/CLAUDE.md\n\n${claudeMd}`
  ];

  // Best-effort: also re-inject the always-loaded rule file (security.md).
  // Other rule files are auto-loaded by glob and will be re-attached by the
  // harness when the relevant files are touched, so we don't blanket-load them.
  const securityRules = readIfExists(path.join(rulesDir, 'security.md'));
  if (securityRules) {
    sections.push(`## .claude/rules/security.md (always loaded)\n\n${securityRules}`);
  }

  const additionalContext = sections.join('\n\n');

  const output = {
    hookSpecificOutput: {
      hookEventName: 'PostCompact',
      additionalContext
    },
    systemMessage: 'CLAUDE.md re-injected after compaction.',
    suppressOutput: true
  };

  process.stdout.write(JSON.stringify(output));
  process.exit(0);
}

main().catch(err => {
  console.error('[PostCompact] Error:', err.message);
  process.exit(0); // Never block on errors
});
