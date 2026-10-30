#!/usr/bin/env bash
# Fails when a commit in BASE..HEAD is authored or committed under an AI
# agent's identity, or carries a Co-authored-by line naming one: GitHub lists
# both as repository contributors, and master was rewritten on 30.10.2026 to
# remove such entries. A branch started before that rewrite also fails here,
# because its old commits are not on master: rebase it onto origin/master.
#
# usage: scripts/check_commit_authors.sh BASE HEAD
set -euo pipefail

base=${1:-}
head=${2:?usage: check_commit_authors.sh BASE HEAD}
# Identities: agent names and mail domains, not first names a person may have.
agent_ids='hermes|anthropic\.com|codex|copilot|cursoragent|devin-ai|openai\.com'
# Co-author lines: any agent name.
agent_coauthors='anthropic|claude|hermes|codex|copilot|cursor|devin|openai'

# A new branch, or a base that a force push left unreachable: check the head
# commit alone.
if [ -z "$base" ] || [ "$base" = 0000000000000000000000000000000000000000 ] \
  || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
  range=("-1" "$head")
else
  range=("$base..$head")
fi

bad=0
while IFS=$'\t' read -r sha who; do
  echo "::error::$sha is attributed to an agent identity: $who"
  bad=1
done < <(git log --format='%h%x09%an <%ae> / %cn <%ce>' "${range[@]}" \
  | grep -iE "$agent_ids" || true)

while IFS= read -r sha; do
  lines=$(git log -1 --format=%B "$sha" | grep -iE '^co-authored-by:' | grep -iE "$agent_coauthors" || true)
  if [ -n "$lines" ]; then
    echo "::error::$sha has an agent co-author line: $lines"
    bad=1
  fi
done < <(git rev-list --abbrev-commit "${range[@]}")

if [ "$bad" -ne 0 ]; then
  echo "Commits must be authored by people, without agent co-author lines."
  echo "Fix with an interactive rebase (or rebase onto origin/master if the"
  echo "branch predates the 30.10.2026 history rewrite)."
  exit 1
fi
echo "commit authors: ok ($(git rev-list --count "${range[@]}") commits)"
