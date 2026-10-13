#!/usr/bin/env bash
# Publish dist/ to the gh-pages branch, for a free public URL.
#
# Netlify builds on every push that touches site/, and those builds are a metered
# resource. This needs no build, no CLI to install and no new account: it writes
# a commit straight from dist/ using the git credentials that already push this
# repo, and GitHub Pages serves the result.
#
# It never touches your working tree, your index or your current branch. The
# commit is assembled with plumbing against a temporary index file, so an
# interrupted run leaves nothing behind to clean up.
#
# One manual step, once: repo Settings -> Pages -> Source = gh-pages branch.
#
# Usage: scripts/publish_pages.sh
set -euo pipefail

cd "$(dirname "$0")/.."
BRANCH=gh-pages

python3 scripts/build_dist.py >/dev/null
[ -f dist/index.html ] || { echo "dist/index.html missing after build"; exit 1; }

# GitHub Pages runs Jekyll unless told not to, and Jekyll drops files whose
# names begin with an underscore. Nothing here starts with one today; this is
# so that stops being something anybody has to remember.
touch dist/.nojekyll

# A temporary index, so the real one is untouched. -f because dist/ is ignored,
# which is correct for the repo and irrelevant here.
GIT_INDEX_FILE="$(mktemp -t veridis-pages-index.XXXXXX)"
export GIT_INDEX_FILE
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
rm -f "$GIT_INDEX_FILE"

git --work-tree=dist add -A -f .
TREE="$(git write-tree)"

# Keep the branch's history rather than orphaning it every publish, so the
# deploy log is readable and a bad publish can be pointed back one commit.
if PARENT="$(git rev-parse --verify --quiet "refs/heads/$BRANCH")"; then
  COMMIT="$(git commit-tree "$TREE" -p "$PARENT" -m "publish $(date -u +%Y-%m-%dT%H:%MZ)")"
else
  COMMIT="$(git commit-tree "$TREE" -m "publish $(date -u +%Y-%m-%dT%H:%MZ)")"
fi
git update-ref "refs/heads/$BRANCH" "$COMMIT"

SLUG="$(git config --get remote.origin.url \
  | sed -E 's#(git@|https://)github.com[:/]##; s/\.git$//')"
USER="${SLUG%%/*}"
REPO="${SLUG##*/}"

echo "built $(find dist -type f | wc -l | tr -d ' ') files, commit ${COMMIT:0:10}"
git push -f origin "$BRANCH"
echo
echo "https://${USER}.github.io/${REPO}/"
echo
echo "If that 404s, set Settings -> Pages -> Source to the gh-pages branch once."
