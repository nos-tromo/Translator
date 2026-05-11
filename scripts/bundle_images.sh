#!/usr/bin/env bash
set -euo pipefail

# YYYY-MM-DD[-<short-sha>]; sha omitted when not in a git repo.
# Override by exporting TRANSLATOR_VERSION beforehand.
_git_sha=$(git rev-parse --short HEAD 2>/dev/null || true)
export TRANSLATOR_VERSION="${TRANSLATOR_VERSION:-$(date +%Y-%m-%d)${_git_sha:+-${_git_sha}}}"
echo "TRANSLATOR_VERSION=$TRANSLATOR_VERSION"

# Persist the version so production hosts can run 'make no-build-*' without
# git or the original build date. Copy this file alongside docker-compose.yml.
echo "$TRANSLATOR_VERSION" > .translator-version

# Build locally-defined services
docker compose build

# Partition compose's image list and ensure local tag bindings exist:
#   built  = local-only names like "translator-backend" (already tagged by build)
#
# Docker Desktop sometimes drops the name:tag binding when you pull
# `name:tag@digest`, leaving only the digest. We re-tag explicitly so
# `docker save` produces a tarball that loads back with both tag and digest
# bindings — which compose needs for its `image: name:tag@digest` references.
built=()
pulled=()
while IFS= read -r img; do
  [[ -z "$img" ]] && continue
  if [[ "$img" == */* ]]; then
    if [[ "$img" =~ ^(.+):([^@]+)@(sha256:[a-f0-9]+)$ ]]; then
      name="${BASH_REMATCH[1]}"
      tag="${BASH_REMATCH[2]}"
      digest="${BASH_REMATCH[3]}"
      docker tag "${name}@${digest}" "${name}:${tag}"
      pulled+=("${name}:${tag}")
    else
      pulled+=("$img")
    fi
  else
    built+=("$img")
  fi
done < <(docker compose config --images)

echo "Built images:  ${built[*]:-<none>}"

if (( ${#built[@]} > 0 )); then
  docker save "${built[@]}" | gzip > "translator-built-${TRANSLATOR_VERSION}.tar.gz"
fi

echo "Wrote: translator-built-${TRANSLATOR_VERSION}.tar.gz"
