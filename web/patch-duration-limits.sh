#!/usr/bin/env bash
# patch-duration-limits.sh — Relax duration limits & options for Docker builds.
#
# Two patches applied per file:
#   1. LOCAL_MAX_DURATION caps raised (regex matches any values)
#   2. Duration options list expanded to 5s increments up to max
#
# Fails the build if expected patterns aren't found (signals upstream
# refactored and this script needs updating).
#
# Usage: bash patch-duration-limits.sh file1.tsx [file2.tsx ...]

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
MAX_540P=30
MAX_720P=25
MAX_1080P=20

# Duration choices shown in the dropdown (filtered by per-resolution max)
DURATION_OPTIONS="5, 10, 15, 20, 25, 30"
# ────────────────────────────────────────────────────────────────────────

PAT_LIMITS="'1080p':\s*[0-9]+"
PAT_OPTIONS='\[5,\s*6,\s*8,\s*10,\s*20\]'

for file in "$@"; do
  # ── Check patterns exist ──
  if ! grep -qP "$PAT_LIMITS" "$file"; then
    echo "ERROR: Duration limits pattern not found in $file"
    echo "Upstream likely refactored LOCAL_MAX_DURATION — update this patch script."
    exit 1
  fi
  if ! grep -qP "$PAT_OPTIONS" "$file"; then
    echo "ERROR: Duration options pattern not found in $file"
    echo "Upstream likely changed the duration array — update this patch script."
    exit 1
  fi

  # ── Patch max duration caps ──
  sed -i -E "s/'540p':\s*[0-9]+/'540p': $MAX_540P/g"   "$file"
  sed -i -E "s/'720p':\s*[0-9]+/'720p': $MAX_720P/g"    "$file"
  sed -i -E "s/'1080p':\s*[0-9]+/'1080p': $MAX_1080P/g" "$file"

  # ── Patch duration options list ──
  sed -i -E "s/\[5,\s*6,\s*8,\s*10,\s*20\]/[$DURATION_OPTIONS]/g" "$file"

  echo "Patched $file"
done

echo "Done. Caps: 540p→${MAX_540P}s, 720p→${MAX_720P}s, 1080p→${MAX_1080P}s"
echo "Options: [${DURATION_OPTIONS}] (filtered by resolution cap)"
