#!/usr/bin/env bash
# patch-generation-limits.sh — Relax generation limits for Docker builds.
#
# Patches applied per file:
#   1. LOCAL_MAX_DURATION caps raised (regex matches any values)
#   2. Duration options list expanded
#   3. FPS options list expanded
#   4. Video resolution options (patchable, kept at model-safe defaults)
#
# Fails the build if expected patterns aren't found (signals upstream
# refactored and this script needs updating).
#
# Usage: bash patch-generation-limits.sh file1.tsx [file2.tsx ...]
#
# NOTE on resolution: The LTX model was trained at 540p/720p/1080p.
# Higher resolutions (1440p/2160p) exist only for API mode and would
# also need backend RESOLUTION_MAP changes to work locally. Only add
# them here if you've tested with your GPU + model combination.

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────
# Max duration per resolution (seconds).
# Add new resolutions here AND in the options below if needed.
MAX_540P=30
MAX_720P=25
MAX_1080P=20

# Duration choices shown in the dropdown (filtered by per-resolution max)
DURATION_OPTIONS="5, 10, 15, 20, 25, 30"

# FPS choices for local generation
FPS_OPTIONS="24, 25, 50, 60"

# Video resolution choices for local generation.
# ⚠ Only add resolutions that are in the backend's RESOLUTION_MAP.
#   Default local map: 540p, 720p, 1080p.
#   To add 1440p/2160p you must also patch the backend handler.
RESOLUTION_OPTIONS_ASC="'540p', '720p', '1080p'"
RESOLUTION_OPTIONS_DESC="'1080p', '720p', '540p'"
# ────────────────────────────────────────────────────────────────────────

PAT_LIMITS="'1080p':\s*[0-9]+"
PAT_DURATION='\[5,\s*6,\s*8,\s*10,\s*20\]'
PAT_FPS='\[24,\s*25,\s*50\]'
PAT_RES_ASC="\['540p',\s*'720p',\s*'1080p'\]"
PAT_RES_DESC="\['1080p',\s*'720p',\s*'540p'\]"

for file in "$@"; do
  echo "Checking $file ..."

  # ── Check patterns exist ──
  if ! grep -qP "$PAT_LIMITS" "$file"; then
    echo "ERROR: Duration limits pattern not found in $file"
    echo "Upstream likely refactored LOCAL_MAX_DURATION — update this patch script."
    exit 1
  fi
  if ! grep -qP "$PAT_DURATION" "$file"; then
    echo "ERROR: Duration options pattern not found in $file"
    echo "Upstream likely changed the duration array — update this patch script."
    exit 1
  fi
  if ! grep -qP "$PAT_FPS" "$file"; then
    echo "ERROR: FPS options pattern not found in $file"
    echo "Upstream likely changed the FPS array — update this patch script."
    exit 1
  fi
  if ! grep -qP "$PAT_RES_ASC|$PAT_RES_DESC" "$file"; then
    echo "ERROR: Resolution options pattern not found in $file"
    echo "Upstream likely changed the resolution array — update this patch script."
    exit 1
  fi

  # ── Patch max duration caps ──
  sed -i -E "s/'540p':\s*[0-9]+/'540p': $MAX_540P/g"   "$file"
  sed -i -E "s/'720p':\s*[0-9]+/'720p': $MAX_720P/g"    "$file"
  sed -i -E "s/'1080p':\s*[0-9]+/'1080p': $MAX_1080P/g" "$file"

  # ── Patch duration options list ──
  sed -i -E "s/\[5,\s*6,\s*8,\s*10,\s*20\]/[$DURATION_OPTIONS]/g" "$file"

  # ── Patch FPS options list ──
  sed -i -E "s/\[24,\s*25,\s*50\]/[$FPS_OPTIONS]/g" "$file"

  # ── Patch resolution options (both orderings) ──
  sed -i -E "s/\['540p',\s*'720p',\s*'1080p'\]/[$RESOLUTION_OPTIONS_ASC]/g" "$file"
  sed -i -E "s/\['1080p',\s*'720p',\s*'540p'\]/[$RESOLUTION_OPTIONS_DESC]/g" "$file"

  echo "  ✓ Patched"
done

echo ""
echo "Done."
echo "  Caps: 540p→${MAX_540P}s, 720p→${MAX_720P}s, 1080p→${MAX_1080P}s"
echo "  Durations: [${DURATION_OPTIONS}]"
echo "  FPS: [${FPS_OPTIONS}]"
echo "  Resolutions: [${RESOLUTION_OPTIONS_ASC}]"
