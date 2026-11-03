#!/usr/bin/env bash
# list_code -- collate all files of a given extension into one txt file
# Usage: list_code --<ext> [--out <output_file>] [--dir <root_dir>]
# Example: list_code --ts
#          list_code --py --out my_project.txt
#          list_code --ts --dir /path/to/project

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
ROOT_DIR="$(pwd)"
OUTPUT_FILE=""
EXT=""

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --*)
      # first --xxx that isn't --out or --dir is the extension
      if [[ "$1" == "--out" ]]; then
        OUTPUT_FILE="$2"; shift 2
      elif [[ "$1" == "--dir" ]]; then
        ROOT_DIR="$2"; shift 2
      elif [[ -z "$EXT" ]]; then
        EXT="${1#--}"; shift        # strip leading --
      else
        echo "Unknown option: $1" >&2; exit 1
      fi
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$EXT" ]]; then
  echo "Usage: list_code --<extension> [--out <output_file>] [--dir <root_dir>]"
  echo "  e.g. list_code --ts"
  echo "  e.g. list_code --py --out snapshot.txt --dir ./src"
  exit 1
fi

# default output filename
if [[ -z "$OUTPUT_FILE" ]]; then
  OUTPUT_FILE="${ROOT_DIR}/code_snapshot_${EXT}.txt"
fi

# ── collect files ─────────────────────────────────────────────────────────────
mapfile -t FILES < <(
  find "$ROOT_DIR" \
    -type f \
    -name "*.${EXT}" \
    ! -path "*/.git/*" \
    ! -path "*/node_modules/*" \
    ! -path "*/__pycache__/*" \
    ! -path "*/.venv/*" \
    | sort
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No .${EXT} files found under: $ROOT_DIR"
  exit 0
fi

# ── write output ──────────────────────────────────────────────────────────────
: > "$OUTPUT_FILE"          # truncate / create

PREV_DIR=""
FILE_COUNT=0

for FILEPATH in "${FILES[@]}"; do
  # path relative to root for cleaner display
  REL_PATH="${FILEPATH#$ROOT_DIR/}"
  DIR_PATH="$(dirname "$REL_PATH")"
  FILENAME="$(basename "$REL_PATH")"

  # print a directory banner whenever we enter a new directory
  if [[ "$DIR_PATH" != "$PREV_DIR" ]]; then
    if [[ -n "$PREV_DIR" ]]; then
      printf '\n' >> "$OUTPUT_FILE"
    fi
    if [[ "$DIR_PATH" == "." ]]; then
      printf '# %-70s\n\n' "(root)" >> "$OUTPUT_FILE"
    else
      printf '# %-70s\n\n' "$DIR_PATH/" >> "$OUTPUT_FILE"
    fi
    PREV_DIR="$DIR_PATH"
  fi

  printf '===== %s =====\n' "$REL_PATH" >> "$OUTPUT_FILE"
  cat "$FILEPATH"                       >> "$OUTPUT_FILE"
  printf '\n'                           >> "$OUTPUT_FILE"

  FILE_COUNT=$(( FILE_COUNT + 1 ))
done

echo "✓  Wrote $FILE_COUNT .${EXT} file(s) → $OUTPUT_FILE"
