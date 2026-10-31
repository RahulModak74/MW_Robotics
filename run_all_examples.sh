#!/usr/bin/env bash
# Run every example in order. Add --quick for a fast smoke test (~2 min total).
# Full run takes about 6-8 minutes on a laptop CPU.
set -e
cd "$(dirname "$0")"
for f in examples/0*.py; do
  echo; echo ">>> $f $*"
  python3 "$f" "$@"
done
echo; echo "All examples finished. Plots are in examples/outputs/"
