#!/usr/bin/env bash
# run_gated_58.sh - the MW-gated flight computer with follow-through, full search (config D: N 64, H 20, S 4), on the 58 tests.
# Measured with the Python search: 58/58, median decision 2.6 ms, 78.7 % of steps within 20 ms (results/gated_58of58.md).
#   ST=<student.pt> E=<held-out motion folder> ./run_gated_58.sh            Python search (the measured 58/58)
#   FBW_CPP=1 ST=... E=... ./run_gated_58.sh                                 same search in the C++ kernel (build ../kernel first)
#   FBW_CPP=1 FBW_LA_DT=0.002 ...                                            UNMEASURED: 2 ms look-ahead physics, about half the cost
# One job at a time; the Python search takes hours on a laptop. Resumable.
source "$(dirname "$0")/common.sh"
O=$OUT/gated${TAG}; mkdir -p "$O"
export FBW_H=${FBW_H:-20} FBW_N=${FBW_N:-64} FBW_S=${FBW_S:-4} FBW2_RELEASE=1 FBW2_THR=${FBW2_THR:-0.0136}
case ${1:-all} in score) score ;; *) all_58; score ;; esac
