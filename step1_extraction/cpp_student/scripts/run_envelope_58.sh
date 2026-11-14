#!/usr/bin/env bash
# run_envelope_58.sh - the two-level force envelope with a cheap search (N 16, H 15, S 2), synchronous, MW-gated, on the 58 tests.
# Measured with the Python search: 50/58, 82 % of steps within 20 ms (results/two_envelope_50of58.txt).
#   MW < lo            -> no correction
#   lo <= MW < full    -> soft envelope (SOFT x the flight computer's correction)
#   MW >= full         -> hard envelope (the full correction)
#   ST=<student.pt> E=<held-out motion folder> [FBW_CPP=1] ./run_envelope_58.sh
source "$(dirname "$0")/common.sh"
O=$OUT/envelope${TAG}; mkdir -p "$O"
export FBW_PID=1 FBW_PID_SOFT=${FBW_PID_SOFT:-0.6} FBW_PID_LO=${FBW_PID_LO:-0.0136} FBW_PID_FULL=${FBW_PID_FULL:-0.025}
export FBW2_THR=${FBW2_THR:-0.0136} FBW_N=${FBW_N:-16} FBW_H=${FBW_H:-15} FBW_S=${FBW_S:-2}
case ${1:-all} in score) score ;; *) all_58; score ;; esac
