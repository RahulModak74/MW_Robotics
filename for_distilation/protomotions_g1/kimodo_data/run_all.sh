#!/usr/bin/env bash
# run_all.sh - whole chain, unattended: encode -> generate -> convert -> collect -> evallibs -> train -> jobs -> eval -> agg
set -e
cd "$(dirname "$0")"; mkdir -p logs
# the 4-bit encoder needs ~5.5 GB of the 6 GB GPU: wait for any other Kimodo generation (e.g. quick100.sh) to finish first
while pgrep -f "gen_motions.py --prompts .*quick100" > /dev/null; do sleep 30; done
for st in encode generate convert collect; do
  echo "[$(date +%T)] pipeline $st"; ./run_pipeline.sh $st > logs/$st.log 2>&1 || { echo "FAILED at $st (logs/$st.log)"; exit 1; }
done
for st in evallibs train jobs eval agg; do
  echo "[$(date +%T)] study $st"; ./run_study.sh $st > logs/study_$st.log 2>&1 || { echo "FAILED at study $st (logs/study_$st.log)"; exit 1; }
done
echo "[$(date +%T)] ALL DONE"
