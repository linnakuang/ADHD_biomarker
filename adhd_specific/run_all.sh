#!/bin/bash
# Run the D-series. D1 first (D4/D7 read its table), then the rest in parallel.
cd "$(dirname "$0")"
PY=~/miniforge/bin/python
$PY d1_adhd_specificity.py > results/d1_nohup.log 2>&1
for s in d4_presentation d5_prepost_dx d6_dual_diagnosis d7_utilization ${EXTRA}; do
  $PY $s.py > results/${s%%_*}_nohup.log 2>&1 &
done
wait
echo ALL DONE
