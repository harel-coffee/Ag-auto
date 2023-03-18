#!/bin/bash
cd /home/hnoorazar/lagoon_codes/01_run_offs/01_cum_runs/qsubs

for runname in {1..20}
do
qsub ./q_ann_$runname.sh
qsub ./q_month_$runname.sh
qsub ./q_chunk_$runname.sh
qsub ./q_wtr_yr_$runname.sh
done
