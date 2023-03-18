#!/bin/bash

cd /home/hnoorazar/chilling_codes/current_draft/02_for_Springer/01_generate_02_stats/qsubs/

for runname in sept
do
qsub ./q_model_dyn_$runname.sh
qsub ./q_obs_dyn_$runname.sh
done
