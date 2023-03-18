#!/bin/bash

cd /home/hnoorazar/chilling_codes/current_draft/02/three_seasons/qsubs/

for runname in sept mid_sept oct mid_oct nov mid_nov
do
qsub ./q_model_dyn_$runname.sh
qsub ./q_obs_dyn_$runname.sh
done
