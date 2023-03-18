#!/bin/bash

cd /home/hnoorazar/chilling_codes/current_draft/02_for_Springer/01_generate_02_stats

for runname in sept
do
cp 02_q_modeled_dynamic.sh ./qsubs/q_model_dyn_$runname.sh
sed -i s/chill_sea/$runname/g ./qsubs/q_model_dyn_$runname.sh
done

for runname in sept
do
cp 02_q_observed_dynamic.sh ./qsubs/q_obs_dyn_$runname.sh
sed -i s/chill_sea/$runname/g ./qsubs/q_obs_dyn_$runname.sh
done

