#!/bin/bash

cd /home/hnoorazar/chilling_codes/current_draft/02/three_seasons/

for runname in sept mid_sept oct mid_oct nov mid_nov
do
cp q_modeled_dynamic.sh ./qsubs/q_model_dyn_$runname.sh
sed -i s/chill_sea/$runname/g ./qsubs/q_model_dyn_$runname.sh
done

for runname in sept mid_sept oct mid_oct nov mid_nov
do
cp q_observed_dynamic.sh ./qsubs/q_obs_dyn_$runname.sh
sed -i s/chill_sea/$runname/g ./qsubs/q_obs_dyn_$runname.sh
done

