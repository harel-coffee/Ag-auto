#!/bin/bash

for model_carbon in bcc_csm1_1_m/rcp45 bcc_csm1_1_m/rcp85 BNU_ESM/rcp45 BNU_ESM/rcp85 CanESM2/rcp45 CanESM2/rcp85 CNRM_CM5/rcp45 CNRM_CM5/rcp85 GFDL_ESM2G/rcp45 GFDL_ESM2G/rcp85 GFDL_ESM2M/rcp45 GFDL_ESM2M/rcp85
do
cd /home/hnoorazar/analog_codes/03_find_analogs/R_codes/matchit/$model_carbon

for (( i = 1; i <= 295; i++ ))
do
qsub ./qsub_set$i
done

cd ..
done
