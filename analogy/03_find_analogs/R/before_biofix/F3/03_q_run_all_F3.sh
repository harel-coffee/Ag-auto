#!/bin/bash

cd /home/hnoorazar/analog_codes/03_find_analogs/rcp85_qsubs_F3/
for runname in bcc-csm1-1-m BNU-ESM CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
#qsub ./q_rcp85_no_precip_no_gen3_$runname.sh
qsub ./q_rcp85_w_precip_w_gen3_$runname.sh
qsub ./q_rcp85_no_precip_w_gen3_$runname.sh
#qsub ./q_rcp85_w_precip_no_gen3_$runname.sh
done

cd /home/hnoorazar/analog_codes/03_find_analogs/rcp45_qsubs_F3/
for runname in bcc-csm1-1-m BNU-ESM CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
#qsub ./q_rcp45_no_precip_no_gen3_$runname.sh
qsub ./q_rcp45_w_precip_w_gen3_$runname.sh
qsub ./q_rcp45_no_precip_w_gen3_$runname.sh
#qsub ./q_rcp45_w_precip_no_gen3_$runname.sh
done
