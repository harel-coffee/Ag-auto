#!/bin/bash

for runname in bcc-csm1-1-m BNU-ESM  CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
cd /home/hnoorazar/analog_codes/03_find_analogs/location_level/merge/rcp85_qsubs/
qsub q_rcp85_w_precip_w_gen3_$runname.sh
done
