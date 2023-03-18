#!/bin/bash

# bcc-csm1-1-m BNU-ESM  CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
for runname in bcc-csm1-1-m BNU-ESM  CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
cd /home/hnoorazar/analog_codes/03_find_analogs/location_level/rcp45_qsubs/
qsub q_rcp45_w_precip_w_gen3_$runname.sh
done
