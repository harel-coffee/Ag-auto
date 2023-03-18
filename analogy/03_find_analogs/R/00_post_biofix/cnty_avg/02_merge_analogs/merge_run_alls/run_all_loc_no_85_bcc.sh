#!/bin/bash

for runname in bcc-csm1-1-m BNU-ESM  CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg/merge/rcp85_qsubs/
qsub q_rcp85_no_precip_$runname.sh
done
