#!/bin/bash

cd /home/hnoorazar/analog_codes/04_analysis/parallel/qsubs

for runname in bcc-csm1-1-m BNU-ESM CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
qsub ./cnty_cnt_w_precip_45_$runname.sh
qsub ./cnty_cnt_w_precip_85_$runname.sh
qsub ./cnty_cnt_no_precip_45_$runname.sh
qsub ./cnty_cnt_no_precip_85_$runname.sh
done
