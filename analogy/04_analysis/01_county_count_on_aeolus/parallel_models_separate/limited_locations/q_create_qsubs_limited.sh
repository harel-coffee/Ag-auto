#!/bin/bash

cd /home/hnoorazar/analog_codes/04_analysis/parallel/limited_locations

for runname in bcc-csm1-1-m BNU-ESM CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
cp q_cnty_cnt_w_precip_45_limited.sh    ./qsubs/cnty_cnt_w_precip_45_$runname.sh
cp q_cnty_cnt_w_precip_85_limited.sh    ./qsubs/cnty_cnt_w_precip_85_$runname.sh
cp q_cnty_cnt_no_precip_45_limited.sh ./qsubs/cnty_cnt_no_precip_45_$runname.sh
cp q_cnty_cnt_no_precip_85_limited.sh ./qsubs/cnty_cnt_no_precip_85_$runname.sh

sed -i s/all_model_names/$runname/g ./qsubs/cnty_cnt_w_precip_45_$runname.sh
sed -i s/all_model_names/$runname/g ./qsubs/cnty_cnt_w_precip_85_$runname.sh
sed -i s/all_model_names/$runname/g ./qsubs/cnty_cnt_no_precip_45_$runname.sh
sed -i s/all_model_names/$runname/g ./qsubs/cnty_cnt_no_precip_85_$runname.sh
done


