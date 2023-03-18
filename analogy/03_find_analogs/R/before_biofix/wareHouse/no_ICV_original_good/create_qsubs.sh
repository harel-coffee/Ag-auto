#!/bin/bash

cd /home/hnoorazar/analog_codes/03_find_analogs/R_codes/

mkdir rcp45
for runname in bcc-csm1-1-m BNU-ESM CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
cp template.sh ./rcp45/q_rcp45_$runname.sh
sed -i s/emission_type/rcp45/g ./rcp45/q_rcp45_$runname.sh
sed -i s/model_type/"$runname"/g ./rcp45/q_rcp45_$runname.sh
done

mkdir rcp85
for runname in bcc-csm1-1-m BNU-ESM CanESM2 CNRM-CM5 GFDL-ESM2G GFDL-ESM2M
do
cp template.sh ./rcp85/q_rcp85_$runname.sh
sed -i s/emission_type/rcp85/g ./rcp85/q_rcp85_$runname.sh
sed -i s/model_type/"$runname"/g ./rcp85/q_rcp85_$runname.sh
done