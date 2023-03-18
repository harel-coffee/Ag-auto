#!/bin/bash
cd /home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/

outer=1
for emission in rcp45 rcp85
do
  for a_model in "bcc-csm1-1-m" "BNU-ESM" "CanESM2" "CNRM-CM5" "GFDL-ESM2G" "GFDL-ESM2M"
  do
    for location in "48.96875_-119.40625" "47.21875_-119.90625" "46.34375_-119.21875" "46.46875_-120.40625" "47.40625_-120.34375"
    do
      cp 00_LM_CM_template.sh         ./CM_qsubs/q_$outer.sh
      sed -i s/outer/"$outer"/g       ./CM_qsubs/q_$outer.sh
      sed -i s/emission/"$emission"/g ./CM_qsubs/q_$outer.sh
      sed -i s/a_model/"$a_model"/g       ./CM_qsubs/q_$outer.sh
      sed -i s/location/"$location"/g ./CM_qsubs/q_$outer.sh
      let "outer+=1" 
    done
  done
done



