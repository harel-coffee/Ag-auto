#!/bin/bash
cd /home/hnoorazar/codling_moth_2021/drivers/03_all_sensitivity_codes/00_weibul_ScaleShift_CM_CMPOP/

outer=1

for categories in "bcc-csm1-1-m" "BNU-ESM" "CanESM2" "CNRM-CM5" "GFDL-ESM2G" "GFDL-ESM2M" "observed"
do
  for scale_shift in 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20
  do
    cp 00_weibellScaleCMPOP_template.sh    ./qsubs/q_$outer.sh
    sed -i s/outer/"$outer"/g             ./qsubs/q_$outer.sh
    sed -i s/categories/"$categories"/g   ./qsubs/q_$outer.sh
    sed -i s/scale_shift/"$scale_shift"/g ./qsubs/q_$outer.sh
    let "outer+=1" 
  done
done



