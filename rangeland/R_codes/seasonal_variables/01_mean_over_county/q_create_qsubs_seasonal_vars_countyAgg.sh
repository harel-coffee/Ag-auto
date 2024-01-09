#!/bin/bash
cd /home/h.noorazar/rangeland/seasonal_variables/01_mean_over_county

outer=1

for a_state in Alabama Arkansas California Colorado Florida Georgia Idaho Illinois Iowa Kansas Kentucky Louisiana Mississippi Missouri Montana Nebraska New_Mexico North_Dakota Oklahoma Oregon South_Dakota Tennessee Texas Virginia Wyoming
do
  cp q_temp_seasonal_vars_countyAgg.sh ./qsubs/seasonal_vars_countyAgg_$outer.sh
  sed -i s/outer/"$outer"/g            ./qsubs/seasonal_vars_countyAgg_$outer.sh
  sed -i s/a_state/"$a_state"/g        ./qsubs/seasonal_vars_countyAgg_$outer.sh
  let "outer+=1" 
done