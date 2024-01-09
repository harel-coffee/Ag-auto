#!/bin/bash

cd /home/h.noorazar/rangeland/seasonal_variables/01_mean_over_county/qsubs

for outer in {1..25}
do
sbatch ./seasonal_vars_countyAgg_$outer.sh
done
