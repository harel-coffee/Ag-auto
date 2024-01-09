#!/bin/bash

cd /home/h.noorazar/rangeland/seasonal_variables/00_gridwise/qsubs

for runname in {1..25}
do
sbatch ./seasonal_vars_gridwise_$runname.sh
done
