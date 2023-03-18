#!/bin/bash

cd /home/hnoorazar/analog_codes/04_analysis/rcp85_qsubs/
for runname in 2026_2050 2051_2075 2076_2095
do
# qsub ./q_rcp85_no_precip_no_gen3_$runname.sh
qsub ./q_rcp85_w_precip_w_gen3_$runname.sh
qsub ./q_rcp85_no_precip_w_gen3_$runname.sh
# qsub ./q_rcp85_w_precip_no_gen3_$runname.sh
done

cd /home/hnoorazar/analog_codes/04_analysis/rcp45_qsubs/
for runname in 2026_2050 2051_2075 2076_2095
do
# qsub ./q_rcp45_no_precip_no_gen3_$runname.sh
qsub ./q_rcp45_w_precip_w_gen3_$runname.sh
qsub ./q_rcp45_no_precip_w_gen3_$runname.sh
# qsub ./q_rcp45_w_precip_no_gen3_$runname.sh
done
