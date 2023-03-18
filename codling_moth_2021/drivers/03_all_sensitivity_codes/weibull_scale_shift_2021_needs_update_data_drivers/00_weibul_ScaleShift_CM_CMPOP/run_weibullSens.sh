#!/bin/bash

cd /home/hnoorazar/codling_moth_2021/drivers/03_all_sensitivity_codes/00_weibul_ScaleShift_CM_CMPOP/qsubs/
for runname in {1..148}
do
qsub ./q_$runname.sh
done
