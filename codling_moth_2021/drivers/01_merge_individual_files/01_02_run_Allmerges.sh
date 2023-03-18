#!/bin/bash

cd /home/hnoorazar/codling_moth_2021/drivers/01_merge_individual_files/qsubs/
for runname in {1..12}
do
qsub ./q_$runname.sh
done
