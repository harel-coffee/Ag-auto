#!/bin/bash

cd /home/hnoorazar/codling_moth_2021/drivers/00_local_modeled/CM_qsubs/
for runname in {1..60}
do
qsub ./q_$runname.sh
done
