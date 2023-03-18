#!/bin/bash

for runname in Can
do
cd /home/hnoorazar/analog_codes/03_find_analogs/fine/rcp45_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/q_rcp45_wgen3_batch_8 | while read LINE ; do
qsub $LINE
done
done