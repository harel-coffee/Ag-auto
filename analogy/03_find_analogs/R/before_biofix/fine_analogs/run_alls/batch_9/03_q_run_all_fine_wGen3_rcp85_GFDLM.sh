#!/bin/bash

for runname in GFDLM
do
cd /home/hnoorazar/analog_codes/03_find_analogs/fine/rcp85_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/q_rcp85_wgen3_batch_9 | while read LINE ; do
qsub $LINE
done
done
