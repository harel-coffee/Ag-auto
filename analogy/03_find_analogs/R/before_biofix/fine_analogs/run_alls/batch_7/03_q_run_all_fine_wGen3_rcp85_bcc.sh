#!/bin/bash

###  BNU Can CNRM GFDLG GFDLM
for runname in bcc
do
cd /home/hnoorazar/analog_codes/03_find_analogs/fine/rcp85_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/q_rcp85_wgen3_batch_7 | while read LINE ; do
qsub $LINE
done
done
