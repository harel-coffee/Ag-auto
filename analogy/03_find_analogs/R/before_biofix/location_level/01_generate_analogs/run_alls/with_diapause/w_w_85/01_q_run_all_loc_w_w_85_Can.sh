#!/bin/bash

# bcc BNU Can CNRM GFDLG GFDLM
for runname in Can 
do
cd /home/hnoorazar/analog_codes/03_find_analogs/location_level/rcp85_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/q_rcp85_w_precip_w_gen3 | while read LINE ; do
qsub $LINE
done
done
