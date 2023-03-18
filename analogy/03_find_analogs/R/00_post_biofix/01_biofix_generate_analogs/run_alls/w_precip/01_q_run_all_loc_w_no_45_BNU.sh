#!/bin/bash

# bcc BNU Can CNRM GFDLG GFDLM
for runname in BNU 
do
cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs/rcp45_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/post_biofix/q_rcp45_w_precip | while read LINE ; do
qsub $LINE
done
done
