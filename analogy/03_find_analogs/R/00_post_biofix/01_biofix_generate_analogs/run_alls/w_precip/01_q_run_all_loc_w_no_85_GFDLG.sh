#!/bin/bash

# bcc BNU Can CNRM GFDLG GFDLM
for runname in GFDLG 
do
cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs/rcp85_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/post_biofix/q_rcp85_w_precip | while read LINE ; do
qsub $LINE
done
done
