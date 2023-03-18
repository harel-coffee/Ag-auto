#!/bin/bash

# BNU Can CNRM GFDLG GFDLM
for runname in bcc 
do
cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg/rcp85_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/post_biofix/q_rcp85_w_precip_cnty_avg | while read LINE ; do
qsub $LINE
done
done
