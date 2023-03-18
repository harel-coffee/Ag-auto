#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=00:05:00
#PBS -l mem=1gb
#PBS -q fast

# bcc BNU Can CNRM GFDLG GFDLM
for runname in Can 
do
cd /home/hnoorazar/analog_codes/00_post_biofix/02_find_analogs_county_avg/rcp45_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/post_biofix/q_rcp45_no_precip_cnty_avg | while read LINE ; do
qsub $LINE
done
done
