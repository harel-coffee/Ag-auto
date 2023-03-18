#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=00:05:00
#PBS -l mem=1gb
#PBS -q fast

## bcc BNU Can CNRM GFDLG GFDLM
for runname in GFDLG 
do
cd /home/hnoorazar/analog_codes/03_find_analogs/location_level/rcp85_qsubs/$runname
cat /home/hnoorazar/analog_codes/parameters/q_rcp85_no_precip_w_gen3 | while read LINE ; do
qsub $LINE
done
done

