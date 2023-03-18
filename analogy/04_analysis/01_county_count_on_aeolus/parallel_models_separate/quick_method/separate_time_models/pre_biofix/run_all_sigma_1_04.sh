#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=00:05:00
#PBS -l mem=1gb
#PBS -q fast

cd /home/hnoorazar/analog_codes/04_analysis/parallel/quick/rcp45/
cat /home/hnoorazar/analog_codes/parameters/sigma_1_job_names | while read LINE ; do
qsub $LINE
done

cd /home/hnoorazar/analog_codes/04_analysis/parallel/quick/rcp85/
cat /home/hnoorazar/analog_codes/parameters/sigma_1_job_names | while read LINE ; do
qsub $LINE
done