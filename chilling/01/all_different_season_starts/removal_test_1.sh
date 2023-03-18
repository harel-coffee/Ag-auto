#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N removal_test_1

## Define compute options
#PBS -l nodes=1:ppn=1,walltime=00:00:10
#PBS -l mem=1gb
#PBS -q hydro
#PBS -t 1-72

## Define path for output & error logs
#PBS -k o
##PBS -j oe
#PBS -e /home/hnoorazar/chilling_codes/current_draft/01/other_three_seasons/error/removal_test_1.E
#PBS -o /home/hnoorazar/chilling_codes/current_draft/01/other_three_seasons/error/removal_test_1.O

## Define path for reporting
#PBS -m abe

dir_list=()
while IFS= read -d $'\0' -r file ; do
dir_list=("${dir_list[@]}" "$file")
done < <(find /data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/ -mindepth 2 -maxdepth 2 -type d ! -path '*bcc*' -print0)

echo dir_list



exit 0
