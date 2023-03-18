#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N summarize_prob_data
## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=8gb
#PBS -l walltime=04:00:00
#PBS -q batch

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /fastscratch/mpruett/summarize_data.e
#PBS -o /fastscratch/mpruett/summarize_data.o

## Define path for reporting
#PBS -M matthew.pruett@wsu.edu
#PBS -m abe

# -------- -------- -------- --------
# Actually do something
# -------- -------- -------- --------

cd $PBS_O_WORKDIR

module purge

module load R

Rscript --vanilla scripts/summarize_prob_data.R