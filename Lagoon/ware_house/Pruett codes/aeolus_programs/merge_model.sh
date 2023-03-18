#!/bin/bash

## Export all environment variables in the qsub command's environment to the
## batch job.
#PBS -V

## Define a job name
#PBS -N merge_model

## Define compute options
#PBS -l nodes=1:dev:ppn=1
#PBS -l mem=4gb
#PBS -l walltime=01:00:00
#PBS -q batch

## Define path for output & error logs
#PBS -k o
  ##PBS -j oe
#PBS -e /fastscratch/mpruett/merge_model.e
#PBS -o /fastscratch/mpruett/merge_model.o

## Define path for reporting
#PBS -M matthew.pruett@wsu.edu
#PBS -m abe

# -------- -------- -------- --------
# Actually do something
# -------- -------- -------- --------

cd $PBS_O_WORKDIR

module purge

module load R

Rscript --vanilla merge_models_aeolus.R