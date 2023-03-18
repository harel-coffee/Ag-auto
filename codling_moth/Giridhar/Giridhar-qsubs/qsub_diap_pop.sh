#!/bin/bash
#v#!/usr/bin/env Rscript
# job name
PBS -N RCodMoth-diap_pop
# set the requested resources
#PBS -l mem=20gb,nodes=1,walltime=6:00:00
PBS -l nodes=1:dev:ppn=1
PBS -l mem=2gb
PBS -l walltime=12:00:00
#PBS -q hydro
# Request email on (a)bort, (b)eginning, and (e)nd.
#PBS -m abe

# email when finished
#PBS -M giridhar.manoharan@wsu.edu

cd $PBS_O_WORKDIR

# First we ensure a clean running environment:
module purge

# Load modules (if needed)
module load R/R-3.2.2_gcc

#./CodlingMoth.R  /home/kiran/histmetdata/vic_inputdata0625_pnw_combined_05142008/ /data/kiran/CodlingMoth/hist/
#./testR.R
/data/hydro/users/giridhar/giridhar/codmoth_pop/rel_abs_diap_pop.R

exit 0
