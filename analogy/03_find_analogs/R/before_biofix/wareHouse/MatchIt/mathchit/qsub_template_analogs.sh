#PBS -N thisjobname_set1
#PBS -l mem=2gb,nodes=1:scratch,walltime=20:00:00
#PBS -m abe

SERVER=$PBS_O_HOST
WORKDIR=/home/hnoorazar/analog_codes/03_find_analogs/R_codes/matchit/
SCP=/usr/bin/scp
SSH=/usr/bin/ssh

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo ' '
echo ' '

############################################################
#    Execute the run.  Do not run in the background.       #
############################################################

runprogram()
{
  echo In runprogram:
  echo present working directory:
  pwd

  Rscript --vanilla /home/hnoorazar/analog_codes/03_find_analogs/R_codes/matchit/d_driver.R file_name
}

early()
{
 echo ' '
 echo ' ############ WARNING:  EARLY TERMINATION #############'
 echo ' '
 }

# trap 'early; stageout' 2 9 15

##################################################
#   Staging in, running the job, and staging out #
#   were specified above as functions.  Now      #
#   call these functions to perform the actual   #
#   file transfers and program execution.        #
##################################################

runprogram
exit 0

