#PBS -N thisjobname_set1
#PBS -l mem=2gb,nodes=1:scratch,walltime=20:00:00
#PBS -m abe



SERVER=$PBS_O_HOST
WORKDIR=/scratch
SCP=/usr/bin/scp
SSH=/usr/bin/ssh

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo ' '
echo ' '

###############################################################
#    Transfer files from server to local disk.                #
###############################################################

stagein()
{
	 echo ' '
	 echo Transferring files from server to compute node
	 echo Writing files in node directory  ${WORKDIR}
	 echo present directory is 
	 pwd
	 cd ${WORKDIR}
	 rm -r thisjobname_set1
	 mkdir thisjobname_set1
	 cd thisjobname_set1
	 echo now present directory is
	 	pwd
	 echo list of files is 
	  	ls
	 rm -r cropcells_set1
	 rm -r metdata
	 mkdir cropcells_set1
	 mkdir metdata
	 # transfer soil file
	 cp -r  /data/kiran/Data/VIC/params/soil/soilparamsensitivities/thisjobname/newparams/soil_param.set1 ./
	 
	 # transfer crop parameter and lib files
	 cp -r  /data/kiran/Data/VIC/params/crop/cropparam_Sep27_2011_noprecip_WSDA_alloutside_pasture50p ./
	 cp -r  /data/kiran/Data/VIC/params/crop/croplib_allcrops ./

	 # transfer veg param and lib file
	 cp -r  /data/kiran/Data/VIC/params/veg/vegparam_allcrops ./
	 cp -r  /data/kiran/Data/VIC/params/veg/veglib ./

	 # transfer CropSyst folder
	 cp -r  /data/kiran/Data/VIC/params/CropSyst ./
	
	 # transfer global paramter file
         cp -r  /data/kiran/Data/VIC/params/global/kirtifirstpaper/hist/global_param_set1 ./
	 # transfer the executable file
	 cp -r  /data/kiran/VIC/VIC_Crop/VICCROP407 ./

	 # transfer the met data

	 cat /data/kiran/Data/VIC/params/metdatalistmultipleset/list_set1 | while read LINE ; do
	 echo "copying $LINE..."
	 #cp   /data-failing/part2/kirti/vic_inputdata0625_pnw_combined_05142008/$LINE ./metdata/
	cp   /data/kiran/vic_inputdata0625_pnw_combined_05142008/$LINE ./metdata/
	done
	cd metdata
	echo number of files in met data
	ls | wc -l
	cd ..
	echo now directory should to back
	pwd

}

############################################################
#    Execute the run.  Do not run in the background.       #
############################################################

runprogram()
{
	echo In runprogram:
	echo present working directory:
	pwd
	./VICCROP407 -g global_param_set1
}

###########################################################
#   Copy necessary files back to permanent directory.     #
###########################################################

stageout()
{
	 echo ' '
	 echo Transferring files from compute nodes to server
	 echo Number of files in results:
	 ls cropcells_set1 | wc -l
	 cp -r ./cropcells_set1/* /data/kiran/firstpaper/thisjobname/
	 echo after deleting number of files:
	 cd ..
	 rm -r thisjobname_set1
	 echo current directory is
	 pwd	
	 echo list after deteting hist
	 ls
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

stagein
runprogram
stageout 

exit


