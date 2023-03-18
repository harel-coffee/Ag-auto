
## Define a job name
#PBS -N python_analogs_85
#PBS -q hydro
#PBS -l nodes=1:ppn=4,walltime=99:00:00
#PBS -l mem=4gb
#PBS -e /home/hnoorazar/analog_codes/03_find_analogs/error/e_85_analog
#PBS -o /home/hnoorazar/analog_codes/03_find_analogs/error/o_85_analog
#PBS -m abe

module load python/2.7.8
module load scipy/0.17.1-python2.7.8
cd /home/hnoorazar/analog_codes/03_find_analogs/

python d_analogs.py
