## Define a job name
#PBS -N py_alg_CNRM_CM5
#PBS -q hydro
#PBS -l nodes=1:ppn=4,walltime=99:00:00
#PBS -l mem=4gb
#PBS -e /home/hnoorazar/analog_codes/03_find_analogs/models_separ/error/e_CNRM_CM5_45_analog
#PBS -o /home/hnoorazar/analog_codes/03_find_analogs/models_separ/error/o_CNRM_CM5_45_analog
#PBS -m abe

module load python/2.7.8
module load scipy/0.17.1-python2.7.8
cd /home/hnoorazar/analog_codes/03_find_analogs/models_separ/

python d_analogs_separ.py CNRM_CM5 rcp45
