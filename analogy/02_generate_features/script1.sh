#PBS -N Q1DetAdjDriver1
#PBS -q wudb
#PBS -l nodes=1:ppn=4,walltime=99:00:00
#PBS -l mem=2gb
#PBS -e error1.txt
#PBS -o output1.txt
#PBS -M noorazar.h@gmail.com
#PBS -m abe

module load python/2.7.8
module load scipy/0.17.1-python2.7.8
cd /home/mhn.namak/hossein/Q1deterministicAdj
python driver1.py
