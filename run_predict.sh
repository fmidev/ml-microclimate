#!/bin/sh
#
#SBATCH --job-name=RESICL
#SBATCH --account=project_2002138 
#SBATCH --time=05:00:00 
#SBATCH --partition=fmi
#SBATCH --mem=185G
#SBATCH -N 5
#SBATCH -n 5
#SBATCH -c 40
#SBATCH -o RESICL.out


# sbatch run_predict.sh 2010 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2011 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2012 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2013 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2014 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2015 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2016 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2017 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2018 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2019 "['15cm_relhum','150cm_relhum']"
# sbatch run_predict.sh 2020 "['15cm_relhum','150cm_relhum']"

#sbatch run_predict.sh 2010 "['15cm_temp',]"
#sbatch run_predict.sh 2011 "['15cm_temp',]"
#sbatch run_predict.sh 2012 "['15cm_temp',]"
#sbatch run_predict.sh 2013 "['15cm_temp',]"
#sbatch run_predict.sh 2014 "['15cm_temp',]"
#sbatch run_predict.sh 2015 "['15cm_temp',]"
#sbatch run_predict.sh 2016 "['15cm_temp',]"
#sbatch run_predict.sh 2017 "['15cm_temp',]"
#sbatch run_predict.sh 2018 "['15cm_temp',]"
#sbatch run_predict.sh 2019 "['15cm_temp',]"
#sbatch run_predict.sh 2020 "['15cm_temp',]"



code_dir='/users/kamarain/resiclim-microclimate'
data_dir='/fmi/scratch/project_2002138/resiclim-microclimate'



# Prepare the miniconda xesmf_env environment 
export PATH="/fmi/projappl/project_2002138/miniconda/bin:$PATH"
export LD_LIBRARY_PATH="/fmi/projappl/project_2002138/miniconda/lib"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

eval "$(conda shell.bash hook)"
conda activate venv



cd $code_dir/




year=$1
trgt=$2
for month in $(seq 5 1 9)
do
    echo $year $month
    srun -n1 -N1 -c40 --exclusive python $code_dir/predict.py $year $month $trgt &
done
wait

