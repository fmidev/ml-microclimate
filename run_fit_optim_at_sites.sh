#!/bin/sh -l
#SBATCH --job-name=resiclim
#SBATCH --time=72:00:00 
#SBATCH --partition=compute
#SBATCH -N 21
#SBATCH -n 21
#SBATCH -c 128  
#SBATCH --mem=240G
#SBATCH -o fit_resic.out





# Prepare the Miniconda environment
#export PATH="/fmi/projappl/project_2002138/mconda_newest/bin:$PATH"
#export PATH="/lustre/tmp/kamarain/miniconda/bin:$PATH"
source /lustre/tmp/kamarain/miniconda/etc/profile.d/conda.sh
conda activate full_env


declare -a years=("2019" "2020" "2021" "2022" "2023" "2024" "2025")
declare -a regions=("RAS" "TII" "KIL" "PAL" "ULV" "VAR")



# Optimize, but do not fit, do not predit
optimize="True"
fit="False"

count=0
for year in "${years[@]}"; do
    for region in "${regions[@]}"; do
        echo "Processing region: $region, year: $year"
        srun -n1 -N1 -c128 --exclusive python fit_optim_at_sites.py $region $year $optimize $fit &
        
        count=$((count + 1))
        if (( count % 21 == 0 )); then
            wait
            echo "Batch of 21 jobs completed."
        fi
    done
done


# Do not optimize, but fit, do not predit
optimize="False"
fit="True"

count=0
for year in "${years[@]}"; do
    for region in "${regions[@]}"; do
        echo "Processing region: $region, year: $year"
        srun -n1 -N1 -c128 --exclusive python fit_optim_at_sites.py $region $year $optimize $fit &
        
        count=$((count + 1))
        if (( count % 21 == 0 )); then
            wait
            echo "Batch of 21 jobs completed."
        fi
    done
done



# Do not optimize, do not fit, but predit
optimize="False"
fit="False"

count=0
for year in "${years[@]}"; do
    for region in "${regions[@]}"; do
        echo "Processing region: $region, year: $year"
        srun -n1 -N1 -c128 --exclusive python fit_optim_at_sites.py $region $year $optimize $fit &
        
        count=$((count + 1))
        if (( count % 21 == 0 )); then
            wait
            echo "Batch of 21 jobs completed."
        fi
    done
done
