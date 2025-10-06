#!/bin/sh -l
#SBATCH --job-name=gen_resic
#SBATCH --time=08:00:00 
#SBATCH --partition=compute
#SBATCH -N 12
#SBATCH -n 12
#SBATCH -c 128  
#SBATCH --mem=240G
#SBATCH -o gen_resic.out

# Prepare the Tykky Miniconda environment
source /lustre/tmp/kamarain/miniconda/etc/profile.d/conda.sh
conda activate full_env

declare -a regions=("RAS" "TII" "KIL" "PAL" "ULV" "VAR")
declare -a years=("2019" "2020" "2021" "2022" "2023" "2024")
declare -a months=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")

for region in "${regions[@]}"; do
  for year in "${years[@]}"; do
    echo "Processing region: $region, year: $year"

    jobcount=0
    for month in "${months[@]}"; do
      echo "  Launching month: $month"
      srun -N1 -n1 -c128 --exclusive python generate_new_spatiotemporal_data.py "$region" "$year" "$month" &

      jobcount=$((jobcount+1))
      if [ $jobcount -eq 12 ]; then
        wait  # Wait for the current batch of 12 parallel jobs to finish
        jobcount=0
      fi
    done

    wait  # Final wait in case total months are not divisible by 12
  done
done

