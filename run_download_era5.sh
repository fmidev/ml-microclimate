#!/bin/sh
#

# nohup sh run_download_era5.sh > log_era5.txt &



code_dir=/users/kamarain/download_ERA5



source /lustre/tmp/kamarain/miniconda/etc/profile.d/conda.sh
conda activate full_env



# Static fields
mkdir -p /fmi/scratch/project_2002138/ERA5_orig_NFin/
cd /fmi/scratch/project_2002138/ERA5_orig_NFin/

python download_era5_staticfields_from_cds.py


# Dynamic fields
declare -a vars=('pmsl' 'te2m' 'dw2m' 'mtpr' 'msdlwrf' 'msdswrf' 'tclc' 'v10m' 'u10m' 'meva' 'slhf' 'sshf' 'mx2t' 'mn2t' 'sprs' 'msnlwrf' 'tsksr' 'ssrd' 'snw' 'sknt') 
for var in "${vars[@]}"
do
   echo $var
   python download_era5_sfc_from_ecmwf.py $var &
done



