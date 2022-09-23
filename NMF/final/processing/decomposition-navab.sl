#!/bin/bash

#SBATCH -G 0
#SBATCH --mem 16G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --job-name decompose
#SBATCH --output slurm-logs-decomp/slurm-%j.out

MLM_LICENSE_FILE=27000@license-0.engr.unr.edu

root_dir=/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide

f=$1
echo $f

if [[ -d $root_dir/$f/2021/decompose-navab ]]; then
  rm $root_dir/$f/2021/decompose-navab/*.mat
else
  echo "Creating folder $root_dir/$f/2021/decompose-navab"
  mkdir -p $root_dir/$f/2021/decompose-navab
fi

cd ..
time matlab -nodisplay -nodesktop -nosplash -r "decompose_patches_navab('$root_dir/$f/2021'); exit();"
