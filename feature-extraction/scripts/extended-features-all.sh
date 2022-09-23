#!/bin/bash

rm extended-feature-logs/*.out

script_dir=$(pwd)
while read -r wsi; do 
  sbatch extended-features.sl $wsi
done < file-list-LGG.txt
