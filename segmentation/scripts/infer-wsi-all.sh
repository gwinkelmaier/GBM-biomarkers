#!/bin/bash

# Submit WSI for job
while read -r wsi; do 
  sbatch infer-wsi.sl 20220110-162436 $wsi
done < file-list-LGG.txt
