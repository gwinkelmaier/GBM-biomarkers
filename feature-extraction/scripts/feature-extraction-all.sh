#!/bin/bash

script_dir=$(pwd)
while read -r wsi; do 
  sbatch feature-extraction.sl $wsi
done < file-list-LGG.txt
