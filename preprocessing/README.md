# GBM Preprocessing 

## Description

### SVS Magnification
Determine what magnification the SVS file was imaged at.
This is done with a python script to read the 'aperio.AppMag' property and create a list of 
20x and 40x images for down-the-line processing.

### Tiling
WSI files are tiled into patches of 224x224.  This size is determined by the input size of the
Machine Learning (segmentation) models used in this project.  PyHIST is used to determine valid 
tissue section and to create the actual tiles.

###### PyHIST
[PyHIST](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008349) 
uses a Canny Edge Detection algorithm with graph connection to determine where the tissue 
is within an svs file.  The parameters used with the PyHIST 
[codebase](https://github.com/manuel-munoz-aguirre/PyHIST)
is found below.  

- output-downsample: 1
	- Tile Magnification: 20x
- patch-size: 224
- mask-downsample: 2
	- Mask Magnification: 10x
- content-threshold: 0.05
	- Keeps tiles with at least 5% Tissue

### Pen Detection
Tiled tissue sections are passed through an SVM Classifier trained to detect pen-marks within
pin-hole views of H&E WSI.  The model expects image histograms for each channel for the RGB tile.
This step curates a list of tiles and their associated probability of containing pen-marks.
This result is used to remove tiles from the image bank prior to nuclear segmentation.

## Files

###### src
Houses python files

###### jobs
Houses Shell scripts, slurm files, and logs.

## Execution Steps
- conda env: PyHIST
  - run "python src/find-mag.py
  - move text files into jobs/lists
  - run "jobs/tile.sh"

- conda env: tf_gpu
  - run "jobs/pen-filter.sh"

## Output locations
Final outputs, e.g. tiles & pen-probabilites, are save relative to the svs file in a subfolder titled 2021

###### Hidden Folders
Some hidden folders are generated for the purpose of visualizing pipeline steps.  For example, the folder 
'2021/.penmarks' contains an image bank of blank images that represent patches flagged by the pen-detection 
model.  This folder is used in combination with the QuPATH automation script labeled 'import_pen_detection.groovy'.
Running the script on a WSI folder that contains '2021/.penmarks' will draw red squares around patches that
are removed from the tiles image bank due to the presence of pen-marks.
