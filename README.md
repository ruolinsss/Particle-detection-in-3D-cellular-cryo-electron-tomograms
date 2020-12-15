# Particle-detection-in-3D-cellular-cryo-electron-tomograms

## What is this?
This reporsitory provides a framework to localize and detect particles in 3D cellular cryo-electron tomography. The main contribution in this repository is to solve the challenges of high density of the particles as well as the less-accurate groundtruth. The framework detects the localization of particles and gives instance label to each particle. 
 
The framework first uses two trained networks to predict the particle masks (MaskNet) and the particle centers (CenterNet), respectively. Then several post processing methods are applied to generate the instance labelled results and filter out outliers. Finally, instance labelled results in mrc file and the center coordinates of all detected particles in txt file are generated.

## Installation:

## Data decription and preprocessing:

The original data from biologists includes a mrc file describing the raw cryo-electron tomography data and several xml files used to generate groundtruth, which describe each particle's position, orientation, class and the index of the tomogram that contains it. Using this information, 3D label maps could be generated with particle masks with different shapes and orientations around each manually labeled position (this algorithm is not included in this repository). 

For MaskNet, the raw data and the generated groundtruth are used to train. The groundtruth is binary, in which 0 denotes background and 1 denotes foreground.

For CenterNet, a new label is created to denote the particle centers. In the new label, the center of each particle has value 1 and the boundary pixels have value 0, the pixels in between have values between 0 and 1 and are calculated according to the distance to the nearst boundary pixels. In other words, each pixel of a particle represents the normalized distance to the nearest background pixel. The groundtruth here is between 0 and 1 and this CenterNet could be considered as a regression neural network.


## Model:


![image](https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/framework.png)

## Usage:

### Train:
This model is a simple Unet with 2 transition down and 2 transition up block.
<img align="right"  src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/model_structure.png" width="360">



### Inference:
<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/predict_result.png" width="480">

### Postprocessing:
post processing choice:
1. remove_min_size: filter out small objects
2. check_center: for each particles, check how many centers it has. we will only split particles in mask if it has more than 2 centers.
3. filter_outlier: filter out the outliers (there are fewer than neighbor_number neighbor particles within distance_threshold pixels)

### Evaluation:
<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/Evaluation.png" width="480">

## Results:
<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/Final_result.png" width="1000">

|| Precision  | Recall     | Runtime in GPU |
|---------------| ---------- | ---------- | -------------- |
|Baseline       |            |            | ~ 1h           |
|check_center   |            |            |+ ~ 20min       |
|filter_outlier |            |            |+ ~ 10min       |
|Combined       |            |            |+ ~ 30min       |

