# Particle-detection-in-3D-cellular-cryo-electron-tomograms

## What is this?
This reporsitory provides a framework to localize and detect particles in 3D cellular cryo-electron tomography. The main contribution in this repository is to solve the challenges of high density of the particles as well as the less-accurate groundtruth. The framework detects the localization of particles and gives instance label to each particle.

The framework first uses two trained networks to predict the particle masks and the particle centers, respectively. Then several post processing methods are applied to generate the instance labelled results and filter out outliers. Finally, instance labelled results in mrc file and the center coordinates of all detected particles in txt file are generated.


## Installation:

## Data:



## Model:

![image](https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/framework.png)

## Usage:

### Train:
![image](https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/model_structure.png | width=20)
### Inference:
### Postprocessing:
### Evaluation:e

## Results: