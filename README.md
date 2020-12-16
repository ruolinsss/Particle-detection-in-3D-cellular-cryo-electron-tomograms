# Particle-detection-in-3D-cellular-cryo-electron-tomograms

## What is this?
This reporsitory provides an end to end framework to localize and detect particles in 3D cellular cryo-electron tomography. The main contribution in this repository is to solve the challenges of high density of the particles as well as the less-accurate groundtruth. The framework detects the localization of particles and gives instance label to each particle. 
 
The framework first uses two trained networks to predict the particle masks (SegNet) and the particle centers (CenterNet), respectively. Then several post processing methods are applied to generate the instance labelled results and filter out outliers. Finally, instance labelled results in mrc file and the center coordinates of all detected particles in txt file are generated.

## Installation:

## Data decription and preprocessing:

The original data from biologists includes a mrc file describing the raw cryo-electron tomography data and several xml files used to generate groundtruth, which describe each particle's position, orientation, class and the index of the tomogram that contains it. Using this information, 3D label maps could be generated with particle masks with different shapes and orientations around each manually labeled position (this algorithm is not included in this repository). 

For SegNet, the raw data and the generated groundtruth are used to train. The groundtruth is binary, in which 0 denotes background and 1 denotes foreground.

For CenterNet, a new label is created to denote the particle centers. In the new label, the center of each particle has value 1 and the boundary pixels have value 0, the pixels in between have values between 0 and 1 and are calculated according to the distance to the nearest boundary pixels. In other words, each pixel of a particle represents the normalized distance to the nearest background pixel. The groundtruth here is between 0 and 1 and this CenterNet could be considered as a regression neural network.


## Model:

There are three optional post processing strategies:   
    
- Post processing 1: Filter out small objects in the predicted mask. Since there will be some single or only several neighbor pixels predicted as foreground by the network by mistake, we could set the parameter remove_min_size as the minimum particle size in the groundtruth, such that all connected component in the prediction smaller than this value will be filtered out.   

- Post processing 2: When we want to fuse the predicted masks and predicted centers, we will first find the local maxima in each particle centers, and then consider this pixel as a 'center' of one particle, every 'center' will have a different value. Then there are two methods to obtain the instance labelled mask. The first one (check_center=False) is, for every foreground pixel in the predicted mask, find the nearst predicted center and have the same value as the center. The second one (check_center=True) is, for each particles, check how many centers it has first. we will only split particles in mask if it has more than 2 centers. The second method could avoid the case when some parts of one particle belongs to a center outside.

- Post processing 3: Since we want to find the particles on the membranes, they are always dense in some areas and will not be alone outside the membrane areas. Therefore we could filter out some alone particles to decrease the false positive. When filter_outlier is True, we will filter particles, which have fewer than neighbor_number nearest neighbor particles within distance_threshold pixels. For example, with the default value, we consider two particles within 30 pixels are neighbors. If one 
particle does not have more than 3 neighbors, it will be counted as outlier and will be removed.



![image](https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/framework.png)

## Usage:

### Training:

### Testing:

** Require arguments **:

* -tomo: The path of the testing tomogram.

** Optional arguments **:

* -m: The path of the trained SegNet model weights.
* -c: The path of the trained CenterNet model weights.
* -o: The path of the saved mrc file and txt file.
* -ps: Patch size to be fed into the inference model.
* -vs: Voxel size for different dataset, e.g. 14.08 for spinach data.
* -mt: Threshold to filter the small noise in predicted mask.
* -ct: Threshold to filter the small noise in predicted center.
* -rs: Particles smaller than this size will be removed.
* -check_center: Whether to use check center strategy in the post processing.
* -filter_outlier: Whether to use filter outlier strategy in the post processing.
* -nn: Define how many neighbor particles are considered in filter outlier strategy. Required when filter_outlier is True.
* -dt: Define the distance to be considered in filter outlier strategy. Required when filter_outlier is True.

** Example run **:

```
Python end2end_framework.py -tomo './tomo.mrc'
```

#### Inference:
<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/predict_result.png" width="480">

#### Postprocessing:

#### Evaluation:
<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/Evaluation.png" width="480">

## Results:
<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/Final_result.png" width="1000">

|| Precision  | Recall     | Runtime in GPU |
|---------------| ---------- | ---------- | -------------- |
|Baseline       |            |            | ~ 1h           |
|check_center   |            |            |+ ~ 20min       |
|filter_outlier |            |            |+ ~ 10min       |
|Combined       |            |            |+ ~ 30min       |

