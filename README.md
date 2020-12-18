# Particle-detection-in-3D-cellular-cryo-electron-tomograms

## What is this?
This reporsitory provides an end to end framework to localize and detect particles in 3D cellular cryo-electron tomography. The main contribution in this repository is to solve the challenges of high density of the particles as well as the less-accurate groundtruth. The framework detects the localization of particles and gives instance label to each particle. 
 
The framework first uses two trained networks to predict the particle masks (SegNet) and the particle centers (CenterNet), respectively. Then several post processing methods are applied to generate the instance labelled results and filter out outliers. Finally, instance labelled results in mrc file and the center coordinates of all detected particles in txt file are generated. They will be saved called pred_tomo.mrc and pred_center_coordinate.txt in the preset path.

## Installation:

## Data decription and preprocessing:

The original data from biologists includes a mrc file describing the raw cryo-electron tomography data and several xml files used to generate groundtruth, which describe each particle's position, orientation, class and the index of the tomogram that contains it. Using this information, 3D label maps could be generated with particle masks with different shapes and orientations around each manually labeled position (this algorithm is not included in this repository). 

For SegNet, the raw data and the generated groundtruth are used to train. The groundtruth is binary, in which 0 denotes background and 1 denotes foreground.

For CenterNet, a new label is created to denote the particle centers. In the new label, the center of each particle has value 1 and the boundary pixels have value 0, the pixels in between have values between 0 and 1 and are calculated according to the distance to the nearest boundary pixels. In other words, each pixel of a particle represents the normalized distance to the nearest background pixel. The groundtruth here is between 0 and 1 and this CenterNet could be considered as a regression neural network.


## Model:

![image](https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/framework.png)

The overall end-to-end framework structure is shown in the figure above. In testing, the raw data is input to the two trained network (SegNet and CenterNet) and predict the particle masks as well as particle centers, respectively. After that, several post processing methods are chosen and applied to generate the final instance labelled particles.

There are three optional post processing strategies:   
    
- Post processing 1: Filter out small objects in the predicted mask. Since there will be some single or only several neighbor pixels predicted as foreground by the network by mistake, the parameter *remove_min_size* could be set as the minimum particle size in the groundtruth, such that all connected component in the prediction smaller than this value will be filtered out.   

- Post processing 2: When we want to fuse the predicted masks and predicted centers, we will first find the local maxima in each particle centers, and then consider this pixel as a 'center' of one particle, every 'center' will have a different value. Then there are two methods to obtain the instance labelled mask. The first one (*check_center* =False) is, for every foreground pixel in the predicted mask, find the nearst predicted center and have the same value as the center. The second one (*check_center* =True) is, for each particles, check how many centers it has first. we will only split particles in mask if it has more than 2 centers. The second method could avoid the case when some parts of one particle belongs to a center outside.

- Post processing 3: Since we want to find the particles on the membranes, they are always dense in some areas and will not be alone outside the membrane areas. Therefore we could filter out some alone particles to decrease the false positive. When filter_outlier is True, we will filter particles, which have fewer than *neighbor_number* nearest neighbor particles within *distance_threshold* pixels. For example, with the default value, we consider two particles within 30 pixels are neighbors. If one particle does not have more than 3 neighbors, it will be counted as outlier and will be removed.

User could choose different post processing strategies and/or different parameters to get the final results. Using default values in post processing could possibly filter out many false positives while only lossing few true postives, which comes at the cost of higher computational time.

## Usage:

### Training:

For training the network ```train.py``` could be used. You should modify several parameters in the *main* function, such as *path_data* .

After training, trained model weights from SegNet will be saved called 'mask_model.h5'  and trained model weights from CenterNet will be saved called 'center_model.h5'  in the *model_path*.

**Example run**:

```
python train.py
```

### Testing:

After training, an end-to-end framework could be used easily to detect the particles on an unseen data. The following arguments should/can be given:

**Require arguments**:

* -tomo: The path of the testing tomogram.

**Optional arguments**:

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

**Example run**:

```
python end2end_framework.py -tomo './tomo.mrc'
```

### Inference:

User could also run inference (```test/inference.py```) individually and save the original output from the trained network (mask prediction or center prediction). You should modify several parameters in the *main* function, such as *tomo_path* .

**Example run**:

```
python test/inference.py
```

Here you could find an example output (2D slice) of trained SegNet and CenterNet.
<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/predict_result.png" width="480">

### Postprocessing:

User could also run post processing (```test/postprocessing.py```) individually given the output of inference function. You should also modify several parameters in the *main* function, such as *pred_mask_path* .

**Example run**:

```
python test/postprocessing.py
```

### Evaluation:

To evaluate the performance of the detected particles, please use the ```evaluation.py``` script. This can be used to compare the areas of the detected particles and the particle locations in the groundtruth.

In more detail, in this function, particles in prediction and groundtruth are divided into 4 categories: 
- 1. True Positive: TP, correctly segmented particles.
- 2. Merged particles: wrongly segmented particles - several particles in groundtruth are considered as 1 object in prediction. 
- 3. False Positive: FP, no corresponding particles in the groundtruth.
- 4. False Negative:FN, missed particles in the groundtruth.
    
And then several quantitative evaluation results could be given:
- 1. Precision (TP / particles in prediction)
- 2. Recall (TP / particles in groundtruth)
- 3. Merged rate (merged particle number / particles in groundtruth): since the center coordinate of each individual particle is also important in this project, this metrics is also included to check whether adjacent particles could be separated appropriately.
    
In addtion, to evaluate the results visually, we also give options to plot one 2d slice of the visually evaluation results or to save the whole 3d visually evaluation volume into mrc file. In these plots, TP particles have value=1 and color green, merged particles have value=2 and color yellow while FN/FP have value=3 and color red. Comparing the number of particles in different colors, user could have a feeling of how good the results are.

To run the evaluation, you should also modify several parameters in the *main* function in ```evaluation.py```, such as *path_pred* .

**Example run**:

```
python evaluation.py
```

Here you could find an example 2D slice of the visually evaluation results.
(TP: Green, merged particles: yellow, FP in the prediction: Red and FN in the groundtruth: Red)

<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/Evaluation.png" width="480">

## Results:

Below shows one example of the final files you could get. After running ```end2end_framework.py```, you could obtain a instance labelled particle detection in mrc file (as Figure b) shows) and the center coordinates of all detected particles in txt file (as Figure d) shows). And after running ```evaluation.py```, you could also obtain an evaluated results (as Figure c) shows) using different values to denote TP, FP, FN and merged particles.

<img src="https://github.com/HelmholtzAI-Consultants-Munich/Particle-detection-in-3D-cellular-cryo-electron-tomograms/blob/dev/README_files/Final_result.png" width="1000">

The performance of the different post processing strategies on the test set is given here:

|| Precision  | Recall     | Runtime in GPU |
|---------------| ---------- | ---------- | -------------- |
|Baseline       |            |            | ~ 1h           |
|check_center   |            |            |+ ~ 20min       |
|filter_outlier |            |            |+ ~ 10min       |
|Combined       |            |            |+ ~ 30min       |

