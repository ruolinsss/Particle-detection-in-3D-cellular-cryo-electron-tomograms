from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def evaluation(seg,gt):
    """
    This function receives predicted particle file and instance labelled groundtruth to do the evaluation. This function takes ~45 min for mask with size of (464, 928, 928) in GPU.
    
    According to the statistical measurements, this function will record the instance number in prediction and groundtruth in the following two categories: 
    1. Ture Positive (TP, correctly segmented particles),
    2. merged particles (wrongly segmented particles, could also be considered FP/FN - several particles in groundtruth are considered as 1 object in prediction). 
    The rest instance numbers in prediction could be considered as False Positive (FP, no corresponding particles in the groundtruth) and in groundtruth could be considered as False Negative (FN, missed particles in the groundtruth)
    
    This function also prints following quantitative evaluation results:
    1. Precision (TP / particles in prediction)
    2. Recall (TP / particles in groundtruth)
    3. Merged rate (merged particle number / particles in groundtruth): since the center coordinate of each individual particle is also important in this project, this metrics is also included to check whether adjacent particles could be separated appropriately.
    
    Input
    ----------
        seg: 3d ndarray
            Instance labelled predicted particles, which could be the output of postprocessing.py
        gt: 3d ndarray
            Instance labelled groundtruth, each particle should has a different value
        
    Returns
    -------
        updated_TP_particle: list
            A list includes the instance values of TP particles in the prediction
        merged_particle: list
            A list includes the instance values of merged particles in the prediction
        updated_TP_particle_gt: list
            A list includes the instance values of TP particles in the groundtruth
        merged_particle_gt: list
            A list includes the instance values of merged particles in the groundtruth
    """
    total_gt = np.unique(gt)
    total_seg = np.unique(seg)
    
    TP_particle = []
    TP_particle_gt = []
    merged_particle = []
    merged_particle_gt = []
    label_list = []
    
    for value in total_gt[1:]:
        
        tmpseg = np.copy(seg)
        seg_label = Counter(tmpseg[gt == value]).most_common(1)[0][0] # find the most common class in the crossponding areas
        if seg_label != 0 and seg_label not in label_list:   # if several objects are connected in the segmentation images, they are considered as wrongly segmented objects
            TP_particle.append(seg_label)
            TP_particle_gt.append(value)
        elif seg_label != 0 and seg_label in label_list:
            merged_particle.append(seg_label)
            merged_particle_gt.append(value)
            
        label_list.append(seg_label)
        
    # remove merged particle in TP
    updated_TP_particle = []
    for i in range(len(TP_particle)):
        if TP_particle[i] not in merged_particle:
            updated_TP_particle.append(TP_particle[i])
        else:
            merged_particle_gt.append(TP_particle_gt[i])            
    updated_TP_particle_gt = [e for e in TP_particle_gt if e not in merged_particle_gt]
    
    print('Recall',len(updated_TP_particle_gt)/len(total_gt))
    print('Precision',len(updated_TP_particle)/len(total_seg))
    print('Merged Rate',len(merged_particle_gt)/len(total_gt))
    
    return updated_TP_particle,merged_particle,updated_TP_particle_gt,merged_particle_gt



def get_visually_evaluation(seg,gt,TP_particle,merged_particle,TP_particle_gt,merged_particle_gt, 
                            plot=False,save_path=None,axis=None):
    '''
    This function shows evaluation result visually. All particles will be divided into 3 classes: TP, merged particles and FP(for prediction)/ FN(for groundtruth). Particles segmented correctly (TP) have value = 1, merged particles have value = 2 and FN in groundtruth or FP in prediction have value = 3. It takes around 26 min to get the results with the size of (464, 928, 928) in GPU.
    If plot is True, this function will generate and save a 2d slice plot to evaluate the result visually. In this plot, TP particles have color green, merged particles have color yellow while FN/FP have color red. Comparing the number of particles in different colors, we could have a feeling of how good this algorithm is.
    
    Input
    ----------
        seg: 3d ndarray
            Instance labelled predicted particles, which could be the output of postprocessing.py
        gt: 3d ndarray
            Instance labelled groundtruth, each particle should has a different value            
        TP_particle: list
            A list includes the instance values of TP particles in the prediction
        merged_particle: list
            A list includes the instance values of merged particles in the prediction
        TP_particle_gt: list
            A list includes the instance values of TP particles in the groundtruth
        merged_particle_gt: list
            A list includes the instance values of merged particles in the groundtruth
        plot: bool - default False
            If True, two plots (evaluated prediction and evaluated groundtruth) will be saved in the save_path.
        save_path: string - default None
            The path two plots will be saved in. Need if plot is True.
        axis: int - default None
            The x axis slice we want to see in the evaluated result, e.g. 230 means it plots and saves evaluated_seg[230,:,:] 
            and evaluated_gt[230,:,:]. Need if plot is True.
            
    Returns
    -------
        evaluated_seg: 3d ndarray 
            Evaluated prediction, only have 4 values: 0 denotes background pixels, 1 denotes TP partcles, 
            2 denotes merged particles while 3 denotes FP
        evaluated_gt: 3d ndarray
            Evaluated groundtruth, only have 4 values: 0 denotes background pixels, 1 denotes TP partcles, 
            2 denotes merged particles while 3 denotes FN
            
    '''
    evaluated_seg = np.copy(seg)
    evaluated_gt = np.copy(gt)

    for i in merged_particle:
        evaluated_seg[evaluated_seg==i] = 2
    for i in TP_particle:
        evaluated_seg[evaluated_seg==i] = 1
    evaluated_seg = np.where((evaluated_seg!=0) & (evaluated_seg!=1)&(evaluated_seg!=2),3,evaluated_seg)

    for i in merged_particle_gt:
        evaluated_gt[evaluated_gt == i] = 2
    for i in TP_particle_gt:
        evaluated_gt[evaluated_gt == i] = 1
    evaluated_gt = np.where((evaluated_gt!=0) & (evaluated_gt!=1)&(evaluated_gt!=2),3,evaluated_gt)
    
    if plot == True:
        assert save_path != None and axis != None, 'save_path and axis should be given to plot visually evaluation result!'
        plot_visually_evaluation_example(evaluated_seg,save_path+'visually_evaluation_seg.png',axis=axis)
        plot_visually_evaluation_example(evaluated_gt,save_path+'visually_evaluation_gt.png',axis=axis)
    
    return evaluated_seg,evaluated_gt


def plot_visually_evaluation_example(evaluated_file,save_path,axis):
    '''
    This function generates and saves a 2d slice plot to evaluate the result visually. In this plot, TP particles have color green, merged particles have color yellow while FN/FP have color red. Comparing the number of particles in different colors, we could have a feeling of how good this algorithm is.
    
    Input
    ----------
        evaluated_file: 3d ndarray 
            Evaluated file, only have 4 values: 0 denotes background pixels, 1 denotes TP partcles, 
            2 denotes merged particles while 3 denotes FP
        save_path: string - default None
            The path two plots will be saved in. Need if plot is True.
        axis: int - default None
            The x axis slice we want to see in the evaluated result, e.g. 230 means it plots and saves evaluated_seg[230,:,:] 
            and evaluated_gt[230,:,:]. Need if plot is True.
            
    Returns
    -------
            
    '''
    # black: background(0)  green: TP(1)   yellow: merged(2)    red: fn/fp(3)
    colors = ['black', 'green', 'yellow', 'red', ]
    cmap = mpl.colors.ListedColormap(colors)
    
    plt.figure(figsize=(10,10))
    plt.imshow(evaluated_file[axis],cmap=cmap)
    plt.savefig(save_path)
    