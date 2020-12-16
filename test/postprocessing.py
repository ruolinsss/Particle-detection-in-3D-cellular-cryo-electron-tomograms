import numpy as np
import time

from utils.utils import read_mrc,write_mrc,write_txt
from utils.postprocessing_helper import find_center,get_center_coords,remove_outlier
from skimage.morphology import remove_small_objects  

def postprocessing(mask,
                   center,
                   mask_threshold = 0.1,
                   center_threshold = 0.1,
                   remove_min_size = 64,
                   check_center = True,
                   filter_outlier = True,
                   neighbor_number = 3,
                   distance_threshold = 30):
    """
    This function does the post processing for the inferenced results.   
    There are three optional post processing strategies:   
    
        Post processing 1: remove_min_size: Filter out small objects in the predicted mask. Since there will be some single or 
                           only several pixels predicted as foreground by the network by mistake, we could set the parameter 
                           remove_min_size as the minimum particle size in the groundtruth, such that all connected component in 
                           the prediction smaller than this value will be filtered out.   
                           
        Post processing 2: check_center: When we want to fuse the predicted masks and predicted centers, we will first find the 
                           local maxima in each particle centers, and then consider this pixel as a 'center' of one particle, 
                           every 'center' will have a different value. 
                           Then there are two methods to obtain the instance labelled mask. The first one (check_center=False) 
                           is, for every foreground pixel in the predicted mask, find the nearst predicted center and have the 
                           same value as the center. The second one (check_center=True) is, for each particles, check how many 
                           centers it has first. we will only split particles in mask if it has more than 2 centers. The second 
                           method could avoid the case when some parts of one particle belongs to a center outside.
        
        Post processing 3: filter_outlier: Since we want to find the particles on the membranes, they are always dense in some 
                           areas and will not be alone outside the membrane areas. Therefore we could filter out some alone 
                           particles to decrease the false positve. When filter_outlier is True, we will filter particles, which 
                           have fewer than neighbor_number nearst neighbor particles within distance_threshold pixels. For 
                           example, with the default value, we consider two particles within 30 pixels are neighbors. If one 
                           particle does not have more than 3 neighbors, it will be counted as outlier and will be removed.
    
    Input
    ----------
        mask: 3d ndarray
            predicted mask, output from inference function with mode mask          
        center: 3d ndarray
            predicted center, output from inference function with mode center  
        mask_threshold: int - default 0.1
            Threshold to filter out the small noise in predicted mask.
        center_threshold: int - default 0.1
            Threshold to filter out the small noise in predicted center.
        remove_min_size: int - default 64
            Particles smaller than this size will be removed.
        check_center: bool - default True
            Whether to use check center strategy in the post processing.
        filter_outlier: bool - default True
            Whether to use filter outlier strategy in the post processing.
        neighbor_number: int - default 3
            Define how many neighbor particles are considered in filter outlier strategy. 
            Required when filter_outlier is True.
        distance_threshold: int - default 30
            Define the distance to be considered in filter outlier strategy. 
            Required when filter_outlier is True.
    
    Returns
    -------
        processed_mask: ndarray
            Mask after filtering outliers, every particle has a different value
        coords: 2d list
            A list contains the XYZ coordinates of all particles in the filted_mask
    """
    
    # post processing 1
    mask = np.where(mask > mask_threshold, True, False)
    mask = remove_small_objects(mask,remove_min_size) 
    mask = np.where(mask==True, 1, 0)
    center = np.where(center >= center_threshold, center, 0)

    # post processing 2
    processed_mask = find_center(mask,center,check_center=check_center)
    coords = get_center_coords(processed_mask)

    # post processing 3
    if filter_outlier == True:
        processed_mask,coords = remove_outlier(processed_mask,coords,
                                               neighbor_number=neighbor_number,distance_threshold=distance_threshold)
    processed_mask = processed_mask.astype(np.float32)
    
    return processed_mask,coords
 
    
    
if __name__=='__main__':
    
    pred_mask_path = './output/mask_pred.mrc' 
    pred_center_path = './output/center_pred.mrc' 
    pred_path = 'output/'    
    voxel_size = 14.08 # voxel size for different dataset, 14.08 for spinach data
    
    mask,header = read_mrc(pred_mask_path)
    center,_ = read_mrc(pred_center_path)
    
    processed_mask,coords = postprocessing(mask,center)
    
    write_mrc(processed_mask,pred_path+'pred_tomo.mrc',header_dict=header)
    write_txt(coords,pred_path+'pred_center_coordinate.txt',voxel_size=voxel_size)
    