import numpy as np
from skimage.measure import label
from scipy import ndimage as ndi
import scipy
from skimage.morphology import remove_small_objects  
from scipy.spatial.distance import pdist,squareform

def local_maxima_3D(data, order=1):
    """
    Detects local maxima in a 3D array

    Input
    ----------
        data: 3d ndarray
        order: int
            How many points on each side to use for the comparison
        
    Returns
    -------
        mask_local_maxima: 3d ndarray
            A 3D array with the same shape as the input data, where the local maxima has value 1 and other pixels = 0. 
        coords: ndarray
            Coordinates of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint, mode='mirror')
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    return mask_local_maxima,coords

def do_kdtree(maxima,pixels):
    """
    kd-tree for quick nearest-neighbor lookup. For input pixels, finds the corresponding nearst maxima.

    Input
    ----------
        maxima: ndarray
            Coordinates of the local maxima here
        pixels: ndarray
            Coordinates of the pixels here
        
    Returns
    -------
        dist: ndarray of floats
            The distances to the nearest neighbors. 
        indexes: ndarray
            The locations of the neighbors in maxima.
    """
    mytree = scipy.spatial.cKDTree(maxima)
    dist, indexes = mytree.query(pixels)
    
# if the distance between two center is too small (<3), merge them    
#     nearst_pair = mytree.query_pairs(2.5,output_type = "ndarray")  
#     for pairs in nearst_pair:
#         combined_x_y_arrays[pairs[0]] = np.mean([combined_x_y_arrays[pairs[0]],combined_x_y_arrays[pairs[1]]],axis=0).astype(np.int16)
#     new_combined_x_y_arrays = np.delete(combined_x_y_arrays,[i for i in np.unique(nearst_pair[:,1]) if i not in nearst_pair[:,0]],axis=0)

    return dist,indexes

def find_center(mask,center,check_center=False):
    """
    Using predicted center to allocate pixels in mask. After this function, every particle will have a different value to solve the separation problem.

    Input
    ----------
        mask: ndarray
            predicted mask
        center: ndarray
            predicted center, should have the same shape as mask
        check_center: bool - default False
            False: every pixel will be given the value of the nearst maxima
            True: for each particles, check how many centers it has. we will only split particles in mask if it has more than 2  
                  centers. Setting check_center = True could help to reduce false positive but it will take around 1 hour for 
                  mask with size of (464, 928, 928).
        
    Returns
    -------
        final: ndarray
            The propossed results. In final, every particle has a different value.
    """
    _,coords = local_maxima_3D(center,1)
    if check_center == False:
        coords_mask = np.asarray(np.where(mask>0)).T
        _,indexes = do_kdtree(coords,coords_mask)
        final = np.copy(mask)
        for i in range(len(indexes)):
            final[tuple(coords_mask[i])] = indexes[i]
    else:
        labeled_mask = label(mask)
        final = np.copy(mask)
        for i in np.unique(labeled_mask)[1:]:
            coords_mask = np.asarray(np.where(labeled_mask==i)).T
            particle_coords = []
            for j in range(coords.shape[0]):
                if np.all(coords_mask==coords[j],axis=(1)).any():
                    particle_coords.append(coords[j])
            if len(particle_coords) >= 2:
                _ ,indexes = do_kdtree(particle_coords,coords_mask)
                for k in range(len(indexes)):
                    final[tuple(coords_mask[k])] = indexes[k]+1          
        final = label(final)
        
    return final

def get_center_coords(mask):
    """
    Given instance-labelled mask (every particle has a different value), this function calculates the center coordinates of all particles. This function takes ~1 hour for mask with size of (464, 928, 928).

    Input
    ----------
        mask: ndarray
            predicted mask, every particle has a different value
        
    Returns
    -------
        coords: 2d list, e.g. [[coordX1,coordY1,coordZ1],[coordX2,coordY2,coordZ2]...]
            A list contains the XYZ coordinates of all particles in the mask
    """
    coords = []
    for i in np.unique(final)[1:]:
        volumeZ,volumeY,volumeX = np.mean(np.where(final==i),axis=-1)  # calculate the center
        coords.append([int(volumeZ),int(volumeY),int(volumeX)])
        
    return coords


def remove_outlier(mask,coords,neighbor_number=3,distance_threshold=30):
    """
    Given instance-labelled mask and the coordinate list of all particle center, this function will filter out the outliers, which don't have more than neighbor_number(3) nearst neighbor particles within distance_threshold (30) pixels.

    Input
    ----------
        mask: ndarray
            predicted mask, every particle has a different value
        coords: 2d list
            A list contains the XYZ coordinates of all particles in the mask
        neighbor_number: int - default 3
            threshold value, define how many neighbor particles are considered
        distance_threshold: int - default 30
            threshold value, define the distance to be considered
        
    Returns
    -------
        filted_mask: ndarray
            mask after filtering outliers, every particle has a different value
        coords_updated: 2d list
            A list contains the XYZ coordinates of all particles in the filted_mask
    """
    # sort the distance between each particles
    distance_matrix = pdist(coords)   # Calculate pairwise distances between coords
    distance_matrix = squareform(distance_matrix)   # Convert the distance matrix to a square-form distance matrix
    distance_matrix[distance_matrix==0] = 1000
    
    nearst_distance = np.partition(distance_matrix,neighbor_number,axis=1)[:,:neighbor_number]
    filted_mask = np.copy(mask)
    coords_updated = coords
    # check whether the dictances between one center and the nearst neighbor_number(3) centers are all smaller than distance_threshold(30). If no, considering it to be the outlier and remove it
    for i in np.where((nearst_distance>distance_threshold).any(axis=1)==True)[0]: 
        c = coords[i]
        value = mask[tuple(c)]
        if value > 0:
            filted_mask[mask==value] = 0   # remove the outliers from final mrc
            coords_updated.remove(c)
            
    return filted_mask,coords_updated