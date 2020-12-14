import numpy as np
import time

from utils.utils import read_mrc,write_mrc,write_txt
from utils.postprocessing_helper import find_center,get_center_coords,remove_outlier
from skimage.morphology import remove_small_objects  

pred_mask_path = '/home/haicu/ruolin.shen/projects/3dpd/output/mask_pred.mrc' 
pred_center_path = '/home/haicu/ruolin.shen/projects/3dpd/output/center_pred.mrc' 
pred_path = 'output/'
voxel_size = 14.08 # voxel size for different dataset, 14.08 for spinach data

mask,header = read_mrc(pred_mask_path)
center,_ = read_mrc(pred_center_path)

# post processing parameters
# post processing choice:
# 1. remove_min_size: filter out small objects
# 2. check_center: for each particles, check how many centers it has. we will only split particles in mask if it has more than 2 centers.
# 3. filter_outlier: filter out the outliers (there are fewer than neighbor_number neighbor particles within distance_threshold pixels)

mask_threshold = 0.1
center_threshold = 0.1
remove_min_size = 64  # Remove objects smaller than the specified size.
check_center = True
filter_outlier = True
neighbor_number = 3
distance_threshold = 30

mask = np.where(mask > mask_threshold, True, False)
# post processing 1
mask = remove_small_objects(mask,remove_min_size) 
mask = np.where(mask==True, 1, 0)
center = np.where(center >= center_threshold, center, 0)

# post processing 2
start = time.time()
processed_mask = find_center(mask,center,check_center=check_center)
end = time.time()
print("Check center function takes %0.2f seconds to run" % (end - start))
start = time.time()
coords = get_center_coords(processed_mask)
end = time.time()
print("get center coords function takes %0.2f seconds to run" % (end - start))

# post processing 3
if filter_outlier == True:
    start = time.time()
    processed_mask,coords = remove_outlier(processed_mask,coords,
                                           neighbor_number=neighbor_number,distance_threshold=distance_threshold)
    end = time.time()
    print("remove outlier function takes %0.2f seconds to run" % (end - start))


# save processed_mask to mrc file and save coords to txt file
processed_mask = processed_mask.astype(np.float32)
write_mrc(processed_mask,pred_path+'pred_tomo.mrc',header_dict=header)
write_txt(coords,pred_path+'pred_center_coordinate.txt',voxel_size=voxel_size)