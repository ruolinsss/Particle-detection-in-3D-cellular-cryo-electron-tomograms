from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
sess = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
print("GPU available? ", sess)

import numpy as np
import time
import argparse

from test.utils.models import my_model
from test.utils.utils import read_mrc,write_mrc,write_txt
from test.inference import inference
from test.postprocessing import postprocessing

def run(tomo_path,
        mask_path='./output/mask_model.h5',
        center_path='./output/center_model.h5',
        output_path='output/',
        patch_size=160,
        voxel_size=14.08,
        mask_threshold = 0.1,
        center_threshold = 0.1, 
        remove_min_size = 64,  
        check_center = True, 
        filter_outlier = True, 
        neighbor_number = 3,
        distance_threshold = 30):
    """
    This function is where the end2end framework is run, including inference and postprocessing. The result of this function will be a instance labelled particle detection in mrc file and the center coordinates of all detected particles in txt file.
        
    Input
    ----------
        tomo_path: string
            The path of the testing tomogram.
        mask_path: string - default './output/mask_model.h5'
            The path of the trained SegNet model weights. If user hasn't modified the output path in train, the trained model 
            will be saved with the default path.
        center_path: string - default './output/center_model.h5'
            The path of the trained CenterNet model weights. If user hasn't modified the output path in train, the trained model 
            will be saved with the default path.
        output_path: string - default 'output/'
            The path of the saved mrc file and txt file.
        patch_size: int - default 160
            Patch size to be fed into the inference model.
        voxel_size: int - default 14.08
            Voxel size for different dataset, e.g. 14.08 for spinach data.
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
    """
    start = time.time()
    
    # inference
    pred_mask, header_dict = inference(tomo_path,mask_path,dim=patch_size,mode='mask')
    pred_center, _ = inference(tomo_path,center_path,dim=patch_size,mode='center')
    
    # post processing
    post_mask,coords = postprocessing(pred_mask,pred_center,
                                  mask_threshold,center_threshold,
                                  remove_min_size,check_center,
                                  filter_outlier,neighbor_number,distance_threshold)
    
    # save processed_mask to mrc file and save coords to txt file
    write_mrc(post_mask,output_path+'pred_tomo.mrc',header_dict=header_dict)
    write_txt(coords,output_path+'pred_center_coordinate.txt',voxel_size=voxel_size)
    
    end = time.time()
    print("Model took %0.2f seconds to predict" % (end - start))



def get_args():
    '''
    Required arguments
    ------------------
        -tomo: The path of the testing tomogram.
        
    Optional arguments
    ------------------
        -m: The path of the trained SegNet model weights.
        -c: The path of the trained CenterNet model weights.
        -o: The path of the saved mrc file and txt file.
        -ps: Patch size to be fed into the inference model.
        -vs: Voxel size for different dataset, e.g. 14.08 for spinach data.
        -mt: Threshold to filter the small noise in predicted mask.
        -ct: Threshold to filter the small noise in predicted center.
        -rs: Particles smaller than this size will be removed.
        -check_center: Whether to use check center strategy in the post processing.
        -filter_outlier: Whether to use filter outlier strategy in the post processing.
        -nn: Define how many neighbor particles are considered in filter outlier strategy. Required when filter_outlier is True.
        -dt: Define the distance to be considered in filter outlier strategy. Required when filter_outlier is True.
        
    '''
    parser = argparse.ArgumentParser(description='Run the end-2-end framework to detect particles in 3D cellular cryo-electron tomograms', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            
    parser.add_argument('--tomo_path', '-tomo', metavar='INPUT', required=True,
                        help='The path of the testing tomogram.')
    parser.add_argument('--mask_path', '-m', default= './output/mask_model.h5',
                        help='The path of the trained SegNet model weights.')
    parser.add_argument('--center_path', '-c', default='./output/center_model.h5', 
                        help='The path of the trained CenterNet model weights.')
    parser.add_argument('--output_path', '-o', default='./output/',
                        help='The path of the saved mrc file and txt file.')    
    parser.add_argument('--patch_size', '-ps',type=int, default=160,
                        help='Patch size to be fed into the inference model.')
    parser.add_argument('--voxel_size', '-vs',type=int, default=14.08,
                        help='Voxel size for different dataset, e.g. 14.08 for spinach data.')
    parser.add_argument('--mask_threshold', '-mt',type=int, default=0.1,
                        help='Threshold to filter the small noise in predicted mask.')
    parser.add_argument('--center_threshold', '-ct', type=int, default=0.1,
                        help='Threshold to filter the small noise in predicted center.')
    parser.add_argument('--remove_min_size', '-rs',type=int, default= 64,
                        help='Particles smaller than this size will be removed.')
    parser.add_argument('--check_center', '-check_center',type=bool, default=True, 
                        help='Whether to use check center strategy in the post processing.')
    parser.add_argument('--filter_outlier', '-filter_outlier',type=bool, default=True,
                        help='Whether to use filter outlier strategy in the post processing.')    
    parser.add_argument('--neighbor_number', '-nn', type=int, default=3,
                        help='Define how many neighbor particles are considered in filter outlier strategy. Required when filter_outlier is True.')
    parser.add_argument('--distance_threshold', '-dt',type=int, default=30,
                        help='Define the distance to be considered in filter outlier strategy. Required when filter_outlier is True.')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()
    run(args.tomo_path, 
        mask_path=args.mask_path, 
        center_path=args.center_path, 
        output_path=args.output_path,
        patch_size=args.patch_size,
        voxel_size=args.voxel_size,
        mask_threshold = args.mask_threshold,
        center_threshold = args.center_threshold, 
        remove_min_size = args.remove_min_size, 
        check_center = args.check_center, 
        filter_outlier = args.filter_outlier, 
        neighbor_number = args.neighbor_number,
        distance_threshold = args.distance_threshold)
