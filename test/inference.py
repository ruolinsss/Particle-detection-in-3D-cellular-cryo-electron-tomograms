import sys
sys.path.append('../../') # add parent folder to path
import numpy as np
import time
from test.utils.models import my_model
from test.utils.utils import read_mrc,write_mrc

def inference(tomo_path,weights_path,dim=160,mode='mask'):
    """
    This function enables to segment a tomogram. As tomograms are too large to be processed in one take, the tomogram is decomposed in smaller overlapping 3D patches. Modified from https://gitlab.inria.fr/serpico/deep-finder/-/blob/master/deepfinder/inference.py#L19
    
    Input
    ----------
        tomo_path: string
            The path of the testing tomogram.
        weights_path: string
            The path of the trained model.
        dim: int
            Patch size to be fed into the model.
        mode: 'mask' or 'center' - default 'mask'
            'mask': using simple UNet and dice loss to predict the mask of particles
            'center': using simple UNet and mse loss to predict the center of particles
        
    Returns
    -------
        predArray: numpy array
            Contains predicted score maps, having the same shape as testing tomogram.
    """
    # Segmentation, parameters for dividing data in patches:
    pcrop = 25  # how many pixels to crop from border (net model dependent)
    poverlap = 2 * pcrop + 5 # patch overlap (in pixels) 
    l = np.int(dim / 2)
    lcrop = np.int(l - pcrop)
    step = np.int(2 * l + 1 - poverlap)
    
    # load data and model
    test_data,header_dict = read_mrc(tomo_path)
    test_data = (test_data - np.mean(test_data)) / np.std(test_data)  # normalize
    test_data = np.pad(test_data, pcrop, mode='constant', constant_values=0)  # zeropad
    test_size = test_data.shape
    
    model = my_model(dim, mode=mode)
    model.load_weights(weights_path)

    # Get patch centers:
    pcenterX = list(range(l, test_size[0] - l, step))  # list() necessary for py3
    pcenterY = list(range(l, test_size[1] - l, step))
    pcenterZ = list(range(l, test_size[2] - l, step))

    # If there are still few pixels at the end:
    if pcenterX[-1] < test_size[0] - l:
        pcenterX = pcenterX + [test_size[0] - l, ]
    if pcenterY[-1] < test_size[1] - l:
        pcenterY = pcenterY + [test_size[1] - l, ]
    if pcenterZ[-1] < test_size[2] - l:
        pcenterZ = pcenterZ + [test_size[2] - l, ]

    Npatch = len(pcenterX) * len(pcenterY) * len(pcenterZ)
    print('Data array is divided in ' + str(Npatch) + ' patches ...')

    # Process data in patches:
    predArray = np.zeros(test_size , dtype=np.float16)
    normArray = np.zeros(test_size, dtype=np.int8)
    patchCount = 1

    for x in pcenterX:
        for y in pcenterY:
            for z in pcenterZ:
                print('Segmenting patch ' + str(patchCount) + ' / ' + str(Npatch) + ' ...')
                patch = test_data[x - l:x + l, y - l:y + l, z - l:z + l]
                patch = np.reshape(patch, (1, dim, dim, dim, 1))  

                pred = model.predict(patch, batch_size=1)

                predArray[x - lcrop:x + lcrop, y - lcrop:y + lcrop, z - lcrop:z + lcrop] = predArray[
                                                                                              x - lcrop:x + lcrop,
                                                                                              y - lcrop:y + lcrop,
                                                                                              z - lcrop:z + lcrop]  +                                                                                                         np.float16(pred[0,
                                                                                                   l - lcrop:l + lcrop,
                                                                                                   l - lcrop:l + lcrop,
                                                                                                   l - lcrop:l + lcrop,0])
                normArray[x - lcrop:x + lcrop, y - lcrop:y + lcrop, z - lcrop:z + lcrop] = normArray[
                                                                                           x - lcrop:x + lcrop,
                                                                                           y - lcrop:y + lcrop,
                                                                                           z - lcrop:z + lcrop] + np.ones(
                    (dim - 2 * pcrop, dim - 2 * pcrop, dim - 2 * pcrop), dtype=np.int8)

                patchCount += 1

    # Normalize overlaping regions:
    predArray = predArray / normArray
    predArray = predArray[pcrop:-pcrop, pcrop:-pcrop, pcrop:-pcrop]  # unpad
    
    return predArray,header_dict
    
    

if __name__=='__main__':
    '''
    Following information should be given:
    
    tomo_path: string
        Path of the testing data/
    weights_path: string
        Path of the saved trained model weights.
    output_path: sting
        Path to save the predicted mask/center. 
    mode: string, 'mask' or 'center'
        Whether you want to predict mask or center. Should be corresponding to the given weights_path.
    patch_size: int
        Inference patch size, must be multiple of 4  
    '''
    
    tomo_path    = '/home/haicu/ruolin.shen/projects/train/tomo17_re.mrc' 
    weights_path = '/home/haicu/ruolin.shen/projects/train/results/model_cell_center.h5' 
    output_path = 'output/'
    mode = 'center'
    patch_size   = 160 
    
    start = time.time()
    pred, header_dict = inference(tomo_path,weights_path,dim=patch_size,mode=mode)
    end = time.time()
    print("Model took %0.2f seconds to predict" % (end - start))
    write_mrc(pred,output_path+mode+'_pred.mrc',header_dict = header_dict)


