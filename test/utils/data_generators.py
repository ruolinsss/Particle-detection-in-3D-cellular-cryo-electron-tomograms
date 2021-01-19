import numpy as np
import random
import keras

#from utils import load_data, get_patch_position, dist_label, read_xml
from test.utils.utils import load_data, get_patch_position, read_xml
from test.utils.data_generator_helper import random_rotation_3d, dist_label

# def random_rotation_3d(image, gt, max_angle, order=1):
#     """ Randomly rotate an image by a random angle (-max_angle, max_angle).

#     Arguments:
#     max_angle: `float`. The maximum rotation angle.
#     image1: 3d image (z,y,x)
#     order: whether use interpolation or not:
#            1: use linear interpolation
#            0: not use interpolation (for instance label)

#     Returns:
#     batch of rotated 3D images
#     """    
#     # rotate along x-axis
#     angle = random.uniform(-max_angle, max_angle)
# #     angle=45
#     imagex = scipy.ndimage.interpolation.rotate(image, angle,  axes=(0, 1), reshape=False, order=1)
#     gtx = scipy.ndimage.interpolation.rotate(gt, angle,  axes=(0, 1), reshape=False, order=order)

# #     # rotate along y-axis
# #     angle=45
#     angle = random.uniform(-max_angle, max_angle)
#     imagey = scipy.ndimage.interpolation.rotate(imagex, angle, axes=(0, 2), reshape=False, order=1)
#     gty = scipy.ndimage.interpolation.rotate(gtx, angle,  axes=(0, 2), reshape=False, order=order)

# #     #rotate along z-axis
#     angle = random.uniform(-max_angle, max_angle)
# #     angle=45
#     imagez = scipy.ndimage.interpolation.rotate(imagey, angle, axes=(1, 2), reshape=False, order=1)
#     gtz = scipy.ndimage.interpolation.rotate(gty, angle,  axes=(1, 2), reshape=False, order=order)
    
#     return imagez,gtz

class DataGenerator(keras.utils.Sequence):
    """
    Data Generator with real time data augmentation (180 degree rotation around tilt axis) for training process. 

    Input
    ----------
        path_data: string
            The path of the training/validation tomogram.
        path_target: string
            The path of the training/validation groundtruth.
        objlist: string
            The path of the training/validation particle coordinates xml file.
        mode: 'mask' or 'center' - default 'mask'
            'mask': predict the mask of particle
            'center': predict the center of particles.
        batch_size: int
        dim: int
            Patch size for training or validation.
        Lrnd: int
            Random shifts applied when sampling data- and target-patches (in voxels).
        n_channels: int
            Channel of training tomogram.
        augmentation: bool - default True
            whether need real time data augmentation or not.
        shuffle: bool - default True
            whether to shuffle the order of training patches.
        random_seed: bool - default False
            whether to fix the initialization of a pseudorandom generator.
    Returns
    -------
    """
    def __init__(self, path_data, path_target, objlist,
                 mode = 'mask', batch_size=4, dim=56, Lrnd = 5, n_channels=1, augmentation=True,
                 shuffle=True, random_seed=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle 
        self.augmentation = augmentation
        self.mode = mode
        self.Lrnd = Lrnd        
        if random_seed == True:
            np.random.seed(1)
        
#         self.train = train
#         if train==True:
#             objlist1 = read_xml(objlist[0])
#             objlist2 = read_xml(objlist[1])
#             self.objlist =  objlist1+objlist2[300:500]+objlist2[750:820]+objlist2[850:]
#         else:
        self.objlist = read_xml(objlist)
        self.data, self.target = load_data(path_data, path_target)
        self.p_in = np.int(np.floor(dim / 2))
        
        self.len_train = len(self.objlist)
#         self.indexes = list(np.arange(0,self.len_train))
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.on_epoch_end()
        return int(np.floor(self.len_train / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = list(np.arange(0,self.len_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y

    def __data_generation(self,indexes):
        'Generates data containing batch_size samples' 
        # Initialization
        data_batch = np.empty((self.batch_size, self.dim, self.dim, self.dim, self.n_channels))
        target_batch = np.empty((self.batch_size, self.dim, self.dim, self.dim, 1))
        # Generate data
        for i, ID in enumerate(indexes):
#             if self.train:
            tomoID = int(self.objlist[ID]['tomo_idx'])
#             else:
#                 tomoID = 0
            tomodim = self.data[tomoID].shape
            sample_data = self.data[tomoID]
            sample_target = self.target[tomoID]
            x, y, z = get_patch_position(tomodim, self.p_in, self.objlist[ID], self.Lrnd)

            patch_data = sample_data[z-self.p_in:z+self.p_in, y-self.p_in:y+self.p_in, x-self.p_in:x+self.p_in]
            patch_target = sample_target[z-self.p_in:z+self.p_in, y-self.p_in:y+self.p_in, x-self.p_in:x+self.p_in] * 100            
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize

            # Data augmentation (180degree rotation around tilt axis):
            if self.augmentation:
                patch_data,patch_target = random_rotation_3d(patch_data,patch_target,180,order=0) # random rotation
#                 rotate = np.random.randint(0,4)
#                 patch_data = np.rot90(patch_data, k=rotate, axes=(0, 2))
#                 patch_target = np.rot90(patch_target, k=rotate, axes=(0, 2))

        
            if self.mode == 'mask':
                patch_target = np.where(patch_target>1,1,0)
#                 if self.augmentation:
#                     patch_data,patch_target = random_rotation_3d(patch_data,patch_target,180, order=0)

            elif self.mode == 'center':
                patch_target = dist_label(patch_target)
#                 if self.augmentation:
#                     patch_data,patch_target = random_rotation_3d(patch_data,patch_target,180, order=1)
                    
            target_batch[i,:,:,:,0] = patch_target        
            data_batch[i,:,:,:,0] = patch_data
            
        return data_batch, target_batch
