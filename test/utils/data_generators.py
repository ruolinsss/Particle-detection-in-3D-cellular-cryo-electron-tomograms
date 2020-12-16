import numpy as np
import random
import keras

from utils.utils import load_data, get_patch_position, dist_label, read_xml

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
            tomoID = 0#int(self.objlist[ID]['tomo_idx'])
            tomodim = self.data[tomoID].shape
            sample_data = self.data[tomoID]
            sample_target = self.target[tomoID]
            x, y, z = get_patch_position(tomodim, self.p_in, self.objlist[ID], self.Lrnd)

            patch_data = sample_data[z-self.p_in:z+self.p_in, y-self.p_in:y+self.p_in, x-self.p_in:x+self.p_in]
            patch_target = sample_target[z-self.p_in:z+self.p_in, y-self.p_in:y+self.p_in, x-self.p_in:x+self.p_in] * 100            
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize

            # Data augmentation (180degree rotation around tilt axis):
            if self.augmentation and np.random.uniform() < 0.5:
                patch_data = np.rot90(patch_data, k=2, axes=(0, 2))
                patch_target = np.rot90(patch_target, k=2, axes=(0, 2))

            data_batch[i,:,:,:,0] = patch_data
        
            if self.mode == 'mask':
                patch_target = np.where(patch_target>1,1,0)
                target_batch[i,:,:,:,0] = patch_target

            elif self.mode == 'center':
                target_batch[i,:,:,:,0] = dist_label(patch_target)
            
        return data_batch, target_batch
