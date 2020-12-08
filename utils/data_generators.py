import numpy as np
import random
import keras

from utils.utils import load_data, get_patch_position, dist_label, read_xml

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path_data, path_target, objlist,
                 mode = 'mask', batch_size=4, dim=56, Lrnd = 5, n_channels=1, augmentation_prob=True,
                 shuffle=True, random_seed=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle 
        self.augmentation = augmentation_prob
        self.mode = mode
        self.Lrnd = Lrnd         #random shifts applied when sampling data- and target-patches (in voxels)
        if random_seed == True:
            np.random.seed(1)
                
        self.objlist = read_xml(objlist)
        self.data, self.target = load_data(path_data, path_target)
        self.p_in = np.int(np.floor(dim / 2))
        
        self.len_train = len(self.objlist)
        self.indexes = list(np.arange(0,self.len_train))
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.on_epoch_end()
        return int(np.floor(self.len_train / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
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
            tomoID = int(self.objlist[ID]['tomo_idx'])
            tomodim = self.data[tomoID].shape
            sample_data = self.data[tomoID]
            sample_target = self.target[tomoID]
            x, y, z = get_patch_position(tomodim, self.p_in, self.objlist[ID], self.Lrnd)

            patch_data = sample_data[z-self.p_in:z+self.p_in, y-self.p_in:y+self.p_in, x-self.p_in:x+self.p_in]
            patch_target = sample_target[z-self.p_in:z+self.p_in, y-self.p_in:y+self.p_in, x-self.p_in:x+self.p_in]            
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize

            # Data augmentation (180degree rotation around tilt axis):
            if self.augmentation and np.random.uniform() < 0.5:
                patch_data = np.rot90(patch_data, k=2, axes=(0, 2))
                patch_target = np.rot90(patch_target, k=2, axes=(0, 2))

            data_batch[i,:,:,:,0] = patch_data
        
        if self.mode == 'mask':
            target_batch[i,:,:,:,0] = patch_target
            return data_batch, target_batch
        
        elif self.mode == 'center':
            target_batch[i,:,:,:,0] = dist_label(patch_target)
            return data_batch, target_batch
