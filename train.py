# keras 2.2.4 conda env instantdl
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
sess = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
print("GPU available? ", sess)


import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger, ReduceLROnPlateau
from keras.models import Model

from test.utils.models import my_model
from test.utils.losses import model_loss,dice_coef_loss
from test.utils.data_generators import DataGenerator


def train(path_data,path_target,valid_data,valid_target,train_list,valid_list,model_path,
          dim=56,epoch=10,batch_size=4,lr=0.0001,mode='mask'):
    """
    This function trains the network.
    
    Input
    ----------
        path_data: string
            The path of the training tomogram.
        path_target: string
            The path of the training groundtruth.
        valid_data: string
            The path of the validation tomogram.
        valid_target: string
            The path of the validation groundtruth.
        train_list: string
            The path of the training particle coordinates xml file.
        valid_list: string
            The path of the validation particle coordinates xml file.
        model_path: string
            The path to save the trained model
        dim: int
            Patch size for training or validation.
        epoch: int
        batch_size: int            
        mode: 'mask' or 'center' - default 'mask'
            'mask': predict the mask of particle
            'center': predict the center of particles.
    Returns
    -------
    """

    training_generator = DataGenerator(path_data, path_target, train_list, mode=mode, dim=dim, batch_size=batch_size,augmentation=True)
    validation_generator = DataGenerator(valid_data, valid_target, valid_list, mode=mode, dim=dim, batch_size=batch_size,augmentation=False)

    model = my_model(dim, mode = mode)
    model.summary()
    opt = Adam(lr=lr, beta_1=0.95, beta_2=0.99)
    model.compile(optimizer=opt, loss = model_loss(mode=mode))

    Early_Stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.000001)
    model_checkpoint = ModelCheckpoint(filepath = model_path+'/'+mode+'_model.h5', verbose=1, save_best_only=True) 
    callbacks_list = [model_checkpoint, Early_Stopping,reduce_lr]
    print('begin training ...')
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=epoch,
                        verbose=1,
                        callbacks = callbacks_list)

if __name__=='__main__':
    '''
    Following information should be given:
    
    path_data: string
        Path of the training data
    path_target: string
        Path of the groundtruth
    valid_data: string
        Path of the validation data
    valid_target: string
        Path of the validation groundtruth
    train_list: string
        Path of the xml file including information about the training data, such as particle coordinates
    valid_list: string
        Path of the xml file including information about the validation data, such as particle coordinates
    model_path: sting
        Path to save the trained model weights. 
        Model from SegNet will be saved called ''mask_model.h5'' in the model_path.
        Model from CenterNet will be saved called ''center_model.h5'' in the model_path.
    dim: int
        Training patch size    
    '''
        
    path_data = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/Tomo32_denoised_bin4.mrc',
                '/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/Tomo17_bin4_denoised.mrc']  
    path_target = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach_back/labelmap0.mrc',
                  '/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach_back/labelmap1.mrc'] 

#     valid_data = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/Tomo17_bin4_denoised.mrc'] 
#     valid_target = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach_back/labelmap1.mrc']
    
#     train_list = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/pos_file0.xml',
#                   '/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/pos_file1.xml']
#     valid_list = '/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/pos_file1.xml'
    
    train_list = '/home/haicu/ruolin.shen/projects/3dpd/train.xml'
    valid_list = '/home/haicu/ruolin.shen/projects/3dpd/val.xml'
    
    model_path = 'output'
    dim = 56
    
    # train mask model first
    train(path_data,path_target,path_data,path_target,train_list,valid_list,model_path,
          dim=dim,epoch=500,batch_size=8,lr=0.0005,mode='mask')
    # train center model 
    train(path_data,path_target,path_data,path_target,train_list,valid_list,model_path,
          dim=dim,epoch=500,batch_size=8,lr=0.00001,mode='center')


