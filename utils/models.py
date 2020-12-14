from keras.layers import Input, concatenate
from keras.models import Model
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D

def my_model(dim_in, mode = 'mask'):
    '''
    Build the training model - a simple UNet, modified from https://gitlab.inria.fr/serpico/deep-finder/-/blob/master/deepfinder/models.py#L14
    
    Input
    ----------
        dim_in: int
            the input patch size
        mode: 'mask' or 'center' - default 'mask'
            'mask': using simple UNet and dice loss to predict the mask of particles
            'center': using simple UNet and mse loss to predict the center of particles
    '''
    
    input = Input(shape=(dim_in,dim_in,dim_in,1))
    
    x    = Conv3D(32, (3,3,3), padding='same', activation='relu')(input)
    high = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    
    x = MaxPooling3D((2,2,2), strides=None)(high)
    
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    mid = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    
    x = MaxPooling3D((2,2,2), strides=None)(mid)
    
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2,2,2), data_format='channels_last')(x)
    x = Conv3D(64, (2,2,2), padding='same', activation='relu')(x)
    
    x = concatenate([x, mid])
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    
    x = UpSampling3D(size=(2,2,2), data_format='channels_last')(x)
    x = Conv3D(48, (2,2,2), padding='same', activation='relu')(x)
    
    x = concatenate([x, high])
    x = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    
    if mode == 'mask':
        output = Conv3D(1, (1,1,1), padding='same', activation='sigmoid')(x)
    elif mode == 'center':
        output = Conv3D(1, (1,1,1), padding='same', activation='relu')(x)
#         output_cell = Conv3D(1, (1,1,1), padding='same', activation='sigmoid',name='cell')(x)
#         output_mask = Conv3D(1, (1,1,1), padding='same', activation='sigmoid',name='mask')(x)
    
#         model = Model(input, [output_cell,output_mask])
    else:
        raise ValueError('Mode should be either ''mask'' or ''center''')
    
    model = Model(input, output)
    return model