from keras import backend as K
import keras

def dice_coef_loss(y_true, y_pred):
    """
    This function calculates a standard dice loss between y_true and y_pred
    
    Input
    ----------
        y_true: tensor
            groundtruth
        y_pred: tensor
            prediction from network
        
    Returns
    -------
        loss: float
            Calculated dice loss
    """
    intersection = K.sum(y_true * y_pred, axis=[1,2,3,4])
    union = K.sum(y_true, axis=[1,2,3,4]) + K.sum(y_pred, axis=[1,2,3,4])
    loss = 1 - K.mean( (2. * intersection + 1e-9) / (union + 1e-9), axis=0)
    return loss

def model_loss(mode='mask'):
    '''
    This function returns the caluclated model loss according to different modes.
    
    Input
    ----------
        mode: 'mask' or 'center' - default 'mask'
            'mask': using simple UNet and dice loss to predict the mask of particles
            'center': using simple UNet and mse loss to predict the center of particles
    '''
    def loss(y_true, y_pred):
        if mode == 'mask':
            return dice_coef_loss(y_true, y_pred)
        elif mode == 'center':
            mse = tf.keras.losses.MeanSquaredError()
            return mse(y_true, y_pred)
        else:
            raise ValueError('Mode should be either ''mask'' or ''center''')
    
    return loss
    
