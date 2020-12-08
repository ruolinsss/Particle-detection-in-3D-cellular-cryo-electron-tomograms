# keras 2.2.4 conda env instantdl
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping,CSVLogger, ReduceLROnPlateau
from keras.models import Model

from utils.models import my_model
from utils.losses import model_loss
from utils.data_generators import DataGenerator


def train(path_data,path_target,valid_data,valid_target,train_list,valid_list,model_path,
          dim=56,epoch=10,batch_size=4,mode='mask'):
    
    training_generator = DataGenerator(path_data, path_target, train_list, mode=mode, dim=dim, batch_size=batch_size)
    validation_generator = DataGenerator(valid_data, valid_target, valid_list, mode=mode, dim=dim, batch_size=batch_size)

    model = my_model(dim, mode = mode)
    model.summary()
    opt = Adam(lr=0.000005, beta_1=0.95, beta_2=0.99)
    model.compile(optimizer=opt, loss = model_loss(mode=mode))

    Early_Stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.0000001)
    model_checkpoint = ModelCheckpoint(filepath = model_path+'/'+mode+'_model-{epoch:02d}.h5', verbose=1, save_best_only=True) 
    callbacks_list = [model_checkpoint, Early_Stopping,reduce_lr]
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=epoch,
                        verbose=1,
                        callbacks = callbacks_list)

if __name__=='__main__':
    path_data = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/Tomo32_denoised_bin4.mrc']
    path_target = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach_back/labelmap0.mrc']

    valid_data = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/Tomo17_bin4_denoised.mrc']
    valid_target = ['/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach_back/labelmap1.mrc']

    train_list = '/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/pos_file0.xml'
    valid_list = '/home/haicu/ruolin.shen/DeepFinder_usage/deep-finder/spinach/pos_file1.xml'
    
    model_path = '.output'
    dim = 56
    # train mask model first
    train(path_data,path_target,valid_data,valid_target,train_list,valid_list,model_path,
          dim=dim,epoch=5,batch_size=4,mode='mask')
    # train center model 
    train(path_data,path_target,valid_data,valid_target,train_list,valid_list,model_path,
          dim=dim,epoch=20,batch_size=4,mode='center')

   
