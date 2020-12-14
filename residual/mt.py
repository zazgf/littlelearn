import numpy as np
from keras import layers
from keras.layers import Input, Add,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os
from keras import callbacks
import time
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import EarlyStopping, ModelCheckpoint
import imageio
import keras.backend as K
import math
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
K.set_session(session)

method_2 = False
method_3 = True

train_data_path = 'data/train'
validation_data_path = 'data/valid'

img_height = 150
img_width = 150
classes = 2


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters = F1, kernel_size=(1,1) ,strides=(1,1),padding='valid',name=conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2,kernel_size=(f,f),strides=(1,1),padding='same',name = conv_name_base + '2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding='valid',name = conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)
    X= Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X
def convolutional_block(X, f, filters, stage, block, s = 2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X
def ResNet50(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = AveragePooling2D((2,2), name="avg_pool")(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model

if __name__ == '__main__':
    model = ResNet50(input_shape = (img_width, img_height, 3), classes = classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    if method_2:
        DATA_DIR = 'data'
        TRAIN_DIR = os.path.join(DATA_DIR, 'train')
        VALID_DIR = os.path.join(DATA_DIR, 'valid')
        SIZE = (64, 64)
        BATCH_SIZE = 6
        num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
        num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

        num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
        num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

        gen = keras.preprocessing.image.ImageDataGenerator()
        val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

        batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
        val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    if method_3:
        batch_size = 4
        
        num_train_samples = sum([len(files) for r, d, files in os.walk(train_data_path)])
        num_valid_samples = sum([len(files) for r, d, files in os.walk(validation_data_path)])

        samples_per_epoch = math.floor(num_train_samples/batch_size)
        validation_steps = math.floor(num_valid_samples/batch_size)

        epochs = 2
        train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_path,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

    if method_2:
        early_stopping = EarlyStopping(patience=10)
        checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)
        log_dir = './tf-log/'
        tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
        cbks = [tb_cb]
        with tf.device('/GPU:0'):
            model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=20, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
        plot_model(model, to_file='model.png')
        SVG(model_to_dot(model).create(prog='dot', format='svg'))
        model.save("myMode")
    if method_3:
        log_dir = './tf-log/'
        tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
        cbks = [tb_cb]
        start = time.time()
        with tf.device('/GPU:0'):
            model.fit_generator(
                train_generator,
                steps_per_epoch=samples_per_epoch,
                epochs=epochs,
                callbacks=cbks,
                validation_data=validation_generator,
                validation_steps=validation_steps)
        target_dir = './models/'
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        model.save('./models/model.h5')
        model.save_weights('./models/weights.h5')
        end = time.time()
        dur = end-start
        if dur<60:
            print("Execution Time:",dur,"seconds")
        elif dur>60 and dur<3600:
            dur=dur/60
            print("Execution Time:",dur,"minutes")
        else:
            dur=dur/(60*60)
            print("Execution Time:",dur,"hours")
