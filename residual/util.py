import open3d as o3d
import numpy as np
import os
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from keras.layers import Input, Add,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import imageio
import keras.backend as K
import math
# K.set_image_data_format('channels_last')
# K.set_learning_phase(1)
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# K.set_session(session)
NUM_POINT = 1024
npoints = 3
classes = 2
root_path = "lidar_data"
train_dir = os.path.join(root_path,'s1')
test_dir = os.path.join(root_path,'s2')

T1=[]
TE=[]
for i, ret in enumerate(os.walk(train_dir)):
      for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue    
        
        pcd = o3d.io.read_point_cloud(os.path.join(train_dir,filename))
        point_set = np.asarray(pcd.points)
        point_set = point_set[:NUM_POINT, 0:npoints]
        point_set = np.array([point_set])
        point_set = point_set[0]
        # np.reshape(point_set,[32,32,3])
        T1.append(point_set)
        TE.append([0,1])
for i, ret in enumerate(os.walk(test_dir)):
      for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue    
        pcd = o3d.io.read_point_cloud(os.path.join(test_dir,filename))
        point_set = np.asarray(pcd.points)
        point_set = point_set[:NUM_POINT, 0:npoints]
        point_set = np.array([point_set])
        point_set = point_set[0]
        # np.reshape(point_set,[32,32,3])
        print(point_set.shape)
        T1.append(point_set)
        TE.append([1,0])
T1=np.asarray(T1)
TE=np.asarray(TE)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(T1,TE,test_size=0.2,random_state=0)
print(x_train.shape)
print(y_train)
from lidar_net import ResNet50,identity_block,convolutional_block
if __name__ == '__main__':
  model = ResNet50(input_shape = (1024, 3), classes = classes)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  model.fit(x_train, y_train, epochs = 2, batch_size = 32)
