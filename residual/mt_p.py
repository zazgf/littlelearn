#coding=utf-8
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import EarlyStopping, ModelCheckpoint
import imageio
import tensorflow as tf
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
K.set_session(session)



model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
test_path = 'datasets/test'

model = load_model(model_path)
model.load_weights(model_weights_path)

img_width, img_height = 150, 150

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  with tf.device('/GPU:0'):
    array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 1:
    print("判断：石块较少")
  elif answer == 0:
    print("判断：石块较多")
  elif answer == 2:
    print("Predicted: ok")

  return answer

start = time.time()
for i, ret in enumerate(os.walk(test_path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    
    print(ret[0] + '/' + filename)
    result = predict(ret[0] + '/' + filename)
    print(" ")

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