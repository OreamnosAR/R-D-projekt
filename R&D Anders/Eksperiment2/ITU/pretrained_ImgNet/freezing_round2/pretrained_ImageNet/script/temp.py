import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from keras import backend as K

image_size = 256 # https://pubs.rsna.org/doi/pdf/10.1148/ryai.210315
num_epoches = 200 #With early stopping. ITU
batch_size = 16 #Breast = 16 jf. artikelen fra ITU: https://arxiv.org/pdf/2302.08272.pdf
lr = 0.00001 #ITU = fixed 1e-5    vs  mei = varierende
model_name = "ResNet50_repli_RadImgNet"
structure = 'freezeall'
database = 'RadImageNet'
#rescale=1./255? YES
current_directory = os.environ['HOME']

def get_compiled_model(k):
    base_model = ResNet50(weights=None, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
    #for layer in base_model.layers:
    #    layer.trainable = False
    
    y = base_model.output
    y = Dropout(0.5)(y)
    predictions = Dense(2, activation='softmax')(y)
    model = Model(inputs=base_model.input, outputs=predictions)

    i = 1
    w_path = os.path.join(current_directory,"radimagenet/WorkspaceV2/Replication_ITU_study/pretrained_RadImageNet/weights/breast_data/breast-freezeall-fold"+str(i+1)+"-RadImageNet-ResNet50_repli_RadImgNet-256-16-1e-05.h5")
    model_dir  = os.path.join(current_directory,w_path)
    model.load_weights(model_dir)

    for layer in base_model.layers[:-k]: #Whole model??!
        layer.trainable = False

    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss=BinaryCrossentropy(), metrics=[keras.metrics.AUC(name='auc')])
    return model

model = get_compiled_model(1) #Train final classification layer
model.summary()
# 4098 trainable params

model = get_compiled_model(0) #Train whole model
model.summary()
# 23,591,810 trainable params

del model
K.clear_session()

