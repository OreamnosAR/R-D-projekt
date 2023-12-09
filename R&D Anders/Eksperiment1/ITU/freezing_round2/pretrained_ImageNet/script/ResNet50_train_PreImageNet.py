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

#python WorkspaceV2/Repli_ITU_freezing_strategy/pretrained_ImageNet/script/ResNet50_train_PreImageNet.py

#THIS is round 2 of the freeze strategy...
image_size = 256 # https://pubs.rsna.org/doi/pdf/10.1148/ryai.210315
num_epoches = 200 #With early stopping. ITU
batch_size = 16 #Breast = 16 jf. artikelen fra ITU: https://arxiv.org/pdf/2302.08272.pdf
lr = 0.00001 #ITU = fixed 1e-5    vs  mei = varierende
model_name = "ResNet50_repli_RadImgNet"
structure = 'freezeall_step2'
database = 'RadImageNet'
#rescale=1./255? YES
current_directory = os.environ['HOME']
structure = 'unfreezeall'

def get_compiled_model(fold_idx):
    base_model = ResNet50(weights=None, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
    y = base_model.output
    y = Dropout(0.5)(y)
    predictions = Dense(2, activation='softmax')(y)
    model = Model(inputs=base_model.input, outputs=predictions)

    w_path = os.path.join(current_directory,"radimagenet/WorkspaceV2/Replication_ITU_study/pretrained_ImageNet/weights/breast_data/breast-freezeall-fold"+str(i+1)+"-ImageNet-ResNet50_repli_ImgNet-256-16-1e-05.h5")
    model_dir  = os.path.join(current_directory,w_path)
    model.load_weights(model_dir)

    if structure == 'unfreezeall':
        print('Train whole model')
    #for layer in model.layers[:]: 
    #    layer.trainable = False

    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss=BinaryCrossentropy(), metrics=[keras.metrics.AUC(name='auc')])
    return model

def run_model(fold_idx):
    model = get_compiled_model(fold_idx)
    model.summary()
    ### Set train steps and validation steps
    train_steps = len(train_generator.labels)/batch_size
    val_steps = len(validation_generator.labels)/batch_size
    
    #### set the path to save models having lowest validation loss during training
    save_model_dir = os.path.join(current_directory, "radimagenet/WorkspaceV2/Repli_ITU_freezing_strategy/pretrained_ImageNet/weights/breast_data/")
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)    
    filepath = os.path.join(current_directory,"radimagenet/WorkspaceV2/Repli_ITU_freezing_strategy/pretrained_ImageNet/weights/breast_data/breast-"+structure+"-fold" + str(i+1) + "-" + database + "-" + model_name + "-" + str(image_size) + "-" + str(batch_size) + "-"+str(lr)+ ".h5") 
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)#, start_from_epoch=30?...nope...

    history = model.fit(train_generator,
                        epochs=num_epoches,
                        steps_per_epoch=train_steps,
                        validation_data=validation_generator,
                        validation_steps=val_steps,
                        use_multiprocessing=True,
                        workers=0, #10, #No GPU atm
                        callbacks=[checkpoint, earlyStop])
    ### Save training loss
    train_auc = history.history['auc']
    val_auc = history.history['val_auc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    d_loss = pd.DataFrame({'train_auc':train_auc, 'val_auc':val_auc, 'train_loss':train_loss, 'val_loss':val_loss})
    save_loss_dir = os.path.join(current_directory,"radimagenet/WorkspaceV2/Repli_ITU_freezing_strategy/pretrained_ImageNet/training_history")
    if not os.path.exists(save_loss_dir):
        os.mkdir(save_loss_dir)
    temp = str(save_loss_dir)+"/breast-"+structure+"-fold" + str(i+1) + "-" + database + "-" + model_name + "-" + str(image_size) + "-" + str(batch_size) + "-"+str(lr)+ ".csv"
    d_loss.to_csv(temp, index=False)
    del model
    K.clear_session()

print('Started PreImageNet!')
for i in range(5): #5 = fivefold cross-validation
    train_path = os.path.join(current_directory,"radimagenet/Workspace/Replicate_article/data/breast_splits/train_fold"+str(i+1)+".csv")
    val_path = os.path.join(current_directory,"radimagenet/Workspace/Replicate_article/data/breast_splits/val_fold"+str(i+1)+".csv")
    df_train=pd.read_csv(train_path)
    df_val=pd.read_csv(val_path)

    train_data_generator = ImageDataGenerator(
                                    rescale=1./255,
                                    preprocessing_function=preprocess_input,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    data_generator = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
    train_generator = train_data_generator.flow_from_dataframe(
        dataframe=df_train,
        directory=current_directory,
        x_col = 'path',
        y_col = 'class',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='categorical')
    validation_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        directory=current_directory,
        x_col = 'path',
        y_col = 'class',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='categorical')   
    num_classes =len(train_generator.class_indices)
    print('lr set to',lr)
    run_model(i)
#Might take up to 7-8h
