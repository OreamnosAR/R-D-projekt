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
lr = 0.00001 #ITU = fixed 1e-5  
model_name = "Eks2_ITU_ResNet50_ImgNet"
current_directory = os.environ['HOME']

def get_compiled_model():
    #ImageNet
    base_model = ResNet50(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    y = base_model.output
    y = Dropout(0.5)(y)
    predictions = Dense(2, activation='softmax')(y)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss=BinaryCrossentropy(), metrics=[keras.metrics.AUC(name='auc')])
    return model

def run_model():
    model = get_compiled_model()
    ### Set train steps and validation steps
    train_steps = len(train_generator.labels)/batch_size
    val_steps = len(validation_generator.labels)/batch_size
    
    #### set the path to save models having lowest validation loss during training
    save_model_dir = os.path.join(current_directory,"radimagenet/Anders/Eksperiment2/ITU/freezing_round1/pretrained_ImageNet/weights/") 
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)    
    filepath = os.path.join(current_directory,"radimagenet/Anders/Eksperiment2/ITU/freezing_round1/pretrained_ImageNet/weights/" + model_name + ".h5") 
    
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
    save_loss_dir = os.path.join(current_directory,"radimagenet/Anders/Eksperiment2/ITU/freezing_round1/pretrained_ImageNet/training_history")

    if not os.path.exists(save_loss_dir):
        os.mkdir(save_loss_dir)
    temp = str(save_loss_dir)+"/" + model_name + ".csv"
    d_loss.to_csv(temp, index=False)
    del model
    K.clear_session()

print('Started PreImageNet!')
for i in range(1):
    train_path = os.path.join(current_directory,"radimagenet/Anders/Eksperiment2/Data/Datasplits/train_pocData.csv")
    val_path = os.path.join(current_directory,"radimagenet/Anders/Eksperiment2/Data/Datasplits/val_pocData.csv")
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
    data_generator = ImageDataGenerator(rescale=1./255,
                                        preprocessing_function=preprocess_input)
    train_generator = train_data_generator.flow_from_dataframe(
        dataframe=df_train,
        directory=current_directory,
        x_col = 'filepath',
        y_col = 'label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='categorical')
    validation_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        directory=current_directory,
        x_col = 'filepath',
        y_col = 'label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode='categorical')   
    num_classes =len(train_generator.class_indices)
    print('lr set to',lr)
    run_model()