import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#Imgae into pandas Dataframe
filenames= os.listdir("./dogs-vs-cats/train")
categories=[]
for filename in filenames:
    
    category=filename.split(".")[0]
    if category=="dog":
        categories.append(1)
    else:   
       categories.append(0)
        
        
df=pd.DataFrame({'filename':filenames,
                  'category':categories
                })  
print(df.head())
print(df.shape)
print(df['category'].value_counts())
print(df['category'].value_counts().plot(kind='bar'))

#Using Convolution Neural Network
model1 = Sequential()

model1.add(Conv2D(128, (3, 3), activation='relu', input_shape=(128,128,3)))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
model1.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

df['category']=df["category"].replace({0:'cat',1:'dog'})
train_df,validate_df=train_test_split(df,test_size=.20,random_state=4)
train_df=train_df.reset_index(drop=True)
validate_df=validate_df.reset_index(drop=True)
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=16
print(total_validate)

#Augmentation and prepare images for our model1
train_datagen=ImageDataGenerator(
    rotation_range=30,
    rescale=1./255,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    width_shift_range=0.4,
    height_shift_range=0.4
)

train_generator=train_datagen.flow_from_dataframe(
    
    train_df,
    "./dogs-vs-cats/train",
    x_col='filename',
    y_col='category',
    target_size=(128,128),
    class_mode='categorical',
    batch_size=batch_size
    
)
validate_datagen=ImageDataGenerator(
    rescale=1./255
)

validate_generator=validate_datagen.flow_from_dataframe(
    validate_df,
    "./dogs-vs-cats/train",
    x_col='filename',
    y_col='category',
    target_size=(128,128),
    class_mode='categorical',
    batch_size=batch_size
    
)
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

#Train and lets see the accuracy rate our model1
history1=model1.fit_generator(
    train_generator,
    steps_per_epoch=total_train//batch_size, 
    epochs=15, 
    verbose=1,
    callbacks=callbacks, 
    validation_data=validate_generator,
    validation_steps=total_validate//batch_size,
        
)
