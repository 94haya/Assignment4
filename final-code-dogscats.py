from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D,Dense,Flatten,Dropout
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,load_img
from IPython.display import display
from PIL import Image
from keras.models import load_model 
import numpy as np
from keras.preprocessing import image


classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Conv2D(32,(3,3))) 
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Conv2D(32,(3,3))) 
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))
classifier.summary()
classifier.compile(optimizer ='rmsprop',
                   loss ='binary_crossentropy',
                   metrics =['accuracy'])
train_datagen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/9942038100/Desktop/project/train',
                                                target_size=(64,64),
                                                batch_size= 32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('C:/Users/9942038100/Desktop/project/test',
                                           target_size = (64,64),
                                           batch_size = 32,
                                           class_mode ='binary')
classifier.fit_generator(training_set,
                        steps_per_epoch =625,
                        epochs = 30,
                        validation_data =test_set,
                        validation_steps = 5000)
classifier.save('catdog_cnn_model.h5')  
classifier = load_model('catdog_cnn_model.h5')

test_image =image.load_img('C:/Users/9942038100/Desktop/image.jpeg',target_size =(64,64))
test_image =image.img_to_array(test_image)
test_image =np.expand_dims(test_image, axis =0)
result = classifier.predict(test_image)
if result[0][0] >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
