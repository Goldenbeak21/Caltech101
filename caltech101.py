import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
import scipy
import os
import imageio
import image 
import cv2

def imread(path):
    img = imageio.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img
    
path = '/content/drive/My Drive/101_ObjectCategories'
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
imgs = []
labels = []
# LOAD ALL IMAGES 
for i, category in enumerate(categories):
    print(i)
    for f in os.listdir(path + "/" + category):
        print(f)
        ext = os.path.splitext(f)
        if ext[1] not in valid_exts:
            continue
        fullpath = os.path.join(path + "/" + category, f)
        img = cv2.imread(fullpath)
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img[:,:,0] -= 123.68
        img[:,:,1] -= 116.78
        img[:,:,2] -= 103.94
        imgs.append(img) # NORMALIZE IMAGE 
        label_curr = i
        labels.append(label_curr)
print ("Num imgs: %d" % (len(imgs)))
print ("Num labels: %d" % (len(labels)) )
print (ncategories)


import cv2
img = cv2.imread(fullpath)
plt.imshow(img)
img.shape
img1 = cv2.resize(img, (32,32))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1)
img1.shape


import np_utils
from np_utils import *
from tensorflow.keras.utils import *
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imgsf,labels1,test_size=0.20 ,random_state=0)
X_train = np.array(X_train)
X_test = np.array(X_test)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False) 

datagen.fit(X_train)



-----ALEXNET----

from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense



classifier = Sequential()

classifier.add(Conv2D(64,(3,3), input_shape=(32,32,3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2))

#adding a second convolution layer to improve the performance (No of features has increased as the size of the image reduces)
classifier.add(Conv2D(128,(3,3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2,data_format='channels_last'))

classifier.add(Flatten())

#creating the ANN (part of the CNN) where the pixels are the input  
classifier.add(Dense(256, activation='relu'))

#using softmax as the ouptut has more than one nodes, so sigmoid won't give values that add up to one
classifier.add(Dense(101, activation='softmax'))


# Optimizer - what method are you usung to optimize the loss (gradient descent etc.)
# loss - what function are you considering to optimize (mean squared error, absolute difference)
# loss is like a metric but the loss values are considered while back propagating where as the 
# metrics values are not, metrics is used for other purposes such as call backs
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#history = classifier.fit_generator(datagen.flow(X_train, y_train, batch_size=32), epochs=25)
hist = classifier.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=32),
                    epochs=20,
                    validation_data=(X_test, y_test))

-----LENET----
classifier = Sequential()

classifier.add(Conv2D(32,3, input_shape=(32,32,3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2))

classifier.add(Conv2D(64,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2))

#adding a second convolution layer to improve the performance (No of features has increased as the size of the image reduces)
classifier.add(Conv2D(128,3, padding='same', activation='relu'))
classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(Conv2D(512,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2,data_format='channels_last'))

classifier.add(Flatten())

#creating the ANN (part of the CNN) where the pixels are the input  
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(128, activation='relu'))

#using softmax as the ouptut has more than one nodes, so sigmoid won't give values that add up to one
classifier.add(Dense(101, activation='softmax'))

classifier.summary()

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = classifier.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=32),
                    epochs=20,
                    validation_data=(X_test, y_test))

----VGG----
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


classifier = Sequential()

classifier.add(Conv2D(64,3, input_shape=(224,224,3), padding='same', activation='relu'))
classifier.add(Conv2D(64,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2))

classifier.add(Conv2D(128,3, padding='same', activation='relu'))
classifier.add(Conv2D(128,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2))


classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2,data_format='channels_last'))

classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2,data_format='channels_last'))

classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(Conv2D(256,3, padding='same', activation='relu'))
classifier.add(MaxPooling2D(2,2,data_format='channels_last'))

classifier.add(Flatten())

#creating the ANN (part of the CNN) where the pixels are the input  
classifier.add(Dense(4096, activation='relu'))
classifier.add(Dense(2048, activation='relu'))

#using softmax as the ouptut has more than one nodes, so sigmoid won't give values that add up to one
classifier.add(Dense(101, activation='softmax'))

classifier.summary()

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = classifier.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=32),
                    epochs=20,
                    validation_data=(X_test, y_test))
