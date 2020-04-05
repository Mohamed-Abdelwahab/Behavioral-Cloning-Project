import os
import csv

samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i,line in enumerate(reader):
        samples.append(line)
        
samples = samples[1:]
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import matplotlib.image as mpimg
import numpy as np
import sklearn
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Lambda, Cropping2D
from keras.layers.core import Flatten, Dense, Dropout, SpatialDropout2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def generator(samples, batch_size=32):
    num_samples = len(samples)
    
   
    while 1: 
        sklearn.utils.shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3): #we are taking 3 images, first one is center, second is left and third is right
                        
                        name = '../../opt/carnd_p3/data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = mpimg.imread(name) # reads an image in RGB since in drive.py it is RGB
                        center_angle = float(batch_sample[3]) #getting the steering angle measurement
                        images.append(center_image)
                        
                        # introducing correction for left and right images
                        # if image is in left we increase the steering angle by 0.2
                        # if image is in right we decrease the steering angle by 0.2
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)
                        
                        # Code for Augmentation of data.
                        # We take the image and just flip it and negate the measurement
                        
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
                        #here we got 6 images from one image    
                        
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
'''def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../../opt/carnd_p3/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image =  mpimg.imread(name)
                #print(batch_sample[3])
                center_angle =float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)'''

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()

# trim image to only see section with road
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#Nvidia model
model.add(Conv2D(24, (5, 5), activation="relu", name="conv_1", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", name="conv_2", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", name="conv_3", strides=(2, 2)))
model.add(SpatialDropout2D(.5, dim_ordering='default'))

model.add(Conv2D(64, (3, 3), activation="relu", name="conv_4", strides=(1, 1)))
model.add(Conv2D(64, (3, 3), activation="relu", name="conv_5", strides=(1, 1)))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)  

#saving model
model.save('model.h5')

# Done
print('Done.')


