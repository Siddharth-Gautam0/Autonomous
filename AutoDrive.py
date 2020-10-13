import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

## Keras
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import cv2
import pandas as pd
import random
import ntpath

## Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

datadir = 'Your path here'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)

def path_tail(path):
    head, tail = ntpath.split(path)
    return tail

## Remove path of images
data['center'] = data['center'].apply(path_tail)
data['left'] = data['left'].apply(path_tail)
data['right'] = data['right'].apply(path_tail)

def ImageData(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
      indexed_data = data.iloc[i]
      center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
      image_path.append(os.path.join(datadir, center.strip()))
      steering.append(float(indexed_data[3]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = ImageData(datadir + '/IMG', data)
X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=0)


#Important for Tuning
def PreprocessImage(img):
  img = npimg.imread(img)

  ## Crop image to remove unnecessary features
  img = img[60:-25, :, :]

  ## Change to YUV image
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

  ## Gaussian blur
  img = cv2.GaussianBlur(img, (3, 3), 0)

  ## Decrease size for easier processing
  img = cv2.resize(img, (100, 100))

  ## Normalize values
  img = img / 127.5-1.0
  return img

## Get any image
image = image_paths[100]
original_image = npimg.imread(image)
preprocessed_image = PreprocessImage(image)

## Compare original and preprocessed image
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[1].imshow(preprocessed_image)
axes[1].set_title('Preprocessed Image')

X_train = np.array(list(map(PreprocessImage, X_train)))
X_valid = np.array(list(map(PreprocessImage, X_valid)))

plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis('off')
print(X_train.shape)

def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    return model
model = nvidia_model()

history = model.fit(X_train, Y_train, epochs=3, validation_data=(X_valid, Y_valid), batch_size=32, verbose=1, shuffle=0)

model.save('Your path here .h5')
