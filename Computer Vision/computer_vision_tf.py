# -*- coding: utf-8 -*-
"""Computer_vision_tf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-Uc9J1ZcZGISVBjMWBDXvac_f9psVdUg

# Computer vision Introduction using CNN 
# (Convolutional Neural Network)
* CNN should have non linear activations
* tf.keras.layers.ConvXD where X:
  - 1 = 1D text based
  - 2 = 2D Images
  - 3 = 3D Videos

## Imports
"""

import pathlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import zipfile
import os

"""## Lets now get the data
We will only be using the pizza and steak part of the food101 dataset
"""

# Download zip file of pizza_steak images
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

# Time to unzip the file
zip_ref = zipfile.ZipFile("pizza_steak.zip","r")
zip_ref.extractall()
zip_ref.close()

# View files
!ls pizza_steak

!ls pizza_steak/train/

"""## Visualizing the Data"""

# Lets traverse through pizza steak directory and list number of files
for dirpath,dirnames,filenames in os.walk("pizza_steak"):
  #print(f" There are {len(dirnames)} Directories and {len(filenames)} images in {dirpath}.")
  if len(dirnames) < 1:
    print(f"There are {len(filenames)} images in {dirpath}.")
  elif  len(filenames) <= 1:
    print(f"There are {len(dirnames)} Directories in {dirpath}.")

# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))

num_steak_images_train

num_pizza_images_test = len(os.listdir("pizza_steak/test/pizza"))
num_pizza_images_test

# Lets get the classnames 
data_dir = pathlib.Path("pizza_steak/train")
class_names = np.array(sorted(item.name for item in data_dir.glob("*")))

class_names = class_names[1:]
class_names

"""### Visualize our images"""

def view_random_image(target_dir,target_class):
  """
  INFO:Gets a random image from a selected directory and class
  """
  target_folder = target_dir + "/" +  target_class
  random_image = random.sample(os.listdir(target_folder),1)
  # Plot out the image
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")
  print(f" Image shape: {img.shape}") # Shows the shape of the image

  return img

# View a random image
img = view_random_image(target_dir = "pizza_steak/train",
                        target_class = "pizza")

"""## Lets try to slowly build a CNN (convoluted neural network)"""

# Set up random seed
tf.random.set_seed(42)

# Preprocces data (get all images between 0 and 1 , callded normalization or scaling )
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Setup paths to our data directories
train_dir = "/content/pizza_steak/train"
test_dir = "/content/pizza_steak/test"

# Import data from directories and turn them into batches
train_data = train_datagen.flow_from_directory(directory = train_dir,
                                               target_size = (224,224),
                                               class_mode = "binary",
                                               batch_size = 32,
                                               seed = 42)


test_data = test_datagen.flow_from_directory(directory = test_dir,
                                             target_size = (224,224),
                                             class_mode = "binary",
                                             batch_size = 32,
                                             seed = 42)

# Build a CNN model same as Tiny VGG on the cnn explainer site

model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 10,
                           kernel_size = 3,
                           activation = "relu",
                           input_shape = (224,224,3)),
    tf.keras.layers.Conv2D(10, 3, activation = "relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,padding = "valid"),
    tf.keras.layers.Conv2D(10,3,activation = "relu"),
    tf.keras.layers.Conv2D(10,3,activation = "relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1 ,activation= "sigmoid")
])

# TIme to compile our model
model_1.compile(optimizer= tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ["accuracy"])

# Fit the model
history_1 = model_1.fit(train_data,
                        epochs = 5,
                        steps_per_epoch = len(train_data),
                        validation_data = test_data,
                        validation_steps =  len(test_data))

model_1.evaluate(test_data)

model_1.evaluate(train_data)

# Setup random seed
tf.random.set_seed(42)

# Create model 
model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (224,224,3)),
    tf.keras.layers.Dense(4,activation = "relu"),
    tf.keras.layers.Dense(4,activation = "relu"),
    tf.keras.layers.Dense(1,activation = "sigmoid"),

])

# COmpile the model
model_2.compile(optimizer= tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ["accuracy"])


# Fit the model
history_2 = model_2.fit(train_data,
                        epochs = 5,
                        steps_per_epoch = len(train_data),
                        validation_data = test_data,
                        validation_steps =  len(test_data))

model_2.evaluate(test_data)

model_2.evaluate(train_data)

"""### Lets try to improve model 2 by adding an extra layer and more hidden connections

"""

# Model 3

# Setup random seed
tf.random.set_seed(42)

# Build model
model_3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (224,224,3)),
    tf.keras.layers.Dense(126,activation = "relu"),
    tf.keras.layers.Dense(100,activation = "relu"),
    tf.keras.layers.Dense(64,activation = "relu"),
    tf.keras.layers.Dense(1,activation = "sigmoid")
])


# Compile the model
model_3.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

# Fit the model
history_3 = model_3.fit(train_data,
                        epochs = 5,
                        steps_per_epoch = len(train_data),
                        validation_data = test_data,
                        validation_steps = len(test_data))

model_3.evaluate(test_data)

model_3.evaluate(train_data)

"""## Lets view our model summaries"""

model_1.summary()

model_2.summary()

model_3.summary()

"""## Notice how the model 1 (CNN) only has 31k parameters while model 3 has over 18m parameters and yet it still not delivers the same performance.

# Steps we took

1. Become one with the data (visualize it)
2. Preprocess the data, prepare it for model training (scaling/normalizing and turning into batches)
3. Created a model (start with a baseline)
4. Fit the model
5. Evaluate the model
6. Adjust different hyperparameters and improve our model 
7. Repeat until satisfied

## 1.  Become one with the data
"""

# Visualize Data
plt.figure()
plt.subplot(1,2,1)
steak_img = view_random_image("pizza_steak/train","steak")
plt.subplot(1,2,2)
pizza_img = view_random_image("pizza_steak/train","pizza")

"""## 2. Preporcess the data"""

# Define directory dataset paths
train_dir = "pizza_steak/train"
test_dir = "pizza_steak/test"

# Turn data into batches using test data generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/255.)
test_datagen = ImageDataGenerator(rescale = 1/255.)

# Load in our image data from directories 
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size = (224,224),
                                               class_mode = "binary",
                                               batch_size = 32)
test_data = test_datagen.flow_from_directory(directory = test_dir,
                                             target_size = (224,224),
                                             class_mode ="binary",
                                             batch_size = 32)

# Get a sample of a train data batch
images,labels = train_data.next() # Get the net batch of images / labels in train data
len(images),len(labels)

# View the first batch of labels
labels

"""## 3. Create a CNN model

start with a baseline. A baseline is a simple model or an existing one that you set up when beginning a machine learning experiment and then as you keep experimenting you try to beat
"""

# Lets try to make the model creating a little easier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Activation
from tensorflow.keras import Sequential


# Create the baseline model

baseline_model = Sequential([
      Conv2D(filters = 10,
             kernel_size = 3,
             strides = 1,
             padding = "valid",
             activation="relu",
             input_shape = (224,224,3)), # Specify shape in input layer
      Conv2D(10,3,activation="relu"),
      Conv2D(10,3,activation="relu"),
      Flatten(),
      Dense(1,activation="sigmoid")         # Output layer
])

