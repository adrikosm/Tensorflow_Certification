# -*- coding: utf-8 -*-
"""transfer_learning_tuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d61i-OwmRz5LMJMVU1uVX_yBl0WKYA4i

## Imports
"""

import numpy as np
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import pathlib
import zipfile
import os
import random
import datetime

"""## Import some of our helper functions"""

!wget https://github.com/adrikosm/Tensorflow_Certification/blob/main/Helper_Functions.py