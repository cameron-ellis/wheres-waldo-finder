# first import libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os

# tensorflow
import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model

#import time
import time
#get the images directory
image_dir = os.getcwd()

#get the background and waldo image directory
background_dir = image_dir + '/background.png'
waldo_dir = image_dir + '/waldo.png'
wilma_dir = image_dir + '/wilma.png'

#background image
background_im = Image.open(background_dir)
background_im