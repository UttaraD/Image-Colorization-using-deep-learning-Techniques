from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, UpSampling2D, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape
from tensorflow.keras import layers
from keras.models import Model
from keras.losses import binary_crossentropy
import glob
from numpy import asarray
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from os import listdir
from tensorflow.keras.models import Model, load_model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
from skimage.io import imsave
from skimage.transform import resize
from skimage.io import imshow
from PIL import Image
import cv2

st.set_page_config(layout="wide")

st.header(""" Image Colorization """)

st.subheader("Upload an image here")

model = tf.keras.models.load_model('colorize_autoencoder.h5',custom_objects=None,compile=True)
image_file=st.file_uploader('File uploader')
if image_file is not None:
    image=Image.open(image_file)

    st.image(
    image,caption=f"This is the image",
    use_column_width=False
    )
    img1_color = []
    img1 = np.array(image)
    img1 = resize(img1, (112, 112))
    img1_color.append(img1)
    img1_color= np.array(img1_color,dtype=float)
    img1_color =rgb2lab(1.0/255*img1_color)[:,:,:,0]
    img1_color =img1_color.reshape(img1_color.shape+(1,))
    #st.write(img1_color)
    output1=model.predict(img1_color)
    #st.write(output1)
    # output1=output1*128


    result=np.zeros((112,112,3))
    result[:,:,0]=img1_color[0][:,:,0]
    result[:,:,1:]=output1[0]
    color_img=lab2rgb(result)
    imshow(lab2rgb(result))
    st.image(color_img*128,width= 200,clamp = True)

