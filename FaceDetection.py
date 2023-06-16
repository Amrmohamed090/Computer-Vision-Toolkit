from Filters import saveImg_unique, read_rgb
from io import StringIO
from scipy import ndimage
from random import randint
from PIL import Image
import base64
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('Agg')

import sys
from scipy.misc import face
import os
from sklearn import preprocessing

def detect_faces():
    image=read_rgb(gray=True, normalize=False)
    # Load the pre-trained classifiers for face
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale3(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels = True
        )
    print(faces)
    rects = faces[0]
    neighbours = faces[1]
    weights = faces[2]
    return rects

def draw_faces():
    image=read_rgb(gray=False, normalize=False)
    faces=detect_faces()
     # Draw a rectangle around the faces
    for (x, y, w, h) in faces: 
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0 , 0), 4)
    result_img=np.copy(image)
    path = saveImg_unique(result_img, "./static/img/output/",rgb=True, save_on_current=True)
    return path