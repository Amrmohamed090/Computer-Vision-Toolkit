import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
import base64
from PIL import Image
import cv2
from io import StringIO 
import base64
from random import randint
from scipy import ndimage

def read_rgb(gray=False, path = None):
    if path is None:
        img = cv2.imread('./static/img/input/current.png') #0 is to read the as gray scale
    else:
        img = cv2.imread(path) #0 is to read the as gray scale
    if gray:
        img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img=rgb2gray(img)
    
    # normalize the value of each pixel to be between 0 &1
    img=img/225
    img = np.array(img)
    
    return img

def saveImg_unique(img,path, rgb=True, save_on_current = True):
    path_img = f"{path}{randint(1,100000)}.png"
    with open(path_img, 'wb') as f:
        
        cv2.imwrite(path_img, img)
    if save_on_current:
        with open('./static/img/input/current.png', 'wb') as f:
            
            cv2.imwrite('./static/img/input/current.png', img)
        
    return path_img


def window(D0,x,y):  #D0 is the cutoff frequncy 
    h=np.zeros((x,y),dtype=np.float64())
    for i in range(x):
        for j in range(y):
            D =np.sqrt((i-x/2)**2 + (j-y/2)**2)   #D is the radius from the center
            if D <= D0:
                h[i,j]=1
            else:
                h[i,j]=0
    return h

def Freq_filter(D0,type,path=None):
    if path is None:
        img=read_rgb(gray=True)
    else:
        img=read_rgb(gray=True, path=path)
    imgFr=np.fft.fft2(img)
    imgFr=np.fft.fftshift(imgFr)
    x,y =img.shape
    if type =="low":
        win=window(D0,x,y)
    elif type =="high":
        win=1-window(D0,x,y)
    fliter_img= imgFr*win
    fliter_img=np.fft.fftshift(fliter_img)
    fliter_img=np.abs(np.fft.ifft2(fliter_img))
    fliter_img = fliter_img*225
    if path is None:
        path = saveImg_unique(fliter_img, "./static/img/output/", rgb=False, save_on_current = False)
        return path
    else:
        return fliter_img
# operators of different filter

def hybridFd(img1_path,img2_path,t = 1,low=20,high=10):
    if t== 1:
        img1=Freq_filter(10,'high',path=img1_path)
        img2=Freq_filter(20,'low',path=img2_path)
    else:
        img1=Freq_filter(10,'low',path=img1_path)
        img2=Freq_filter(20,'high',path=img2_path)
        
    result=img1+img2
    path = saveImg_unique(result, "./static/img/output/", rgb=False, save_on_current = False)
    return path