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
from Filters import saveImg_unique,read_rgb

def Histogram_Computation(image, r=256):

    image_Height = image.shape[0]
    image_Width = image.shape[1]

    Histogram = np.zeros([r], np.int32)

    for x in range(0, image_Height):
        for y in range(0, image_Width):
            Histogram[int(image[x, y])] += 1

    return Histogram


def rgb2gray(rgb_image):
    gary_img= np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
    # gary_img=np.array(gary_img)
    return gary_img

def equalizeImage(gray_image=None):
    onCurrent = False
    if gray_image is None:
        gray_image = cv2.imread('./static/img/input/current.png',0)  #0 is to read the as gray scale
        # img=rgb2gray(img)
        onCurrent = True
    
    img = gray_image.copy()
   
    
    x,y=gray_image.shape

    # Calc histogram
    histogram, bins = np.histogram(img.ravel(), bins=256, range=(0, 255))

    # Probability distribution
    pdf = histogram/histogram.sum()

    # Commulative
    cdf = pdf.cumsum()

    # commulative * max gray level
    normed_cdf = np.round(cdf * 255).astype(np.int64)
    equalized_img = np.zeros((x, y))

    for i in range(0, x):
        for j in range(0, y):
            r = img[i, j]
            s = normed_cdf[r]
            equalized_img[i, j] = s
    if onCurrent:
        return saveImg_unique(equalized_img,"./static/img/output/",save_on_current = False)
    else:
        return equalized_img

def normalizeImage(gray_image=None):
    onCurrent = False
    if gray_image is None:
        gray_image = cv2.imread('./static/img/input/current.png',0)  #0 is to read the as gray scale
        onCurrent = True
    # img=rgb2gray(img)
    
    lmin = float(gray_image.min())
    lmax = float(gray_image.max())
    normalized_img = np.floor((gray_image-lmin)/(lmax-lmin)*255.0)
 
    if onCurrent:
        return saveImg_unique(normalized_img,"./static/img/output/",save_on_current = False)
    else:
        return normalized_img

def histogram():
    gray_image = cv2.imread('./static/img/input/current.png',0)
    plt.subplot(2, 2, 1)
    plt.tight_layout(pad=2.5)
    plt.title('Original Histogram')
    hist, bins = np.histogram(gray_image.ravel(), bins=256, range=(0, 255))
    intensity = np.arange(0, 256, 1)
    hist1 = Histogram_Computation(gray_image)

    plt.bar(intensity, hist1)


    plt.subplot(2, 2, 3)
    plt.title('Equalized Histogram')
    hist2 = Histogram_Computation(equalizeImage(gray_image=gray_image))
    plt.bar(intensity, hist2)

    normImg =  normalizeImage(gray_image)
    maxx = normImg.max()
    minn = normImg.min()
    plt.subplot(2, 2, 4)
    plt.title('Normalize Histogram')
    hist3 = Histogram_Computation(normImg)
    intensity = np.arange(-128,128,1)
    plt.bar(intensity, hist3)

    path = "./static/img/output/" + str(randint(1,100000))+".png"
    plt.savefig(path)
    plt.clf()
    plt.clf()
    plt.clf()
    return path


def rgb_histogram():
    rgb_img=cv2.imread("./static/img/input/current.png")
    img = cv2.imread('./static/img/input/current.png',0)

    ## BGR #################################################################
    b = rgb_img[:, :, 0]
    g = rgb_img[:, :, 1]
    r = rgb_img[:, :, 2]

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.subplot(1, 3, 1)
    plt.title('Blue Histogram')
    intensity = np.arange(0, 256, 1)
    Histogram_b = Histogram_Computation(b)
    plt.bar(intensity, Histogram_b)


    plt.subplot(1, 3, 2)
    plt.title('Green Histogram')
    Histogram_g = Histogram_Computation(g)
    plt.bar(intensity, Histogram_g, color = "green")


    plt.subplot(1, 3, 3)
    plt.title('Red Histogram')
    Histogram_r = Histogram_Computation(r)
    plt.bar(intensity, Histogram_r, color = "red")
    path1 = "./static/img/output/" + str(randint(1,100000))+".png"
    plt.savefig(path1)
    plt.clf()
    plt.clf()
    plt.clf()
    ### CDF Curve #################################################################
    # Calc histogram
    histogram, bins = np.histogram(img.ravel(), bins=256, range=(0, 255))
    # Probability distribution
    pdf = histogram/histogram.sum()
    # Commulative
    cdf = pdf.cumsum()
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.hist(img.flatten(),
            bins=256,
            range=[0, 256],
            color='r')
    ax.set_xlabel('pixel intensity')
    ax.set_ylabel('#pixels')
    ax.set_xlim(0, 255)


    ax2 = ax.twinx()
    ax2.plot(cdf, color='b')
    ax2.set_ylabel('cdf')
    ax2.set_ylim(0, 1)
    path2 = "./static/img/output/" + str(randint(1,100000))+".png"
    plt.savefig(path2)
    plt.clf() 
    return [path1,path2]



def globalThreshold(img,thr=125):
  x,y=img.shape
  new_image = np.zeros((x,y))
  for i in range(0,x):
    for j in range(0,y):
      if (img[i][j] < thr):
        new_image[i][j] = 0
      else:
        new_image[i][j] = 255
  path = saveImg_unique(new_image, "./static/img/output/", rgb=False, save_on_current = False)
  return path

def localThreshold(img,size=20):
  x,y= img.shape
  new_image = np.zeros((x,y))
  for i in range(0,x):
            for j in range(0,y):
              thr=np.mean(img[i:size+i, j:size+j])
              if (img[i][j] < thr):
                new_image[i][j] = 0
              else:
                new_image[i][j] = 255
  path = saveImg_unique(new_image, "./static/img/output/", rgb=False, save_on_current = False)
  return path

