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


def optimal_Thres(image=None,output_img=False):
    if image is None:
      image = read_rgb(gray=True, normalize=False)
    x, y = image.shape
    background = np.zeros((x, y), dtype=np.int32)
    obj = np.ones((x, y), dtype=np.int32)

    background[0, 0] = 1
    background[0, y-1] = 1
    background[x-1, 0] = 1
    background[x-1, y-1] = 1
    corners = np.where(obj > 0)
    obj[corners] = 0
    T_old = 0

    while (1):
        mean_background = np.mean(image[background == 1])
        mean_obj = np.mean(image[obj == 0])
        T = (mean_background+mean_obj)/2
        T = round(T, 2)
        if (T == T_old):
            break
        else:
            background[image < T] = 1
            obj[image < T] = 0
            T_old = T
    thr_img = np.copy(image)
    thr_img[background > 0] = 0
    thr_img[obj > 0] = 1
    path = saveImg_unique(thr_img, "./static/img/output/",
                          rgb=False, save_on_current=False)
    if(output_img):
      return thr_img
    else:
      return path



def localized(n=4,thersholdMethod='opt'):
  img=read_rgb(gray=True, normalize=False)
  image_array=np.copy(img)
    # Get the dimensions of the image
  height, width = image_array.shape
  
  # Calculate the size of each sub-image
  sub_height = height // n   #flooring
  sub_width = width // n
  
  # Create an empty array to hold the recombined image
  recombined_image = np.zeros((height, width), dtype=np.uint8)
  
  # Loop through each sub-image
  for i in range(n):
      for j in range(n):
          # Calculate the start and end indices for this sub-image
          start_i = i * sub_height
          end_i = (i + 1) * sub_height
          start_j = j * sub_width
          end_j = (j + 1) * sub_width
          if(thersholdMethod=='opt'):
            ther_image=optimal_Thres(image_array[start_i:end_i, start_j:end_j],True)
          elif(thersholdMethod=='otsu'):
            # ther_image=otsu_thres(image_array[start_i:end_i, start_j:end_j],True)
            ther_image=otsuThreshold(image_array[start_i:end_i, start_j:end_j],True)
          # Copy this sub-image into the recombined image
          recombined_image[start_i:end_i, start_j:end_j] = ther_image
  path = saveImg_unique(recombined_image, "./static/img/output/",
                        rgb=False, save_on_current=False)
  return path



def otsuThreshold_value():
    image = read_rgb(gray=True, normalize=False)
    #Get size of image
    rows, cols =  image.shape
    #Plotting image histogram
    #We are interested on H (histogram), other values that plt.hist returns will be ignored here
    H, binEdges = np.histogram(image.ravel(), 256)
    # Getting relative histogram (pdf)
    pdf = H /(rows*cols)
    # Get cdf for all gray levels
    cdf = np.cumsum(pdf)
    #Initialization
    othresh = 1
    maxVarB = 0
    for t in range(1,255):
      #gray levels belongs to background 
      bg = np.arange(0,t)
      #object gray levels
      obj = np.arange(t, 256)
      #Calculation of mean gray level for object and background
      mBg    = sum(bg*pdf[0:t])/cdf[t]
      mObj   = sum(obj*pdf[t:256])/(1-cdf[t])
      # Calculate between class variance
      varB = cdf[t] * (1-cdf[t]) *(mObj - mBg)**2
      #Pick up max variance and corresponding threshold
      if varB > maxVarB:
          maxVarB= varB
          othresh = t
    return othresh

def otsuThreshold(gray_image=None ,output_img=False):
  if gray_image is None:
    gray_image = read_rgb(gray=True, normalize=False)
  threshold=otsuThreshold_value()
  final_img=255* ( gray_image > threshold ) 
  path = saveImg_unique(final_img, "./static/img/output/",
                        rgb=False, save_on_current=False)
  if(output_img):
    return final_img
  else:
    return path

def otsuThreshold_value_spectral(num_modes):
    image = read_rgb(gray=True, normalize=False)
    #Get size of image
    rows, cols =  image.shape
    #Plotting image histogram
    #We are interested on H (histogram), other values that plt.hist returns will be ignored here
    H, binEdges = np.histogram(image.ravel(), 256)
    # Getting relative histogram (pdf)
    pdf = H /(rows*cols)
    # Get cdf for all gray levels
    cdf = np.cumsum(pdf)
    #Initialization
    othresh = np.zeros(num_modes)
    maxVarB = np.zeros(num_modes)
    for i in range(num_modes):
        if i == 0:
            t_min = 1
            t_max = 255
        else:
            t_min = othresh[i-1] + 1
            t_max = 255
        t_min=int(t_min)
        for t in range(t_min, t_max):
            #gray levels belongs to background 
            bg = np.arange(0,t)
            #object gray levels
            obj = np.arange(t, 256)
            #Calculation of mean gray level for object and background
            mBg    = sum(bg*pdf[0:t])/cdf[t]
            mObj   = sum(obj*pdf[t:256])/(1-cdf[t])
            # Calculate between class variance
            varB = cdf[t] * (1-cdf[t]) *(mObj - mBg)**2
            #Pick up max variance and corresponding threshold
            if varB > maxVarB[i]:
                maxVarB[i] = varB
                othresh[i] = t
    return othresh

def segment_image_otsu(n):
  image = read_rgb(gray=True, normalize=False)
  thresholds=otsuThreshold_value_spectral(n)
  num_levels=n
  num_modes = len(thresholds) + 1
  rows, cols = image.shape
  output = np.zeros((rows, cols, num_modes), dtype=np.uint8)
  for i in range(num_modes):
      if i == 0:
          output[:,:,i] = (image < thresholds[i]).astype(np.uint8) * int(num_levels/(num_modes-1)*i)
      elif i == num_modes - 1:
          output[:,:,i] = (image >= thresholds[i-1]).astype(np.uint8) * int(num_levels/(num_modes-1)*i)
      else:
          output[:,:,i] = ((image >= thresholds[i-1]) & (image < thresholds[i])).astype(np.uint8) * int(num_levels/(num_modes-1)*i)
  segmented_image = np.sum(output, axis=-1)
  segmented_image=segmented_image.astype(np.uint8)
  segmented_image=segmented_image*100
  path = saveImg_unique(segmented_image, "./static/img/output/",
                        rgb=False, save_on_current=False)
  return path

def apply_spectral_kmean(threshold=220, num_clusters=4):
  gray = read_rgb(gray=True, normalize=False)
  # Compute the Fourier transform of the image
  f = np.fft.fft2(gray)
  # Shift the zero-frequency component to the center of the spectrum
  fshift = np.fft.fftshift(f)
  # Compute the magnitude spectrum
  magnitude_spectrum = 20 * np.log(np.abs(fshift))
  # Apply the spectral threshold
  spectral_mask = np.ones_like(magnitude_spectrum)
  spectral_mask[magnitude_spectrum < threshold] = 0
  # Apply the spectral mask to the shifted spectrum
  filtered_spectrum = fshift * spectral_mask
  # Shift the filtered spectrum back to the original position
  filtered_spectrum_shifted = np.fft.ifftshift(filtered_spectrum)
  # Compute the inverse Fourier transform to obtain the filtered image
  filtered_image = np.abs(np.fft.ifft2(filtered_spectrum_shifted))
  path = saveImg_unique(filtered_image, "./static/img/output/",
                        rgb=False, save_on_current=False)
  return path