import cv2
import numpy as np
from random import randint
from FeatureDetection import SIFT_2
import time
import matplotlib.pyplot as plt


########################### SIFT ####################################


def read_rgb(gray=False,normalize = True):
    img = cv2.imread('./static/img/input/current.png') #0 is to read the as gray scale
    if gray:
        img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img=rgb2gray(img)
    
    # normalize the value of each pixel to be between 0 &1
    if normalize:
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





  
def matching(descriptor1 , descriptor2 , match_calculator):
    
    keypoints1 = descriptor1.shape[0]
    keypoints2 = descriptor2.shape[0]
    matches = []

    for kp1 in range(keypoints1):

        distance = -np.inf
        y_index = -1
        for kp2 in range(keypoints2):

         
            value = match_calculator(descriptor1[kp1], descriptor2[kp2])

            if value > distance:
              distance = value
              y_index = kp2
        
        match = cv2.DMatch()
        match.queryIdx = kp1
        match.trainIdx = y_index
        match.distance = distance
        matches.append(match)
    matches= sorted(matches, key=lambda x: x.distance, reverse=True)
    return matches
 



def calculate_ssd(descriptor1 , descriptor2):

    ssd = 0
    for m in range(len(descriptor1)):
        ssd += (descriptor1[m] - descriptor2[m]) ** 2

    ssd = - (np.sqrt(ssd))
    return ssd






def main_ssd(path1,path2):
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2 ,0)
    img1_rgb = cv2.imread(path1)
    img2_rgb = cv2.imread(path2)

    
    # generate keypoints and descriptor using sift
    sift = cv2.SIFT_create()
    keypoints_1, descriptor1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptor2 = sift.detectAndCompute(img2,None)




    start_ssd = time.time()
    matches_ssd = matching(descriptor1, descriptor2, calculate_ssd)
    matched_image_ssd = cv2.drawMatches(img1_rgb, keypoints_1, img2_rgb, keypoints_2,
                                    matches_ssd[:30], img2_rgb, flags=2)
    end_ssd = time.time()
    ssd_time = end_ssd - start_ssd
    
    path = saveImg_unique(matched_image_ssd, "./static/img/output/", rgb=False, save_on_current = False)
    return path , ssd_time




def calculate_ncc(descriptor1 , descriptor2):


    out1_normalized = (descriptor1 - np.mean(descriptor1)) / (np.std(descriptor1))
    out2_normalized = (descriptor2 - np.mean(descriptor2)) / (np.std(descriptor2))

    correlation_vector = np.multiply(out1_normalized, out2_normalized)

    correlation = float(np.mean(correlation_vector))

    return correlation



def main_ncc(path1,path2):
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2 ,0)
    img1_rgb = cv2.imread(path1)
    img2_rgb = cv2.imread(path2)

    # generate keypoints and descriptor using sift
    sift = cv2.SIFT_create()
    keypoints_1, descriptor1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptor2 = sift.detectAndCompute(img2,None)

    start_ncc = time.time()
    matches_ncc = matching(descriptor1, descriptor2, calculate_ncc)
    matched_image_ncc = cv2.drawMatches(img1_rgb, keypoints_1, img2_rgb, keypoints_2,matches_ncc[:30], img2_rgb, flags=2)
    end_ncc = time.time()

    ncc_time =  end_ncc - start_ncc

    path = saveImg_unique(matched_image_ncc, "./static/img/output/", rgb=False, save_on_current = False)
    return path ,ncc_time
  

