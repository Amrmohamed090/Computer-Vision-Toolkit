from Filters import saveImg_unique, read_rgb
from io import StringIO
from scipy import ndimage
from random import randint
import random
from PIL import Image
import base64
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib

from numba import jit, cuda

matplotlib.use('Agg')

def RGB_to_LUV(rgb):
    # Convert RGB values to the range [0, 1]
    rgb = rgb / 255.0
    
    # Convert RGB to XYZ
    matrix = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])
    xyz = np.dot(rgb, matrix.T)
    
    # Define the reference white point and calculate un and vn
    ref_white = np.array([0.95047, 1.00000, 1.08883])
    ref_xyz = np.dot(ref_white, matrix.T)
    un = (4 * ref_xyz[0]) / (ref_xyz[0] + (15 * ref_xyz[1]) + (3 * ref_xyz[2]))
    vn = (9 * ref_xyz[1]) / (ref_xyz[0] + (15 * ref_xyz[1]) + (3 * ref_xyz[2]))
    
    # Convert XYZ to LUV
    y = xyz[..., 1]
    mask = y > 0.008856
    l = np.where(mask, 116 * np.power(y, 1/3) - 16, 903.3 * y)
    d = xyz.sum(axis=-1)
    u = np.where(mask, 13 * l * (4 * xyz[..., 0] / d - un), 0)
    v = np.where(mask, 13 * l * (9 * xyz[..., 1] / d - vn), 0)
    
    # Scale LUV values to the range [0, 255]
    luv = np.stack([2.55 * l, u + 128, v + 128], axis=-1)
    return luv.astype(np.uint8)

############## Kmean ###############

def Kmean(k=5,number_of_iterations=100,color_space="RGB"):
    image = read_rgb(gray=False, normalize=False)
    #initialize random k
    if color_space == "LUV":
        image =  RGB_to_LUV(image)
    
    return Kmean_calculation (image,k=k,number_of_iterations=number_of_iterations)


def Kmean_calculation (image,k=5,number_of_iterations=100):
    
    k_means = np.zeros((k,5))
    
        
    for i in range(len(k_means)):
        for j in range(len(k_means[i])):
            k_means[i][j] = random.uniform(50, 100)
        k_means[i][len(k_means[i])-2] = random.uniform(0, image.shape[0])
        k_means[i][len(k_means[i])-1] = random.uniform(0, image.shape[1])
    
    #the fun start
    for _ in range(number_of_iterations):
        old_k_means = k_means.copy()
        #creating clusters list
        k_lists = [] 
        '''
        [
        [[1,2,3],[4,5,6],[7,8,9]],
        [[1,2,3],[4,5,6]]
        [[1,3,4],[53,45,3]]
        ]
        '''
        
        for n, k_value in enumerate(k_means):
            k_lists.append([])
        
        
        for i,row in enumerate(image):
            for j,pixels in enumerate(row):
                min_ = 999999
                k_ = 999999
                vector = np.array(list(pixels)+[i,j])
                for n, k_values in enumerate(k_means):
                    distance = np.linalg.norm(vector-k_values)
                    if distance<min_:
                        min_ = distance
                        k_ = n
                k_lists[k_].append(vector)
        
        for n,cluster in enumerate(k_lists):
            for m,vector in enumerate(np.array(cluster).T):
                k_means[n][m] = vector.mean()
        if np.linalg.norm(k_means-old_k_means) < 1:
            break
        else:
            old_k_means = k_means.copy()
    
    for i,row in enumerate(image):
            for j,pixels in enumerate(row):
                min_ = 999999
                k_ = 999999
                vector = np.array(list(pixels)+[i,j])
                for n, k_values in enumerate(k_means):
                    distance = np.linalg.norm(vector-k_values)
                    if distance<min_:
                        min_ = distance
                        k_ = n
                image[i][j] = k_means[k_][:len(k_means[k_])-2]
    
    path = saveImg_unique((image).astype(np.uint8), "./static/img/output/",rgb=True, save_on_current=False)
    return path
############## end Kmean ###################    


############# region growing #################            
def region_growing( seed_point, threshold):

    image = read_rgb(gray=True, normalize=False)
    # Initialize segmented image with all black pixels
    segmented_image = np.zeros_like(image)

    # Get image shape
    height, width = image.shape

    # Initialize list of points to visit, starting with the seed point
    to_visit = [seed_point]

    # Loop through each point in the to_visit list and grow the region
    while len(to_visit) > 0:
        # Pop the first point from the to_visit list
        current_point = to_visit.pop(0)

        # Check if the current point is within the image bounds
        if (current_point[0] >= 0 and current_point[0] < width and
            current_point[1] >= 0 and current_point[1] < height):
            
            # Check if the current point has not been visited
            if segmented_image[current_point[1], current_point[0]] == 0:
                # Mark the current point as visited by setting it to white in the segmented image
                segmented_image[current_point[1], current_point[0]] = 255

                # Check the intensity difference between the current point and its neighbors
                neighbors = get_neighbors(current_point, width, height)
                for neighbor in neighbors:
                    intensity_diff = np.abs(int(image[current_point[1], current_point[0]]) - int(image[neighbor[1], neighbor[0]]))
                    if intensity_diff <= threshold:
                        # Add the neighbor point to the to_visit list
                        to_visit.append(neighbor)

    path = saveImg_unique(segmented_image, "./static/img/output/",
                        rgb=True, save_on_current=False)
    return path

def get_neighbors(point, width, height):
   
    x, y = point
    neighbors = [(x-1, y-1), (x, y-1), (x+1, y-1),
                 (x-1, y),             (x+1, y),
                 (x-1, y+1), (x, y+1), (x+1, y+1)]

    # Filter out coordinates that are outside of the image bounds
    neighbors = filter(lambda p: p[0] >= 0 and p[0] < width and p[1] >= 0 and p[1] < height, neighbors)

    return neighbors

############# end region growing #################   

############# agglomerative ###################
def compute_init_matrix(points) :
    Dis_Mat = [[-1]*len(points) for i in range(0,len(points))]
    for i in range(len(points)) :
        for j in range(i,len(points)) :
            if(j == i) : 
                Dis_Mat[j][i] = -1
                continue
            Dis_Mat[j][i] = compute_Dis(points[i],points[j],i,j)
    return Dis_Mat

def compute_Dis(p1,p2,i,j) :
    p1 = np.append(p1,i)
    p2 = np.append(p2,j)
    return np.sqrt(np.sum(np.square(np.subtract(p1,p2))))

def Min_Dis(matrix) :
    minimum = [1,0]
    for i in range(len(matrix)) :
        for j in range(len(matrix[0])) :
            if((matrix[i][j] == -1)) : continue
            if(matrix[i][j] < matrix[minimum[0]][minimum[1]]) :
                minimum = [i,j]
    return minimum

def segment(image, number_of_clusters = 15) :
    

    points = image.reshape(image.shape[0] * image.shape[1] , 3)
    dindogram = [[i] for i in range(len(points))]
    if(number_of_clusters > len(points)) : raise Exception("Clusters exceeded points!!")
    Dis_Mat = compute_init_matrix(points)
    while len(dindogram) != number_of_clusters :
        minimum = Min_Dis(Dis_Mat)
        new_cluster = [dindogram[minimum[0]],dindogram[minimum[1]]]
        flat_new_cluster = [item for sublist in new_cluster for item in sublist]
        dindogram.pop(np.max(minimum))
        dindogram[np.min(minimum)] = flat_new_cluster
        Update_Mat(Dis_Mat,minimum[0],minimum[1])
    return points,dindogram

def Update_Mat(Dis_Mat,indx1,indx2) :
    maximum_indx = max([indx1,indx2])
    single_link(Dis_Mat,indx1,indx2)
    Dis_Mat.pop(maximum_indx)
    for i in range(len(Dis_Mat)) :
        Dis_Mat[i].pop(maximum_indx)
    

def single_link(Dis_Mat,indx1,indx2) :
    minimum_indx = min([indx1,indx2])
    for i in range(len(Dis_Mat)) :
        if(i == indx1) : continue
        if(i == indx2) : continue
        if(i < indx1) :
            if(i < indx2) :
                distanc_1 = Dis_Mat[indx1][i]
                distanc_2 = Dis_Mat[indx2][i]
                m = min([distanc_1,distanc_2])
                Dis_Mat[minimum_indx][i] = m
            else :
                distanc_1 = Dis_Mat[indx1][i]
                distanc_2 = Dis_Mat[i][indx2]
                m = min([distanc_1,distanc_2])
                if(minimum_indx == indx2) : Dis_Mat[i][minimum_indx] = m
                else : Dis_Mat[i][minimum_indx] = m
                
        else :
            if(i < indx2) :
                distanc_1 = Dis_Mat[i][indx1]
                distanc_2 = Dis_Mat[indx2][i]
                if(minimum_indx == indx1) : Dis_Mat[i][minimum_indx] = m
                else : Dis_Mat[i][minimum_indx] = m
            else :
                distanc_1 = Dis_Mat[i][indx1]
                distanc_2 = Dis_Mat[i][indx2]
                m = min([distanc_1,distanc_2])
                Dis_Mat[i][minimum_indx] = m

def draw(points,dindogram,Image_Before) :
    colors = []
    while len(colors) != len(dindogram):
        color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        if(color not in colors) : colors.append(color)
    for i in range(len(dindogram)) :
        for j in range(len(dindogram[i])) :
            indx = dindogram[i][j]
            points[indx] = colors[i]
    image = points.reshape(Image_Before.shape)
    path = saveImg_unique((image).astype(np.uint8), "./static/img/output/",
                        rgb=False, save_on_current=False)
    return path

def agglomerative(number_of_clusters=15,color_space="RGB"):
    
    Image_Before = read_rgb(gray=False, normalize=False)
    if color_space == "LUV":
        Image_Before =  RGB_to_LUV(Image_Before)
    image = np.array(Image_Before)
    points,dindogram = segment(image,number_of_clusters)
    return draw(points,dindogram,Image_Before)

############# end agglomerative ###################

############# mean shift ####################
from scipy.spatial import KDTree

def mean_shift( radius=10, threshold=10,color_space="RGB"):
    image = read_rgb(gray=False, normalize=False)
    if color_space == "LUV":
        image =  RGB_to_LUV(image)
    #create array of vectors
    vectors_list = []
    for i,row in enumerate(image):
        for j,pixels in enumerate(row):
            vector = np.array(list(pixels)+[i,j])
            vectors_list.append(vector)
    vectors_array = np.array(vectors_list)
    '''
    steps:
    1- iterate over each vectors
    2-find the other vectors that lie in the window
    3-calculate the mean then the vector is = mean
    4-repeate "from step 2" until convergance
    '''
    tree = KDTree(vectors_array)
    milestone = 10
    for i,vector in enumerate(vectors_array):
        vector_climbing = vector.copy()
        
        for _ in range(200):
            target_point = np.array(vector_climbing)
            indices = tree.query_ball_point(target_point, radius)
            vectors_within_radius = vectors_array[indices]
            new_mean = vectors_within_radius.T.mean(axis=1)
            
            if np.linalg.norm(new_mean-vector_climbing)*100/np.linalg.norm(new_mean) < threshold:
                vector_climbing = new_mean
                break
            else:
                vector_climbing = new_mean
        
        image[vector[3]][vector[4]] = vector_climbing[:3]
        
        if ((i+1)/len(vectors_array))*100> milestone :
            milestone+=10
            print(f"{int(((i+1)/len(vectors_array))*100)}%")
    path = saveImg_unique((image).astype(np.uint8), "./static/img/output/",
                        rgb=False, save_on_current=False)
    return path

################## end mean shift #########################