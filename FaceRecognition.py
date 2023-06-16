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
import scipy.linalg as s_linalg
import sys
from scipy.misc import face
import os
from sklearn import preprocessing
import FaceDetection

def crop_face(image, faces):

     # Draw a rectangle around the faces
    min = 999999
    t = ()
    for (x, y, w, h) in faces:
        if w*h < min:
            t = (x, y, w, h)
    
    x, y, w, h = t
    
    length = max(w, h)
    center_y = int(y+ h/2)
    center_x = int(x+ w/2)
    
    x = int(center_x - length/2)
    y = int(center_y - length/2)
    cropped_image = image[y:y+length,x:x+length]
    cropped_image = cv2.resize(cropped_image, (50,50))
    return cropped_image



class datasetClass:
    def __init__(self, required_no):
        # required_no number of images for each person from dataset
        dataset_name = "\DatasetCV"
        dir = "Faces" + dataset_name 
        
        self.images_path_for_training = []
        self.images_label_for_training = []
        self.no_of_images_for_training = []
        
        self.images_path_for_testing = []
        self.images_label_for_testing = []
        self.no_of_images_for_testing = []

        self.images_target = []
        
        per_no = 0
        for name in os.listdir(dir):
            dir_path = os.path.join(dir, name)
            if os.path.isdir(dir_path):
                if len(os.listdir(dir_path)) >= required_no:
                    i = 0
                    for img_name in os.listdir(dir_path):
                        img_path = os.path.join(dir_path, img_name)
                        if i < required_no:
                            self.images_path_for_training += [img_path]
                            self.images_label_for_training += [per_no]
                            
                            if len(self.no_of_images_for_training) > per_no:
                                self.no_of_images_for_training[per_no] += 1
                            else :
                                self.no_of_images_for_training += [1]
                                
                            if i == 0:
                                self.images_target += [name]
                            
                        else :
                            self.images_path_for_testing += [img_path]
                            self.images_label_for_testing += [per_no]
                            
                            if len(self.no_of_images_for_testing) > per_no:
                                self.no_of_images_for_testing[per_no] += 1
                            else :
                                self.no_of_images_for_testing += [1]
                        i += 1
                    per_no += 1
                    
                    
class imageToMatrixClass:
    
    def __init__(self, images_path, images_width, images_height):
        self.images_path =images_path
        self.images_width = images_width
        self.images_height = images_height
        self.images_size = images_height * images_width
        
    def get_matrix(self):
        
        col = len(self.images_path)
        img_mat = np.zeros((self.images_size, col))
        
        i = 0
        for path in self.images_path:
            print(type(self.images_path))
            gray = cv2.imread(path, 0)
            faces = FaceDetection.detect_faces()
            gray = crop_face(gray, faces)
            gray_resized = cv2.resize(gray,(self.images_width, self.images_height))
            mat_gray = np.asmatrix(gray_resized)
            vec = mat_gray.ravel()
            
            img_mat [: ,i] = vec
            
            i +=1
        return img_mat

class PCA:
    
    def __init__(self, image_matrix, image_label, image_targets, no_of_elements, images_width, images_height, quality_percent):
        self.image_matrix = image_matrix
        self.image_label = image_label
        self.image_targets = image_targets
        self.no_of_elements = no_of_elements
        self.images_width = images_width
        self.images_height = images_height
        self.quality_percent = quality_percent

        mean = np.mean(self.image_matrix , 1)
        self.mean_face = np.asmatrix(mean).T
        self.image_matrix -= self.mean_face
        
    # p value is the number of cols will be taken from output of eign values because the only few cols have a maximum variation
    def give_p_val(self, eign_val):
        
        sum_original = np.sum(eign_val)
        sum_threshold = sum_original * self.quality_percent/100
        sum_temp = 0
        p = 0
        # The eigen values from svd is returned as 1d arr not a diagonal matrix
        while sum_temp <sum_threshold :
            sum_temp += eign_val[p]
            p += 1
        return p
        
    def reduce_dim(self):

        u, eign_val, v_t = s_linalg.svd(self.image_matrix, full_matrices=True) 
        # function used to find the eigen vector U and eigen_val of the covariance matrix of a given matrix
        p = self.give_p_val(eign_val)
        self.new_bases = u[: , 0:p]
        self.new_coordinates = np.dot (self.new_bases.T, self.image_matrix)
        return self.new_coordinates
    
    def new_cords(self, single_image):
        print(single_image.shape)
        img_vec = np.asmatrix(single_image).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.image_label) + img_vec))/(len(self.image_label)+ 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)
    
    def recognize_face(self, new_cords_of_image):
        classes = len(self.no_of_elements)
        start = 0
        dist = []
        for i in range(classes):
            temp_imgs = self.new_coordinates [: ,int(start):int(start + self.no_of_elements[i])]
            mean_temp = np.asmatrix(np.mean(temp_imgs, 1)).T
            start = start + self.no_of_elements[i]
            dist_temp = np.linalg.norm(new_cords_of_image - mean_temp)
            dist += [dist_temp]

        min_pos = np.argmin(dist)
        return self.image_targets[min_pos]
    
    def image_form_path(self, path):
        gray = cv2.imread(path, 0)
        return cv2.resize(gray, (self.images_width, self.images_height))
    
    def new_to_old_cords(self, new_cords):
        return self.mean_face + (np.asmatrix(np.dot(self.new_bases * new_cords))).T
    
    def show_image(self, label_to_show, old_cords):
        old_cords_matrix = np.reshape(old_cords, [self.images_width, self.images_height])
        old_cords_integers= np.array(old_cords_matrix, dtype=np.uint8)
        resized_image= cv2.resize(old_cords_integers, (500, 500))
        cv2.imshow(label_to_show, resized_image)
        cv2.waitKey()
        
    def show_eigen_faces(self, min_pix_int, max_pix_int, eig_face_no):
        ev = self.new_bases[: , eig_face_no:eig_face_no + 1]
        min_orig = np.min(ev)
        max_orig = np.max(ev)

        ev= min_pix_int + (((max_pix_int - min_pix_int)/(max_orig - min_orig)) * ev)
        self.show_image("Eigen Face" +str(eig_face_no), ev )

def create_recognition_utility():
    no_images_of_one_person = 2 
    
    dataset_obj= datasetClass(no_images_of_one_person)
    images_paths_for_training= dataset_obj.images_path_for_training
    print("#"*30)
    print("training for :")
    print(images_paths_for_training)
    print("#"*30)
    labels_for_training = dataset_obj.images_label_for_training
    images_target = dataset_obj.images_target
    no_of_elements_for_training = dataset_obj.no_of_images_for_training
    

        
    img_width, img_height =50 , 50
    imageToMatrixClassobj= imageToMatrixClass(images_paths_for_training, img_width, img_height)
    img_matrix = imageToMatrixClassobj.get_matrix()

    PCA_obj = PCA(img_matrix, labels_for_training, images_target, no_of_elements_for_training, img_width, img_height, quality_percent = 90)
    PCA_obj.reduce_dim()
    return PCA_obj
def recognize2(PCA_obj):

    image=read_rgb(gray=True, normalize=False)
    faces = FaceDetection.detect_faces()
    image = crop_face(image, faces)
              
    new_cords_for_image = PCA_obj.new_cords(image)
    find_name = PCA_obj.recognize_face(new_cords_for_image)
    return f"./static/DatasetCVcrop/{find_name}/normal.png"                  

def recognize():
    
    image=read_rgb(gray=True, normalize=False)
    faces = FaceDetection.detect_faces()
    image = crop_face(image, faces)
    
    no_images_of_one_person = 2 
    dataset_obj= datasetClass(no_images_of_one_person)
    images_paths_for_training= dataset_obj.images_path_for_training
    print("#"*30)
    print("training for :")
    print(images_paths_for_training)
    print("#"*30)
    labels_for_training = dataset_obj.images_label_for_training
    images_target = dataset_obj.images_target
    no_of_elements_for_training = dataset_obj.no_of_images_for_training
    
    images_paths_for_testing= dataset_obj.images_path_for_testing
    labels_for_testing = dataset_obj.images_label_for_testing
    no_of_elements_for_testing= dataset_obj.no_of_images_for_testing
        
    img_width, img_height =50 , 50
    imageToMatrixClassobj= imageToMatrixClass(images_paths_for_training, img_width, img_height)
    img_matrix = imageToMatrixClassobj.get_matrix()

    my_algo_class_obj = PCA(img_matrix, labels_for_training, images_target, no_of_elements_for_training, img_width, img_height, quality_percent = 90)
    my_algo_class_obj.reduce_dim()
    
    
    new_cords_for_image = my_algo_class_obj.new_cords(image)
    find_name = my_algo_class_obj.recognize_face(new_cords_for_image)
    return f"./static/DatasetCVcrop/{find_name}/normal.png"