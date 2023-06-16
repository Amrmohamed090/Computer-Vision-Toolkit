from flask import Flask , render_template,request, url_for,session
from flask_cors import CORS
from flask_session import Session
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
import os
import base64
import json
import Filters
import traceback
import Filters
import Frequency
from random import randint
import cv2
import histograms
from histograms import hybridFd
import Hough
import active_contour
import FeatureDetection
import MatchingImage 
import time
import Thresholding
import Segmentation
from Segmentation import RGB_to_LUV
import FaceDetection
import FaceRecognition

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
Session(app)

CORS(app)
def saveImg(img,path):
    with open(path, 'wb') as f:
        f.write(img)

def check_for_session_state():
    path = './static/img/output'
    old_images = os.listdir(path)
    for file in old_images:
        os.remove(path+'/'+file)     
    try:
        if session["uploaded_img"]:
            pass
    except:
        traceback.print_exc()
          
        session["uploaded_img"] = ""


def saveImg_unique(img,path, rgb=True, save_on_current = True):
    path_img = f"{path}{randint(1,100000)}.png"
    with open(path_img, 'wb') as f:
        
        f.write(img)
    if save_on_current:
        with open('./static/img/input/current.png', 'wb') as f:
            f.write(img)
    return path_img




@app.route("/noise_req",methods=["GET","POST"])
def noise_req():
    if request.method == "POST":
        size=request.form["size"]
        size=int(size)
        sigma=request.form["sigma"]
        sigma=float(sigma)
        D0=request.form["D0"]
        D0=int(D0)

        if request.form["noise_type"] == "uniform_noise":
            path = Filters.uniformNoise()
        elif request.form["noise_type"] == "gaussian_noise":
            path = Filters.GaussianNoise(sigma=sigma)
        elif request.form["noise_type"] == "salt_noise":
            path = Filters.salt_pepperNoise()
        elif request.form["noise_type"] == "average_filter":
            path = Filters.avr_filter(size=size)
        elif request.form["noise_type"] == "median_filter":
            path = Filters.med_filter(size=size)
        elif request.form["noise_type"] == "gaussian_filter":
            path = Filters.gass_filter(size=size,sigma=sigma)
        elif request.form["noise_type"] == "sobel":
            path = Filters.filter(kernal= "sobel")
        elif request.form["noise_type"] == "roberts":
            path = Filters.filter_robert()
        elif request.form["noise_type"] == "prewitt":
            path = Filters.filter(kernal = "prewitt")
        elif request.form["noise_type"] == "low_pass":
            path = histograms.Freq_filter(D0,'low')
        elif request.form["noise_type"] == "high_pass":
            path = histograms.Freq_filter(D0,'high')
        elif request.form["noise_type"] == "canny":
            path = Filters.canny(low_threshold= float(request.form["low_thresh"]),high_threshold = float(request.form["high_thresh"]),sigma=float(request.form["sigmaEd"]))
        print(path)
    return json.dumps({1:path})
@app.route("/hough_req",methods=["GET","POST"])
def hough_req():
    if request.method == "POST":
        threshold=request.form["threshold"]
        threshold=int(threshold)
        thetastep=request.form["thetastep"]
        thetastep=int(thetastep)
        rstep=request.form["rstep"]
        rstep=int(rstep)


        if request.form["hough_type"] == "houghLine":
            path = Hough.drawHoughLines(rStep=rstep,angleStep=thetastep,ther=threshold)
        elif request.form["hough_type"] == "houghCircle":
            r_min = 50
            r_max = 200
            delta_r = rstep
            num_thetas = int(300/thetastep)
            bin_threshold = (float(threshold)/num_thetas)
            path = Hough.houghCirc(r_min, r_max, delta_r, num_thetas, bin_threshold)
        print(path)
    return json.dumps({1:path})

@app.route("/freq_request",methods=["GET","POST"])
def freq_request():
    sizethr= request.form["sizethr"] 
    sizethr= int(sizethr)
    thr= request.form["thr"] 
    thr= int(thr)
    if request.method == "POST":
        if request.form["req_type"] == "histogram":
            path = Frequency.histogram()
        elif request.form["req_type"] == "equalize":
            path = Frequency.equalizeImage()
        elif request.form["req_type"] == "normalize":
            path = Frequency.normalizeImage()
        elif request.form["req_type"] == "local_thresholding":
            imgThr = cv2.imread('./static/img/input/current.png',0)
            path = Frequency.localThreshold(imgThr,size=sizethr)
        elif request.form["req_type"] == "global_thresholding":
            imgThr = cv2.imread('./static/img/input/current.png',0)
            path = Frequency.globalThreshold(imgThr,thr=thr)
        elif request.form["req_type"] == "rgb_histogram":
            path = Frequency.rgb_histogram()
        print(path)
    return json.dumps({1:path})

@app.route("/hybrid_req",methods=["GET","POST"])
def hybrid_req():
    D0Low= request.form["D0Low"] 
    D0Low= int(D0Low)
    D0High= request.form["D0High"] 
    D0High= int(D0High)
    if request.method == "POST":
        img1_path = saveImg_unique(base64.b64decode( request.form["img_upload1"].split(',')[1]), "./static/img/output/", rgb=True, save_on_current = False)
        img2_path = saveImg_unique(base64.b64decode( request.form["img_upload2"].split(',')[1]), "./static/img/output/", rgb=True, save_on_current = False)
        path1 = hybridFd(img1_path=img1_path,img2_path=img2_path,low=D0Low,high=D0High)
        path2 = hybridFd(img1_path=img1_path,img2_path=img2_path,t=2,low=D0Low,high=D0High)
    return json.dumps({1: path1,2:path2})

@app.route("/feature",methods=["GET","POST"])
def feature():
    if request.method == "POST":
        start = time.time()
        if request.form["f"] == "harris":
            path = FeatureDetection.harris()
            
        if request.form["f"] == "lambda":
            path = FeatureDetection.corners_with_lambda(threshold=float(request.form["threshold"]))
        if request.form["f"] == "SIFT":
            path = FeatureDetection.SIFT()
        
        end = time.time()
        t = end - start
        return json.dumps({1: path,2:round(t,2)})

@app.route("/feature_matching_req",methods=["GET","POST"])
def feature_matching_req():
    if request.method == "POST":
        
        img1_path = saveImg_unique(base64.b64decode( request.form["img_upload1"].split(',')[1]), "./static/img/output/", rgb=True, save_on_current = False)
        img2_path = saveImg_unique(base64.b64decode( request.form["img_upload2"].split(',')[1]), "./static/img/output/", rgb=True, save_on_current = False)
        if request.form["type"] == "SSD":
            path,time = MatchingImage.main_ssd(img1_path,img2_path)
        elif request.form["type"] == "NCC":
            path,time = MatchingImage.main_ncc(img1_path,img2_path)
        time = round(time,2)
        return json.dumps({1: path,2:time})

@app.route("/thresholding_req",methods=["GET","POST"])
def thresholding_req():
    if request.method == "POST":
        number_slices=request.form["number_slices"]
        mode=request.form["mode"]
        number_slices=int(number_slices)



        if request.form["thresholding_type"] == "optimal":
            if(mode=='local'):
                path = Thresholding.localized(n=number_slices)
            else:
                path = Thresholding.optimal_Thres()
        elif request.form["thresholding_type"] == "Otsu":
            if(mode=='local'):
                path = Thresholding.localized(n=number_slices,thersholdMethod='otsu')
            else:
                path = Thresholding.otsuThreshold()
        elif request.form["thresholding_type"] == "spectral_Otsu":
            # path = Thresholding.Spectral_otsu_thres(num_thresholds=number_slices)
            path = Thresholding.segment_image_otsu(n=number_slices)
        elif request.form["thresholding_type"] == "spectral_kmean":
            # path = Thresholding.Spectral_otsu_thres(num_thresholds=number_slices)
            path = Thresholding.apply_spectral_kmean(num_clusters=number_slices)
        print(path)
    return json.dumps({1:path})

@app.route("/facedetection_req",methods=["GET","POST"])
def facedetection_req():
    if request.method == "POST":


        if request.form["facedetection_type"] == "Detect":
            path = FaceDetection.draw_faces()
        
        elif request.form["facedetection_type"] == "Recognize":
            path = FaceRecognition.recognize()
        print(path)
    return json.dumps({1:path})
    

@app.route("/active_contour_req",methods=["GET","POST"])
def active_contour_req():
    
    path = active_contour.start(float(request.form["radius"]),float(request.form["centerx"]),float(request.form["centery"]),int(request.form["area"]))
   
    # activecontour.active_contour(request.form["img_upload1"])
    return json.dumps({1: path})

@app.route("/segmentation_req",methods=["GET","POST"])
def segmentation_req():
    if request.method == "POST":
        segmentation_type = request.form["segmentation_type"]
        
        if   segmentation_type == "kmean":
            path = Segmentation.Kmean (k=int(request.form["k"]),number_of_iterations=int(request.form["max_iteration"]),color_space=request.form["color_space"])
            
        elif segmentation_type == "region_growing":
            path =  Segmentation.region_growing( (int(request.form["seed_x"]),int(request.form["seed_y"])), float(request.form["thresh_region"]))
        
        elif segmentation_type == "agglomerative":
            path =  Segmentation.agglomerative(number_of_clusters=int(request.form["number_of_clusters"]),color_space=request.form["color_space"])
        elif segmentation_type == "mean_shift":
            path = Segmentation.mean_shift( radius=int(request.form["raduis"]), threshold=int(request.form["threshold_mean"]),color_space=request.form["color_space"])
    
    return json.dumps({1:path})

@app.route("/rgb_to_luv",methods=["GET","POST"])
def rgb_to_luv():
    image = cv2.imread("./static/img/input/current.png")
    image = RGB_to_LUV(image)
    plt.imshow(image)
    path = "./static/img/output/" + str(randint(1,100000))+".png"
    plt.savefig(path)
    return json.dumps({1:path})
#start render tempates
@app.route("/edgedetection",methods=["GET","POST"])
def edgedetection():
    return render_template("edgedetection.html", img = session["uploaded_img"])

@app.route("/equalize",methods=["GET","POST"])
def equalize():
    return render_template("equalize.html", img = session["uploaded_img"])

@app.route("/hybrid",methods=["GET","POST"])
def hybrid():
    return render_template("hybrid.html", img = session["uploaded_img"])
@app.route("/hough",methods=["GET","POST"])
def hough():
    return render_template("hough.html", img = session["uploaded_img"])
@app.route("/active_contour",methods=["GET","POST"])
def active():
    return render_template("active_contour.html", img = session["uploaded_img"])
@app.route("/thresholding",methods=["GET","POST"])
def thresholding():
    return render_template("thresholding.html", img = session["uploaded_img"])
@app.route("/segmentation",methods=["GET","POST"])
def segmentation():
    return render_template("segmentation.html", img = session["uploaded_img"])
@app.route("/facedetection",methods=["GET","POST"])
def facedetection():
    #session["PCA"] = FaceRecognition.create_recognition_utility()
    return render_template("facedetection.html", img = session["uploaded_img"])

@app.route("/",methods=["GET","POST"])
def main():
    #initializing the session states
    check_for_session_state()
    return render_template("index.html", img = session["uploaded_img"])

@app.route("/noise",methods=["GET","POST"])
def noise():
    return render_template("noise.html", img = session["uploaded_img"])

@app.route("/feature_extraction",methods=["GET","POST"])
def feature_extraction():
    return render_template("feature_extraction.html", img = session["uploaded_img"])

@app.route("/feature_matching",methods=["GET","POST"])
def feature_matching():
    return render_template("feature_matching.html", img = session["uploaded_img"])

#end render tempates
@app.route("/reset_current_img",methods=["GET","POST"])
def reset_current_img():
    if request.method=="POST":
        if session["uploaded_img"]  != "":
            saveImg(base64.b64decode(session["uploaded_img"] .split(',')[1]),'./static/img/input/current.png')
    return json.dumps({1: session["uploaded_img"]})



@app.route('/saveImg',methods =['POST',"GET"])
def save_Img():
    if request.method == "POST":
        #save original images
       session["uploaded_img"] = request.form["img_upload"]
    
        
       saveImg(base64.b64decode( request.form["img_upload"].split(',')[1]),'./static/img/input/uploaded.png')
       path = saveImg_unique(base64.b64decode( request.form["img_upload"].split(',')[1]), "./static/img/output/", rgb=True,save_on_current = True)
        
        
    return json.dumps({1: path})

if __name__ == "__main__":
    app.run(debug=True)