import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import sobel
from skimage.filters import gaussian
import math
from random import randint
import mapbox_earcut as earcut

###################
##################
#############
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
def plt_img_contour(im,c):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap=plt.cm.gray)
    ax.scatter(np.array(c)[:, 1], np.array(c)[:, 0],color="blue")
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, im.shape[1], im.shape[0], 0])
    
    


def distance(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))
def slop(p1,p2):
    return np.linalg.norm(p1-p2)

    
def resample(contour,nsamples):
    
    length_contour = 0
    for i in range(len(contour)-1):
        length_contour += np.linalg.norm(np.array(contour[i])-np.array(contour[i+1]))
    sr = length_contour/nsamples
    
    new_contour = [list(contour[0])]

    ctrl_i = 1
    ncontour = 1

    ss = sr
    bends_count = 0
    while(ncontour <nsamples):

        d = distance(contour[ctrl_i],new_contour[-1])
        if d > ss:
            increment_x = np.cos(np.arctan2(contour[ctrl_i][0]-new_contour[-1][0],contour[ctrl_i][1]-new_contour[-1][1])) * ss
            increment_y = np.sin(np.arctan2(contour[ctrl_i][0]-new_contour[-1][0],contour[ctrl_i][1]-new_contour[-1][1])) * ss
            if (contour[ctrl_i][1]-new_contour[-1][1]) >= 0:
                increment_x = abs(increment_x)
            else:
                increment_x = -1 * abs(increment_x)
            if (contour[ctrl_i][0]-new_contour[-1][0]) >= 0:
                increment_y = abs(increment_y)
            else:
                increment_y = -1 * abs(increment_y)
                
            
            
            new_x = (new_contour[-1][1] + increment_x)
            new_y = (new_contour[-1][0] + increment_y)
            
            for _ in range(bends_count):
                new_contour.pop()


            new_contour.append([new_y,new_x])
            

            ncontour = ncontour + 1
            bends_count = 0
            ss = sr

            
        else:
            ss = ss- d
            new_contour.append(list(contour[ctrl_i]))
            ctrl_i = ctrl_i + 1
            bends_count = bends_count + 1
 
    new_contour.append(new_contour[0])
    return new_contour

def calculate_area(cnt,img):
    verts = np.array(cnt).reshape(-1, 2)
    rings = np.array([len(verts)])
    result = earcut.triangulate_float32(verts, rings)
    start = 0
    finish = len(verts[result])
    area = 0
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)
    ax.scatter(np.array(cnt)[:, 1], np.array(cnt)[:, 0])
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    while (start<finish):
        a = np.linalg.norm(np.array(verts[result][start])-np.array(verts[result][start+1]))
        b = np.linalg.norm(np.array(verts[result][start])-np.array(verts[result][start+2]))
        c = np.linalg.norm(np.array(verts[result][start+1])-np.array(verts[result][start+2]))
        s = (a+b+c)/2
        area += np.sqrt((s*(s-a)*(s-b)*(s-c)))
        plt.plot([verts[result][start][1],verts[result][start+1][1],verts[result][start+2][1]],
                [verts[result][start][0],verts[result][start+1][0],verts[result][start+2][0]])
        start+=3
    
    for i, point in enumerate(cnt):
        if cnt[-1][0] != cnt[0][0] or cnt[-1][0] != cnt[0][0]:
            length = distance(cnt[-1],cnt[0])
        else:
            length = 0
            
        for i in range(len(cnt)-1):
            length += distance(cnt[i],cnt[i+1])
            
            
            
    
    ax.text(0.95, 0.01, f'Area: {round(area)} perimeter: {round(length)}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='white', fontsize=15)
    
    
        
            
    path = "./static/img/output/" + str(randint(1,100000))+".png"
    print(path)
    plt.savefig(path)
    plt.clf()
    return path

def first_greedy(img, contour,alpha=1,beta=1,gamma=1):
    s = 5
    
    
    for _ in range(500):
        rg = list(range(len(contour)))
        
        for i in rg:
            x , y = contour[i]
            x_prev , y_prev = contour[i-1]
            if i == len(contour)-1:
                x_next , y_next = contour[0]
            else :
                x_next , y_next = contour[i+1]
            window = img[round(x)-int((s-1)/2):round(x)+int((s-1)/2)+1,round(y)-int((s-1)/2):round(y)+int((s-1)/2)+1]
            window = alpha * window + beta * ((x_next - x)**2 + (y_next-y)**2) + gamma * ((x_next-(2*x)+x_prev)**2+(y_next-(2*y)+y_prev)**2)
            new_s = s
            
            try:
                mx_index = np.unravel_index(window.argmax(), window.shape)
                
            except:
                print("diverged")
                return contour
            
                
            contour[i] = np.array([contour[i][0]+mx_index[0]-int(s/2), contour[i][1]+mx_index[1]-int(s/2)])
        
        contour = contour[0 : len(contour)-1]
        contour = np.roll(contour,-1,axis=0)
        contour = resample(contour,100)
        
        plt_img_contour(img,contour)
        
        
        
    return contour

def start(radius,centerx,centery,area, alpha=1, beta=1,gamma=1):


    img = read_rgb(gray=True,normalize = False)

    s = np.linspace(0, 2*np.pi, 100)
    y_coordinates = centery + radius*np.sin(s)
    x_coordinates = centerx + radius*np.cos(s)
    init = np.array([y_coordinates, x_coordinates]).T

    new_img = img.copy()
    new_img =sobel(new_img)
    new_img = gaussian(new_img, 3, preserve_range=False)
    new_img = new_img**2
    cnt = init.copy()
    cnt = np.array(first_greedy(new_img, cnt,alpha,beta,gamma))
    

    if area == 0:
    
        
        fig, ax = plt.subplots()
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(cnt[:, 1], cnt[:, 0],color="blue")
        ax.plot(init[:, 1], init[:, 0],color="red")
        ax.set_xticks([]), ax.set_yticks([])

        path = "./static/img/output/" + str(randint(1,100000))+".png"
        print(path)
        plt.savefig(path)
        plt.clf()
        plt.clf()
        plt.clf()
        return path
        
    else:
        path = calculate_area(list(cnt),img)
        return path
    
