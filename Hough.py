from io import StringIO
from Filters import saveImg_unique
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
from collections import defaultdict


def read_rgb():
    # 0 is to read the as gray scale
    img = cv2.imread('./static/img/input/current.png')
    return img


def houghLine(rStep=1, angleStep=1, ther=100):
    img = read_rgb()
    edgesImg = cv2.Canny(img, 50, 150, apertureSize=3, L2gradient=True)
    x, y = edgesImg.shape
    thetas = np.arange(-90, 90, angleStep)
    thetas = np.deg2rad(thetas)
    # get the diagonal of the img max r
    img_diag = int(np.round(np.sqrt(x * x + y * y)))
    r = np.arange(-img_diag, img_diag, rStep)
    accumulator = np.zeros((len(r), len(thetas)), dtype=np.int64)
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    for i in range(x):
        for j in range(y):
            if (edgesImg[i][j] != 0):
                for z in range(len(thetas)):
                    rs = (i*cos_theta[z])+(j*sin_theta[z])
                    # dont sure why we add img diagonal
                    rs = int(np.round(rs))+img_diag
                    accumulator[rs][z] += 1  # increment at radius and theta

    radius_idx, theta_idx = np.where(accumulator > ther)
    radius = r[radius_idx]
    theta = thetas[theta_idx]
    return radius, theta


def drawHoughLines(rStep=1, angleStep=1, ther=100):
    radius, thetas = houghLine(rStep, angleStep, ther)
    img1 = read_rgb()
    for i in range(len(radius)):
        r = radius[i]
        theta = thetas[i]
        # Stores the value of cos(the) in a
        a = np.cos(theta)
        # Stores the value of sin(the) in b
        b = np.sin(theta)
        # x0 stores the value rcos(the)
        x0 = a*r
        # y0 stores the value rsin(the)
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
    path = saveImg_unique(img1, "./static/img/output/",
                          rgb=True, save_on_current=True)
    return path

def houghCirc(r_min=50, r_max=200, delta_r=1, num_thetas=100, bin_threshold=0.29):
        image = read_rgb()
        edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_image = cv2.Canny(edge_image, 50, 400)
        # image size
        img_height, img_width = edge_image.shape[:2]

        # R and Theta ranges
        dtheta = int(360 / num_thetas)

        # Thetas is bins created from 0 to 360 degree with increment of the dtheta
        thetas = np.arange(0, 360, step=dtheta)

        # Radius ranges from r_min to r_max
        rs = np.arange(r_min, r_max, step=delta_r)

        # Calculate Cos(theta) and Sin(theta) it will be required later
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        '''Evaluate candidate circles dx and dy for different delta radius
    based on the the parametric equation of circle.
    x = x_center + r * cos(t) and y = y_center + r * sin(t),
    where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)'''

        circle_candidates = []
        for r in rs:
            for t in range(num_thetas):
                circle_candidates.append(
                    (r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))

        '''Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not
    already present in the dictionary instead of throwing exception.'''
        accumulator = defaultdict(int)

        for y in range(img_height):
            for x in range(img_width):
                if edge_image[y][x] != 0:  # white pixel(edge)
                    # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                    for r, rcos_t, rsin_t in circle_candidates:
                        x_center = x - rcos_t
                        y_center = y - rsin_t
                        # vote for current candidate
                        accumulator[(x_center, y_center, r)] += 1

        # Output image with detected lines drawn
        output_img = image.copy()
        # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold)
        out_circles = []

        # Sort the accumulator based on the votes for the candidate circles
        for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
            x, y, r = candidate_circle
            current_vote_percentage = votes / num_thetas
            if current_vote_percentage >= bin_threshold:
                out_circles.append((x, y, r, current_vote_percentage))
                # print(x, y, r, current_vote_percentage)

        # Post process to exclude circles that are too close to each other (Remove nearby duplicate circles based on pixel_threshold)
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in out_circles:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
        out_circles = postprocess_circles

        # Draw shortlisted circles on the output image
        for x, y, r, v in out_circles:
            output_img = cv2.circle(output_img, (x, y), r, (255, 0, 0), 2)

        path = saveImg_unique(
            output_img, "./static/img/output/", rgb=True, save_on_current=True)
        return path
