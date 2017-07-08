import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from pprint import pprint
print("OpenCV Version : %s " % cv2.__version__)
from copy import deepcopy

def plotBGR2RGB(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
    return(1)

# Given an image matrix, detects a face, draws a rectangle around it, and returns it
def face_eye_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## Change the path as per your local directory
    face_detector = cv2.CascadeClassifier("/home/mudassir/anaconda2/pkgs/opencv3-3.2.0-np111py27_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier("/home/mudassir/anaconda2/pkgs/opencv3-3.2.0-np111py27_0/share/OpenCV/haarcascades/haarcascade_eye.xml")
    rects_face = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(2, 2), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects_face:
       cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
       rects_eye = eye_detector.detectMultiScale(gray[y:y+h, x:x+w], scaleFactor=1.001, minNeighbors=2, minSize=(1, 1), flags=cv2.CASCADE_SCALE_IMAGE)
       for (x_e, y_e, w_e, h_e) in rects_eye:
           cv2.rectangle(image, (x + x_e, y + y_e), (x + x_e + w_e, y + y_e + h_e), (255, 0, 0), 2)
    return(image)

# Given path to an image, executes the face detection pipeline and plots resulting images
def face_plot(imgpath):
    image_p = cv2.imread(imgpath)
    image_f = deepcopy(image_p)
    ## Uncomment below code to run the script
    #image_f = face_eye_detector(image_f)
    res = np.hstack((image_p, image_f))
    #plt.figure(figsize=(20,10))
    #plotBGR2RGB(res)
    return(image_f)
