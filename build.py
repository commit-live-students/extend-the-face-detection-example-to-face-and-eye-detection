import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from pprint import pprint
from copy import deepcopy


def plotBGR2RGB(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
    return(1)

# change face_detector to faceeye_detector
def faceeye_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier("/Users/arunabhsingh/anaconda2/pkgs/opencv-3.2.0-np111py27_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    eye_detector = cv2.CascadeClassifier("/Users/arunabhsingh/anaconda2/pkgs/opencv-3.2.0-np111py27_0/share/OpenCV/haarcascades/haarcascade_eye.xml")
    rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in rects:
        print (x,y,w,h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        temp_gray = gray[y:y+h,x:x+w]
        eye_rects = eye_detector.detectMultiScale(temp_gray, scaleFactor=1.01, minNeighbors=4, minSize=(10,10), flags=cv2.CASCADE_SCALE_IMAGE)
        for (a, b, c, d) in eye_rects:
            cv2.rectangle(image, (x+a, y+b), (x+a + c, y+b + d), (255, 0, 0), 2)
    return(image)




def face_plot(imgpath):
    image_p = cv2.imread(imgpath)
    image_f = deepcopy(image_p)
    image_f = faceeye_detector(image_f)
    res = np.hstack((image_p, image_f))
    plt.figure(figsize=(20, 10))
    plotBGR2RGB(res)
    return (image_f)