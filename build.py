import cv2
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def plotBGR2RGB(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
    return(1)

# change face_detector to faceeye_detector
def face_plot(imgpath):
    image_p = cv2.imread(imgpath)
    image_f = deepcopy(image_p)
    image_f = faceeye_detector(image_f)
    res = np.hstack((image_p, image_f))
    plt.figure(figsize=(20, 10))
    plotBGR2RGB(res)
    return (image_f)