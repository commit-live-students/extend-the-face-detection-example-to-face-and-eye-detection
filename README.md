# Extend the face detection example to face and eye detection

### The Setting
Let's look at the helper functions we had built for Face Detection:

    %matplotlib inline
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
    def face_detector(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detector = cv2.CascadeClassifier("/Users/soumendra/anaconda3/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
        rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return(image)

    # Given path to an image, executes the face detection pipeline and plots resulting images
    def face_plot(imgpath):
        image_p = cv2.imread(imgpath)
        image_f = deepcopy(image_p)
        image_f = face_detector(image_f)
        res = np.hstack((image_p, image_f))
        plt.figure(figsize=(20,10))
        plotBGR2RGB(res)
        return(image_f)
        
### Problem Statement
Create a new function called `faceeye_detector()` to replace the function `face_detector()`. The new function `faceeye_detector()` identifies eyes in addition to the face, and draws rectangles around eyes as well.

* It should take an image (numpy array) as input
* The output should be an image (numpy array) which is the same as input image, but now with rectangles drawn around any faces and eyes present in the image

### Hint
* https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

