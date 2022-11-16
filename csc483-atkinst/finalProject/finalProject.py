import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def contourPlanaria(img):
    
    copy = img.copy()
    
    bilateralImg = cv2.bilateralFilter(copy, 7, 100, 100)
    
    grayscaleImg = cv2.cvtColor(bilateralImg, cv2.COLOR_BGR2GRAY)
    
    #hsv = cv2.cvtColor(bilateralImg, cv2.COLOR_BGR2HSV)
    
    #hue,saturation,value = cv2.split(hsv)
    
    grayMax = grayscaleImg.max()
    
    #_bin,binaryThresh = cv2.threshold(grayscaleImg,128, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph1 = cv2.morphologyEx(grayscaleImg, cv2.MORPH_CLOSE, kernel, iterations = 5)
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_OPEN, kernel, iterations = 3)
    morph3 = cv2.morphologyEx(morph2, cv2.MORPH_CLOSE, kernel, iterations = 3)
         
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))    
    #morphed1 = cv2.morphologyEx(binaryThresh, cv2.MORPH_OPEN, element) 
    #morphed2 = cv2.morphologyEx(morphed1, cv2.MORPH_CLOSE, element)
    #_3, thresh3 = cv2.threshold(morphed2, 128, 255, cv2.THRESH_BINARY_INV)
    #thresh1 = cv2.adaptiveThreshold(morph3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)
    ret, thresh = cv2.threshold(morph3,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    #_2, thresh2 = cv2.threshold(morph3, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    #_2, thresh4 = cv2.threshold(morph3, 0, 255, cv2.THRESH_BINARY_INV)
    gaussFilter = cv2.GaussianBlur(thresh,(5,5),0)
    edges = cv2.Canny(thresh,100,200)
    plt.imshow(edges)
    plt.show()
    #medianFilter = cv2.medianBlur(thresh,5)
    #image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(gaussFilter)
    keyptsDrawn = cv2.drawKeypoints(gaussFilter, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    origImg, contours, hierarchy = cv2.findContours(image = edges, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
   
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    #h = 0
    #iter = 100
    #while h < iter:
        #morph2 = cv2.morphologyEx(bilateralImg, cv2.MORPH_CLOSE, kernel)
        #h += 1

    #j = 0
    #iterable = 25
    #while j < iterable:
        #morph3 = cv2.morphologyEx(morph2, cv2.MORPH_OPEN, kernel)
        #j += 1
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))


    #gradient_image = cv2.morphologyEx(morph3, cv2.MORPH_GRADIENT, kernel)

    #image_channels = np.split(np.asarray(morph3), 3, axis=2)

    #channel_height, channel_width, _ = image_channels[0].shape

    #for i in range(0, 3):
        #_, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        #image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))


    #image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
    #detector = cv2.SimpleBlobDetector_create()
    #keypoints = detector.detect(image_channels)
    #keyptsDrawn = cv2.drawKeypoints(image_channels, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #denoise = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    #pla1gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #blur = cv2.medianBlur(pla1gray,5)
    #blur1 = cv2.GaussianBlur(pla1gray,(5,5),0)
    #thresh1 = cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)

    return contours
