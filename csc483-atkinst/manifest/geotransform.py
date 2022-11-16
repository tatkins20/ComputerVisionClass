import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

def AffineTransform(img, affinematrix):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    #print(gray.shape)
    height, width = gray.shape
    index_y, index_x = np.indices((height, width))
    outLinHomgPts = np.stack((index_x.ravel(),index_y.ravel(), np.ones(index_y.size)))
    inLinPts = np.round(affinematrix.dot(outLinHomgPts).astype(int))
    minX, minY = np.amin(inLinPts, axis=1)
    inLinPts -= np.array([[minX],[minY]])
    affineMaxX, affineMaxY = np.amax(inLinPts, axis=1)
    input = np.ones((affineMaxY+1, affineMaxX+1), dtype=np.uint8) * 0 
    input[inLinPts[1], inLinPts[0]] = gray.ravel()
    return input
