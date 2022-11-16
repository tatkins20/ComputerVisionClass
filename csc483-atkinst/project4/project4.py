import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

def SobelEdgeDetectorX(image):
    #gaussianblur = cv2.GaussianBlur(image, (5,5), 0)
    #gaussianimg = cv2.filter2D(image, cv2.CV_32F, gaussianfilter)
    kernelx = np.asarray([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]], dtype = 'float32')
    sobeledgeimg = cv2.filter2D(image, cv2.CV_32F, kernelx)
    return sobeledgeimg

def SobelEdgeDetectorY(image):
    #gaussianblur = cv2.GaussianBlur(image, (5,5), 0)
    #gaussianimg = cv2.filter2D(image, cv2.CV_32F, gaussianfilter)
    kernely = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = 'float32')
    sobeledgeimg = cv2.filter2D(image, cv2.CV_32F, kernely)
    return sobeledgeimg

def HarrisDetector(img,k = 0.04):
    '''
    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale or color (your choice)
                (i recommmend greyscale)
    -   k: k value for Harris detector
    Returns:
    -   R: A numpy array of shape (m,n) containing R values of interest points
    '''
    #img = img.astype('float32')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    lx = SobelEdgeDetectorX(gray)
    lx = lx.astype('float32')
    ly = SobelEdgeDetectorY(gray)
    ly = ly.astype('float32')
    lxx = np.multiply(lx, lx)
    lxx = lxx.astype('float32')
    lxy = np.multiply(lx, ly)
    lxy = lxy.astype('float32')
    lyy = np.multiply(ly, ly)
    lyy = lyy.astype('float32')
    #gaussianblur = cv2.getGaussianKernel(5,5)
    convolvedLXX = cv2.GaussianBlur(lxx, (5,5), cv2.BORDER_DEFAULT)
    convolvedLXY = cv2.GaussianBlur(lxy, (5,5), cv2.BORDER_DEFAULT)
    convolvedLYY = cv2.GaussianBlur(lyy, (5,5), cv2.BORDER_DEFAULT)
    convolvedLXX = convolvedLXX.astype('float32')
    convolvedLXY = convolvedLXY.astype('float32')
    convovledLYY = convolvedLYY.astype('float32')
    #convolvedLXX = cv2.filter2D(lxx, cv2.CV_32F, gaussianblur)
    #convolvedLXY = cv2.filter2D(lxy, cv2.CV_32F, gaussianblur)
    #convolvedLYY = cv2.filter2D(lyy, cv2.CV_32F, gaussianblur)
    rList = []
    #rArray = np.empty(gray.shape)
    #rArray = rArray.astype('float64')

    det = (convolvedLXX * convolvedLYY - convolvedLXY ** 2)
    trace = (convolvedLXX + convolvedLYY)
    r = det - k * (trace ** 2)
            #np.insert(rArray,[y,x], r)
            #det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
            #trace = a[0][0] + a[1][1]
            #r = det - k * (trace ** 2)
    rList.append(r)

    rArray = np.array(rList)
    rArray = np.reshape(rArray, gray.shape)
    #rArray = np.empty(1)
    #for element in rList:
        #if element > 0.01 * max(rList):
            #np.append(rArray, element)
    return rArray


def SuppressNonMax(Rvals, numPts):
    '''
    Args:
    -   Rvals: A numpy array of shape (m,n,1), containing Harris response values
    -   numPts: the number of responses to return

    Returns:
    x: A numpy array of shape (N,) containing x-coordinates of interest points
    y: A numpy array of shape (N,) containing y-coordinates of interest points
    confidences (optional): numpy nd-array of dim (N,) containing the strength
    of each interest point
    '''

    rList = []
    Rvals = np.float32(Rvals)
    for y in range(Rvals.shape[0]):
        for x in range(Rvals.shape[1]):
            if (Rvals[y][x]> 0.01 * Rvals.max()):
                rList.append((x, y, Rvals[y][x]))

    rList = sorted(rList, key=lambda tup: tup[2], reverse=True)
    #arbitrary = 100
    #topRList = []
    #for i in range(arbitrary):
        #topRList.append(rList[i])
    #print(rList)

    #distances = []
    radii = []
    xRadii = []
    yRadii = []
    #rListCopy = rList[1:]
    unsupX, unsupY, unsupConfidence = rList[0]
    xRadii.append(unsupX)
    yRadii.append(unsupY)
    for i in range(1, len(rList)):
        distances = []
        x,y, confidence = rList[i]
        index = i-1
        while (index >=0):
            newX, newY, newConfidence = rList[index]
            distance = math.sqrt((newX - x) ** 2 + (newY - y) ** 2)
            distances.append(distance)
            index = index-1

        minDist = min(distances)
        radii.append((x, y, minDist))

    radii = sorted(radii, key=lambda tup: tup[2], reverse=True)
    #print(radii)


    for i in range(0, numPts-1):
        xcord, ycord, rad = radii[i]
        xRadii.append(xcord)
        yRadii.append(ycord)

    xRadii = np.asarray(xRadii)
    yRadii = np.asarray(yRadii)

    return xRadii, yRadii
    #return rList
    #return x, y, confidences

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import cv2
    img = cv2.imread('testimage.pgm')
    k = 0.04
    Responses = HarrisDetector(img,k)
    print(SuppressNonMax(Responses,20))
