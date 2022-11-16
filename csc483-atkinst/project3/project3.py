import cv2
import numpy as np
import math

def SobelEdgeDetectorX(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
    else:
        gray = image
        gray = np.float32(gray)
    gaussianblur = cv2.GaussianBlur(gray, (5,5), 0)
    #gaussianimg = cv2.filter2D(image, cv2.CV_32F, gaussianfilter)
    kernelx = np.asarray([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    sobeledgeimg = cv2.filter2D(gaussianblur, cv2.CV_32F, kernelx)
    return sobeledgeimg

def SobelEdgeDetectorY(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
    else:
        gray = image
        gray = np.float32(gray)
    gaussianblur = cv2.GaussianBlur(gray, (5,5), 0)
    #gaussianimg = cv2.filter2D(image, cv2.CV_32F, gaussianfilter)
    kernely = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobeledgeimg = cv2.filter2D(gaussianblur, cv2.CV_32F, kernely)
    return sobeledgeimg

def EdgeMagnitude(vertEdge, horzEdge):
    magnitude = np.hypot(vertEdge, horzEdge)
    return magnitude

def EdgeOrientation(vertEdge, horzEdge):
    orientation = np.arctan2(vertEdge, horzEdge)
    return orientation

def NonMaxSupress(edgeMagImg, edgeOrientImg):
    angleBins = np.array([0, 45, 90, 135])
    #imgAngles = np.rad2deg(edgeOrientImg)
    #imgAngles = np.linspace(imgAngles)
    #roundedAngles = np.digitize(imgAngles, angleBins)
    #print(roundedAngles)
    #print(edgeMagImg)
    for y in range(len(edgeMagImg)-1):
        for x in range(len(edgeMagImg[y])-1):
            imgAngles = np.rad2deg(edgeOrientImg[y])
            #imgAngles = np.linspace(imgAngles)
            roundedAngles = np.digitize(imgAngles, angleBins)
            #print(imgAngles, roundedAngles)
            #print(imgAngles)
            if angleBins[roundedAngles[y]-1] == 0:
                if edgeMagImg[y][x-1] <= edgeMagImg[y][x] and edgeMagImg[y][x+1] <= edgeMagImg[y][x]:
                    edgeMagImg[y][x] = 0
            elif angleBins[roundedAngles[x]-1] == 45:
                if edgeMagImg[y-1][x] <= edgeMagImg[y][x] and edgeMagImg[y+1][x] <= edgeMagImg[y][x]:
                    edgeMagImg[y][x] = 0
            elif angleBins[roundedAngles[x]-1] == 90:
                if edgeMagImg[y-1][x-1] <= edgeMagImg[y][x] and edgeMagImg[y+1][x+1] <= edgeMagImg[y][x]:
                    edgeMagImg[y][x] = 0
            elif angleBins[roundedAngles[x]-1] == 135:
                if edgeMagImg[y-1][x+1] <= edgeMagImg[y][x] and edgeMagImg[y+1][x-1] <= edgeMagImg[y][x]:
                    edgeMagImg[y][x] = 0
    return edgeMagImg

def TwoThresholds(edgeMagImg, lowThreshold, highThreshold):
    lowThresh = lowThreshold
    highThresh = highThreshold
    #print(edgeMagImg)
    array = np.copy(edgeMagImg)
    edgeCategoryList = []

    for y in range(len(edgeMagImg)-1):
        edgeList = []
        for x in range(len(edgeMagImg[y])-1):
            categoryList = []
            if edgeMagImg[y][x] > highThresh:
                category = "HIGH"
                array[y][x] = 1
            elif (lowThresh <= edgeMagImg[y][x] or edgeMagImg[y][x] <= highThresh):
                category = "WEAK"
                array[y][x] = 0.5
            elif edgeMagImg[y][x] < lowThresh:
                edgeMagImg[y][x] = 0
                category = "DISCARD"
                array[y][x] = 0
                categoryList.append(category)
            edgeList.append(categoryList)
    edgeCategoryList.append(edgeList)
    edgeCategoryList = np.array(edgeCategoryList)
    return edgeMagImg, array

def EdgeTracking(thresholdedImg):
    for y in range(len(thresholdedImg[0])-1):
        for x in range(len(thresholdedImg[0][y])-1):
            if (0 <= y < thresholdedImg[1].shape[0] - 1 and 0 <= x < thresholdedImg[1].shape[1] - 1):
                if thresholdedImg[1][y][x] == 0.5:
                    if (thresholdedImg[1][y][x-1] == 1 or
                    thresholdedImg[1][y][x+1] == 1 or
                    thresholdedImg[1][y-1][x] == 1 or
                    thresholdedImg[1][y+1][x] == 1 or
                    thresholdedImg[1][y-1][x+1] == 1 or
                    thresholdedImg[1][y-1][x-1] == 1 or
                    thresholdedImg[1][y+1][x-1] == 1 or
                    thresholdedImg[1][y+1][x+1] == 1):
                        continue
                    else:
                        thresholdedImg[0][y][x] = 0

    return thresholdedImg[0]

def Canny(image, lowThreshold, highThreshold):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
    else:
        gray = img
        gray = np.float32(gray)
    gaussianblur = cv2.GaussianBlur(gray, (5,5), 0)
    sobelX = SobelEdgeDetectorX(gray)
    sobelY = SobelEdgeDetectorY(gray)
    edgeMagnitude = EdgeMagnitude(sobelX, sobelY)
    edgeOrientation = EdgeOrientation(sobelY, sobelX)
    nonMaxSupress = NonMaxSupress(edgeMagnitude, edgeOrientation)
    thresholded = TwoThresholds(nonMaxSupress, lowThreshold, (highThreshold))
    edgeTracking = EdgeTracking(thresholded)
    return edgeTracking

def myHoughLines(image, rho, theta, threshold):
    #newimg = image.copy()
    #newimg[img < threshold] = 0
    thetas = np.deg2rad(np.arange(theta, dtype='float32'))
    rhos = np.arange(rho, dtype='float32')
    #votes = np.zeros((2 * rho, theta),dtype='uint8')
    votes = {}
    height = image.shape[0]
    width = image.shape[1]

    for x in range(height):
        for y in range(width):
            for thetaVal in range(len(thetas)):
                rhoVal = x * math.cos(thetas[thetaVal]) + y * math.sin(thetas[thetaVal])
                if votes.get((rhoVal, thetas[thetaVal])):
                    votes[(rhoVal, thetas[thetaVal])] += 1
                else:
                    votes[(rhoVal, thetas[thetaVal])] = 1

    linesList =  []

    for rhoAndTheta, vote in votes.items():
        rhoThetaList = []
        if vote > threshold:
            #voteList = []
            #voteList.append(np.argwhere(vote))
            #votelist.append(np.argwhere(eachPixel))
            rhoVal = rhoAndTheta[0]
            thetaVal = rhoAndTheta[1]
            rhoThetaList.append(rhoVal)
            rhoThetaList.append(thetaVal)
            #rhoThetaList.append(voteList)
            linesList.append(rhoThetaList)

                #linesList.append((votes[np.nonzero(eachPixel)], eachPixel[np.nonzero(vote)]))
                #lines = votes[np.nonzero(eachPixel)][np.nonzero(vote)]
    lines = np.array(linesList)

    return lines

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import cv2

    test_image = cv2.imread('../images/project3/demo.pgm')
    gaussianblur = cv2.GaussianBlur(test_image, (5,5), 0)
    sobelX = SobelEdgeDetectorX(test_image)
    sobelY = SobelEdgeDetectorY(test_image)
    edgeMagnitude = EdgeMagnitude(sobelX, sobelY)
    edgeOrientation = EdgeOrientation(sobelY, sobelX)
    nonMaxSupress = NonMaxSupress(edgeMagnitude, edgeOrientation)
    thresholded = TwoThresholds(nonMaxSupress, 100, (100*1.5))
    edgeTracking = EdgeTracking(thresholded)
    plt.imshow((edgeTracking).astype(np.uint8))
    plt.show()
