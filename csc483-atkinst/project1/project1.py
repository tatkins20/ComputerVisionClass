#project1.py
import numpy
import matplotlib.pyplot as plt
import cv2

def loadppm(filename):
    '''Given a filename, return a numpy array containing the ppm image
    input: a filename to a valid ascii ppm file
    output: a properly formatted 3d numpy array containing a separate 2d array
            for each color
    notes: be sure you test for the correct P3 header and use the dimensions and depth
            data from the header
            your code should also discard comment lines that begin with #
    '''
    numColors = 3
    file = open(filename)
    pThree = file.readline().splitlines()
    if pThree[0] != "P3":
        return "not correct P3 header"
    height,width= file.readline().split()
    h = int(height)
    w = int(width)
    max_color = file.readline().splitlines()
    data = file.read()
    values = data.split()

    redValues = [values[i] for i in range(0, len(values), numColors)]
    greenValues = [values[i+1] for i in range(0, len(values), numColors)]
    blueValues = [values[i+2] for i in range(0, len(values), numColors)]

    red = []
    green = []
    blue = []
    numValues = 0
    for i in range(0, w):
        redRow = []
        blueRow = []
        greenRow = []
        for j in range(0, h):
            numValues += 1
            redRow.append(redValues[numValues-1])
            greenRow.append(greenValues[numValues-1])
            blueRow.append(blueValues[numValues-1])
        red.append(redRow)
        green.append(greenRow)
        blue.append(blueRow)

    redArray = numpy.array(red, dtype='uint8')
    greenArray = numpy.array(green, dtype='uint8')
    blueArray = numpy.array(blue,dtype='uint8')
    threeDimArray = numpy.dstack((redArray, greenArray, blueArray))
    return threeDimArray


def GetGreenPixels(img):
    '''given a numpy 3d array containing an image, return the green channel'''
    return img[:,:,1]

def GetBluePixels(img):
    '''given a numpy 3d array containing an image, return the blue channel'''
    return img[:,:,2]

def GetRedPixels(img):
    '''given a numpy 3d array containing an image, return the red channel'''
    return img[:,:,0]

def convertToGreyscale(img):
    '''given a numpy 3d array containing an image, return grayscale image'''
    splitArray = numpy.dsplit(img,1)
    multiDimArray = numpy.array(splitArray, dtype = 'uint8')
    grayList = []
    #print(multiDimArray)
    w = len(multiDimArray[0])
    h = len(multiDimArray[0][0])

    for row in range (len(multiDimArray[0])):
        for i in range(len(multiDimArray[0][row])):
            pixelSum = 0
            grayTempList = []
            for j in range(len(multiDimArray[0][row][0])):
                pixelSum += multiDimArray[0][row][i][j]
            pixelAvg = int(pixelSum/3)
            grayTempList.append(pixelAvg)
            grayTempList.append(pixelAvg)
            grayTempList.append(pixelAvg)
            grayList.append(grayTempList)
    #print(grayTempList)
    #print(grayList)

    gray = []
    numValues = 0
    for i in range(0, w):
        grayRow = []
        for j in range(0, h):
            numValues += 1
            grayRow.append(grayList[numValues-1])
        gray.append(grayRow)
    grayArray = numpy.array(gray,dtype='uint8')
    #print(grayArray)
    return grayArray

def thresholdingImg(img):
    grayImg = convertToGreyscale(img)
    thresholdList = []

    w = len(grayImg[0])
    h = len(grayImg)

    for row in range(h):
        for i in range(w):
            threshList = []
            for j in range(len(grayImg[row][i])):
                if (int(grayImg[row][i][j]) < 128):
                    pixelValue = 0
                    threshList.append(pixelValue)
                else:
                    pixelValue = 255
                    threshList.append(pixelValue)
            thresholdList.append(threshList)

    #print(thresholdList)
    threshold = []
    numValues = 0
    for i in range(0, h):
        threshRow = []
        for j in range(0, w):
            numValues += 1
            threshRow.append(thresholdList[numValues-1])
        threshold.append(threshRow)
    threshArray = numpy.array(threshold,dtype='uint8')
    return threshArray


def histogramEqualization(img):
    grayImg = convertToGreyscale(img)
    h = len(grayImg)
    w = len(grayImg[0])
    rgbInfoNumLength = 3
    totalNumValues = h * w * rgbInfoNumLength
    MaxColorRange = 256
    n = 255

    colorValues = {}
    cdf = {}
    for value in range(MaxColorRange):
        colorValues.update({value:0})
    colorCountDict = colorValues.copy()
    allImgValues = numpy.ravel(grayImg)
    #print(allImgValues)
    for imgValue in allImgValues:
        if (imgValue in colorCountDict):
            colorCount = colorCountDict[imgValue] + 1
            colorCountDict.update({imgValue:colorCount})
            
    #print(colorCountDict)

    lastSeenValue = colorCountDict[0]
    for value in range(MaxColorRange):
        currentValue = colorCountDict[value] + lastSeenValue
        cdf.update({value:currentValue})
        lastSeenValue = currentValue
    #print(cdf)

    cdf_min = min(cdf, key=lambda k: cdf[k])

    histEqNums = []
    for row in grayImg:
        rowList = []
        for value in row:
            val = value[0]
            if cdf[val] == 0 or cdf[val] == (h*w):
                rowList.append(val)
            else:
                histEq = int(((cdf[val] - cdf_min)/((h*w) - cdf_min)) * (MaxColorRange - 1))
                rowList.append(histEq)
            #print(rowList)
        histEqNums.append(rowList)
            
    #histEqNumsArray = numpy.asarray(histEqNums)
    #histEqNumsArray = numpy.reshape(histEqNumsArray, (h,w))
    #print(histEqNums)
    equalized = []
    numElements = 0
    for i in range(0, h-1):
        histEqRow = []
        for j in range(0, w-1):
            histEqRow.append(histEqNums[i][j])
            numElements += 1
        equalized.append(histEqRow)
    histEqArray = numpy.array(equalized,dtype='uint8')
    #print(histEqArray)
    return histEqArray

if __name__ == "__main__":
  #put any command-line testing code you want here.
   #note this code in this block will only run if you run the module from the command line
  # (i.e. type "python3 project1.py" at the command prompt)
  # or within your IDE
  # it will NOT run if you simply import your code into the python shell.

  #rgb = loadppm("../images/simple.ascii.ppm")
  #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis

  # uncomment the lines below to test
  #plt.imshow(rgb)
  #plt.show()

  #rgb = loadppm("../images/zebra.ascii.ppm")
  #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis

  #rgb = loadppm("../images/simple.ascii.ppm")
  #green = GetGreenPixels(rgb)

  #you know the routine
  #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  #plt.imshow(green,cmap='gray', vmin=0, vmax=255)
  #plt.show()

  #red = GetRedPixels(rgb)
  #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  #plt.imshow(red,cmap='gray', vmin=0, vmax=255)
  #plt.show()

  #blue = GetBluePixels(rgb)
  #plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
  #plt.imshow(blue,cmap='gray', vmin=0, vmax=255)
  #plt.show()

  #code to test greyscale conversions of the colored boxes and the zebra



  #code to create black/white monochrome image



  #code to create/test normalized greyscale image



  zebra = loadppm("../images/simple.ascii.ppm")
  histogram = histogramEqualization(zebra)
  plt.xticks([]), plt.yticks([])
  plt.imshow(histogram)
  plt.show()
