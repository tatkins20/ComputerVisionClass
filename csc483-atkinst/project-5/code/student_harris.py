import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


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
    #img = img.astype('uint8')

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
    else:
        gray = img
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
    #rList.append(r)

    #rArray = np.array(rList)
    rArray = np.reshape(r, gray.shape)
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
    #print(len(rList))
    xList = []
    yList = []
    for i in range(1, len(rList)):
        x, y, confidence = rList[i]
        xList.append(x)
        yList.append(y)
    #print(rList)
    xList = np.array(xList)
    yList = np.array(yList)
    #distances = []
    radii = []
    xRadii = []
    yRadii = []
    #rListCopy = rList[1:]
    unsupX, unsupY, unsupConfidence = rList[0]
    xRadii.append(unsupX)
    yRadii.append(unsupY)
    for i in range(1, len(rList)-1):
        #distances = []
        index = i-1
        distances = np.sqrt((xList[i] - xList[:i]) ** 2 + (yList[i] - yList[:i]) ** 2)
        #while (index >=0):
            #newX, newY, newConfidence = rList[index]
            #distance = math.sqrt((newX - x) ** 2 + (newY - y) ** 2)
            #distances.append(distance)
            #index = index-1

        minDist = np.argmin(distances)
        radii.append((xList[i], yList[i], distances[minDist]))

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

def get_interest_points(image, feature_width):
    """

    JR adds: to ensure compatability with project 4A, you simply need to use
    this function as a wrapper for your 4A code.  Guidelines below left
    for historical reference purposes.

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    k = 0.04
    Responses = HarrisDetector(image,k)
    #print(Responses)
    #print(img[Responses>0.01*Responses.max()])
    #print(Responses)
    #img[Responses>0.01*Responses.max()]=[255,0,0]
    x, y = SuppressNonMax(Responses,feature_width)
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################

    #raise NotImplementedError('`get_interest_points` function in ' +
    #'`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    #raise NotImplementedError('adaptive non-maximal suppression in ' +
    #'`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y

if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import *
    from student_feature_matching import match_features
    from student_sift import get_features
    from student_harris import get_interest_points
    from IPython.core.debugger import set_trace

    # Notre Dame
    image1 = load_image('../data/Notre Dame/921919841_a30df938f2_o.jpg')
    image2 = load_image('../data/Notre Dame/4191453057_c86028ce1f_o.jpg')
    eval_file = '../data/Notre Dame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.pkl'

    # # Mount Rushmore -- this pair is relatively easy (still harder than Notre Dame, though)
    # image1 = load_image('../data/Mount Rushmore/9021235130_7c2acd9554_o.jpg')
    # image2 = load_image('../data/Mount Rushmore/9318872612_a255c874fb_o.jpg')
    # eval_file = '../data/Mount Rushmore/9021235130_7c2acd9554_o_to_9318872612_a255c874fb_o.pkl'

    # # Episcopal Gaudi -- This pair is relatively difficult
    # image1 = load_image('../data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg')
    # image2 = load_image('../data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg')
    # eval_file = '../data/Episcopal Gaudi/4386465943_8cf9776378_o_to_3743214471_1b5bbfda98_o.pkl'


    scale_factor = 0.5
    image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
    image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)
    image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    feature_width = 16 # width and height of each local feature, in pixels.
    x1, y1 = get_interest_points(image1_bw, 100)
    x2, y2 = get_interest_points(image2_bw, 100)
