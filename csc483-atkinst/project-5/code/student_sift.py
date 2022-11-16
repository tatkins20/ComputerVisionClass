import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def SobelEdgeDetectorX(image):
    gaussianblur = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
    #gaussianimg = cv2.filter2D(image, cv2.CV_32F, gaussianfilter)
    kernelx = np.asarray([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]], dtype = 'float32')
    sobeledgeimg = cv2.filter2D(gaussianblur, cv2.CV_32F, kernelx)
    return sobeledgeimg

def SobelEdgeDetectorY(image):
    gaussianblur = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
    #gaussianimg = cv2.filter2D(image, cv2.CV_32F, gaussianfilter)
    kernely = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = 'float32')
    sobeledgeimg = cv2.filter2D(gaussianblur, cv2.CV_32F, kernely)
    return sobeledgeimg

def EdgeMagnitude(vertEdge, horzEdge):
    magnitude = np.hypot(vertEdge, horzEdge)
    return magnitude

def EdgeOrientation(vertEdge, horzEdge):
    orientation = np.arctan2(vertEdge, horzEdge)
    return orientation

def get_features(image, x, y, feature_width, scales=None):
    """
    JR Writes: To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    maximal points you may need to implement a more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    scale  - laplascian pyramid - in SIFT paper
    Below for advanced implementation:

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    if len(image.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = np.float32(gray)
    else:
        image = np.float32(image)
    assert image.ndim == 2, 'Image must be grayscale'
    fv = []
    numBins = 8
    numSubSection = 4
    binWidth = 360//numBins
    subSectionWidth = feature_width//numSubSection
    # Creating 8 orientation bins
    #orientBins = np.arange(0, 360, 40)
    #print(orientBins)
    # Creating 16x16 neighborhood around keypoints
    xKey = 0
    yKey = 0
    #sixteenBySixteen = [[] for feature in range(feature_width)]
    for index in range(len(x)):
        #sixteenBySixteen = []
        #sixteenBySixteen = np.empty((16,16))
        #print(sixteenBySixteen)
        #print(x[index], y[index])
        xKey = x[index]
        yKey = y[index]
        top, left = max(0, yKey-feature_width//2), max(0, xKey-feature_width//2)
        bottom, right = min(image.shape[0], yKey+feature_width//2), min(image.shape[1], xKey+feature_width//2)
        #sixteenBySixteen[top:bottom][left:right] = image[top:bottom, left:right]
        #print(sixteenBySixteen)
        patch = np.asarray(image[top:bottom, left:right])
        patch = cv2.GaussianBlur(patch, (5,5), 0)
        #print(patch)
        #patch = np.reshape(patch, (16,16))
        # xLeftBound = xKey - 8
        # xRightBound = xKey + 8
        # yBottomBound = yKey + 8
        # yTopBound = yKey - 8
        # verticalBound = yTopBound
        # #get 16x16 window
        # while verticalBound < yBottomBound:
        #     horizontalBound = xLeftBound
        #     while horizontalBound < xRightBound:
        #         sixteenBySixteen = np.append(sixteenBySixteen, image[verticalBound][horizontalBound])
        #         horizontalBound = horizontalBound + 1
        #     verticalBound = verticalBound +1
    # use project 3 gradient code to get the gradient using sobel filters
        windowedXEdges = cv2.Sobel(patch.astype(np.float32),cv2.CV_32F,1,0,ksize=5)
        windowedYEdges  = cv2.Sobel(patch.astype(np.float32),cv2.CV_32F,0,1,ksize=5)
        windowedMagnitude = EdgeMagnitude(windowedXEdges, windowedYEdges)
        windowedOrientation = EdgeOrientation(windowedXEdges, windowedYEdges)
        windowedOrientation = np.rad2deg(windowedOrientation)
        #print(windowedOrientation)
        featvec = np.zeros(numBins * numSubSection ** 2, dtype=np.float32)
    # use histogram on 4x4 windows of that gradient to give us feature vectors
        #print(patch)
        for i in range(0, subSectionWidth):
            for j in range(0, subSectionWidth):
                subtop, subleft = i * subSectionWidth, j * subSectionWidth
                subbottom, subright = min(image.shape[0], (i+1)*subSectionWidth), min(image.shape[1], (j+1)*subSectionWidth)
                hist, binEdges = np.histogram(windowedOrientation[subtop:subbottom, subleft:subright], 8, (-180, 180))
                featvec[i*subSectionWidth*numBins + j*numBins:i*subSectionWidth*numBins + (j+1)*numBins] = hist.flatten()
        #print(np.unique(windowedOrientation))
        featvec /= max(1e-6, np.linalg.norm(featvec))
        featvec[featvec>0.2] = 0.2
        featvec /= max(1e-6, np.linalg.norm(featvec))
        fv.append(featvec)


    #raise NotImplementedError('`get_features` function in ' +
        #'`student_sift.py` needs to be implemented')
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return np.array(fv)**.8

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
## Create feature vectors at each interest point (Szeliski 4.1.2)
    image1_features = get_features(image1_bw, x1, y1, feature_width, 1)
    image2_features = get_features(image2_bw, x2, y2, feature_width, 1)
