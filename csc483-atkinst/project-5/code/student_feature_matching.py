import numpy as np
from math import *
import operator

def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:finalProject
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################

    #raise NotImplementedError('`match_features` function in ' +
        #'`student_feature_matching.py` needs to be implemented')
    f1matches = []
    f2matches = []
    confidences = []
    THRESHOLD = 1
    featdistances = []
    for ft1_index, ft1 in enumerate(features1):
        discriptordistances = []
        ft1index = ft1_index
        for ft2_index, ft2 in enumerate(features2):
            squares = []
            #difference = np.subtract(ft1,ft2)
            #square = np.square(difference)
            #summation = np.sum(square)
            #distance = np.sqrt(summation)
            for F1_d in range (len(features1[ft1_index])):
                square = (features1[ft1_index][F1_d] - features2[ft2_index][F1_d]) ** 2
                squares.append(square)
                summation = sum(squares)
                distance = sqrt(summation)
            ft2index = ft2_index
            discriptordistances.append((ft1index, ft2index, distance))
            #featdistances.append(distance3)
        sorteddistances = sorted(discriptordistances, key=lambda tup: tup[2])
        #print(sorteddistances)
        #sorteddistances = sorted(featdistances)
        #discriptordistances2 = np.asarray(discriptordistances)
        ft1i, ft2i, nrstneighbor = sorteddistances[0]
        ft1i2, ft2i2, scndnrstneighbor = sorteddistances[1]
        nearestNeighbor = nrstneighbor
        sndNearestNeighbor = scndnrstneighbor
        #print(nearestNeighbor, sndNearestNeighbor)
        if sndNearestNeighbor>0:
            NNDR = nearestNeighbor/sndNearestNeighbor
            if(NNDR < THRESHOLD):
                f1matches.append(ft1i)
                f2matches.append(ft2i)
                confidences.append(NNDR)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    f1matches = f1matches[0:100]
    f2matches = f2matches[0:100]
    matches = np.column_stack((f1matches, f2matches))
    #matches = matches.reshape(len(x1), 2)
    print(matches)
    confidences = np.array(confidences)
    print(confidences)
    return matches, confidences
