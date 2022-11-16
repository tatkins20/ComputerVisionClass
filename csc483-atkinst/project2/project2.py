import cv2
import numpy as np
from utils import load_image, save_image

#def convolve(img, kernel, x, y):
    

def my_imfilter(image, filter):
    
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can finish grading
   before the heat death of the universe.
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  #print("filter looks like: ",'\n', filter, '\n',
  #"----------------------------", '\n',
  #"image looks like: ",'\n', image)
                     
  redChannel = image[:,:,0]
  greenChannel = image[:,:,1]
  blueChannel = image[:,:,2]
  #print(redChannel)
  imageChannelHeight = image[:,:,0].shape[0]
  imageChannelWidth = image[:,:,0].shape[1]
  imageHeight = image.shape[0]
  imageWidth = image.shape[1]
  imageRGBnumbers = image.shape[2]
  imageNumPixels = image.shape[2]
  filterHeight = filter.shape[1]
  filterWidth = filter.shape[0]
  #print(filter.shape[0]//2,filter.shape[1]//2)  
  #print(len(image.shape))
  #print(filter.shape)
  #print(image.shape)

  convolutedImg = np.copy(image)
  h = filterHeight//2
  w = filterWidth//2
  for x in range(h, imageHeight-h):
    for y in range(w, imageWidth-w):
        cumulation = 0
        for m in range(filterHeight):
            for n in range(filterWidth):
                xOffset = int(abs(x-h*m))
                yOffset = int(abs(y-w*n))
                if abs(x-h*m) >= imageHeight or abs(y-w*n) >= imageWidth:
                    #padShape = np.zeros((abs(image.shape[0] + abs(image.shape[0] - xOffset)), 
                    #abs(image.shape[1] + abs(image.shape[1] - yOffset)), 
                    #image.shape[2]),dtype='uint8')
                    #padShape[0:image.shape[0],0:image.shape[1]] = image
                    #convolutedImg = padShape
                    convolutedImg = np.pad(image,((h,w),(h,w),(h,w)),'constant')
                    #print(paddedImg)
                    cumulation = cumulation + np.dot(filter[n][m],convolutedImg[x-h*m][y-w*n])
                    #convolutedImg = convolutedImg.copy(paddedImg)
                    convolutedImg[x-h*m][y-w*n] = cumulation
                else:
                    cumulation = cumulation + np.dot(filter[n][m],image[x-h*m][y-w*n])
                    convolutedImg[x-h*m][y-w*n] = cumulation
                #else:
                    #cumulation = cumulation + filter[m] * image[x-h*m]
                    #convolutedImg[x-h*m] = cumulation
                    
  filtered_image = convolutedImg

  #raise NotImplementedError('#`my_imfilter` function in `project2.py` ' + 'needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###

  raise NotImplementedError('`create_hybrid_image` function in ' +
    '`student_code.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image

if __name__ == "__main__" :
    test_image = load_image('../project2/images/cat.bmp')
    test_image = cv2.resize(test_image, (0, 0), fx=0.7, fy=0.7)
    identity_filter = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    identity_image = my_imfilter(test_image, identity_filter)
    #plt.imshow(identity_image)
    #done = save_image('../results/identity_image.jpg', identity_image)
