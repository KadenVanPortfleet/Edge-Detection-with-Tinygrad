import os
os.environ["DEBUG"] = "2"
import sys
import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.helpers import DEBUG, flatten
from tinygrad import nn 
from tinygrad.tensor import dtypes
import numpy as np
from scipy import signal
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import cm
np.set_printoptions(threshold=sys.maxsize)
here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'crowdhd.jpg')

if __name__ == "__main__":
    #Sobel Operator kernel
    kernel = Tensor(
        [[[[-1,0,1],
           [-2,0,2],
           [-1,0,1]]]]
    )

    #Create Random 1x1x3x3 Matrix
    eyeD = Tensor.randn(1,1,3,3)
    #eyeD = eyeD.reshape(1,1,3,3)

    img = Image.open(filename).convert('L') #<- The convert part makes it grey scale (easier to work with [2D Matrix instead of 3D])
    #Convert imported image to numpy array with float16 precision.
    img_array = np.array(img, np.float16)
    #Print out img dimensions for user to see/debug.
    imgHeight = img_array.shape[0]
    imgWidth = img_array.shape[1]
    print(f'Image Width: {imgWidth}\n Image Height: {imgHeight}')

    #Convert numpy array to Tinygrad tensor for computation on GPU using OpenCL
    img_tinyT = Tensor(img_array, requires_grad=False)
    #Reshape necessary to perform convolution with library.
    img_tinyT = img_tinyT.reshape(1,1,imgHeight,imgWidth)
    #Convert greyscale values 0-255 to decimal float equivalents.
    img_tinyT /= 255
    #Print value (pixel) at 200,200 for debugging.
    print(img_tinyT[0,0,200,200].numpy())

    #perform convolution with kernel defined at the top. This specific kernel is for right-edge detection. No bias, stride = 1
    img_tinyT = img_tinyT.conv2d(kernel, None, 1, 1, 1, 1)
    
    #perform sigmoid
    img_tinyT = img_tinyT.sigmoid()

    #Reshape back to 2D tensor to convert back to numpy array
    img_tinyT = img_tinyT.reshape(imgHeight,imgWidth)

    #Normalize values between 0 and 1
    img_tinyT = (img_tinyT-0.5)*2

    #perform relu and multiple to 0-255 range.
    img_tinyT = img_tinyT.relu()*255
    #img_tinyT = img_tinyT*255

    #Round to nearest whole number for pixel output.
    img_outArray = img_tinyT.numpy().round()

    #show resulting array
    print(f'array: \n{img_outArray[200]}\n')

    #Display output image.
    im = Image.fromarray(np.uint8(img_outArray) , 'L')
    im.show()