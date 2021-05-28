import numpy as np
import math
import glob
import os
import sys
import argparse
import cv2




def extract_strokes(image,fg_stroke_value=2,bg_stroke_value=1):
    if(type(image) is str):
        #if input is a filepath, read image into I
        I = cv2.imread(image,cv2.IMREAD_UNCHANGED)
    elif(type(image) is np.ndarray):
        #if input is a image, assign image to I
        I = image
    else:
        raise NameError('Failed to read image. Input must be a filename or a ndarray!')

    if(I.shape[2]==4):
        alpha = I[:,:,3] #extract alpha channel
    else:
        raise NameError('Failed to alpha channel. Pls make sure input image has 4 channels. I.shape={}'.format(I.shape))
    fg_Pixels = (alpha == fg_stroke_value)
    bg_Pixels = (alpha == bg_stroke_value)

    fg_mask = fg_mask=np.zeros(alpha.shape,dtype=np.uint8) + fg_Pixels
    fg_mask_invert = 1 - fg_mask
    dismap_fg = cv2.distanceTransform(fg_mask_invert, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    bg_mask=np.zeros(alpha.shape,dtype=np.uint8) + bg_Pixels
    bg_mask_invert = 1 - bg_mask
    dismap_bg = cv2.distanceTransform(bg_mask_invert, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    return dismap_fg, dismap_bg
