import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import numpy
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float


def Get_Maximum(x , size):
      
        image_max = ndi.maximum_filter(x, size, mode='constant')
        # Comparison between image_max and im to find the coordinates of local maxima
        xy = peak_local_max (x, min_distance=65)
        return xy
    
