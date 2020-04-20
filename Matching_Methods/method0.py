import numpy as np
from scipy.signal import correlate2d
#import itk
from skimage import io
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os.path import isfile , join
from PIL import Image
#from cvutils import rgb2gray
from skimage.color import rgb2gray
import cv2
#from skimage.color import rgb2gray
import Feature_ext
import importlib
importlib.reload(Feature_ext)


#______Method 0: Direct 2D correlation of the image with the template

def match_template_corr( x , temp ):
    y = np.empty(x.shape)
    y = correlate2d(x,temp,'same')
    return y


def image():
    
    imgs_dir1 = 'images'
    imgs_names1 = ['cells2.jpg']
    imgs_fnames1 = [ join( imgs_dir1, img_name1) for img_name1 in imgs_names1 ]
    imgs_rgb1 = [ np.array(Image.open(img1)) for img1 in imgs_fnames1 ]
    imgs_gray = [ rgb2gray( img1 ) for img1 in imgs_rgb1 ]
    
    imgs_dir = 'images'
    imgs_names = ['cells_templete.jpg']
    imgs_fnames = [ join( imgs_dir, img_name) for img_name in imgs_names ]
    imgs_rgb = [ np.array(Image.open(img)) for img in imgs_fnames ]
    Templete_gray = [ rgb2gray( img ) for img in imgs_rgb ]
    
    matches_corr = [ match_template_corr(x,h) for (x,h) in zip(imgs_gray,Templete_gray)]
    matches_corr_maxima = [ Feature_ext.Get_Maximum(x,min(t.shape)//8) for (x,t) in zip(matches_corr,Templete_gray)]
    patches = zip(imgs_gray,Templete_gray,  matches_corr, matches_corr_maxima)
    return patches 

def method_0func(patches):
    for i,(im,temp,mcorr,pcorr) in enumerate(patches):
        def get_rect_on_maximum(y,template):
            ij = np.unravel_index(np.argmax(y), y.shape)
            x, y = ij[::-1]
            # highlight matched region
            htemp, wtemp = template.shape
            rect = plt.Rectangle((x-wtemp/2, y-htemp/2), wtemp, htemp, edgecolor='r', facecolor='none')
            return rect,x,y
        
        def make_rects(plt_object,xy,template):
            htemp, wtemp = template.shape
            for ridx in range(xy.shape[0]):
                y,x = xy[ridx]
                r =  plt.Rectangle((x-wtemp/2, y-htemp/2), wtemp, htemp, edgecolor='g', facecolor='none')
                plt_object.add_patch(r)
        
        def make_circles(plt_object,xy,template):
            htemp, wtemp = template.shape
            for ridx in range(xy.shape[0]):
                y,x = xy[ridx]
                plt_object.plot(x, y, 'o', markeredgecolor='g', markerfacecolor='none', markersize=20)
        
        r,x,y = get_rect_on_maximum(mcorr,temp)#####get maximum to put rects
        #______________________show matching space_____________________            
        fig1, ax1 = plt.subplots(figsize = (5, 10))
        plt.autoscale(True)
        
        ax1.imshow(mcorr, cmap=plt.get_cmap('gray'))
        matching_space_image=make_circles(ax1, pcorr,temp)####put circles on image
        ax1.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=20)
        
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('matching_space.jpg', dpi=900, bbox_inches='tight',pad_inches=0)
        plt.show(matching_space_image) 
        
        #_______________________show detected patterns___________________-
        fig2, ax2 = plt.subplots(figsize = (5, 10))
        plt.autoscale(True)    
        
        ax2.imshow(im, cmap=plt.get_cmap('gray'))
        make_rects( ax2 , pcorr, temp )
        detected_patterns_image=ax2.add_patch(r)  ####put rects on image
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('template_matching.jpg', dpi=900, bbox_inches='tight',pad_inches=0)
        plt.show(detected_patterns_image)  
        
    
#patches=image()
#method_0func(patches)