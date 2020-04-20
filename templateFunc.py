import numpy as np
from scipy.signal import correlate2d
#import itk
from skimage import io
#from itkwidgets import view
#import itkwidgets
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

#______Method 1: Direct 2D correlation of the image with the zero-mean template

def match_template_corr_zmean( x , temp ):
    return match_template_corr(x , temp - temp.mean())

##_______Method 2: SSD

def match_template_ssd( x , temp ):
    term1 = np.sum( np.square( temp ))
    term2 = -2*correlate2d(x, temp,'same')
    term3 = correlate2d( np.square( x ), np.ones(temp.shape),'same' )
    ssd = np.maximum( term1 + term2 + term3 , 0 )
    return 1 - np.sqrt(ssd)

#_____Method 3: Normalized cross-correlation
    
def match_template_xcorr( f , t ):
    f_c = f - correlate2d( f , np.ones(t.shape)/np.prod(t.shape), 'same') 
    t_c = t - t.mean()
    numerator = correlate2d( f_c , t_c , 'same' )
    d1 = correlate2d( np.square(f_c) , np.ones(t.shape), 'same')
    d2 = np.sum( np.square( t_c ))
    denumerator = np.sqrt( np.maximum( d1 * d2 , 0 )) # to avoid sqrt of negative
    response = np.zeros( f.shape )
    valid = denumerator > np.finfo(np.float32).eps # mask to avoid division by zero
    response[valid] = numerator[valid]/denumerator[valid]
    return response


def draw_match_0(patches):
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
        
    
def draw_match_1(patches):
    for i,(im,temp,mcorrz,pcorrz) in enumerate(patches):
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
        
        r,x,y = get_rect_on_maximum(mcorrz,temp)#####get maximum to put rects
        #______________________show matching space_____________________            
        fig1, ax1 = plt.subplots(figsize = (5, 10))
        plt.autoscale(True)
        
        ax1.imshow(mcorrz, cmap=plt.get_cmap('gray'))
        matching_space_image=make_circles(ax1, pcorrz,temp)####put circles on image
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
        make_rects( ax2 , pcorrz, temp )
        detected_patterns_image=ax2.add_patch(r)  ####put rects on image
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('template_matching.jpg', dpi=900, bbox_inches='tight',pad_inches=0)
        plt.show(detected_patterns_image)  
        
        
def draw_match_2(patches):
    for i,(im,temp,mssd,pssd) in enumerate(patches):
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
        
        r,x,y = get_rect_on_maximum(mssd,temp)#####get maximum to put rects
        #______________________show matching space_____________________            
        fig1, ax1 = plt.subplots(figsize = (5, 10))
        plt.autoscale(True)
        
        ax1.imshow(mssd, cmap=plt.get_cmap('gray'))
        matching_space_image=make_circles(ax1, pssd,temp)####put circles on image
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
        make_rects( ax2 , pssd, temp )
        detected_patterns_image=ax2.add_patch(r)  ####put rects on image
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('template_matching.jpg', dpi=900, bbox_inches='tight',pad_inches=0)
        plt.show(detected_patterns_image)  
        
def draw_match_3(patches):
    for i,(im,temp,mxcorr,pxcorr) in enumerate(patches):
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
        
        r,x,y = get_rect_on_maximum(mxcorr,temp)#####get maximum to put rects
        #______________________show matching space_____________________            
        fig1, ax1 = plt.subplots(figsize = (5, 10))
        plt.autoscale(True)
        
        ax1.imshow(mxcorr, cmap=plt.get_cmap('gray'))
        matching_space_image=make_circles(ax1, pxcorr,temp)####put circles on image
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
        make_rects( ax2 , pxcorr, temp )
        detected_patterns_image=ax2.add_patch(r)  ####put rects on image
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('template_matching.jpg', dpi=900, bbox_inches='tight',pad_inches=0)
        plt.show(detected_patterns_image)  
                        
   
         
