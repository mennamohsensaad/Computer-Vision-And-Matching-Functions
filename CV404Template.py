
"""
@author: Nashwa
"""
import numpy as np
from scipy.signal import correlate2d
from skimage import io
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os.path import isfile , join
from PIL import Image
from skimage.color import rgb2gray
import Feature_ext
import importlib
importlib.reload(Feature_ext)


####Input_Image
imgs_dir1 = 'images'
imgs_names1 = ['chess.jpg','cells2.png']
imgs_fnames1 = [ join( imgs_dir1, img_name1) for img_name1 in imgs_names1 ]
imgs_rgb1 = [ np.array(Image.open(img1)) for img1 in imgs_fnames1 ]
imgs_gray = [ rgb2gray( img1 ) for img1 in imgs_rgb1 ]

####Input_Template
imgs_dir = 'images'
imgs_names = ['chess_templete.jpg','cells_templete.png']
imgs_fnames = [ join( imgs_dir, img_name) for img_name in imgs_names ]
imgs_rgb = [ np.array(Image.open(img)) for img in imgs_fnames ]
Templete_gray = [ rgb2gray( img ) for img in imgs_rgb ]


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


matches_corr = [ match_template_corr(x,h) for (x,h) in zip(imgs_gray,Templete_gray)]
matches_corr_zmean = [ match_template_corr_zmean(x,h) for (x,h) in zip(imgs_gray,Templete_gray)]
matches_ssd = [ match_template_ssd(x,h) for (x,h) in zip(imgs_gray,Templete_gray)]
matches_xcorr = [ match_template_xcorr(x,h) for (x,h) in zip(imgs_gray,Templete_gray)]



matches_corr_maxima = [ Feature_ext.Get_Maximum(x,min(t.shape)//8) for (x,t) in zip(matches_corr,Templete_gray)]
matches_corr_zmean_maxima = [ Feature_ext.Get_Maximum(x,min(t.shape)//8) for (x,t) in zip(matches_corr_zmean,Templete_gray)]
matches_ssd_maxima = [ Feature_ext.Get_Maximum(x,min(t.shape)) for (x,t) in zip(matches_ssd,Templete_gray)]
matches_xcorr_maxima = [ Feature_ext.Get_Maximum(x,min(t.shape)//8) for (x,t) in zip(matches_xcorr,Templete_gray)]

methods_n = 4
patches = zip(imgs_gray,Templete_gray,
              matches_corr,matches_corr_zmean,matches_ssd,matches_xcorr,
             matches_corr_maxima,matches_corr_zmean_maxima,matches_ssd_maxima,matches_xcorr_maxima)


fig, ax = plt.subplots(len(imgs_gray)*(methods_n+1),2,figsize = (20, 40))
plt.autoscale(True)


for i,(im,temp,mcorr,mcorrz,mssd,mxcorr,pcorr,pcorrz,pssd,pxcorr) in enumerate(patches):
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
            plt_object.plot(x, y, 'o', markeredgecolor='g', markerfacecolor='none', markersize=10)
            

    row = (methods_n+1)*i 
    ax[row,0].imshow(im, cmap = 'gray')
    ax[row,1].imshow(temp, cmap = 'gray')
    
    ax[row+ 1,0].imshow(im, cmap = 'gray')
    r,x,y = get_rect_on_maximum(mcorr,temp)
    make_rects( ax[row + 1,0] , pcorr, temp )
    ax[row + 1,0].add_patch(r)
    ax[row + 1,1].imshow(mcorr, cmap = 'gray')
    make_circles(ax[row + 1,1], pcorr,temp)
    ax[row + 1,1].plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    
    r,x,y = get_rect_on_maximum(mcorrz,temp)
    ax[row + 2,0].imshow(im, cmap = 'gray')
    make_rects( ax[row + 2,0] , pcorrz, temp )
    ax[row + 2,0].add_patch(r)
    ax[row + 2,1].imshow(mcorrz, cmap = 'gray')
    make_circles(ax[row + 2,1], pcorrz,temp)
    ax[row + 2,1].plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    
    r,x,y = get_rect_on_maximum(mssd,temp)
    ax[row + 3,0].imshow(im, cmap = 'gray')
    make_rects( ax[row + 3,0] , pssd, temp )
    ax[row + 3,0].add_patch(r)
    ax[row + 3,1].imshow(mssd, cmap = 'gray')
    make_circles(ax[row + 3,1], pssd,temp)
    ax[row + 3,1].plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    
    r,x,y = get_rect_on_maximum(mxcorr,temp)
    ax[row + 4,0].imshow(im, cmap = 'gray')
    make_rects( ax[row + 4,0] , pxcorr, temp )
    ax[row + 4,0].add_patch(r)
    ax[row + 4,1].imshow(mxcorr, cmap = 'gray')
    make_circles(ax[row + 4,1], pxcorr,temp)
    ax[row + 4,1].plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()



