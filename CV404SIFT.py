from scipy.signal import convolve2d,gaussian
from skimage.transform import rescale
from skimage.transform import resize
import matplotlib.cm as cm
from os.path import isfile , join
from PIL import Image
import numpy as np
from scipy import signal
from math import sqrt
import logging
import sys
import matplotlib.pyplot as plt
#import itertools
#import random
import cv2
from math import sin, cos
import glob
from skimage.transform import rotate


class SIFT(object):
    def __init__(self,imgs_dir):
	    self.imgs_dir=imgs_dir
        
    def pipeline(self,input_img ):
                if self.iteration==0:
                   self.current_image="wholeImage"
                elif self.iteration==1:
                   self.current_image="pattern1"
                else :
                   self.current_image="pattern2"   
                print("finding initalize parmeter for "+self.current_image+"....")
                self.initalize_par()
                self.im=input_img
                img_max = input_img.max()
                print("finding image dog for "+self.current_image+"....")
                self.img_dogs, self.img_octaves = self.image_dog(input_img)
                #self.visualze_scale_dog()
                print("finding dog keypoints for "+self.current_image+"....")
                keypoints = self.dog_keypoints( self.img_dogs , img_max , 0.03 )
                # Each keypoint has four parameters (i,j,scale_idx,orientation)
                print("finding dog keypoints orientations for "+self.current_image+"....")
                self.keypoints_ijso = self.dog_keypoints_orientations( self.img_octaves , keypoints , 36 )
                print("extracting sift descriptors128 for "+self.current_image+"....")
                #self.visualize_points_orirntation()
                points,descriptors = self.extract_sift_descriptors128(self.img_octaves , self.keypoints_ijso , 8)
                print("Ending extract sift descriptors128 for "+self.current_image+"....")
                print("go to matching for "+self.current_image+"...")
                self.iteration=self.iteration+1
                return points, descriptors    

    def initalize_par(self):               
                logging.basicConfig(stream=sys.stdout, level=logging.INFO)
                self.logger = logging.getLogger('SIFT')
                # The following are suggested by SIFT author
                self.N_OCTAVES = 4 
                self.N_SCALES = 5 
                self.SIGMA = 1.6
                self.K = sqrt(2)
                #five different blurred images (sigma, k*sigma, k^2 *sigma, k^3 *sigma, k^4 *sigma)
                SIGMA_SEQ = lambda s: [ (self.K**i)*s for i in range(self.N_SCALES) ] # (s, √2s , 2s, 2√2 s , 4s ) # i is the counter for power of k
                self.SIGMA_SIFT = SIGMA_SEQ(self.SIGMA)  # list of scales 
                self.KERNEL_RADIUS = lambda s : 2 * int(round(s))
                self.KERNELS_SIFT = [self.gaussian_kernel2d(kernlen = 2 * self.KERNEL_RADIUS(s) + 1,std = s) 
                                for s in self.SIGMA_SIFT ]
                """fig, ax = plt.subplots(1,len(KERNELS_SIFT),figsize = (15, 10))
                for i in range(len(KERNELS_SIFT)):
                    ax[i].imshow(KERNELS_SIFT[i])"""

###########################################STEP1################################################################
    def rgb2gray(self,rgb_image):
            return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
        
    def gaussian_kernel1d(self,kernlen,std):
            """Returns a 1D Gaussian window."""
            kernel1d = signal.gaussian(kernlen, std=std)
            kernel1d = kernel1d.reshape(kernlen, 1)
            return kernel1d / kernel1d.sum()
        
    def gaussian_kernel2d(self,kernlen,std):
            """Returns a 2D Gaussian kernel array."""
            gkern1d =self.gaussian_kernel1d(kernlen,std)
            gkern2d = np.outer(gkern1d, gkern1d)
            return gkern2d
    def sift_resize(self,img, ratio = None):
            ratio = ratio if ratio is not None else np.sqrt((1024*1024) / np.prod(img.shape[:2]))
            newshape = list(map( lambda d : int(round(d*ratio)), img.shape[:2])) 
            img = resize( img, newshape , anti_aliasing = True )
            return img,ratio
        
    def image_dog(self, img):
            octaves = []
            dog = []
            base = rescale( img, 2, anti_aliasing=False) 
            octaves.append([ convolve2d( base , kernel , 'same', 'symm') 
                            for kernel in self.KERNELS_SIFT ])
            dog.append([ s2 - s1 
                        for (s1,s2) in zip( octaves[0][:-1], octaves[0][1:])])
            for i in range(1,self.N_OCTAVES):
                base = octaves[i-1][2][::2,::2] # 2x subsampling 
                octaves.append([base] + [convolve2d( base , kernel , 'same', 'symm') 
                                         for kernel in self.KERNELS_SIFT[1:] ])
                dog.append([ s2 - s1 
                            for (s1,s2) in zip( octaves[i][:-1], octaves[i][1:])])
                """
                self.logger.info('Done {}/{} octaves'.format(i+1, N_OCTAVES))
                """
            return dog , octaves
        
        
    def visualze_scale_dog(self):
               
                ####show image with scales and guassian
                plt.imshow(self.img); plt.axis('off')
                plt.show()
                fig, ax = plt.subplots(self.N_OCTAVES,self.N_SCALES,figsize = (15, 10))
                for octave_idx in range(self.N_OCTAVES):
                    img_octave = self.img_octaves[octave_idx]
                    for scale_idx in range(self.N_SCALES):
                        subplot = ax[octave_idx,scale_idx]
                        subplot.axis('off')
                        img_scale = img_octave[scale_idx]
                        subplot.set_title('Image on (Oct={},S={})'.format(octave_idx+1,scale_idx+1))
                        subplot.imshow(img_scale, cmap = 'gray')
                
                ###show difference of guassian (DOG)
                plt.imshow(self.img);plt.axis('off')
                plt.show()
                fig, ax = plt.subplots(self.N_OCTAVES,self.N_SCALES-1,figsize = (15, 10))
                for octave_idx in range(self.N_OCTAVES):
                    img_octave_dogs = self.img_dogs[octave_idx]
                    for dog_idx in range(len(img_octave_dogs)):
                        subplot = ax[octave_idx,dog_idx]
                        subplot.axis('off')
                        img_dog = img_octave_dogs[dog_idx]
                        subplot.set_title('Image on (Oct={},dog={})'.format(octave_idx+1,dog_idx+1))
                        subplot.imshow(img_dog, cmap = 'gray')
###########################################STEP2################################################################
    def cube_extrema(self, img1, img2, img3 ):
            value = img2[1,1]
        
            if value > 0:
                return all([np.all( value >= img ) for img in [img1,img2,img3]]) # test map
            else:
                return all([np.all( value <= img ) for img in [img1,img2,img3]]) # test map
        
    def corners(self, dog , r = 10 ):
            threshold = ((r + 1.0)**2)/r
            dx = np.array([-1,1]).reshape((1,2))
            dy = dx.T
            dog_x = convolve2d( dog , dx , boundary='symm', mode='same' )
            dog_y = convolve2d( dog , dy , boundary='symm', mode='same' )
            dog_xx = convolve2d( dog_x , dx , boundary='symm', mode='same' )
            dog_yy = convolve2d( dog_y , dy , boundary='symm', mode='same' )
            dog_xy = convolve2d( dog_x , dy , boundary='symm', mode='same' )
            
            tr = dog_xx + dog_yy
            det = dog_xx * dog_yy - dog_xy ** 2
            response = ( tr**2 +10e-8) / (det+10e-8)
            
            coords = list(map( tuple , np.argwhere( response < threshold ).tolist() ))
            return coords
        
    def contrast(self, dog , img_max, threshold = 0.03 ):
            dog_norm = dog / img_max
            coords = list(map( tuple , np.argwhere( np.abs( dog_norm ) > threshold ).tolist() ))
            return coords
        
        
      
    def dog_keypoints( self,img_dogs , img_max , threshold ):
            octaves_keypoints = []
            for octave_idx in range(self.N_OCTAVES):
                img_octave_dogs = img_dogs[octave_idx]
                keypoints_per_octave = []
                for dog_idx in range(1, len(img_octave_dogs)-1):
                    dog = img_octave_dogs[dog_idx]
                    keypoints = np.full( dog.shape, False, dtype = np.bool)
                    candidates = set( (i,j) for i in range(1, dog.shape[0] - 1) for j in range(1, dog.shape[1] - 1))
                    search_size = len(candidates)
                    candidates = candidates & set(self.corners(dog)) & set(self.contrast( dog , img_max, threshold ))
                    search_size_filtered = len(candidates)
                    """
                    self.logger.info('Search size reduced by: {:.1f}%'.format( 100*(1 - search_size_filtered/search_size )))
                    """
                    for i,j in candidates:
                        slice1 = img_octave_dogs[dog_idx -1][i-1:i+2, j-1:j+2]
                        slice2 = img_octave_dogs[dog_idx   ][i-1:i+2, j-1:j+2]
                        slice3 = img_octave_dogs[dog_idx +1][i-1:i+2, j-1:j+2]
                        if self.cube_extrema( slice1, slice2, slice3 ):
                            keypoints[i,j] = True
                    keypoints_per_octave.append(keypoints)
                octaves_keypoints.append(keypoints_per_octave)
            return octaves_keypoints
            
     
###############################STEP3##########################################################################
    def dog_keypoints_orientations(self,img_gaussians , keypoints , num_bins):  # 36 Intervals from 0 to 360 degree 
            kps = []
            for octave_idx in range(self.N_OCTAVES):   # 4 octave 
                img_octave_gaussians = img_gaussians[octave_idx]   # 5 guassian img
                octave_keypoints = keypoints[octave_idx]
                for idx,scale_keypoints in enumerate(octave_keypoints):
                    scale_idx = idx + 1 ## idx+1 to be replaced by quadratic localiz
                    gaussian_img = img_octave_gaussians[ scale_idx ] 
                    sigma = 1.5 *( self.SIGMA * ( 2 ** octave_idx ) * ( self.K ** (scale_idx)))  # sigma_theta=1.5*sigma_applied_on_img 
                    radius = self.KERNEL_RADIUS(sigma)
                    kernel = self.gaussian_kernel2d(std = sigma, kernlen = 2 * radius + 1)
                    gx,gy,magnitude,direction = self.sift_gradient(gaussian_img)
                    direction_idx = np.round( direction * num_bins / 360 ).astype(int) #related each angle with bin in hist which include this angles         
                    # direction_idx is array include indecis from 0 to 35 related to angles
                    #ex  if direction_idx=0 this mean angle value in interval 0 to 10 deg
                    for i,j in map( tuple , np.argwhere( scale_keypoints ).tolist() ): # tuples mean pairs
                        window = [i-radius, i+radius+1, j-radius, j+radius+1] #take window arround point with dimension (uper ,lower left,right)
                        mag_win = self.padded_slice( magnitude , window )   #convert window to slice # if window exit img ;padde this part by 0
                        weight = mag_win * kernel     #multiply by guassian kernal to gave it wight arounfd key point 
                        dir_idx = self.padded_slice( direction_idx, window )
                        hist = np.zeros(num_bins, dtype=np.float32)
                        
                        for bin_idx in range(num_bins):
                            hist[bin_idx] = np.sum( weight[ dir_idx == bin_idx ] )
                    # this for lop to convert from index to degree again 
                        for bin_idx in np.argwhere( hist >= 0.8 * hist.max() ).tolist():  # take max point and any point higer than 80% of max point 
                            angle = (bin_idx[0]+0.5) * (360./num_bins) % 360  #bin_idx[0]+0.5 to be in half interval
                            kps.append( (i,j,octave_idx,scale_idx,angle))
            return kps
        
    def padded_slice(self,img, sl):
            output_shape = np.asarray(np.shape(img))
            output_shape[0] = sl[1] - sl[0]
            output_shape[1] = sl[3] - sl[2]
            src = [max(sl[0], 0),
                   min(sl[1], img.shape[0]),
                   max(sl[2], 0),
                   min(sl[3], img.shape[1])]
            dst = [src[0] - sl[0], src[1] - sl[0],
                   src[2] - sl[2], src[3] - sl[2]]
            output = np.zeros(output_shape, dtype=img.dtype)
            output[dst[0]:dst[1],dst[2]:dst[3]] = img[src[0]:src[1],src[2]:src[3]]
            return output
        
    def sift_gradient(self,img):
            dx = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]])
            dy = dx.T
            gx = signal.convolve2d( img , dx , boundary='symm', mode='same' )
            gy = signal.convolve2d( img , dy , boundary='symm', mode='same' )
            magnitude = np.sqrt( gx * gx + gy * gy )
            direction = np.rad2deg( np.arctan2( gy , gx )) % 360
            return gx,gy,magnitude,direction
        
    def colors_by_radius(self,cmap):
            cmap = cm.get_cmap(cmap)
            colors_radius = np.empty((self.N_OCTAVES,self.N_SCALES),dtype=float)
            for oi in range(self.N_OCTAVES):
                for si in range(self.N_SCALES):
                    colors_radius[oi,si] = np.log(2)* oi + np.log(self.K)* (si) 
            colors_radius = (colors_radius-colors_radius.min())/(colors_radius.max()-colors_radius.min())
            colors_radius = cmap(colors_radius)
            return colors_radius
        
    def drawKeypoints(self,image, keypoints , radius_colors, circle_colors):
            copy = image.copy()
            for x, y, octave_idx, scale_idx, angle in keypoints:
                radius = self.SIGMA * ( 2 ** octave_idx ) * ( self.K ** (scale_idx)) *3
                color_c = circle_colors[(octave_idx,scale_idx)]
                color_r = radius_colors[(octave_idx,scale_idx)]
                thickness = (octave_idx+1)
                y *= 2 ** (octave_idx)
                x *= 2 ** (octave_idx)
                cv2.circle(copy, (y, x), int(round(radius)), color_c, thickness)
                cv2.circle(copy, (y, x), 2, color_c, thickness)
        
                angle = np.deg2rad(angle)
                circ_x = int(round(y + np.sin(angle) * radius))
                circ_y = int(round(x + np.cos(angle) * radius))
                cv2.line(copy, (y, x), (circ_x, circ_y), color_r, thickness)
            return copy
    def visualize_points_orirntation(self):
                circle_colors = self.colors_by_radius('viridis')
                radius_colors = self.colors_by_radius('hot') 
                # print(len(list(map(sum, *self.img_octaves)))
                #self.logger.info(" ".join( map( lambda i: str(i.shape) , itertools.chain(*self.img_octaves) )))
                #print(self.img_octaves[0][0].shape)
                img = resize(self.imgs_rgb[0],self.img_octaves[0][0].shape)
                #self.logger.info( img.shape )
                
                for i,j,octave_idx,scale_idx,angle in self.keypoints_ijso:
                    h,w = self.img_octaves[octave_idx][scale_idx].shape
                    assert i < h and j < w , "out of boundaries. shape:{}, ({})".format(str((h,w)))
                
                plt.figure(figsize=(30,30)) 
                plt.axis('off')
                plt.imshow( self.drawKeypoints( img , self.keypoints_ijso, radius_colors, circle_colors))
                # plt.imshow( drawKeypoints( img , self.keypoints_ijso[3], radius_colors, circle_colors))         

##############################################STEP4################################################
    def extract_sift_descriptors128(self,img_gaussians, keypoints, num_bins):    # 8 is the sub region
            descriptors = []; points = [];  data = {} #cach varriable (dichinary)
            for (i,j,oct_idx,scale_idx, orientation) in keypoints:
                # A)Caching 
                if 'index' not in data or data['index'] != (oct_idx,scale_idx):
                    data['index'] = (oct_idx,scale_idx)
                    gaussian_img = img_gaussians[oct_idx][ scale_idx ] 
                    sigma = 1.5 *( self.SIGMA * ( 2 ** oct_idx ) * ( self.K ** (scale_idx)))  # sigma_theta=1.5*sigma_applied_on_img 
                    data['kernel'] = self.gaussian_kernel2d(std = sigma, kernlen = 16)                
        
                    gx,gy,magnitude,direction = self.sift_gradient(gaussian_img)
                    data['magnitude'] = magnitude
                    data['direction'] = direction
        
                window_mag = self.rotated_subimage(data['magnitude'],(j,i), orientation, 16,16)
                window_mag = window_mag * data['kernel']
                window_dir = self.rotated_subimage(data['direction'],(j,i), orientation, 16,16)
                window_dir = (((window_dir - orientation) % 360) * num_bins / 360.).astype(int)  #  to convert from index to degree again 
        
                #B) DOG  "4x4" x16  to calculate histogram for each subregion  with 8 intervals 
                features = []
                for sub_i in range(4):
                    for sub_j in range(4):
                        sub_weights = window_mag[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                        sub_dir_idx = window_dir[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                        hist = np.zeros(num_bins, dtype=np.float32)
                        for bin_idx in range(num_bins):
                            hist[bin_idx] = np.sum( sub_weights[ sub_dir_idx == bin_idx ] )
                        features.extend( hist.tolist())
                        
                #c) combine +normalize        
                features = np.array(features) 
                features /= (np.linalg.norm(features))
                np.clip( features , np.finfo(np.float16).eps , 0.2 , out = features )  # each value more than 0.2 ;put it 0.2
                assert features.shape[0] == 128, "features missing!"
                features /= (np.linalg.norm(features))
                descriptors.append(features)
                points.append( (i ,j , oct_idx, scale_idx, orientation))
            return points , descriptors    # return list of points and list of discriptors
        
        
    def interpixels_image(self, img ):
            inter = np.zeros( (img.shape[0]+1, img.shape[1]+1) )
            inter[:-1,:-1] += img
            inter[1:,1:] += img
            inter[:-1,1:] += img
            inter[1:,:-1] += img
            return inter/4.
        
        
    def rotated_subimage(self,image, center, theta, width, height):
            theta *= 3.14159 / 180 # convert to rad
            
            
            v_x = (cos(theta), sin(theta))
            v_y = (-sin(theta), cos(theta))
            s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
            s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)
        
            mapping = np.array([[v_x[0],v_y[0], s_x],
                                [v_x[1],v_y[1], s_y]])
        
            return cv2.warpAffine(image,mapping,(width, height),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)
        
        
        
  
        
    
        ###################################Feature matching##################################
    def kp_list_2_opencv_kp_list(self,kp_list):
        
            opencv_kp_list = []
            for kp in kp_list:
                opencv_kp = cv2.KeyPoint(x=kp[1] * (2**(kp[2]-1)),
                                         y=kp[0] * (2**(kp[2]-1)),
                                         _size=kp[3],
                                         _angle=kp[4],
        #                                  _response=kp[IDX_RESPONSE],
        #                                  _octave=np.int32(kp[2]),
                                         # _class_id=np.int32(kp[IDX_CLASSID])
                                         )
                opencv_kp_list += [opencv_kp]
        
            return opencv_kp_list
        
        
        
    def combine_img_with_pattern(self):
                self.iteration=0
                imgs_dirs = ['sift_0{}'.format(i) for i in range(2,3) ] #range(1,5)
                image_patterns = {}
                image_patterns_gray = {}
                images = {}
                images_gray = {}
                for img_dir in imgs_dirs:
                    img = np.array(Image.open('images/{}/img.jpg'.format(img_dir)))
                    img,ratio = self.sift_resize(img)
                    images[img_dir] = img
                    image_patterns[img_dir] = []
                    for filename in glob.glob('images/{}/0*.jpg'.format(img_dir)):
                        pattern,_ = self.sift_resize(np.array(Image.open(filename)), ratio )
                        image_patterns[img_dir].append(pattern)
                        image_patterns[img_dir].append(rotate(pattern, 90))
                    images_gray[img_dir] = self.rgb2gray( images[img_dir] )
                    image_patterns_gray[img_dir] = [ self.rgb2gray( img ) for img in image_patterns[img_dir] ]
                
                image_patterns_sift = {}
                images_sift = {}
                for img_dir in imgs_dirs:
                    images_sift[img_dir] = self.pipeline(images_gray[img_dir])
                    image_patterns_sift[img_dir] = []
                    for pattern in image_patterns_gray[img_dir]:
                        image_patterns_sift[img_dir].append( self.pipeline( pattern ))
                
                for img_dir in imgs_dirs:
                    img = images[img_dir]
                    img_sift = images_sift[img_dir]
                    for i in range(len(image_patterns[img_dir])):
                        pattern = image_patterns[img_dir][i]
                        pattern_sift = image_patterns_sift[img_dir][i]
                        self.match(img, img_sift[0], img_sift[1], pattern, pattern_sift[0], pattern_sift[1]) 
             
        
    def match(self,img_a, pts_a, desc_a, img_b, pts_b, desc_b):
            
            img_a, img_b = tuple(map( lambda i: np.uint8(i*255), [img_a,img_b] ))
            
            desc_a = np.array( desc_a , dtype = np.float32 )
            desc_b = np.array( desc_b , dtype = np.float32 )
        
            pts_a = self.kp_list_2_opencv_kp_list(pts_a)
            pts_b = self.kp_list_2_opencv_kp_list(pts_b)
        
            # create BFMatcher object
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc_a,desc_b,k=2)
            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.25*n.distance:
                    good.append(m)
        
            # Sort them in the order of their distance.
        #     matches = sorted(good, key = lambda x:x.distance)
        #     distances = np.array(list(map(lambda x:x.distance,matches)),dtype=float)
        #     print(distances)
        #     distances = (distances - distances.min())/(distances.max()-distances.min())
        #     colors = np.array(cm.get_cmap('viridis')(distances)).tolist()
            # cv2.drawMatchesKnn expects list of lists as matches.
        #      draw_matches(img_a,pts_a,img_b,pts_b,matches,colors,30)
            img_match = np.empty((max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)
        #     cv2.drawMatchesKnn(img_a,pts_a,img_b,pts_b,good,outImg = img_match, matchColor=None,
        #                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #     plt.figure(figsize=(20,20))
        #     plt.imshow(img_match)
        #     plt.show()
            cv2.drawMatches(img_a,pts_a,img_b,pts_b,good, outImg = img_match,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(20,20))
            plt.imshow(img_match)
            plt.show()
        
"""    
    def draw_matches(self,img1, kp1, img2, kp2, matches, colors,count): 
            # We're drawing them side by side.  Get dimensions accordingly.
            # Handle both color and grayscale images.
            if len(img1.shape) == 3:
                new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
            elif len(img1.shape) == 2:
                new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
            new_img = np.zeros(new_shape, type(img1.flat[0]))  
            # Place images onto the new image.
            new_img[0:img1.shape[0],0:img1.shape[1]] = img1
            new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
            
            # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
            r = 3
            thickness = 1
            for idx in range(min(count,len(matches))):
                m = matches[idx]
                # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
                # wants locs as a tuple of ints.
                end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
                end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
                cv2.line(new_img, end1, end2, colors[idx], thickness)
                cv2.circle(new_img, end1, r, colors[idx], thickness)
                cv2.circle(new_img, end2, r, colors[idx], thickness)
            
            plt.figure(figsize=(2,2))
            plt.imshow(new_img)
            plt.show()
"""
#imgs_dir = 'images'
#sift=SIFT(imgs_dir)        
#sift.combine_img_with_pattern()