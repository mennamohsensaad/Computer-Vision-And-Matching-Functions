from PyQt5 import QtWidgets,QtGui , QtCore ,Qt
from PyQt5.QtWidgets import   QFileDialog  ,QWidget,QApplication
from PyQt5.QtGui import QPixmap,QPainter,QPen
from MainWindow import Ui_MainWindow
import numpy as np
from  qimage2ndarray import array2qimage
from scipy import ndimage
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve
from PIL import Image
from numpy import asarray
import cv2
import math
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import gaussian
from skimage.filters import sobel
from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_float
from scipy.ndimage import imread
import handleImageClass as handleImage
from collections import defaultdict
from os import listdir
import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy.misc import imsave
from matplotlib.pyplot import imread
from imageio import imread
import qimage2ndarray
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import seaborn as sns
from PIL.ImageQt import ImageQt
from os.path import isfile , join



from scipy.signal import convolve2d,gaussian
from skimage.transform import rescale
from skimage.transform import resize
import matplotlib.cm as cm
from os.path import isfile , join

from scipy import signal
from math import sqrt
import logging
from math import sin, cos
import glob
from skimage.transform import rotate
from sympy import Symbol
import sympy as sym
import time
import SIFT_class as c


class CV(QtWidgets.QMainWindow):
    def __init__(self):
        super(CV, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = 75
        self.strong_pixel = 255
        self.sigma = 1
        self.kernel_size = 5
        self.lowThreshold = 0.05
        self.highThreshold = 0.15
        self.click=0
        #self.ui.comboBox_shape.currentIndexChanged.connect(self.IMPOSE_Lines_Circles)
        self.ui.pushButton_Hough_load.clicked.connect(self.LoadImage_Hough)
        self.ui.pushButton_apply_hough.clicked.connect(self.IMPOSE_Lines_Circles)
        self.ui.pushButton_opencv.clicked.connect(self.Apply_canny_with_opencv)
        self.ui.pushButton_snake_load.clicked.connect(self.LoadImage2)
        self.ui.Reset.clicked.connect(self.Reset)
        self.ui.clear_anchors.clicked.connect(self.Clear_anchors)
        self.ui.snake_apply.clicked.connect(self.Apply_snake)
        self.ui.label_snake_input.mousePressEvent=self.getPixel
        self.ui.pushButton_Harris_load.clicked.connect(self.LoadImage3)
        self.ui.Harris_ApplyButton_2.clicked.connect(self.Apply_Harris)
        ####__________HYBRID__________
        self.ui.pushButton_histograms_load_2.clicked.connect(self.button_clicked1)##LOAD IMAGE 1 HYBRID
        self.ui.pushButton_histograms_load_3.clicked.connect(self.button_clicked2)##LOAD IMAGE 2 HYBRID
        self.ui.pushButton_histograms_load_4.clicked.connect(self.button_clicked3)##OUTPUT HYBRID
        #####_________HISTOGRAM__________
        self.draw_curve=0
        self.ui.pushButton_histograms_load.clicked.connect(self.LoadImage)
        self.ui.comboBox_9.currentIndexChanged.connect(self.Draw_histogram)
        self.ui.comboBox_7.currentIndexChanged.connect(self.check_Effect_to_image)
        self.ui.comboBox_8.currentIndexChanged.connect(self.Choose_curve)
        ##___________FILTERS____________
        self.ui.pushButton_filters_load.clicked.connect(self.button_clicked)
        self.ui.comboBox.currentIndexChanged.connect(self.add_noise)
        self.ui.comboBox_2.currentIndexChanged.connect(self.show_filters)
        ##___________SIFT____________
        self.obj=c.SIFT(self)
        self.iteration=0
        self.ui.pushButton_load_TM_A_2.clicked.connect(self.button_SIFTA)##LOAD IMAGE 1 SIFT
        self.ui.pushButton_load_TM_B_2.clicked.connect(self.button_SIFTB)##LOAD IMAGE 2 SIFT
        self.ui.pushButton_SIFT_match.clicked.connect(self.combine_img_with_pattern_GUI)##OUTPUT Match

    
    def LoadImage_Hough(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG *.JPEG *.JFIF);;img file (*.jpg *.PNG *.JPEG *.JFIF)")
        if self.fileName:
            self.pixmap = QPixmap(self.fileName)
            self.input_img =mpimg.imread(self.fileName)
            self.ui.label_Hough_input.setPixmap(self.pixmap)
            self.ui.label_Hough_output.clear()
            self.ui.label_Hough_output_2.clear()
            pixels = asarray(self.input_img)
            print(pixels.shape)
            self.ui.lineEdit_size_Hough.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")
    
    def rgb2gray(self,rgb):
        #return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
          r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
          gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
          return gray 
    
    def Apply_canny(self):
       self.canny_filter()
       self.ui.label_Hough_output.setPixmap(QPixmap("canny_edges.jpg"))
    
#    
    def canny_filter(self):
              #self.gray_img
#              
            #self.gray_img =cv2.imread(self.fileName,0)
            canny_img_final = []
            input_size=(self.ui.lineEdit_mask_size.text())
            #check if gray image or color 
            pixels = asarray(self.input_img)
            pixels = pixels.astype('float32')
            try:
                if (pixels.shape[2] == 3 or pixels.shape[2] == 4 ):
                    self.gray_img =self.rgb2gray( self.input_img)
                   
            #elif (pixels.shape[2] == 1):
            except IndexError:  #(pixels.shape[2] == None):
                     self.gray_img =self.input_img
                     
        
            #plt.imshow(self.gray_img)
            guass=self.gaussian(int(input_size),int(input_size),1)
            self.img_smoothed = convolve(self.gray_img,guass)
            self.gradientMat, self.thetaMat =self.sobel_filter_for_canny(self.img_smoothed)
            self.nonMaxImg =self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg =self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            canny_img_final.append(img_final)
            print(canny_img_final)
            self.visualize(canny_img_final, 'gray')                 

    

    def sobel_filter_for_canny(self,img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    
    def non_max_suppression(self,img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z
    
    def threshold(self,img):
#        lowThreshold = 0.05
#        self.highThreshold = 0.1
        LhighThreshold = img.max() * self.highThreshold;
        LlowThreshold = self.highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= LhighThreshold)
        zeros_i, zeros_j = np.where(img < LlowThreshold)

        weak_i, weak_j = np.where((img <= LhighThreshold) & (img >= LlowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self,img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img


    def visualize(self,imgs, format=None, gray=False):
            #plt.figure(figsize=(20, 40))
            for i, img in enumerate(imgs):
                if img.shape[0] == 3:
                    img = img.transpose(1,2,0)
                img=Image.fromarray(np.uint8(img))    
                img.save("canny_edges.jpg")    
#___________________________________to check accuracy of our implementation to canny edges detector ____________________
#___________________________  try canny with opencv to compare results with our implemetation for canny_________________
    def Apply_canny_with_opencv(self):    
       img = cv2.imread(self.fileName,0)
       edges = cv2.Canny(img,100,200)
       print (edges)
       pixels=np.array(edges)
        #gray2qimage
       iamge=array2qimage(pixels*1000)
       pixmap = QPixmap( iamge)
       self.ui.label_Hough_output_2.setPixmap(pixmap)
       self.ui.label_Hough_output_2.show
       
       
#________________________________________________HoughFunctions__________________________________________________________
       
       
    

#           ________________________________gaussian & sobel & laplacian __________________
    
    def gaussian(self,m,n,sigma):
        gaussian=np.zeros((m,n))
        m=m//2
        n=n//2
        for x in range (-m,m+1):
            for y in range (-n,n+1):
                x1=sigma*math.sqrt(2*np.pi)
                x2=np.exp(-(x**2+y**2)/(2*sigma**2))
                gaussian[x+m,y+n]=(1/x1)*x2  
        return gaussian
    
    def gaussian_filter(self,m,n,sigma,img):
        g=self.gaussian(m,n,sigma)
        img.convolve(g)
        return img
        
    
    def edge(self,img,threshold):
        # Laplacian with sobel to detect the edges
        laplacian = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        print(sum(sum(laplacian)))
        sobel_x = ([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = ([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        gx = img.convolve(sobel_x)
        gy = img.convolve(sobel_y)
    
        G = np.sqrt(gx ** 2 + gy ** 2) 
        
    
        G[G<threshold] = 0
        Lap = img.convolve(laplacian)
    
        M,N = Lap.shape
    
        temp = np.zeros((M+2,N+2))                                                 
        temp[1:-1,1:-1] = Lap                                                       
        result = np.zeros((M,N))                                                   
        for i in range(1,M+1):
            for j in range(1,N+1):
                #Looking for a negative pixel and checking its 8 neighbors
                if temp[i,j]<0:                                                    
                    for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                            if temp[i+x,j+y]>0:
                                result[i-1,j-1] = 1                                
        #img.load(np.array(np.logical_and(result,G),dtype=np.uint8))
        img.load(np.array(np.logical_and(result,G),dtype=np.uint8))
        return img
    
    def detectCircles(self,img,threshold,region,radius):
        M,N = img.shape
        [R_max,R_min] = radius
    
        R = R_max - R_min
        #3D accumulator array :radius, X ,Y 
        A = np.zeros((R_max,M+2*R_max,N+2*R_max))
        B = np.zeros((R_max,M+2*R_max,N+2*R_max))
    
        theta = np.arange(0,360)*np.pi/180
        #edges = np.nonzero(img[:,:])
        #For non Zero elements
        edges = np.argwhere(img[:,:])
                                                      
        for K in range(R):
            r = R_min+K
            
            circle = np.zeros((2*(r+1),2*(r+1)))
            #Finding out the center 
            (m,n) = (r+1,r+1)                                                       
            for angle in theta:
                x = int(np.round(r*np.cos(angle)))
                y = int(np.round(r*np.sin(angle)))
                circle[m+x,n+y] = 1
            constant = np.argwhere(circle).shape[0]
            #print(circle)
            #print(constant)
            for x,y in edges:                                                     
                X = [x-m+R_max,x+m+R_max]                                           
                Y= [y-n+R_max,y+n+R_max]                                            
                A[r,X[0]:X[1],Y[0]:Y[1]] += circle
            A[r][A[r]<threshold*constant/r] = 0
    
        for r,x,y in np.argwhere(A):
            temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
            try:
                p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
            except:
                continue
            B[r+(p-region),x+(a-region),y+(b-region)] = 1
    
        return B[:,R_max:-R_max,R_max:-R_max]   
           
           
    
    def displayCircles(self,A):
        img = imread(self.fileName)
        fig =plt.figure()
        plt.imshow(img)
        circleCoordinates = np.argwhere(A)                                          
        circle = []
        for r,x,y in circleCoordinates:
            circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
            fig.add_subplot(111).add_artist(circle[-1])
            
        #self.ui.label_filters_input.setPixmap(self.input_iamge)
        #plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
#        plt.savefig("fff.jpg")
       
        
        plt.savefig('detection_of_circles.jpg', dpi=900, bbox_inches='tight',pad_inches=0)
        
        
        
        



#____________________________________________HOUGH-TRAANSFORM-LINE-DETECTION_________________________________________

#hough transform
    def Hough_Line(self,image):
    #######First find image dimensions
    # y for rows and x for columns 
        Y = image.shape[0]
        X = image.shape[1]
        DIAGONAL = int(np.round(np.sqrt(X**2 + Y ** 2)))
        thetas = np.deg2rad(np.arange(0, 180))
        ########what is radius Range
        rs = np.linspace(-DIAGONAL, DIAGONAL, 2*DIAGONAL)
        #2. Create accumulator array and initialize to zero
        accumulator = np.zeros((2 * DIAGONAL, len(thetas)))
        #3. Loop for each edge pixel 
        for y in range(Y):
            for x in range(X):
                # Check if it is an edge pixel
                #  NB: y -> rows , x -> columns
                if image[y,x] > 0:
                    #4. Loop for each theta
                    # Map edge pixel to hough space
                     for k in range(len(thetas)):
    
                        #5. calculate $\rho$
                        # Calculate space parameter
                        r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
    
                        accumulator[int(r) + DIAGONAL,k] += 1
        return accumulator, thetas, rs
#extract lines from image
#######Draw__lines
    def Draw_Lines(self,accumulator,thetas,rhos,threshold):
                  self.lines = defaultdict()
                  
                  self.acc2 = np.zeros(accumulator.shape)
                  for rho_idx in range(len(rhos)) :
                    for theta_idx in range(len(thetas)) :
                        if accumulator[rho_idx, theta_idx] > threshold :
                            theta = thetas[theta_idx]
                            rho = rhos[rho_idx]
                            # print (angle , rho , accumulator[angle_index , rho])
                            self.lines[(rho,theta)] = accumulator[rho_idx, theta_idx]
                            
                            self.acc2[rho_idx,theta_idx] = accumulator[rho_idx, theta_idx]
                  return self.lines,self.acc2
              

              
    def IMPOSE_Lines_Circles (self):
     
     self.hough = str(self.ui.comboBox_shape.currentText())
       
     print(self.hough)
     if self.hough=="Circles":
            self.ui.label_Hough_output.clear()
            self.ui.label_Hough_output_2.clear() 
            input_size=(self.ui.lineEdit_mask_size.text())
            print(input_size)
            #self.fileName
            self.img =handleImage.imarray(self.fileName)
            self.filter_img =self.gaussian_filter(int(input_size),int(input_size),2,self.img)            
            self.image = self.edge(self.filter_img,128) 
            
                              
            self.image = self.detectCircles(self.image,13,15,[40,10])
            self.output_iamge=self.displayCircles(self.image)
#            elif self.fileName=="images10.jpg":       
#                 self.image = self.detectCircles(self.image,9,15,[40,10])
#                 self.output_iamge=self.displayCircles(self.image)
           
            #self.output_iamge=self.displayCircles(self.image)
            
                  
            
            #self.ui.label_Hough_output.setPixmap(self.output_iamge)
            self.ui.label_Hough_output.setPixmap(QPixmap("detection_of_circles.jpg"))
            self.ui.label_Hough_output.show()
            
     elif self.hough=="Lines": 
        self.ui.label_Hough_output.clear()
        self.ui.label_Hough_output_2.clear() 
        input_size=(self.ui.lineEdit_mask_size.text())
        print(input_size)      
        imag = cv2.imread(self.fileName)
#        width = 128
#        height = 128 # keep original height
#        dim = (width, height)
#             
#            # resize image
#        imag = cv2.resize(imag, dim, interpolation = cv2.INTER_AREA)
        
        #img = cv2.imread('images/beauflor-spirit-chessboard-vinyl-flooring-p755-3264_image.jpg')
        # Convert the img to grayscale 
        gray = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY) 
        # Apply edge detection method on the image 
        self.edges = cv2.Canny(gray,50,150,apertureSize = 3) 
        #cv2.imshow('gray_scale',gray_scale)
        
        accumulator,thetas,rhos = self.Hough_Line(self.edges)
        self.lines, self.acc2 = self.Draw_Lines( accumulator,thetas,rhos,90)
        #cv2.imshow('acc2',acc2)
        
        for (rho,theta), val in self.lines.items():
           
            a = np.cos(theta)
            b = np.sin(theta)
            pt0 = rho * np.array([a,b])
            pt1 = tuple((pt0 + 1000 * np.array([-b,a])).astype(int))
            pt2 = tuple((pt0 - 1000 * np.array([-b,a])).astype(int))
            cv2.line(imag, pt1, pt2, (0,0,255), 3)
          
        pixels=np.array(imag)
        #gray2qimage
        im=array2qimage(pixels)
        pixmap = QPixmap(im)
        self.ui.label_Hough_output.setPixmap(pixmap)
        #self.ui.label_Hough_output.show
        #plt.imshow(pixmap)
        
    
    
     elif self.hough =="Canny" : 
           self.ui.label_Hough_output.clear()
           self.canny_filter()
           self.ui.label_Hough_output.setPixmap(QPixmap("canny_edges.jpg"))   
          
     else :  

          self.ui.label_Hough_output.clear()       
  



#____________________________________________Active_contour_Model_________________________________________________
  
    def LoadImage2(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG *.JPEG *.JFIF);;img file (*.jpg *.PNG *.JPEG *.JFIF)")
        if self.fileName:
            self.Reset()
            self.pixmap = QPixmap(self.fileName)
            self.input_img =mpimg.imread(self.fileName)
            self.ui.label_snake_input.setPixmap(self.pixmap)
            self.ui.label_snake_output.clear()
            self.ui.label_snake_output_running.clear()
            pixels = asarray(self.input_img)
            self.Image =io.imread(self.fileName,0)
            #print(self.Image)
            self.ui.lineEdit_size_snake.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")
            
    
    def Apply_snake(self):
        
        snake_list=self.active_contour_model(self.fileName)
        #print(snake_list)
        self.calc_area_preimeter(snake_list)
        chainCode=self.DriverFunction(snake_list) 
        self.ui.lineEdit_chain_code.setText(""+str(chainCode)+"")#+str('x')+""+str(" ")+""
        
        
    def intp(self,x_list,y_list):
        new_x = []
        new_y = []
        lenth = len(x_list)
        print('contour lenghth : ',lenth)
        if lenth < 400:
            for i in range(1,lenth-1):
                new_x.append(x_list[i - 1])
                new_y.append(y_list[i - 1])
                inp_x = (x_list[i-1] + x_list[i]) //2
                inp_y = (y_list[i-1] + y_list[i]) //2
    
                new_x.append(inp_x)
                new_y.append(inp_y)
                #print(new_x,new_y)        
        else:
            return x_list,y_list
        return new_x,new_y
    
    
    def active_contour_model(self,imageName):
        img =io.imread(imageName,0)
        color_img = img
        gray_img = rgb2gray(img)
        gaussian_img =gaussian(gray_img,2)
        #gaussian_img_norm=normalizeRange(gaussian_img, minVal=0, maxVal=1)
        #signal.convolve2d(gray_img, gaussian_Filter(5, (3,3)), mode='same') #gaussian(gray_img,3)  #convolve image with guassian fi
        alpha=float(self.ui.lineEdit_alpha.text())
        beta=float(self.ui.lineEdit_beta.text())
        gamma=float(self.ui.lineEdit_gamma.text())
        snake = self.active_contour( gaussian_img,color_img,
                               self.init_contour, alpha, beta, gamma)
        " alpha=0.015, beta=10, gamma=0.001"
        QApplication.processEvents()
        self.pixmap = QPixmap(self.fileName)
        painter= QtGui.QPainter(self.pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.green)
        penRect.setWidth(4)
        painter.setPen(penRect)
        for i in range(len(snake) - 1): 
            a = snake[i] 
            painter.drawPoint(a[0],a[1])
        painter.end()
        result=self.pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_snake_output.setPixmap(result)
        QApplication.processEvents()
        return snake
    
    def inital_contour(self,):
        c=[self.x,self.y]
        r=np.sqrt((self.x-self.x1) ** 2+(self.y-self.y1) ** 2)
        theta=np.linspace(0, 2*np.pi, 200) # min, max, number of divisions
        x=c[0]+r*np.cos(theta)
        y=c[1]+r*np.sin(theta)
        x_list = x
        #print ( x_list)
        y_list = y
        new_x_list,new_y_list=self.intp(x_list,y_list)
        self.init_contour = np.array([new_x_list,new_y_list]).T
        #print(init)
        QApplication.processEvents()
        painter= QtGui.QPainter(self.pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.red)
        penRect.setWidth(4)
        painter.setPen(penRect)
        for i in range(len(self.init_contour)): 
            a = self.init_contour[i] 
            #print(b)
            painter.drawPoint(a[0],a[1])
        
        painter.end()
        result=self.pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_snake_input.setPixmap(result)
        QApplication.processEvents()
    
    def active_contour(self,image,color, snake, alpha, beta,gamma,
                   w_line=0, w_edge=1, 
                   bc='periodic', max_px_move=1.0,
                   max_iterations=3000, convergence=0.25):
        max_iterations = int(max_iterations)
        if max_iterations <= 0:
            raise ValueError("max_iterations should be >0.")
        convergence_order = 10
        img = img_as_float(image)
        
        RGB = img.ndim == 3
    
        if w_edge != 0:
            if RGB:
                edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]),
                        sobel(img[:, :, 2])]
            else:
                edge = [sobel(img)]
                
    
            for i in range(3 if RGB else 1):
                edge[i][0, :] = edge[i][1, :]
                edge[i][-1, :] = edge[i][-2, :]
                edge[i][:, 0] = edge[i][:, 1]
                edge[i][:, -1] = edge[i][:, -2]
        else:
            edge = [0]
            
        #print(edge)    
    #    img_norm=normalizeRange(edge, minVal=0, maxVal=1)  #normalize(edge,0,1)
    #    edge=img_norm
        if RGB:
            img = w_line*np.sum(img, axis=2) \
                + w_edge*sum(edge)
        else:
            img = w_line*img + w_edge*edge[0]
    
    
        intp = RectBivariateSpline(np.arange(image.shape[1]), #img
                                   np.arange(image.shape[0]),
                                   img.T, kx=2, ky=2, s=0)
    
    
        x, y = snake[:, 0].astype(np.float), snake[:, 1].astype(np.float)
        xsave = np.empty((convergence_order, len(x)))
        ysave = np.empty((convergence_order, len(x)))
    
    
        n = len(x)
    
        a = np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1) - 2*np.eye(n)
        b = np.roll(np.eye(n), -2, axis=0) + \
            np.roll(np.eye(n), -2, axis=1) - \
            4*np.roll(np.eye(n), -1, axis=0) - \
            4*np.roll(np.eye(n), -1, axis=1) + \
            6*np.eye(n)
        A = -alpha*a + beta*b
        #print(A.shape)
        #print((gamma*np.eye(n)).shape)
    
        inv = np.linalg.inv(A + gamma*np.eye(n))
#        fig, ax = plt.subplots(figsize=(7, 7))
#        ax.imshow(color)
    
    
        for i in range(max_iterations):
    
            fx = intp(x, y, dx=1, grid=False)
            fy = intp(x, y, dy=1, grid=False)
    
            xn = inv @ (gamma*x + fx)
            yn = inv @ (gamma*y + fy)
    
    
            dx = max_px_move*np.tanh(xn-x)
            dy = max_px_move*np.tanh(yn-y)
    
            x += dx
            y += dy
    
            j = i % (convergence_order+1)
            if j < convergence_order:
                xsave[j, :] = x
                ysave[j, :] = y
            else:
                dist = np.min(np.max(np.abs(xsave-x[None, :]) +  np.abs(ysave-y[None, :]), 1))
                if dist < convergence:
                    break
    
            
            snake = np.array([x, y]).T
              
            #pause time for some sec befor next iteration 
            sec=0
            self.ui.lineEdit_iteration_num.setText(""+str(i)+"")
            for i in range(9000):
                sec=sec+1   
            
            self.running_snake(img,snake)
 
        return np.array([x, y]).T
    
    def running_snake(self,img,snake):
        QApplication.processEvents()
        #self.pixmap = QPixmap(img)
        pixels=np.array(img)
         #gray2qimage
        iamge=array2qimage(pixels*1000)
        #new_img = Image.fromarray(iamge)
        pixmap = QPixmap(iamge)
        #self.pixmap = QPixmap(self.fileName)
        painter= QtGui.QPainter(pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.blue)
        penRect.setWidth(4)
        painter.setPen(penRect)
        for i in range(len(snake)): 
            a =  snake[i] 
            #print(b)
            painter.drawPoint(a[0],a[1])
        
        painter.end()
        result=pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_snake_output_running.setPixmap(result)
        QApplication.processEvents()
        
    def getPixel(self,event):
        
          if (self.click==0):
              self.x=math.floor((event.pos().x()*self.pixmap.width())/self.ui.label_snake_input.frameGeometry().width()) 
              self.y=math.floor((event.pos().y()*self.pixmap.height())/self.ui.label_snake_input.frameGeometry().height())
              self.click=self.click+1  
              #print(self.x,self.y)
            
          elif(self.click==1):
              self.x1=math.floor((event.pos().x()*self.pixmap.width())/self.ui.label_snake_input.frameGeometry().width()) 
              self.y1=math.floor((event.pos().y()*self.pixmap.height())/self.ui.label_snake_input.frameGeometry().height())
              self.click=self.click+1
              #print(self.x1,self.y1)
          self.frame_on_Anchors()
              
    def frame_on_Anchors(self):
        
        QApplication.processEvents()
        painter= QtGui.QPainter(self.pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.red)
        penRect.setWidth(2)
        painter.setPen(penRect)
        if (self.click==1):
           painter.drawRect(self.x,self.y,4, 4)
        elif (self.click==2):
           painter.drawRect(self.x1,self.y1,4, 4)
           
        else:
            print("end")
        painter.end()
        result=self.pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_snake_input.setPixmap(result)
        QApplication.processEvents()
        if (self.click==2):
             self.inital_contour()
        else:
            self.click=self.click
    

    def Clear_anchors(self):
        self.click=0
        self.pixmap = QPixmap(self.fileName)
        self.ui.label_snake_input.setPixmap(self.pixmap)
        self.ui.label_snake_output.clear()
        self.ui.label_snake_output_running.clear() 
        self.ui.lineEdit_area_of_contour.setText(""+str("")+"")
        self.ui.lineEdit_Perimeter_of_contour.setText(""+str("")+"")
        self.ui.lineEdit_chain_code.setText(""+str("")+"")
        self.ui.lineEdit_iteration_num.setText(""+str("")+"")
    
    def Reset(self): 
        self.Clear_anchors()
        self.ui.lineEdit_chain_code.setText(""+str("")+"")
        self.ui.lineEdit_iteration_num.setText(""+str("")+"")
        self.ui.lineEdit_alpha.setText(""+str("")+"")
        self.ui.lineEdit_beta.setText(""+str("")+"")
        self.ui.lineEdit_gamma.setText(""+str("")+"")
        self.ui.lineEdit_area_of_contour.setText(""+str("")+"")
        self.ui.lineEdit_Perimeter_of_contour.setText(""+str("")+"")
        
    def DriverFunction(self,ListOfPoints): 
        chainCode = self.generateChainCode(ListOfPoints)
        #print(chainCode)
        chainCodeString = "".join(str(e) for e in chainCode)
        #print (chainCodeString)
        return chainCodeString
    
    
    def getChainCode(self,x1, y1, x2, y2): 
        codeList = [5, 6, 7, 4, 0, 3, 2, 1]
        # to detect direction
        dx = x2 - x1 
        dy = y2 - y1 
        hashKey = 3 * dy + dx + 4
        if  int(hashKey) >7:
             hashKey=7
        elif int(hashKey) <0 :
             hashKey=0
        else:
            hashKey=hashKey
        #print(hashKey)
        return codeList[int(hashKey)] 

    def generateChainCode(self,ListOfPoints): 
        chainCode = [] 
        for i in range(len(ListOfPoints) - 1): 
            a = ListOfPoints[i] 
    #        print(a)
            #print(a[0])
            b = ListOfPoints[i + 1] 
            #print(b)
            chainCode.append(self.getChainCode(a[0], a[1], b[0], b[1])) 
        return chainCode 
    
    def calc_area_preimeter(self,snake):
        
        area=0
        primeter=0
        for i in range(len(snake)-1): 
            a =snake[i] 
            b=snake[i + 1] 
            #print(b)
            x,y=a[0],a[1]
            nex_x,nex_y=b[0],b[1]
            area += (x*nex_y-y*nex_x)/10
            primeter +=abs((x-nex_x)+(y-nex_y)*1j)/5
        area = area / 12
        self.ui.lineEdit_area_of_contour.setText(""+str(float(area))+""+""+""+str(' ')+""+str('m^2')+"")
        self.ui.lineEdit_Perimeter_of_contour.setText(""+str(float(primeter))+""+""+str(' ')+""+""+str('m')+"")
        return area,primeter
        

#    
#    def Area_and_perimeter(self):
#        
#        thresh = cv.threshold(self.Image,127,255,0)
#        im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
#        area = cv.contourArea(contours[0])
#        self.ui.lineEdit_area_of_contour.setText(""+str(area)+"")
#        perimeter = cv.arcLength(contours[0],True)
#        self.ui.lineEdit_Perimeter_of_contour.setText(""+str(perimeter)+"")
 
    
#_________________________________________Harris_corner_detectors____________________________________________________    
     
    def LoadImage3(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG *.JPEG *.JFIF);;img file (*.jpg *.PNG *.JPEG *.JFIF)")
        if self.fileName:
            self.pixmap = QPixmap(self.fileName)
            self.input_img =mpimg.imread(self.fileName)
            self.ui.label_Harris_input.setPixmap(self.pixmap)
            self.ui.label_Harris_output.clear()
            pixels = asarray(self.input_img)
            #print(pixels.shape)
            self.ui.lineEdit_size_Harris.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")
            
     
        
    #Import Libraries


#Functions used in Phase I

#Define RGB2gray function
#    def rgb2gray(self,img) :
#        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    
    
    #Detemine gradient function for Fx and Fy using sobel filter(normlized)
    def gradient_x(self,img) :
        grad_img = ndimage.convolve(img, np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]))
        return grad_img/np.max(grad_img)
    
    def gradient_y(self,img) :
        grad_img = ndimage.convolve(img, np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]))
        return grad_img/np.max(grad_img)
    
    
    #Harris Corner Detector Implementation and test
    def Apply_Harris (self) :
        
        self.ui.label_Harris_output.clear()
        input_img = cv2.imread(self.fileName)
        ratio =float(self.ui.ratio.text())
        
        #Phase I : Find filtered grdient
        #Load the input image
#        input_img = imageio.imread(img_path)
        
        #Convert the image to grayscale
        gray_input_img = rgb2gray(input_img)
        #Apply gaussian blurring
        #blur_img = ndimage.gaussian_filter(gray_input_img, sigma = 1.0)
        blur_img = gaussian(gray_input_img, sigma = 0.8)
        #Find gradient Fx
        x_grad = self.gradient_x(blur_img)
        #Find gradient Fy
        y_grad = self.gradient_y(blur_img)
        
            
        #Phase II : Find corners
        xx_grad = x_grad * x_grad
        yy_grad = y_grad * y_grad
        xy_grad = x_grad * y_grad
        tuple_data = [] #Contains y, x Co-ordinates and its corner response
        k = 0.05
        max = 0
        
        for i in range(1, int(input_img.shape[0] - 1)) :
                for j in range(1, int(input_img.shape[1] - 1)) :
                    window_x = xx_grad[i-4 : i+5 , j-4 : j+5]
                    window_y = yy_grad[i-4 : i+5 , j-4 : j+5]
                    window_xy = xy_grad[i-4 : i+5 , j-4 : j+5]
                    sum_xx = np.sum(window_x)
                    sum_yy = np.sum(window_y)
                    sum_xy = np.sum(window_xy)
                    determinant = (sum_xx * sum_yy) - (sum_xy * sum_xy)
                    trace = sum_xx + sum_yy
                    R = determinant - (k * trace * trace)
                    tuple_data.append((i, j, R))
                    if(R > max) :
                        max = R
        
        #L contains y, x co-ordinate(whose value is greater than threshold) and their corner response of those co-ordinates
        L = []
        L2 = []
        threshold_T = ratio * max
        
        for res in tuple_data :
            i, j, R = res
            L2.append([i, j, R])    #without using thresholding
            if R > threshold_T :
                L.append([i, j, R]) #using thresholding
        
              
        
        
        #Phase III : Non maximal suppression
        sorted_L = sorted(L, key = lambda x: x[2], reverse = True)
        final_L = [] #final_l contains list after non maximal suppression
        final_L.append(sorted_L[0][:-1])
        dis = 10
        xc, yc = [], []
        for i in sorted_L :
            for j in final_L :
                if(abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis) :
                    break
            else :
                final_L.append(i[:-1])
                xc.append(i[1])
                yc.append(i[0])
        
                
        sorted_L2 = sorted(L2, key = lambda x: x[2], reverse = True)
        final_L2 = [] #final_l contains list after non maximal suppression
        final_L2.append(sorted_L2[0][:-1])
        xc2, yc2 = [], []
        for i in sorted_L2 :
            for j in final_L2 :
                if(abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis) :
                    break
            else :
                final_L2.append(i[:-1])
                xc2.append(i[1])
                yc2.append(i[0])
        
        pixmap = QPixmap(self.fileName)
        painter= QtGui.QPainter(pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.red)
        penRect.setWidth(4)
        painter.setPen(penRect)
        for i in range(len(xc)): 
            x=  xc[i]+2
            y= yc[i]
            #print(b)
            painter.drawPoint(x,y)
        
        painter.end()
        result=pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_Harris_output.setPixmap(result)
        
        
        #without using thresholding
        #plt.imshow(input_img, cmap = plt.get_cmap('gray'))
        #plt.plot(xc2, yc2, '*', color='purple')
        #plt.show()    
        ##_______________HYBRID_____________
    def button_clicked1(self):  
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG *.JPEG *.JFIF);;img file (*.jpg *.PNG *.JPEG *.JFIF)")
        if fileName:
            pixmap = QPixmap(fileName)
            self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation) 
            self.color_img1 =mpimg.imread(fileName)
            width = 256
            height = 256 # keep original height
            dim = (width, height)
             
            # resize image
            self.color_img1 = cv2.resize(self.color_img1, dim, interpolation = cv2.INTER_AREA)
            self.gray_img1 =self.rgb2gray(self.color_img1) 
            
            self.pixels1 = np.asarray(self.color_img1)
            self.pixels1 = self.pixels1.astype('float32')
            
            self.Display_image1()
            self.Label1_Name()
            self.size1()
            
            
#    def rgb2gray(self,rgb_image):
#        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])  # ... mean  all rgb values     
    
    def button_clicked2(self):  
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.JPG *.PNG *.JPEG *.JFIF);;img file (*.jpg *.JPG *.PNG *.JPEG *.JFIF)")
        if fileName:
            pixmap = QPixmap(fileName)
            self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio,QtCore.Qt.FastTransformation) 
            self.color_img2 =mpimg.imread(fileName)
            #plt.imshow(self.color_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#            einstein = ndimage.imread("einstein.png", flatten=True)
            width = 256
            height = 256 # keep original height
            dim = (width, height)
             
            # resize image
            self.color_img2 = cv2.resize(self.color_img2, dim, interpolation = cv2.INTER_AREA)
            self.gray_img2 =self.rgb2gray(self.color_img2)
            
            self.pixels2 = np.asarray(self.color_img2)
            self.pixels2 = self.pixels2.astype('float32')
#            print(self.pixels2.shape)
            
            self.Display_image2()
            self.Label2_Name()
            self.size2()
            
            
    def button_clicked3(self):  

           self.hybrid   = self.hybridImage (self.gray_img2, self.gray_img1, 25, 10)
#           misc.imsave("marilyn-einstein.png", numpy.real(hybrid))
           
           output_hybird = np.array(self.hybrid).astype(np.int32)
           #output_hybird = np.array(numpy.real(hybrid)*200).astype(np.uint8)
           output_hybird = qimage2ndarray.array2qimage(output_hybird)
           output_hybird = QPixmap(output_hybird)
           output_hybird = output_hybird.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
           
           self.ui.label_histograms_output_2.setPixmap(output_hybird)
           self.ui.label_histograms_output_2.show
       
        
    def Display_image1(self):
        self.ui.label_histograms_input_2.setPixmap(self.pixmap)####for input image 1
        self.ui.label_histograms_input_2.show
    
    def Display_image2(self):
        self.ui.label_histograms_hinput_2.setPixmap(self.pixmap)#####for input image 2
        self.ui.label_histograms_hinput_2.show
        
    def Display_image3(self):
        
        self.ui.label_histograms_output_2.setPixmap(self.pixmap)#####for input image 2
        self.ui.label_histograms_output_2.show     
        

        
    def Label1_Name(self):
        
        #self.ui.label_12 = QLabel(self)
        self.ui.label_12.setText('Name:Marylin')
      
    def Label2_Name(self):
        
        self.ui.label_15.setText('Name:Einstein')


    def size1(self):
        self.ui.lineEdit_11.setText(""+str(self.pixels1.shape[0])+""+str('x')+""+str(self.pixels1.shape[1])+"")

    def size2(self):
        self.ui.lineEdit_12.setText(""+str(self.pixels2.shape[0])+""+str('x')+""+str(self.pixels2.shape[1])+"")
        
      


# sample values from a spherical gaussian function from the center of the image
    def makeGaussianFilter(self,numRows, numCols, sigma, highPass=True):
       centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
       centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)
       def gaussian(i,j):
              coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
              return 1 - coefficient if highPass else coefficient
    
       return numpy.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])

######apply filter by doing coord. multiplication
    def filterFT(self,imageMatrix, filterMatrix):
        ##########apply fourier
       shiftedDFT = fftshift(fft2(imageMatrix))
#       misc.imsave("dft.png", self.scaleSpectrum(shiftedDFT))
       filteredDFT = shiftedDFT * filterMatrix
#       misc.imsave("Lowpassfilter.png", self.scaleSpectrum(filteredDFT))
       return ifft2(ifftshift(filteredDFT))
       
    ####get ride of High freq. comp.(((((((((( Apply fourier to image matrix))))))&&&&& make gaussian for lpf *g(x,y))
    def lowPass(self,imageMatrix, sigma):
       n,m = imageMatrix.shape
       return self.filterFT(imageMatrix, self.makeGaussianFilter(n, m, sigma, highPass=False))
    
    ####get ride of Low freq. comp (((((((((( Apply fourier to image matrix)))))) &&&&&  make gaussian for lpf *1-g(x,y))
    def highPass(self,imageMatrix, sigma):
       n,m = imageMatrix.shape
       return self.filterFT(imageMatrix, self.makeGaussianFilter(n, m, sigma, highPass=True))
    
    ######compining lowpass part of an image with highpass part of another image.
    def hybridImage(self ,highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
        
       highPassed = self.highPass(highFreqImg, sigmaHigh)
       lowPassed = self.lowPass(lowFreqImg, sigmaLow)    
       return highPassed + lowPassed
        ###___________________HISTOGRAM__________
    def LoadImage(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG *.JPEG *.JFIF);;img file (*.jpg *.PNG *.JPEG *.JFIF)")
        if self.fileName:
            pixmap = QPixmap(self.fileName)
            self.pixmap = pixmap.scaled(256,256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) 
            self.input_img =mpimg.imread(self.fileName)
            self.gray =cv2.imread(self.fileName,0)
            #plt.imshow(self.color_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
            self.ui.label_histograms_input.setPixmap(self.pixmap)
            self.ui.label_histograms_input.show
            #to show size of the image 
            pixels = asarray(self.input_img)
            print(pixels.shape)
            self.ui.lineEdit_4.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")
        
        
#    def rgb2gray(self,rgb_image):
#        return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])  # ... mean  all rgb values 


#___________________________________________________________________________________________________________
    def Draw_histogram(self):
         self.color_of_histogram = str(self.ui.comboBox_9.currentText())
         print(self.color_of_histogram)
         img = Image.open(self.fileName).convert('YCbCr')
         equized_image=Image.open("equlized_image.jpg").convert('YCbCr')
         #Convert our image to numpy array, calculate the histogram
         self.img = np.array(img)
         self.equized_image = np.array(equized_image)    
         if self.color_of_histogram=="Gray ":
                self.ui.input_histogram.clear()
                self.ui.output_histogram.clear()
                img_arr = np.asarray(self.gray)
                flat = img_arr.flatten()
                gray_hist = self.make_histogram(flat)
                gray_equalized_hist=self.make_histogram(self.new_equalized_img)
                plotWindow = self.ui.input_histogram
                plotWindow.plot(gray_hist, pen='w')
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot(gray_equalized_hist, pen='w')
    
               
         elif self.color_of_histogram=="Red":
                self.ui.input_histogram.clear()
                self.ui.output_histogram.clear()
            
                # Extract 2-D arrays of the RGB channels: red
                red_pixels=self.img[:,:,0]
                red_pixels_equalized=self.equized_image[:,:,0]
                # Flatten the 2-D arrays into 1-D
                red_vals =  red_pixels.flatten()
                red_vals_equalized =red_pixels_equalized.flatten()
                red_hist = self.make_histogram_of_Color_im(red_vals,self.img)
                red_hist_equalized=self.make_histogram_of_Color_im(red_vals_equalized,self.equized_image)
                plotWindow = self.ui.input_histogram
                plotWindow.plot( red_hist, pen='r')
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot( red_hist_equalized, pen='r')
    
     
         elif self.color_of_histogram=="Green ":
                self.ui.input_histogram.clear()
                self.ui.output_histogram.clear()
                # Extract 2-D arrays of the RGB channels: green
                green_pixels=self.img[:,:,1]
                green_pixels_equalized=self.equized_image[:,:,1]
                # Flatten the 2-D arrays into 1-D
                green_vals =green_pixels.flatten()
                green_vals_equalized =green_pixels_equalized.flatten()
                green_hist = self.make_histogram_of_Color_im(green_vals,self.img)
                green_hist_equalized=self.make_histogram_of_Color_im(green_vals_equalized,self.equized_image)
                plotWindow = self.ui.input_histogram
                plotWindow.plot( green_hist, pen='g')
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot(green_hist_equalized, pen='g')
                
         elif self.color_of_histogram=="Blue ":
                self.ui.input_histogram.clear()
                self.ui.output_histogram.clear()
                
                # Extract 2-D arrays of the RGB channels: blue
                blue_pixels=self.img[:,:,1]
                blue_pixels_equalized=self.equized_image[:,:,1]
                # Flatten the 2-D arrays into 1-D
                blue_vals =blue_pixels.flatten()
                blue_vals_equalized =blue_pixels_equalized.flatten()
                blue_hist = self.make_histogram_of_Color_im(blue_vals,self.img)
                blue_hist_equalized=self.make_histogram_of_Color_im(blue_vals_equalized,self.equized_image)
                plotWindow = self.ui.input_histogram
                plotWindow.plot( blue_hist, pen='b')
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot(blue_hist_equalized, pen='b')
                
         else :
            self.ui.input_histogram.clear()
            self.ui.output_histogram.clear()
            colors = ('r', 'g', 'b')
            channel_ids = (0, 1, 2)
            # create the histogram plot, with three lines, one for each color
            for channel_id, c in zip(channel_ids, colors):
                                print (channel_id)
                                print(c)
                                # Extract 2-D arrays of the RGB channels: blue
                                color_pixels=self.img[:,:,channel_ids]
                                color_pixels_equalized=equized_image[:,:,channel_ids]    
                                # Flatten the 2-D arrays into 1-D
                                color_vals =color_pixels.flatten()
                                color_vals_equalized =color_pixels_equalized.flatten()
                                color_hist = self.make_histogram_of_Color_im(color_vals,self.img)
                                color_hist_equalized=self.make_histogram_of_Color_im(color_vals_equalized,equized_image)
                                plotWindow = self.ui.input_histogram
                                plotWindow.plot( color_hist, pen=c)
                                plotWindow2 = self.ui.output_histogram
                                plotWindow2.plot(color_hist_equalized, pen=c)
                                
#______________________________________Build histogram______________________________________________
    def make_histogram(self,img):
        # Take a flattened greyscale image and create a historgram from it 
        histogram = np.zeros(256, dtype=int)
        for i in range(img.size):
            histogram[img[i]] += 1
        return histogram                                 

    def make_histogram_of_Color_im(self, y_vals,img):
        """ Take an image and create a historgram from it's luma values """
        histogram = np.zeros(256, dtype=int)
        for y_index in range(y_vals.size):
            histogram[y_vals[y_index]] += 1
        return histogram                                

#______________________________________check options ________________________________________                               
    def check_Effect_to_image(self):
        effect= str(self.ui.comboBox_7.currentText())
        print(effect)
        if effect== "Normalize" :
              self.check_color_or_Gray_Normalize()
        elif effect=="Equalize ":
               self.check_color_or_Gray_Equalize()
        elif effect=="Global Thresholding ":
              thre=(self.ui.lineEdit_10.text())
              thre=float(thre)
              print(thre)
              self.global_threshold(thre)
        else :
             ratio = (self.ui.lineEdit_10.text())
             ratio=float(ratio)
             size = (self.ui.lineEdit_9.text())
             size=int(size)
             self.Local_thresholding(size ,ratio)
            
        
        
    def check_color_or_Gray_Normalize(self):
        pixels = asarray(self.input_img)
        pixels = pixels.astype('float32')
        #check if its RGB or Gray
        try:
            if (pixels.shape[2] == 3):
                self.normalize_color_image(self.input_img)
                print("3")
            
        #elif (pixels.shape[2] == 1):
        except IndexError:  #(pixels.shape[2] == None):
                 self.normalize_grey_image(self.input_img)
                 print("1")
    


    def check_color_or_Gray_Equalize(self):
        pixels = asarray(self.input_img)
        pixels = pixels.astype('float32')
        print(pixels.shape)
        #check if its RGB or Gray
        try:
            if (pixels.shape[2] == 3):
                self.equalize_color_image()
                print("3")
            
        #elif (pixels.shape[2] == 1):
        except IndexError:  #(pixels.shape[2] == None):
                 self.Equilize_grey_Image()
                 print("1")
#______________________________choose curves_____________________________________________________                 
    def  Choose_curve(self):
          curve= str(self.ui.comboBox_8.currentText())
          print(curve)
          if curve== "Cumlative curve " :
              #print("cumaltive")
              self.draw_curve=1
              print(self.draw_curve)
              self.check_color_or_Gray_Equalize()
        
          else :
             self.distribution_curve()
             #print("distribution")
             
             
    def  distribution_curve(self):
        img_arr = np.asarray(self.input_img)
        #mg_float = img_arr.astype('float32')
        flat = img_arr.flatten()
        sns.distplot(flat, hist=True, kde=True, color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 5})         
#        plotWindow2 = self.ui.output_histogram
#        plotWindow2.plot(flat, pen='w') 
        """note that this is plot in consol """ 
        
    def make_cumsum(self,histogram):
        # Create an array that represents the cumulative sum of the histogram 
        cumsum = np.zeros(256, dtype=int)
        cumsum[0] = histogram[0]
        for i in range(1, histogram.size):
            cumsum[i] = cumsum[i-1] + histogram[i]
        return cumsum     
    def make_cumsum_of_Color_im(self,histogram):
        """ Create an array that represents the cumulative sum of the histogram """
        cumsum = np.zeros(256, dtype=int)
        cumsum[0] = histogram[0]
        for i in range(1, histogram.size):
            cumsum[i] = cumsum[i-1] + histogram[i]
        return cumsum    
            
#______________________________________Normalize image_______________________________________________
    
    def normalize_grey_image(self,img):
        #image = Image.open(img).convert('L')
        pixels = asarray(img)
        #img_h = pixels.shape[0]
        #img_w = pixels.shape[1]
        pixels = pixels.astype('float32')
        
        #need only one as its only one channel
        old_min = pixels.min()
        old_max = pixels.max()
        old_range = old_max - old_min
        
        for rows in range (pixels.shape[0]):
            for col in range (pixels.shape[1]):
                pixels[rows, col]  = (pixels[rows, col] - old_min) / old_range
 
        pixels=np.array(pixels)
        #plt.imshow(pixels)
        img=array2qimage(pixels*255)
        #img.show()
        #img.save('norm_grey_img.png')
        pixmap = QPixmap(img)
        self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.ui.label_histograms_output.setPixmap(self.pixmap)
        self.ui.label_histograms_output.show
    
    
#____________normalize color image________________
        
    def normalize_color_image(self,img):
        
        #read image as array to take size of image
        pixels = asarray(img)
        pixels = pixels.astype('float32')
        #print (pixels.shape)
        #print(pixels.shape[1])
        
        #get minimum and maximum intensity values of image(for each channel) and set a range out of them
        old_minR = pixels[..., 0].min()
        old_minG = pixels[..., 1].min()
        old_minB = pixels[..., 2].min()
        
        old_maxR = pixels[..., 0].max()
        old_maxG = pixels[..., 1].max()
        old_maxB = pixels[..., 2].max()
        
        #Or: max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]), np.amax(img[:,:,2])])
        
        old_rangeR = old_maxR - old_minR
        old_rangeG = old_maxG - old_minG
        old_rangeB = old_maxB - old_minB
        
        #formula for normalization from (0-255): Inew = (Iold-old_min) * (new_range/old_range) + new_min
        #formula for normalization from (0-1): Inew = (Iold-old_min) /old_range
        
        #for each pixel change its intensity using the formula above
        for rows in range (pixels.shape[0]):
            for col in range (pixels.shape[1]):
                pixels[rows, col,0]  = (pixels[rows, col,0] - old_minR) / old_rangeR
                pixels[rows, col,1]  = (pixels[rows, col,1] - old_minG) / old_rangeG
                pixels[rows, col,2]  = (pixels[rows, col,2] - old_minB) / old_rangeB
                #print(pixels[rows, col])
        
        pixels=np.array(pixels)
        #plt.imshow(pixels)
        iamge=array2qimage(pixels*255)
        pixmap = QPixmap(iamge)
        self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.ui.label_histograms_output.setPixmap(self.pixmap)
        self.ui.label_histograms_output.show
        
    
  
       
#___________________________________________Equalize image______________________________________________________
    def Equilize_grey_Image(self):
        img_arr = np.asarray(self.gray)
        #img_float = img_arr.astype('float32')
        self.img_h = img_arr.shape[0]
        self.img_w = img_arr.shape[1]
        flat = img_arr.flatten()
        #hist = np.histogram(flat, bins=256, range=(0, 1))
        hist = self.make_histogram(flat)
#        plotWindow2 = self.ui.output_histogram
#        plotWindow2.plot(hist, pen='w')
        cumilative_curve = self.make_cumsum(hist)
        if (self.draw_curve==1):
                #print("yyyyyyyyyyyyyes")
                self.ui.output_histogram.clear()
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot(cumilative_curve, pen='w') 
        else:        
            new_intensity = self.make_mapping(cumilative_curve, self.img_h,self.img_w)
            self.new_equalized_img = self.apply_mapping(flat,new_intensity) #new_img is 1D
            #hist_equ= self.make_histogram(self.new_equalized_img)
      
            self.output_image = Image.fromarray(np.uint8(self.new_equalized_img.reshape((self.img_h,self.img_w))))
            #plt.imshow(self.output_image)
            image=ImageQt(self.output_image)
            self.pix=QPixmap.fromImage(image).scaled(256, 256,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.ui.label_histograms_output.setPixmap(self.pix)
      
      
    def make_mapping(self,cumsum, img_h, img_w):
        # Create a mapping s.t. each old colour value is mapped to a new
         #   one between 0 and 255 
        mapping = np.zeros(256, dtype=int)
    
        grey_levels = 256
        for i in range(grey_levels):
            mapping[i] = max(0, round((grey_levels*cumsum[i])/(img_h*img_w))-1)
        return mapping

    #create the mapped image
    #new_image[i]=mapping[img[i]]
    #The output of this function is an array containing the pixel values of the new, histogram equalized image! 
    #All that needs doing now is restructuring and rendering / saving it
    def apply_mapping(self,img, mapping):
        # Apply the mapping to our image 
        new_image = np.zeros(img.size, dtype=int)
        for i in range(img.size):
            new_image[i] = mapping[img[i]]
        return new_image
    
#____________equalize color image________________________ 
        
    def equalize_color_image(self):
       # Load image, convert it to YCbCr format ten store width and height into constants
        img = Image.open(self.fileName).convert('YCbCr')
        self.IMG_W, self.IMG_H = img.size
        
        # Convert our image to numpy array, calculate the histogram, cumulative sum,
        # mapping and then apply the mapping to create a new image
        img = np.array(img)
        y_vals = img[:,:,0].flatten()
        histogram = self.make_histogram_of_Color_im( y_vals,img)
#        plotWindow2 = self.ui.output_histogram
#        plotWindow2.plot( histogram, pen='w')
        
        cumilative_curve = self.make_cumsum_of_Color_im(histogram)
        if (self.draw_curve==1):
                #print("yyyyyyyyyyyyyes")
                self.ui.output_histogram.clear()
                plotWindow2 = self.ui.output_histogram
                plotWindow2.plot( cumilative_curve, pen='w') 
        else:      
            mapping = self.make_mapping_of_Color_im(histogram,  cumilative_curve)
            new_image = self.apply_mapping_of_Color_im(img, mapping)
            #new_histogram=self.make_histogram_of_Color_im(new_image)
            #plotWindow2 = self.ui.output_histogram
            #plotWindow2.plot( cumsum, pen='w')
            # Save the image
            self.equalized_color_image = Image.fromarray(np.uint8(new_image), "YCbCr")
            self.equalized_color_image.save("equlized_image.jpg")
            #self.pix=QPixmap.fromImage(image).scaled(256, 256,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.ui.label_histograms_output.setPixmap(QPixmap("equlized_image.jpg").scaled(256, 256,QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
          
       
    def make_mapping_of_Color_im(self,histogram, cumsum):
        """ Create a mapping s.t. each old luma value is mapped to a new
            one between 0 and 255. Mapping is created using:
             - M(i) = max(0, round((luma_levels*cumsum(i))/(h*w))-1)
            where luma_levels is the number of luma levels in the image """
        mapping = np.zeros(256, dtype=int)
        luma_levels = 256
        for i in range(histogram.size):
            mapping[i] = max(0, round((luma_levels*cumsum[i])/(self.IMG_H*self.IMG_W))-1)
        return mapping
    def apply_mapping_of_Color_im(self,img, mapping):
        """ Apply the mapping to our image """
        new_image = img.copy()
        new_image[:,:,0] = list(map(lambda a : mapping[a], img[:,:,0]))
        return new_image


#_________________________________________global thresholding ____________________________________________
        
    def global_threshold (self,threshold):
        gray_img =cv2.imread(self.fileName,0)  
        img = asarray( gray_img)
        #img = img.astype('float32')
        print(img)
        for row in range (img.shape[0]):
            for col in range (img.shape[1]):
                if img[row, col] < threshold:
                    img[row, col] = 0
                else:
                    img[row, col] = 255
        #new_img = img.astype(np.uint8) #made it save safely
        pixels=np.array(img)
        iamge=array2qimage(pixels)
        #new_img = Image.fromarray(iamge)
        pixmap = QPixmap(iamge)
        self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.ui.label_histograms_output.setPixmap(self.pixmap)
        self.ui.label_histograms_output.show
        threshold=0
        
        
#______________________________________local thresholding_________________________________________        

    def Local_thresholding(self,size,ratio):
         gray_img =cv2.imread(self.fileName,0)  
         image_array = asarray( gray_img)
         #gray_img =self.rgb2gray(self.input_img)
         #pixels = asarray(gray_img)
         #pixels = pixels.astype('float32')
         #image_array = np.array(self.input_img)
         print(image_array)        
         new_array=np.ones(shape=(len(image_array),len(image_array[0])))
         for row  in range( len(image_array)- size + 1 ):
             for col  in range( len(image_array[0]) - size + 1 ):
                 #for row1  in range( len(thelist)- size + 1 ):
                 window=image_array[row:row+size,col:col+size]
                 minm=window.min()
                 maxm=window.max()
                 #print(minm,maxm)
                 threshold =minm+((maxm-minm)*ratio)
                 #print(threshold)
                 if window[0,0] < threshold:
                     new_array[row,col]=0
                     print('ok1')
                        #new_array.append(0)
                     #print ('t')
                    #print('x')      
                 else:
                    new_array[row,col]=1
                    print('ok2')
         print(new_array)           
         pixels=np.array(new_array)
         #gray2qimage
         iamge=array2qimage(pixels*50)
        #new_img = Image.fromarray(iamge)
         pixmap = QPixmap(iamge)
         self.pixmap = pixmap.scaled(256, 256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
         self.ui.label_histograms_output.setPixmap(self.pixmap)
         self.ui.label_histograms_output.show
         print(new_array)
         #plt.imshow(new_array, cmap = plt.get_cmap('gray'))    
             
        ######________________FILTERS_________________
        
    def button_clicked(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.JPEG *.JFIF);;img file (*.jpg *.JPEG *.JFIF)")
        if self.fileName:
            self.ui.label_filters_input.clear()
            pixmap = QPixmap(self.fileName)
            self.pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) 
            self.color_img =mpimg.imread(self.fileName)
            self.gray_img =self.rgb2gray(self.color_img)
            self.ui.lineEdit.setText(""+('image')+"")
            self.ui.lineEdit_2.setText(""+str(self.gray_img.shape[0])+""+str('x')+""+str(self.gray_img.shape[1])+"")
            #print(self.fileName[66:100])
            #for i in range (self.fileName[6]):
            self.Display_image() 
#            
##    def rgb2gray(self,rgb):
##        #return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
##          r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
##          gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
##          return gray
    def Display_image(self):
#        if self.fileName[66:100]=="House.jpg" or self.fileName[66:100]=="Pyramids2.jpg" or self.fileName[66:100]=="some-pigeon.jpg":
#             self.input_iamge=np.array(self.gray_img)
#        else:
#             self.input_iamge=np.array(self.gray_img*200)
#        #print (input_iamge)
        
        self.input_iamge=np.array(self.gray_img).astype(np.int32)
        self.input_iamge1=qimage2ndarray.array2qimage(self.input_iamge)
        self.input_iamge=QPixmap(self.input_iamge1)
        self.ui.label_filters_input.setPixmap(self.input_iamge)
        self.ui.label_filters_input.show()
#
#        
    def corr(self,mask):
        row,col=self.gray_img.shape
        m,n=mask.shape
        new=np.zeros((row+m-1,col+n-1))
        n=n//2
        m=m//2
        filtered=np.zeros(self.gray_img.shape)
        new[m:new.shape[0]-m,n:new.shape[1]-n]=self.gray_img
        for i in range (m,new.shape[0]-m):
            for j in range (n,new.shape[1]-n):
                temp=new[i-m:i+m+1,j-m:j+m+1]
                result=temp*mask
                filtered[i-m,j-n]=result.sum()      
        return filtered
#
    def gaussian_2(self,m,n,sigma):
        gaussian_2=np.zeros((m,n))
        m=m//2
        n=n//2
        for x in range (-m,m+1):
            for y in range (-n,n+1):
                x1=sigma*math.sqrt(2*np.pi)
                x2=np.exp(-(x**2+y**2)/(2*sigma**2))
                gaussian_2[x+m,y+n]=(1/x1)*x2  
        return gaussian_2

    def gaussian_filter_2(self,m,n,sigma):
        g=self.gaussian_2(m,n,sigma)
        n=self.corr(g)
        return n
#    
    def mean(self,k):
        #meanFilter=[]
        n=k*k
        print(n)
        meanFilter=(np.ones((k,k)))*(1/n)
        #print(meanFilter)
        filt=self.corr(meanFilter)
        #print("9898")
        return filt
#    
    def median_filter(self,mask):
        m,n=self.gray_img.shape
        median = np.zeros((m,n))
        temp = []
        mask_center = mask // 2  # to get the center value

    
        for i in range(m):
            for j in range(n): #(i,j)for loop for image
                for u in range(mask):
                    for v in range(mask):#(u,v)for loop for image 
                       if (i + u - mask_center < 0) or (i + u - mask_center > m - 1):
                            temp.append(0)
                       elif (j + u - mask_center < 0) or (j + mask_center > n - 1):
                            temp.append(0)
                       else:                     
                         temp.append(self.gray_img[i + u - mask_center][j + v - mask_center])
    
                temp.sort()
                median[i][j] = temp[len(temp) // 2]
                temp = []
        return median

    def prewitt(self):
        n,m = np.shape(self.gray_img)
        Gx= np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Gy= np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
         
        filt= np.zeros(shape=(n, m))
        
        for i in range(n - 2):
            for j in range(m - 2):
                gx = np.sum(np.multiply(Gx, self.gray_img[i:i + 3, j:j + 3])) 
                gy = np.sum(np.multiply(Gy, self.gray_img[i:i + 3, j:j + 3])) 
                filt[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
        
        return filt



    def robert(self):
            n,m= np.shape(self.gray_img) 
            Gx = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
            Gy = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
            filt= np.zeros(shape=(n, m)) 
            
            for i in range(n - 2):
                for j in range(m - 2):
                    gx = np.sum(np.multiply(Gx, self.gray_img[i:i + 3, j:j + 3]))  
                    gy = np.sum(np.multiply(Gy, self.gray_img[i:i + 3, j:j + 3]))  
                    filt[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  
            
            return  filt
#
#
    def sobel_2(self):
            n,m= np.shape(self.gray_img)
            Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
              
            filt= np.zeros(shape=(n, m))  
            
            for i in range(n - 2):
                for j in range(m - 2):
                    gx = np.sum(np.multiply(Gx, self.gray_img[i:i + 3, j:j + 3]))  
                    gy = np.sum(np.multiply(Gy, self.gray_img[i:i + 3, j:j + 3]))  
                    filt[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  
            
            return  filt
#    
#    
#    
    def sobel_filter_for_canny_2(self,img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
#    
    def non_max_suppression_2(self,img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z
#    
    def threshold_2(self,img):
#        lowThreshold = 0.05
#        self.highThreshold = 0.1
        LhighThreshold = img.max() * self.highThreshold;
        LlowThreshold = self.highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= LhighThreshold)
        zeros_i, zeros_j = np.where(img < LlowThreshold)

        weak_i, weak_j = np.where((img <= LhighThreshold) & (img >= LlowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)
#
    def hysteresis_2(self,img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img
#
#
#    
#    
#
############################canny
    def canny_filter_2(self):
              #self.gray_img
#              
            #self.gray_img =cv2.imread(self.fileName,0)
            canny_img_final = []
            input_size=(self.ui.lineEdit_3.text())
            #plt.imshow(self.gray_img)
            guass=self.gaussian_2(int(input_size),int(input_size),1)
            self.img_smoothed = convolve(self.gray_img,guass)
            self.gradientMat, self.thetaMat =self.sobel_filter_for_canny_2(self.img_smoothed)
            self.nonMaxImg =self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg =self.threshold_2(self.nonMaxImg)
            img_final = self.hysteresis_2(self.thresholdImg)
            canny_img_final.append(img_final)
            print(canny_img_final)
            self.visualize(canny_img_final, 'gray')
#
    def visualize_2(self,imgs, format=None, gray=False):
            #plt.figure(figsize=(20, 40))
            for i, img in enumerate(imgs):
                if img.shape[0] == 3:
                    print("rrrrrrrrrrrrr")
                    img = img.transpose(1,2,0)
                img=Image.fromarray(np.uint8(img))    
                img.save("canny_edges.jpg")    
#                plt_idx = i+1
#                plt.subplot(2, 2, plt_idx)
#                plt.imshow(img, format)
#            plt.show()
##            
#        
#     
########################################## add noise#################################    
#    
    def gaussian_noise( self,mu, sigma, im_size ):
        randGaussian=np.random.normal( mu, sigma, im_size) #np.random.normal Gaussian noise
        return randGaussian
    
    def im_gaussian_noise(self,mu, sigma):
        g_noise= self.gaussian_noise(mu,sigma, self.gray_img.shape)
        img_w_g_noise = self.gray_img + g_noise
        return img_w_g_noise
    
            
    def Random_Uniform(self,percent):
        img_noisy=np.zeros(self.gray_img.shape)
        uniform = np.random.random(self.gray_img.shape) 
        cleanPixels_ind=uniform > percent
        noise = (uniform <= (percent)); 
        img_noisy[cleanPixels_ind]=self.gray_img[cleanPixels_ind]
        img_noisy[noise] = 0.7
        return img_noisy           
            
    
    def salt_pepper_noise(self,percent):
        img_noisy=np.zeros(self.gray_img.shape)
        salt_pepper = np.random.random(self.gray_img.shape) # Uniform distribution
        cleanPixels_ind=salt_pepper > percent
        #NoisePixels_ind=salt_pepper <= percent
        pepper = (salt_pepper <= (0.5* percent)); # pepper < half percent
        
        salt = ((salt_pepper <= percent) & (salt_pepper > 0.5* percent)); 
        img_noisy[cleanPixels_ind]=self.gray_img[cleanPixels_ind]
        img_noisy[pepper] = 0
        img_noisy[salt] = 1
        return img_noisy
  
    def show_filters(self): 
        self.filters = str(self.ui.comboBox_2.currentText())
        
        if self.filters=="Gaussian":
            #self.input_iamge=array2qimage.qimage2ndarray(self.input_iamge)
            input_size=(self.ui.lineEdit_3.text())
            self.filter_img =self.gaussian_filter_2(int(input_size),int(input_size),2)
            self.filter=np.array(self.filter_img)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()
        
        elif self.filters=="Mean":   
                input_size=(self.ui.lineEdit_3.text())
                self.filter_img =self.mean(int(input_size))
                self.filter=np.array(self.filter_img)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show() 
                 
#       
        elif self.filters=="Median": 
            input_size=(self.ui.lineEdit_3.text())
            self.filter_img =self.median_filter(int(input_size))
            self.filter=np.array(self.filter_img)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()  
        
        
        elif self.filters=="Prewitt":    
            self.filter_img =self.prewitt()
            self.filter=np.array(self.filter_img)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()
            
              
        elif self.filters=="Roberts":    
            self.filter_img =self.robert()
            self.filter=np.array(self.filter_img)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()
            
        elif self.filters=="Sobel":    
            self.filter_img =self.sobel_2()
            self.filter=np.array(self.filter_img)
            self.input_iamge=qimage2ndarray.array2qimage(self.filter)
            self.output_iamge=QPixmap(self.input_iamge)
            self.ui.label_filters_output.setPixmap(self.output_iamge)
            self.ui.label_filters_output.show()  
            
        elif self.filters=="Canny": 
            self.canny_filter_2()
            self.ui.label_filters_output.setPixmap(QPixmap("canny_edges.jpg"))
  
        else:
            print("2")
    #def add_noise(self):
    def add_noise(self): 
            self.filters = str(self.ui.comboBox.currentText())
            
            if self.filters=="Gaussian":
                #self.input_iamge=array2qimage.qimage2ndarray(self.input_iamge)
                self.filter_img =self.im_gaussian_noise(0,0.3)
                self.filter=np.array(self.filter_img)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show()
            
            elif self.filters=="Uniform":    
                self.filter_img =self. Random_Uniform(0.3)
                self.filter=np.array(self.filter_img)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show() 
                print("1212")
    #       
            elif self.filters=="Salt-papper":    
                self.filter_img =self.salt_pepper_noise(0.3)
                self.filter=np.array(self.filter_img)
                self.input_iamge=qimage2ndarray.array2qimage(self.filter)
                self.output_iamge=QPixmap(self.input_iamge)
                self.ui.label_filters_output.setPixmap(self.output_iamge)
                self.ui.label_filters_output.show()  
    
     ######________________SIFT_________________    
        
    def button_SIFTA(self):  
        self.fileName_A, self.filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName_A:
            
            pixmap = QPixmap(self.fileName_A)
            self.pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) 
            #self.ui.lineEdit.setText(""+('image')+"")
            self.img_SIFRA=mpimg.imread(self.fileName_A)
            self.ui.lineEdit_SIFT_size_A.setText(""+str(self.img_SIFRA.shape[0])+""+str('x')+""+str(self.img_SIFRA.shape[1])+"")
            self.ui.label_SIFT_input_A.setPixmap(self.pixmap)####for input image 1
            self.ui.label_SIFT_input_A.show
            #self.Display_SIFTA()
            
            
            
            
    def button_SIFTB(self):  
        self.fileName_B, self.filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName_B:
            
            pixmap = QPixmap(self.fileName_B)
            self.pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.img_SIFRB=mpimg.imread(self.fileName_B)
            #self.ui.lineEdit.setText(""+('image')+"")
            self.ui.lineEdit_SIFT_size_B.setText(""+str(self.img_SIFRB.shape[0])+""+str('x')+""+str(self.img_SIFRB.shape[1])+"")
            self.ui.label_SIFT_input_B.setPixmap(self.pixmap)#####for input image 2
            self.ui.label_SIFT_input_B.show
            #self.Display_SIFTB()      
       
            
    def combine_img_with_pattern_GUI(self):
                    stat_t=time.process_time()
                    #self.ui.lineEdit_current_process.setText(" Show consol to see steps of alghorithm running  ")
                    img = np.array(Image.open(self.fileName_A))
                    img,ratio = self.obj.sift_resize(img)
                    image_patterns = []
                    pattern,_ = self.obj.sift_resize(np.array(Image.open(self.fileName_B)), ratio )
                    image_patterns.append(pattern)
                    image_patterns.append(rotate(pattern, 90))
                    image_gray = self.obj.rgb2gray( img )
                    image_patterns_gray = [ self.obj.rgb2gray( img ) for img in image_patterns ]
                    img_sift = self.pipeline_GUI(image_gray)
                    image_patterns_sift = []
                    for pattern in image_patterns_gray:
                        image_patterns_sift.append( self.pipeline_GUI( pattern ))
                    for i in range(len(image_patterns)):
                            self.patt_N=i
                            pattern = image_patterns[i]
                            pattern_sift = image_patterns_sift[i]
                            self.match(img, img_sift[0], img_sift[1], pattern, pattern_sift[0], pattern_sift[1]) 
                
                    t=time.process_time()-stat_t
                    self.ui.lineEdit_SIFT_Time_Elapsed.setText(""+str(t)+"")
                    
                    
    def match(self,img_a, pts_a, desc_a, img_b, pts_b, desc_b):
            #self.ui.lineEdit_current_process.setText("feature matching for "+self.current_image+"....")
            img_a, img_b = tuple(map( lambda i: np.uint8(i*255), [img_a,img_b] ))
            
            desc_a = np.array( desc_a , dtype = np.float32 )
            desc_b = np.array( desc_b , dtype = np.float32 )
        
            pts_a = self.obj.kp_list_2_opencv_kp_list(pts_a)
            pts_b = self.obj.kp_list_2_opencv_kp_list(pts_b)
        
            # create BFMatcher object
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc_a,desc_b,k=2)
            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.25*n.distance:
                    good.append(m)
            img_match = np.empty((max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)
            cv2.drawMatches(img_a,pts_a,img_b,pts_b,good, outImg = img_match,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#            plt.figure(figsize=(20,20))
#            plt.imshow(img_match)
#            plt.show()
            pixels=np.array(img_match)
            im=array2qimage(pixels)
            pixmap = QPixmap(im)
            if self.patt_N==0:
                  self.ui.label_SIFT_Features_Matching.setPixmap(pixmap)
            elif self.patt_N==1:
                 self.ui. label_SIFT_Features_Matching_with_rotate.setPixmap(pixmap)
                 self.ui.lineEdit_current_process.setText("Algorithm ended successfully")
           
            
            
        
    def pipeline_GUI(self,input_img):
                if self.iteration==0:
                   self.current_image="wholeImage"
                elif self.iteration==1:
                   self.current_image="pattern1"
                else :
                   self.current_image="pattern2"
                #self.ui.lineEdit_current_process.setText("finding initalize parmeter for "+self.current_image+".......")
                print("finding initalize parmeter for "+self.current_image+".......")
                self.obj.initalize_par()
                self.im=input_img
                img_max = input_img.max()
                print("finding image dog for "+self.current_image+"....")
                self.img_dogs, self.img_octaves =self.obj.image_dog(input_img)
                #self.visualze_scale_dog()
                print("finding dog keypoints for "+self.current_image+"....")
                keypoints =self.obj.dog_keypoints( self.img_dogs , img_max , 0.03 )
                # Each keypoint has four parameters (i,j,scale_idx,orientation)
                print("finding dog keypoints orientations for "+self.current_image+"....")
                self.keypoints_ijso = self.obj.dog_keypoints_orientations( self.img_octaves , keypoints , 36 )
                #self.visualize_points_orirntation()
                print("extracting sift descriptors128 for "+self.current_image+"....")
                points,descriptors = self.obj.extract_sift_descriptors128(self.img_octaves , self.keypoints_ijso , 8)
                print("Ending extract sift descriptors128 for "+self.current_image+"....")
                print("matching for "+self.current_image+"...")
                self.iteration=self.iteration+1
                return points, descriptors   
        
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    application =CV()
    application.show()
  
    
  
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
