import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
import math
#from astropy.convolution import convolve



###_____________________________________________________Al-Shimaa____________________________###

#____________________________________________________handle Image_______________________________________#
class imarray(object):

	def __init__(self,path=None,mode='L'):
		if path == None:
			return
		try :
			self.__image = imread(path, mode = mode)

		except :
			print("Error! : %s"%path)
			return
		try :
			self.__image = np.asarray(self.__image)
			self.__dimension = self.__image.shape
			self.__type = path.split(".")[-1]
		except :
			print("Internal Error! Image file not supported")
    
#	def repr(self):
#		
#		return repr(self.__image)
#
#	def cmp(self,img):
#		return cmp(self,img)
    

	def __getitem__(self,coordinates):
		return self.__image[coordinates]
    
	def load(self, image) :
		image = np.asarray(image,dtype=np.uint8)
		if len(image.shape) == 2 :
			self.__image = image
#			try :
#				self.__dimension = self.__image.shape
#			except :
#				print(" Error!")
		else :
			print(" Error")

	def getShape(self):
		return self.__dimension
	shape = property(getShape)

	def getExtension(self):
		return self.__type
	ext = property(getExtension)
#
#	def displayImage(self,mode='Greys_r'):
##		try:
##			plt.imshow(self.__image,cmap=mode)
##		except:
##			print("Image could not be displayed")
##			return
#		plt.show()
#	disp = property(displayImage)

#	def save(self,name):
#		plt.imsave(name,self.__image)

	def convolve(self,mask):
		mask = np.asarray(mask,dtype=np.float32)
		if len(mask.shape) != len(self.__dimension):
			print("Invalid Mask Dimensions")
		m,n = mask.shape
		padY = int(np.floor(m/2))
		padX = int(np.floor(n/2))
		M,N = self.__dimension
		padImg = np.ones((M+padY*2,N+padX*2))*128
		fImage = np.zeros((M+padY*2,N+padX*2))
		padImg[padY:-padY,padX:-padX] = self.__image

		for yInd in range(padY,M+padY):
			for xInd in range(padX,N+padX):
				fImage[yInd,xInd] = sum(sum(padImg[yInd-padY:yInd+m-padY,xInd-padX:xInd+n-padX]*mask))

		return fImage[padY:-padY,padX:-padX]
#________________________________________________________gaussian & sobel & laplacian __________________________________#

def gaussian(m,n,sigma):
    gaussian=np.zeros((m,n))
    m=m//2
    n=n//2
    for x in range (-m,m+1):
        for y in range (-n,n+1):
            x1=sigma*math.sqrt(2*np.pi)
            x2=np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+m,y+n]=(1/x1)*x2  
    return gaussian

def gaussian_filter(m,n,sigma,img):
    g=gaussian(m,n,sigma)
    img.convolve(g)
    return img
    

def edge(img,threshold):
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

def detectCircles(img,threshold,region,radius):
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

def displayCircles(A):
    img = imread(file_path)
    fig = plt.figure()
    plt.imshow(img)
    circleCoordinates = np.argwhere(A)                                          
    circle = []
    for r,x,y in circleCoordinates:
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
          
    plt.show()

file_path = './images/Coins.jpg'
img = imarray(file_path)
image = gaussian_filter(3,3,1,img)                                                
image = edge(image,128)                                               
image = detectCircles(image,13,15,[40,10])
displayCircles(image)