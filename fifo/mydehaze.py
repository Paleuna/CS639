import cv2
import math
import numpy as np
import statistics
import sys
import os


#Section 3
def getDark(img,kernal_size):
    dc = np.minimum(np.minimum(img[:,:,0],img[:,:,1]),img[:,:,2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernal_size,kernal_size))
    dark = cv2.erode(dc,kernel)
    return dark

#Section 4.1
def getTransmission(img,A,sz):
    omega = 0.95;
    
    R = img[:,:,0] / A[0,0]
    G = img[:,:,1] / A[0,1]
    B = img[:,:,2] / A[0,2]

    normalized = np.dstack((R,G,B)) 

    transmission = 1 - omega*getDark(normalized,sz)
    return transmission

#modified Section 4.2
#inspired by https://github.com/martiansideofthemoon/blind-dehazing/blob/master/dark_prior/guidedfilter.py
def Guidedfilter(img,transmission,filter_size=60,epsilion=0.0001):
    imageMean= cv2.boxFilter(img,cv2.CV_64F,(filter_size, filter_size))
    transMean = cv2.boxFilter(transmission, cv2.CV_64F,(filter_size, filter_size))
    itMean = cv2.boxFilter(img*transmission,cv2.CV_64F,(filter_size, filter_size))
    itCov = itMean - imageMean*transMean 
    imageVar   = cv2.boxFilter(img*img,cv2.CV_64F,(filter_size, filter_size))- imageMean*imageMean

    a = itCov/(imageVar + epsilion)
    b = transMean - a*imageMean

    aMean = cv2.boxFilter(a,cv2.CV_64F,(filter_size, filter_size))
    bMean = cv2.boxFilter(b,cv2.CV_64F,(filter_size, filter_size))

    res = aMean*img + bMean
    return res

def TransmissionRefine(img,transmission):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    t = Guidedfilter(gray,transmission,filter_size=60,epsilion=0.0001)

    return t;


#Section 4.3
def Recover(img,t,A,t_default = 0.1):
    t = cv2.max(t,t_default)
    R = (img[:,:,0]-A[0,0])/t + A[0,0]
    G = (img[:,:,1]-A[0,1])/t + A[0,1]
    B = (img[:,:,2]-A[0,2])/t + A[0,2]
    recovered= np.dstack((R,G,B)) 
    return recovered

#Section4.4
def getA(img,dark):
    pixelNum = img.shape[0] * img.shape[1]
    selectedPixel = int(max(math.floor(pixelNum/1000),1)) #select at least 1 pixel
    darkvector = np.reshape(dark,-1)
    R = np.reshape(img[:,:,0],-1)
    G = np.reshape(img[:,:,1],-1)
    B = np.reshape(img[:,:,2],-1)
    
    idx = darkvector.argsort()
    idx = idx[pixelNum-selectedPixel::]
    A = np.zeros([1,3])
    for i in range(0,selectedPixel):
        A[0,0] = A[0,0] + R[idx[i]]
        A[0,1] = A[0,1] + G[idx[i]]
        A[0,2] = A[0,2] + B[idx[i]]
    
    A = A/selectedPixel
    
    return A

def start(path,inputpath,outputpath,mid,picture_format=".png"):
    # iterate through all file+    
    relativeinputpath = "." + inputpath
    for file in os.listdir(relativeinputpath):
        # Check whether file is in text format or not
        if file.endswith(picture_format):
            image_read_relative_name = "."+inputpath+mid+file  
            image_write_relative_name = "."+outputpath+mid+file  
            # call read text file function
            imageProcess(image_read_relative_name, image_write_relative_name)
            
            
def imageProcess(image_read_relative_name, image_write_relative_name):
    with open(image_read_relative_name, 'r') as f:
        img = cv2.imread(image_read_relative_name)
        
        #bilateral 是一个filter        
        img = cv2.bilateralFilter(img, 15, 75, 75)
        #这个是dehaze
        img = dehaze(img)
        # 用这行之前需要把img改成float或者uint格式，需要你自己实现一下
        # img = cv2.bilateralFilter(img, 15, 75, 75)
        cv2.imwrite(image_write_relative_name,img)

        
def dehaze(img):
    dark = getDark(img,15)
    A = getA(img,dark)
    transmission = getTransmission(img,A,15)
    transmissionBetter = TransmissionRefine(img,transmission)
    res = Recover(img,transmissionBetter,A,0.1)
    # 加50因为提高亮度
    return res+50


if __name__ == '__main__':
    path = r"markhan/myproject" 
    inputpath = r"/input"
    outputpath = r"/output"
    mid  = "/"
    picture_format = ".png"
    start(path,inputpath,outputpath,mid,picture_format)
