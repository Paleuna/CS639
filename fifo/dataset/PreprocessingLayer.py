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

                      
def PreprocessingLayer(img):  #img should be a np array    
    #bilateral filter        
    img = cv2.bilateralFilter(img, 15, 75, 75)
    #dehaze
    dark = getDark(img,15)
    A = getA(img,dark)
    transmission = getTransmission(img,A,15)
    transmissionBetter = TransmissionRefine(img,transmission)
    res = Recover(img,transmissionBetter,A,0.1) + 50
    # add 50 to increase the lightness
    return res
