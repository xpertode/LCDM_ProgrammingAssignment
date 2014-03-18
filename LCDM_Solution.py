'''
Author:Rishabh Yadav<rishabh.yadav.ece11@iitbhu.ac.in>
Date:08-03-2014 01:39 IST
-->Highly Modular
-->Uses astropy,numpy,pylab and matplotlib libraries.
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
import pylab 




def main():
    ###Please set the location field as per the location of M86.fits in your system.
    location='C:\Users\DELL\Downloads\Compressed\M86.fits\M86.fits'
    #location='M86.fits'

    #Read Image
    img=readImage(location)
    img=removeNoise(img)
    
    #Compute covariance matrix
    cov = determineCovariance(img)

    #Create Subplot
    fig, ax = plt.subplots()
    
    cmap=matplotlib.cm.cool         #Aliasing to keep things simple
    
    #Configure the plot.Set origin to lower and colormap to grayscale.
    ax.imshow(img,cmap=cm.gray,origin='lower')  

    #Find the center of the image
    center_x,center_y=findCenter(img)

    #Plot the image and draw principal axes of the light distribution.
    drawPrincipalAxis(center_x,center_y, cov, ax,img)

    #Mark the center point of the image in the plot.
    plt.plot(center_x,center_y, marker='.',linewidth='5',color='b')

    #Set the range of Y-axis and X-axis.
    pylab.ylim([100,200])
    pylab.xlim([60,180])

    #Set the Figures's subtitle,X-label and Y-label.
    fig.suptitle('GSoC 2014 Solution:R.J.Brunner(Image Pixel Plot)', fontsize=10) 
    plt.xlabel('X', fontsize=18)
    plt.ylabel('Y', fontsize=16)

    #Save the plot in png format
    fig.savefig('GSoC 2014 Solution:R.J.Brunner(Image Pixel Plot).png')

    #Show the plot
    plt.show()




'''
Function to read image and update the default datatype to prevent overflow.
'''
def readImage(location):
    f=fits.open(location)
    img=f[0].data
    img=img.astype(np.int64, copy=False) #To prevent overflow in long_scalars dtype is updated to int64
    return img





'''
Function to find the center of image.
'''
def findCenter(img):
    max_intensity=img.max()
    a=np.where(img==max_intensity)
    x=a[1][0]
    y=a[0][0]
    return [x,y]




'''
Update the image's numpy 2D matrix
such that the noisy points(points whose intensity is below the threshold) are labelled dark.
Threshold is defined as:
threshold=mean + standard deviation
'''
def removeNoise(img):
    threshold=img.mean()+img.std()
    for i in range(len(img)):
        for j in range(len(img)):
            if img[i,j]<threshold:
                img[i,j]=0
    return img



'''
Function to compute the moment of order p+q.
p-->moment w.r.t x
q-->moment w.r.t y

'''

def calculateMoment(data,p,q):
    rows,cols = data.shape
    y, x = np.mgrid[:rows,:cols]
    data = data * x**p * y**q
    return data.sum()





'''
Function to calculate the covariance of the image.
'''
def determineCovariance(data):
    totalIntensity = data.sum()
    m10 = calculateMoment(data, 1, 0)#Moment of order 1;p=1,q=0.
    m01 = calculateMoment(data, 0, 1)#Moment of order 1;p=0,q=1.

    #Calculate x_bar and y_bar which is required to compute the central moment
    x_bar = m10 / totalIntensity
    y_bar = m01 / totalIntensity

    #Central moments upto order 2
    mu11 = (calculateMoment(data, 1, 1) - x_bar * m01) / totalIntensity
    mu20 = (calculateMoment(data, 2, 0) - x_bar * m10) / totalIntensity
    mu02 = (calculateMoment(data, 0, 2) - y_bar * m01) / totalIntensity

    #Construct Covariance Matrix
    cov = np.array([[mu20, mu11], [mu11, mu02]])
    
    return cov





'''
Function to draw the principal axis
'''
def drawPrincipalAxis(x_bar, y_bar, cov, ax,image):

    '''Make lines a length of stddev.'''
    def drawLines(eigen_values, eigen_vectors, mean, i):
        length = np.sqrt(eigen_values[i])
        vector = length * eigen_vectors[:,i] / np.hypot(*eigen_vectors[:,i])/2.5  #Scaling the lines
        x,y  = np.vstack((mean-vector, mean, mean+vector)).T
        return x, y
    
    mean = np.array([x_bar, y_bar])   
    eigen_values, eigen_vectors = np.linalg.eigh(cov) #Inbuilt function to compute the eigen values and eigen vector from given covariance matrix.
    #Draw principal axes
    ax.plot(*drawLines(eigen_values, eigen_vectors, mean, 0), color='yellow')
    ax.plot(*drawLines(eigen_values, eigen_vectors, mean, -1), color='green')
    ax.axis('image')


if __name__ == '__main__':
    main()
