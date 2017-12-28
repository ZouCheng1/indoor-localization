from __future__ import print_function
#import sys, os
import math
import cv2
import numpy
import scipy.signal
#import pylab
def get_contour(file_name):
    gray_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    if gray_image.shape[1] > gray_image.shape[0]:
        gray_image = numpy.rot90(gray_image, 3)
    m2 = cv2.blur(gray_image, (50,50))
    thresholded_img = cv2.adaptiveThreshold(m2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 2)
    _,contours, heirarchy = cv2.findContours(thresholded_img, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    length,width = gray_image.shape[0],gray_image.shape[1]
    #contour_image = gray_image.copy()
	       #cv2.drawContours(contour_image, contours, -1,255 ,3 ) 
    #contours_kept_image = cv2.imread(file_name, cv2.IMREAD_COLOR)
    #cv2.draw(contour_image,contours,-1,255,3)
    #kept_center = (int(contours_kept_image.shape[1] / 2), int(contours_kept_image.shape[0] / 2))
    return (gray_image,contours,length,width)
def posi_get(contours,length,width):
    centers = []
    radius = []
    for contour in contours:
        center ,radiu = cv2.minEnclosingCircle(contour)
        center = list(map(int,center))
        radiu = int(radiu)
        center=(center[1],center[0])
        reject = False
        if radiu < 50:
            continue
        for pt in contour:
            assert len(pt)==1
            pt =pt[0]
            #pt.reshape(2)
            if pt[0]<10 or pt[1]<10 or pt[1]>(length-10) or pt[0] > (width -10):
                reject = True 
                break
        if reject:
            continue
        contour_area = cv2.contourArea(contour)
        circle_area = math.pi * radiu**2
        if (contour_area / circle_area) < .5:
            continue
        radius.append(radiu)
        centers.append(center)
        number_of_transmitters = len(centers)
    return centers ,radius, number_of_transmitters
def get_freq( centers ,radius, number_of_transmitters,gray_image):
    Fs=55556.0
    NFFT = 2**14
    gain = 5
    estimated_frequencies = []
    radii = radius
    #light_circles = gray_image.copy()
    for i in range(number_of_transmitters):
        try:
            row_start = max(0, centers[i][0] - radii[i])
            row_end = min(gray_image.shape[0]-1, centers[i][0] + radii[i])
            column_start = max(0, centers[i][1] - radii[i])
            column_end = min(gray_image.shape[1]-1, centers[i][1] + radii[i])
            image_slice = gray_image[row_start:row_end, column_start:column_end]
            image_row = numpy.sum(image_slice, axis=0)
            y = image_row * numpy.hamming(image_row.shape[0])
            L = len(y)  
            f = Fs/2 * numpy.linspace(0,1,NFFT/2.0+1)
            Y = numpy.fft.fft(y*gain,NFFT)/float(L)  
            Y_plot = 2*abs(Y[0:int(NFFT/2.0+1)])
            peaks = scipy.signal.argrelmax(Y_plot)[0]
            idx = numpy.argmax(Y_plot[peaks])  
            peak_freq = f[peaks[idx]]
            estimated_frequencies.append(peak_freq) 
        except:
            estimated_frequencies.append(10)
    estimated_frequencies = numpy.array(estimated_frequencies)
    return estimated_frequencies
def img_proc(file_name):
    gray_image,contours,length,width = get_contour(file_name);
    centers ,radius, number_of_transmitters = posi_get(contours,length,width)
    estimated_frequencies = get_freq( centers ,radius, number_of_transmitters,gray_image)
    return centers,gray_image,radius,number_of_transmitters,estimated_frequencies
    
    


    
    

        
            
            
               
                