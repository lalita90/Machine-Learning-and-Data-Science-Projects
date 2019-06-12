# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:20:39 2019

@author: LALITA
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
F=[1, 2, 1, 3, 2, 3, 1, 2, 3, 8, 7, 8, 9, 9, 7, 8]
#F=np.pad(F, (1, 1), 'constant')

con_ai=np.convolve(F, [1,1,1])  #slightly blurred 
con_aii=np.convolve(F, [1,0,-1])    #edge detection
#https://en.wikipedia.org/wiki/Kernel_(image_processing)

#%%Q1b

def filterrotate(kernel):
    kernel_copy= kernel.copy()
    for i in range(kernel.shape[0]):
      for j in range(kernel.shape[1]):
        kernel_copy[i][j]= kernel[kernel.shape[0]-i-1][kernel.shape[1]-j-1]
    return kernel_copy
def convolve(image, kernel):
	
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	
	pad = int(kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_CONSTANT,value=[255,255,255])
	output = np.zeros((iH, iW), dtype="float32")
  # loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			
			
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
 
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()
 
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k
      # rescale the output image to be in the range [0, 255]
	#output = rescale_intensity(output, in_range=(0, 255))
	#output = (output * 255).astype("uint8")
 
	# return the output image
	return output
#%%Q1bi
img=np.array([[255, 168, 200, 180, 195], 
                 [204, 189, 178, 167, 128],
                 [184, 190, 216, 250, 145],
                 [178, 201, 105, 119, 197],
                 [87, 158, 138, 201, 169]
                ])
    
filter=np.array([[1, 2, 1], 
                  [2, 4, 2],
                  [1, 2, 1]])
filter=(1/16)*filter    #gaussian blur
#rotate the kernel by 180 degrees horizontally and vertically
filter_rot=filterrotate(filter)
#perform convolution
img_con= convolve(img,filter_rot)
#%%Q1bii
image= np.array([[164, 188, 164, 161, 195], 
                 [178, 201, 197, 150, 137],
                 [174, 168, 181, 190, 184],
                 [131, 179, 176, 185, 198],
                 [92, 185, 179, 133, 167]
                ])

kernel_1= np.array([[1, 1, 1], 
                  [1, 1, 1],
                  [1, 1, 1]])


kernel_2= np.array([[-1, 2, -1], 
                  [0, 0, 0],
                  [1, 2, 1]])

kernel_3= np.array([[-1, -1, -1], 
                  [-1, 9, -1],
                  [-1, -1, -1]])

#rotate the kernel by 180 degrees horizontally and vertically
kernel_1_rot=filterrotate(kernel_1)
kernel_2_rot=filterrotate(kernel_2)  
kernel_3_rot=filterrotate(kernel_3)

#perform convolution
image_con_1= convolve(image,kernel_1_rot)   #blurs
image_con_2= convolve(image,kernel_2_rot)   #smoothning
image_con_3= convolve(image,kernel_3_rot)   #sharpening
print(image_con_1)
print(image_con_2)
print(image_con_3)


fig5, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(image) 
axarr[0,1].imshow(image_con_1) 
axarr[1,0].imshow(image)
axarr[1,1].imshow(image_con_2) 
axarr[2,0].imshow(image)
axarr[2,1].imshow(image_con_3)
plt.show()




