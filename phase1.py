

def shear(image,degree,direction):
  res = np.zeros((2*image.shape[0],2*image.shape[1],3))

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      v = (i - image.shape[1]/2)
      w = (j - image.shape[0]/2)
      if(direction=="y"):
        x = v 
        y = w + v*degree
      elif(direction=="x"):
        x = v + w*degree 
        y = w  
      res[int(y + res.shape[0]/2),int(x + res.shape[1]/2)]=image[j,i]     
  return res


def scale(image,qX,qY):

  res = np.zeros((image.shape[0],image.shape[1],3))
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      v = i - image.shape[1]/2
      w = j - image.shape[0]/2
      x = v*qX
      y = w*qY
      res[int(y + res.shape[0]/2), int( x + res.shape[1]/2)] = image[j,i]
  return res

def merge(image,x,y,cube):
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      newX = int(x-(image.shape[0]/2)+i)
      newY = int(y-(image.shape[1]/2)+j)
      if(newX>=0 and newX<cube.shape[0] and newY>=0 and newY<cube.shape[1]):
        if(np.all(image[i,j])):
         cube[newX,newY]= image[i,j]
  # cv2_imshow(cube)
  return cube


import cv2
from PIL import Image
from google.colab.patches import cv2_imshow

import numpy as np
import pandas as pd
import cv2

# # # 1.
 
cube=cv2.imread("Cube.png")
# cube = cv2.cvtColor(cube,cv2.COLOR_BGR2GRAY)
# 1.1
lena = cv2.imread("lena.bmp")
# lena = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)

girl = cv2.imread("girl.bmp")
# girl= cv2.cvtColor(girl,cv2.COLOR_BGR2GRAY)
barbara = cv2.imread("barbara.bmp")


sheared_barbara = shear(barbara,0.43,"x")
scaled_barbara = scale(sheared_barbara,.99,0.45)
cube = merge(scaled_barbara,232,472,cube)


#  scaling lena:
scaling = [[0.9765625,0,0],[0,0.9765625,0],[0,0,1]]
lena_scaled = np.zeros((512,512,3))
barbara_scaled = np.zeros((1000,1000,3))

for i in range(501):
  for j in range(501):
    for k in range(3):
      lena_scaled[i][j][k]= lena[int(i*0.9765625)][int(j*0.9765625)][k]

for i in range(500):
  for j in range(500):
    for k in range(3):
      cube[346+i,333+j,k]=lena_scaled[i][j][k]


image = shear(girl,0.45,"y")
image2 = scale(image,0.43,0.99)
result = merge(image2,481,222,cube)
cv2_imshow(result)

im = Image.open("Map1.gif")
im.seek(0)
im.save("map1.png")

im = Image.open("Map2.gif")
im.seek(0)
im.save("map2.png")


map1 = cv2.imread("map1.png")
map1 = cv2.cvtColor(map1,cv2.COLOR_BGR2GRAY)
map2 = cv2.imread("map2.png")
map2 = cv2.cvtColor(map2,cv2.COLOR_BGR2GRAY)
print(type(map1))
print(map1.shape,map2.shape)

def line_interpolation_map(v,w):
  y = 0.97153*v-0.29462*w+0.00028696*v*w+97.625
  if y > 405:
    y = 405
  x = 0.1915*v+1.0358*w-0.0001089*w*v+7.9987 
  if x > 414:
    x = 414
  return x,y 

map12 = np.zeros((415,406))
print(map12.shape)

for i in range(394):
  for j in range(369):
    x,y = line_interpolation_map(j,i)
    map12[int(x)][int(y)] = map2[i][j] 

cv2_imshow(map12)

def sobel_filter(image):
  n = 5
  sobel_Xfilter = np.array([[1,2,3,2,1],
                             [2,3,5,3,2],
                             [0,0,0,0,0],
                             [-2,-3,-5,-3,-2],
                             [-1,-2,-3,-2,-1]])
  
  sobel_Yfilter = np.array([[-1,-2,0,2,1],
                             [-2,-3,0,3,2],
                             [-3,-5,0,5,3],
                             [-2,-3,0,3,2],
                             [-1,-2,0,2,1]])

  Gx = np.zeros((image.shape[0],image.shape[1]))
  Gy = np.zeros((image.shape[0],image.shape[1]))

  for i in range(image.shape[0]):
    if image.shape[0] - i < n:
      break
    for j in range(image.shape[1]):
      if image.shape[1] - j < n:
        break
      Gx[i][j] = (sobel_Xfilter * image[i:i+5,j:j+5]).sum()

  for i in range(image.shape[0]):
    if image.shape[0] - i < n:
      break
    for j in range(image.shape[1]):
      if image.shape[1] - j < n:
        break
      Gy[i][j] = (sobel_Yfilter * image[i:i+5,j:j+5]).sum()    
     
  G =  Gy + Gx
  
  return G

# 2
mosque= cv2.imread("mosque.bmp")
mosque = cv2.cvtColor(mosque,cv2.COLOR_BGR2GRAY)

def robert_filter(image):

  robert_Xfilter = np.array([[1,0],[0,-1]])
  robert_Xfilter = np.array([[0,1],[-1,0]])

  Gx = np.zeros((image.shape[0],image.shape[1]))
  Gy = np.zeros((image.shape[0],image.shape[1]))

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      Gx[i][j] = (robert_Xfilter * image[i:i+2,j:j+2]).sum()
      Gy[i][j] = (robert_Xfilter * image[i:i+2,j:j+2]).sum()

  G = np.sqrt(np.power(Gx,2)+np.power(Gy,2))
  
  return G


def mean_filter(image,n):

  mean_filter = np.ones((n,n))/(n*n)
  G = np.zeros((image.shape[0],image.shape[1]))

  for i in range(image.shape[0]):
    if image.shape[0] - i < n:
      break
    for j in range(image.shape[1]):
      if image.shape[1] - j < n:
        break
      G[i][j] = (mean_filter * image[i:i+n,j:j+n]).sum()

  return G 

mosque_robert = robert_filter(mosque)
cv2_imshow(mosque_robert)

mosque_mean_3 = mean_filter(mosque,3)
mosque_mean_3_robert = robert_filter(mosque_mean_3)
cv2_imshow(mosque_mean_3_robert)

mosque_mean_5 = mean_filter(mosque,5)
mosque_mean_5_robert = robert_filter(mosque_mean_5)
cv2_imshow(mosque_mean_5_robert)

mosque_mean_7 = mean_filter(mosque,7)
cv2_imshow(mosque_mean_7)
mosque_mean_7_robert = robert_filter(mosque_mean_7)
cv2_imshow(mosque_mean_7_robert)


def psedu_mean_nx_filter(image,n):

  if n == 5:
    mean_filter =  (np.array([[1,2,3,2,1],
                            [2,4,6,4,2],
                            [3,6,9,6,3],
                            [2,4,6,4,2],
                            [1,2,3,2,1]]))/81
  elif n == 7:  
        mean_filter =  (np.array([[1,3,6,7,6,3,1],
                            [3,9,18,21,18,9,3],
                            [6,18,36,42,36,18,6],
                            [7,21,42,49,42,21,7],
                            [6,18,36,42,36,18,6],
                            [3,9,18,21,18,9,3],
                            [1,3,6,7,6,3,1]])/729) 

  G = np.zeros((488,695))

  for i in range(image.shape[0]):
    if image.shape[0] - i < n:
      break
    for j in range(image.shape[1]):
      if image.shape[1] - j < n:
        break
      G[i][j] = (mean_filter * image[i:i+n,j:j+n]).sum()

  return G 




mosque_mean_3 = mean_filter(mosque,3)
mosque_mean_3_2x = mean_filter(mosque_mean_3,3)
cv2_imshow(mosque_mean_3_2x)

mosque_mean_3 = psedu_mean_nx_filter(mosque,5)
cv2_imshow(mosque_mean_3)


mosque_mean_3 = mean_filter(mosque,3)
mosque_mean_3_2x = mean_filter(mosque_mean_3,3)
mosque_mean_3_3x = mean_filter(mosque_mean_3_2x,3)
cv2_imshow(mosque_mean_3_3x)

mosque_mean_7 = psedu_mean_nx_filter(mosque,7)
cv2_imshow(mosque_mean_7)

import tensorflow as tf
tf.test.gpu_device_name()

mean_filter =  (np.array([[1,3,6,7,6,3,1],
                            [3,9,18,21,18,9,3],
                            [6,18,36,42,36,18,6],
                            [7,21,42,49,42,21,7],
                            [6,18,36,42,36,18,6],
                            [3,9,18,21,18,9,3],
                            [1,3,6,7,6,3,1]]))
        mean_filter.sum() 

        x = np.array([[1,2,3,2,1],
                            [2,4,6,4,2],
                            [3,6,9,6,3],
                            [2,4,6,4,2],
                            [1,2,3,2,1]])
        # x.sum()

# 3




he1 = cv2.imread("he1.jpg")
he2 = cv2.imread("he2.jpg")
he3 = cv2.imread("he3.jpg")
he4 = cv2.imread("he4.jpg")
def hiseq(he1):
  
  # preprocessing

  he1_gray = cv2.cvtColor(he1, cv2.COLOR_BGR2GRAY)
  unique, counts = np.unique(he1_gray, return_counts=True)
  
# count of each pixel 
  dict_he1 =dict(zip(unique, counts))

# sum of all pixels
  sum_he1 = sum(dict_he1.values())

# preprocessing
  if dict_he1.get(0)==None:
    dict_he1[0]=0

# count of each pixel / sum of all pixels
  for i in dict_he1 :
      dict_he1[i] = dict_he1[i] / sum_he1
   
  dict_accum = {}
  dict_accum[0] = dict_he1.get(0)
 
# cumulative sum 
  for i in range(1,256):
    if dict_he1.get(i):
      dict_accum[i] =  dict_he1.get(i) + dict_accum.get(i-1)
    else:
      dict_accum[i]= dict_accum.get(i-1)

# cumulative sum * max of pixels range
  for i in range(256):
      dict_accum[i] = round(dict_accum.get(i)*255)

#  equalize image
  for i in range(he1_gray.shape[0]):
    for j in range(he1_gray.shape[1]):
       he1_gray[i][j] = dict_accum[he1_gray[i][j]]
  # cv2_imshow(he1_gray)
  return he1_gray

hiseq(he1)  
hiseq(he2) 
hiseq(he3) 
hiseq(he4)

# preprocessing
f_rgb = cv2.imread("he4.jpg")
f = cv2.cvtColor(f_rgb,cv2.COLOR_BGR2GRAY)

f_he = hiseq(f_rgb)

# apply equation
for i in np.arange(0.1,0.6,0.1):
  g = i*f+(1-i)*f_he
  cv2_imshow(g)

def LHE(he1,xWin,yWin):
  
  # preprocessing

  he1_gray = cv2.cvtColor(he1, cv2.COLOR_BGR2GRAY)
  he1_local= np.empty((xWin, yWin), float)

  # for m in range(0,he1_gray.shape[0]-xWin,int((xWin-1)/2)):
  #   for n in range(0,he1_gray.shape[1]-yWin,int((yWin-1)/2)):
  
  for m in range(he1_gray.shape[0]-xWin):
    if m+xWin > he1_gray.shape[0] :
          break
    for n in range(he1_gray.shape[1]-yWin):

       

        he1_local = he1_gray[m:m+xWin,n:n+yWin]

        unique, counts = np.unique(he1_local, return_counts=True)
        
      # count of each pixel 
        dict_he1 =dict(zip(unique, counts))

      # sum of all pixels
        sum_he1 = sum(dict_he1.values())

        # preprocessing
        if dict_he1.get(0)==None:
            dict_he1[0]=0

        # count of each pixel / sum of all pixels
        for i in dict_he1 :
              dict_he1[i] = dict_he1[i] / sum_he1
          
        dict_accum = {}
        dict_accum[0] = dict_he1.get(0)
        
        # cumulative sum 
        for i in range(1,256):
            if dict_he1.get(i):
              dict_accum[i] =  dict_he1.get(i) + dict_accum.get(i-1)
            else:
              dict_accum[i]= dict_accum.get(i-1)

        # cumulative sum * max of pixels range
        for i in range(256):
              dict_accum[i] = round(dict_accum.get(i)*255)

        #  equalize image
        for i in range(he1_local.shape[0]):
            for j in range(he1_local.shape[1]):
              he1_gray[i+m][j+n] = dict_accum[he1_local[i][j]]
  cv2_imshow(he1_gray)
  # return he1_gray
  
LHE(he1,201,201)

gussian_3x3 = np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]
                        ])/16


def gussian_filter(image,kernel):
    
    n = kernel.shape[0]
    G = np.zeros((image.shape[0],image.shape[1]))

    for i in range(image.shape[0]):
      if image.shape[0] - i < n:
        break
      for j in range(image.shape[1]):
        if image.shape[1] - j < n:
          break
        G[i][j] = (kernel * image[i:i+n,j:j+n]).sum()

    return G


# 4
child = cv2.imread("child.jpg")
child = cv2.cvtColor(child,cv2.COLOR_BGR2GRAY)

def mean_filter(image,n):

  mean_filter = np.ones((n,n))/(n*n)
  G = np.zeros((image.shape[0],image.shape[1]))

  for i in range(image.shape[0]):
    if image.shape[0] - i < n:
      break
    for j in range(image.shape[1]):
      if image.shape[1] - j < n:
        break
      G[i][j] = (mean_filter * image[i:i+n,j:j+n]).sum()

  
  return G 

child_unsharp = mean_filter(child,3)
unsharp_mask = child - child_unsharp
result1 = child + unsharp_mask
cv2_imshow(result1)


child_unsharp_gussian = gussian_filter(child,gussian_3x3)
unsharp_mask_gussian = child - child_unsharp_gussian
result2 = child + unsharp_mask_gussian
cv2_imshow(result2)


child_unsharp = mean_filter(child,5)
unsharp_mask = child - child_unsharp
result3 = child + unsharp_mask
cv2_imshow(result3)


child_unsharp = mean_filter(child,7)
unsharp_mask = child - child_unsharp
result5 = child + unsharp_mask
cv2_imshow(result5)


child_unsharp = mean_filter(child,9)
unsharp_mask = child - child_unsharp
result9 = child + unsharp_mask
cv2_imshow(result9)

def median_filter(image):

  
  G = np.zeros((image.shape[0],image.shape[1]))
  temp = np.zeros((3,3))
  for i in range(image.shape[0]):
    if image.shape[0] - i < 3:
      break
    for j in range(image.shape[1]):
      if image.shape[1] - j < 3:
        break
      temp = image[i:i+3,j:j+3]
      G[i][j] = np.median(temp)

  
  return G 

# static gussian filter

kernel = np.array([[-1,-2, -1],[-2, 12, -2],[-1,-2,-1]])/16
derivative = gussian_filter(child,kernel)

res = derivative+child
cv2_imshow(res)

# dynamic gaussian filter

kernel1 = np.array([[0,0, 0],[-2, 2, 0],[0,0,0]])
derivative1 = gussian_filter(child,kernel1)/16
print(derivative1.sum())

kernel2 = np.array([[-1,0, 0],[0, 1, 0],[0,0,0]])
derivative2 = gussian_filter(child,kernel2)
print(derivative2.sum())


# lower than threshold
# kernel3 = np.array([[0,-2, 0],[0, 2, 0],[0,0,0]])
# derivative3 = gussian_filter(child,kernel3)
# print(derivative3.sum())

# lower than threshold
# kernel4 = np.array([[0,0, -1],[0, 1, 0],[0,0,0]])
# derivative4 = gussian_filter(child,kernel4)
# print(derivative4.sum())

# lower than threshold
# kernel5 = np.array([[0,0, 0],[0, 2, -2],[0,0,0]])
# derivative5 = gussian_filter(child,kernel5)
# print(derivative5.sum())


# lower than threshold
# kernel6 = np.array([[0,0, 0],[0, 1, 0],[0,0,-1]])
# derivative6 = gussian_filter(child,kernel6)
# print(derivative6.sum())

kernel7 = np.array([[0,0, 0],[0, 2, 0],[0,-2,0]])
derivative7 = gussian_filter(child,kernel7)
print(derivative7.sum())

kernel8 = np.array([[0,0, 0],[0, 1, 0],[-1,0,0]])
derivative8 = gussian_filter(child,kernel8)
print(derivative8.sum())

cv2_imshow(derivative1+derivative2+derivative7+derivative8+child)

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('child.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()