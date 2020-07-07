from ipdb import set_trace as pdb
import cv2
import numpy as np

img = cv2.imread('./data/456.jpg')
img_arr = np.array(img)

pdb()
img_g = cv2.GaussianBlur(img, (5,5), 0)
print(img_g.shape)
cv2.imwrite('./results/testing/gaussianfilter.jpg',img_g)

img_p = np.pad(img_arr,((5,5),(5,5),(0,0)),'symmetric')
print(img_p.shape)
cv2.imwrite('./results/testing/padding.jpg',img_p)