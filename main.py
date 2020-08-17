import os
import shutil
import cv2
import numpy as np
import torch

from tqdm import tqdm
from ipdb import set_trace as pdb
from CropandStitch import crop,stitch
from RCF.run_rcf import make_single_rcf


clean = True 


def enhance_filter(input):
    # # LAB enhance
    # lab = cv2.cvtColor(input, cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(3, 3))
    # lab_planes[0] = clahe.apply(lab_planes[0])
    # lab = cv2.merge(lab_planes)
    # input = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # BGR enhance
    b,g,r = cv2.split(input)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    input = cv2.merge([b,g,r])

    # # Filter
    # # d=5 for real-time, 9 for offline application
    # # two sigma values can be the same, < 10 won't have effect, > 150 make cartoonish image.
    # input = cv2.bilateralFilter(input, d=9, sigmaColor=150, sigmaSpace=150)

    return input


if __name__ == '__main__':

    img_dir = './data/general/'
    #img_list = os.listdir(img_dir)
    with open('./data/test.lst', 'r') as f:
        img_list = f.read().splitlines()

    edge_path = './edge/rcf/' # for saving edge result

    # ========================================== set scale attribute ==========================================

    heightnum = 4
    widthnum = 3
    enlarge = 2
    padding_size = 5

    # ========================================== crop into pieces ==========================================
    for file in img_list:
        crop(heightnum, widthnum, img_dir, file)

    cropped_dir = './cropped/'
    cropped_list = os.listdir(cropped_dir)

    # ========================================== make RCF edge ==========================================
    for img in tqdm(cropped_list, total=len(cropped_list)):
        # print('------')
        input_image = cv2.imread(cropped_dir+img)
        input_image = enhance_filter(input_image)
        # padding for elimating border
        pad_image = np.pad(input_image,((padding_size,padding_size),(padding_size,padding_size),(0,0)),'symmetric') 
        # print('input: ', input_image.shape)
        large_image = cv2.resize(pad_image, (0,0), fx=enlarge, fy=enlarge)
        # print('large: ', large_image.shape)
        make_single_rcf(large_image, edge_path+'rcf_'+img, enlarge,padding_size)

    if clean :
        shutil.rmtree('./cropped/')  
        os.mkdir('./cropped/')  

    print(' ----- RCF End ----- ')
    # ========================================== padding to eliminate border ==========================================

    # pad_dir = './edge/rcf/'
    # pad_list = os.listdir(pad_dir)
    # for file in pad_list:
    #     img = cv2.imread(pad_dir+file)
    #     print(file,img.shape)
    #     img = np.pad(img,((5,5),(5,5),(0,0)),'symmetric') 
    #     print(file,img.shape)
    #     img = img[5:img.shape[0]-5, 5:img.shape[1]-5]
    #     print(file,img.shape)


    # ========================================== stitching back ==========================================
    stitch(heightnum, widthnum, enlarge, edge_path)
    print(' ----- Stitch End ----- ')
