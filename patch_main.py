import argparse
import os
#from dataset.joint_dataset import get_loader
#from joint_solver import Solver
from ipdb import set_trace as pdb
import cv2
import numpy as np


def get_patches(img):
    img_h, img_w = img.shape[:2]        # (604, 401)

    # h_step = img_h // 5     # 1
    # w_step = img_w // 5     # 0
    h_step = w_step = 500

    h_list = [h_step*i if h_step*i<img_h else img_h for i in range(1, img_h//h_step+2)]
    w_list = [w_step*i if w_step*i<img_w else img_w for i in range(1, img_w//w_step+2)]
    if len(h_list) >= 2:
        h_list = h_list[:-2] + [h_list[-1]]
    if len(w_list) >= 2:
        w_list = w_list[:-2] + [w_list[-1]]

    print(h_list)
    print(w_list)

    patches = []
    now_y = 0
    for y in h_list:
        row = []
        now_x = 0
        for x in w_list:
            row.append(img[now_y:y, now_x:x])
            now_x = x
        patches.append(row)
        now_y = y
    return patches


def remove_border(edge_img):
    edge_img[0] = 255       # first row
    edge_img[-1] = 255      # last row
    edge_img[:, 0] = 255    # first column
    edge_img[:, -1] = 255   # last column
    return edge_img



if __name__ == '__main__':
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'

    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    # Testing settings
    parser.add_argument('--model', type=str, default='models/final_edge.pth') # Snapshot

    parser.add_argument('--test_fold', type=str, default='results/edge')
    parser.add_argument('--test_mode', type=int, default=0) # 0->edge
    # parser.add_argument('--test_fold', type=str, default='results/sal') # Test results saving folder
    # parser.add_argument('--test_mode', type=int, default=1) # 0->edge, 1->saliency
    
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):os.mkdir(config.save_folder)
    if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)

    config.test_root = './data/test/Imgs/'
    test_loader = get_loader(config)
    test = Solver(test_loader, config)

    img_dir = './data/test/Imgs/'
    img_name = 'IMG_ (62)'
    img = cv2.imread(img_dir + img_name + '.jpg')
    patches = get_patches(img)
    p = test.test_patch(patches[0][0])

    img_edges = []
    for i,row in enumerate(patches):
        row_edges = []
        for j,image in enumerate(row):
            edge_img = test.test_patch(image)
            # edge_img = remove_border(edge_img)
            row_edges.append(edge_img)
        img_edges.append(np.concatenate(row_edges, axis=1))
    img_edges = np.concatenate(img_edges, axis=0)
    cv2.imwrite(f'results/edge/{img_name}_edge_patch.png', img_edges)
    