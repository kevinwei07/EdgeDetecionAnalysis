from ipdb import set_trace as pdb
import cv2
import os
import numpy as np

def crop(h,w,img_file):
    
    img_name, img_ext = img_file.rsplit('.', 1)
    img = cv2.imread('./data/'+img_file)
    print(img_name,' shape = ',img.shape)

    height, width = img.shape[:2]
    heightnum = h
    widthnum = w

    for h in range (0,heightnum): 
        for w in range (0,widthnum):
            num = h*widthnum+w+1
            #print(num)
            patch = img[(height//heightnum)*h:(height//heightnum)*(h+1),(width//widthnum)*w:(width//widthnum)*(w+1)]
            cv2.imwrite('./cropped/'+img_name+'_'+str(num)+'.jpg',patch)

def stitch(h,w,enlarge,img_path):
    
    #ori_img = cv2.imread('./data/chord.jpg')
    #height,width = ori_img.shape[:2]
    heightnum = h
    widthnum = w
    enlarge = enlarge
    desc = f'_{heightnum}*{widthnum}_{enlarge}x'

    edge_list = os.listdir(img_path)

    for picnum in range(0,len(edge_list)//(h*w)):
        row_image = []
        final = []
        picname = edge_list[picnum*(h*w)].rsplit('.')[0].strip('rcf_').strip('_1') # IMG_ (45)
        for i in range (0,heightnum): 
            rowlist=[]
            for j in range (0,widthnum):
                num = i*widthnum+j+1
                img = cv2.imread(img_path+edge_list[picnum*(h*w)+num-1])
                #blank[(height//heightnum)*h:(height//heightnum)*(h+1),(width//widthnum)*w:(width//widthnum)*(w+1)] = img[:,:]
                row = np.array(img)
                rowlist.append(row)
            row_image.append(np.concatenate(rowlist,axis=1))
        final=np.concatenate(row_image,axis=0)
        print('final shape = ',final.shape)
        cv2.imwrite('./results/fulledge/'+picname+desc+'.jpg',final)

# if __name__ == '__main__':

#     heightnum = 3
#     widthnum = 5

#     crop(heightnum,widthnum)
#     #stitch(heightnum,widthnum,'./stitched/')
