import os, cv2
import numpy as np

def img_enhance(path):
    LCA_data_path = path + 'LCA/data/'
    RCA_data_path = path + 'RCA/data/'

    LCA_enhance_path = path + 'LCA/enhance/'
    RCA_enhance_path = path + 'RCA/enhance/'

    if not os.path.exists(LCA_enhance_path): os.makedirs(LCA_enhance_path)
    if not os.path.exists(RCA_enhance_path): os.makedirs(RCA_enhance_path)

    name = [i for i in os.listdir(LCA_data_path)]
    name.sort(key=lambda x:int(x.split('.')[0]))
    LCA_img_path = [LCA_data_path + i for i in name]
    RCA_img_path = [RCA_data_path + i for i in name]

    LCA_img_gray = [cv2.imread(img, 0) for img in LCA_img_path]
    RCA_img_gray = [cv2.imread(img, 0) for img in RCA_img_path]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_list1 = []
    clahe_list2 = []
    for v in LCA_img_gray:
        clahe_list1.append(clahe.apply(v)) 
    for v in RCA_img_gray:
        clahe_list2.append(clahe.apply(v)) 
    clahe_arr1 = np.array(clahe_list1) # (150, 512, 512)
    clahe_arr2 = np.array(clahe_list2) # (150, 512, 512)
    # n, h, w = clahe_arr1.shape
    # LCA_img_clahe = clahe_arr1.reshape(n, 1, h, w)
    # n, h, w = clahe_arr2.shape
    # RCA_img_clahe = clahe_arr2.reshape(n, 1, h, w)
    for i, v in enumerate(clahe_arr1):
        cv2.imwrite(LCA_enhance_path + name[i], v)
    for i, v in enumerate(clahe_arr2):
        cv2.imwrite(RCA_enhance_path + name[i], v)

if __name__ == '__main__':

    path = '../data/CA_DSA/'
    img_enhance(path)