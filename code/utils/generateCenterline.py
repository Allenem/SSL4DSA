import os
import cv2
import numpy as np
from skimage import morphology


def skeleton_demo(binary):
    binary[binary == 255] = 1
    # print(np.sum(binary))
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    # cv2.imshow("skeleton", skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return skeleton


def main(inputfolder, outputfolder):
    file_list = os.listdir(inputfolder)
    img_list = [inputfolder + i for i in file_list]
    skeleton_list = [outputfolder + i for i in file_list]
    for i, v in enumerate(img_list):
        img = cv2.imread(v)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (512, 512)
        skeleton = skeleton_demo(img_gray)
        cv2.imwrite(skeleton_list[i], skeleton)
    print('Finished folder "{}"!'.format(inputfolder))


if __name__ == '__main__':
    inputfolders = ['../../data/Coronary_DSA/LCA/label/', '../../data/Coronary_DSA/RCA/label/']
    outputfolders = ['../../data/Coronary_DSA/LCA/skeleton/', '../../data/Coronary_DSA/RCA/skeleton/']
    
    for idx in range(len(inputfolders)):
        inputfolder = inputfolders[idx]
        outputfolder = outputfolders[idx]

        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        
        main(inputfolder, outputfolder)