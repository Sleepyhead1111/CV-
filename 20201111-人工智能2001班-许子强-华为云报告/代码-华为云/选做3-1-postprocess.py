from skimage import measure
import cv2
import os
import glob
import numpy as np

def lobe_post_processing(image):
    '''
        肺实质后处理
        :param -> image: 肺实质数组
        :return-> array：numpy
    '''
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image
    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num + 1)]
    area_list = [region[i - 1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x - 1])[::-1]
    # 去除面积较小的连通域
    if len(num_list_sorted) > 1:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[1:]:
            # label[label==i] = 0
            if (area_list[i - 1] * 2) < max(area_list):
                # print(i-1)
                label[region[i - 1].slice][region[i - 1].image] = 0
        # num_list_sorted = num_list_sorted[:1]
    label[label > 0] = 255
    return label

if __name__ == '__main__':
    image_arr = glob.glob('result_single/*')
    image_arr.sort()
    
    allImg = np.zeros((len(image_arr), 512, 512))
    
    save_path = 'result_single_post'
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(len(image_arr)):
        print('i:', i)
        image = cv2.imread(image_arr[i], cv2.IMREAD_GRAYSCALE)
        allImg[i][:][:] = image
        
    post = lobe_post_processing(allImg)  #  ndarray
    
    for idx in range(len(image_arr)):
        print('idx:', idx)
        cv2.imwrite(os.path.join(save_path, image_arr[idx].split('/')[1]), post[idx])
    