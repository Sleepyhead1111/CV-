import numpy as np
import cv2


# 线性级增加亮度
def liner_trans(image, gamma):   # gamma大于1时图片变亮，小于1图片变暗
    image = np.float32(image) * gamma // 1
    image[image > 255] = 255
    image = np.uint8(image)
    return image


# 添加speckle噪声
def add_speckle(image):
    # 随机生成一个服从分布的噪声
    gauss = np.random.randn(image.shape[0], image.shape[1], image.shape[2])
    # 添加speckle噪声
    noisy_img = image + image * gauss
    # 归一化图像的像素值
    noisy_image = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_image


if __name__ == '__main__':
    image = cv2.imread('d2.png')  # 宽1920(shape[1])，高1080(shape[0])，3通道(shape[2])
    cv2.imshow(winname='Destiny2', mat=image)
    cv2.waitKey(0)
    # 旋转
    rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # ROTATE_90_CLOCKWISE(顺时针旋转90度), ROTATE_180(旋转180度), ROTATE_90_COUNTERCLOCKWISE(逆时针旋转90度)
    cv2.imshow(winname='rotate', mat=rotate)
    cv2.waitKey(0)
    # 垂直翻转
    ub = cv2.flip(image, 0)  # 图像翻转，flipcode控制翻转效果。flipcode = 0：沿x轴翻转；flipcode > 0：沿y轴翻转；flipcode < 0：x,y轴同时翻转
    cv2.imshow(winname='ub', mat=ub)
    cv2.waitKey(0)
    # 水平翻转
    lr = cv2.flip(image, 1)
    cv2.imshow(winname='lr', mat=lr)
    cv2.waitKey(0)
    # 增亮
    light = liner_trans(image, gamma=1.5)  # gamma大于1时图片变亮
    cv2.imshow(winname='light', mat=light)
    cv2.waitKey(0)
    # 增加噪音
    noise = add_speckle(image)
    cv2.imshow(winname='noise', mat=noise)
    cv2.waitKey(0)
    # 保存图像
    cv2.imwrite('d2_rotate.png', rotate)
    cv2.imwrite('d2_ub.png', ub)
    cv2.imwrite('d2_lr.png', lr)
    cv2.imwrite('d2_noise.png', noise)
    cv2.imwrite('d2_light.png', light)

    cv2.destroyAllWindows()

