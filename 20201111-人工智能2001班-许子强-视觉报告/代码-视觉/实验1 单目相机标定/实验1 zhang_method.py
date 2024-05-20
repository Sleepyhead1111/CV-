import os
import numpy as np
import cv2


# 求输入数据的归一化矩阵
def normalizing_input_data(coor_data):
    x_avg = np.mean(coor_data[:, 0])
    y_avg = np.mean(coor_data[:, 1])
    sx = np.sqrt(2) / np.std(coor_data[:, 0])
    sy = np.sqrt(2) / np.std(coor_data[:, 1])
    norm_matrix = np.matrix([[sx, 0, -sx * x_avg],
                            [0, sy, -sy * y_avg],
                            [0, 0, 1]])
    return norm_matrix


# 求单应矩阵H
def get_H(pic_coor, real_coor):
    # 获得归一化矩阵
    pic_norm_mat = normalizing_input_data(pic_coor)
    real_norm_mat = normalizing_input_data(real_coor)

    M = []
    for i in range(len(pic_coor)):
        # 转换为齐次坐标
        single_pic_coor = np.array([pic_coor[i][0], pic_coor[i][1], 1])
        single_real_coor = np.array([real_coor[i][0], real_coor[i][1], 1])
        # 坐标归一化
        pic_norm = np.dot(pic_norm_mat, single_pic_coor)
        real_norm = np.dot(real_norm_mat, single_real_coor)
        # 构造M矩阵
        M.append(np.array([-real_norm.item(0), -real_norm.item(1), -1,
                           0, 0, 0,
                           pic_norm.item(0) * real_norm.item(0), pic_norm.item(0) * real_norm.item(1),
                           pic_norm.item(0)
                           ]))
        M.append(np.array([0, 0, 0,
                           -real_norm.item(0), -real_norm.item(1), -1,
                           pic_norm.item(1) * real_norm.item(0), pic_norm.item(1) * real_norm.item(1),
                           pic_norm.item(1)]))
    # 利用SVD求解M * h = 0中h的解
    U, S, VT = np.linalg.svd((np.array(M, dtype='float')).reshape((-1, 9)))
    # 最小的奇异值对应的奇异向量，S求出来按大小排列的，最后的最小
    H = VT[-1].reshape((3, 3))
    H = np.dot((np.dot(np.linalg.inv(pic_norm_mat), H)), real_norm_mat)
    H /= H[-1, -1]
    return H


# 返回H，pic_coor为角点在图像上的坐标，real_coor为角点在现实世界中的坐标
def get_homography(pic_coor, real_coor):
    refined_homographies = []
    for i in range(len(pic_coor)):
        H = get_H(pic_coor[i], real_coor[i]).reshape(-1, 1)
        refined_homographies.append(H)
    return np.array(refined_homographies)


# 返回pq位置对应的v向量
def create_v(p, q, H):
    H = H.reshape(3, 3)
    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])


# 得到相机内参矩阵A
def get_intrinsic_param(H):
    # 构建V矩阵
    V = np.array([])
    for i in range(len(H)):
        V = np.append(V, np.array([create_v(0, 1, H[i]), create_v(0, 0, H[i]) - create_v(1, 1, H[i])]))
    # 求解V*b = 0中的b
    U, S, VT = np.linalg.svd((np.array(V, dtype='float')).reshape((-1, 6)))
    # 最小的奇异值对应的奇异向量，S求出来按大小排列的，最后的最小
    b = VT[-1]
    # 求相机内参
    w = b[0] * b[2] * b[5] - b[1] * b[1] * b[5] - b[0] * b[4] * b[4] + 2 * b[1] * b[3] * b[4] - b[2] * b[3] * b[3]
    d = b[0] * b[2] - b[1] * b[1]

    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d ** 2 * b[0])
    gamma = np.sqrt(w / (d ** 2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d
    return np.array([
        [alpha, gamma, uc],
        [0, beta, vc],
        [0, 0, 1]
    ])


# 返回每一幅图像的外参矩阵[R|t]
def get_extrinsic_param(H, intrinsics_param):
    extrinsic_param = []
    inv_intrinsics_param = np.linalg.inv(intrinsics_param)
    for i in range(len(H)):
        h0 = (H[i].reshape(3, 3))[:, 0]
        h1 = (H[i].reshape(3, 3))[:, 1]
        h2 = (H[i].reshape(3, 3))[:, 2]
        scale_factor = 1 / np.linalg.norm(np.dot(inv_intrinsics_param, h0))
        r0 = scale_factor * np.dot(inv_intrinsics_param, h0)
        r1 = scale_factor * np.dot(inv_intrinsics_param, h1)
        t = scale_factor * np.dot(inv_intrinsics_param, h2)
        r2 = np.cross(r0, r1)

        R = np.array([r0, r1, r2, t]).transpose()
        extrinsic_param.append(R)
    return extrinsic_param


if __name__ == '__main__':
    file_dir = r'./image'
    pic_name = os.listdir(file_dir)
    # 角点为7*7
    w = 7
    h = 7
    cross_corners = [w, h]
    real_coor = np.zeros((cross_corners[0] * cross_corners[1], 3), np.float32)  # 每个点的真实坐标 54*3
    real_coor[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 设定初始值
    real_points = []
    real_points_x_y = []
    pic_points = []
    for pic in pic_name:
        # 读取图片
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv2.imread(pic_path)

        # 寻找到圆盘角点， ret为标志位，成功为true，失败为false；pic_coor为角点在图片中的像素位置，格式为54*1*2
        ret, pic_coor = cv2.findCirclesGrid(pic_data, (cross_corners[0], cross_corners[1]), None)
        print(ret)
        if ret:
            # 添加每幅图的对应3D-2D坐标
            pic_coor = pic_coor.reshape(-1, 2)  # 整理为54*2
            pic_points.append(pic_coor)
            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:, :2])

    # 求单应矩阵
    H = get_homography(pic_points, real_points_x_y)
    # 求内参
    intrinsic_param = get_intrinsic_param(H)
    # 求对应每幅图外参
    extrinsic_param = get_extrinsic_param(H, intrinsic_param)
    # 打印相机参数
    print(f'intrinsic_param:\n{intrinsic_param}')
    print(f'extrinsic_param:')
    for item in extrinsic_param:
        print(item)


