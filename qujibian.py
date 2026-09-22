import cv2
import numpy as np
import os

def undistort_image(image_path, output_path):
    """
    消除图片畸变

    参数:
    image_path: 输入图片路径
    output_path: 输出图片路径
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    # 获取图片尺寸
    h, w = img.shape[:2]

    # 相机内参矩阵 (需要根据相机标定结果填写)
    # 这里使用示例值，实际应用中需要通过相机标定获得
    # fx, fy: 焦距
    # cx, cy: 主点坐标
    camera_matrix = np.array([
        [1.91947981e+03, 0.00000000e+00, 1.26566418e+03],  
        [0.00000000e+00, 1.91941642e+03, 7.04833607e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ], dtype=np.float32)

    # 畸变系数 (需要根据相机标定结果填写)
    # k1, k2: 径向畸变系数
    # p1, p2: 切向畸变系数 
    # k3: 高阶径向畸变系数
    dist_coeffs = np.array([-5.61757289e-01, 3.56650327e-01, -3.45290566e-04, -3.71246347e-04, -1.21195119e-01], dtype=np.float32)

    # 计算最优相机矩阵
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # 去畸变
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 裁剪图像 (可选)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]

    # 保存结果
    cv2.imwrite(output_path, undistorted_img)
    print(f"去畸变完成，结果已保存到: {output_path}")

    # 显示结果 (可选)
    cv2.imshow('Original', img)
    cv2.imshow('Undistorted', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    # 替换为你的图片路径
    input_image = r"D:\tt\tietapingheng\tietapingheng\picture\jinzhun_1.bmp"
    output_image = r"D:\tt\tietapingheng\tietapingheng\new_picture\jinzhun_1_new.png"
    
    undistort_image(input_image, output_image)