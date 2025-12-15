"""
眼在手外 计算得是 相机相对于基座得 齐次变换矩阵

基座相对于机械臂末端得齐次变换矩阵 == 机械臂末端相对于基座得齐次变换矩阵得逆 （也就是机械臂位姿变换得齐次变换矩阵得逆）

"""

import csv
import numpy as np


def _normalize_calibration_data(calibration_data, angles_in_degrees=False):
    """Normalize calibration data into (x, y, z, rx, ry, rz) tuples.

    Args:
        calibration_data: Iterable containing either 6 or 7 values per entry.
            If 7 values are present, the first value is treated as an image
            filename and ignored for pose construction.
        angles_in_degrees: When True, convert rx/ry/rz from degrees to radians.

    Returns:
        list[tuple]: Normalized pose tuples with angles in radians.
    """

    normalized = []
    for entry in calibration_data:
        if len(entry) == 7:
            _, x, y, z, rx, ry, rz = entry
        elif len(entry) == 6:
            x, y, z, rx, ry, rz = entry
        else:
            raise ValueError(
                "Calibration data must contain either 6 values (x, y, z, rx, ry, rz) "
                "or 7 values with a filename prefix."
            )

        if angles_in_degrees:
            rx, ry, rz = np.deg2rad([rx, ry, rz])

        normalized.append((x, y, z, rx, ry, rz))

    return normalized


# 打开文本文件
def poses2_main(tag=None, calibration_data=None, angles_in_degrees=False):


    if calibration_data is not None:
        pose_entries = _normalize_calibration_data(calibration_data, angles_in_degrees)
    else:
        with open(f"{tag}", "r",encoding="utf-8") as f:
            # 读取文件中的所有行
            lines = f.readlines()
        pose_entries = [float(i)  for line in lines for i in line.split(',')]
        pose_entries = [pose_entries[i:i + 6] for i in range(0, len(pose_entries), 6)]

    matrices = []

    for pose in pose_entries:
        matrices.append(inverse_transformation_matrix(pose_to_homogeneous_matrix(pose)))


    # 将齐次变换矩阵列表存储到 CSV 文件中
    save_matrices_to_csv(matrices, f'RobotToolPose.csv')

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz@Ry@Rx  # 先绕 x轴旋转 再绕y轴旋转  最后绕z轴旋转
    return R


def pose_to_homogeneous_matrix(pose):

    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H

def inverse_transformation_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]

    # 计算旋转矩阵的逆矩阵
    R_inv = R.T

    # 计算平移向量的逆矩阵
    t_inv = -np.dot(R_inv, t)

    # 构建逆变换矩阵
    T_inv = np.identity(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


def save_matrices_to_csv(matrices, file_name):
    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)

if __name__ == "__main__":
    # 假设已经将位姿列表转换为齐次变换矩阵列表
    # 示例：
    tag = ''
    poses2_main(tag)