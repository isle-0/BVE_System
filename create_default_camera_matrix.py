import numpy as np
import argparse

def create_default_camera_matrix():
    parser = argparse.ArgumentParser(description='创建默认相机内参矩阵')
    parser.add_argument('--width', type=int, default=1920, help='图像宽度（像素）')
    parser.add_argument('--height', type=int, default=1080, help='图像高度（像素）')
    parser.add_argument('--output', type=str, default='camera_matrix.npy', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 创建默认相机内参矩阵
    # 假设焦距为图像宽度，主点在图像中心
    fx = args.width
    fy = args.width
    cx = args.width / 2
    cy = args.height / 2
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # 保存矩阵
    np.save(args.output, camera_matrix)
    print(f"默认相机内参矩阵已保存到 {args.output}")
    print("相机内参矩阵:")
    print(camera_matrix)
    print("\n注意：这是一个默认值，仅用于测试。对于实际应用，请使用准确的相机标定结果。")

if __name__ == '__main__':
    create_default_camera_matrix() 