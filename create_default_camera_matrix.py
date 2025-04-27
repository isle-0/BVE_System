import numpy as np
import argparse

def create_default_camera_matrix():
    parser = argparse.ArgumentParser(description='Create a default camera intrinsic matrix')
    parser.add_argument('--width', type=int, default=1920, help='Image width (pixels)')
    parser.add_argument('--height', type=int, default=1080, help='Image height (pixels)')
    parser.add_argument('--output', type=str, default='camera_matrix.npy', help='Output file path')
    
    args = parser.parse_args()
    
    # Create a default camera intrinsic matrix
    # Assume the focal length is the image width, and the principal point is in the center of the image
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
    print(f"Camera intrinsic matrix saved to {args.output}")
    print("Camera intrinsic matrix:")
    print(camera_matrix)
    print("\nNote: This is a default value for testing purposes. For practical applications, please use accurate camera calibration results.")

if __name__ == '__main__':
    create_default_camera_matrix() 