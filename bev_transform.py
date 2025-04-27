import numpy as np
import cv2
from typing import Tuple, List

class BEVTransformer:
    def __init__(self, camera_matrix: np.ndarray, camera_height: float):
        """
        Initialize the BEV transformer
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            camera_height: Camera installation height (meters)
        """
        self.camera_matrix = camera_matrix
        self.camera_height = camera_height
        
        # Print camera parameters for debugging
        print(f"Camera intrinsic matrix:\n{camera_matrix}")
        print(f"Camera height: {camera_height} meters")
        
    def compute_bev_transform(self, image_size: tuple, bev_size: tuple,
                            bev_range: tuple) -> np.ndarray:
        """
            Compute the transformation matrix from the image to the BEV
            
            Args:
                image_size: Input image size (height, width)
                bev_size: BEV size (height, width)
                bev_range: BEV range (x_min, x_max, y_min, y_max)
            
        Returns:
            transform_matrix: 3x3 transformation matrix
        """
        # Get camera parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Calculate projection point from image to ground
        ground_point = np.array([cx, cy + self.camera_height * fy, 1])
        
        # Calculate transformation matrix
        transform_matrix = np.array([
            [fx / self.camera_height, 0, -cx / self.camera_height],
            [0, fy / self.camera_height, -cy / self.camera_height],
            [0, 1 / self.camera_height, 0]
        ])
        
        return transform_matrix
        
    def image_to_bev(self,
                     panorama: np.ndarray,
                     points_2d: np.ndarray,
                     bev_size: Tuple[int,int],
                     bev_range: Tuple[float,float,float,float]) -> np.ndarray:
        """
        Project the ground point cloud to the BEV (Bird's Eye View)
        
        Args:
            panorama: Original panoramic image for color sampling
            points_2d: Nx4 array (X, Z, row, col)
            bev_size: (width, height) Output BEV image size
            bev_range: (x_min, x_max, z_min, z_max) Ground coordinate range, unit: meters
        Returns:
            bev_image: BEV Bird's Eye View image
        """
        bev_w, bev_h = bev_size
        x_min, x_max, z_min, z_max = bev_range
        bev = np.zeros((bev_h, bev_w, 3), dtype=panorama.dtype)
        for x, z, i, j in points_2d:
            if x_min <= x <= x_max and z_min <= z <= z_max:
                u = int((x - x_min) / (x_max - x_min) * (bev_w - 1))
                v = int((1 - (z - z_min) / (z_max - z_min)) * (bev_h - 1))
                color = panorama[int(i), int(j)]
                bev[v, u] = color
        return bev
    

    def simple_projection(self, image: np.ndarray, bev_size: Tuple[int, int],
                         bev_range: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Generate a Bird's Eye View (BEV) image using a simple projection method
        
        Args:
            image: Input image
            bev_size: BEV size (width, height)
            bev_range: BEV range (x_min, x_max, y_min, y_max) unit: meters
            
        Returns:
            bev_image: BEV image
        """
        # 创建鸟瞰图画布
        bev_image = np.zeros((bev_size[1], bev_size[0], 3), dtype=np.uint8)
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 计算像素到米的转换比例
        x_scale = bev_size[0] / (bev_range[1] - bev_range[0])
        y_scale = bev_size[1] / (bev_range[3] - bev_range[2])
        
        # 创建网格点
        y_coords, x_coords = np.mgrid[0:h, 0:w].reshape(2, -1)
        
        # 限制处理点的数量，避免内存问题
        max_points = 100000
        if len(x_coords) > max_points:
            # 随机采样点
            indices = np.random.choice(len(x_coords), max_points, replace=False)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
        
        # 使用基于图像位置的估计深度
        # 假设远处的点深度更大
        depths = 0.1 + 0.9 * (y_coords / h)  # 从0.1到1.0的深度值
        
        # 将深度值转换为实际距离（米）
        depths = 2 + depths * 10  # 2米到12米范围
        
        # 计算3D点坐标
        # 使用相机内参矩阵将2D像素坐标转换为3D点
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 计算相对于相机的3D坐标
        X = (x_coords - cx) * depths / fx
        Y = (y_coords - cy) * depths / fy
        Z = depths
        
        # 计算鸟瞰图坐标 - 使用正确的坐标映射方式
        bev_x = (X - bev_range[0]) * x_scale
        bev_y = (Z - bev_range[2]) * y_scale
        
        # 过滤掉超出鸟瞰图范围的点
        valid_indices = (bev_x >= 0) & (bev_x < bev_size[0]) & (bev_y >= 0) & (bev_y < bev_size[1])
        bev_x = bev_x[valid_indices]
        bev_y = bev_y[valid_indices]
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        
        if len(bev_x) == 0:
            print("Warning: Simple projection method also has no points within the BEV range, trying to expand the range")
            # 尝试放宽范围
            expanded_range = (
                bev_range[0] - 5,  # 左侧扩展5米
                bev_range[1] + 5,  # 右侧扩展5米
                bev_range[2],      # 前方不变
                bev_range[3] + 10  # 前方扩展10米
            )
            
            # 重新计算比例
            x_scale = bev_size[0] / (expanded_range[1] - expanded_range[0])
            y_scale = bev_size[1] / (expanded_range[3] - expanded_range[2])
            
            # 重新计算坐标
            bev_x = (X - expanded_range[0]) * x_scale
            bev_y = (Z - expanded_range[2]) * y_scale
            
            # 再次过滤
            valid_indices = (bev_x >= 0) & (bev_x < bev_size[0]) & (bev_y >= 0) & (bev_y < bev_size[1])
            bev_x = bev_x[valid_indices]
            bev_y = bev_y[valid_indices]
            x_coords = x_coords[valid_indices]
            y_coords = y_coords[valid_indices]
            
            if len(bev_x) == 0:
                print("Error: Unable to generate a valid BEV image")
                # 返回一个带有网格的空白图像
                bev_image = np.ones((bev_size[1], bev_size[0], 3), dtype=np.uint8) * 255
                self._add_grid(bev_image)
                return bev_image
        
        # 将坐标转换为整数
        bev_x = bev_x.astype(np.int32)
        bev_y = bev_y.astype(np.int32)
        
        # 获取对应点的颜色
        colors = image[y_coords, x_coords]
        
        # 绘制点到鸟瞰图
        for i in range(len(bev_x)):
            # 确保坐标在有效范围内
            if 0 <= bev_x[i] < bev_size[0] and 0 <= bev_y[i] < bev_size[1]:
                bev_image[bev_y[i], bev_x[i]] = colors[i]
        
        # 添加网格
        self._add_grid(bev_image)
        
        return bev_image
        
    def _add_grid(self, image: np.ndarray, grid_size: int = 50):
        """
        Add a grid to the image
        
        Args:
            image: Input image
            grid_size: Grid size (pixels)
        """
        h, w = image.shape[:2]
        
        # 绘制垂直线
        for x in range(0, w, grid_size):
            cv2.line(image, (x, 0), (x, h), (128, 128, 128), 1)
            
        # 绘制水平线
        for y in range(0, h, grid_size):
            cv2.line(image, (0, y), (w, y), (128, 128, 128), 1)
            
    def project_objects_to_bev(self, object_positions: dict, bev_size: tuple,
                             bev_range: tuple) -> dict:
        """
        Project 3D object positions to the BEV coordinate system
        
        Args:
            object_positions: Object position dictionary
            bev_size: BEV size
            bev_range: BEV range
            
        Returns:
            bev_positions: Object positions in the BEV coordinate system
        """
        # 计算像素到米的转换比例
        x_scale = bev_size[0] / (bev_range[1] - bev_range[0])
        y_scale = bev_size[1] / (bev_range[3] - bev_range[2])
        
        bev_positions = {}
        for obj_id, pos in object_positions.items():
            x, y, z = pos
            
            # 计算鸟瞰图坐标 - 使用正确的坐标映射方式
            bev_x = (x - bev_range[0]) * x_scale
            bev_y = (z - bev_range[2]) * y_scale
            
            # 检查是否在鸟瞰图范围内
            if 0 <= bev_x < bev_size[0] and 0 <= bev_y < bev_size[1]:
                bev_positions[obj_id] = (bev_x, bev_y)
                
        return bev_positions 