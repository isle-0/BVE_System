import numpy as np
import cv2
from typing import Tuple, List

class BEVTransformer:
    def __init__(self, camera_matrix: np.ndarray, camera_height: float):
        """
        初始化鸟瞰图转换器
        
        Args:
            camera_matrix: 相机内参矩阵 (3x3)
            camera_height: 相机安装高度（米）
        """
        self.camera_matrix = camera_matrix
        self.camera_height = camera_height
        
        # 打印相机参数以便调试
        print(f"相机内参矩阵:\n{camera_matrix}")
        print(f"相机高度: {camera_height}米")
        
    def compute_bev_transform(self, image_size: tuple, bev_size: tuple,
                            bev_range: tuple) -> np.ndarray:
        """
        计算从图像到鸟瞰图的变换矩阵
        
        Args:
            image_size: 输入图像大小 (height, width)
            bev_size: 鸟瞰图大小 (height, width)
            bev_range: 鸟瞰图范围 (x_min, x_max, y_min, y_max)
            
        Returns:
            transform_matrix: 3x3变换矩阵
        """
        # 获取相机参数
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 计算图像到地面的投影点
        ground_point = np.array([cx, cy + self.camera_height * fy, 1])
        
        # 计算变换矩阵
        transform_matrix = np.array([
            [fx / self.camera_height, 0, -cx / self.camera_height],
            [0, fy / self.camera_height, -cy / self.camera_height],
            [0, 1 / self.camera_height, 0]
        ])
        
        return transform_matrix
        
    def image_to_bev(self, image: np.ndarray, depth_map: np.ndarray,
                     bev_size: Tuple[int, int],
                     bev_range: Tuple[float, float, float, float]) -> np.ndarray:
        """
        将图像和深度图转换为鸟瞰图
        
        Args:
            image: 输入图像
            depth_map: 深度图 (归一化到0-1范围)
            bev_size: 鸟瞰图大小 (width, height)
            bev_range: 鸟瞰图范围 (x_min, x_max, y_min, y_max) 单位：米
            
        Returns:
            bev_image: 鸟瞰图
        """
        # 打印输入参数以便调试
        print(f"输入图像尺寸: {image.shape}")
        print(f"深度图尺寸: {depth_map.shape}")
        print(f"鸟瞰图尺寸: {bev_size}")
        print(f"鸟瞰图范围: {bev_range}")
        
        # 检查图像和深度图尺寸是否匹配
        if image.shape[:2] != depth_map.shape:
            print(f"警告: 图像尺寸 {image.shape[:2]} 与深度图尺寸 {depth_map.shape} 不匹配")
            # 调整深度图大小以匹配图像
            depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
            print(f"已调整深度图尺寸为: {depth_map.shape}")
        
        # 创建鸟瞰图画布
        bev_image = np.zeros((bev_size[1], bev_size[0], 3), dtype=np.uint8)
        
        # 计算像素到米的转换比例
        x_scale = bev_size[0] / (bev_range[1] - bev_range[0])
        y_scale = bev_size[1] / (bev_range[3] - bev_range[2])
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 创建网格点
        y_coords, x_coords = np.mgrid[0:h, 0:w].reshape(2, -1)
        
        # 限制处理点的数量，避免内存问题
        max_points = 100000
        if len(x_coords) > max_points:
            # 随机采样点
            indices = np.random.choice(len(x_coords), max_points, replace=False)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
            print(f"随机采样 {max_points} 个点进行处理")
        
        # 获取这些点的深度值
        depths = depth_map[y_coords, x_coords]
        
        # 打印深度值的基本统计信息
        print(f"深度值最小值: {depths.min()}, 最大值: {depths.max()}, 平均值: {depths.mean()}")
        
        # 如果深度图几乎全黑，使用基于图像位置的估计深度
        if depths.max() < 0.01:  # 如果最大深度值小于0.01
            print("警告: 深度图几乎全黑，使用基于图像位置的估计深度")
            # 使用基于图像位置的估计深度
            # 假设远处的点深度更大
            depths = 0.1 + 0.9 * (y_coords / h)  # 从0.1到1.0的深度值
        
        # 过滤掉深度为0的点
        valid_indices = depths > 0
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        depths = depths[valid_indices]
        
        if len(depths) == 0:
            print("警告: 没有有效的深度值，使用简单的投影方法")
            return self.simple_projection(image, bev_size, bev_range)
        
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
        
        # 打印3D坐标范围以便调试
        print(f"[DEBUG] X范围: {X.min():.2f} ~ {X.max():.2f}")
        print(f"[DEBUG] Z范围: {Z.min():.2f} ~ {Z.max():.2f}")
        print(f"[DEBUG] BEV X范围: {bev_range[0]} ~ {bev_range[1]}")
        print(f"[DEBUG] BEV Y范围: {bev_range[2]} ~ {bev_range[3]}")
        
        # 计算鸟瞰图坐标 - 使用正确的坐标映射方式
        # 使用bev_range进行坐标映射，而不是硬加中心偏移量
        bev_x = (X - bev_range[0]) * x_scale
        bev_y = (Z - bev_range[2]) * y_scale
        
        # 过滤掉超出鸟瞰图范围的点
        valid_indices = (bev_x >= 0) & (bev_x < bev_size[0]) & (bev_y >= 0) & (bev_y < bev_size[1])
        bev_x = bev_x[valid_indices]
        bev_y = bev_y[valid_indices]
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        
        if len(bev_x) == 0:
            print("警告: 没有点落在鸟瞰图范围内，尝试放宽范围")
            # 尝试放宽范围
            expanded_range = (
                bev_range[0] - 5,  # 左侧扩展5米
                bev_range[1] + 5,  # 右侧扩展5米
                bev_range[2],      # 前方不变
                bev_range[3] + 10  # 前方扩展10米
            )
            print(f"尝试使用扩展范围: {expanded_range}")
            
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
                print("警告: 即使放宽范围后仍然没有点落在鸟瞰图范围内，使用简单的投影方法")
                return self.simple_projection(image, bev_size, bev_range)
        
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
        
        # 如果鸟瞰图几乎全黑，使用简单的投影方法
        if np.sum(bev_image) < 1000:  # 如果图像中非零像素很少
            print("警告: 鸟瞰图几乎全黑，使用简单的投影方法")
            return self.simple_projection(image, bev_size, bev_range)
        
        # 添加网格
        self._add_grid(bev_image)
        
        return bev_image
    
    def simple_projection(self, image: np.ndarray, bev_size: Tuple[int, int],
                         bev_range: Tuple[float, float, float, float]) -> np.ndarray:
        """
        使用简单的投影方法生成鸟瞰图
        
        Args:
            image: 输入图像
            bev_size: 鸟瞰图大小 (width, height)
            bev_range: 鸟瞰图范围 (x_min, x_max, y_min, y_max) 单位：米
            
        Returns:
            bev_image: 鸟瞰图
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
            print("警告: 简单投影方法也没有点落在鸟瞰图范围内，尝试放宽范围")
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
                print("错误: 无法生成有效的鸟瞰图")
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
        在图像上添加网格
        
        Args:
            image: 输入图像
            grid_size: 网格大小（像素）
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
        将3D对象位置投影到鸟瞰图坐标系
        
        Args:
            object_positions: 对象位置字典
            bev_size: 鸟瞰图大小
            bev_range: 鸟瞰图范围
            
        Returns:
            bev_positions: 鸟瞰图坐标系中的对象位置
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