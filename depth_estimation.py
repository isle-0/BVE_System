import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import cv2
import os

class DepthEstimator:
    def __init__(self, model_name: str = "Intel/dpt-large"):
        """
        初始化深度估计器
        
        Args:
            model_name: MiDaS模型名称
        """
        print(f"正在加载MiDaS模型: {model_name}")
        # 加载MiDaS模型和特征提取器
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        
        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            print("使用GPU进行深度估计")
            self.model = self.model.to("cuda")
        else:
            print("使用CPU进行深度估计")
        
        # 设置为评估模式
        self.model.eval()
        print("MiDaS模型加载完成")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        估计图像的深度图
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            depth_map: 深度图 (归一化到0-1范围)
        """
        return self.midas_depth_estimation(image)
    
    def midas_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """
        使用MiDaS模型估计深度图
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            depth_map: 深度图 (归一化到0-1范围)
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_rgb)
        
        # 准备输入
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        
        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 打印原始图像尺寸
        print(f"[DEBUG] 原始图像尺寸: {image.shape}")
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # 获取原始深度值
        depth_raw = prediction.squeeze().cpu().numpy()
        
        # 打印原始深度值统计
        print(f"[DEBUG] 原始深度图最小值: {depth_raw.min():.2f}, 最大值: {depth_raw.max():.2f}, 平均值: {depth_raw.mean():.2f}")
        
        # 使用分位数进行归一化
        min_val = np.percentile(depth_raw, 2)  # 2%分位数
        max_val = np.percentile(depth_raw, 98)  # 98%分位数
        
        print(f"[DEBUG] 自适应深度映射范围: {min_val:.2f} ~ {max_val:.2f}")
        
        # 裁剪和归一化
        depth_clipped = np.clip(depth_raw, min_val, max_val)
        depth_norm = (depth_clipped - min_val) / (max_val - min_val)
        
        # 映射到0.5-3米范围（更适合近场场景）
        depth_meters = 0.5 + depth_norm * 2.5
        
        print(f"[DEBUG] 最终深度图范围: {depth_meters.min():.2f}m ~ {depth_meters.max():.2f}m")
        
        # 保存深度分布直方图
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(depth_raw.flatten(), bins=100, color='skyblue')
            plt.axvline(min_val, color='red', linestyle='--', label='2% cutoff')
            plt.axvline(max_val, color='green', linestyle='--', label='98% cutoff')
            plt.title("Raw MiDaS Depth Distribution")
            plt.legend()
            plt.savefig("depth_histogram.png")
            plt.close()
            print("[DEBUG] 已保存深度分布直方图: depth_histogram.png")
        except Exception as e:
            print(f"[DEBUG] 保存深度分布直方图失败: {str(e)}")
        
        return depth_meters
    
    def visualize_depth(self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        将深度图可视化为彩色图像
        
        Args:
            depth_map: 深度图 (归一化到0-1范围)
            colormap: OpenCV颜色映射
            
        Returns:
            depth_vis: 可视化的深度图
        """
        # 将深度图转换为8位无符号整数
        depth_vis = (depth_map * 255).astype(np.uint8)
        
        # 应用颜色映射
        depth_vis = cv2.applyColorMap(depth_vis, colormap)
        
        return depth_vis
        
    def get_3d_points(self, image: np.ndarray, depth_map: np.ndarray, 
                     camera_matrix: np.ndarray) -> np.ndarray:
        """
        将深度图转换为3D点云
        
        Args:
            image: 输入图像
            depth_map: 深度图
            camera_matrix: 相机内参矩阵
            
        Returns:
            points_3d: Nx3的3D点云数组
        """
        height, width = depth_map.shape
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # 创建像素坐标网格
        y, x = np.mgrid[0:height, 0:width]
        
        # 计算3D坐标
        Z = depth_map
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        
        # 将坐标堆叠成Nx3数组
        points_3d = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # 移除无效点（深度为0或无穷大的点）
        valid_mask = ~np.isnan(points_3d).any(axis=1) & ~np.isinf(points_3d).any(axis=1)
        points_3d = points_3d[valid_mask]
        
        return points_3d 