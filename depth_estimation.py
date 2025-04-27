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
        Initialize the depth estimator
        
        Args:
            model_name: DPT model name, default using Intel/dpt-large
        """
        print(f"Loading DPT model: {model_name}")
        # 加载DPT模型和特征提取器
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        
        # 如果有GPU则使用GPU
        if torch.cuda.is_available():
            print("Using GPU for depth estimation")
            self.model = self.model.to("cuda")
        else:
            print("Using CPU for depth estimation")
        
        # 设置为评估模式
        self.model.eval()
        print("DPT model loaded successfully")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate the depth map of the image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            depth_map: Depth map (unit: meters)
        """
        return self.dpt_depth_estimation(image)
    
    def dpt_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """
        Use the DPT model to estimate the depth map
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            depth_map: 深度图 ( 单位：米)
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
        print(f"[DEBUG] Original image size: {image.shape}")
        
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
        print(f"[DEBUG] Raw depth map - min: {depth_raw.min():.2f}, max: {depth_raw.max():.2f}, mean: {depth_raw.mean():.2f}")
        
        # 使用分位数进行归一化
        min_val = np.percentile(depth_raw, 2)  # 2%分位数
        max_val = np.percentile(depth_raw, 98)  # 98%分位数
        
        print(f"[DEBUG] Adaptive depth mapping range: {min_val:.2f} ~ {max_val:.2f}")
        
        # 裁剪和归一化
        depth_clipped = np.clip(depth_raw, min_val, max_val)
        depth_norm = (depth_clipped - min_val) / (max_val - min_val)
        
        # 映射到0.5-3米范围（更适合近场场景）
        depth_meters = 0.5 + depth_norm * 3.5
        
        print(f"[DEBUG] Final depth map range: {depth_meters.min():.2f}m ~ {depth_meters.max():.2f}m")
        
        # 保存深度分布直方图
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(depth_raw.flatten(), bins=100, color='skyblue')
            plt.axvline(min_val, color='red', linestyle='--', label='2% cutoff')
            plt.axvline(max_val, color='green', linestyle='--', label='98% cutoff')
            plt.title("Raw DPT Depth Distribution")
            plt.legend()
            plt.savefig("depth_histogram.png")
            plt.close()
            print("[DEBUG] Depth distribution histogram saved: depth_histogram.png")
        except Exception as e:
            print(f"[DEBUG] Failed to save depth distribution histogram: {str(e)}")
        
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
        
    def get_3d_points(self, depth_map: np.ndarray, camera_height: float) -> np.ndarray:
        """
        Convert the depth map to a 3D point cloud (suitable for panoramic equirectangular images)

        Args:
            depth_map: Depth map (unit: meters)
            camera_height: Camera installation height (meters)

        Returns:
            points_3d: Nx4 的地面平面点云 (X: 横向  , Z: 前向, row: 像素行, col: 像素列)
        """
        import math
        h, w = depth_map.shape
        pts = []
        # 对每个像素按球面映射计算射线方向，再和地面 (y = -camera_height) 求交点
        for i in range(h):
            phi = (0.5 - (i + 0.5) / h) * math.pi  # 垂直角 φ ∈ [-π/2, π/2]
            for j in range(w):
                theta = (((j + 0.5) / w) - 0.5) * 2 * math.pi  # 水平角 θ ∈ [-π, π]
                dx = math.cos(phi) * math.sin(theta)
                dy = math.sin(phi)
                dz = math.cos(phi) * math.cos(theta)
                if dy >= 0:
                    continue  # 朝上不交地面
                t = -camera_height / dy
                x = dx * t
                z = dz * t
                pts.append([x, z, i, j])
        if not pts:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(pts, dtype=np.float32)