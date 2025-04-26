import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class BEVVisualizer:
    def __init__(self):
        # 定义不同类别的颜色和图标
        self.colors = {
            'person': (255, 0, 0),    # 红色
            'car': (0, 255, 0),       # 绿色
            'truck': (0, 0, 255),     # 蓝色
            'bus': (255, 255, 0),     # 青色
            'motorcycle': (255, 0, 255), # 紫色
            'bicycle': (0, 255, 255)   # 黄色
        }
        
        # 加载图标
        self.icons = {}
        self.load_icons()
        
    def load_icons(self):
        """
        加载目标类别的图标
        """
        # 这里应该加载实际的图标文件
        # 为了示例，我们使用简单的几何图形代替
        icon_size = (32, 32)
        
        # 行人图标
        person_icon = np.zeros((*icon_size, 3), dtype=np.uint8)
        cv2.circle(person_icon, (16, 16), 8, (255, 0, 0), -1)
        self.icons['person'] = person_icon
        
        # 车辆图标
        car_icon = np.zeros((*icon_size, 3), dtype=np.uint8)
        cv2.rectangle(car_icon, (8, 12), (24, 20), (0, 255, 0), -1)
        self.icons['car'] = car_icon
        
        # 其他类别的图标类似...
        
    def draw_bev_map(self, bev_image: np.ndarray, objects: Dict,
                     show_grid: bool = True) -> np.ndarray:
        """
        在鸟瞰图上绘制检测到的目标
        
        Args:
            bev_image: 鸟瞰图
            objects: 检测到的目标信息
            show_grid: 是否显示网格
            
        Returns:
            visualized_image: 可视化后的图像
        """
        # 复制输入图像
        visualized_image = bev_image.copy()
        
        # 添加网格
        if show_grid:
            self._add_grid(visualized_image)
            
        # 绘制每个目标
        for obj_class, obj_info in objects.items():
            if obj_class in self.colors:
                position = obj_info['position']
                confidence = obj_info['confidence']
                
                # 获取图标
                icon = self.icons.get(obj_class)
                if icon is not None:
                    # 在目标位置绘制图标
                    x, y = position
                    h, w = icon.shape[:2]
                    visualized_image[y-h//2:y+h//2, x-w//2:x+w//2] = icon
                    
                # 添加标签
                label = f"{obj_class}: {confidence:.2f}"
                cv2.putText(visualized_image, label, (x+20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[obj_class], 2)
                
        return visualized_image
        
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
            
    def visualize_results(self, original_image: np.ndarray,
                         bev_image: np.ndarray,
                         objects: Dict) -> None:
        """
        可视化原始图像、鸟瞰图和检测结果
        
        Args:
            original_image: 原始图像
            bev_image: 鸟瞰图
            objects: 检测到的目标信息
        """
        # 创建图形
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')
        
        # 显示鸟瞰图
        plt.subplot(132)
        plt.imshow(cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB))
        plt.title('鸟瞰图')
        plt.axis('off')
        
        # 显示带标注的鸟瞰图
        plt.subplot(133)
        annotated_bev = self.draw_bev_map(bev_image, objects)
        plt.imshow(cv2.cvtColor(annotated_bev, cv2.COLOR_BGR2RGB))
        plt.title('带标注的鸟瞰图')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show() 