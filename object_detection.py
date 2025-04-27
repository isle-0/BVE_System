import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
import numpy as np
import cv2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        
        # COCO数据集的类别
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
    def detect(self, image: np.ndarray) -> tuple:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            boxes: 检测框坐标 [x1, y1, x2, y2]
            labels: 类别标签
            scores: 置信度分数
        """
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PyTorch张量
        image_tensor = F.to_tensor(image_rgb)
        
        # 进行检测
        with torch.no_grad():
            prediction = self.model([image_tensor.to(self.device)])
            
        # 获取检测结果
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        
        # 过滤低置信度的检测结果
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        return boxes, labels, scores
        
    def get_object_positions(self, boxes: np.ndarray, labels: np.ndarray,
                          scores: np.ndarray, depth_map: np.ndarray,
                           camera_matrix: np.ndarray) -> dict:
        """
        计算检测到的目标的3D位置
        
        Args:
            boxes: 检测框坐标
            depth_map: 深度图
            camera_matrix: 相机内参矩阵
            
        Returns:
            object_positions: 包含目标类别和3D位置的字典
        """
        object_positions = {}
        
        for idx, (box, label_idx, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # 获取目标区域的平均深度
            object_depth = depth_map[y1:y2, x1:x2].mean()
            
            # 计算目标中心点的3D位置
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            X = (center_x - cx) * object_depth / fx
            Y = (center_y - cy) * object_depth / fy
            Z = object_depth

            label_name = self.COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            object_id = f"{label_name}_{idx}"  # 加编号，防止同名覆盖
            object_positions[object_id] = {
                'position': np.array([X, Y, Z]),
                'confidence': score
            }
            
        return object_positionsimport torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
import numpy as np
import cv2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        
        # COCO数据集的类别
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
    def detect(self, image: np.ndarray) -> tuple:
        """
        检测图像中的目标
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            boxes: 检测框坐标 [x1, y1, x2, y2]
            labels: 类别标签
            scores: 置信度分数
        """
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PyTorch张量
        image_tensor = F.to_tensor(image_rgb)
        
        # 进行检测
        with torch.no_grad():
            prediction = self.model([image_tensor.to(self.device)])
            
        # 获取检测结果
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        
        # 过滤低置信度的检测结果
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        return boxes, labels, scores
        
    def get_object_positions(self, boxes: np.ndarray, labels: np.ndarray,
                          scores: np.ndarray, depth_map: np.ndarray,
                           camera_matrix: np.ndarray) -> dict:
        """
        计算检测到的目标的3D位置
        
        Args:
            boxes: 检测框坐标
            depth_map: 深度图
            camera_matrix: 相机内参矩阵
            
        Returns:
            object_positions: 包含目标类别和3D位置的字典
        """
        object_positions = {}
        
        for idx, (box, label_idx, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # 获取目标区域的平均深度
            object_depth = depth_map[y1:y2, x1:x2].mean()
            
            # 计算目标中心点的3D位置
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            X = (center_x - cx) * object_depth / fx
            Y = (center_y - cy) * object_depth / fy
            Z = object_depth

            label_name = self.COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            object_id = f"{label_name}_{idx}"  # 加编号，防止同名覆盖
            object_positions[object_id] = {
                'position': np.array([X, Y, Z]),
                'confidence': score
            }
            
        return object_positions