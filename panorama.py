import cv2
import numpy as np
from typing import List, Tuple

class PanoramaStitcher:
    def __init__(self):
        self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        
    def stitch_images(self, images: List[np.ndarray]) -> Tuple[bool, np.ndarray]:
        """
        将多张图像拼接成全景图
        
        Args:
            images: 输入图像列表
            
        Returns:
            success: 是否成功
            panorama: 拼接后的全景图
        """
        if len(images) < 2:
            raise ValueError("需要至少两张图像进行拼接")
            
        # 确保所有图像具有相同的大小
        target_size = (images[0].shape[1], images[0].shape[0])
        resized_images = [cv2.resize(img, target_size) for img in images]
        
        # 执行图像拼接
        status, panorama = self.stitcher.stitch(resized_images)
        
        if status == cv2.Stitcher_OK:
            return True, panorama
        else:
            return False, None
            
    def get_homography(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        计算两张图像之间的单应性矩阵
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            homography: 3x3单应性矩阵
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 使用SIFT特征检测和匹配
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # 特征匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 应用比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) < 10:
            raise ValueError("没有足够的匹配点")
            
        # 获取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
        
    def stitch_quad_images(self, front: np.ndarray, back: np.ndarray,
                          left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        专门用于拼接四张图像的函数
        
        Args:
            front: 前方图像
            back: 后方图像
            left: 左方图像
            right: 右方图像
            
        Returns:
            panorama: 拼接后的全景图
        """
        # 首先拼接左右图像
        success, left_right = self.stitch_images([left, right])
        if not success:
            raise ValueError("左右图像拼接失败")
            
        # 然后拼接前后图像
        success, front_back = self.stitch_images([front, back])
        if not success:
            raise ValueError("前后图像拼接失败")
            
        # 最后拼接两个部分
        success, panorama = self.stitch_images([left_right, front_back])
        if not success:
            raise ValueError("最终拼接失败")
            
        return panorama 