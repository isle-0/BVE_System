import argparse
import os
import cv2
import numpy as np

from object_detection import ObjectDetector
from panorama import PanoramaStitcher
from depth_estimation import DepthEstimator
from bev_transform import BEVTransformer
from visualization import BEVVisualizer

def load_images(input_dir: str) -> list:
    """
    从指定目录加载所有图像
    
    Args:
        input_dir: 输入图像目录
        
    Returns:
        images: 图像列表
    """
    images = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                print(f"已加载图像: {filename}")
    return images

def create_default_camera_matrix() -> np.ndarray:
    """
    创建默认的相机内参矩阵
    
    Returns:
        camera_matrix: 3x3相机内参矩阵
    """
    # 使用提供的相机内参矩阵
    camera_matrix = np.array([
        [2827,    0, 2016],
        [   0, 2827, 1512],
        [   0,    0,    1]
    ])
    return camera_matrix

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='全景图像拼接和鸟瞰图生成')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--camera_matrix', type=str, help='相机内参矩阵文件路径')
    parser.add_argument('--camera_height', type=float, default=1.5, help='相机安装高度（米）')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--bev_range', type=float, nargs=4,
                        default=[-2, 2, -2, 2],
                        help='鸟瞰图范围 [x_min x_max z_min z_max]，单位：米。推荐 [-R, R, -R, R] 来覆盖全 360°')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载图像
    print("正在加载图像...")
    images = load_images(args.input_dir)
    if not images:
        print("错误：未找到图像")
        return
    
    # 加载或创建相机内参矩阵
    if args.camera_matrix and os.path.exists(args.camera_matrix):
        camera_matrix = np.load(args.camera_matrix)
        print("已加载相机内参矩阵")
    else:
        camera_matrix = create_default_camera_matrix()
        print("使用默认相机内参矩阵")
    
    # 初始化全景拼接器
    print("正在初始化全景拼接器...")
    stitcher = PanoramaStitcher()
    
    # 拼接图像
    print("正在拼接图像...")
    success, panorama = stitcher.stitch_images(images)
    if not success:
        print("错误：图像拼接失败")
        return
    
    # 保存全景图
    panorama_path = os.path.join(args.output_dir, 'panorama.jpg')
    cv2.imwrite(panorama_path, panorama)
    print(f"已保存全景图: {panorama_path}")
    
    # 初始化深度估计器
    print("正在初始化深度估计器...")
    depth_estimator = DepthEstimator()
    
    # 估计深度图
    print("正在估计深度图...")
    depth_map = depth_estimator.estimate_depth(panorama)
    
    # 可视化深度图
    print("正在可视化深度图...")
    depth_vis = depth_estimator.visualize_depth(depth_map)
    
    # 保存深度图
    depth_path = os.path.join(args.output_dir, 'depth_map.jpg')
    cv2.imwrite(depth_path, depth_vis)
    print(f"已保存深度图: {depth_path}")
    
    # 初始化BEV转换器
    print("正在初始化BEV转换器...")
    bev_transformer = BEVTransformer(camera_matrix, args.camera_height)
    
    # 设置鸟瞰图范围
    bev_range = tuple(args.bev_range)
    print(f"使用鸟瞰图范围: {bev_range}")

    # 生成鸟瞰图
    print("正在生成鸟瞰图...")
    bev_size = (1200, 1200)  # 鸟瞰图大小
    points_2d = depth_estimator.get_3d_points(depth_map, args.camera_height)
    bev_image = bev_transformer.image_to_bev(panorama, points_2d, bev_size, bev_range)
    
    # 保存鸟瞰图
    bev_path = os.path.join(args.output_dir, 'bev_image.jpg')
    cv2.imwrite(bev_path, bev_image)
    print(f"已保存鸟瞰图: {bev_path}")

    # ===== 检测行人和车辆 =====
    print("正在检测全景图中的对象...")
    detector = ObjectDetector()
    boxes, labels, scores = detector.detect(panorama)

    # 在panorama上画框
    for box, label_idx, score in zip(boxes, labels, scores):
        label = detector.COCO_INSTANCE_CATEGORY_NAMES[label_idx]
        # 只画person和car
        if label == 'person':
            color = (0, 0, 255)  # 红色框 - person
        elif label == 'car':
            color = (0, 255, 0)  # 绿色框 - car
        else:
            continue  # 其他类别跳过
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(panorama, (x1, y1), (x2, y2), color, 2)
        cv2.putText(panorama, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 保存标注后的panorama
    annotated_path = os.path.join(args.output_dir, 'panorama_with_objects.jpg')
    cv2.imwrite(annotated_path, panorama)
    print(f"已保存标注后的全景图: {annotated_path}")

    # ===== 获取3D位置，投影到BEV，并标到BEV上 =====
    print("正在估计目标在BEV图上的位置...")
    object_positions = detector.get_object_positions(boxes, labels, scores, depth_map, camera_matrix)
    # 筛选person和car，并且组装好给visualizer用的数据格式
    bev_objects = {}
    for name, info in object_positions.items():
        if name.startswith('person') or name.startswith('car'):
            bev_objects[name] = info

    # 位置转换到BEV坐标系
    position_only = {k: v['position'] for k, v in bev_objects.items()}
    bev_positions = bev_transformer.project_objects_to_bev(position_only, bev_size, bev_range)

    # 把位置映射回来给BEVVisualizer
    for k, (bev_x, bev_y) in bev_positions.items():
        bev_objects[k]['position'] = (int(bev_x), int(bev_y))

    # 用BEVVisualizer来画目标
    # visualizer = BEVVisualizer()
    # bev_annotated = visualizer.draw_bev_map(bev_image, bev_objects)
    # 自己画标注
    bev_annotated = bev_image.copy()
    for name, info in bev_objects.items():
        x, y = info['position']
        if 'person' in name:
            color = (0, 0, 255)  # 红色，注意OpenCV是BGR
            cv2.circle(bev_annotated, (x, y), radius=5, color=color, thickness=-1)
        elif 'car' in name:
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(bev_annotated, (x - 5, y - 5), (x + 5, y + 5), color, thickness=-1)

    # 保存带目标标注的BEV图
    bev_output_path = os.path.join(args.output_dir, 'bev_image_with_objects.jpg')
    cv2.imwrite(bev_output_path, bev_annotated)
    print(f"已保存标注后的鸟瞰图: {bev_output_path}")

    print("处理完成！")

if __name__ == '__main__':
    main()