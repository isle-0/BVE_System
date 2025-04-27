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
    Load all images from the specified directory
    
    Args:
        input_dir: Input image directory
        
    Returns:
        images: List of images
    """
    images = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                print(f"Loaded image: {filename}")
    return images

def create_default_camera_matrix() -> np.ndarray:
    """
    Create default camera intrinsic matrix
    
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
    """
    # Use provided camera intrinsic matrix
    camera_matrix = np.array([
        [2827,    0, 2016],
        [   0, 2827, 1512],
        [   0,    0,    1]
    ])
    return camera_matrix

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Panorama stitching and BEV generation')
    parser.add_argument('--input_dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--camera_matrix', type=str, help='Camera intrinsic matrix file path')
    parser.add_argument('--camera_height', type=float, default=1.5, help='Camera installation height (meters)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--bev_range', type=float, nargs=4,
                        default=[-2, 2, -2, 2],
                        help='BEV range [x_min x_max z_min z_max], unit: meters. Recommended [-R, R, -R, R] to cover full 360°')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    print("Loading images...")
    images = load_images(args.input_dir)
    if not images:
        print("Error: No images found")
        return
    
    # Load or create camera intrinsic matrix
    if args.camera_matrix and os.path.exists(args.camera_matrix):
        camera_matrix = np.load(args.camera_matrix)
        print("Loaded camera intrinsic matrix")
    else:
        camera_matrix = create_default_camera_matrix()
        print("Using default camera intrinsic matrix")
    
    # Initialize panorama stitcher
    print("Initializing panorama stitcher...")
    stitcher = PanoramaStitcher()
    
    # Stitch images
    print("Stitching images...")
    success, panorama = stitcher.stitch_images(images)
    if not success:
        print("Error: Image stitching failed")
        return
    
    # Save panorama
    panorama_path = os.path.join(args.output_dir, 'panorama.jpg')
    cv2.imwrite(panorama_path, panorama)
    print(f"Saved panorama: {panorama_path}")
    
    # Initialize depth estimator
    print("Initializing depth estimator...")
    depth_estimator = DepthEstimator()
    
    # Estimate depth map
    print("Estimating depth map...")
    depth_map = depth_estimator.estimate_depth(panorama)
    
    # Visualize depth map
    print("Visualizing depth map...")
    depth_vis = depth_estimator.visualize_depth(depth_map)
    
    # Save depth map
    depth_path = os.path.join(args.output_dir, 'depth_map.jpg')
    cv2.imwrite(depth_path, depth_vis)
    print(f"Saved depth map: {depth_path}")
    
    # Initialize BEV transformer
    print("Initializing BEV transformer...")
    bev_transformer = BEVTransformer(camera_matrix, args.camera_height)
    
    # Set BEV range
    bev_range = tuple(args.bev_range)
    print(f"Using BEV range: {bev_range}")

    # Generate BEV image
    print("Generating BEV image...")
    bev_size = (1200, 1200)  # BEV image size
    points_2d = depth_estimator.get_3d_points(depth_map, args.camera_height)
    bev_image = bev_transformer.image_to_bev(panorama, points_2d, bev_size, bev_range)
    
    # Save BEV image
    bev_path = os.path.join(args.output_dir, 'bev_image.jpg')
    cv2.imwrite(bev_path, bev_image)
    print(f"Saved BEV image: {bev_path}")

    # ===== Detect pedestrians and vehicles =====
    print("Detecting objects in panorama...")
    detector = ObjectDetector()
    boxes, labels, scores = detector.detect(panorama)

    # Draw boxes on panorama
    for box, label_idx, score in zip(boxes, labels, scores):
        label = detector.COCO_INSTANCE_CATEGORY_NAMES[label_idx]
        # Only draw person and car
        if label == 'person':
            color = (0, 0, 255)  # Red box - person
        elif label == 'car':
            color = (0, 255, 0)  # Green box - car
        else:
            continue  # Skip other categories
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(panorama, (x1, y1), (x2, y2), color, 2)
        cv2.putText(panorama, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save annotated panorama
    annotated_path = os.path.join(args.output_dir, 'panorama_with_objects.jpg')
    cv2.imwrite(annotated_path, panorama)
    print(f"Saved annotated panorama: {annotated_path}")

    # ===== Get 3D positions, project to BEV, and annotate on BEV =====
    print("Estimating object positions on BEV map...")
    object_positions = detector.get_object_positions(boxes, labels, scores, depth_map, camera_matrix)
    # Filter person and car, and prepare data format for visualizer
    bev_objects = {}
    for name, info in object_positions.items():
        if name.startswith('person') or name.startswith('car'):
            bev_objects[name] = info

    # Convert positions to BEV coordinates
    position_only = {k: v['position'] for k, v in bev_objects.items()}
    bev_positions = bev_transformer.project_objects_to_bev(position_only, bev_size, bev_range)

    # Map positions back for BEVVisualizer
    for k, (bev_x, bev_y) in bev_positions.items():
        bev_objects[k]['position'] = (int(bev_x), int(bev_y))

    # Use BEVVisualizer to draw objects
    # visualizer = BEVVisualizer()
    # bev_annotated = visualizer.draw_bev_map(bev_image, bev_objects)
    # Draw annotations manually
    bev_annotated = bev_image.copy()
    for name, info in bev_objects.items():
        x, y = info['position']
        if 'person' in name:
            color = (0, 0, 255)  # Red, note OpenCV uses BGR
            cv2.circle(bev_annotated, (x, y), radius=5, color=color, thickness=-1)
        elif 'car' in name:
            color = (0, 255, 0)  # Green
            cv2.rectangle(bev_annotated, (x - 5, y - 5), (x + 5, y + 5), color, thickness=-1)

    # Save BEV image with object annotations
    bev_output_path = os.path.join(args.output_dir, 'bev_image_with_objects.jpg')
    cv2.imwrite(bev_output_path, bev_annotated)
    print(f"Saved annotated BEV image: {bev_output_path}")

    print("Processing completed!")

if __name__ == '__main__':
    main()