# Monocular Bird's-Eye-View (BEV) System

Our group project implements a Bird's-Eye-View system based on a monocular camera. It can generate a top-down view from images captured from different angles and annotate detected objects (such as pedestrians, vehicles, etc.) within this view. The system achieves the complete transformation process from raw images to a BEV map through the coordination of multiple modules.

## Features
This system integrates several key computer vision modules to achieve monocular BEV perception:

*   **Panorama Stitching:** 
While the core BEV transformation operates on data conceptually from a single viewpoint, panorama stitching can be employed as a preliminary step to create a wider Field of View (FoV) input image. This might involve stitching images captured sequentially by a moving camera or from a rotating sensor setup. The primary importance of stitching in this context is to provide the subsequent BEV transformation module with a broader view of the environment, allowing the resulting BEV map to cover a larger spatial area, which is crucial for applications requiring enhanced situational awareness. However, it's important to recognize that the stitching process itself can introduce errors. Misalignments due to inaccurate keypoint matching or limitations of the geometric model (e.g., the homography's planarity assumption) can create local geometric inconsistencies in the stitched panorama. Since the BEV transformation relies heavily on the geometric integrity of its input, these stitching errors can propagate, leading to distortions or inaccuracies in the final BEV map.

*   **Monocular Depth Estimation:** 
Depth estimation is arguably the most critical component for monocular BEV generation. It aims to infer the distance from the camera to each point in the scene, providing the essential third dimension missing from the 2D input image. This depth information is fundamental for projecting image pixels or features into the 3D world coordinate system, which is a prerequisite for generating the top-down BEV representation.

Modern approaches predominantly rely on deep learning models, such as Convolutional Neural Networks (CNNs) and, more recently, Vision Transformers (ViTs). These models are trained on large-scale datasets containing RGB images paired with corresponding ground truth depth data (often acquired using LiDAR or other sensors). They learn to predict a dense depth map, assigning a depth value to each pixel, or sometimes a probability distribution over possible depths for each pixel. This project utilizes the MiDaS (Monocular Depth Estimation for Autonomous Systems) model family, which is widely recognized for its robustness and accuracy across diverse scenes, achieved through training on a mix of multiple datasets. MiDaS typically predicts relative inverse depth, meaning the output represents 1/depth up to an unknown scale and shift

The accuracy of this depth estimation step directly dictates the geometric fidelity of the final BEV map. Errors in predicted depth values will cause objects and structures to be misplaced along the viewing ray when projected into 3D, resulting in incorrect positioning in the BEV output. Furthermore, because monocular vision lacks an inherent metric scale reference , the depth predicted by models like MiDaS is often relative. To generate a BEV map in meaningful metric units (e.g., meters), which is necessary for planning and navigation , the system must incorporate a mechanism to resolve this scale ambiguity. This could involve using known parameters like camera height above the ground, leveraging learned object size priors, or integrating information from other sources like visual odometry. 

*   **Object Detection and Tracking:** 
Object detection serves to identify and locate relevant entities within the scene, such as vehicles, pedestrians, cyclists, and traffic signs. This provides crucial semantic context to the geometric representation.

This task is typically accomplished using deep learning-based object detectors. Models like YOLO (You Only Look Once), Faster R-CNN, or DETR are trained on extensive datasets (e.g., COCO, KITTI, nuScenes) annotated with bounding boxes and class labels for various object categories. This project employs YOLOv5, a popular choice known for its excellent balance of speed and accuracy, making it suitable for real-time or near-real-time applications. YOLOv5 is a single-stage detector featuring architectures like CSPDarknet as the backbone and PANet in the neck, offering various model sizes (n, s, m, l, x) to cater to different computational budgets.

The output of the object detector (bounding boxes and class labels in the perspective view) is combined with the estimated depth information to project the detected objects onto the BEV map. This allows the system to represent not just the geometry of the scene but also the locations and types of dynamic and static objects within it, which is vital for tasks like collision avoidance, behavior prediction, and interaction planning. Performing detection in the perspective view leverages the high resolution and rich texture information available in the input image. However, the accuracy of the object's final position in the BEV map becomes dependent on the accuracy of the depth estimated for that object. An alternative architectural choice involves transforming image features to the BEV space first and then performing detection on the BEV feature map. This might offer better handling of perspective distortion and occlusion but could be affected by information loss or artifacts introduced during the view transformation process itself.

*   **Bird's-Eye-View Transformation:** 
The BEV transformation module is the core component responsible for converting the visual information from the camera's native perspective view (PV) into the desired top-down Bird's-Eye-View (BEV) coordinate frame.

Fundamentally, this transformation relies on geometric principles, utilizing the camera's calibration parameters (both intrinsic and extrinsic) and depth information (either estimated or assumed). The process involves projecting points or features from the 2D image plane out into the 3D world and then orthographically projecting these 3D points down onto a predefined 2D ground plane representing the BEV map.

The choice of transformation method involves trade-offs between simplicity, computational cost, reliance on intermediate estimates (like depth), assumptions about the world (flat ground), and robustness to potential errors (e.g., inaccurate calibration ). Regardless of the method, this transformation is the crucial step that enables the system to represent and reason about the environment in the geometrically consistent top-down view required for effective planning and navigation.

*   **Object Position Annotation:** 
This final module processes the generated BEV data (which might be a dense feature map, a semantic segmentation map, or a list of object coordinates) and renders it into a human-understandable format.Visualization is essential for debugging the system, qualitatively assessing performance, comparing different algorithms or parameters, and providing intuitive feedback in real-world applications like advanced driver-assistance systems (ADAS). An interpretable BEV representation can significantly aid in identifying potential failure modes or limitations of the perception system.

## System Architecture

The system consists of the following core modules:

1.  **Panorama Stitching Module (`panorama.py`)**
    *   Uses OpenCV's `Stitcher` class for image stitching.
    *   Automatically detects overlapping regions between images.
    *   Implements seamless stitching using feature point matching and image fusion algorithms.
    *   Supports continuous stitching of multiple images.

2.  **Depth Estimation Module (`depth_estimation.py`)**
    *   Performs monocular depth estimation based on deep learning models.
    *   Uses a pre-trained MiDaS model.
    *   Generates pixel-level depth maps.
    *   Supports visualization and post-processing of depth maps.

3.  **Object Detection Module (`object_detection.py`)**
    *   Uses the YOLOv5 model for object detection.
    *   Supports detection of multiple object categories (pedestrians, vehicles, etc.).
    *   Implements object tracking and ID assignment.
    *   Provides visualization of detection results.

4.  **Bird's-Eye-View Transformation Module (`bev_transform.py`)**
    *   Implements the transformation from image space to BEV space.
    *   Uses the camera intrinsic matrix for projection transformation.
    *   Supports custom BEV range and resolution.
    *   Handles reprojection of depth information and object positions.

5.  **Visualization Module (`visualization.py`)**
    *   Provides various visualization tools.
    *   Supports visualization of depth maps and object detection results.
    *   Generates visualization effects for the BEV map.
    *   Offers an interactive visualization interface (potential feature).

## Installation

1.  **Clone the repository:**
```bash
git clone [repository-url]
cd BEV_System
```
2.  **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

1.  **Prepare Image Data:**
    *   Capture scenes from different angles using a monocular camera.
    *   Ensure sufficient overlap between images.
    *   Supported image formats: JPG, PNG, JPEG.

2.  **Run the System:**
    ```bash
    python main.py --input_dir /path/to/images --output_dir /path/to/output
    ```

3.  **Command-Line Arguments:**
    *   `--input_dir`: Directory containing input images (required).
    *   `--output_dir`: Directory for output results (required).
    *   `--camera_matrix`: Path to the camera intrinsic matrix file (optional).
    *   `--camera_height`: Camera installation height in meters (default: 1.5).
    *   `--bev_range`: BEV range `[x_min x_max y_min y_max]` in meters (default: `[-2, 2, 0, 4]`).
    *   `--debug`: Enable debug mode (optional).

## Project Structure

```
BEV_System/
├── main.py                    # Main program entry point
├── panorama.py               # Panorama stitching module
├── depth_estimation.py       # Depth estimation module
├── object_detection.py       # Object detection module
├── bev_transform.py          # BEV transformation module
├── visualization.py          # Visualization module
├── create_default_camera_matrix.py  # Camera parameter configuration utility
├── requirements.txt          # Project dependencies
├── output/                   # Output directory
└── pics/                     # Example image directory
```

## Implementation Details

### Camera Parameter Configuration

*   **Default Camera Intrinsic Matrix:**
```
[2827,    0, 2016]
[   0, 2827, 1512]
[   0,    0,    1]
```
*   Supports loading custom camera parameters from a file.

### Depth Estimation

*   Uses the MiDaS model for depth estimation.
*   Supports selection from various pre-trained models.
*   Provides post-processing and optimization for depth maps.

### Bird's-Eye-View Transformation

*   Based on principles of projective geometry.
*   Considers camera height and installation angle.
*   Supports custom transformation range and resolution.

## Notes and Considerations

*   Ensure camera parameters are correctly calibrated for accurate results.
*   Image quality significantly impacts the final BEV output.
*   Shooting in well-lit environments is recommended.
*   Sufficient overlap between images is necessary for successful panorama stitching.
*   Depth estimation accuracy depends on scene complexity and lighting conditions.

## Performance Optimization (Potential Areas)

*   Multi-threading for image stitching process.
*   GPU acceleration support (if available).
*   Image preprocessing optimization.
*   Memory usage optimization.

## Future Improvements (Potential Directions)

*   Support for real-time processing.
*   Addition of more object categories for detection.
*   Optimization of depth estimation accuracy.
*   Improvement of the BEV transformation algorithm.
*   Addition of more visualization options.
```
