# 单目相机鸟瞰图(BEV)系统

这个项目实现了一个基于单目相机的鸟瞰图(BEV)系统，可以从不同角度拍摄的图像生成俯视图，并在其中标注检测到的目标（如行人、车辆等）。

## 功能特点

- 全景图像拼接
- 单目深度估计
- 目标检测和跟踪
- 鸟瞰图转换和可视化
- 目标位置标注

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd BEV
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备图像数据：
   - 使用单目相机从不同角度拍摄场景
   - 确保图像之间有足够的重叠区域

2. 运行系统：
```bash
python main.py --input_dir /path/to/images --output_dir /path/to/output
```

## 项目结构

```
BEV/
├── main.py              # 主程序入口
├── panorama.py          # 全景图像拼接模块
├── depth_estimation.py  # 深度估计模块
├── object_detection.py  # 目标检测模块
├── bev_transform.py     # 鸟瞰图转换模块
└── visualization.py     # 可视化模块
```

## 注意事项

- 确保相机参数已正确标定
- 图像质量会影响最终效果
- 建议在光线充足的环境下拍摄 