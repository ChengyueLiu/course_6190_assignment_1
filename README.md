# Detectron2 Balloon Detection Project

This project implements instance segmentation for balloons using the Detectron2 framework. It utilizes a pre-trained Mask R-CNN model fine-tuned on a balloon dataset to detect and segment balloons in images.

## Prerequisites

- Python 3.8+
- CUDA 12.1+ (tested on CUDA 12.4)
- Sufficient GPU memory (8GB+ recommended)
- Linux/macOS/Windows

## Project Setup

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv detectron2_env
source detectron2_env/bin/activate  # Linux/macOS
# detectron2_env\Scripts\activate    # Windows

# Upgrade pip and install basic tools
python -m pip install --upgrade pip
pip install setuptools wheel

# Install PyTorch (CUDA 12.1 version, compatible with CUDA 12.4)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install opencv-python
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install pyyaml==5.1
pip install tensorboard
pip install ninja

# Install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
python -c "import detectron2; print('Detectron2 version:', detectron2.__version__)"
```

### 2. Project Structure Setup
```bash
# Create project directory
mkdir detectron2_project
cd detectron2_project

# Create necessary directories
mkdir data src output

# Download and extract dataset
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip -d data/
```

## Project Structure
```
detectron2_project/
├── data/
│   └── balloon/
│       ├── train/
│       │   ├── via_region_data.json
│       │   └── *.jpg
│       └── val/
│           ├── via_region_data.json
│           └── *.jpg
├── output/
│   ├── model_final.pth     # Trained model
│   └── prediction_*.jpg    # Prediction results
├── src/
│   ├── train.py           # Training script
│   └── inference.py       # Inference script
└── README.md
```

## Usage

### 1. Training
The training script (`src/train.py`) fine-tunes a pre-trained Mask R-CNN model on the balloon dataset:
```bash
python src/train.py
```
This will:
- Load the balloon dataset
- Configure the model parameters
- Train for 300 iterations
- Save the model to `output/model_final.pth`

### 2. Inference
The inference script (`src/inference.py`) runs detection on validation images:
```bash
python src/inference.py
```
This will:
- Load the trained model
- Run inference on random validation images
- Save visualized results to the output directory

## Model Details

- Base Architecture: Mask R-CNN
- Backbone: ResNet50-FPN
- Training Parameters:
  - Base Learning Rate: 0.00025
  - Batch Size: 2
  - Max Iterations: 300
  - ROI Batch Size per Image: 128

## Results

After training, the model can:
- Detect balloons in images
- Generate precise segmentation masks
- Provide confidence scores for detections

Results are saved as images in the `output` directory with visualized bounding boxes and segmentation masks.

## Acknowledgments

- Dataset: [Balloon Dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)
- Framework: [Detectron2](https://github.com/facebookresearch/detectron2)




Literature Review
Method
Experiments
	1. 训练结果
	2. 关键因素分析
	3. 超参数设置分析
	4. 关键组件的消融实验
	5. 优缺点分析
	6. 结果可视化及讨论