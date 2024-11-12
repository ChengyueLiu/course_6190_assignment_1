import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import os
import cv2
import random
import json
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def setup_cityscapes_metadata():
    """设置Cityscapes元数据"""
    # 注册数据集以获取元数据
    register_coco_instances(
        "cityscapes_val",
        {},
        "data/cityscapes_subset/annotations/val.json",
        "data/cityscapes_subset/leftImg8bit/val"
    )
    return MetadataCatalog.get("cityscapes_val")


def get_val_images():
    """获取验证集图片路径列表"""
    # 读取COCO格式的标注文件
    with open("data/cityscapes_subset/annotations/val.json", 'r') as f:
        annotations = json.load(f)

    image_dir = "data/cityscapes_subset/leftImg8bit/val"
    dataset_dicts = []

    for img in annotations["images"]:
        dataset_dicts.append({
            "file_name": os.path.join(image_dir, img["file_name"]),
            "image_id": img["id"]
        })

    return dataset_dicts


def main():
    # 模型目录
    model_dir = "./output/cityscapes_subset/models"

    # 创建输出目录
    output_dir = "./output/cityscapes_subset/inference"
    os.makedirs(output_dir, exist_ok=True)

    # 配置推理参数
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Cityscapes的8个类别
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")  # 使用训练好的模型
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 设置检测阈值

    # 创建预测器
    predictor = DefaultPredictor(cfg)

    # 设置元数据
    cityscapes_metadata = setup_cityscapes_metadata()

    # 获取验证集图片
    dataset_dicts = get_val_images()

    # 随机选择5张图片进行推理
    for d in random.sample(dataset_dicts, min(5, len(dataset_dicts))):
        # 读取图片
        im = cv2.imread(d["file_name"])
        if im is None:
            print(f"Warning: Could not read image {d['file_name']}")
            continue

        # 进行推理
        outputs = predictor(im)

        # 可视化结果
        v = Visualizer(im[:, :, ::-1],
                       metadata=cityscapes_metadata,
                       scale=0.8,  # 城市场景通常较大，稍微调大scale
                       instance_mode=ColorMode.IMAGE_BW  # 使用黑白模式突出实例
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # 保存结果
        output_path = os.path.join(output_dir, f"pred_{os.path.basename(d['file_name'])}")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        print(f"Saved prediction to {output_path}")

        # 打印检测到的实例数量
        num_instances = len(outputs["instances"])
        print(f"Detected {num_instances} instances in {os.path.basename(d['file_name'])}")

        # 打印每个类别的检测数量
        if num_instances > 0:
            pred_classes = outputs["instances"].pred_classes.cpu().numpy()
            class_counts = {}
            for class_id in pred_classes:
                class_name = cityscapes_metadata.thing_classes[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            print("Class distribution:", class_counts)
        print("-" * 50)


if __name__ == "__main__":
    main()