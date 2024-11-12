# src/train.py

import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import os
import cv2
import xml.etree.ElementTree as ET
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def get_voc_dicts(subset_dir, split):
    """Load and parse VOC subset annotations."""
    dataset_dicts = []
    split_file = os.path.join(subset_dir, "ImageSets", "Main", f"{split}.txt")

    # 读取split文件中的图像ID列表
    with open(split_file, "r") as f:
        image_ids = [x.strip() for x in f.readlines()]

    # 遍历每张图片
    for image_id in image_ids:
        record = {}
        # 获取图像路径和尺寸
        filename = os.path.join(subset_dir, "JPEGImages", f"{image_id}.jpg")
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width

        # 解析XML标注
        anno_file = os.path.join(subset_dir, "Annotations", f"{image_id}.xml")
        tree = ET.parse(anno_file)
        root = tree.getroot()

        # 处理每个目标
        objs = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in selected_classes:
                continue

            bbox = obj.find("bndbox")
            obj_dict = {
                "bbox": [
                    float(bbox.find("xmin").text),
                    float(bbox.find("ymin").text),
                    float(bbox.find("xmax").text),
                    float(bbox.find("ymax").text)
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": selected_classes.index(class_name),
                "iscrowd": 0
            }
            objs.append(obj_dict)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_voc_dataset(subset_dir):
    """注册VOC数据集到detectron2"""
    global selected_classes
    selected_classes = [
        'person', 'car', 'cat', 'dog', 'chair',
        'bottle', 'bus', 'bicycle'
    ]

    for d in ["train", "val"]:
        dataset_name = f"voc_subset_{d}"
        if dataset_name in DatasetCatalog:
            DatasetCatalog.remove(dataset_name)
        DatasetCatalog.register(
            dataset_name,
            lambda d=d: get_voc_dicts(subset_dir, d)
        )
        MetadataCatalog.get(dataset_name).set(thing_classes=selected_classes)

    return selected_classes


def setup_cfg(num_classes, output_dir):
    """设置训练配置"""
    cfg = get_cfg()

    # 使用 Faster R-CNN 配置
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))

    # 数据集配置
    cfg.DATASETS.TRAIN = ("voc_subset_train",)
    cfg.DATASETS.TEST = ("voc_subset_val",)  # 添加验证集

    # 数据加载配置
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True  # 过滤空标注的图片

    # 预训练权重
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )

    # 训练参数配置
    cfg.SOLVER.IMS_PER_BATCH = 2  # 如果GPU内存够用,可以增加到4
    cfg.SOLVER.BASE_LR = 0.001  # 稍微增加学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 100  # 添加warmup
    cfg.SOLVER.MAX_ITER = 2000  # 增加迭代次数

    # 学习率调度
    cfg.SOLVER.STEPS = (1000, 1500)  # 在这些迭代点降低学习率
    cfg.SOLVER.GAMMA = 0.1  # 学习率衰减因子

    # 优化器配置
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001

    # 模型配置
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # 稍微增加以提高小目标检测
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # 训练时的置信度阈值
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    # 数据增强
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # 训练过程配置
    cfg.TEST.EVAL_PERIOD = 200  # 每200次迭代评估一次
    cfg.SOLVER.CHECKPOINT_PERIOD = 200  # 每200次迭代保存一次模型

    # 输出配置
    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()  # 防止配置被意外修改
    return cfg


def main():
    # 1. 创建输出目录
    model_dir = "./output/voc/models"
    os.makedirs(model_dir, exist_ok=True)

    # 2. 注册数据集
    subset_dir = "./data/VOC2012_subset"
    selected_classes = register_voc_dataset(subset_dir)

    # 3. 配置训练参数
    cfg = setup_cfg(num_classes=len(selected_classes), output_dir=model_dir)

    # 4. 开始训练
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()