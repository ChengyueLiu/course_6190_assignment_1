import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
import json
import torch


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()

        # 添加TensorBoard日志记录
        hooks.insert(-1, TensorboardXWriter(self.cfg.OUTPUT_DIR))

        return hooks


def get_balloon_dicts(img_dir):
    """Load and parse balloon dataset annotations."""
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def main():
    # 创建输出目录
    output_dir = "./output/voc"
    model_dir = os.path.join(output_dir, "models")
    log_dir = os.path.join(output_dir, "logs")
    eval_dir = os.path.join(output_dir, "eval")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # 注册数据集
    # 注册数据集
    for d in ["train", "val"]:
        dataset_name = f"balloon_{d}"
        data_path = os.path.join("data", "balloon", d)
        if dataset_name in DatasetCatalog:
            DatasetCatalog.remove(dataset_name)
        DatasetCatalog.register(dataset_name, lambda d=d: get_balloon_dicts(f"data/balloon/{d}"))
        MetadataCatalog.get(dataset_name).set(thing_classes=["balloon"])


    # 配置训练参数
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # 数据集设置
    cfg.DATASETS.TRAIN = ("voc_subset_train",)
    cfg.DATASETS.TEST = ("voc_subset_val",)

    # 数据加载设置
    cfg.DATALOADER.NUM_WORKERS = 2

    # 模型初始化
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # 按照原论文设置超参数
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 120000
    cfg.SOLVER.STEPS = (90000,)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001

    # ROI头部设置
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(selected_classes)

    # 评估设置
    cfg.TEST.EVAL_PERIOD = 5000

    # 输出设置
    cfg.OUTPUT_DIR = output_dir

    # 训练
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()