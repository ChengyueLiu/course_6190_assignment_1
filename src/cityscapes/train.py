import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances  # 改用这个

def main():
    # 创建输出目录
    model_dir = "./output/cityscapes_subset/models"
    os.makedirs(model_dir, exist_ok=True)

    # 注册数据集 - 使用 register_coco_instances
    root = "./data/cityscapes_subset"
    register_coco_instances(
        "cityscapes_train",
        {},
        os.path.join(root, "annotations/train.json"),  # 需要COCO格式的标注文件
        os.path.join(root, "leftImg8bit/train")
    )
    register_coco_instances(
        "cityscapes_val",
        {},
        os.path.join(root, "annotations/val.json"),
        os.path.join(root, "leftImg8bit/val")
    )

    # 配置训练参数
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    # 数据集配置
    cfg.DATASETS.TRAIN = ("cityscapes_train",)
    cfg.DATASETS.TEST = ("cityscapes_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # 模型配置
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Cityscapes有8个类别

    # 训练配置
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = [3000, 4000]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.OUTPUT_DIR = model_dir

    # 开始训练
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()