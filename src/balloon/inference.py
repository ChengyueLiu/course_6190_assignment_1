# src/inference.py
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import os
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from src.balloon.train import get_balloon_dicts  # 导入数据加载函数


def main():
    model_dir = "./output/balloon/models"  # 模型保存目录
    # 确保输出目录存在
    output_dir = "./output/balloon/inference"
    os.makedirs(output_dir, exist_ok=True)

    # 配置推理参数
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")  # 使用训练好的模型
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    # 在验证集上进行推理
    val_data_path = "data/balloon/val"
    dataset_dicts = get_balloon_dicts(val_data_path)
    balloon_metadata = MetadataCatalog.get("balloon_train")

    # 随机选择3张图片进行推理
    for d in random.sample(dataset_dicts, min(3, len(dataset_dicts))):
        # 读取图片
        im = cv2.imread(d["file_name"])
        if im is None:
            print(f"Warning: Could not read image {d['file_name']}")
            continue

        # 进行推理
        outputs = predictor(im)

        # 可视化结果
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # 保存结果
        output_path = os.path.join(output_dir, f"prediction_{os.path.basename(d['file_name'])}")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        print(f"Saved prediction to {output_path}")


if __name__ == "__main__":
    main()