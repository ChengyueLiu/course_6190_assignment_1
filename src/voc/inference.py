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
# 导入数据加载函数
from src.voc.train import get_voc_dicts

def setup_cfg(model_dir, num_classes=8):
    """设置推理配置"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    return cfg

def main():
    # 设置目录
    model_dir = "./output/voc/models"
    output_dir = "./output/voc/inference"
    os.makedirs(output_dir, exist_ok=True)

    # 配置推理参数
    cfg = setup_cfg(model_dir)
    predictor = DefaultPredictor(cfg)

    # 直接获取验证集数据，类似balloon的方式
    val_data_path = "./data/VOC2012_subset"  # VOC数据集路径
    dataset_dicts = get_voc_dicts(val_data_path, "val")  # 直接调用函数获取数据
    metadata = MetadataCatalog.get("voc_subset_val")  # 获取元数据

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
                      metadata=metadata,
                      scale=0.8,
                      instance_mode=ColorMode.IMAGE_BW
                      )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # 保存结果
        output_path = os.path.join(output_dir, f"prediction_{os.path.basename(d['file_name'])}")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

        # 打印检测结果
        instances = outputs["instances"].to("cpu")
        for box, cls, score in zip(instances.pred_boxes, instances.pred_classes, instances.scores):
            class_name = metadata.thing_classes[cls]
            print(f"Found {class_name} with score: {score:.3f}")

        print(f"Saved prediction to {output_path}")

if __name__ == "__main__":
    main()