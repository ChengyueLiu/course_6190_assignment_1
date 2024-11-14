import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import os
import cv2
import json
import torch
import random
import matplotlib.pyplot as plt
from datetime import datetime
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, HookBase
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


class LossTracker(HookBase):
    def __init__(self):
        super().__init__()
        self.losses = []

    def after_step(self):
        # Record the loss after each iteration
        latest_loss = self.trainer.storage.latest()
        if "total_loss" in latest_loss:
            self.losses.append(latest_loss["total_loss"])


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


class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_tracker = LossTracker()
        self.register_hooks([self.loss_tracker])

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def analyze_and_save_predictions(predictor, dataset_dicts, dataset_name, output_dir):
    """Analyze predictions and save good/bad cases."""
    metadata = MetadataCatalog.get(dataset_name)

    # Create directories for visualization
    viz_dir = os.path.join(output_dir, "visualizations")
    good_cases_dir = os.path.join(viz_dir, "good_cases")
    bad_cases_dir = os.path.join(viz_dir, "bad_cases")
    os.makedirs(good_cases_dir, exist_ok=True)
    os.makedirs(bad_cases_dir, exist_ok=True)

    results = []
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        # Calculate prediction quality score
        if len(instances) > 0:
            scores = instances.scores.numpy()
            quality_score = np.mean(scores)
        else:
            quality_score = 0.0

        result = {
            "file_name": d["file_name"],
            "quality_score": quality_score,
            "instances": instances
        }
        results.append(result)

    # Sort results by quality score
    results.sort(key=lambda x: x["quality_score"], reverse=True)

    # Save top 5 good cases and bottom 5 bad cases
    for i, result in enumerate(results[:5]):  # Good cases
        im = cv2.imread(result["file_name"])
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
        out = v.draw_instance_predictions(result["instances"])
        output_path = os.path.join(good_cases_dir, f"good_case_{i + 1}.jpg")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    for i, result in enumerate(results[-5:]):  # Bad cases
        im = cv2.imread(result["file_name"])
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
        out = v.draw_instance_predictions(result["instances"])
        output_path = os.path.join(bad_cases_dir, f"bad_case_{i + 1}.jpg")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    return viz_dir


def save_training_curves(loss_tracker, output_dir):
    """Save training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_tracker.losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')
    plt.grid(True)

    curves_dir = os.path.join(output_dir, "training_curves")
    os.makedirs(curves_dir, exist_ok=True)
    plt.savefig(os.path.join(curves_dir, "loss_curve.png"))
    plt.close()

    # Save raw loss data
    with open(os.path.join(curves_dir, "loss_data.json"), "w") as f:
        json.dump(loss_tracker.losses, f)


def save_hyperparameters(cfg, output_dir):
    """Save hyperparameter settings in a readable format."""
    hyper_params = {
        "learning_rate": cfg.SOLVER.BASE_LR,
        "max_iterations": cfg.SOLVER.MAX_ITER,
        "batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "roi_batch_size": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        "learning_rate_decay_steps": cfg.SOLVER.STEPS,
        "learning_rate_decay_gamma": cfg.SOLVER.GAMMA,
        "rpn_batch_size": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
        "rpn_positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
        "roi_positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
        "nms_threshold": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
    }

    params_dir = os.path.join(output_dir, "parameters")
    os.makedirs(params_dir, exist_ok=True)

    with open(os.path.join(params_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyper_params, f, indent=4)


def evaluate_model(cfg, dataset_name):
    """Evaluate model and save detailed metrics."""
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "metrics"))
    val_loader = build_detection_test_loader(cfg, dataset_name)
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Save detailed metrics
    metrics_dir = os.path.join(cfg.OUTPUT_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file = os.path.join(metrics_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save evaluation summary
    summary = {
        "segmentation": {
            "mAP": metrics["segm"]["AP"],
            "AP50": metrics["segm"]["AP50"],
            "AP75": metrics["segm"]["AP75"],
            "APs": metrics["segm"]["APs"],
            "APm": metrics["segm"]["APm"],
            "APl": metrics["segm"]["APl"],
        },
        "detection": {
            "mAP": metrics["bbox"]["AP"],
            "AP50": metrics["bbox"]["AP50"],
            "AP75": metrics["bbox"]["AP75"],
            "APs": metrics["bbox"]["APs"],
            "APm": metrics["bbox"]["APm"],
            "APl": metrics["bbox"]["APl"],
        }
    }

    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return metrics, predictor


# 在main函数中修改输出目录的设置部分：

def main():
    # Create output directories with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"./output/balloon_{timestamp}"

    # Create separate directories for detectron2 and custom outputs
    detectron_output = os.path.join(base_dir, "detectron2_output")
    custom_output = os.path.join(base_dir, "analysis_output")

    os.makedirs(detectron_output, exist_ok=True)
    os.makedirs(custom_output, exist_ok=True)

    # ... (其他代码保持不变) ...

    # 配置detectron2输出目录
    cfg.OUTPUT_DIR = detectron_output

    # 保存自定义输出到analysis_output目录
    save_hyperparameters(cfg, custom_output)

    # Training
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save training curves to custom output
    save_training_curves(trainer.loss_tracker, custom_output)

    # Evaluation and visualization
    print("Evaluating model and generating visualizations...")
    metrics, predictor = evaluate_model(cfg, "balloon_val", custom_output)

    # Analyze and save predictions
    dataset_dicts = DatasetCatalog.get("balloon_val")
    viz_dir = analyze_and_save_predictions(predictor, dataset_dicts, "balloon_val", custom_output)

    print("\nTraining and evaluation completed. Results saved to:")
    print(f"\n{base_dir}/")
    print("├── detectron2_output/")
    print("│   ├── model checkpoints")
    print("│   ├── tensorboard logs")
    print("│   └── default inference results")
    print("└── analysis_output/")
    print("    ├── parameters/")
    print("    ├── training_curves/")
    print("    ├── metrics/")
    print("    └── visualizations/")


# 修改evaluate_model函数的参数：
def evaluate_model(cfg, dataset_name, custom_output_dir):
    """Evaluate model and save detailed metrics."""
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
    val_loader = build_detection_test_loader(cfg, dataset_name)
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Save detailed metrics to custom output directory
    metrics_dir = os.path.join(custom_output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file = os.path.join(metrics_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save evaluation summary
    summary = {
        "segmentation": {
            "mAP": metrics["segm"]["AP"],
            "AP50": metrics["segm"]["AP50"],
            "AP75": metrics["segm"]["AP75"],
            "APs": metrics["segm"]["APs"],
            "APm": metrics["segm"]["APm"],
            "APl": metrics["segm"]["APl"],
        },
        "detection": {
            "mAP": metrics["bbox"]["AP"],
            "AP50": metrics["bbox"]["AP50"],
            "AP75": metrics["bbox"]["AP75"],
            "APs": metrics["bbox"]["APs"],
            "APm": metrics["bbox"]["APm"],
            "APl": metrics["bbox"]["APl"],
        }
    }

    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    return metrics, predictor

if __name__ == "__main__":
    main()