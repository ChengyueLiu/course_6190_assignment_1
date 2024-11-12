import os
import json
import numpy as np
from cityscapesscripts.helpers.labels import labels
from cityscapesscripts.preparation.json2instanceImg import json2instanceImg
from PIL import Image
import glob


def convert_cityscapes_to_coco(cityscapes_root, split):
    """
    Convert Cityscapes annotations to COCO format
    """
    # 确保输出目录存在
    os.makedirs(os.path.join(cityscapes_root, "annotations"), exist_ok=True)

    # 准备COCO格式数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 添加类别信息
    category_dict = {}
    cat_id = 1
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            category_dict[label.name] = cat_id
            coco_data["categories"].append({
                "id": cat_id,
                "name": label.name,
                "supercategory": "object"
            })
            cat_id += 1

    # 处理图像和标注
    img_id = 1
    ann_id = 1

    image_dir = os.path.join(cityscapes_root, f"leftImg8bit/{split}")
    annotation_dir = os.path.join(cityscapes_root, f"gtFine/{split}")

    for city in os.listdir(image_dir):
        city_img_dir = os.path.join(image_dir, city)
        city_ann_dir = os.path.join(annotation_dir, city)

        for img_file in glob.glob(os.path.join(city_img_dir, "*_leftImg8bit.png")):
            # 处理图像
            img = Image.open(img_file)
            width, height = img.size

            image_info = {
                "id": img_id,
                "file_name": os.path.relpath(img_file, image_dir),
                "height": height,
                "width": width
            }
            coco_data["images"].append(image_info)

            # 处理对应的标注文件
            base_name = os.path.basename(img_file).replace("_leftImg8bit.png", "")
            json_file = os.path.join(city_ann_dir, f"{base_name}_gtFine_polygons.json")

            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    cityscape_ann = json.load(f)

                for obj in cityscape_ann["objects"]:
                    if obj["label"] in category_dict:
                        # 转换多边形坐标
                        polygon = obj["polygon"]
                        x_coords = [p[0] for p in polygon]
                        y_coords = [p[1] for p in polygon]

                        # 计算边界框
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        width = x_max - x_min
                        height = y_max - y_min

                        annotation = {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": category_dict[obj["label"]],
                            "segmentation": [sum([[x, y] for x, y in zip(x_coords, y_coords)], [])],
                            "area": width * height,
                            "bbox": [x_min, y_min, width, height],
                            "iscrowd": 0
                        }
                        coco_data["annotations"].append(annotation)
                        ann_id += 1

            img_id += 1

    # 保存COCO格式的标注文件
    output_file = os.path.join(cityscapes_root, f"annotations/{split}.json")
    with open(output_file, "w") as f:
        json.dump(coco_data, f)


def format_convert(cityscapes_root = "./data/cityscapes_subset"):
    for split in ["train", "val"]:
        convert_cityscapes_to_coco(cityscapes_root, split)