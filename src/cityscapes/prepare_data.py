import os
import random
import shutil
from collections import defaultdict

from tqdm import tqdm


def prepare_cityscapes_subset(
        base_dir='./data',
        split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
        subset_name='cityscapes_subset'
):
    """
    将Cityscapes验证集数据分割成新的训练/验证/测试子集

    Args:
        base_dir (str): 原始数据集根目录，包含gtFine和leftImg8bit文件夹
        split_ratios (dict): 新子集的分割比例
        subset_name (str): 新子集的目录名
    Returns:
        str: 子集目录路径
    """
    try:
        # 检查原始数据集目录
        gt_val_dir = os.path.join(base_dir, 'gtFine', 'val')
        img_val_dir = os.path.join(base_dir, 'leftImg8bit', 'val')
        if not all(os.path.exists(d) for d in [gt_val_dir, img_val_dir]):
            raise ValueError(f"Required directories not found in {base_dir}")

        # 创建子集目录
        subset_dir = os.path.join(base_dir, subset_name)
        for split in split_ratios.keys():
            for subdir in ['gtFine', 'leftImg8bit']:
                os.makedirs(os.path.join(subset_dir, subdir, split), exist_ok=True)

        # 收集所有图片信息
        print("Scanning validation set...")
        image_list = []
        for city in os.listdir(gt_val_dir):
            city_dir = os.path.join(gt_val_dir, city)
            if not os.path.isdir(city_dir):
                continue

            for filename in os.listdir(city_dir):
                if filename.endswith('_gtFine_color.png'):
                    base_name = filename.replace('_gtFine_color.png', '')
                    image_info = {
                        'city': city,
                        'base_name': base_name,
                        'gtFine_files': {
                            'color': f'{base_name}_gtFine_color.png',
                            'instanceIds': f'{base_name}_gtFine_instanceIds.png',
                            'labelIds': f'{base_name}_gtFine_labelIds.png',
                            'polygons': f'{base_name}_gtFine_polygons.json'
                        },
                        'leftImg8bit_file': f'{base_name}_leftImg8bit.png'
                    }

                    # 验证所有文件是否存在
                    all_files_exist = True
                    for gt_file in image_info['gtFine_files'].values():
                        if not os.path.exists(os.path.join(gt_val_dir, city, gt_file)):
                            all_files_exist = False
                            print(f"Warning: Missing gtFine file: {gt_file}")
                            break

                    if not os.path.exists(os.path.join(img_val_dir, city, image_info['leftImg8bit_file'])):
                        all_files_exist = False
                        print(f"Warning: Missing image file: {image_info['leftImg8bit_file']}")

                    if all_files_exist:
                        image_list.append(image_info)

        if not image_list:
            raise ValueError("No valid images found in validation set")

        # 随机打乱并分割数据
        random.shuffle(image_list)
        total_images = len(image_list)
        train_size = int(total_images * split_ratios['train'])
        val_size = int(total_images * split_ratios['val'])

        split_data = {
            'train': image_list[:train_size],
            'val': image_list[train_size:train_size + val_size],
            'test': image_list[train_size + val_size:]
        }

        # 复制文件到新目录
        print("\nCopying files to new directory structure...")
        for split_name, images in split_data.items():
            print(f"\nProcessing {split_name} set ({len(images)} images)...")
            for image_info in tqdm(images):
                city = image_info['city']

                # 创建城市目录
                for subdir in ['gtFine', 'leftImg8bit']:
                    os.makedirs(os.path.join(subset_dir, subdir, split_name, city), exist_ok=True)

                # 复制gtFine文件
                for gt_file in image_info['gtFine_files'].values():
                    src = os.path.join(gt_val_dir, city, gt_file)
                    dst = os.path.join(subset_dir, 'gtFine', split_name, city, gt_file)
                    shutil.copy2(src, dst)

                # 复制图像文件
                src = os.path.join(img_val_dir, city, image_info['leftImg8bit_file'])
                dst = os.path.join(subset_dir, 'leftImg8bit', split_name, city, image_info['leftImg8bit_file'])
                shutil.copy2(src, dst)

        print("\nSubset creation completed successfully!")
        for split_name, images in split_data.items():
            print(f"{split_name.capitalize()} set: {len(images)} images")

        return subset_dir

    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")
        raise

import numpy as np
from PIL import Image
from tqdm import tqdm
def get_dataset_info(subset_dir='./data/cityscapes_subset', selected_classes=None):
    """
    统计数据集信息，包括每个分割集的图像数量和类别分布

    Args:
        subset_dir (str): 子集目录路径
        selected_classes (list, optional): 需要统计的特定类别列表，默认统计所有类别
    Returns:
        dict: 数据集统计信息
    """
    # Cityscapes类别ID映射
    CLASS_ID_MAP = {
        24: 'person',
        25: 'rider',
        26: 'car',
        27: 'truck',
        28: 'bus',
        31: 'train',
        32: 'motorcycle',
        33: 'bicycle'
    }

    if selected_classes is None:
        selected_classes = list(CLASS_ID_MAP.values())

    stats = {}
    print("\nCollecting dataset statistics...")

    for split in ['train', 'val', 'test']:
        stats[split] = {
            'total_images': 0,
            'per_class': defaultdict(int),
            'images_per_class': defaultdict(int)
        }

        split_gt_dir = os.path.join(subset_dir, 'gtFine', split)
        if not os.path.exists(split_gt_dir):
            print(f"Warning: Directory not found: {split_gt_dir}")
            continue

        print(f"\nAnalyzing {split} set...")
        for city in tqdm(os.listdir(split_gt_dir)):
            city_dir = os.path.join(split_gt_dir, city)
            if not os.path.isdir(city_dir):
                continue

            for filename in os.listdir(city_dir):
                if not filename.endswith('_gtFine_instanceIds.png'):
                    continue

                stats[split]['total_images'] += 1

                try:
                    # 读取实例标注
                    instance_path = os.path.join(city_dir, filename)
                    instance_img = np.array(Image.open(instance_path))

                    # 统计每个类别的实例
                    classes_in_image = set()
                    unique_instances = np.unique(instance_img)

                    for instance_id in unique_instances:
                        if instance_id == 0:  # 跳过背景
                            continue

                        class_id = instance_id // 1000
                        if class_id in CLASS_ID_MAP:
                            class_name = CLASS_ID_MAP[class_id]
                            if class_name in selected_classes:
                                stats[split]['per_class'][class_name] += 1
                                classes_in_image.add(class_name)

                    # 更新类别出现的图像数
                    for class_name in classes_in_image:
                        stats[split]['images_per_class'][class_name] += 1

                except Exception as e:
                    print(f"Error processing {instance_path}: {str(e)}")
                    continue

    # 打印统计信息
    print("\nDataset Statistics Summary:")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        split_stats = stats[split]
        total_instances = sum(split_stats['per_class'].values())

        print(f"\n{split.upper()} Set:")
        print(f"Total images: {split_stats['total_images']}")

        if total_instances > 0:
            print("\nInstances per class:")
            for cls in sorted(selected_classes):
                instances = split_stats['per_class'][cls]
                images = split_stats['images_per_class'][cls]
                percentage = (instances / total_instances) * 100
                print(f"  {cls:10s}: {instances:5d} instances in {images:3d} images ({percentage:5.1f}%)")

    return stats

if __name__ == "__main__":
    # 示例使用
    subset_dir = prepare_cityscapes_subset(
        base_dir='./data',
        split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15},
        subset_name='cityscapes_subset'
    )