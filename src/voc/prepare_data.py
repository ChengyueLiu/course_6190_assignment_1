# src/data_setup.py

import os
import random
import shutil
import xml.etree.ElementTree as ET
import cv2
import torchvision
from collections import defaultdict

# 全局配置参数
DEFAULT_CONFIG = {
    # 数据集类别
    "SELECTED_CLASSES": [
        'person', 'car', 'cat', 'dog', 'chair',
        'bottle', 'bus', 'bicycle'
    ],

    # 每个类别的图片数量
    "IMAGES_PER_CLASS": 125,

    # 数据集分割比例
    "SPLIT_RATIOS": {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    },

    # 目录配置
    "BASE_DIR": './data',
    "SUBSET_NAME": 'VOC2012_subset',

    # VOC数据集配置
    "VOC_YEAR": '2012',
    "VOC_IMAGE_SET": 'train'
}


def prepare_voc_subset(
        base_dir=DEFAULT_CONFIG["BASE_DIR"],
        selected_classes=DEFAULT_CONFIG["SELECTED_CLASSES"],
        images_per_class=DEFAULT_CONFIG["IMAGES_PER_CLASS"],
        split_ratios=DEFAULT_CONFIG["SPLIT_RATIOS"],
        voc_year=DEFAULT_CONFIG["VOC_YEAR"],
        voc_image_set=DEFAULT_CONFIG["VOC_IMAGE_SET"],
        subset_name=DEFAULT_CONFIG["SUBSET_NAME"]
):
    """
    One-time setup: download VOC and create balanced subset

    Args:
        base_dir (str): Root directory for dataset storage
        selected_classes (list): List of class names to include
        images_per_class (int): Number of images to select per class
        split_ratios (dict): Train/val/test split ratios
        voc_year (str): VOC dataset year
        voc_image_set (str): VOC dataset image set
        subset_name (str): Name of the created subset directory
    Returns:
        str: Path to subset directory
    """

    def download_voc():
        """Download VOC dataset"""
        print("Downloading VOC dataset...")
        dataset = torchvision.datasets.VOCSegmentation(
            root=base_dir,
            year=voc_year,
            image_set=voc_image_set,
            download=True
        )
        return os.path.join(base_dir, 'VOCdevkit', f'VOC{voc_year}')

    def create_balanced_subset(voc_dir, subset_dir):
        """Create balanced dataset subset"""
        print(f"Creating balanced subset with {images_per_class} images per class")

        # Create directory structure
        for split in split_ratios.keys():
            os.makedirs(os.path.join(subset_dir, 'Annotations'), exist_ok=True)
            os.makedirs(os.path.join(subset_dir, 'JPEGImages'), exist_ok=True)
            os.makedirs(os.path.join(subset_dir, 'ImageSets', 'Main'), exist_ok=True)

        # Collect images by class
        class_to_images = defaultdict(list)
        anno_dir = os.path.join(voc_dir, 'Annotations')

        print("Scanning dataset for class distribution...")
        for anno_file in os.listdir(anno_dir):
            if not anno_file.endswith('.xml'):
                continue

            tree = ET.parse(os.path.join(anno_dir, anno_file))
            root = tree.getroot()

            image_classes = set()
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name in selected_classes:
                    image_classes.add(cls_name)
                    class_to_images[cls_name].append(anno_file[:-4])

        # Select balanced subset for each class
        selected_images = set()
        for cls in selected_classes:
            available_images = len(class_to_images[cls])
            if available_images < images_per_class:
                print(f"Warning: Only {available_images} images available for class {cls}")
                selected = class_to_images[cls]
            else:
                selected = random.sample(class_to_images[cls], images_per_class)
            selected_images.update(selected)

        # Convert to list and shuffle
        selected_images = list(selected_images)
        random.shuffle(selected_images)

        # Calculate split sizes
        total_images = len(selected_images)
        train_size = int(total_images * split_ratios['train'])
        val_size = int(total_images * split_ratios['val'])

        # Create splits
        splits = {
            'train': selected_images[:train_size],
            'val': selected_images[train_size:train_size + val_size],
            'test': selected_images[train_size + val_size:]
        }

        # Copy files and create split files
        print("\nCreating subset and copying files...")
        for split_name, image_ids in splits.items():
            split_file = os.path.join(subset_dir, 'ImageSets', 'Main', f'{split_name}.txt')
            with open(split_file, 'w') as f:
                for image_id in image_ids:
                    f.write(f'{image_id}\n')

                    # Copy image and annotation files
                    for ext, src_dir in [('.jpg', 'JPEGImages'), ('.xml', 'Annotations')]:
                        src = os.path.join(voc_dir, src_dir, image_id + ext)
                        dst = os.path.join(subset_dir, src_dir, image_id + ext)
                        shutil.copy2(src, dst)

        return splits

    try:
        # Create main directories
        os.makedirs(base_dir, exist_ok=True)
        subset_dir = os.path.join(base_dir, subset_name)

        # Download dataset if needed
        voc_dir = download_voc()

        # Create and populate subset
        splits = create_balanced_subset(voc_dir, subset_dir)

        print("\nOne-time setup completed successfully!")
        return subset_dir

    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")
        raise


def get_dataset_info(
        subset_dir=os.path.join(DEFAULT_CONFIG["BASE_DIR"], DEFAULT_CONFIG["SUBSET_NAME"]),
        selected_classes=DEFAULT_CONFIG["SELECTED_CLASSES"]
):
    """
    Get and display dataset statistics
    Can be called multiple times as needed

    Args:
        subset_dir (str): Path to subset directory
        selected_classes (list): List of class names to analyze
    Returns:
        dict: Dataset statistics
    """
    # Initialize statistics dictionary
    stats = {}
    for split in ['train', 'val', 'test']:
        stats[split] = {
            'total': 0,
            'per_class': {cls: 0 for cls in selected_classes},
            'images_per_class': {cls: 0 for cls in selected_classes}
        }

        # Read split file
        split_file = os.path.join(subset_dir, 'ImageSets', 'Main', f'{split}.txt')

        try:
            with open(split_file, 'r') as f:
                image_ids = [x.strip() for x in f]

            stats[split]['total'] = len(image_ids)

            # Process each image
            for image_id in image_ids:
                anno_file = os.path.join(subset_dir, 'Annotations', f'{image_id}.xml')
                tree = ET.parse(anno_file)
                root = tree.getroot()

                classes_in_image = set()

                for obj in root.findall('object'):
                    cls = obj.find('name').text
                    if cls in selected_classes:
                        stats[split]['per_class'][cls] += 1
                        classes_in_image.add(cls)

                for cls in classes_in_image:
                    stats[split]['images_per_class'][cls] += 1

        except FileNotFoundError:
            print(f"Warning: Could not find split file for {split}")
            continue
        except Exception as e:
            print(f"Error processing {split} split: {str(e)}")
            continue

    # Print statistics
    print("\nDataset Statistics:")
    print("=" * 50)

    for split, split_stats in stats.items():
        print(f"\n{split.upper()} Set:")
        print(f"Total images: {split_stats['total']}")

        print("\nInstances per class:")
        for cls, count in split_stats['per_class'].items():
            print(f"  {cls:10s}: {count:5d} instances in {split_stats['images_per_class'][cls]:5d} images")

        total_instances = sum(split_stats['per_class'].values())
        if total_instances > 0:
            print("\nClass distribution (% of total instances):")
            for cls, count in split_stats['per_class'].items():
                percentage = (count / total_instances) * 100
                print(f"  {cls:10s}: {percentage:5.1f}%")

    return stats

