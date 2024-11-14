from detectron2.data import DatasetCatalog


def balloon():
    from src.balloon import train, inference
    # train.main()
    inference.main()


def voc():
    from src.voc.prepare_data import prepare_voc_subset, get_dataset_info
    from src.voc import train, inference
    # download and select subset of voc
    # prepare_voc_subset()

    # statistics of voc dataset
    # get_dataset_info()

    train.main()
    inference.main()


def cityscapes():
    from src.cityscapes.prepare_data import prepare_cityscapes_subset, get_dataset_info
    from src.cityscapes import train
    from src.cityscapes.city_2_coco_script import format_convert

    # 把 val 分成 subset的train, val, test
    # prepare_cityscapes_subset()(one time)

    # 统计
    # get_dataset_info()

    # city to coco format(one time)
    # format_convert()

    # train
    train.main()


if __name__ == '__main__':
    balloon()
    # voc()
    # cityscapes()
