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


if __name__ == '__main__':
    # balloon()
    voc()
