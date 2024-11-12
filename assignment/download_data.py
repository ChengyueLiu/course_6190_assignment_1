import torchvision


def download_voc_dataset():
    dataset = torchvision.datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train',
        download=True
    )