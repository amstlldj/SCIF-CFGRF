from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder  # 如果是基于文件夹结构的数据集
import os

def create_seu_dataset(data_path, batch_size, **kwargs):
    train = kwargs.get("train", True)
    download = kwargs.get("download", False)

    # 定义图像的转换操作
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 缩放为32x32尺寸
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 设置DataLoader的参数
    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4),
    )

    # 创建数据集对象，假设数据是以类名为子文件夹的结构组织
    dataset = ImageFolder(root=data_path, transform=transform)

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)

    return dataloader
