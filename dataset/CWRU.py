import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_cwru_dataset(data_path, batch_size, **kwargs):
    # 确保数据路径指向包含类别文件夹的文件夹
    train = kwargs.get("train", True)
    download = kwargs.get("download", False)

    # 定义数据预处理步骤
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 缩放为32x32尺寸
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 使用 ImageFolder 来读取数据，自动标记类别
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # 打印数据集的大小和类别
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    # 配置 DataLoader
    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4),
    )

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)

    return dataloader

