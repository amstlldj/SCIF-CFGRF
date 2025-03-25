import torch
from torch.utils.data import Dataset, DataLoader

class ListDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        初始化CWRU数据集，加载文件中的列表。
        :param file_path: 文件路径，文件中保存了一个包含多个张量的列表
        :param transform: 可选的预处理方法
        """
        self.data_list = torch.load(file_path)  # 加载保存的列表
        self.transform = transform

    def __len__(self):
        """返回列表的长度，表示数据集中的样本数量"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        返回指定索引的数据，并应用预处理
        :param idx: 数据的索引
        :return: 处理后的数据（三元组：图像、噪声、标签）
        """
        image, noise, label = self.data_list[idx]  # 获取对应的图像、噪声和标签
        if self.transform:
           image = self.transform(image)  # 对图像应用预处理
           noise = self.transform(noise)  # 对噪声应用预处理
        return image, noise, label  # 返回三元组

def create_list_dataset(data_path, **kwargs):
    """
    创建列表的DataLoader
    :param data_path: 存储数据列表的文件路径
    :param batch_size: 批次大小，默认为1，代表只读入列表中的第一个元素，元素形状为（batch_size,3,32,32）
    :param kwargs: 其他可选参数（如transform, shuffle等）
    :return: 数据加载器
    """
    transform = kwargs.get("transform", None)
    batch_size = kwargs.get("batch_size", 1)

    # 创建自定义数据集
    dataset = ListDataset(file_path=data_path, transform=transform)

    # 打印数据集的大小
    print(f"Dataset size: {len(dataset)}")

    # 配置 DataLoader
    loader_params = dict(
        shuffle=kwargs.get("shuffle", False),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", False),
        num_workers=kwargs.get("num_workers", 4),
    )

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)

    return dataloader


