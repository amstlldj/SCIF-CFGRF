from dataset.MNIST import create_mnist_dataset
from dataset.CIFAR import create_cifar10_dataset
from dataset.Custom import create_custom_dataset
from dataset.CWRU import create_cwru_dataset
from dataset.SEU import create_seu_dataset
from dataset.List import create_list_dataset

def create_dataset(dataset: str, **kwargs):
    if dataset == "mnist":
        return create_mnist_dataset(**kwargs)
    elif dataset == "cifar":
        return create_cifar10_dataset(**kwargs)
    elif dataset == "custom":
        return create_custom_dataset(**kwargs)
    elif dataset == "cwru":
        return create_cwru_dataset(**kwargs)
    elif dataset == "seu":
        return create_seu_dataset(**kwargs)
    elif dataset == "list":
        return create_list_dataset(**kwargs)
    else:
        raise ValueError(f"dataset except one of {'mnist', 'cifar', 'custom', 'cwru', 'seu', 'list'}, but got {dataset}")
