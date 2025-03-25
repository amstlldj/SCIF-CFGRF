from typing import Optional, Union
import torch
from tqdm import tqdm
import torchvision.utils
from torchvision.utils import make_grid
from PIL import Image
from pathlib2 import Path
import yaml
import os
import numpy as np
from argparse import ArgumentParser
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from sklearn.manifold import TSNE
import csv

def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg

def train_one_epoch(trainer, loader, optimizer, device, epoch, total_epoch):
    trainer.train()
    total_loss, total_num = 0., 0

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        for _, (images, labels) in enumerate(data):
            optimizer.zero_grad()

            x_0 = images.to(device)
            labels = labels.to(device)

            loss, x_noise = trainer(x_0, labels)
                
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += x_0.shape[0]

            data.set_description(f"Epoch: {epoch}/{total_epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / total_num,
            })

    return total_loss / total_num


def save_image(images: torch.Tensor, labels: Optional[torch.Tensor] = None, 
               show: bool = True, path: Optional[str] = None, format: Optional[str] = None, 
               to_grayscale: bool = False, batch_idx: int = 0, **kwargs): 
    """
    Save each image to the corresponding folder based on its label.
    
    Parameters:
        images: A tensor with shape (batch_size, channels, height, width).
        labels: A tensor of labels with shape (batch_size,) or None. Default is None.
        show: Whether to display the image after saving. Default `True`.
        path: Path to save the image. If None (default), will not save the image.
        format: Image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: Whether to convert PIL image to grayscale. Default `False`.
        batch_idx: The index of the current batch being saved (used for file naming).
        **kwargs: Other arguments for `torchvision.utils.make_grid`.

    Returns:
        None. Each image is saved in the corresponding label folder.
    """
    # Denormalize images to [0, 1] range
    images = images * 0.5 + 0.5  # Assuming input range [-1, 1], adjust if needed

    # Convert to uint8
    images = (images * 255).clamp(0, 255).to(torch.uint8)

    # Ensure labels is a 1D tensor (if it's a 2D tensor, convert to class indices)
    if labels is not None and labels.ndimension() > 1:
        labels = labels.argmax(dim=1)  # If labels are one-hot encoded, use argmax to get class indices
    
    # Save images to label-specific folders
    for i, image in enumerate(images):
        label = labels[i].item() if labels is not None else None
        
        # Convert image to PIL
        im = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())  # Convert tensor to numpy and then to PIL
        
        # Optionally convert to grayscale
        if to_grayscale:
            im = im.convert("L")
        
        # Define save path based on label
        label_folder = os.path.join(path, str(label)) if path else str(label)
        os.makedirs(label_folder, exist_ok=True)

        # Save image with batch_idx and image_idx to avoid overwriting
        image_path = os.path.join(label_folder, f"batch_{batch_idx}_image_{i}.{format if format else 'png'}")
        im.save(image_path, format=format)

        # Optionally display the image
        if show:
            im.show()

def save_sample_image(images: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                      show: bool = True, path: Optional[str] = None, format: Optional[str] = None, 
                      to_grayscale: bool = False, batch_idx: int = 0, save_as_gif: bool = False, 
                      gif_duration: float = 1, max_steps: int = 51, **kwargs):
    images = (images * 0.5 + 0.5).clamp(0, 1) * 255
    images = images.to(torch.uint8)

    print(f"Batch {batch_idx}: images shape = {images.shape}, received labels = {labels}, labels shape = {labels.shape if labels is not None else None}")
    
    # 处理 labels，确保与 images 的 batch_size 匹配
    if labels is not None:
        if labels.ndimension() == 2 and labels.shape[0] == images.shape[0]:
            # labels 是 (64, 4)，取第一列作为标签
            labels_processed = labels[:, 0]  # 形状变为 (64,)
        elif labels.ndimension() == 1 and labels.shape[0] == images.shape[0]:
            labels_processed = labels  # 已经是 (64,)
        else:
            raise ValueError(f"Labels shape {labels.shape} does not match images batch size {images.shape[0]}")
    else:
        labels_processed = [None] * images.shape[0]

    # 逐样本处理
    for i, (image_set, label) in enumerate(zip(images, labels_processed)):
        label_value = label.item() if label is not None else None
        label_folder = os.path.join(path, str(label_value)) if path else str(label_value)
        os.makedirs(label_folder, exist_ok=True)

        image_list = []
        step_size = max(1, len(image_set) // (max_steps - 1))  # 动态步长，限制帧数
        for j in range(0, len(image_set), step_size):
            image = image_set[j]
            im = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
            if to_grayscale:
                im = im.convert("L")
            image_list.append(im)

        final_image = image_set[-1].permute(1, 2, 0).cpu().numpy()
        final_im = Image.fromarray(final_image)
        if to_grayscale:
            final_im = final_im.convert("L")
        if len(image_list) < max_steps:
            image_list.append(final_im)

        if save_as_gif:
            gif_path = os.path.join(label_folder, f"batch_{batch_idx}_image_{i}_diffusion_process.gif")
            image_list[0].save(gif_path, save_all=True, append_images=image_list[1:], 
                              duration=int(gif_duration * 1000), loop=0)
            print(f"GIF saved at {gif_path}")
        else:
            result_image = Image.new('RGB' if not to_grayscale else 'L', 
                                   (image_list[0].width * len(image_list), image_list[0].height))
            for idx, im in enumerate(image_list):
                result_image.paste(im, (idx * im.width, 0))
            image_path = os.path.join(label_folder, f"batch_{batch_idx}_image_{i}_diffusion_result.{format if format else 'png'}")
            result_image.save(image_path, format=format)
            print(f"Image saved at {image_path}")
            if show:
                result_image.show()

def generator_parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim", "rf"])
    parser.add_argument('--model', type=str, default="unet", choices=['unet', 'dm', 'ws', 'wc'], help='Select trainer type')

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=16)

    # sampler param
    parser.add_argument("--result_only", default=True, action="store_true")
    parser.add_argument("--interval", type=int, default=50, help="Save every interval steps, interval=1 means each step is saved.")

    # DDIM sampler param
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

    # save image param
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("--gif", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)
    parser.add_argument("-if", "--original_image_folder", type=str, default=None)
    parser.add_argument("-mp", "--match_save_path", type=str, default=None)
    parser.add_argument("--to_grayscale", default=False, action="store_true")
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--target_class_count", type=int, default=64)
    parser.add_argument("--cosine_threshold", type=float, default=0.8)
    parser.add_argument("--gif_speed", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=101)

    args = parser.parse_args()
    return args

def train_parse_option():
    parser = ArgumentParser()
    parser.add_argument('--trainer', type=str, default="rf", choices=['gaussian', 'rf'], help='Select trainer type')
    parser.add_argument('--model', type=str, default="unet", choices=['unet', 'dm', 'ws', 'wc'], help='Select trainer type')
    parser.add_argument("--scheduler", type=str, default="ReduceLR", choices=["StepLR", "ReduceLR"])
    parser.add_argument("--earlystopping", default=True, action="store_true")

    args = parser.parse_args()
    return args

def generate_batches(num_batches, batch_size, num_classes):
    """
    生成一个包含多个批次的列表，批次内容重复。
    每10/5个元素为一组，每个元素是一个包含10/5个张量的列表，张量内容为0到9/0到4。
    
    Args:
        num_batches (int): 批次数量，必须是10/5的倍数。
        batch_size (int): 每次生成数量，由训练好的模型文件的数据加载batch_size决定，传入cp["config"]["Dataset"]["batch_size"]即可。
    Returns:
        list: 包含num_batches个批次的列表，每个批次是一个包含10/5个张量的列表。
    """
    if num_batches % num_classes != 0:
        raise ValueError(f"num_batches must be a multiple of {num_classes}!")
    
    batches = []
    # 每10/5个元素重复同样的批次
    for i in range(num_batches // num_classes):
        # 每个批次包含10/5个张量
        batch = [torch.full((batch_size,), i % num_classes) for i in range(num_classes)]  # 0到9/0到4的张量
        batches.extend(batch)  # 将批次添加到总列表中
    
    return batches


