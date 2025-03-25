import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import random
import time
from glob import glob
import shutil
import os
import sys
import itertools

# 计算余弦相似度
def cosine_similarity(a, b):
    return F.cosine_similarity(a, b)

# 图像预处理并缩放到32x32
def preprocess_and_resize(image_path, target_size=(32, 32)):
    # 打开图像并转换为RGB
    image = Image.open(image_path).convert('RGB')
    
    # 调整图像大小为32x32
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # 定义转换操作：转换为Tensor并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),          # 转为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    image = transform(image).unsqueeze(0)  # 添加batch维度
    return image

# 提取图像特征（这里我们假设使用简单的像素值作为特征）
def extract_features(image_path, target_size=(32, 32)):
    image = preprocess_and_resize(image_path, target_size)
    return image.view(1, -1)  # 将图像展平为一维向量

def filter_images_by_cosine_similarity(original_folder, generated_folder, batch_id, num_classes, threshold=0.8):
    """
    仅对当前 batch 生成的图片进行余弦相似度比较，并删除低于阈值的图片。
    适用于 original_images 数量小于 generated_images 的情况，动态判断是否循环匹配。

    参数:
    original_folder (str): 真实图像所在的根文件夹路径（包含 0-9 目录）。
    generated_folder (str): 生成图像所在的根文件夹路径（包含 0-9 目录）。
    batch_id (int): 当前 batch 的编号。
    threshold (float): 余弦相似度的阈值，低于该值的图片将被删除。
    """
    count = 0

    # 遍历 0-9/0-4 类的文件夹
    for class_label in range(num_classes):
        original_class_folder = os.path.join(original_folder, str(class_label))
        generated_class_folder = os.path.join(generated_folder, str(class_label))

        # 检查原始文件夹和生成文件夹是否存在
        if not os.path.exists(original_class_folder) or not os.path.exists(generated_class_folder):
            print(f"文件夹不存在，跳过类 {class_label}：{original_class_folder} 或 {generated_class_folder}")
            continue  # 跳过该类

        # 获取原始图片列表
        original_images = sorted([f for f in os.listdir(original_class_folder) if f.endswith(('.png', '.jpg'))])
        
        # 获取当前 batch 生成的图片列表
        generated_images = sorted([f for f in os.listdir(generated_class_folder) 
                                   if f.startswith(f"batch_{batch_id}_") and f.endswith(('.png', '.jpg'))])

        if not original_images or not generated_images:
            continue  # 跳过空文件夹

        # 当原始图片数量少于生成图片时，循环使用原始图片
        if len(original_images) < len(generated_images):
            original_iter = itertools.cycle(original_images)
        else:
            original_iter = iter(original_images)

        for gen_image_name in generated_images:
            orig_image_name = next(original_iter)  # 获取匹配的原始图片
            
            # 读取图片路径
            orig_image_path = os.path.join(original_class_folder, orig_image_name)
            gen_image_path = os.path.join(generated_class_folder, gen_image_name)

            # 提取特征
            orig_features = extract_features(orig_image_path, target_size=(32, 32))
            gen_features = extract_features(gen_image_path, target_size=(32, 32))

            # 计算余弦相似度
            cos_sim = cosine_similarity(orig_features, gen_features)

            # 删除低于阈值的图片
            if cos_sim < threshold:
                count += 1
                print(f"删除图像: {gen_image_name}，余弦相似度: {cos_sim.item():.4f}")
                os.remove(gen_image_path)

    print(f'共删除 {count} 张生成图片。')

def check_image_counts(root_folder, x, num_classes):
    """检查 root_folder 下 0-9/0-4 类文件夹的图像数量，若有类别图像数均 > x，则返回 True"""
    check = 0
    for j in range(num_classes):  # 遍历 0-9/0-4 类文件夹
        folder_path = os.path.join(root_folder, str(j))
        # 确保文件夹存在
        if not os.path.exists(folder_path):
            print(f"Warning: 文件夹 {folder_path} 不存在！")
            continue
        
        # 统计所有图片文件，确保获取所有扩展名
        image_files = glob(os.path.join(folder_path, "*"))
        image_count = len(image_files)  # 统计所有文件
        print(f"类别 {j} 图像数: {image_count}")
        
        if image_count > x:
            print(f"类别 {j} 的图像数满足条件 (> {x})")
            check = check + 1
        if check == num_classes:
            return True  # 如果所有类别图像数大于x，返回True

def trim_images_to_x_per_class(root_folder, x, num_classes):
    """对每个类文件夹进行检查，若图像数量大于 x，则随机删除多余图片"""
    for jj in range(num_classes):  # 遍历 0-9/0-4 类文件夹
        folder_path = os.path.join(root_folder, str(jj))
        images = glob(os.path.join(folder_path, "*.*"))  # 获取所有图片文件路径
        image_count = len(images)

        if image_count > x:
            # 计算需要删除的图片数量
            num_to_delete = image_count - x
            images_to_delete = random.sample(images, num_to_delete)  # 随机选择要删除的图片
            
            for image_path in images_to_delete:
                os.remove(image_path)  # 删除图片文件
            print(f"已从类 {jj} 删除 {num_to_delete} 张多余的图片，当前数量为 {x} 张")

