from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from joblib import dump,load
import gc
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import gc

def normalize(data):
    """"
    归一化数据
    
    参数:
    data (numpy.ndarray) - 需要归一化的数据
    
    返回:
    numpy.ndarray - 归一化后的数据
    """
    # 计算数据的均值和标准差
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    
    # 归一化数据
    normalized_data = (data - data_mean) / data_std
    
    return normalized_data

def split_data_with_overlap(data, window_size=512, overlap_ratio=0.5):
    """"
    切割时序数据为重叠的窗口
    :param data: 输入的时序数据
    :param window_size: 每个窗口的大小
    :param overlap_ratio: 窗口之间的重叠比例:param overlap_ratio: 窗口之间的重叠比例
    :return: 切割后的数据列表
    :step_size:step_size是在切割数据时用来确定每个切割窗口之间的步长的参数。它的计算方式是根据窗口大小(window_size)和重叠率(overlap_ratio)来确定的。
    例如:
    如果窗口大小是512,重叠率是0.5,那么step_size就是256。这意味着每个窗口向后移动256个数据点,与前一个窗口有一半的重叠。
    如果重叠率是0.75,那么step_size就是128。这意味着每个窗口向后移动128个数据点,与前一个窗口有3/4的重叠。
    """
    step_size = int(window_size * (1 - overlap_ratio)) # step_size是在切割数据时用来确定每个切割窗口之间的步长的参数。它的计算方式是根据窗口大小(window_size)和重叠率(overlap_ratio)来确定的。
    slices = []

    for start in range(0, len(data) - window_size + 1, step_size):
        slices.append(data[start:start + window_size])
    
    return np.array(slices)

def make_datasets(data_file_csv, label_list=[], split_rate=[0.7, 0.2, 0.1],window_size=512,overlap_ratio=0.5):
    """
    从CSV文件生成数据集,并按照指定比例划分
    :param data_file_csv: CSV文件路径
    :param label_list: 标签列表,如果有多个标签
    :param split_rate: 训练集、验证集和测试集的比例
    :return: 训练集、验证集和测试集
    """
    # 读取数据
    data = pd.read_csv(data_file_csv).values
    
    # 假设数据是一维的
    
    # 切割数据
    sliced_data = split_data_with_overlap(data, window_size,overlap_ratio)
    
    # 划分数据集
    total_samples = len(sliced_data)
    train_size = int(total_samples * split_rate[0])
    val_size = int(total_samples * split_rate[1])
    
    train_data = sliced_data[:train_size]
    val_data = sliced_data[train_size:train_size + val_size]
    test_data = sliced_data[train_size + val_size:]
    
    return train_data, val_data, test_data

def make_data(data, train_fraction=0.8):
    """
    随机抽取数据集，将 80% 数据作为训练集，20% 数据作为测试集。
    
    参数:
    data (np.ndarray): 输入的数据集，形状为 (2500, 256)
    train_fraction (float): 用于训练集的比例，默认值为 0.8
    
    返回:
    tuple: 训练集和测试集
    """
    # 确保数据是 numpy 数组
    data = np.array(data)
    
    # 获取数据的总行数
    num_samples = data.shape[0]
    
    # 计算训练集的大小
    num_train_samples = int(num_samples * train_fraction)
    
    # 随机打乱数据索引
    indices = np.random.permutation(num_samples)
    
    # 获取训练集和测试集的索引
    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]
    
    # 划分训练集和测试集
    train_set = data[train_indices]
    test_set = data[test_indices]
    
    return train_set, test_set

"""
# 采用驱动端数据
data_columns = ['X099_DE_time', 'X107_DE_time', 'X120_DE_time', 'X132_DE_time', 'X171_DE_time',

                'X187_DE_time','X199_DE_time','X211_DE_time','X224_DE_time','X236_DE_time']

columns_name = ['de_normal','de_7_inner','de_7_ball','de_7_outer','de_14_inner','de_14_ball','de_14_outer','de_21_inner','de_21_ball','de_21_outer']

file_names = ['99.mat','107.mat','120.mat','132.mat','171.mat','187.mat','199.mat','211.mat','224.mat','236.mat']

data_total = []

for index in range(10):
    
    # 读取各个类别的MAT文件
    
    data = loadmat(f'D:/jupyternotebook/X-1/x_data/{file_names[index]}')

    data_total.append(data[data_columns[index]].reshape(-1))

print(len(data_total))

data_total

data_len = []
for i in range(10):
    data_total[i] = normalize(data_total[i]) # 对各类分别归一化处理
    data_len.append(len(data_total[i]))

np.min(data_len) # 查看最短长度

data_total

for i in range(10):
    data_total[i] = split_data_with_overlap(data_total[i],256,0.81)
    print(data_total[i].shape[0]) # 对各类数据集进行切片形成数据集

for i in range(10):
    indices = np.random.choice(data_total[i].shape[0], 2500, replace=False)
    data_total[i] = data_total[i][indices,:]
    print(data_total[i].shape[0]) # 对各类数据集进行随机下采样

train = []
test = []
for i in range(10):
    make_train,make_test = make_data(data_total[i],0.8)
    train.append(make_train)
    test.append(make_test) # 将每个类分别划分为训练集和测试集存储进入train和test列表
    print(train[i].shape)
    print(test[i].shape)

# 保存数据
dump(train,'train')
dump(test,'test')

# 读入数据
train = load('train')
test = load('test')

for i in range(10):
    print(train[i].shape)
    print(test[i].shape) # 查看载入数据形状

# 数据集路径
data_set = [train,test]
train_path = 'D:/jupyternotebook/X-1/CWRU_data/train_set/'
test_path = 'D:/jupyternotebook/X-1/CWRU_data/test_set/'

train_set_path = []
for i in range(10):
    train_set_path.append(train_path+f'{i}/')
print(train_set_path)

test_set_path = []
for i in range(10):
    test_set_path.append(test_path+f'{i}/')
print(test_set_path)

data_set_path = [train_set_path,test_set_path]
"""

def makeTimeFrequencyImage(data, img_path, img_size,sampling_period=1.0/12000,totalscale=128,wavename='cmor1-1'):
    """
    生成时频图像并保存到指定路径。
    
    参数:
    data (numpy.ndarray): 输入数据,通常为一维时间序列数据。
    img_path (str): 生成图像的保存路径,包括文件名和格式。
    img_size (tuple): 生成图像的尺寸,例如 (128, 128)。
    """
    fc = pywt.central_frequency(wavename)
    cparam = 2*fc*totalscale
    scales = cparam/np.arange(totalscale,0,-1)
    #cols = 10
    #rows = (data.shape[0] + cols - 1) // cols
    # 计算连续小波变换
    #fig,axes = plt.subplots(rows,10,figsize=(30,rows*3))
    #axes = axes.flatten()  # 展平子图数组
    for i in range(data.shape[0]):
        plt.figure(figsize=img_size)
        coeffs, freqs = pywt.cwt(data[i,:], scales,wavename,sampling_period)
        # 计算小波系数的绝对值
        amp = np.abs(coeffs)
        #freq_max = freqs.max()
        t = np.linspace(0, sampling_period, 256, endpoint=False)
        #c = axes[i].contourf(t,freqs,amp,cmap='jet')
        c = plt.contourf(t,freqs,amp,cmap='jet')
        #plt.colorbar(c,ax=axes[i])
        #plt.colorbar(c)
        # 保存每个子图
        # 隐藏坐标轴
        plt.xticks([])
        plt.yticks([])

        plt.savefig(img_path+f'time_frequency_image_{i}.png')
        plt.close()
    gc.collect()
    print('over')

def GenerateImageDataset(data_set,path_list,img_size,sampling_period=1.0/12000,totalscale=128,wavename='cmor1-1'):
    """
    data_set:是数据列表，存放有训练、验证、测试集
    path_list：是路径列表，存放有各个数据集各类型数据的存储路径
    该函数将生成各个数据集的各类型图像并存储到对应位置
    """""
    for i in range(len(path_list)):
        user_input = input("是否继续执行（0结束/1执行下一次图片生成）:")
        if user_input == '1':
           makeTimeFrequencyImage(data_set[i],path_list[i],img_size,sampling_period,totalscale,wavename)
        else:
            print("结束!")
            break
    gc.collect() # 释放内存
    print('all over')

"""
GenerateImageDataset(data_set[1],data_set_path[1],(10,5))
GenerateImageDataset(data_set[1][0:],data_set_path[1][0:],(10,5))
"""

def copy_random_images(train_set_dir, target_dir, num_images):
    # 遍历0-9的文件夹
    for i in range(10):
        source_folder = os.path.join(train_set_dir, str(i))
        target_folder = os.path.join(target_dir, str(i))

        # 如果源文件夹存在
        if os.path.exists(source_folder):
            # 创建目标文件夹（如果不存在）
            os.makedirs(target_folder, exist_ok=True)
            
            # 获取源文件夹中的所有图片文件
            images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

            # 如果文件数量小于要复制的数量，则调整要复制的数量
            num_to_copy = min(num_images, len(images))

            # 随机抽取图片
            selected_images = random.sample(images, num_to_copy)

            # 复制文件
            for image in selected_images:
                src = os.path.join(source_folder, image)
                dst = os.path.join(target_folder, image)
                shutil.copy(src, dst)

"""
train_set_directory = 'train_set'
target_directory = 'BR1_100'目标目录 = 'BR1_100'
num_images_to_copy = x  # 将x替换为你想要复制的图片数量

copy_random_images(train_set_directory, target_directory, num_images_to_copy)
"""

def copy_images(train_set_dir, target_dir, num_images):
    # 复制train_set/0 文件夹中的所有图片
    source_folder_0 = os.path.join(train_set_dir, '0')
    target_folder_0 = os.path.join(target_dir, '0')

    if os.path.exists(source_folder_0):
        os.makedirs(target_folder_0, exist_ok=True)
        images_0 = [f for f in os.listdir(source_folder_0) if os.path.isfile(os.path.join(source_folder_0, f))]
        
        # 复制所有图片
        for image in images_0:
            src = os.path.join(source_folder_0, image)
            dst = os.path.join(target_folder_0, image)
            shutil.copy(src, dst)

    # 从1-9的文件夹中随机抽取图片并复制
    for i in range(1, 10):
        source_folder = os.path.join(train_set_dir, str(i))
        target_folder = os.path.join(target_dir, str(i))

        if os.path.exists(source_folder):
            os.makedirs(target_folder, exist_ok=True)
            images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
            
            # 如果文件数量小于要复制的数量，则调整要复制的数量
            num_to_copy = min(num_images, len(images))

            # 随机抽取图片
            selected_images = random.sample(images, num_to_copy)

            # 复制文件
            for image in selected_images:
                src = os.path.join(source_folder, image)
                dst = os.path.join(target_folder, image)
                shutil.copy(src, dst)
# 该函数可以从1-9类的训练集中随机抽取指定数量的图片用于构造不同BR的训练集，同时0类训练集保持2000的数量

"""
train_set_directory = 'D:/jupyternotebook/X-1/CWRU_data/train_set'
target_directory = 'D:/jupyternotebook/X-1/CWRU_data/BR1_100_train_set'
num_images_to_copy =  20  # 你想要从1-9文件夹中随机抽取的图片数量

copy_images(train_set_directory, target_directory, num_images_to_copy)

train_set_directory = 'D:/jupyternotebook/X-1/CWRU_data/train_set'
target_directory = 'D:/jupyternotebook/X-1/CWRU_data/BR1_1_train_set'
num_images_to_copy =  2000  # 你想要从1-9文件夹中随机抽取的图片数量

copy_images(train_set_directory, target_directory, num_images_to_copy)
"""

def copy_images(train_set_dir, target_dir, num_images):
    for i in range(10):  # 遍历0到9类
        source_folder = os.path.join(train_set_dir, str(i))
        target_folder = os.path.join(target_dir, str(i))

        if os.path.exists(source_folder):
            os.makedirs(target_folder, exist_ok=True)
            images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)) and f.lower().endswith(('.jpg', '.png'))]
            
            # 如果文件数量小于要复制的数量，则调整要复制的数量
            num_to_copy = min(num_images, len(images))

            # 随机抽取图片
            selected_images = random.sample(images, num_to_copy)

            # 复制文件
            for image in selected_images:
                src = os.path.join(source_folder, image)
                dst = os.path.join(target_folder, image)
                shutil.copy(src, dst)
    print('over!')

# 示例调用
# copy_images('train_set', 'target_set', 5)  # 从每个类随机抽取5张图片
"""
# 使用示例
source_root = 'D:/jupyternotebook/X-1/CWT-IL-ACSAWGAN-GP/Data/CWRU/BR1_2_train_set'  # 源数据集根目录
target_root = 'D:/jupyternotebook/X-1/CWT-IL-ACSAWGAN-GP/Data/CWRU/BR1_2_train_set_balance'  # 目标平衡数据集根目录
num_images = 1000  # 每个文件夹需要复制的图片数量
copy_images(source_root,target_root,num_images)
"""

def copy_and_modify_images(a_folder, b_folder, x, y):
    # 确保目标文件夹存在
    if not os.path.exists(b_folder):
        os.makedirs(b_folder)

    for i in range(10):
        a_subfolder = os.path.join(a_folder, str(i))
        b_subfolder = os.path.join(b_folder, str(i))
        
        # 确保子文件夹存在
        if not os.path.exists(a_subfolder):
            continue
        if not os.path.exists(b_subfolder):
            os.makedirs(b_subfolder)

        # 筛选图片文件
        images_a = [f for f in os.listdir(a_subfolder) if os.path.isfile(os.path.join(a_subfolder, f))]
        images_b = [f for f in os.listdir(b_subfolder) if os.path.isfile(os.path.join(b_subfolder, f))]

        if i == 0:
            # 对0文件夹处理y张图片
            to_delete_b = random.sample(images_b, min(y, len(images_b)))
            for img in to_delete_b:
                os.remove(os.path.join(b_subfolder, img))
            to_copy_a = random.sample(images_a, min(y, len(images_a)))
            for img in to_copy_a:
                shutil.copy(os.path.join(a_subfolder, img), b_subfolder)
        else:
            # 对1-9文件夹处理x张图片
            to_delete_b = random.sample(images_b, min(x, len(images_b)))
            for img in to_delete_b:
                os.remove(os.path.join(b_subfolder, img))
            to_copy_a = random.sample(images_a, min(x, len(images_a)))
            for img in to_copy_a:
                shutil.copy(os.path.join(a_subfolder, img), b_subfolder)

# 使用示例
#a_folder_path = 'D:/jupyternotebook/X-1/CWT-IL-ACSAWGAN-GP/Data/CWRU/BR1_2_train_set'  # 替换为a文件夹路径
#b_folder_path = 'D:/jupyternotebook/X-1/CWT-IL-ACSAWGAN-GP/Gen-Data/CWRU/CWT-IL-ACSAWGAN-GP/BR1_2/gen_img_save'  # 替换为b文件夹路径
#x = 1000  # 其他1-9文件夹复制的图片数量

# 调用函数
#copy_and_modify_images(a_folder_path, b_folder_path, x, y)  # 修改路径和x, y值

"""
说明：
该函数旨在将平衡的训练集的真实图片数据全部替换到对应生成数据当中形成平衡后的生成和真实训练样本用于训练分类器，x是要替换的数量，即训练集0-9每个文件夹下的图片数据量
# 使用示例
a_folder_path = 'D:/jupyternotebook/X-1/CWT-IL-ACSAWGAN-GP/Data/CWRU/BR1_400_train_set'  # 替换为a文件夹路径
b_folder_path = 'D:/jupyternotebook/X-1/CWT-IL-ACSAWGAN-GP/Comparison Models/DSEA-SMOTE/BR1_400/images'  # 替换为b文件夹路径
y = 2000  # 第0文件夹复制的图片数量
x = 5  # 其他1-9文件夹复制的图片数量
"""