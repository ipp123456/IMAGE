from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import time
import os
from efficientnet.model import EfficientNet
import pickle
from PIL import Image
import numpy as np
import json
from image_process import cut_pic

def get_file_name(path): # 获取目录下的所有文件
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

def get_pic_name(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = root+'\\'+file
            # if file_path[-3:] == 'jpg':
            file_list.append([file_path,file[:-4]])
    return file_list

def make_dirs(path):
    """Make Directory If not Exists""" # 创建文件夹
    if not os.path.exists(path):
        os.makedirs(path)
# some parameters
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_name = 'efficientnet-b4'
pth_map = {
    'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
}
model_ft = EfficientNet.from_name(net_name)


input_size = 224
state_dict = torch.load('model/efficientnet-b4.pth', map_location=torch.device('cpu'))
# model_ft.load_state_dict(state_dict)

data_transforms =  transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

with open("dict.json", "r") as f:
    dict = json.load(f)
inverse_dic={}
for key,val in dict.items():
    inverse_dic[val] = key

# 预处理图片
pic_list = get_pic_name('test')
save_path= 'test_processed'
make_dirs(save_path)
for pic_path, pic_name in pic_list:
    cut_pic(pic_path, pic_name, save_path)

#测试图片
file_list = get_file_name(save_path)

for image_dir in file_list:
    img = Image.open(image_dir)
    img2 = data_transforms(img).unsqueeze(0)

    logits = state_dict(img2)
    y = F.softmax(logits, dim=-1)
    label = np.argmax(y.detach().cpu().numpy(), axis=1)[0]
    score = y[0,label].item()
    print(image_dir, inverse_dic[label], score)


