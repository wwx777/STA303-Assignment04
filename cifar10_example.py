# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from tqdm import tqdm

import torchvision.models as models
import torchvision.transforms as transforms

from torchcp.classification.predictors import ClusterPredictor, ClassWisePredictor, SplitPredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS
from torchcp.classification import Metrics
from torchcp.utils import fix_randomness
from examples.common.dataset import build_dataset
import torchvision
import torchvision.transforms as transforms



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()

    fix_randomness(seed=args.seed)

    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    first_layer_weights = model.conv1.weight

    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(model_device)


        # 定义数据集下载路径
    download_path = "C:/Users/user/data/CIFAR10"

# 定义数据预处理转换
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 可选的归一化操作
    ])
    mode='train'
    if mode == "train":
        dataset = dset.CIFAR10(download_path, train=True, download=True, transform=transform)
    elif mode == "test":
        dataset = dset.CIFAR10(download_path, train=False, download=True, transform=transform)
   
    print(len(dataset))
    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True)

    
    #######################################
    # A standard process of conformal prediction
    #######################################    
    alpha = args.alpha
    print(f"Experiment--Data : cifar10, Model : {model_name}, Score : THR, Predictor : SplitPredictor, Alpha : {alpha}")
    score_function = THR()
    predictor = SplitPredictor(score_function, model)
    print(f"The size of calibration set is {len(cal_dataset)}.")
    predictor.calibrate(cal_data_loader, alpha)
    predictor.evaluate(test_data_loader)
    # 假设predictor.evaluate返回一个包含评估指标的字典
    evaluation_results = predictor.evaluate(test_data_loader)

    # 打印评估结果
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")

