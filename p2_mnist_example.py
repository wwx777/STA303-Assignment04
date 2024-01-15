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

# 计算权重在通道上的平均值，从而将它们转换为单通道
# 新的权重将有形状 [64, 1, 7, 7]
    new_weights = first_layer_weights.mean(dim=1, keepdim=True)

# 用新的单通道权重替换第一层的权重
    model.conv1.weight = torch.nn.Parameter(new_weights)

# 修改第一层的输入通道数为1
    model.conv1.in_channels = 1
   
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(model_device)


    dataset = build_dataset('mnist')

    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [30000, 30000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True)

    
    #######################################
    # A standard process of conformal prediction
    #######################################    
    alpha = args.alpha
    print(f"Experiment--Data : Mnist, Model : {model_name}, Score : THR, Predictor : ClusterPredictor, Alpha : {alpha}")
    score_function = THR()
    predictor = ClusterPredictor(score_function, model)
    print(f"The size of calibration set is {len(cal_dataset)}.")
    predictor.calibrate(cal_data_loader, alpha)
    predictor.evaluate(test_data_loader)
    evaluation_results = predictor.evaluate(test_data_loader)
     # 打印评估结果
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")


