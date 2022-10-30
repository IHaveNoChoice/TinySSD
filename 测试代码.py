import torch
import torchvision
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob

net = TinySSD(num_classes=1)
net = net.to('cpu')

# 加载模型参数
net.load_state_dict(torch.load('net_30.pkl', map_location=torch.device('cpu')))

name = 'detection/test/2.jpg'
X = torchvision.io.read_image(name).unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

output = predict(X)
display(img, output.cpu(), threshold=0.6)