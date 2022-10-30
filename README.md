# TinySSD
0.环境
Python>=3.8
CUDA>=11.6


1.生成数据:
生成训练样本，放在detection/sysu_train/下
调用detection/create_train.py实现自动合成

detection/sysu_train/images是合成的训练样本
detection/sysu_train/label.csv是标注信息

2.数据处理及载入
(1)read_data实现数据读入
(2)Dataset自定义数据集
(3)load_data实现数据集载入

3.训练流程
（1） 生成多尺度的锚框，为每个锚框预测类别和偏移量
（2） 为每个锚框标注类别和偏移量
（3） 根据类别和偏移量的预测和标注值计算损失函数
