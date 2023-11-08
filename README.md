# Deeplearning-UNet
Prostate Medical Image Segmentation base in Unet 

Unet网络模型是参考B站up主霹雳吧啦Wz的，我的数据集是使用的Prostate MRI Image Segmentation 2012(Promise12)挑战赛提供的前列腺MRI图像公开数据集，其中src文件夹下是模型源码，建议看up的视频，我主要就是对数据集的预处理。
注意：这只是我自己记录我的小项目，路径什么的都没改，需要自己去改。

预处理：主要是把数据集中的三维数据抽出来变成二维的图像，然后把一些没用的数据去除，之后直方图均衡化一下。
