import SimpleITK as sitk
import numpy as np
import os


def img_resmaple(ori_img_file, target_img_file, new_spacing, resamplemethod=sitk.sitkNearestNeighbor):
    """
        @param ori_img_file: 原始的itk图像路径，一般为.mhd
        @param target_img_file: 保存路径
        @param new_spacing: 目标重采样的spacing，如[0.585938, 0.585938, 0.4]
        @param resamplemethod: itk插值⽅法: sitk.sitkLinear-线性、sitk.sitkNearestNeighbor-最近邻、sitk.sitkBSpline等，SimpleITK源码中会有各种插值的方法，直接复制调用即可
    """
    data = sitk.ReadImage(ori_img_file)  # 根据路径读取mhd文件
    original_spacing = data.GetSpacing()  # 获取图像重采样前的spacing
    original_size = data.GetSize()  # 获取图像重采样前的分辨率

    # 有原始图像size和spacing得到真实图像大小，用其除以新的spacing,得到变化后新的size
    new_shape = [
        int(np.round(original_spacing[0] * original_size[0] / new_spacing[0])),
        int(np.round(original_spacing[1] * original_size[1] / new_spacing[1])),
        int(np.round(original_spacing[2] * original_size[2] / new_spacing[2])),
    ]
    print("处理后新的分辨率:{}".format(new_shape))

    # 重采样构造器
    resample = sitk.ResampleImageFilter()

    resample.SetOutputSpacing(new_spacing)  # 设置新的spacing
    resample.SetOutputOrigin(data.GetOrigin())  # 原点坐标没有变，所以还用之前的就可以了
    resample.SetOutputDirection(data.GetDirection())  # 方向也未变
    resample.SetSize(new_shape)  # 分辨率发生改变
    resample.SetInterpolator(resamplemethod)  # 插值算法
    data = resample.Execute(data)  # 执行操作

    sitk.WriteImage(data, target_img_file)  # 将处理后的数据，保存到一个新的mhdw文件中


dir_path = r'D:\softwarefiles\Test1\prostate2012\TrainingData_Part3'
save_dir = r'D:\softwarefiles\Test1\prostate2012\TrainingData_Part3\resample'
list = os.listdir(dir_path)
# print(list)
file_name = [name for name in list if '.mhd' in name]
print(file_name)
file_path = [os.path.join(dir_path, filename) for filename in file_name]
# print(file_path)
save_path = [os.path.join(save_dir, filename) for filename in file_name]
# print(save_path)
new_spacing = [0.625, 0.625, 1.5]
for i in range(len(file_path)):
    img_resmaple(file_path[i], save_path[i], new_spacing)
