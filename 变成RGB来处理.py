import shutil
import cv2
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

# image = Image.open(r"D:\softwarefiles\Test1\prostate2012\Training_part1\labels\1_0_mask.png").convert('RGB')
# # print(image)
# # array = np.array(image)
# print(array.shape)  # 单张图片的shape：[512, 512, 3]
# print(array.sum())  # 没用的掩模图的像素值和为0
# 图像和掩模图都是RGBA格式
# label像素值获取
label_path = r"D:\softwarefiles\Test1\prostate2012\Training_part1\labels"
label_dir = os.listdir(label_path)
label_dir.sort(key=lambda x: int(x.split('_', 1)[0]))  # 将文件名排序:这里只按'_'前的数字排序了，后面的还是乱的
# print(label_dir)
label_filename_list = []
for i in range(38):
    filename = [name for name in label_dir if f'{i}_' in name]
    filename_right = []
    for name in filename:
        if name.split('_', 1)[0] == f'{i}':
            filename_right.append(name)
    filename_right.sort(key=lambda x: int((x.split('_', 1)[1]).split('.', 1)[0]))
    for j in range(len(filename_right)):
        label_filename_list.append(filename_right[j])

print('label_filename_list', label_filename_list)
labels = [Image.open(os.path.join(label_path, file_name)).convert('RGB').resize((512, 512)) for file_name in label_filename_list]
label_list = [np.asarray(label) for label in labels]
labels_after = [cv2.resize(image, (512, 512)) for image in label_list]
label_array = np.asarray(labels_after)
index = [i.sum() > 0 for i in label_array]
img_prostate = label_array[index]
label_dirs = np.asarray(label_dir)
prostate_filename = label_dirs[index]

# print(len(img_prostate))  #458
# image像素值获取
image_path = r"D:\softwarefiles\Test1\prostate2012\Training_part1\images"
image_dir = os.listdir(image_path)
image_dir.sort(key=lambda x: int(x.split('_', 1)[0]))
# print(image_dir)
image_filename_list = []
for i in range(38):
    filename = [name for name in image_dir if f'{i}_' in name]
    filename_right = []
    for name in filename:
        if name.split('_', 1)[0] == f'{i}':
            filename_right.append(name)
    filename_right.sort(key=lambda x: int((x.split('_', 1)[1]).split('.', 1)[0]))
    for j in range(len(filename_right)):
        image_filename_list.append(filename_right[j])

images = [Image.open(os.path.join(image_path, file_name)).convert('RGB') for file_name in image_filename_list]
image_list = [np.asarray(image) for image in images]
images_after = [cv2.resize(image, (512, 512)) for image in image_list]
image_array = np.asarray(images_after).astype(np.float16)
# print(image_array.shape) # [800,512,512,3]
img_patient = image_array[index]
image_dirs = np.asarray(image_dir)
patient_filename = image_dirs[index]  # 这里得到每个有用的image的文件名


print(img_patient.shape)

# plt.hist(image_array.reshape(-1, ), bins=50)  # 输出像素直方图
# plt.show()

# 可视化展示
# j = 1
# for i in range(11, 20):
#     plt.subplot(3, 3, j)
#     plt.imshow(img_patient[i], cmap='gray')
#     plt.axis('off')
#     j += 1
# plt.show()
# ============================================
# 自定义window函数

def windowing(imgs, window_width, window_center):
    minWindow = float(window_center) - 0.5 * float(window_width)
    new_img = (imgs - minWindow) / float(window_width)
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    return new_img


image_w = windowing(img_patient, 350, 50)
print(image_w.max())
print(image_w.shape)


# j = 1
# for i in range(11, 20):
#     plt.subplot(3, 3, j)
#     plt.imshow(image_w[i], cmap='gray')
#     plt.axis('off')
#     j += 1
# plt.show()

# ====================================================
# 自定义批量均衡化函数
def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_res = np.zeros_like(imgs)
    for i in range(3):
        img_res[i, :, :] = clahe.apply(np.array(imgs[i, :, :]))
    return img_res / 255


image_clahe = np.zeros((1146, 3, 512, 512))
for i, img in enumerate(img_patient):
    image_clahe[i] = clahe_equalized(np.transpose(img, (2, 1, 0)))

print(image_clahe.shape)
image_clahe_ = np.zeros((1146, 512, 512, 3))
for i, img in enumerate(image_clahe):
    image_clahe_[i] = np.transpose(img, (2, 1, 0))
print(image_clahe_.shape)
print(image_clahe_.max())
# j = 1
# for i in range(11, 20):
#     plt.subplot(3, 3, j)
#     plt.imshow(image_clahe_[i], cmap='gray')
#     plt.axis('off')
#     j += 1
# plt.show()

# =============================================================
tmp = "newtmp"
save_path = "Training_part1"
tmp_path = os.path.join(save_path, tmp)
if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)
image_dir = r"./images"
label_dir = r"./labels"
images_path = os.path.join(tmp_path, image_dir)
labels_path = os.path.join(tmp_path, label_dir)
if not os.path.exists(images_path):
    os.mkdir(images_path)
if not os.path.exists(labels_path):
    os.mkdir(labels_path)
# ===============================================
save_patient_path = images_path
save_prostate_path = labels_path
for path in [save_patient_path, save_prostate_path]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
for i in range(len(img_patient)):
    plt.imsave(os.path.join(save_prostate_path, f"{prostate_filename[i].split('_', 1)[0]}_{i}.png"), img_prostate[i],
               )
    plt.imsave(os.path.join(save_patient_path, f"{patient_filename[i].split('_', 1)[0]}_{i}.png"), image_clahe_[i],
               )


# # ======================================part2数据处理==================================
#
# # label像素值获取
# label_path = r"D:\softwarefiles\Test1\prostate2012\Training_part2\labels"
# label_dir = os.listdir(label_path)
# label_dir.sort(key=lambda x: int(x.split('_', 1)[0]))  # 将文件名排序
# # print(label_dir)
#
# label_filename_list = []
# for i in range(26):
#     filename = [name for name in label_dir if f'{i}_' in name]
#     filename_right = []
#     for name in filename:
#         if name.split('_', 1)[0] == f'{i}':
#             filename_right.append(name)
#     filename_right.sort(key=lambda x: int((x.split('_', 1)[1]).split('.', 1)[0]))
#     for j in range(len(filename_right)):
#         label_filename_list.append(filename_right[j])
#
# labels = [Image.open(os.path.join(label_path, file_name)).convert('RGB') for file_name in label_filename_list]
# label_list = [np.asarray(label) for label in labels]
# labels_after = [cv2.resize(image, (512, 512)) for image in label_list]
# label_array = np.asarray(labels_after)
# index = [i.sum() > 0 for i in label_array]
# img_prostate = label_array[index]
# label_dirs = np.asarray(label_dir)
# prostate_filename = label_dirs[index]
#
# # print(len(img_prostate))  #458
# # image像素值获取
# image_path = r"D:\softwarefiles\Test1\prostate2012\Training_part2\images"
# image_dir = os.listdir(image_path)
# image_dir.sort(key=lambda x: int(x.split('_', 1)[0]))
#
# image_filename_list = []
# for i in range(26):
#     filename = [name for name in image_dir if f'{i}_' in name]
#     filename_right = []
#     for name in filename:
#         if name.split('_', 1)[0] == f'{i}':
#             filename_right.append(name)
#     filename_right.sort(key=lambda x: int((x.split('_', 1)[1]).split('.', 1)[0]))
#     for j in range(len(filename_right)):
#         image_filename_list.append(filename_right[j])
#
# images = [Image.open(os.path.join(image_path, file_name)).convert('RGB') for file_name in image_filename_list]
# image_list = [np.asarray(image) for image in images]
# images_after = [cv2.resize(image, (512, 512)) for image in image_list]
# image_array = np.asarray(images_after)
# # print(image_array.shape) # [800,512,512,4]
# img_patient = image_array[index]
# image_dirs = np.asarray(image_dir)
# patient_filename = image_dirs[index]  # 这里得到每个有用的image的文件名
# # =============================windowing=======
# image_w = windowing(img_patient, 350, 50)
# print(image_w.shape)
#
# # ========================直方图均衡化==========
# image_clahe = np.zeros((162, 3, 512, 512))
# for i, img in enumerate(img_patient):
#     image_clahe[i] = clahe_equalized(np.transpose(img, (2, 1, 0)))
#
# print(image_clahe.shape)
# image_clahe_ = np.zeros((162, 512, 512, 3))
# for i, img in enumerate(image_clahe):
#     image_clahe_[i] = np.transpose(img, (2, 1, 0))
# print(image_clahe_.shape)
# # ===========================
# tmp = "newtmp"
# save_path = "Training_part2"
# tmp_path = os.path.join(save_path, tmp)
# if not os.path.exists(tmp_path):
#     os.mkdir(tmp_path)
# image_dir = r"./images"
# label_dir = r"./labels"
# images_path = os.path.join(tmp_path, image_dir)
# labels_path = os.path.join(tmp_path, label_dir)
# if not os.path.exists(images_path):
#     os.mkdir(images_path)
# if not os.path.exists(labels_path):
#     os.mkdir(labels_path)
# # ===================================
# save_patient_path = images_path
# save_prostate_path = labels_path
# for path in [save_patient_path, save_prostate_path]:
#     if os.path.exists(path):
#         shutil.rmtree(path)
#     os.makedirs(path)
# for i in range(len(img_prostate)):
#     plt.imsave(os.path.join(save_prostate_path, f"{prostate_filename[i].split('_', 1)[0]}_{i}.png"), img_prostate[i],
#                )
#     plt.imsave(os.path.join(save_patient_path, f"{patient_filename[i].split('_', 1)[0]}_{i}.png"), image_clahe_[i],
#                )
#
#


# ===============================part3数据处理=================================

# label像素值获取
label_path = r"D:\softwarefiles\Test1\prostate2012\Training_part3\labels"
label_dir = os.listdir(label_path)
label_dir.sort(key=lambda x: int(x.split('_', 1)[0]))  # 将文件名排序
# print(label_dir)

label_filename_list = []
for i in range(26):
    filename = [name for name in label_dir if f'{i}_' in name]
    filename_right = []
    for name in filename:
        if name.split('_', 1)[0] == f'{i}':
            filename_right.append(name)
    filename_right.sort(key=lambda x: int((x.split('_', 1)[1]).split('.', 1)[0]))
    for j in range(len(filename_right)):
        label_filename_list.append(filename_right[j])

labels = [Image.open(os.path.join(label_path, file_name)).convert('RGB') for file_name in label_filename_list]
label_list = [np.asarray(label) for label in labels]
labels_after = [cv2.resize(image, (512, 512)) for image in label_list]
label_array = np.asarray(labels_after)
index = [i.sum() > 0 for i in label_array]
img_prostate = label_array[index]
label_dirs = np.asarray(label_dir)
prostate_filename = label_dirs[index]

# print(len(img_prostate))  #458
# image像素值获取
image_path = r"D:\softwarefiles\Test1\prostate2012\Training_part3\images"
image_dir = os.listdir(image_path)
image_dir.sort(key=lambda x: int(x.split('_', 1)[0]))

image_filename_list = []
for i in range(26):
    filename = [name for name in image_dir if f'{i}_' in name]
    filename_right = []
    for name in filename:
        if name.split('_', 1)[0] == f'{i}':
            filename_right.append(name)
    filename_right.sort(key=lambda x: int((x.split('_', 1)[1]).split('.', 1)[0]))
    for j in range(len(filename_right)):
        image_filename_list.append(filename_right[j])

images = [Image.open(os.path.join(image_path, file_name)).convert('RGB') for file_name in image_filename_list]
image_list = [np.asarray(image) for image in images]
images_after = [cv2.resize(image, (512, 512)) for image in image_list]
image_array = np.asarray(images_after)
# print(image_array.shape) # [800,512,512,4]
img_patient = image_array[index]
image_dirs = np.asarray(image_dir)
patient_filename = image_dirs[index]  # 这里得到每个有用的image的文件名
# =============================windowing=======
image_w = windowing(img_patient, 350, 50)
print(image_w.shape)

# ========================直方图均衡化==========
image_clahe = np.zeros((158, 3, 512, 512))
for i, img in enumerate(img_patient):
    image_clahe[i] = clahe_equalized(np.transpose(img, (2, 1, 0)))

print(image_clahe.shape)
image_clahe_ = np.zeros((158, 512, 512, 3))
for i, img in enumerate(image_clahe):
    image_clahe_[i] = np.transpose(img, (2, 1, 0))
print(image_clahe_.shape)
# ===========================
tmp = "newtmp"
save_path = "Training_part3"
tmp_path = os.path.join(save_path, tmp)
if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)
image_dir = r"./images"
label_dir = r"./labels"
images_path = os.path.join(tmp_path, image_dir)
labels_path = os.path.join(tmp_path, label_dir)
if not os.path.exists(images_path):
    os.mkdir(images_path)
if not os.path.exists(labels_path):
    os.mkdir(labels_path)
# ===================================
save_patient_path = images_path
save_prostate_path = labels_path
for path in [save_patient_path, save_prostate_path]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
for i in range(len(img_prostate)):
    plt.imsave(os.path.join(save_prostate_path, f"{prostate_filename[i].split('_', 1)[0]}_{i}.png"), img_prostate[i],
               )
    plt.imsave(os.path.join(save_patient_path, f"{patient_filename[i].split('_', 1)[0]}_{i}.png"), image_clahe_[i],
               )