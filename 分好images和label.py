import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def load_itk_image(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def load_mhd():
    numpyImage, numpyOrigin, numpySpacing = load_itk_image(
        r'prostate2012/TrainingData_Part1/Case00.mhd')
    print(numpyOrigin, numpySpacing)
    print(numpyImage.shape)
    # print(numpyImage)

    save_path = r"prostate2012/Trainingimages_part1"
    for i in range(numpyImage.shape[0]):  # z, y, x
        arr = numpyImage[i, :, :]  # 获得第i张的单一数组
        print(arr.shape)
        print(arr[256])

        plt.imsave(os.path.join(save_path, "{}_mask.png".format(i)), arr, cmap='gray')
        print('photo {} finished'.format(i))


def deal_set(dir_path, save_path):
    image_dir = "images"
    label_dir = "labels"
    images_path = os.path.join(save_path, image_dir)
    labels_path = os.path.join(save_path, label_dir)
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    if not os.path.exists(labels_path):
        os.mkdir(labels_path)

    list = os.listdir(dir_path)
    # print(list)
    # 处理images
    images_filename = [filename for filename in list if '.mhd' in filename and '_segmentation' not in filename]
    print(images_filename)
    for i in range(len(images_filename)):
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(os.path.join(dir_path, images_filename[i]))
        print(numpyImage.shape)
        for j in range(numpyImage.shape[0]):  # z, y, x
            arr = numpyImage[j, :, :]  # 获得第i张的单一数组

            plt.imsave(os.path.join(images_path, f"{i}_{j}.png"), arr, cmap='gray')
            print(f'photo {i}_{j} finished')
    labels_filename = [filename for filename in list if '_segmentation.mhd' in filename]
    print(labels_filename)
    for i in range(len(labels_filename)):
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(os.path.join(dir_path, labels_filename[i]))
        print(numpyImage.shape)
        for j in range(numpyImage.shape[0]):
            arr = numpyImage[j, :, :]
            plt.imsave(os.path.join(labels_path, f"{i}_{j}.png"), arr, cmap='gray')
            print(f'label {i}_{j} finished')


if __name__ == "__main__":
    part1_folder = r"D:\softwarefiles\Test1\prostate2012\TrainingData_Part1\resample"
    part1_save_path = "Training_part1"
    deal_set(part1_folder, part1_save_path)
    #
    # deal_set(r"D:\softwarefiles\Test1\prostate2012\TrainingData_Part3\resample", "Training_part3")
    # deal_set(r'D:\softwarefiles\Test1\prostate2012\TrainingData_Part2\resample', "Training_part2")

