import os
import time
import transforms as T
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision
from mydataset2 import Prostate_Dataset
from src.UNet import UNet
import torch.nn.functional as F

# img_path = r"D:\softwarefiles\bishe\dataset\Training_part1\newtmp\images\3_67.png"
#     roi_mask_path = r"D:\softwarefiles\bishe\dataset\Training_part1\newtmp\labels\3_67.png"
#     assert os.path.exists(weights_path), f"weights {weights_path} not found."
#     assert os.path.exists(img_path), f"image {img_path} not found."
#     assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model.pth"
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    img_dir_path = r'D:\softwarefiles\bishe\dataset\Training_part3\newtmp\images'
    roi_mask_dir = r'D:\softwarefiles\bishe\dataset\Training_part3\newtmp\labels'
    img_filename_list = os.listdir(img_dir_path)

    filename_list = []
    for i in range(12):
        filename = [name for name in img_filename_list if f'{i}_' in name]
        filename_right = []
        for name in filename:
            if name.split('_', 1)[0] == f'{i}':
                filename_right.append(name)
        filename_right.sort(key=lambda x: int((x.split('_', 1)[1]).split('.', 1)[0]))
        for j in range(len(filename_right)):
            filename_list.append(filename_right[j])
    img_filename_list = filename_list

    print(img_filename_list)

    mask_filename_list = os.listdir(roi_mask_dir)

    filename_list = []
    for i in range(12):
        filename = [name for name in img_filename_list if f'{i}_' in name]
        filename_right = []
        for name in filename:
            if name.split('_', 1)[0] == f'{i}':
                filename_right.append(name)
        filename_right.sort(key=lambda x: int((x.split('_', 1)[1]).split('.', 1)[0]))
        for j in range(len(filename_right)):
            filename_list.append(filename_right[j])
    mask_filename_list = filename_list

    print(mask_filename_list)
    image_path = [os.path.join(img_dir_path, path) for path in img_filename_list]
    roi_path = [os.path.join(roi_mask_dir, path) for path in mask_filename_list]
    for i in range(len(image_path)):
        # load roi mask
        # roi_img = Image.open(roi_path[i]).convert('L')
        # roi_img = np.array(roi_img)
        # load image
        original_img = Image.open(image_path[i]).convert('RGB')
        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        save_name = img_filename_list[i]

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))
            # output['out'] = F.softmax(output['out'])
            prediction = output['out'].argmax(1).squeeze(0)
            print(prediction.shape)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            # 将不敢兴趣的区域像素设置成0(黑色)
            # prediction[roi_img == 0] = 0
            print(prediction.shape)
            mask = Image.fromarray(prediction)

            mask.save(f"saved_images/{save_name}")


if __name__ == '__main__':
    main()