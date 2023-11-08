import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class Prostate_Dataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(Prostate_Dataset, self).__init__()
        self.flag = "Training_part1" if train else "Training_part3"
        data_root = os.path.join(root, self.flag, 'newtmp')
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]  # 这里得到的是每张图片的名称
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]  # 这里得到的是每张图片的路径
        self.label_list = [os.path.join(data_root, 'labels', i) for i in img_names] # 得到每个标签图片的路径

        # check files
        for i in self.label_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif") # 这里原本的mask，我们没有，所以不用
        #                  for i in img_names]
        # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        label = Image.open(self.label_list[idx]).convert('L')  # L表示灰度图像
        label = np.array(label) / 255
        # roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # roi_mask = 255 - np.array(roi_mask)
        # mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(label)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):  # 这里的batch是一个列表
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))  # 这里得到最大的shape [c, h, w]
    batch_shape = (len(images),) + max_size  # 加上batch这个维度 [b,c,h,w]
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
