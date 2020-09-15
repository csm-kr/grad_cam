import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
from xml.etree.ElementTree import parse
from matplotlib.patches import Rectangle
from torchvision import transforms

from config import device


class VOC_Cls_Dataset(data.Dataset):

    # not background for coco
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    """
    ssd_dataset 읽어드리는 로더
    """
    def __init__(self, root="D:\Data\VOC_ROOT", split='TRAIN', transform=None):
        super(VOC_Cls_Dataset, self).__init__()
        root = os.path.join(root, split)
        self.img_list = sorted(glob.glob(os.path.join(root, '*/JPEGImages/*.jpg')))
        self.anno_list = sorted(glob.glob(os.path.join(root, '*/Annotations/*.xml')))
        self.class_idx_dict = {class_name: i for i, class_name in enumerate(self.class_names)}     # class name : idx
        self.idx_class_dict = {i: class_name for i, class_name in enumerate(self.class_names)}     # idx : class name
        self.split = split
        self.transform = transform

    def __getitem__(self, idx):

        visualize = False
        # --------------------------------------------- img read ------------------------------------------------------
        image = Image.open(self.img_list[idx]).convert('RGB')
        labels = self.parse_voc(self.anno_list[idx])

        if self.transform is not None:
            image = self.transform(image)
        labels = torch.from_numpy(labels)

        if visualize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # tensor to img
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            # img_vis += np.array([123, 117, 104], np.float32)
            img_vis *= std
            img_vis += mean
            img_vis = np.clip(img_vis, 0, 1)

            plt.figure('img')
            plt.imshow(img_vis)
            plt.show()

        return image, labels

    def __len__(self):
        return len(self.img_list)

    def parse_voc(self, xml_file_path):

        tree = parse(xml_file_path)
        root = tree.getroot()
        labels = np.zeros([20], dtype=np.float32)

        for obj in root.iter("object"):

            # 'name' tag 에서 멈추기
            name = obj.find('./name')
            class_name = name.text.lower().strip()
            labels[self.class_idx_dict[class_name]] = 1.

        return labels


if __name__ == "__main__":

    # train_transform
    ubuntu_root = "/home/cvmlserver3/Sungmin/data/VOC_ROOT"
    window_root = "D:\Data\VOC_ROOT"
    root = window_root

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_set = VOC_Cls_Dataset(root, split='TRAIN', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)
        print(labels)

