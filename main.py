import torch
import visdom
import argparse
from vgg_training import VGG
from dataset import VOC_Cls_Dataset
import os
from config import device
import numpy as np
from loss import Multi_Classificaion_Loss
from torchvision import transforms
import time

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def main():
    # 1. arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)  # 173
    parser.add_argument('--lr', type=float, default=1e-3)
    # for sgd optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default="ssd_vgg_16")
    parser.add_argument('--conf_thres', type=float, default=0.01)
    parser.add_argument('--start_epoch', type=int, default=0)        # to resume
    parser.add_argument('--os_type', type=str, default='window',
                        help='choose the your os type between window and ubuntu')

    opts = parser.parse_args()
    print(opts)

    # 3. visdom
    vis = visdom.Visdom()

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
                                               batch_size=opts.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    # 6. network
    model = VGG(pretrained=True).to(device)

    # 7. loss
    criterion = Multi_Classificaion_Loss()

    # 8. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    # for statement
    for epoch in range(opts.start_epoch, opts.epoch):
        tic = time.time()
        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            # sgd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            toc = time.time() - tic

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            # for each steps
            if idx % 10 == 0:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Learning rate: {lr:.7f} s \t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx,
                              len(train_loader),
                              loss=loss,
                              lr=lr,
                              time=toc))

                if vis is not None:
                    # loss plot
                    vis.line(X=torch.ones((1, 1)).cpu() * idx + epoch * train_loader.__len__(),  # step
                             Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                             win='train_loss',
                             update='append',
                             opts=dict(xlabel='step',
                                       ylabel='Loss',
                                       title='training loss',
                                       legend=['Total Loss']))
        save_path = './saves'
        save_file_name = 'vgg_classification_voc'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, save_file_name + '.{}.pth'.format(epoch)))


if __name__ == "__main__":
    main()



