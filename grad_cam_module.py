from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class grad_cam(nn.Module):
    def __init__(self, model, module, layer):
        super(grad_cam, self).__init__()
        self.model = model
        self.module = module
        self.layer = layer
        self.register_hooks()

    def register_hooks(self):
        for modue_name, module in self.model._modules.items():
            if modue_name == self.module:
                for layer_name, module in module._modules.items():
                    if layer_name == self.layer:
                        module.register_forward_hook(self.forward_hook)
                        module.register_backward_hook(self.backward_hook)

    def forward(self, input, target_index):
        outs = self.model(input).squeeze()
        # backward를 통해서 자동으로 backward_hook 함수가 호출되고, self.backward_result에 gradient가 저장됩니다.
        outs[target_index].backward(retain_graph=True)
        # gradient의 평균을 구합니다. (alpha_k^c)
        a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)
        # self.foward_result를 이용하여 Grad-CAM을 계산합니다.
        out = torch.sum(a_k * self.forward_result, dim=0).cpu()
        # normalize
        out = (out + torch.abs(out)) / 2
        out = out / torch.max(out)
        out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), [224, 224])  # 4D로 바꿈
        return out.detach().squeeze().numpy()

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])


if __name__ == '__main__':

    def preprocess_image(img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        preprocessed_img = torch.from_numpy(preprocessed_img)
        preprocessed_img.unsqueeze_(0)
        input = preprocessed_img.requires_grad_(True)
        return input


    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cv2.imshow("cam", np.uint8(255 * cam))
        cv2.waitKey()

    import os
    import cv2
    import glob
    import numpy as np

    from vgg_training import VGG
    model = VGG()
    epoch = 51
    save_path = './saves'
    save_file_name = 'vgg_classification_voc'
    state_dict = torch.load(os.path.join(save_path, save_file_name) + '.{}.pth'.format(epoch))
    model.load_state_dict(state_dict)
    print(model)
    grad_cam = grad_cam(model=model, module='features', layer='30')

    root = 'D:\Data\VOC_ROOT\TEST\VOC2007\JPEGImages'
    img_list = os.listdir(root)
    img_list = sorted(glob.glob(os.path.join(root, '*.jpg')))
    # print(img_list)
    for img_path in img_list:
        img = cv2.imread(img_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)

        # voc labels
        '''
        ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
         'bus', 'car', 'cat', 'chair', 'cow',
         'diningtable', 'dog', 'horse', 'motorbike', 'person',
         'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        '''

        target_index = 14
        mask = grad_cam(input, target_index)
        show_cam_on_image(img, mask)
