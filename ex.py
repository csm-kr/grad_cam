from torchvision import models

vgg16 = models.vgg16(pretrained=True)
print(vgg16)