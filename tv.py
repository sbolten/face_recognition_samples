from torchvision import models
from torchvision import transforms
from PIL import Image

import torch

alexnet = models.alexnet(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
googlenet = models.googlenet(pretrained=True)

transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

img = Image.open("/home/stephan/src/horse.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

resnet50.eval()
alexnet.eval()
googlenet.eval()
#out = alexnet(batch_t)
out = resnet50(batch_t)
#out = googlenet(batch_t)
print(out.shape)

with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#print(classes[index[0]], percentage[index[0]].item())
_, indices = torch.sort(out, descending=True)
[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]