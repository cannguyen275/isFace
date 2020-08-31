import cv2
import torch
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import time
from backbone.mobilenet_v3 import mobilenetv3
from backbone.shufflenet_v2 import shufflenet_v2_x0_5
from helper.utils import load_checkpoint


def get_file_ffolder(path):
    files_name = os.listdir(path)
    file_paths = [os.path.join(path, file_name) for file_name in files_name]
    return file_paths


device = torch.device('cpu')
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
model = shufflenet_v2_x0_5()
# pretrained = torch.load("model/checkpoint_149_0.043281020134628756.tar")
model = load_checkpoint(model, 'model/checkpoint_149_0.010453527558476049.tar')
model.eval()
model = model.to(device)
images_path = get_file_ffolder("Your Image folder")
classes = ['face', 'nonface']
for path in images_path:
    # path = '/home/can/AI_Camera/pose_estimation/hinh_2.jpg'
    print(path)
    img = cv2.imread(path)
    # img = cv2.resize(img, (112, 112))
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    a = time.time()
    output = model(img)
    softmax = nn.Softmax(dim=1)
    smax_out = softmax(output)
    prob, predicted = torch.max(smax_out, 1)
    print(time.time() - a)
    # print(predicted)
    print('Predicted:' + classes[predicted] + "   " + str(prob.data[0] * 100))
