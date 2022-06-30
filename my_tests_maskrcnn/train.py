import torchvision
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import time
import _init_paths
import torch
from torchvision import transforms as T
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN
from pycocotools.coco import COCO
from datasets import MaskRCNN_Dataset


from pycocotools.mask import decode
from utils.img_utils import dialate_boxes
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
import torchvision.models.segmentation

def loadData(dataset, batchSize=3):
  batch_Imgs=[]
  batch_Data=[]
  for i in range(batchSize):
        idx = random.randint(0, len(dataset))
        img, data = dataset[idx]

        batch_Imgs.append(img)
        batch_Data.append(data)  
  
  batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
  batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
  return batch_Imgs, batch_Data

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data/cowbird/images', help='Path to image folder')
parser.add_argument('--annfile', default='data/cowbird/annotations/instance_train.json', help='Path to annotation')
parser.add_argument('--index', type=int, default=8, help='Index to the dataset for an example')
parser.add_argument('--outdir', type=str, default='tests_detection', help='Folder for output images')

if __name__ == "__main__":
    batchSize=2
    imageSize=[256,256]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parser.parse_args()
    root = args.root
    annfile = args.annfile


    normalize_out = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
    # Split data into train and validation, and create dataloaders for them
    dataset = MaskRCNN_Dataset(root=root, annfile=annfile, scale_factor=0.25, transform=normalize_out, output_size=128)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(list(dataset))-100, 100])


    # Load a maskRCNN finetuned on our birds
    network_transform = GeneralizedRCNNTransform(800, 1333, (0,0,0), (1,1,1))
    backbone = resnet_fpn_backbone(backbone_name='resnet101', pretrained=False)
    model = MaskRCNN(backbone, num_classes=2)
    model.transform = network_transform
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    Adam_opt = torch.optim.Adam(params)
    model.train()


    # Training Process
    for i in range(15):
        images, targets = loadData(train_data, batchSize=2)
        # print(targets[0])
        images = list(image.to(device) for image in images)
        targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
        
        if i == 0:
                print(targets[0])
        
        Adam_opt.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        Adam_opt.step()
        
        print(i,'loss:', losses.item())
        if i%5==0:
                torch.save(model.state_dict(), str(i)+".torch")
                print("Save model to:",str(i)+".torch")