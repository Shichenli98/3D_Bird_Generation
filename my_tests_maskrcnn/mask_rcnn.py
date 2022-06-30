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
from utils.evaluation import evaluate_iou, evaluate_euc

def tranform_target(target):
    output = []
    n = len(target['labels'])
    for i in range(n):
        tmp = dict()
        for key in target:
            tmp[key] = target[key][i]
        output.append(tmp)
    return output


def finetune_MaskRCNN(model, loaders, criterion, optimizer, num_epochs=15):
    since = time.time()

    val_iou_history = []
    # Initialization for training
    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0
    # Training
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_iou = 0.0

            # Iterate over data.           
            for input, target in loaders[phase]:
                # target = tranform_target(target)
                print(target, target['boxes'].size())
                input= input.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    output = model(input, target)
                    print(output)
                    loss, [bbox, label, score, pred_mask] = output
                    
                    # we need a better loss function for this
                    iou = criterion(pred_mask, target[-1])
                    euc_dist = evaluate_euc(bbox, target[0])
                    loss = (1 - iou) + (euc_dist / (target[0][-1] * target[0][-2])) + (1 - target[1])**2 + (1 - target[2])**2

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_iou += iou.item() * input.size(0)

                epoch_iou = running_iou / len(loaders[phase].dataset)
                print('{} IoU: {:.4f}'.format(phase, epoch_iou))
            if num_epochs == 1:
                print(output)
            # deep copy the model
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_iou_history.append(epoch_iou)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val IoU: {:4f}'.format(best_iou))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_iou_history


parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data/cowbird/images', help='Path to image folder')
parser.add_argument('--annfile', default='data/cowbird/annotations/instance_train.json', help='Path to annotation')
parser.add_argument('--index', type=int, default=8, help='Index to the dataset for an example')
parser.add_argument('--outdir', type=str, default='tests_detection', help='Folder for output images')

if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = args.root
    annfile = args.annfile

    # normalize_in = T.Normalize(
    #     mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.]
    #     )
    
    # coco = COCO(annfile)
    # available_Ids = coco.getImgIds()

    # imgfiles = coco.loadImgs(available_Ids[:])[0]['file_name']
    # imgpaths = [root + '/' + imgfile for imgfile in imgfiles]
    # imgs = cv2.imread(imgpaths)
    # imgs = [normalize_in(torch.tensor(img).float().permute(2,0,1)) for img in imgs]
    
    normalize_out = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
    # Split data into train and validation, and create dataloaders for them
    dataset = MaskRCNN_Dataset(root=root, annfile=annfile, scale_factor=0.25, transform=normalize_out)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(list(dataset))-100, 100])
    loaders = dict()
    loaders['train'] = torch.utils.data.DataLoader(train_data, batch_size=2)
    loaders['val'] = torch.utils.data.DataLoader(val_data, batch_size=2)

    # Load a maskRCNN finetuned on our birds
    network_transform = GeneralizedRCNNTransform(800, 1333, (0,0,0), (1,1,1))
    backbone = resnet_fpn_backbone(backbone_name='resnet101', pretrained=False)
    model = MaskRCNN(backbone, num_classes=2)
    model.transform = network_transform
    
    params = [p for p in model.parameters() if p.requires_grad]
    Adam_opt = torch.optim.Adam(params)

    finetune_MaskRCNN(model, loaders, evaluate_iou, Adam_opt, num_epochs=1)
    

    