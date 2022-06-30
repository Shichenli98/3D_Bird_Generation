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
from PIL import Image

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class


def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def instance_segmentation_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        # cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imsave('prediction.jpg', img)
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()



if __name__ == "__main__":
    # Initialization
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    # network_transform = GeneralizedRCNNTransform(800, 1333, (0,0,0), (1,1,1))
    # backbone = resnet_fpn_backbone(backbone_name='resnet101', pretrained=True)
    # model = MaskRCNN(backbone, num_classes=2)
    # model.transform = network_transform
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # load trained model
    # model.load_state_dict(torch.load("1.torch"))
    model.to(device) # move model to the right devic
    model.eval() # test mode

    # load data
    imgPath = '/mnt/d/selfie.jfif'
    # instance_segmentation_api(img_path=imgPath)
    # imgPath = '/mnt/d/Research/avian-mesh/data/cowbird/images/00000026.jpg'
    images = cv2.imread(imgPath)
    images = cv2.resize(images, [256, 256], cv2.INTER_LINEAR)
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)
    with torch.no_grad():
        pred = model(images)
    
    # postprocess to output those scores larger than 0.8
    im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
    im2 = im.copy()
    for i in range(len(pred[0]['masks'])):
        msk=pred[0]['masks'][i,0].detach().cpu().numpy()
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        if scr>0.8 :
            im2[:,:,0][msk>0.5] = random.randint(0,255)
            im2[:, :, 1][msk > 0.5] = random.randint(0,255)
            im2[:, :, 2][msk > 0.5] = random.randint(0, 255)
    # cv2.imshow(str(scr), np.hstack([im,im2]))
    # cv2.waitKey()
    cv2.imwrite('prediction.jpg', np.hstack([im,im2]))
    print('Successfully saved')

