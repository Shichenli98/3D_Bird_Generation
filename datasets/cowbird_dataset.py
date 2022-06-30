import os
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.mask import decode
import _init_paths
from utils.img_utils import dialate_boxes

import argparse
from torchvision import transforms as T

class Cowbird_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for instance level task, including detection, instance segmentation, 
    and single view reconstruction. Since data are in COCO format, this class utilize
    COCO API to do most of the dataloading.
    """
    def __init__(self, root, annfile, scale_factor=0.25, output_size=256, transform=None):
        self.root = root
        self.coco = COCO(annfile)
        self.imgIds = self.coco.getImgIds(catIds=1)
        self.imgIds.sort()
        
        self.scale_factor = scale_factor
        self.output_size = output_size
        self.transform = transform
        self.data = self.get_data()
        
    def __getitem__(self, index):
        data = self.data[index]
        x, y, w, h = data['bbox']

        # input image
        img = cv2.imread(data['imgpath'])
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (self.output_size, self.output_size))
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2,0,1).float()/255

        # keypoints
        kpts = data['keypoints'].clone()
        valid = kpts[:,-1] > 0
        kpts[valid,:2] -= torch.tensor([x, y])
        kpts[valid,:2] *= self.output_size / w.float()
        
        # mask
        mask = decode(data['rle'])
        mask = mask[y:y+h, x:x+w]
        mask = cv2.resize(mask, (self.output_size, self.output_size))
        mask = torch.tensor(mask).long()
        
        # meta
        size = data['size'] * self.output_size / w.float()
        meta = {
            'imgpath': data['imgpath'],
            'size': size
            }
                
        return img, kpts, mask, meta
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self):
        data = []
        for imgId in self.imgIds:
            data.extend(self.load_data(imgId))
        return data
    
    def load_data(self, imgId):
        img_dict = self.coco.loadImgs(imgId)[0]
        width = img_dict['width']
        height = img_dict['height']
        
        annIds = self.coco.getAnnIds(imgIds=imgId)
        anns = self.coco.loadAnns(annIds)
        data = []
        for ann in anns:
            path = self.path_from_Id(imgId)
            kpts = torch.tensor(ann['keypoints']).float().reshape(-1, 3)
            bbox = dialate_boxes([ann['bbox']], s=self.scale_factor)[0]
            rle  = self.coco.annToRLE(ann)
            size = max(ann['bbox'][2:])
            
            data.append({
                'imgpath': path,
                'bbox': bbox,
                'keypoints': kpts,
                'rle': rle,  # to save memory, we store rle and convert to mask on the fly 
                'size': size
            })
            
        return data
            
    def path_from_Id(self, imgId):
        img_dict = self.coco.loadImgs(imgId)[0]
        filename = img_dict['file_name']
        path = os.path.join(self.root, filename)
        return path

class MaskRCNN_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for MaskRCNN, including instance level detection and instance segmentation tasks. 
    Should return input, target, where input is a image tensor, target includes features of N bounding boxes in the image (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
    Since data are in COCO format, this class utilize
    COCO API to do most of the dataloading.
    """
    def __init__(self, root, annfile, scale_factor=0.25, output_size=256, transform=None):
        self.root = root
        self.coco = COCO(annfile)
        self.imgIds = self.coco.getImgIds(catIds=1)
        self.imgIds.sort()
        
        self.scale_factor = scale_factor
        self.output_size = output_size
        self.transform = transform
        self.data = self.get_data()
        
    def __getitem__(self, index):
        data = self.data[index]

        # input image
        img = cv2.imread(data['imgpath'])
        img = cv2.resize(img, (self.output_size, self.output_size))
        img = torch.as_tensor(img, dtype=torch.float32)
        
        num_masks = len(data['bboxes'])
        bboxes = torch.zeros([num_masks, 4], dtype=torch.float32)
        masks = torch.zeros([num_masks, self.output_size, self.output_size])
        for i, bbox in enumerate(data['bboxes']):
            # bounding box
            x, y, w, h = bbox
            bboxes[i] = torch.tensor([x, y, x+w, y+h])

            # mask
            mask = decode(data['rles'][i])
            mask = (mask > 0).astype(np.uint8)
            mask = cv2.resize(mask, (self.output_size, self.output_size))
            mask = torch.from_numpy(mask)
            masks[i] = mask
            # mask = torch.tensor(mask).long()
            # if i == 0:
            #     masks = mask
            # else:
            #     masks = torch.cat((masks, mask), 0)
        
        # target
        target = dict()
        target['boxes'] = bboxes
        target['labels'] = torch.ones((num_masks, ), dtype=torch.int64) # label is 1
        target['masks'] = masks
        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self):
        data = []
        for imgId in self.imgIds:
            data.extend(self.load_data(imgId))
        return data
    
    def load_data(self, imgId):
        img_dict = self.coco.loadImgs(imgId)[0]
        # width = img_dict['width']
        # height = img_dict['height']
        
        annIds = self.coco.getAnnIds(imgIds=imgId)
        anns = self.coco.loadAnns(annIds)
        path = self.path_from_Id(imgId)
        data = {'imgpath': path,
                'bboxes': [], 
                'keypoints': [],
                'rles': []
        }
        for i, ann in enumerate(anns):
            
            kpts = torch.tensor(ann['keypoints']).float().reshape(-1, 3)
            bbox = dialate_boxes([ann['bbox']], s=self.scale_factor)[0]
            kpts = np.array(kpts)
            bbox = np.array(bbox)
            rle  = self.coco.annToRLE(ann)
            
            data['bboxes'].append(bbox)
            data['keypoints'].append(kpts)
            data['rles'].append(rle)

        data['bboxes'] = torch.tensor(data['bboxes'])
        data['keypoints'] = torch.tensor(data['keypoints'])
        return [data]
            
    def path_from_Id(self, imgId):
        img_dict = self.coco.loadImgs(imgId)[0]
        filename = img_dict['file_name']
        path = os.path.join(self.root, filename)
        return path

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
    normalize_out = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    dataset = MaskRCNN_Dataset(root=root, annfile=annfile, scale_factor=0.25, transform=normalize_out)
    # print(dataset.data[0])

    print('Start from here: ', dataset[0][1]['masks'].size())
    # torch.save(dataset[0][-1]['masks'], 'file.pt')

    # import pandas as pd

    # t = dataset[0][-1]['masks'] #dummy data

    # t_np = t.numpy() #convert to Numpy array
    # df = pd.DataFrame(t_np) #convert to a dataframe
    # df.to_csv("testfile.csv",index=False) #save to file