import os
import cv2
import json
import torch
import random
import numpy as np
from utils.util import smoothing_mask, total_size, APPLE_CLASSES
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.augmentations import Transform
import json


class ListAppleDataset(Dataset):
    """ Apple Defects Detection Dataset.
    
    Args:

    Returns:
    
    Raises:

    """
    def __init__(self, mode, dataset, root, img_size=(512, 512), transform=None, evaluation=None):
        self.dataset = dataset
        self.mode = mode
        self.img_size = img_size
        self.transform = transform
        self.evaluation = evaluation

        if self.dataset == 'split':
            self.data_folder = os.path.join(root, mode)
            self.load_version_2()
            
        if self.dataset == 'version-1':
            self.data_folder = root
            self.load_version_1()
            
            # splitting into training and validation set
            n = len(self.data)
        
            random.Random(44).shuffle(self.data)  # random.shuffle is an in-place operation
            if self.mode == 'train':
                self.data = self.data[:int(n*0.8)]
            
            elif self.mode == 'valid':
                self.data = self.data[int(n*0.8):]
                
        elif self.dataset in ['version-2', 'version-3']:
            self.data_folder = root
            self.load_version_2()
            
            # splitting into training and validation set
            n = len(self.data)
        
            random.Random(44).shuffle(self.data)  # random.shuffle is an in-place operation
            if self.mode == 'train':
                self.data = self.data[:int(n*0.8)]
            
            elif self.mode == 'valid':
                self.data = self.data[int(n*0.8):]
        
        # else:
        #     raise ValueError(f'Dataset {self.dataset} not found.')
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        if self.dataset == 'version-1':
            annotation, image_path, image = self.get_target(idx)
        elif self.dataset in ['version-2', 'version-3', 'split']:
            annotation, image_path, image = self.get_target(self.data[idx])
        
        sum_size = 1
        height, width, _ = image.shape
        
        if self.evaluation:
            return image, image_path, annotation
        
        mask = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
        area = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
        
        target = np.array(annotation)
        boxes = target[:, :8] if target.shape[0]!=0 else None
        labels = target[:, 8] if target.shape[0]!=0 else None
        
        # apply transform on `image`, `boxes`, `labels`
        image, boxes, labels = self.transform(image, boxes, labels)
        
        num_obj = len(boxes) if boxes is not None else 1
        sum_size = total_size(boxes)
        
        for box, label in zip(boxes, labels):
            mask, area = smoothing_mask(mask, area, box, sum_size/num_obj, label)
        
        image, mask, area = self.annotation_transform(np.array(image), mask, area, self.img_size[1], self.img_size[1])
        
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.astype(np.float32))
        area = torch.from_numpy(area.astype(np.float32))
        sum_size = torch.from_numpy(np.array([sum_size], dtype=np.float32))
        
        return image, mask, area, sum_size
        
        
    def annotation_transform(self, image, mask, area, height, width):
        resize_img = cv2.resize(image, (width, height))
        resized_mask = cv2.resize(mask, (width, height))
        resized_area = cv2.resize(area, (width, height))
        return resize_img, resized_mask, resized_area
    
    
    def load_version_1(self, data_folder):
        with open(os.path.join(data_folder, 'groundtruth.json')) as f:
            data = json.load(f)
        
        filtered_data = [item for item in data if set(item['class_id']) in ({1}, {3}, {1, 3})]

        label_mapping = {1: 0, 3: 1, 5: 2}  # 0: scratch; 1: blight; 2: normal
        
        for data in filtered_data:
            data['class_id'] = [label_mapping[i] for i in data['class_id']]
        
        num_class = set()
        
        for item in filtered_data:
            num_class.update(item['class_id'])
        
        self.num_classes = len(num_class)
        self.data = filtered_data

    def load_version_2(self):
        self.target_transform = None
        self._anno_path = os.path.join(self.data_folder, 'labelTxt', '%s.txt')
        self._coco_imgpath = os.path.join(self.data_folder, 'images', '%s.jpg')

        dataset_list = os.path.join(self.data_folder, 'image_list.txt')
        dataset_list = open(dataset_list, "r")

        annot_path = os.path.join(self.data_folder, 'labelTxt')
        annot_list = os.listdir(annot_path)

        missing_list = []

        num_classes = set()

        for item in annot_list:
            try:
                open_annot = open(os.path.join(annot_path, item), "r")
                label = open_annot.read().split()[-2]
                num_classes.add(label)
            except:
                missing_list.append(item.split('.')[0])  # remove sample without annotation

        ids = []
        for line in dataset_list.read().splitlines():
            ids.append(line)

        self.data = [item for item in ids if item not in missing_list]

        self.num_classes = len(num_classes)

        self.data = sorted(self.data)
        
    
    def get_target(self, idx):
        if self.dataset == 'version-1':
            sample = self.data[idx]
            image_name = sample['name']
            image_path = os.path.join(self.data_folder, 'images', f'{image_name}.jpg')
            image = cv2.imread(image_path)[:,:,::-1]
            height, width, _ = image.shape
            
            temp_boxes = sample['crop_coordinates_ratio']
            class_names = sample['class_name']
            annotation = []
            
            for box, class_name in zip(temp_boxes, class_names):
                cx, cy, w, h = box
                x1 = int((cx - w / 2) * width)
                y1 = int((cy - h / 2) * height)
                x2 = int((cx + w / 2) * width)
                y2 = int((cy - h / 2) * height)
                x3 = int((cx + w / 2) * width)
                y3 = int((cy + h / 2) * height)
                x4 = int((cx - w / 2) * width)
                y4 = int((cy + h / 2) * height)
                
                # perform boundary checks
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                x3 = max(0, min(x3, width - 1))
                y3 = max(0, min(y3, height - 1))
                x4 = max(0, min(x4, width - 1))
                y4 = max(0, min(y4, height - 1))

                annotation.append([x1, y1, x2, y2, x3, y3, x4, y4, APPLE_CLASSES.index(class_name)])
            
        if self.dataset in ['version-2', 'version-3', 'split']:
            
            image_path = self._coco_imgpath % (idx)
            
            image = cv2.imread(image_path)[:,:,::-1]
            size = image.shape[0]
            
            if 'test' in self.mode:
                return [], image_path, image
            
            anno = open(self._anno_path % idx, "r")
            anno = anno.read().splitlines()
            
            annotation = []
            
            for _anno in anno:
                _anno_temp = _anno.split(' ')
                # _anno = [float(float(x)/size) for x in _anno_temp[:8]]
                _anno = [x for x in _anno_temp[:8]]
                _anno.append(APPLE_CLASSES.index(_anno_temp[-2]))
                annotation.append(_anno)

        # else:
        #     raise ValueError(f'Dataset {self.dataset} not found.')
        
        return annotation, image_path, image
    

if __name__ == '__main__':

    """For Apple Dataset"""
    transform_train = Transform(is_train=True, size=(512, 512))
    appledata = ListAppleDataset(mode='train',
                                 dataset = 'split',
                                 root='/mnt/data/dataset/apple-defects/train-test-split/',
                                 img_size=(512, 512),
                                 transform=transform_train)

    apple_loader = DataLoader(appledata, batch_size=4, shuffle=True)
    
    for batch in apple_loader:
        images, masks, areas, total_sizes = batch
        
        