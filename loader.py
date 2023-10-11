import os
import cv2
import json
import torch
import random
import numpy as np
from utils.util import smoothing_mask, total_size
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.augmentations import Transform
from sklearn.model_selection import StratifiedShuffleSplit


class ImageTransform():
    def __init__(self):
        pass
    
    def __call__(self):
        pass


class AppleDataset(Dataset):
    """
    Surface Defective Apple Dataset
    """
    def __init__(self, mode, data_path, img_size=(512, 512), transform=None, evaluation=None):
        self.data_path = data_path
        self.mode = mode
        self.num_classes = 5
        self.img_size = img_size
        self.dataset = self.load_data()
        self.transform = transform
        self.evaluation = evaluation
        
        n = len(self.dataset)
        # split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=44) 
        # # labels = [int(sample['class']) for sample in self.dataset]
        # labels = [sample['class'] for sample in self.dataset]
        

        # for train_idx, valid_idx in split.split(self.dataset, labels):
        #     self.train_set = [self.dataset[i] for i in train_idx] 
        #     self.valid_set = [self.dataset[i] for i in valid_idx]
            
        # if mode == 'train':
        #     self.dataset = self.train_set
        # elif mode == 'valid':
        #     self.dataset = self.valid_set

        
        """Use for object detection splitting"""        
        if mode == 'train':
            self.dataset = self.dataset[:int(n*0.8)]
            
        elif mode == 'valid':
            self.dataset = self.dataset[int(n*0.8):]
            # random.shuffle(self.dataset)
        
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, idx):
        annotation, image_path, image = self.get_annotation(idx)
        sum_size = 1
        height, width = image.size
        
        if self.evaluation:
            return image, image_path, annotation
        
        # if self.transform:
        #     image = self.transform(image)
        
        # target_size = self.img_size[0], self.img_size[1]
        mask = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
        area = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
        
        target = np.array(annotation)
        boxes = target[:, :8] if target.shape[0]!=0 else None
        labels = target[:, 8] if target.shape[0]!=0 else None
        
        # Apply transform on `image`, `boxes`, `labels`
        image, boxes, labels = self.transform(image, boxes, labels)
        
        
        # Recompute the coordinate when image size changes
        # if boxes is not None:
            # target_h, target_w = self.img_size[0], self.img_size[1]
            # new_wh = np.array([target_w / width, target_h / height]) # rescaling factor
            # boxes = boxes * np.tile(new_wh, 4)
            
        # labels = labels.astype(np.int32)
        
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
    
    
    def load_data(self):
        # read ground truth json file
        with open(os.path.join(self.data_path, 'ground-truth','new_gt_multi.json')) as f:
            data = json.load(f)
            # data = [sample for sample in data if sample['class'] == 1]
            data = [sample for sample in data if 5 not in sample['class_id']]
        return data


    def get_annotation(self, idx):
        sample = self.dataset[idx]
        
        image_path = os.path.join(self.data_path, 'images', f"{sample['name']}.jpg")
        # class_id = int(sample['class'])
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        temp_boxes = sample['crop_coordinates_ratio']
        class_ids = sample['class_id']
        annotations = []
        
        # convert format from [cx, cy, w, h] -> [x1, y1, x2, y2, x3, y3, x4, y4, class_id]
        for box, class_id in zip(temp_boxes, class_ids):
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

            annotations.append([x1, y1, x2, y2, x3, y3, x4, y4, class_id])
                   
        return annotations, image_path, image 



class SODAD(Dataset):
    def __init__(self, mode, data_path, img_size = (512, 512), transform=None, evaluation=None):
        self.data_path = data_path
        self.img_size = img_size
        self.transform = transform
        self.evaluation = evaluation

        if mode == 'train':
            self.dataset, self.num_classes = self.load_data(mode)
        elif mode == 'valid':
            self.dataset, self.num_classes = self.load_data(mode)
        
        self.image_names = [*self.dataset]
        
            
    def __len__(self):
        return len(self.image_names)
    
    
    def __getitem__(self, idx):
        annotation, image_path, image = self.get_annotation(idx)
        
        mask = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)
        area = np.zeros((self.img_size[0], self.img_size[1], self.num_classes), dtype=np.float32)

        target = np.array(annotation)
        boxes = target[:, :8] if target.shape[0]!=0 else None
        labels = target[:, 8] if target.shape[0]!=0 else None
        
        # Apply transform on `image`, `boxes`, `labels`
        image, boxes, labels = self.transform(image, boxes, labels)
        
        num_obj = len(boxes) if boxes is not None else 1
        sum_size = total_size(boxes)
        
        for box, label in zip(boxes, labels):
            mask, area = smoothing_mask(mask, area, box, sum_size/num_obj, label)
        
        image, mask, area = self.annotation_transform(np.array(image), mask, area, self.img_size[0], self.img_size[1])

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask.astype(np.float32))
        area = torch.from_numpy(area.astype(np.float32))
        sum_size = torch.from_numpy(np.array([sum_size], dtype=np.float32))
        
        
        return image, mask, area, sum_size
    
    
    def load_annotations(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        
        filtered_annotations = [annotation for annotation in data.get('annotations', []) if annotation.get('ignore') == 0]
        filtered_data = data.copy()
        filtered_data['annotations'] = filtered_annotations
        
        return filtered_data
    
    def load_data(self, mode):
        if mode == 'train':
            annotations_file = 'train.json'
        elif mode == 'valid':  
            annotations_file = 'val.json'
        else:
            raise ValueError(f'Invalid mode: {mode}')  
        
        annotations = self.load_annotations(os.path.join(self.data_path,
                                                         'Annotations', annotations_file))
        
        num_classes = len(set([annotation['category_id'] for annotation in annotations['annotations']]))
        
        # Mapping new label for each category
        category_mapping = {org_label: org_label - 1 for org_label in range(1, num_classes+1)}
        for annotation in annotations['annotations']:
            annotation['category_id'] = category_mapping[annotation['category_id']]
        
        temp_data = {}
        
        # Save annotation and image info in a list
        image_id_to_name = {image['id']: image['file_name'] for image in annotations['images']}
        
        # Save annotation and image info in a dictionary
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            image_name = image_id_to_name.get(image_id)
            if image_name is not None:
                if image_name not in temp_data:
                    temp_data[image_name] = []
                annotation_data = {
                    'image_id': image_id,
                    'bbox': annotation['bbox'],
                    'category_id': annotation['category_id']
                }
                temp_data[image_name].append(annotation_data)
        
        
        return temp_data, num_classes
    
    
    def get_annotation(self, idx):
        image_name = self.image_names[idx]
        sample = self.dataset[image_name]
        image_path = os.path.join(self.data_path, 'Images', image_name)
        image = Image.open(image_path).convert('RGB')
        
        target_annotation = []
        for annot in sample:
            bbox = annot['bbox']
            category_id = annot['category_id']
            x, y, w, h = bbox
            bbox_new = x, y, x+w, y, x+w, y+h, x, y+h
            target_annotation.append(list(bbox_new) + [category_id])
        
        return target_annotation, image_path, image


    def annotation_transform(self, image, mask, area, height, width):
        resize_img = cv2.resize(image, (width, height))
        resized_mask = cv2.resize(mask, (width, height))
        resized_area = cv2.resize(area, (width, height))
        return resize_img, resized_mask, resized_area

if __name__ == '__main__':
    
    # """For Apple Dataset"""
    # transform_train = Transform(is_train=True, size=(512, 512))
    # appledata = AppleDataset(mode='train',
    #                          data_path='/root/data/apple/cropped-apple-bb/',
    #                          img_size=(512, 512),
    #                          transform=transform_train)
    
    # apple_loader = DataLoader(appledata, batch_size=2, shuffle=True)
    # for batch in apple_loader:
    #     images, masks, areas, total_sizes = batch
    #     print(total_sizes)
    #     # import pdb; pdb.set_trace()
    
    
    """For SODA-D Dataset"""
    transform_train = Transform(is_train=True, size=(512, 512))
    appledata = SODAD(mode='valid',
                      data_path='/root/data/SODA-D/',
                      img_size=(512, 512),
                      transform=transform_train)
    import pdb; pdb.set_trace()
    soda_loader = DataLoader(appledata, batch_size=2, shuffle=True)

    for batch in soda_loader:
        images, masks, areas, total_sizes = batch
        print(total_sizes)
        # import pdb; pdb.set_trace()