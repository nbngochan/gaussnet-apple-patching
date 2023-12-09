import argparse
import cv2
import numpy as np
import os
import torch
import time

from basenet.model import Model_factory
from loader import ListAppleDataset
from utils.post_processing import get_center_point_contour
from utils.post_processing import nms, topk, smoothing
from utils.util import APPLE_CLASSES, COLORS
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', 1)

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, help='Root directory of dataset')
parser.add_argument('--dataset', type=str, help='Dataset version name')
parser.add_argument('--checkpoint', type=str, help='Tranined checkpoint')
parser.add_argument('--experiment', default=1, type=int, help='number of experiment')
parser.add_argument('--input_size', default=512, type=int, help='input size')
parser.add_argument('--c_thresh', default=0.1, type=float, help='threshold for center point')
parser.add_argument('--backbone', type=str, default='hourglass104_MRCB_cascade', 
                        help='[hourglass104_MRCB_cascade, hourglass104_MRCB, hhrnet48, DLA_dcn, uesnet101_dcn]')
parser.add_argument('--kernel', default=3, type=int, help='kernel of max-pooling for center point')
parser.add_argument('--scale', default=1, type=float, help='scale factor')

arg = parser.parse_args()
print(arg)


result_img_path = 'img-out/'
if not os.path.exists(result_img_path):
    os.makedirs(result_img_path)
    
"""Data Loader"""
mean = (0.485, 0.456, 0.406)
var = (0.229, 0.224, 0.225)

test_dataset = ListAppleDataset('valid', arg.dataset, arg.root,
                            arg.input_size, transform=None, evaluation=True)
   
"""Network Backbone"""
NUM_CLASSES = {'apple_1': 1, 'apple_2': 2, 'sodad': 9, 'version-2': 3}
num_classes = NUM_CLASSES[arg.dataset]
model = Model_factory(arg.backbone, num_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.cuda()

checkpoint = torch.load(arg.checkpoint)
model.load_state_dict(checkpoint['model'])
checkpoint = None

_ = model.eval()

dest_dir = f'/data/apple/results/{arg.checkpoint}_{arg.experiment}/Task1/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

filedict = {}
for cls in APPLE_CLASSES:
    fd = open(os.path.join(dest_dir, f'Task1_{cls}.txt'), 'a')
    filedict[cls] = fd
    
obj_write = "%s %.3f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" #imgname, conf, box*8

sum_total_time = 0
sum_infer_time = 0
sum_post_time = 0

num = 0

all_preds = []
all_targets = []


# for idx in range(len(test_dataset)):
for idx in range(50):
    image, image_path, ground_truth = test_dataset.__getitem__(idx)
    image_name = os.path.basename(image_path).split('.')[0]
    print(image_name)    
    org_h, org_w, _ = image.shape

    h, w = arg.input_size, arg.input_size
    image = cv2.resize(np.array(image), (w, h))
    
    x = image.copy()
    x = x.astype(np.float32)
    x /= 255
    x -= mean
    x /= var
    x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
    x = x.unsqueeze(0)
    
    with torch.no_grad():
        x = x.to(device)
        
        t1 = time.time()
        out = model(x)
        
        if 'gaussnet' in arg.backbone:
            out = out[1]
        
        out = smoothing(out, arg.kernel)
        peak = nms(out, arg.kernel)
        c_ys, c_xs = topk(peak, k=2000)
    
        
    x = x[0].cpu().detach().numpy()
    out = out[0].cpu().detach().numpy()
    c_xs = c_xs[0].int().cpu().detach().numpy()
    c_ys = c_ys[0].int().cpu().detach().numpy()
    
    x = x.transpose(1, 2, 0)
    # x *= var
    # x += mean
    # x *= 255
    x = x.clip(0, 255).astype(np.uint8)
    
    t2 = time.time()
    
    results = get_center_point_contour(out, arg.c_thresh, arg.scale, (org_w, org_h))
    
    t3 = time.time()
    
    _img = image.copy()
    
    
    # Predicted value
    pred_boxes = []
    pred_labels = []
    
    for result in results:
        box = result['rbox']
        label = result['label']
        color = COLORS[label]

        target_wh = np.array([[w/org_w, h/org_h]], dtype=np.float32)
        box = box * np.tile(target_wh, (4,1))  # shape: (4, 2) [x1, y1], [x2, y2], [x3, y3], [x4, y4]
        
        _img = cv2.drawContours(_img, [box.astype(np.int0)], -1, color, 2)
        
        xmin = np.min(box[:, 0])
        ymin = np.min(box[:, 1])
        xmax = np.max(box[:, 0])
        ymax = np.max(box[:, 1])
        
        # box = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        
        pred_boxes.append(box)
        pred_labels.append(label)
        

    preds = {'boxes': torch.tensor(pred_boxes, dtype=torch.float32),
             'scores': torch.tensor([1.0]*len(pred_boxes)) ,
             'labels': torch.tensor(pred_labels)}
    
    all_preds.append(preds)
    
    # Ground truth value
    true_boxes = []
    true_labels = []
    
    for box in ground_truth:
        label = int(box[8])
        
        box = np.array(box[:8], dtype=np.float32).reshape(-1, 2)
        target_wh = np.array([[w/org_w, h/org_h]], dtype=np.float32)
        box = box * np.tile(target_wh, (4,1))
        
        xmin = np.min(box[:, 0])
        ymin = np.min(box[:, 1])
        xmax = np.max(box[:, 0])
        ymax = np.max(box[:, 1])
        
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        
        true_boxes.append(box)
        true_labels.append(label)
        
    targets = {'boxes': torch.tensor(true_boxes, dtype=torch.float32),
               'labels': torch.tensor(true_labels)}
    
    all_targets.append(targets)
    
    
    # Plotting visualization after prediction
    merge_out = np.max(out, axis=-1)
    merge_out = np.clip(merge_out * 255, 0, 255)

    binary = (merge_out > 0.3*255) * 255
    
    merge_out = cv2.applyColorMap(merge_out.astype(np.uint8), cv2.COLORMAP_JET)
    binary = cv2.applyColorMap(binary.astype(np.uint8), cv2.COLORMAP_JET)
    
    merge_out = cv2.resize(merge_out, (w, h))  # image with bounding box prediction
    binary = cv2.resize(binary, (w, h))  # binary image with thresholding
    
    result_img = cv2.hconcat([_img[:, :, ::-1], merge_out, binary])

    cv2.imwrite("%s/%s.jpg" % (result_img_path, image_name), result_img)

import pdb; pdb.set_trace()
# Calculate mAP
metric = MeanAveragePrecision(iou_type="bbox")
metric.update(all_preds, all_targets)
mAP = metric.compute()
print("Mean Average Precision (mAP):", mAP)

