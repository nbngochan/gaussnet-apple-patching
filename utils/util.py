import numpy as np
import cv2
import os
import torch
import json
import PIL.Image as Image
from sklearn.metrics import auc
import math

APPLE_CLASSES = ['blight', 'scratch', 'normal']
COLORS = [(0, 128, 0),  # Green
            (255, 0, 0),  # Red
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (128, 0, 128)]  # Purple


def is_nan(value):
    return math.isnan(float(value))


def im2double(img):
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype("float") - min_val) / (max_val - min_val)
    return out


def gaussian(shape, sigma=10):
    """
    Gaussian kernel for density map generation
    """
    m = (shape[0] - 1) / 2.  # center point m, n
    n = (shape[1] - 1) / 2.  
    y, x = np.ogrid[-m:m+1, -n:n+1]  # mesh of coordinates
    k = np.exp(-(x*x + y*y) / (2*sigma*sigma))  # gaussian kernel at each point
    k = k / k.max()  # normalize
    k[k<0.0001] = 0  
    return k


def tricube(size, factor=3):
    """
    Tricube kernel for density map generation
    """
    s = (size - 1) / 2
    m, n = [(ss - 1.) / 2. for ss in [size, size]]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    
    k_x = (1 - (abs((x/s)**3)))**factor
    k_y = (1 - (abs((y/s)**3)))**factor
    
    k = k_y.dot(k_x)   
    
    k = k / k.max()  
    return k


def quartic(shape, factor=2):
    m, n = [(s - 1) / 2. for s in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    k = 15/16*(1 - (x*x + y*y))**factor
    k = k / k.max()
    k[k<0.0001] = 0
    
    return k


def triweight(size, factor=3):
    s = (size - 1) / 2
    m, n = [(s - 1.) / 2. for s in [size, size]]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    
    k_x = ((1 - (abs(x/s)**2))**factor)*(35 - 30*((abs(x/s))**2) + 3*abs((x/s))**4)
    k_y = ((1 - (abs(y/s)**2))**factor)*(35 - 30*((abs(y/s))**2) + 3*abs((y/s))**4)
    
    k = k_y.dot(k_x)
    k = k / k.max()
    
    return k


def colormap(density_map, image):
    heatmap = cv2.applyColorMap((density_map * 255).astype('uint8'), cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = cv2.addWeighted(image[:, :, ::-1], 0.5, heatmap, 0.5, 0)
    return overlay

diameter = 50
kernel_map = gaussian((diameter, diameter), 15)
kernel_poly = np.array([[0, 0], [0, diameter], [diameter, diameter], [diameter, 0]], np.float32)  # source quadrilateral coordinates

# diameter = 800
# kernel_map = tricube(diameter, 3)
# kernel_poly = np.array([[0, 0], [0, diameter], [diameter, diameter], [diameter, 0]], np.float32)  # source quadrilateral coordinates


def l2_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def l1_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.abs(x2 - x1) + np.abs(y2 - y1)

def smoothing_mask(mask, area, box, size, label):
    H, W = mask.shape[:2]
    if type(size) is tuple:
        size = size[0] * size[1]
    
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    
    mask_w = max(l2_distance((x1, y1), (x2, y2)), l2_distance((x3, y3), (x4, y4)))
    mask_h = max(l2_distance((x1, y1), (x4, y4)), l2_distance((x2, y2), (x3, y3)))
    
    weight_mask = np.zeros((H, W), dtype=np.float32)
    mask_area = max(1, mask_w * mask_h)
    img_area = size

    transform_mat = cv2.getPerspectiveTransform(kernel_poly, box.astype(np.float32).reshape(4, 2))
    dst = cv2.warpPerspective(kernel_map, transform_mat, (W, H))

    mask_area = (img_area / mask_area)
    
    weight_mask = cv2.fillPoly(weight_mask, pts = box.astype(np.int32).reshape(1, 4, 2), color=mask_area)
    mask[:, :, label] = np.maximum(mask[:, :, label], dst)
    area[:, :, label] = np.maximum(area[:, :, label], weight_mask)
    
    return mask, area

DATA_PATH = '/root/data/apple/cropped-apple-bb/'
def get_annotation(img_name):
    with open(os.path.join(DATA_PATH, 'ground-truth','ground_truth.json')) as f:
        data = json.load(f)
    for item in data:
        if item['name'] == img_name:
            image_path = item['cropped_image_path']
            class_id = int(item['class'])  # 0: defective, 1: normal
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            
            temp_boxes = item['crop_coordinates_ratio']
            num_objs =  len(temp_boxes)
            target_boxes = []

            for box in temp_boxes:
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
                
                target_boxes.append([x1, y1, x2, y2, x3, y3, x4, y4, class_id])
    
    return target_boxes, image_path, img

def total_size(boxes):
    size_sum = 0
    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        mask_w = max(l2_distance((x1, y1), (x2, y2)), l2_distance((x3, y3), (x4, y4)))
        mask_h = max(l2_distance((x1, y1), (x4, y4)), l2_distance((x2, y2), (x3, y3)))
        size_sum += mask_w * mask_h
    return size_sum
    
    
def get_mask(img_name):
    target_boxes, image_path, img = get_annotation(img_name)
    sum_size = 1
    height, width = img.size
    
    mask = np.zeros((height, width, 2), dtype=np.float32)
    area = np.zeros((height, width, 2), dtype=np.float32)
        
    target = np.array(target_boxes)
    
    boxes = target[:, :8]
    labels = target[:, 8]
    
    # target_wh = np.array([width, height], dtype=np.float32)
    # boxes = (boxes.clip(0,1) * np.tile(target_wh, 4)).astype(np.float32)
    
    labels = labels.astype(np.int32)
    
    num_obj = max(len(boxes), 1)
    sum_size = total_size(boxes)
    
    for box, label in zip(boxes, labels):
        mask, area = smoothing_mask(mask, area, box, sum_size/num_obj, label)
    
    img = np.array(img)
    
    return img, mask, area, sum_size


from torchvision.ops import box_iou

def calculate_mAP(predicted_boxes, predicted_labels, ground_truth_boxes, ground_truth_labels, iou_threshold):
  """
  Calculates the mAP score for each class.

  Args:
    predicted_boxes: A torch tensor of predicted bounding boxes in xyxy format.
    predicted_labels: A torch tensor of predicted labels.
    ground_truth_boxes: A torch tensor of ground truth bounding boxes in xyxy format.
    ground_truth_labels: A torch tensor of ground truth labels.

  Returns:
    A dictionary of mAP scores for each class.
  """

  num_classes = predicted_labels.max().item() + 1
  mAP_scores = {}

  for c in range(num_classes):
    # Filter predicted and ground truth boxes and labels for class c
    class_mask_pred = predicted_labels == c
    class_mask_gt = ground_truth_labels == c
    pred_boxes_c = predicted_boxes[class_mask_pred]
    gt_boxes_c = ground_truth_boxes[class_mask_gt]

    # Calculate IoU between predicted and ground truth boxes
    iou = box_iou(pred_boxes_c, gt_boxes_c)

    # Sort predicted boxes by confidence score (descending order)
    # sorted_indices = torch.argsort(-pred_boxes_c[:, 4])
    # pred_boxes_c = pred_boxes_c[sorted_indices]
    # iou = iou[sorted_indices]

    # Calculate precision and recall
    num_pred = pred_boxes_c.size(0)
    true_positives = torch.zeros(num_pred)
    false_positives = torch.zeros(num_pred)
    total_gt_boxes = gt_boxes_c.size(0)

    # Compute TP and FP for class c in term of IoU threshold
    if isinstance(iou_threshold, float):
        for i in range(num_pred):
            if iou.size(1) != 0:
                if iou[i].max().item() >= iou_threshold:
                    true_positives[i] = 1
                    iou[i, iou[i].argmax()] = -1
                else:
                    false_positives[i] = 1

                
    elif iou_threshold is None:
        thresholds = torch.arange(0.5, 1.0, 0.05)
        for i in range(total_gt_boxes):
            max_iou = torch.tensor(-1.0)
            iou_i = iou[:, i]
            if iou_i.numel() > 0:
                
                max_iou, max_idx = iou_i.max(dim=0)
                for threshold in thresholds:
                    mask = (max_iou >= threshold) & (max_idx == i)
                    num_masked = mask.sum().item()
                    if num_masked > 0:
                        true_positives[mask] = 1
                        max_iou[mask] = -1
                false_positives += (max_iou >= 0).float()


    cum_true_positives = torch.cumsum(true_positives, dim=0)
    cum_false_positives = torch.cumsum(false_positives, dim=0)
    precision = cum_true_positives / (cum_true_positives + cum_false_positives + 1e-10)
    recall = cum_true_positives / total_gt_boxes

    # Calculate Average Precision (AP)
    recall = torch.cat((torch.zeros(1), recall, torch.ones(1)))
    precision = torch.cat((torch.zeros(1), precision, torch.zeros(1)))
    for i in range(precision.size(0) - 1, 0, -1):
      precision[i - 1] = torch.max(precision[i - 1], precision[i])

    # Calculate mAP score for class c
    ap = torch.sum((recall[1:] - recall[:-1]) * precision[1:])

    # Save the AP score for class c in the dictionary
    mAP_scores[c] = ap.item()

  return mAP_scores


if __name__ == '__main__':
    kernel_option = {1: 'gaussian', 2: 'tricube'}
    option = int(input("Enter choice: "))
    
    if kernel_option[option] == 'gaussian':
        diameter = 50
        kernel_map = gaussian((diameter, diameter), 12)
        kernel_poly = np.array([[0, 0], [0, diameter], [diameter, diameter], [diameter, 0]], np.float32)  # source quadrilateral coordinates
    else:
        diameter = 800
        kernel_map = tricube(diameter, 3)
        kernel_poly = np.array([[0, 0], [0, diameter], [diameter, diameter], [diameter, 0]], np.float32)  # source quadrilateral coordinates
    
    
    image, mask, area, sum_size = get_mask('23945065_20211025_181018_151.jpg')
    import pdb; pdb.set_trace()
    
    print(sum_size)