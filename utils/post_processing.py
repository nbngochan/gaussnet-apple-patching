import torch
import numpy as np
import cv2

def get_center_point_contour(output, thresh, scale, org_size):
    org_w, org_h = org_size
    height, width, num_classes = output.shape

    c_mask = (output > thresh).astype(np.uint8)
    
    results = []
    
    for cls in range(num_classes):
        mask = c_mask[:, :, cls]
        
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    
        for k in range(1, nLabels):
            size = stats[k, cv2.CC_STAT_AREA]
            
            # make segmentation map
            segmap = np.zeros_like(mask, dtype=np.uint8)
            segmap[labels==k] = 255
            
            contours, _ = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            for cnt in contours:
                # compute the center of the contour
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                x, y, w, h = cv2.boundingRect(cnt)
                x = x * org_w / width
                y = y * org_h / height
                w = w * org_w / width
                h = h * org_h / height
                
                # Calculate the center of the bounding box
                center_x = x + w / 2
                center_y = y + h / 2
                
                # Apply the scale factor from the center
                scaled_w = w * scale
                scaled_h = h * scale
                
                # Adjust the top-left coordinates based on the scaled width and height
                x = center_x - scaled_w / 2
                y = center_y - scaled_h / 2
                
                box = [(x, y), (x + scaled_w, y), (x + scaled_w, y + scaled_h), (x, y + scaled_h)]
                
                results.append({"conf" : max(0.0, min(1.0, output[cy, cx, cls])),
                               "rbox" : box, 
                               "label" : cls})

    return results
        
# def get_center_point_contour(output, thresh, scale, org_size):
#     org_w, org_h = org_size
#     # import pdb; pdb.set_trace()
#     height, width, num_classes = output.shape

#     c_mask = (output > thresh).astype(np.uint8)
    
#     results = []
    
#     for cls in range(num_classes):
#         mask = c_mask[:, :, cls]
        
#         contours, _ = cv2.findContours(mask.astype(np.uint8),
#                                     cv2.RETR_EXTERNAL,
#                                     cv2.CHAIN_APPROX_SIMPLE)
        
#         # Extract bounding box coordinates from contours
#         bounding_boxes = []
        
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             x1, y1 = x, y
#             x2, y2 = x + w, y
#             x3, y3 = x + w, y + h
#             x4, y4 = x, y + h
            
#             x1 = x1 * org_w / width
#             y1 = y1 * org_h / height
#             x2 = x2 * org_w / width
#             y2 = y2 * org_h / height
#             x3 = x3 * org_w / width
#             y3 = y3 * org_h / height
#             x4 = x4 * org_w / width
#             y4 = y4 * org_h / height
            
#             boxes = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
#             # bounding_boxes.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            
#             if boxes:
#                 results.append({"rbox" : boxes,
#                                 "label" : cls})

#     return results
  
        
def smoothing(heat, kernel=3):
    pad = (kernel - 1) // 2
    
    heat = heat.permute(0, 3, 1, 2)  # [B, C, H, W]
    
    heat = torch.nn.functional.avg_pool2d(heat,
                                         (kernel, kernel),
                                         stride=1,
                                         padding=pad)
    
    heat = heat.permute(0, 2, 3, 1)
    
    return heat


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    
    heat = heat.clone().permute(0, 3, 1, 2)  # [B, C, H, W]
    
    hmax = torch.nn.functional.max_pool2d(heat,
                                         (kernel, kernel),
                                         stride=1,
                                         padding=pad)
    
    keep = (hmax == heat).float()
    
    peak = heat * keep
    
    peak = peak.permute(0, 2, 3, 1)
    
    return peak


def topk(scores, k=40):
    batch, height, width, cat = scores.size()
    
    scores = scores.permute(0, 3, 1, 2)
    
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)
    
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    
    return topk_ys, topk_xs



