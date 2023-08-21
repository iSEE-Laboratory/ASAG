
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import math
from torchvision.ops.boxes import box_area



def box_union(boxes1: torch.Tensor, boxes2: torch.Tensor):
    '''
    boxes1: (N, 4), xyxy
    boxes2: (M, 4), xyxy
    '''
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    max_side = torch.maximum(rb[..., 0] - lt[..., 0], rb[..., 1] - lt[..., 1]) # [N, M]
    return max_side, lt, rb

@torch.no_grad()
def AgglomerativeClustering(boxes: torch.Tensor, confidences: torch.Tensor=None, distance_threshold=None, threshold=0):
    if len(boxes) < 2:
        return boxes
    while len(boxes) > 1:
        distance_matrix, lt, rb = box_union(boxes, boxes) # [N, N]
        mask = torch.diag(torch.ones(len(boxes), device=boxes.device)).bool()
        distance_matrix[mask] = 100000
        value, index = torch.min(distance_matrix.flatten(), dim=0)
        if distance_threshold is not None and value >= distance_threshold:
            break
        i = torch.div(index, len(boxes), rounding_mode="floor")
        j = index % len(boxes)
        mask = torch.ones(len(boxes), device=boxes.device).bool()
        mask[i] = 0
        mask[j] = 0
        boxes = torch.cat([boxes[mask], torch.cat([lt[i,j], rb[i,j]])[None]])
        if confidences is not None:
            confidences = torch.cat([confidences[mask], confidences[i:i+1]+confidences[j:j+1]])
    if confidences is not None:
        boxes = boxes[confidences>threshold]
    return boxes

# def AgglomerativeClustering(boxes: torch.Tensor, distance_threshold=None):
#     child = [boxes[i: i+1] for i in range(len(boxes))]
#     while len(boxes) > 1:
#         distance_matrix, lt, rb = box_union(boxes, boxes) # [N, N]
#         mask = torch.diag(torch.ones(len(boxes), device=boxes.device)).bool()
#         distance_matrix[mask] = 100000
#         value, index = torch.min(distance_matrix.flatten(), dim=0)
#         if distance_threshold is not None and value >= distance_threshold:
#             return boxes, child
#         i = torch.div(index, len(boxes), rounding_mode="floor")
#         j = index % len(boxes)
#         if i > j:
#             child1 = child.pop(i)
#             child2 = child.pop(j)
#         else:
#             child2 = child.pop(j)
#             child1 = child.pop(i)
#         child.append(torch.cat([child1, child2]))
#         mask = torch.ones(len(boxes), device=boxes.device).bool()
#         mask[i] = 0
#         mask[j] = 0
#         boxes = torch.cat([boxes[mask], torch.cat([lt[i,j], rb[i,j]])[None]])
#     return boxes, child

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou, iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


# this function is robust to normalize/non-normalize
# if anchor_bboxes and targets are both normalized or both non-normalized, the results are correct

def get_deltas(src_boxes, target_boxes):
    """
    Get box regression transformation deltas (dx, dy, dw, dh)
    Args:
        src_boxes (Tensor): source boxes, e.g., object proposals xyxy
        target_boxes (Tensor): target of the transformation, e.g., ground-truth boxes. xyxy
    """
    assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
    assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

    src_widths = src_boxes[..., 2] - src_boxes[..., 0]
    src_heights = src_boxes[..., 3] - src_boxes[..., 1]
    src_ctr_x = src_boxes[..., 0] + 0.5 * src_widths
    src_ctr_y = src_boxes[..., 1] + 0.5 * src_heights

    target_widths = target_boxes[..., 2] - target_boxes[..., 0]
    target_heights = target_boxes[..., 3] - target_boxes[..., 1]
    target_ctr_x = target_boxes[..., 0] + 0.5 * target_widths
    target_ctr_y = target_boxes[..., 1] + 0.5 * target_heights

    dx = (target_ctr_x - src_ctr_x) / src_widths
    dy = (target_ctr_y - src_ctr_y) / src_heights
    dw = torch.log(target_widths / src_widths)
    dh = torch.log(target_heights / src_heights)

    deltas = torch.stack((dx, dy, dw, dh), dim=-1)
    assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
    return deltas

SCALE_CLAMP = math.log(100000.0 / 16)

def apply_deltas(deltas, boxes):
    """
    Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
    Args:
        deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
            deltas[i] represents k potentially different class-specific
            box transformations for the single box boxes[i].
        boxes (Tensor): boxes to transform, of shape (N, 4) xyxy
    """
    deltas = deltas.float()  # ensure fp32 for decoding precision
    boxes = boxes.to(deltas.dtype)

    widths = boxes[..., 2] - boxes[..., 0]
    heights = boxes[..., 3] - boxes[..., 1]
    ctr_x = boxes[..., 0] + 0.5 * widths
    ctr_y = boxes[..., 1] + 0.5 * heights

    dx = deltas[..., 0::4]
    dy = deltas[..., 1::4]
    dw = deltas[..., 2::4]
    dh = deltas[..., 3::4]

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=SCALE_CLAMP)
    dh = torch.clamp(dh, max=SCALE_CLAMP)

    pred_ctr_x = dx * widths[..., None] + ctr_x[..., None]
    pred_ctr_y = dy * heights[..., None] + ctr_y[..., None]
    pred_w = torch.exp(dw) * widths[..., None]
    pred_h = torch.exp(dh) * heights[..., None]

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return pred_boxes.reshape(deltas.shape)



