
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import copy
from pycocotools import mask as coco_mask

import datasets.transforms as T



class CrowdHumanDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_set, img_folder, ann_file, transforms, return_masks, show=False):
        super(CrowdHumanDetection, self).__init__(img_folder, ann_file)
        self.show = show
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        if image_set == "train":
            self._filter_imgs()
        
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths. Modified from mmdet"""
        dataset_len = len(self.ids)
        valid_index = []
        for i in range(dataset_len):
            id = self.ids[i]
            info = self.coco.loadImgs(id)[0]
            if min(info['width'], info['height']) >= min_size:
                anno = self._load_target(id)
                anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
                boxes = [obj["bbox"] for obj in anno]
                boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                if boxes.shape[0] > 0:
                    valid_index.append(i)
        self.ids = [self.ids[i] for i in valid_index]


    def __getitem__(self, idx):
        img, target = super(CrowdHumanDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        ori_img = copy.deepcopy(img)
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self.show:
            target['ori_img'] = ori_img
        target['labels'] = target['labels'] * 0
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_crowdhuman_transforms(image_set, training_schedule):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [800, 768, 736, 704, 672, 640, 608, 576, 544, 512, 480]
    scales = [800, 768, 736, 704, 672, 640]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1500),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1500),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_crowdhuman(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "CrowdHuman_train", root / "annotations" / 'train.json'),
        "val": (root / "CrowdHuman_val", root / "annotations" / 'val.json'),
    }
    
    img_folder, ann_file = PATHS[image_set]
    show = False
    if image_set == 'val' and args.draw_pic:
        show = True
    dataset = CrowdHumanDetection(image_set, img_folder, ann_file, transforms=make_crowdhuman_transforms(image_set, args.training_schedule), return_masks=False, show=show)
    return dataset
