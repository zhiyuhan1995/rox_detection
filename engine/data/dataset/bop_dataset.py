import os
import json
import torch
import torch.utils.data
from PIL import Image
import faster_coco_eval
import faster_coco_eval.core.mask as coco_mask
from .coco_dataset import ConvertCocoPolysToMask
from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register
import numpy as np

import torchvision.transforms as transform

# Initialize COCO mask backend
faster_coco_eval.init_as_pycocotools()
Image.MAX_IMAGE_PIXELS = None

__all__ = ['BopDetection']

@register()
class BopDetection(DetDataset):
    """
    DEIM dataset for BOP-format COCO annotations.
    Directory structure:
        root/
          camera.json
          train|val|test_pbr/
            <seq_id>/
              scene_gt_coco.json
              rgb/
                000000.png, 000001.png, ...
              depth/, mask/, mask_visib/

    Each sample yields one RGB image and its annotations extracted
    from the per-sequence scene_gt_coco.json.
    """
    __inject__ = ['transforms']

    def __init__(self, root: str, split: str = None, transforms=None, return_masks: bool = False,
                 visib_ratio_thr: float = 0.5):
        self.root = root
        self.split = split
        self.transforms = transforms
        self._transforms = transform.Compose([
            transform.ToTensor(),
            transform.Resize((640, 640)),  # Resize to the size expected by your model
        ])
        self.return_masks = return_masks
        self.visib_ratio_thr = visib_ratio_thr

        # scan sequences
        split_dir = os.path.join(root, split)
        seq_ids = sorted(d for d in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, d)))

        # build item list
        self.items = []  # each item: dict with rgb_path, annos
        i = 0
        for i, seq in enumerate(seq_ids):
            seq_dir = os.path.join(split_dir, seq)
            rgb_dir = os.path.join(seq_dir, '')
            mask_dir = os.path.join(seq_dir, 'mask')
            visib_dir = os.path.join(seq_dir, 'mask_visib')
            coco_file = os.path.join(seq_dir, 'scene_gt_coco.json')
            if not os.path.exists(coco_file):
                continue
            with open(coco_file, 'r') as f:
                coco = json.load(f)
            # index images and annotations
            img_map = {img['id']: img for img in coco.get('images', [])}
            annos = coco.get('annotations', [])

            for img_id, img_info in img_map.items():              
                fname = img_info['file_name']
                rgb_path = os.path.join(rgb_dir, fname[:-3]+"png")
                
                if not os.path.exists(rgb_path):
                    # print(rgb_path)
                    continue
                # filter annotations for this image
                anns = [a for a in annos if a['image_id'] == img_id]

                if not anns:
                    # skip image if no valid annotations
                    continue

                self.items.append({
                    'rgb': rgb_path,
                    'id': img_id,
                    'annotations': anns,
                    'mask_dir': mask_dir,
                    'visib_dir': visib_dir,
                })

        # prepare mask converter
        self.prepare = ConvertCocoPolysToMask(self.return_masks)
    
    def load_item(self, idx:int):
        item = self.items[idx]
        image = Image.open(item['rgb']).convert('RGB')

        # optionally filter by visible/full mask area ratio
        annos = item['annotations']
        if self.visib_ratio_thr > 0.0:
            filtered = []
            for a in annos:
                # mask files: <image_id>_<ann_id>.png
                fn = f"{item['id']:06d}_{a['id']-annos[0]['id']:06d}.png"
                mask_path = os.path.join(item['mask_dir'], fn)
                visib_path = os.path.join(item['visib_dir'], fn)
                if os.path.exists(mask_path) and os.path.exists(visib_path):
                    try:
                        m_full = np.array(Image.open(mask_path))
                        m_vis = np.array(Image.open(visib_path))
                        area_full = np.count_nonzero(m_full)
                        area_vis = np.count_nonzero(m_vis)
                        ratio = area_vis / area_full if area_full > 0 else 0.0
                    except Exception:
                        ratio = 0.0
                else:
                    ratio = 1.0
                a["ratio"] = ratio
                if ratio >= self.visib_ratio_thr:
                    filtered.append(a)
            # print("rgb_path: ", item['rgb'], "image_id: ", a['image_id'], "id:", a['id'], "ratio: ", a["ratio"])
            # print("#######################")
            annos = filtered
        for anno in annos:
            anno["category_id"] = 0
            anno["bbox"] = [anno["bbox"][0], anno["bbox"][1], anno["bbox"][2], anno["bbox"][3]]
        # build target dict
        target = {
            'image_id': idx,
            'annotations': annos
        }

        # convert COCO polys to boxes/masks
        image, target = self.prepare(image, target)

        # convert to torchvision tensors
        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(
                target['boxes'], key='boxes', spatial_size=image.size[::-1]
            )

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')    

        return image, target

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img, target = self.load_item(idx)
        # boxes = target["boxes"]
        # boxes[:, 0] = (boxes[:, 0] + boxes[:,2])/2
        # boxes[:, 1] = (boxes[:, 1] + boxes[:,3])/2
        # boxes[:, 2] = (boxes[:, 2] - boxes[:,0])*2
        # boxes[:, 3] = (boxes[:, 3] - boxes[:,1])*2
        # boxes[:, 0] = boxes[:, 0]/1280
        # boxes[:, 1] = boxes[:, 1]/960
        # boxes[:, 2] = boxes[:, 2]/1280
        # boxes[:, 3] = boxes[:, 3]/960
        # target["boxes"] = boxes
        # img = self._transforms(img)
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)

        return img, target
    
    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


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

    def __call__(self, image: Image.Image, target, **kwargs):
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

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

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
        # boxes[:, 0] = (boxes[:, 0] + boxes[:,2])/2
        # boxes[:, 1] = (boxes[:, 1] + boxes[:,3])/2
        # boxes[:, 2] = (boxes[:, 2] - boxes[:,0])*2
        # boxes[:, 3] = (boxes[:, 3] - boxes[:,1])*2
        # boxes[:, 0] = boxes[:, 0]/1280
        # boxes[:, 1] = boxes[:, 1]/960
        # boxes[:, 2] = boxes[:, 2]/1280
        # boxes[:, 3] = boxes[:, 3]/960


        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
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

        target["orig_size"] = torch.as_tensor([int(w), int(h)])

        return image, target
    

mscoco_category2name = {1: 'part'}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}