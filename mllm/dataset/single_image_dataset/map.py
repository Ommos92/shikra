import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import numpy as np

import torch
from torchvision.ops import box_iou
from torch.utils.data import Dataset


from ..utils import (
    BaseComputeMetrics,
)

from ..process_function import (
    BoxFormatter,
)

from ..root import (
    DATASETS,
    METRICS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    EXPR_PLACEHOLDER,
)



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


@DATASETS.register_module()
class LVISDataset(Dataset):
    def __init__(self, filename, image_folder=None, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.image_folder = image_folder
        self.rng = np.random.default_rng(seed)

        self.data = []
        with open(filename, 'r', encoding='utf8') as f:
            # for line in tqdm(f, desc=f'{self.__class__.__name__} loading ann {self.filename}'):
            for line in f:
                self.data.append(line)

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        
        #should contain the path to the image from the folder mscoco2017 val
        img_path = item['img_path']
        ann_img = self.get_image(item) # Get image annotation


        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {BOXES_PLACEHOLDER} .',
                    'boxes_seq': [[0]],
                }
            ]
        }
        return ret

    def get_image(self, item):
        return {
            'id': item['image']['id'],
            'width': item['image']['width'],
            'height': item['image']['height'],
            'license': item['image']['license'],
            'flickr_url': item['image']['flickr_url'],
            'coco_url': item['image']['coco_url'],
            'date_captured': item['image']['date_captured'],
            'not_exhaustive_category_ids': item['image']['not_exhaustive_category_ids'],
            'neg_category_ids': item['image']['neg_category_ids']
        }
    
    def get_categories(self, item):
        return {
            'id': item['categories']['id'],
            'synset': item['categories']['synset'],
            'synonyms': item['categories']['synonyms'],
            'def': item['categories']['def'],
            'instance_count': item['categories']['instance_count'],
            'image_count': item['categories']['image_count'],
            'frequency': item['categories']['frequency']
        }

    def get_annotation(self, item):
        return {
            'id': item['annotation']['id'],
            'image_id': item['annotation']['image_id'],
            'category_id': item['annotation']['category_id'],
            'segmentation': item['annotation']['segmentation'],
            'area': item['annotation']['area'],
            'bbox': item['annotation']['bbox']
        }



@METRICS.register_module()
class MAPComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0

        pred_boxes, target_boxes = [], []
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans for target: {target}")
                continue
            if extract_pred is None:
                failed += 1
                logger.warning(f"failed to extract ans for pred: {pred}")
                extract_pred = [0, 0, 0, 0]
            target_boxes.append(extract_target)
            pred_boxes.append(extract_pred)

        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes)
            pred_boxes = torch.tensor(pred_boxes)
            # normalized box value is too small, so that the area is 0.
            ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
            ious = torch.einsum('i i -> i', ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            iou = ious.mean().item()
            correct = (ious > 0.5).sum().item()

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != 4:
                return None
            return box
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None





if __name__ == '__main__':

    dataset = LVISDataset('/Users/andrewelkommos/Documents/gitworkspace/shikra/data/lvis_v1_val.json', '/Users/andrewelkommos/Documents/datasets/MSCOCO2017/val2017')

    print(dataset[0])