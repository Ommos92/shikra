import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import numpy as np

from lvis.lvis import LVIS
from lvis.eval import LVISEval

import torch
from torchvision.ops import box_iou
from torch.utils.data import Dataset
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion

import os.path

from ..utils.io import read_img_general

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

#Registers as a dataset
@DATASETS.register_module()
class LVISDataset(LVIS):
    """annotation_path: path to the annotation file"""
    def __init__(self,image_folder=None, **kwargs):
        super().__init__(**kwargs)
        self.image_folder = image_folder  


    def __getitem__(self, index):

        #Get the image annotations and the path to the image
        image_path = self.dataset['images'][index]['coco_url']
        file_name = str.split(image_path, '/')[-1]

        image_id = self.dataset['images'][index]['id']
        # Get the annotation ids for the image
        ann_ids = self.get_ann_ids(img_ids=[image_id])
        annotations = self.load_anns(ids=ann_ids)


        # Retrieve Image
        image = self.get_image(self.image_folder + "/" + file_name)

        #Get the Bounding Box and the Point using the 
        # Retrieve bounding boxes for the image
        bounding_boxes = [np.array(ann['bbox']) for ann in annotations]
        category_ids = [ann['category_id'] for ann in annotations]

        ret = {
            'image': image,
            'target': {
                'boxes': bounding_boxes
            },
        }
        return ret
    
    def __len__(self):
        return len(self.dataset['images'])
    
    def get_image(self, image_path):
        """Get the image from the image folder using the annotation id 
        Args:
            annotation_id: the id of the annotation
            Returns:
                image: the image from the image folder"""
        image_name = image_path.replace('http://images.cocodataset.org/val2017/', self.image_folder + '/')
        if self.image_folder is not None:
            image_path = os.path.join(self.image_folder, image_name)
        image = read_img_general(image_path)
        return image



@METRICS.register_module()
class MAPComputeMetrics(BaseComputeMetrics):
    """
    Computes the Mean Average Precision (MAP) metric for the LVIS dataset over the number of objects detected 
    in the image. Inherits the LVISEval class from the lvis package and the BaseComputeMetrics class from the
    mllm package. LVIS Eval should be able to compute the MAP metric for the LVIS dataset.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']
        self.metric = MeanAveragePrecision(box_format='xywh', iou_type='bbox')

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        
        self.box_formatter
        # Need to calculate the MAP score per number of objects in the image
        average_precisions = []


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