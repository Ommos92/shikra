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

from ..utils import MInstrDataset



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

#Registers as a dataset
@DATASETS.register_module()
class LVISDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
        self.lvis = LVIS(annotation_path=kwargs.get('filename'))
        self.image_folder = kwargs.get('image_folder')
        max_images = 20
        self.max_images = max_images if max_images is not None else len(self.lvis.dataset['images'])


    def __getitem__(self, index):

        #Get the image annotations and the path to the image
        image_path = self.lvis.dataset['images'][index]['coco_url']
        file_name = str.split(image_path, '/')[-1]

        image_id = self.lvis.dataset['images'][index]['id']
        # Get the annotation ids for the image
        ann_ids = self.lvis.get_ann_ids(img_ids=[image_id])
        annotations = self.lvis.load_anns(ids=ann_ids)

        # Retrieve Image
        image = self.get_image(self.image_folder + "/" + file_name)

        #Get the Bounding Box and the Point using the 
        # Retrieve bounding boxes for the image
        bounding_boxes = [np.array(ann['bbox']) for ann in annotations]
        category_ids = [ann['category_id'] for ann in annotations]

        #Get the number of objects in the dataset.
        num_objects = len(bounding_boxes)

        # Retrieve category names
        cats = self.lvis.load_cats(ids=category_ids)
        synonym_list = [synonym for cat in cats for synonym in cat['synonyms']]

        # Need to remove repeating synonyms
        synonym_list = list(set(synonym_list))
        synonym_query = ', '.join(synonym_list)

        #Construct the Question
        question = self.get_template().replace(EXPR_PLACEHOLDER, synonym_query)

        #Need to add in the number of objects detected in the image for the MAP Prediction Score
        # Also need to figure out boxes_seq as well...

        ret = {
            'image': image,
            'target': {
                'boxes': bounding_boxes
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
        print(ret)
        return ret
    
    def __len__(self):
        return self.max_images
    
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']
        self.metric = MeanAveragePrecision(box_format='xywh', iou_type='bbox')
        self.iou = IntersectionOverUnion(box_format='xywh'),
        self.map_scores = {1: [], 2: [], 3: [], 5: [], 8: [], 10: [], 15: [], 20: []}  # store the MAP scores for each number of objects

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        self.box_formatter
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
            self.iou(pred_boxes, target_boxes)

            ious = torch.einsum('i i -> i', ious)  # take diag elem

            iou = ious.mean().item()
            correct = (ious > 0.5).sum().item()

            num_objects = len(target_boxes)  # calculate the number of objects detected
            if num_objects in self.map_scores:
                self.map_scores[num_objects].append(1.0 * correct / len(targets))  # update the MAP score for the current number of objects

        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
            'warning': warn_message,
            'map_scores': self.map_scores,  # return the MAP scores for each number of objects
        }