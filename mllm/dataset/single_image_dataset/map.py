import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import numpy as np
import re

from lvis.lvis import LVIS
from lvis.eval import LVISEval

import torch
from torchvision.ops import box_iou, box_convert
from torch.utils.data import Dataset
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion

from collections import defaultdict
import itertools

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
    QUESTION_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    NUM_OBJS_PLACEHOLDER,
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
        
        #Set this for testing purposes, and truncating the
        max_images = 3
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

        #Define the maximum number of objects
        max_objects = 1
        #Get the Bounding Box and the Point using the 
        # Retrieve bounding boxes for the image
        bounding_boxes = [np.array(ann['bbox']) for ann in annotations[:max_objects]]
        bounding_boxes = box_convert(torch.tensor(bounding_boxes), 'xywh', 'xyxy').tolist()

        category_ids = [ann['category_id'] for ann in annotations[:max_objects]]
        category_names = [self.lvis.cats[cat_id]['name'] for cat_id in category_ids]

        bbox_label_dict = defaultdict(list)

        for name, bbox in zip(category_names, bounding_boxes):
            bbox_label_dict[name].append(bbox)

        bbox_label_str = '; '.join([f'{k} {str(v).replace("], [", ";").replace(", ", ",")}' for k, v in list(bbox_label_dict.items())])
        #Construct the Question
        question = self.get_template().replace(EXPR_PLACEHOLDER, bbox_label_str)

        #Need to add in the number of objects detected in the image for the MAP Prediction Score
        boxes_seq = list(range(1,len(bbox_label_dict)))

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
                    'boxes_seq': [boxes_seq],
                }
            ]
        }
        
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
    """
    Computes the Mean Average Precision (MAP) metric for the LVIS dataset over the number of objects detected 
    in the image. Inherits the LVISEval class from the lvis package and the BaseComputeMetrics class from the
    mllm package. LVIS Eval should be able to compute the MAP metric for the LVIS dataset.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']
        self.mAP = MeanAveragePrecision(box_format='xywh', iou_type='bbox')
    
    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        #Initialize the number of failed predictions and targets
        failed = 0
        target_failed = 0

        self.box_formatter
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            try: 
                # need to do IOU Calculation per sample
                for i in range(len(extract_pred)):
                    pred_box = extract_pred[i]
                    target_box = extract_target[0][i]

                    pred_box = torch.tensor(pred_box)
                    target_box = torch.tensor(target_box)

                    pred_box = box_convert(pred_box, 'xywh', 'xyxy')
                    target_box = box_convert(target_box, 'xywh', 'xyxy')

                    pred = [dict(
                        boxes=pred_box,
                        scores=torch.tensor([1.0]),
                        labels=torch.tensor(0),
                    )]
                    target = [dict(
                        boxes=target_box,
                        labels=torch.tensor(0),
                    )]
                    
                    self.mAP.update(pred, target)
                    

            except:
                extract_pred = None

            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans for target: {target}")
                continue
            if extract_pred is None:
                failed += 1
                logger.warning(f"failed to extract ans for pred: {pred}")
                extract_pred = [0, 0, 0, 0]

        
        # Compute the final mAP Score Per Objects
        result = self.mAP.compute()

        return {
            'mAP': result,
            'target_failed': target_failed,
            'failed': failed
        }

    def extract_ans(self, string: str):
        """
        Needs to extract multiple bounding boxes from the string.

        Args:
            string (str): _description_

        Returns:
            _type_: _description_
        """
        try:
            list_of_boxes = self.box_formatter.extract(string)

            # Find all occurrences of text followed by a bounding box
            #matches = re.findall(r'([A-Za-z\s]+)\[\d+\.\d+,\d+\.\d+,\d+\.\d+,\d+\.\d+\]', string)

            # Remove leading and trailing whitespace from each match
            #class_names = [match.strip() for match in matches]

            #print(class_names)
            return list_of_boxes
        
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None

