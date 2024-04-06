"""

{
  "image": "string",
  "text": "string",
  "boxes_value": [
    "string"
  ],
  "boxes_seq": [
    "string"
  ],
  "server_url": "http://127.0.0.1:12345/shikra"
}


"""

import torch 

import requests
import argparse

import json
import os

import numpy as np
from torch.utils.data import Dataset

import lvis.eval as lvis_eval
import lvis.lvis as lvis
from torchvision.ops import box_iou, box_convert

import random as rand

from PIL import Image

import fast_api_client as fc



class LVISDataset(Dataset):
    def __init__(self, filename='data/lvis_v1_val.json', image_folder=r'/home/ommos92/datasets/LVIS/val2017'):
        super().__init__()
        self.lvis = lvis.LVIS(annotation_path=filename)
        self.image_folder = image_folder
        max_images = 1
        self.max_images = max_images if max_images is not None else len(self.lvis.dataset['images'])
        self.server_url = 'http://127.0.0.1:12345' + "/shikra"

    def __getitem__(self, index):

        #Get the image annotations and the path to the image
        image_path = self.lvis.dataset['images'][index]['coco_url']
        file_name = str.split(image_path, '/')[-1]

        image_id = self.lvis.dataset['images'][index]['id']
        # Get the annotation ids for the image
        ann_ids = self.lvis.get_ann_ids(img_ids=[image_id])
        annotations = self.lvis.load_anns(ids=ann_ids)

        # Retrieve Image
        #image = self.get_image(self.image_folder + "/" + file_name)
        image_path = self.image_folder + "/" + file_name

        #Get the Bounding Box and the Point using the 
        # Retrieve bounding boxes for the image
        bounding_boxes = [np.array(ann['bbox']) for ann in annotations]
        bounding_boxes = box_convert(torch.tensor(bounding_boxes), 'xywh', 'xyxy').tolist()

        category_ids = [ann['category_id'] for ann in annotations]

        #Get the number of objects in the dataset.
        num_objects = len(bounding_boxes)

        # Retrieve category names
        cats = self.lvis.load_cats(ids=category_ids)
        label_list = [cat['name'] for cat in cats]

        # Need to remove repeating names
        label_list = list(set(label_list))
        label_query = ', '.join(label_list)

        # Join the objects with commas and an 'and' before the last one
        objects_str = ', '.join(label_list[:-1]) + ' and ' + label_list[-1]

        # Formulate the question
        question = f"What is the relationship in position between the {objects_str} in the image? Identify each object's Coordinates using these reference bounding boxes <boxes>."
        question = f"You are an AI tasked with finding the bounding boxes <boxes> of objects in the image. Identify the bounding boxes of the {objects_str} in the image and \
        return the coordinates of each object in the format <x1, y1, x2, y2>."

        # Load a set of questions from the json file in templates directory
        # Get the template
        template = json.load(open('mllm/demo/templates/map.json', 'r', encoding='utf8'))

        #Create a list of of size of the number of objects detected in the image [0,1,2, ..., n-1]
        boxes_seq = list(range(1,num_objects-1))


        # Select a random template
        #text = template[rand.randint(0, len(template) - 1)]


        #Need to add in the number of objects detected in the image for the MAP Prediction Score
        # Also need to figure out boxes_seq as well...
        
        response = fc.query(image_path, question, bounding_boxes, [boxes_seq], self.server_url)
        _, image = fc.postprocess(response['response'], image=Image.open(image_path))
        return {"response": response, "image" : image}
    
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
        image = self.read_img_general(image_path)
        return image


    def read_img_general(img_path):

        return Image.open(img_path).convert('RGB')



def create_api_query(image, text, boxes_value, boxes_seq, server_url):

    data = {
        "image": image,
        "text": text,
        "boxes_value": boxes_value,
        "boxes_seq": boxes_seq,
        "server_url": server_url
    }
    response = requests.post(server_url, json=data)
    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create API queries.')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:12345/shikra', help='Server URL for the query')
    args = parser.parse_args()

    #make 100 queries or something
    dataset = LVISDataset(filename='data/lvis_v1_val.json', image_folder=r'/home/ommos92/datasets/LVIS/val2017')
    for i in range(100):
        response = dataset[i]

        try:
            image = response['image']
            image.save(f'mllm/demo/results/{i}.png')
        except:
            pass

        print(response)
