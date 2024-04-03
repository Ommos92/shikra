from lvis import LVIS
from lvis.vis import LVISVis
import numpy as np
import matplotlib.pyplot as plt

ann_path = '/home/ommos92/adv-computer-vision/shikra/data/lvis_v1_val.json'
# Load LVIS dataset
lvis = LVIS(ann_path)

# Get annotation id for a specific image
image_id = 446522
ann_ids = lvis.get_ann_ids(img_ids=[image_id])
annotations = lvis.load_anns(ids=ann_ids)
img = lvis.load_imgs([image_id])

# Retrieve bounding boxes for the image
bounding_boxes = [np.array(ann['bbox']) for ann in annotations]
category_ids = [ann['category_id'] for ann in annotations]

# Retrieve category names
cats = lvis.load_cats(ids=category_ids)
labels_id = [cat['id'] for cat in cats] 
# Retrieve all of the category names
cat_list = [synonym for cat in cats for synonym in cat['name']]
# Format into a single string separated by commas
cat_list = ', '.join(cat_list)




url = str.split(lvis.load_imgs([image_id])[0]['coco_url'], sep='/')
# Load the image using its file name
image = plt.imread("/Users/andrewelkommos/Documents/datasets/MSCOCO17/val2017/" + url[-1])
fig, ax = plt.subplots()
# Display the image
plt.imshow(image)

# Display the annotations
vis = LVISVis(lvis_gt=ann_path, img_dir='/Users/andrewelkommos/Documents/datasets/MSCOCO17/val2017')
for bbox in bounding_boxes:
    vis.vis_bbox(ax, bbox=bbox)


plt.show()