from mllm.dataset.single_image_dataset.map import LVISDataset, MAPComputeMetrics
import lvis.eval as lvis_eval
import lvis.lvis as lvis

from torchmetrics.detection import MeanAveragePrecision as MAP

ann_path = '/Users/andrewelkommos/Documents/gitworkspace/shikra/data/lvis_v1_val.json'

dataset = LVISDataset(annotation_path= ann_path, image_folder = '/Users/andrewelkommos/Documents/datasets/MSCOCO17/val2017')
result = dataset[0]
print(result)





