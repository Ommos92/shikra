from mllm.dataset.single_image_dataset.map import LVISDataset


dataset = LVISDataset(template_string = '/Users/andrewelkommos/Documents/gitworkspace/shikra/data/lvis_v1_val.json', template_folder='/Users/andrewelkommos/Documents/datasets/MSCOCO2017/val2017')
print(dataset[0])