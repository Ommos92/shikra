import logging
from lvis import LVIS, LVISResults, LVISEval

# result and val files for 100 randomly sampled images.
ANNOTATION_PATH = "/Users/andrewelkommos/Documents/gitworkspace/shikra/data/lvis_v1_val.json"
RESULT_PATH = "/Users/andrewelkommos/Documents/gitworkspace/shikra/data/lvis_v1_val_results.json"

ANN_TYPE = 'bbox'

lvis_eval = LVISEval(ANNOTATION_PATH, RESULT_PATH, ANN_TYPE)
lvis_eval.run()
lvis_eval.print_results()