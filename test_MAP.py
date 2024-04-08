import re

s = 'A man[0.602,0.296,0.776,0.710] in a white jacket[0.604,0.326,0.780,0.480] and black apron[0.624,0.452,0.750,0.606] is standing in a kitchen with an apron[0.624,0.452,0.750,0.606] on his waist[0.624,0.452,0.750,0.606].'

# Find all occurrences of text followed by a bounding box
matches = re.findall(r'([A-Za-z\s]+)\[\d+\.\d+,\d+\.\d+,\d+\.\d+,\d+\.\d+\]', s)

# Remove leading and trailing whitespace from each match
class_names = [match.strip() for match in matches]

print(class_names)

from torch import tensor
from torchmetrics.detection import MeanAveragePrecision
preds = [
  dict(
    boxes=tensor([[258.0, 41.0, 606.0, 285.0], [133.2, 78, 625, 222]]),
    scores=tensor([0.5,0.5]),
    labels=tensor([0,1]),
  )
]
target = [
  dict(
    boxes=tensor([[214.0, 41.0, 562.0, 285.0], [131, 68, 631, 224]]),
    labels=tensor([0,1]),
  )
]
metric = MeanAveragePrecision()
metric.update(preds, target)

result = metric.compute()
print(result)