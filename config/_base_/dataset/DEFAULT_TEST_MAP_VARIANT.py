MAP_COMMON_CFG_LOCAL = dict(
    type='LVISDataset',
    image_folder='/dataset/MSCOCO2017/val2017',
    #template_file=r"{{fileDirname}}/template/VQA.json", 
)


DEFAULT_TEST_POINT_VARIANT = dict(
   MAP_OBJ_val=dict(**MAP_COMMON_CFG_LOCAL, version='b', filename='{{fileDirname}}/../../../data/lvis_v1_val.json'),

)