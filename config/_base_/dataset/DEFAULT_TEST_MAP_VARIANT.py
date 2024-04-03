MAP_COMMON_CFG_LOCAL = dict(
    type='LVISDataset',
    template_file=r"{{fileDirname}}/template/MAP.json",
    image_folder=r'/home/ommos92/datasets/LVIS/val2017',
    max_dynamic_size=None, 
)


DEFAULT_TEST_MAP_VARIANT = dict(
   MAP_OBJ_VAL=dict(
        **MAP_COMMON_CFG_LOCAL,
        filename=r'{{fileDirname}}/../../../data/lvis_v1_val.json',
    ),

)