from mmcv import Registry, Config
from opencood.models.fuse_modules import(coalign_fuse, f_cooper_fuse, hmsa, mswin, self_attn, swap_fusion_modules,
    v2v_fuse, V2VAM, v2xvit_basic, where2comm_fuse)
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.fuse_modules.V2VAM import V2V_AttFusion


FUSION_MODELS = Registry('fusion_model')

FUSION_MODELS.register_module(name='Where2comm', module=Where2comm)
FUSION_MODELS.register_module(name='F_Cooper', module=SpatialFusion)
FUSION_MODELS.register_module(name='Cobev', module=SwapFusionEncoder)
FUSION_MODELS.register_module(name='V2VAM', module=V2V_AttFusion)


def get_fusion_model(name, **kwargs):
    # if name == "Where2comm":
    #     base_bev_backbone_cfg = Config.fromfile("./submodules/OpenCOOD/opencood/hypes_yaml/point_pillar_where2comm.yaml").model.args
    #     specific_cfg = Config.fromfile("./submodules/OpenCOOD/opencood/hypes_yaml/point_pillar_where2comm.yaml").model.args.where2comm_fusion
    #     args = {**base_bev_backbone_cfg, **specific_cfg, **kwargs}
    # elif name == "Cobev":
    #     specific_cfg = Config.fromfile("./submodules/OpenCOOD/opencood/hypes_yaml/point_pillar_cobevt.yaml").model.args.fax_fusion
    #     args = {**specific_cfg}
    # elif name == "F_Cooper":
    #     args = kwargs 
    # elif name == "V2VAM":
    #     specific_cfg = Config.fromfile("./submodules/OpenCOOD/opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml").model.args
    #     args = {**specific_cfg, **kwargs}
    # else:
    #     raise ValueError(f"Unsupported fusion type: {name}")
    # #assert name in FUSION_MODELS, f"Fusion model {name} not found!"
    fusion_model_class = FUSION_MODELS.get(name)
    assert fusion_model_class is not None, f"Fusion model {name} not found!"

    if name == "F_Cooper":
        return fusion_model_class() 
    elif name == "Cobev":
        return fusion_model_class(**kwargs)
    else:
        return fusion_model_class(**kwargs)
