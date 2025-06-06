import torch.nn as nn

class BaseFusionModel(nn.Module):
    def __init__(self, **config):
        super(BaseFusionModel, self).__init__()
        self.config = config

    def fuse(self, bev_embed, transformed_sender_bev, img_metas, **kwargs):
        raise NotImplementedError("Each fusion model must implement the fuse method.")