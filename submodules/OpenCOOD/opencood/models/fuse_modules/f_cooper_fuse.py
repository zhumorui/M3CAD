# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of F-cooper maxout fusing.
"""
import torch
import torch.nn as nn


class SpatialFusion(nn.Module):
    def __init__(self):
        super(SpatialFusion, self).__init__()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        x = x.squeeze(0)
        split_x = self.regroup(x, record_len)
        out = []

        for xx in split_x:
            xx = torch.max(xx, dim=0, keepdim=True)[0]
            out.append(xx)
        return torch.cat(out, dim=0)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    test_data = torch.rand(1, 2, 256, 200, 200)
    test_data = test_data.cuda()
    
    record_len = torch.tensor([2]).cuda()

    model = SpatialFusion()
    model.cuda()
    
    output = model(test_data, record_len)
    print(f"Input shape: {test_data.shape}")
    print(f"Output shape: {output.shape}")