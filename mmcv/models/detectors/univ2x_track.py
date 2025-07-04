import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from mmcv.fusion_utils import visualize_bev, plot_trajectories, visualize_all, transform_coordinates, get_sender2ego_4x4, transform_coordinates_with_rotation, build_4x4_from_pos_rot, build_sender2sego_rt
from torchvision.transforms.functional import rotate
from mmcv.utils import auto_fp16
from mmcv.models import DETECTORS
from mmcv.core import bbox3d2result
from mmcv.core.bbox.coder import build_bbox_coder
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.utils.grid_mask import GridMask
import copy
import math
import os
from datetime import datetime
from mmcv.core.bbox.util import normalize_bbox
from mmcv.models import build_loss
from einops import rearrange
from mmcv.models.utils.transformer import inverse_sigmoid
from ..dense_heads.track_head_plugin import MemoryBank, QueryInteractionModule, Instances, RuntimeTrackerBase
from mmcv.fusion_utils import get_fusion_model
from ..fusion_modules import AgentQueryFusion

@DETECTORS.register_module()
class UniV2xTrack(MVXTwoStageDetector):
    """UniV2x tracking part
    """
    def __init__(
        self, 
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        enable_fusion=False,
        fusion_cfg=None,
        debug=False,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        prev_frame_num=0,
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=1,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        ),
        pc_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=1,
        vehicle_id_list=None,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
        #is_ego_agent=False,
        enable_track_fusion = False,
        return_track_query=True,
        save_track_query=False,
        save_track_query_file_root=''
    ):
        super(UniV2xTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length
        self.debug = debug
        self.enable_fusion = enable_fusion
        self.bev_embed_linear = nn.Linear(embed_dims, embed_dims)
        self.bev_pos_linear = nn.Linear(embed_dims, embed_dims)
        if self.enable_fusion:
            if fusion_cfg is not None:
                fusion_model_name = fusion_cfg.pop("type")
                print(f"Using fusion model: {fusion_model_name}")
                print(f"fusion_cfg: {fusion_cfg}")
                self.feature_fusion = get_fusion_model(fusion_model_name, args=fusion_cfg)
            else:
                raise ValueError("fusion_cfg is not provided but enable_fusion is True") 
        # cross-agent query interaction
        self.enable_track_fusion = enable_track_fusion
        if self.enable_track_fusion:
            self.cross_agent_query_interaction = AgentQueryFusion(pc_range=self.pc_range,
                                                                  embed_dims=self.embed_dims)
        self.save_track_query = save_track_query
        self.save_track_query_file_root = save_track_query_file_root
        self.return_track_query = return_track_query        
        
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        
        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode
        self.prev_frame_num = prev_frame_num
        self.prev_frame_infos = []
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = nn.Embedding(self.num_query+1, self.embed_dims * 2)   # the final one is ego query, which constantly models ego-vehicle
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )
        #ego
        self.criterion = build_loss(loss_cfg)
        #sender
        self.criterion_sender = build_loss(loss_cfg)
        self.test_track_instances = None
        self.test_sender_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        if img is None:
            return None
        assert img.dim() == 5
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query = query

        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )

        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """
        Args:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
            velocity (Tensor): (num_query, 2). m/s
                in lidar frame. vx, vy
            global2lidar (np.Array) [4,4].
        Outs:
            ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
        """
        # print(l2g_r1.type(), l2g_t1.type(), ref_pts.type())

        if isinstance(l2g_r1,list):
            l2g_r1 = l2g_r1[0]
        if isinstance(l2g_t1,list):
            l2g_t1 = l2g_t1[0]
        if isinstance(l2g_r2,list):
            l2g_r2 = l2g_r2[0]
        if isinstance(l2g_t2,list):
            l2g_t2 = l2g_t2[0]          
        
        l2g_r1 = l2g_r1.type(torch.float)
        l2g_t1 = l2g_t1.type(torch.float)
        l2g_t2 = l2g_t2.type(torch.float)
        time_delta = time_delta.type(torch.float)

        num_query = ref_pts.size(0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = reference_points + velo_pad * time_delta

        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2

        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)

        ref_pts = ref_pts @ g2l_r

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0]
        )
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1]
        )
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2]
        )

        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)

    def get_history_bev(self, imgs_queue, img_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, 
                    img_metas=img_metas, 
                    prev_bev=prev_bev)
        self.train()
        return prev_bev

    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos
    
    def fusion_bev_features(self, bev_embed, sender_bev_embed, sender_mask):
        """
        Fuse BEV features from ego and sender vehicles.
        
        Args:
            bev_embed (torch.Tensor): Ego BEV features.
                Shape: ((bev_h * bev_w), batch_size, channels)
            sender_bev_embed (torch.Tensor): Sender BEV features.
                Shape: ((bev_h * bev_w), batch_size, channels) 
            sender_mask (torch.Tensor): Mask indicating valid sender BEV regions.
                Shape: (batch_size, bev_h, bev_w)
                
        Returns:
            torch.Tensor: Fused BEV features.
                Shape: ((bev_h * bev_w), batch_size, channels)
        """
        sender_bev_embed = rearrange(sender_bev_embed, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
        bev_embed = rearrange(bev_embed, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
        bev_mask = torch.ones_like(sender_mask)
        mask = torch.stack([bev_mask, sender_mask], dim=-1).cuda()
        input_bev = torch.stack([bev_embed, sender_bev_embed], dim=1).cuda()

        fusion_bev_embed = self.feature_fusion(input_bev, mask)
        fusion_bev_embed = rearrange(fusion_bev_embed, 'b c h w -> (h w) b c')

        return fusion_bev_embed
    
    # def transform_bev(self, bev, ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw):
    #     """带有手动双线性插值的BEV转换"""
    #     device = bev.device
    #     h, w = bev.shape[-2:]
        
    #     pixel_indices = torch.arange(h*w, device=device)
    #     y_indices = pixel_indices // w
    #     x_indices = pixel_indices % w
        
    #     # 转换到[-51.2, 51.2]
    #     x = (x_indices / (w-1) * 102.4) - 51.2
    #     y = (y_indices / (h-1) * 102.4) - 51.2
        
    #     # 创建position矩阵
    #     pos = torch.stack([x, y, torch.zeros_like(x)], dim=1)
        
    #     # 使用transform_coordinates
    #     transformed_pos, _ = transform_coordinates(
    #         pos=pos.to(torch.float32),
    #         yaw=torch.zeros_like(pos[:, 0]),
    #         ego_x=ego_x,
    #         ego_y=ego_y,
    #         ego_yaw=ego_yaw,
    #         sender_x=sender_x,
    #         sender_y=sender_y,
    #         sender_yaw=sender_yaw
    #     )
        
    #     transformed_x = transformed_pos[:, 0]
    #     transformed_y = transformed_pos[:, 1]
        
    #     # 从[-51.2, 51.2]映射回[0, w-1]和[0, h-1]，但保持浮点数
    #     pixel_x = ((transformed_x + 51.2) / 102.4 * (w-1))
    #     pixel_y = ((transformed_y + 51.2) / 102.4 * (h-1))
        
    #     # 计算四个最近的整数坐标点
    #     x0 = torch.floor(pixel_x).long()
    #     x1 = x0 + 1
    #     y0 = torch.floor(pixel_y).long()
    #     y1 = y0 + 1
        
    #     # 计算插值权重
    #     wx1 = pixel_x - x0.float()
    #     wx0 = 1 - wx1
    #     wy1 = pixel_y - y0.float()
    #     wy0 = 1 - wy1
        
    #     # 创建变换后的特征图和mask
    #     transformed_bev = torch.zeros_like(bev)
    #     mask = torch.zeros((h, w), device=device)
        
    #     # 创建有效的索引mask (需要检查所有四个角点)
    #     valid_x0 = (x0 >= 0) & (x0 < w)
    #     valid_x1 = (x1 >= 0) & (x1 < w)
    #     valid_y0 = (y0 >= 0) & (y0 < h)
    #     valid_y1 = (y1 >= 0) & (y1 < h)
        
    #     valid_indices = valid_x0 & valid_x1 & valid_y0 & valid_y1
        
    #     # 对每个有效位置进行双线性插值
    #     for idx in range(h*w):
    #         if valid_indices[idx]:
    #             src_y = y_indices[idx]
    #             src_x = x_indices[idx]
                
    #             # 获取目标位置的四个邻近点
    #             dst_x0, dst_x1 = x0[idx], x1[idx]
    #             dst_y0, dst_y1 = y0[idx], y1[idx]
                
    #             # 获取权重
    #             w00 = wx0[idx] * wy0[idx]
    #             w01 = wx0[idx] * wy1[idx]
    #             w10 = wx1[idx] * wy0[idx]
    #             w11 = wx1[idx] * wy1[idx]
                
    #             # 使用scatter_add_进行加权累加
    #             transformed_bev[:, dst_y0, dst_x0] += bev[:, src_y, src_x] * w00
    #             transformed_bev[:, dst_y0, dst_x1] += bev[:, src_y, src_x] * w10
    #             transformed_bev[:, dst_y1, dst_x0] += bev[:, src_y, src_x] * w01
    #             transformed_bev[:, dst_y1, dst_x1] += bev[:, src_y, src_x] * w11
                
    #             # 更新mask (使用最大权重来决定mask值)
    #             max_weight = torch.max(torch.tensor([w00, w01, w10, w11]))
    #             mask[dst_y0:dst_y1+1, dst_x0:dst_x1+1] = torch.max(
    #                 mask[dst_y0:dst_y1+1, dst_x0:dst_x1+1],
    #                 torch.tensor(max_weight)
    #             )
        
    #     return transformed_bev, mask
    def _get_coop_bev_embed(self, bev_embed_src, bev_pos_src, track_instances, start_idx):
        bev_embed = bev_embed_src
        bev_pos = bev_pos_src
        act_track_instances = track_instances[start_idx:]  

        # print('act_track_instances len:',len(act_track_instances))

        locs = act_track_instances.ref_pts.sigmoid().clone()
        locs[:, 0:1] = locs[:, 0:1] * self.bev_w # w
        locs[:, 1:2] = locs[:, 1:2] * self.bev_h # h

        pixel_len = 1 # 2

        for idx in range(act_track_instances.ref_pts.shape[0]):
            w = int(locs[idx, 0])
            h = int(locs[idx, 1])
            if w >= self.bev_w or w < 0 or h >= self.bev_h or h < 0:
                continue

            for hh in range(max(0, h - pixel_len), min(self.bev_h - 1, h + pixel_len)):
                for ww in range(max(0, w - pixel_len), min(self.bev_w - 1, w + pixel_len)):
                    bev_embed[hh * self.bev_w + ww, :, :] =  bev_embed[hh * self.bev_w + ww, :, :] + self.bev_embed_linear(act_track_instances.query[idx, self.embed_dims:])
                    bev_pos[:, :, hh, ww] = bev_pos[:, :, hh, ww] + self.bev_pos_linear(act_track_instances.query[idx, :self.embed_dims])
 
        return bev_embed, bev_pos
    #TODO: replaced with new transformation
    def transform_bev(self, bev, ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw):
        """改进后的前向映射，无循环版本，与classical_transform_bev结果一致"""
        device = bev.device
        c, h, w = bev.shape

        # 原像素坐标
        pixel_indices = torch.arange(h*w, device=device)
        y_indices = pixel_indices // w
        x_indices = pixel_indices % w

        # 转换到[-51.2, 51.2]
        x = (x_indices.to(torch.float32) / (w-1) * 102.4) - 51.2
        y = (y_indices.to(torch.float32) / (h-1) * 102.4) - 51.2
        
        pos = torch.stack([x, y, torch.zeros_like(x)], dim=1)

        transformed_pos, _ = transform_coordinates(
            pos=pos,
            yaw=torch.zeros_like(pos[:, 0]),
            ego_x=ego_x,
            ego_y=ego_y,
            ego_yaw=ego_yaw,
            sender_x=sender_x,
            sender_y=sender_y,
            sender_yaw=sender_yaw
        )

        transformed_x = transformed_pos[:, 0]
        transformed_y = transformed_pos[:, 1]

        # 映射回[0, w-1],[0, h-1]
        pixel_x = ((transformed_x + 51.2) / 102.4 * (w-1))
        pixel_y = ((transformed_y + 51.2) / 102.4 * (h-1))

        x0 = pixel_x.floor().long()
        y0 = pixel_y.floor().long()
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = pixel_x - x0.float()
        wx0 = 1 - wx1
        wy1 = pixel_y - y0.float()
        wy0 = 1 - wy1

        valid_x0 = (x0 >= 0) & (x0 < w)
        valid_x1 = (x1 >= 0) & (x1 < w)
        valid_y0 = (y0 >= 0) & (y0 < h)
        valid_y1 = (y1 >= 0) & (y1 < h)

        valid = valid_x0 & valid_x1 & valid_y0 & valid_y1

        src_vals = bev[:, y_indices, x_indices]  # [C, N]
        
        # 将transformed_bev展开，方便使用index_add_
        transformed_bev = torch.zeros_like(bev) # [C,H,W]
        transformed_bev_flat = transformed_bev.view(c, -1)  # [C, H*W]

        # 准备函数来散射加法
        def scatter_values_into(dst, vx, vy, wgt):
            # vx, vy和wgt长度相同
            # 将二维(y,x)索引转换为一维
            flat_idx = vy * w + vx
            # 批量加权加到transformed_bev_flat上
            # vals: [C, valid_count]
            vals = src_vals[:, valid] * wgt[valid].unsqueeze(0)
            dst.index_add_(1, flat_idx[valid], vals)

        # 四个角点的散射
        scatter_values_into(transformed_bev_flat, x0, y0, wx0*wy0)
        scatter_values_into(transformed_bev_flat, x1, y0, wx1*wy0)
        scatter_values_into(transformed_bev_flat, x0, y1, wx0*wy1)
        scatter_values_into(transformed_bev_flat, x1, y1, wx1*wy1)

        # 还原
        transformed_bev = transformed_bev_flat.view(c, h, w)
        
        # 创建mask
        mask = torch.zeros((h, w), device=device)
        # mask也同样根据有效区域赋值(与classical逻辑一致)
        # 四个角点任意有效即为1
        mask_idx = torch.zeros_like(valid, dtype=torch.float32)
        # 使用同样的scatter方法给mask赋值
        def scatter_mask(vx, vy):
            flat_idx = vy * w + vx
            mask_idx.index_fill_(0, flat_idx[valid], 1.0)

        scatter_mask(x0, y0)
        scatter_mask(x1, y0)
        scatter_mask(x0, y1)
        scatter_mask(x1, y1)

        mask = mask_idx.view(h, w)

        return transformed_bev, mask
    
    @auto_fp16(apply_to=("img", "prev_bev"))
    def _sender_forward_single_frame_train(
        self,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
    ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(
            img, img_metas,
            prev_img=prev_img, prev_img_metas=prev_img_metas,
        )

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        if l2g_r2 is not None:
            # Update ref_pts for next frame considering each agent's velocity
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances_list.append(track_instances)
        
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]

            out["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion_sender.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[900], img_metas))
        
        # memory bank 
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out
    
    @auto_fp16(apply_to=("img", "prev_bev"))
    def _forward_single_frame_train(
        self,
        img,
        img_metas,
        track_instances,
        sender_track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
        sender_img=None,
        sender_img_metas=None,
        gt_bboxes_3d=None,
        sender_gt_bboxes_3d=None,
        vis_idx=None,
    ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(
            img, img_metas,
            prev_img=prev_img, prev_img_metas=prev_img_metas,
        )

        # calculate sender_bev_embed
        if self.enable_fusion:
            # get sender_bev_embed
            self.eval()
            with torch.no_grad():
                sender_bev_embed, sendr_bev_pos = self.get_bevs(
                    sender_img, sender_img_metas,
                    prev_img=None, prev_img_metas=None,
                )
                if self.debug:
                    # backup a sender_bev_embed for visualization
                    backup_sender_bev = sender_bev_embed.clone()

                ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw = sender_img_metas[0]['can_bus'][:6]

                # sender_bev_embed: [40000, 1, 256] -> [256, 200, 200]
                sender_bev_embed = rearrange(sender_bev_embed, '(h w) b c -> b c h w', h=200, w=200)
                sender_bev_embed = sender_bev_embed.squeeze(0)

                sender_bev_embed, transformed_mask = self.transform_bev(
                    sender_bev_embed, ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw
                )

                # [256, 200, 200] -> [40000, 1, 256]
                sender_bev_embed = rearrange(sender_bev_embed, 'c h w -> (h w) 1 c')
                sender_mask = transformed_mask[None, ..., None]  # Add batch and channel dims
            self.train()

            # fusion
            fusion_bev_embed = self.fusion_bev_features(bev_embed, sender_bev_embed, sender_mask)
        else:
            fusion_bev_embed = None

        if self.debug and self.enable_fusion:
            visualize_bev(bev_embed, backup_sender_bev, sender_bev_embed, fusion_bev_embed, gt_bboxes_3d, sender_gt_bboxes_3d, sender_img_metas, vis_idx=vis_idx, debug=self.debug)
        
        if self.enable_track_fusion:
            print('------track fusion in train------')
            sender_frame_res = self._sender_forward_single_frame_train(
                sender_img,
                sender_img_metas,
                sender_track_instances,
                prev_img,
                prev_img_metas,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_indices,
                all_instances_pred_logits,
                all_instances_pred_boxes,
                #pc_range=self.pc_range,
            )
            sender_track_instances = sender_frame_res["track_instances"]
            sender_boxes = sender_gt_bboxes_3d[0][vis_idx].tensor.to(img.device)
            ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw = sender_img_metas[0]['can_bus'][:6]
            
            sender_transformed_pos, sender_transformed_yaw = transform_coordinates(sender_boxes[:, :3], sender_boxes[:, 6], 
                                                                                        ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw)
            
            sender2ego_rt = get_sender2ego_4x4(sender_x, sender_y, sender_yaw, ego_x, ego_y, ego_yaw)
            # print('sender2ego_rt:\n', sender2ego_rt)

            # N = sender_boxes.shape[0]
            # pts_sender = sender_boxes[:, :3]
            # pts_sender_homo = torch.cat([
            #                                 pts_sender,
            #                                 torch.ones((N, 1), device=pts_sender.device)
            #                             ], dim=1)
            # pts_sender_homo = pts_sender_homo.unsqueeze(-1)        # (N, 4, 1)
            # sender2ego_rt = sender2ego_rt.to(pts_sender.device)
            # pts_ego_mat1 = torch.matmul(sender2ego_rt, pts_sender_homo)  # (N, 4, 1)
            # pts_ego_mat1 = pts_ego_mat1.squeeze(-1)[:, :3] 
            # print('result of get_sender2ego_4x4:', pts_ego_mat1)
            # print("result of transform_coordinates:", sender_transformed_pos)
            fused_track_instances = self.cross_agent_query_interaction(sender_track_instances,      # sendr的 track
                                                                    track_instances,             # ego 本车的 track
                                                                    sender2ego_rt,      # 基于对方的标定信息
                                                                    self.pc_range
                                                                )
            track_nums_src = len(track_instances)
            #track_nums_new = len(fused_track_instances)
            track_instances=fused_track_instances
        else:
            #fused_track_instances=track_instances
            track_nums_src = len(track_instances)
        
        track_nums_new = len(track_instances)    
        add_nums = track_nums_new - track_nums_src    
        if self.enable_fusion:
            fusion_bev_embed, bev_pos = self._get_coop_bev_embed(
                fusion_bev_embed, bev_pos, track_instances, track_nums_new-add_nums)
        else:
            fusion_bev_embed, bev_pos = self._get_coop_bev_embed(
                bev_embed, bev_pos, track_instances, track_nums_new-add_nums)    

        det_output = self.pts_bbox_head.get_detections(
            fusion_bev_embed if (self.enable_fusion or self.enable_track_fusion) else bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )            

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        if l2g_r2 is not None:
            # Update ref_pts for next frame considering each agent's velocity
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances_list.append(track_instances)
        
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]

            out["track_instances"] = track_instances
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
        
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[900], img_metas))
        
        # memory bank 
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        # Step-2 Update track instances using matcher

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict
    
    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    @auto_fp16(apply_to=("img", "points"))
    def forward_track_train(self,
                            img,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_past_traj,
                            gt_past_traj_mask,
                            gt_inds,
                            gt_sdc_bbox,
                            gt_sdc_label,
                            l2g_t,
                            l2g_r_mat,
                            img_metas,
                            timestamp,
                            sender_img=None,
                            sender_img_metas=None,
                            sender_gt_bboxes_3d=None,
                            sender_gxt_labels_3d=None,
                            sender_gt_inds=None,
                            sender_gt_past_traj=None,
                            sender_gt_past_traj_mask=None,
                            sender_gt_sdc_bbox=None,
                            sender_gt_sdc_label=None,
                ):
        """Forward funciton
        Args:
        Returns:
        """
        print('-------Uniad in Bench2DriveZoo forward_track_train----')
        #print('---filename of img metas in train----', img_metas[0][0]['filename'])
        #print('----filename of sendr_img_metas in train----', sender_img_metas[0][0]['filename'])
        
        track_instances = self._generate_empty_tracks()
        sender_track_instances = self._generate_empty_tracks()
        num_frame = img.size(1)
        # init ego gt instances!
        gt_instances_list = []

        for frame_idx in range(num_frame):
            # Initialize ground truth instances
            instance = Instances((1, 1))

            # Process ego's ground truth bounding boxes
            ego_boxes = gt_bboxes_3d[0][frame_idx].tensor.to(img.device)
            ego_boxes = normalize_bbox(ego_boxes, None)  # Normalize ego's bounding boxes
            ego_labels = gt_labels_3d[0][frame_idx]
            ego_inds = gt_inds[0][frame_idx]
            ego_past_traj = gt_past_traj[0][frame_idx].float()
            ego_past_traj_mask = gt_past_traj_mask[0][frame_idx].float()
            
            sdc_boxes = gt_sdc_bbox[0][frame_idx].tensor.to(img.device)
            sdc_boxes = normalize_bbox(sdc_boxes, None)  # Normalize ego's sdc bounding boxes

            # --- This part code is commented because we already use the global gt_bboxes in the dataset --- #

            # # Merge sender's GT if fusion is enabled
            # if self.enable_fusion:
            #     # Transform the coordinates of sender's ground truth bounding boxes and past trajectory
            #     sender_boxes = sender_gt_bboxes_3d[0][frame_idx].tensor.to(img.device)
            #     ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw = sender_img_metas[0][frame_idx]['can_bus'][:6]
                
            #     # Transform coordinates from sender to ego before normalizing
            #     sender_transformed_pos, sender_transformed_yaw = transform_coordinates(sender_boxes[:, :3], sender_boxes[:, 6], 
            #                                                                             ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw)
            #     sender_boxes_transformed = sender_boxes.clone()
            #     sender_boxes_transformed[:, :3] = sender_transformed_pos
            #     sender_boxes_transformed[:, 6] = sender_transformed_yaw

            #     sender_boxes = normalize_bbox(sender_boxes_transformed, None)  # Normalize sender's bounding boxes
            #     sender_labels = sender_gt_labels_3d[0][frame_idx]
            #     sender_inds = sender_gt_inds[0][frame_idx]
            #     sender_past_traj = sender_gt_past_traj[0][frame_idx].float()
            #     sender_past_traj_mask = sender_gt_past_traj_mask[0][frame_idx].float()

            #     # Transform sender's past trajectory to ego's coordinate system
            #     transformed_past_traj, _ = transform_coordinates(
            #         torch.cat([sender_past_traj, torch.zeros(sender_past_traj.shape[0], sender_past_traj.shape[1], 1, device=sender_past_traj.device)], dim=-1).reshape(-1, 3),
            #         torch.zeros(sender_past_traj.shape[0] * sender_past_traj.shape[1], 1, device=sender_past_traj.device),
            #         ego_x, ego_y, ego_yaw,
            #         sender_x, sender_y, sender_yaw
            #     )

            #     # Reshape transformed past trajectory to its original shape
            #     transformed_past_traj = transformed_past_traj.reshape(sender_past_traj.shape[0], sender_past_traj.shape[1], 3)[:, :, :2]

            #     # Update sender's past trajectory with the transformed values
            #     sender_past_traj = transformed_past_traj

            #     # Identify unique sender instances not overlapping with ego's instances
            #     unique_sender_inds = []
            #     for ind, box in zip(sender_inds, sender_boxes):
            #         if (not (ego_inds == ind).any()) and (-51.2 < box[0] < 51.2) and (-51.2 < box[1] < 51.2) and ((box[0] ** 2 + box[1] ** 2) ** 0.5 > 2):
            #             unique_sender_inds.append(ind)

            #     unique_sender_inds = torch.tensor(unique_sender_inds, device=img.device)
                
            #     print(unique_sender_inds)

            #     # Generate mask for unique sender instances
            #     unique_sender_mask = torch.tensor([(ind in unique_sender_inds) for ind in sender_inds], device=img.device)

            #     # Add unique sender data to ego's ground truth
            #     ego_inds = torch.cat([ego_inds, sender_inds[unique_sender_mask]], dim=0)
            #     ego_boxes = torch.cat([ego_boxes, sender_boxes[unique_sender_mask]], dim=0)
            #     ego_labels = torch.cat([ego_labels, sender_labels[unique_sender_mask]], dim=0)
            #     ego_past_traj = torch.cat([ego_past_traj, sender_past_traj[unique_sender_mask]], dim=0)
            #     ego_past_traj_mask = torch.cat([ego_past_traj_mask, sender_past_traj_mask[unique_sender_mask]], dim=0)

            # Update instance with merged ground truth values
            instance.obj_ids = ego_inds
            instance.boxes = ego_boxes
            instance.labels = ego_labels
            instance.past_traj = ego_past_traj
            instance.past_traj_mask = ego_past_traj_mask
            instance.sdc_boxes = torch.cat([sdc_boxes for _ in range(ego_boxes.shape[0])], dim=0)
            instance.sdc_labels = torch.cat([gt_sdc_label[0][frame_idx] for _ in range(ego_labels.shape[0])], dim=0)

            gt_instances_list.append(instance)
        #print('gt_instances_list._fields key:',gt_instances_list[0]._fields.keys())
        
        # init sender gt instances!
        sender_num_frame = sender_img.size(1)
        sender_gt_instances_list = [] 
        for frame_idx in range(sender_num_frame):
            # Initialize ground truth instances
            sender_instance = Instances((1, 1))

            # Process sender's ground truth bounding boxes
            sender_boxes = sender_gt_bboxes_3d[0][frame_idx].tensor.to(img.device)
            sender_boxes = normalize_bbox(sender_boxes, None)  # Normalize ego's bounding boxes
            sender_labels = sender_gxt_labels_3d[0][frame_idx]
            sender_inds = sender_gt_inds[0][frame_idx]
            sender_past_traj = sender_gt_past_traj[0][frame_idx].float()
            sender_past_traj_mask = sender_gt_past_traj_mask[0][frame_idx].float()
            sender_sdc_boxes = sender_gt_sdc_bbox[0][frame_idx].tensor.to(img.device)
            sender_sdc_boxes = normalize_bbox(sender_sdc_boxes, None)  # Normalize ego's sdc bounding boxes
            # Update instance with merged ground truth values
            sender_instance.obj_ids = sender_inds
            sender_instance.boxes = sender_boxes
            sender_instance.labels = sender_labels
            sender_instance.past_traj = sender_past_traj
            sender_instance.past_traj_mask = sender_past_traj_mask
            sender_instance.sdc_boxes = torch.cat([sender_sdc_boxes for _ in range(sender_boxes.shape[0])], dim=0)
            sender_instance.sdc_labels = torch.cat([sender_gt_sdc_label[0][frame_idx] for _ in range(sender_labels.shape[0])], dim=0)
            sender_gt_instances_list.append(sender_instance)

        # Initialize the criterion for single clip processing
        self.criterion.initialize_for_single_clip(gt_instances_list)
        self.criterion_sender.initialize_for_single_clip(sender_gt_instances_list)
        out = dict()
        ego_trajs = []
        sender_trajs = []
        fused_trajs = []
        for i in range(num_frame):
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            # TODO: Generate prev_bev in an RNN way.

            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            img_metas_single = [copy.deepcopy(img_metas[0][i])]
            if sender_img is not None and sender_img_metas is not None:
                try:
                    sender_img_single = torch.stack([img_[i] for img_ in sender_img], dim=0)
                    sender_img_metas_single = [copy.deepcopy(sender_img_metas[0][i])]
                except Exception as e:
                    sender_img_single = None
                    sender_img_metas_single = None

            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []
            # ego forward
            frame_res = self._forward_single_frame_train(
                img_single,
                img_metas_single,
                track_instances,
                sender_track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
                sender_img=sender_img_single if sender_img is not None else None,
                sender_img_metas=sender_img_metas_single if sender_img_metas is not None else None,
                gt_bboxes_3d=gt_bboxes_3d,
                sender_gt_bboxes_3d=sender_gt_bboxes_3d,
                
                vis_idx=i,
            )
            # all_query_embeddings: len=dec nums, N*256
            # all_matched_idxes: len=dec nums, N*2
            track_instances = frame_res["track_instances"]
            # print('track_instance._fields key:',track_instances._fields.keys())
            # for key in track_instances._fields:
            #     val = getattr(track_instances, key)
            #     print(f"{key}: shape = {val.shape if hasattr(val, 'shape') else 'N/A'}, type = {type(val)}")
            # ego_traj = track_instances.pred_boxes[:, :2].cpu().detach().numpy()
            # ego_trajs.append(ego_traj)
            # sender forward, obtain the sener track_instances
            # sender_frame_res = self._sender_forward_single_frame_train(
            #     sender_img_single,
            #     sender_img_metas_single,
            #     sender_track_instances,
            #     prev_img,
            #     prev_img_metas,
            #     l2g_r_mat[0][i],
            #     l2g_t[0][i],
            #     l2g_r2,
            #     l2g_t2,
            #     time_delta,
            #     all_query_embeddings,
            #     all_matched_idxes,
            #     all_instances_pred_logits,
            #     all_instances_pred_boxes,
            #     #pc_range=self.pc_range,
            # )
            # sender_track_instances = sender_frame_res["track_instances"]
            # sender_traj = sender_track_instances.pred_boxes[:, :2].cpu().detach().numpy()
            # sender_trajs.append(sender_traj)
            
            #  fusion tracking
            # if self.enable_track_fusion and (sender_img_single is not None):
            #     # Transform coordinates from sender to ego before normalizing
            #     sender_boxes = sender_gt_bboxes_3d[0][frame_idx].tensor.to(img.device)
            #     ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw = sender_img_metas[0][frame_idx]['can_bus'][:6]
            #     sender2ego_rt = build_sender2sego_rt(sender_boxes[:, :3], sender_boxes[:, 6], 
            #                                                                             ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw)
            #     track_nums_src = len(track_instances)
            #     fused_track_instances = self.cross_agent_query_interaction(sender_track_instances,      # sendr的 track
            #                                                         track_instances,             # ego 本车的 track
            #                                                         sender2ego_rt,      # 基于对方的标定信息
            #                                                         self.pc_range
            #                                                     )
            #     track_nums_new = len(fused_track_instances)
            #     add_nums = track_nums_new - track_nums_src
            #     # updata track_instances
            #     track_instances = fused_track_instances
            #     out["track_instances"] = track_instances
                # # 提取ego, sender, fused轨迹点
                # fused_traj = fused_track_instances.pred_boxes[:, :2].cpu().detach().numpy()
                # fused_trajs.append(fused_traj)  
                #print(f"ego tracks: {ego_traj}")
                #print(f"sender tracks: {sender_traj}")
                #print(f"fused tracks: {fused_traj}")
            # 画图比较
            # time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # save_dir = "/home/UNT/yz0364/cooper_uniad/output/"
            # save_path = os.path.join(save_dir, f"trajectory_plot_{time}.png")
            #visualize_all(gt_instances_list[i], track_instances, sender_track_instances, fused_track_instances)         

        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                    "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        out.update({k: frame_res[k] for k in get_keys})
        
        losses = self.criterion.losses_dict
        # losses_sender = self.criterion_sender.losses_dict
        # # 把二者合并/加权
        # total_losses = {}
        # total_losses.update(losses)
        # for k,v in losses_sender.items():
        #     total_losses[f"sender_{k}"] = v
        return losses, out

    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # For tiny model
            # bev_emb
            bev_embed = outs_track["bev_embed"] # [10000, 1, 256]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(bev_embed, '(h w) b c -> b c h w', h=h, w=w)  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)  # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, 'b c h w -> (h w) b c')
            outs_track["bev_embed"] = bev_embed

            # prev_bev
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    #  [1, 10000, 256]
                    prev_bev = rearrange(prev_bev, 'b (h w) c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> b (h w) c')
                    outs_track["prev_bev"] = prev_bev
                else:
                    #  [10000, 1, 256]
                    prev_bev = rearrange(prev_bev, '(h w) b c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, 'b c h w -> (h w) b c')
                    outs_track["prev_bev"] = prev_bev

            # bev_pos
            bev_pos  = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track

    def _sender_forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        sender_img=None,
        sender_img_metas=None,
        gt_bboxes_3d=None,
        sender_gt_bboxes_3d=None,

    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """

        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)
        det_output = self.pts_bbox_head.get_detections(
                bev_embed, 
                object_query_embeds=track_instances.query,
                ref_points=track_instances.ref_pts,
                img_metas=img_metas,
            )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query 
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, None)
    
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out


    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        sender_track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        sender_img=None,
        sender_img_metas=None,
        gt_bboxes_3d=None,
        sender_gt_bboxes_3d=None,
        enable_fusion_override=True,  # used for inference only 
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """
        #DEB
        ego_id = img_metas[0]['sample_idx'].split('/')[-2]
        sender_id = sender_img_metas[0][0]['sample_idx'].split('/')[-2] if sender_img_metas is not None else None

        print(f"ego_id: {ego_id}, sender_id: {sender_id}")

        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        if self.enable_fusion and enable_fusion_override:
            assert sender_img is not None and sender_img_metas is not None

            sender_img = sender_img[0]
            sender_img_metas = [item[0] for item in sender_img_metas]
            sender_bev_embed, sender_bev_pos = self.get_bevs(
                sender_img, sender_img_metas,
                prev_img=None, prev_img_metas=None,
            )
            
            if self.debug:
                backup_sender_bev = sender_bev_embed.clone()
            ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw = sender_img_metas[0]['can_bus'][:6]

            sender_bev_embed = rearrange(sender_bev_embed, '(h w) b c -> b c h w', h=200, w=200)
            sender_bev_embed = sender_bev_embed.squeeze(0)

            sender_bev_embed, transformed_mask = self.transform_bev(
                sender_bev_embed, ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw
            )

            sender_bev_embed = rearrange(sender_bev_embed, 'c h w -> (h w) 1 c')
            sender_mask = transformed_mask[None, ..., None]

            with torch.no_grad():
                fusion_bev_embed = self.fusion_bev_features(bev_embed, sender_bev_embed, sender_mask)
        else:
            fusion_bev_embed = None

         #DEB
        if self.debug:
            if self.enable_fusion:
                visualize_bev(bev_embed, backup_sender_bev, sender_bev_embed, fusion_bev_embed, gt_bboxes_3d, 
                              sender_gt_bboxes_3d, sender_img_metas, vis_idx=0, debug=self.debug)


        """ Process sender images if available """
        #if False:
        if self.enable_track_fusion and sender_img is not None and sender_img_metas is not None:
            print('------enable_track_fusion (test)------')
            #print('----type----',type(sender_img_metas[0]), sender_img_metas[0].keys())
            sender_frame_res = self._sender_forward_single_frame_inference(
                sender_img,
                sender_img_metas,
                sender_track_instances,
                prev_bev,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta,  # time_delta is used for velo update
                sender_img=None,
                sender_img_metas=None,
            )
            #sender_track_instances = sender_frame_res["track_instances"]
            sender_track_instances = sender_frame_res["track_instances_fordet"]
            self.test_sender_track_instances = sender_track_instances
            """ Perform track fusion if enabled """
            track_nums_src = len(track_instances)
            if isinstance(sender_gt_bboxes_3d[0], list):
                sender_gt_bboxes_3d[0] = LiDARInstance3DBoxes.cat(sender_gt_bboxes_3d[0])
            sender_boxes = sender_gt_bboxes_3d[0].tensor.to(img.device)
            ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw = sender_img_metas[0]['can_bus'][:6]

            sender_transformed_pos, sender_transformed_yaw = transform_coordinates(sender_boxes[:, :3], sender_boxes[:, 6], 
                                                                ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw)
            sender2ego_rt = get_sender2ego_4x4(sender_x, sender_y, sender_yaw, ego_x, ego_y, ego_yaw)
            print('sender2ego_rt:', sender2ego_rt)
            N = sender_boxes.shape[0]
            pts_sender = sender_boxes[:, :3]
            pts_sender_homo = torch.cat([
                                            pts_sender,
                                            torch.ones((N, 1), device=pts_sender.device)
                                        ], dim=1)
            pts_sender_homo = pts_sender_homo.unsqueeze(-1)        # (N, 4, 1)
            sender2ego_rt = sender2ego_rt.to(pts_sender.device)
            pts_ego_mat1 = torch.matmul(sender2ego_rt, pts_sender_homo)  # (N, 4, 1)
            pts_ego_mat1 = pts_ego_mat1.squeeze(-1)[:, :3] 
            print('result of get_sender2ego_4x4 in test:', pts_ego_mat1)
            print("result of transform_coordinates in test:", sender_transformed_pos)

            fused_track_instances = self.cross_agent_query_interaction(
                sender_track_instances,
                track_instances,
                sender2ego_rt,
                self.pc_range
            )
            track_nums_new = len(fused_track_instances)
            add_nums = track_nums_new - track_nums_src
            track_instances = fused_track_instances
            if self.enable_track_fusion:
                fusion_bev_embed,bev_pos = self._get_coop_bev_embed(fusion_bev_embed, bev_pos, track_instances, track_nums_new-add_nums)
            else:
                fusion_bev_embed,bev_pos = self._get_coop_bev_embed(bev_embed, bev_pos, track_instances, track_nums_new-add_nums) 

        if (self.enable_fusion or self.enable_track_fusion) and enable_fusion_override:
            det_output = self.pts_bbox_head.get_detections(
                fusion_bev_embed, 
                object_query_embeds=track_instances.query,
                ref_points=track_instances.ref_pts,
                img_metas=img_metas,
            )
        else:
            det_output = self.pts_bbox_head.get_detections(
                bev_embed, 
                object_query_embeds=track_instances.query,
                ref_points=track_instances.ref_pts,
                img_metas=img_metas,
            )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        # track_instances.track_scores = track_scores  # [300]
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query 
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, None)
       
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
        sender_img=None, 
        sender_img_metas=None,
        gt_bboxes_3d=None,
        sender_gt_bboxes_3d=None, 

    ):
        """only support bs=1 and sequential input"""

        bs = img.size(0)
        # img_metas = img_metas[0]
        #print('---img_metas----', img_metas)
        #print('---filename of img metas (test)----', img_metas[0]['filename'])
        #print('---sender_img_metas----', sender_img_metas)
        #print('----filename of sendr_img_metas (test)----', sender_img_metas[0][0]['filename'])

        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
            self.prev_bev = self.prev_frame_info['prev_bev']
        if self.test_sender_track_instances is None:
            sender_track_instances = self._generate_empty_tracks()
        else:
            sender_track_instances = self.test_sender_track_instances
            
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            sender_track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            sender_img,
            sender_img_metas,
            gt_bboxes_3d,
            sender_gt_bboxes_3d,
            #enable_fusion_override=True
        )

        self.prev_bev = frame_res["bev_embed"]
        self.prev_frame_info['prev_bev'] = self.prev_bev
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]
        self.test_track_instances = track_instances

        results = [dict()]
        get_keys = ["bev_embed", "bev_pos", 
                    "track_query_embeddings", "track_bbox_results", 
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        if self.with_motion_head:
            get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        results[0].update({k: frame_res[k] for k in get_keys})
        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        print('----boxes_3d.center------',results[0]['boxes_3d'].center)

        return results
    
    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        # bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask)[0]
        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # filter out sleep querys
        if instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]
