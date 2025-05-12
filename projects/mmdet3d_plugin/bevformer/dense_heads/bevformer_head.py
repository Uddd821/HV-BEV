# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmdet.models import build_loss


@HEADS.register_module()
class BEVFormerHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_height=6,
                 pc_range=None,
                 loss_height=dict(
                     type='SoftCrossEntropyLoss',
                     reduction='mean',
                     loss_weight=0.25),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.num_height = num_height
        self.pc_range = pc_range
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_height = build_loss(loss_height)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)  # (200*200, 256)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)  # (900, 512)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False): # prev_height=None,
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape  # (1, 6, 256, H, W)
        dtype = mlvl_feats[0].dtype  # torch.float32
        object_query_embeds = self.query_embedding.weight.to(dtype)  # 物体query，(900, 512)，900个长度为512
        bev_queries = self.bev_embedding.weight.to(dtype)  # BEV query, (200*200, 256)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)  # (1, 200, 200), 0值
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # (1, 256, 200, 200)，位置编码

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,  # 多尺寸图像特征, (1, 6, 256, H, W)
                bev_queries,  # (40000, 256)
                self.bev_h,  # 200
                self.bev_w,  # 200
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),  # 网格尺寸，每一个bev网格的真实尺寸, (102.4/200=0.512, 0.512)
                bev_pos=bev_pos,  # (1, 256, 200, 200)
                img_metas=img_metas,  # 图像信息
                prev_bev=prev_bev,  # history bev
                # prev_height=prev_height,
            )  # (1, 40000, 256), (6, 1, 40000, 6)
        else:
            outputs = self.transformer(
                mlvl_feats,  # 多尺寸图像特征, (1, 6, 256, H, W)
                bev_queries,  # (40000, 256)
                object_query_embeds,  # 物体query，(900, 512)
                self.bev_h,  # 200
                self.bev_w,  # 200
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),  # 2.048
                bev_pos=bev_pos,  # (1, 256, 200, 200)
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501 6层
                cls_branches=self.cls_branches if self.as_two_stage else None,  # 6层
                img_metas=img_metas,  # 图像信息，cur_frame
                prev_bev=prev_bev,  # 历史信息 (1, 40000, 256)
                # prev_height=prev_height
            )
        """
        bev_embed:(40000, 1, 256) bev的拉直嵌入
        height_distribution:(6, 1, 40000, 6) bev的高度分布估计
        hs:(6, 900, 1, 256) 内部decoder layer输出的object query
        init_reference_out:(1, 900, 3) 随机初始化的参考点
        inter_references_out:(6, 1, 900, 3) 内部decoder layer输出的参考点
        """
        bev_embed, height_distribution, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)  # (6, 900, 1, 256) -> (6, 1, 900, 256)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):  # 6层都处理
            if lvl == 0:
                reference = init_reference  # (1, 900, 3)
            else:
                reference = inter_references[lvl - 1]  # (1, 900, 3) sigmoid值
            reference = inverse_sigmoid(reference)  # 逆sigmoid
            outputs_class = self.cls_branches[lvl](hs[lvl])  # (1, 900, 256) -> (1, 900, 10)
            tmp = self.reg_branches[lvl](hs[lvl])  # (1, 900, 256) -> (1, 900, 10)

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            # 参考点 + 偏移量
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            # 恢复到lidar坐标
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                                              self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                                              self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                                              self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)  # 将分类预测加入outputs list (1, 900, 10)
            outputs_coords.append(outputs_coord)  # 将回归预测加入outputs list (1, 900, 10)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,  # (40000, 1, 256)
            'height_distribution': height_distribution,  # (6, 1, 40000, 6)
            'all_cls_scores': outputs_classes,  # (6, 1, 900, 10)
            'all_bbox_preds': outputs_coords,  # (6, 1, 900, 3)
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)  # 1 batch_size
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # 1.按照batch处理,计算分类和回归的targets
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # 在batch维度进行拼接
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        # 2.计算该层的分类和回归损失
        # 2.1 classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        # 2.2regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        # 计算回归损失 只计算匹配上gt的预测框损失
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                       :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)  # 将nan替换为0
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_single_inter(self, height_distribution, gt_height_list):

        device = gt_height_list[0].device
        num_imgs = height_distribution.size(0)

        loss_height_sum = torch.tensor([0.0], device=device)
        for bs in range(num_imgs):
            gt_height = gt_height_list[bs]
            bev_ind = gt_height[:, :2].long()
            height_pred = height_distribution[bs]
            Z = self.pc_range[5] - self.pc_range[2]
            z_bins = torch.linspace(0.5, Z - 0.5, self.num_height, dtype=gt_height.dtype, device=device)
            gt_z_values = gt_height[:, 2]

            # 计算每个 ground truth 高度与所有 z_bins 的差异 (N, 6)
            z_diff = torch.abs(gt_z_values.unsqueeze(1) - z_bins.unsqueeze(0))
            sigma = 1.0  # 高斯核的标准差
            z_weights = torch.exp(- (z_diff ** 2) / (2 * sigma ** 2))
            z_prob = z_weights / z_weights.sum(dim=1, keepdim=True)  # (N, 6)

            loss_height = self.loss_height(height_pred, z_prob, bev_ind)
            loss_height_sum += loss_height

        loss_height = loss_height_sum / num_imgs
        # loss_height = gt_height.new_tensor([loss_height.item()])

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_height = torch.nan_to_num(loss_height)
        return loss_height

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_height_distribution = preds_dicts['height_distribution']
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_enc_layers = len(all_height_distribution)
        num_dec_layers = len(all_cls_scores)
        bs, num_height = all_cls_scores.size(1), all_height_distribution.size(-1)
        device = gt_labels_list[0].device

        height_distribution = all_height_distribution.view(num_enc_layers, bs, self.bev_w, self.bev_h, num_height)

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        gt_height_list = []
        bev_grid_size_w = self.real_w / self.bev_w
        bev_grid_size_h = self.real_h / self.bev_h
        for gt_bboxes in gt_bboxes_list:
            gt_centers = gt_bboxes[:, :3]
            # gt_width = gt_bboxes[:, 3]  # 宽度 (沿y轴)
            # gt_length = gt_bboxes[:, 4]  # 长度 (沿x轴)
            # gt_yaw = gt_bboxes[:, 6]  # 偏航角（绕z轴的旋转）
            #
            # # 计算边界框四个角点相对于中心的偏移量
            # dx = gt_length / 2
            # dy = gt_width / 2
            # dx = dx.unsqueeze(1)
            # dy = dy.unsqueeze(1)
            # corner_offsets = torch.tensor([
            #     [1, 1],
            #     [1, -1],
            #     [-1, -1],
            #     [-1, 1]
            # ], device=gt_bboxes.device).float()  # shape (4, 2)
            # corner_offsets = corner_offsets.unsqueeze(0)  # (1, 4, 2)
            # corner_offsets = corner_offsets.repeat(gt_centers.size(0), 1, 1)  # (N, 4, 2)
            # corner_offsets *= torch.cat([dx, dy], dim=1).unsqueeze(1)  # (N, 4, 2)
            #
            # # 计算旋转矩阵
            # cos_yaw = torch.cos(gt_yaw)
            # sin_yaw = torch.sin(gt_yaw)
            # R = torch.stack((cos_yaw, -sin_yaw, sin_yaw, cos_yaw), dim=-1).reshape(-1, 2, 2)
            #
            # # 获取每个边界框的四个角点在 BEV 平面中的绝对坐标
            # corners = R @ corner_offsets.permute(0, 2, 1)  # (N, 2, 4)
            # corners += gt_centers[:, :2].unsqueeze(-1)  # 中心偏移
            #
            # # 将角点坐标转换为 BEV 网格索引坐标
            # y_shifted = (self.pc_range[3] - corners[:, 0, :]) /  bev_grid_size_h
            # x_shifted = (self.pc_range[4] - corners[:, 1, :])  / bev_grid_size_w
            # # x_shifted = (corners[:, 0, :] - self.pc_range[0]) / bev_grid_size_h  # (N, 4)
            # # y_shifted = (corners[:, 1, :] - self.pc_range[1]) / bev_grid_size_w  # (N, 4)
            # # 转换为整数索引，并限制在 BEV 网格范围内
            # grid_x = torch.clamp(x_shifted.floor().long(), 0, self.bev_w - 1)
            # grid_y = torch.clamp(y_shifted.floor().long(), 0, self.bev_h - 1)
            #
            # # 获取每个边界框的BEV网格覆盖范围
            # min_x = torch.min(grid_x, dim=1)[0]  # (N,)
            # max_x = torch.max(grid_x, dim=1)[0]  # (N,)
            # min_y = torch.min(grid_y, dim=1)[0]  # (N,)
            # max_y = torch.max(grid_y, dim=1)[0]  # (N,)
            #
            # # 生成体积覆盖的 BEV 网格索引，并计算对应的高度
            # gt_bev_height = []
            # for i in range(len(min_x)):
            #     x_indices = torch.arange(min_x[i], max_x[i] + 1, device=min_x.device)
            #     y_indices = torch.arange(min_y[i], max_y[i] + 1, device=min_x.device)
            #     grid_x, grid_y = torch.meshgrid(x_indices, y_indices)
            #
            #     # 展平并将网格索引与高度对应
            #     grid_indices = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # shape (num_grids, 2)
            #     heights = torch.full((grid_indices.shape[0], 1), gt_centers[i, 2] - self.pc_range[2], device=grid_indices.device)  # shape (num_grids, 1)
            #     gt_bev_height.append(torch.cat([grid_indices, heights], dim=1))  # shape (num_grids, 3)
            #
            # # 合并所有边界框的结果
            # gt_bev_height = torch.cat(gt_bev_height, dim=0)  # shape (M, 3)
            # gt_height_list.append(gt_bev_height)

            gt_bev_w_ind = torch.tensor(((gt_centers[..., 0] - self.pc_range[0]) / self.real_w) * self.bev_w, dtype=int)
            gt_bev_h_ind = torch.tensor(((gt_centers[..., 1] - self.pc_range[1]) / self.real_h) * self.bev_h, dtype=int)
            gt_bev_height = torch.stack([gt_bev_w_ind, gt_bev_h_ind, gt_bboxes[:, 2] - self.pc_range[2]], dim=1)
            gt_height_list.append(gt_bev_height)

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_height_list = [gt_height_list for _ in range(num_enc_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        losses_height = multi_apply(self.loss_single_inter,
                    height_distribution, all_gt_height_list)
        losses_height = losses_height[0]

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_height'] = losses_height[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        num_enc_layer = 0
        for loss_height_i in losses_height[:-1]:
            loss_dict[f'd{num_enc_layer}.loss_height'] = loss_height_i
            num_enc_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list
