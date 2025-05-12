# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Di Wu
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader, digit_version, TORCH_VERSION
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from torch.cuda import empty_cache

# import time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class GCN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GraphStructureLearning(nn.Module):

    def __init__(self, num_nodes, in_channels):
        super(GraphStructureLearning, self).__init__()
        # 使用一个线性层来学习边权重
        self.fc = nn.Linear(in_channels * num_nodes, num_nodes * num_nodes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch_size, num_nodes, in_channels)
        batch_size, num_nodes, in_channels = x.size()

        # Flatten the node features to (batch_size, num_nodes * in_channels)
        x = x.view(batch_size, -1)

        # 学习边权重
        edge_weights = self.fc(x)  # (batch_size, num_nodes * num_nodes)

        # reshape edge_weights to (batch_size, num_nodes, num_nodes)
        edge_weights = edge_weights.view(batch_size, num_nodes, num_nodes)

        edge_weights = torch.sigmoid(edge_weights)

        return edge_weights


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False)

    def forward(self, x, edge_index):
        # First GAT layer
        x = F.relu(self.gat1(x, edge_index))
        # Second GAT layer
        x = self.gat2(x, edge_index)
        return x


@ATTENTION.register_module()
class GraphSpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 num_points_in_pillar=4,
                 num_nodes=4,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 # gcn_batch_size=64,  # 添加分批大小参数
                 # deformable_attention_batch_size=64,  # 添加分批大小参数
                 device='cuda',
                 **kwargs
                 ):
        super(GraphSpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.num_points_in_pillar = num_points_in_pillar
        self.num_nodes = num_nodes
        # self.gcn_batch_size = gcn_batch_size
        # self.deformable_attention_batch_size = deformable_attention_batch_size
        self.sampling_offsets = nn.Linear(embed_dims, num_points_in_pillar * num_nodes * 2)
        self.attention_weights = nn.Linear(embed_dims, num_points_in_pillar * num_nodes)  # 4*4
        # self.gsl = GraphStructureLearning(num_nodes, embed_dims) # 图结构学习模块
        # self.gcn = GCN(embed_dims, embed_dims // 2, embed_dims // 4) # 图卷积模块
        # self.gat = GAT(embed_dims, embed_dims // 2, embed_dims // 4, heads=4) # 图注意力模块
        self.fusion_weights = nn.Linear(embed_dims * 2, 2)
        self.init_weight()
        self.device = device

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        xavier_init(self.sampling_offsets, distribution='uniform', bias=0.)
        xavier_init(self.attention_weights, distribution='uniform', bias=0.)
        xavier_init(self.fusion_weights, distribution='uniform', bias=0.)
        self._is_init = True

    @force_fp32(apply_to=('ref_points_neighbors', 'img_metas'))
    def reference_point_sampling(self, ref_points_neighbors, pc_range, img_metas):
        # ref_points_neighbors:(1, 40000, 4, 4, 3), lidar2img: lidar -> ego -> camera -> 内参 -> img
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = ref_points_neighbors.new_tensor(lidar2img)
        ref_points_neighbors = ref_points_neighbors.clone()

        ref_points_neighbors[..., 0:1] = ref_points_neighbors[..., 0:1] * \
                                         (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_points_neighbors[..., 1:2] = ref_points_neighbors[..., 1:2] * \
                                         (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_points_neighbors[..., 2:3] = ref_points_neighbors[..., 2:3] * \
                                         (pc_range[5] - pc_range[2]) + pc_range[2]

        ref_points_neighbors = torch.cat(
            (ref_points_neighbors, torch.ones_like(ref_points_neighbors[..., :1])),
            -1)  # cat: (1, 40000, 4, 4, 1), ref_points_neighbors: (1, 40000, 4, 4, 4)
        ref_points_neighbors = ref_points_neighbors.permute(2, 0, 1, 3, 4)  # (4, 1, 40000, 4, 4)
        D, B, num_query, num_neighbors = ref_points_neighbors.size()[:4]
        num_cam = lidar2img.size(1)
        ref_points_neighbors = ref_points_neighbors.view(
            D, B, 1, num_query, num_neighbors, 4).repeat(1, 1, num_cam, 1, 1, 1).unsqueeze(
            -1)  # (4, 1, 40000, 4, 4) -> (4, 1, 1, 40000, 4, 4) -> (4, 1, 6, 40000, 4, 4) -> (4, 1, 6, 40000, 4, 4, 1)

        lidar2img_neighbors = lidar2img.view(
            1, B, num_cam, 1, 1, 4, 4).repeat(D, 1, 1, num_query, num_neighbors,
                                              1,
                                              1)  # (1, 6, 4, 4) -> (1, 1, 6, 1, 1, 4, 4) -> (4, 1, 6, 40000, 4, 4, 4)

        ref_points_neighbors_cam = torch.matmul(lidar2img_neighbors.to(torch.float32),
                                                ref_points_neighbors.to(torch.float32)).squeeze(
            -1)  # 矩阵乘法, (4, 1, 6, 40000, 4, 4) 投影到相机上是一个4维量（ud,vd,d,1）
        eps = 1e-5

        bev_neighbors_mask = (ref_points_neighbors_cam[..., 2:3] > eps)  # (4, 1, 6, 40000, 4, 1)
        ref_points_neighbors_cam = ref_points_neighbors_cam[..., 0:2] / torch.maximum(
            ref_points_neighbors_cam[..., 2:3],
            torch.ones_like(ref_points_neighbors_cam[..., 2:3]) * eps)  # (4, 1, 6, 40000, 4, 2)

        ref_points_neighbors_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        ref_points_neighbors_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_neighbors_mask = (bev_neighbors_mask & (ref_points_neighbors_cam[..., 1:2] > 0.0)
                              & (ref_points_neighbors_cam[..., 1:2] < 1.0)
                              & (ref_points_neighbors_cam[..., 0:1] < 1.0)
                              & (ref_points_neighbors_cam[..., 0:1] > 0.0))  # (4, 1, 6, 40000, 4, 1)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_neighbors_mask = torch.nan_to_num(bev_neighbors_mask)  # nan->number, default 0
        else:
            bev_neighbors_mask = bev_neighbors_mask.new_tensor(
                np.nan_to_num(bev_neighbors_mask.cpu().numpy()))

        ref_points_neighbors_cam = ref_points_neighbors_cam.permute(2, 1, 3, 0, 4,
                                                                    5)  # (4, 1, 6, 40000, 4, 2) -> (6, 1, 40000, 4, 4, 2)
        bev_neighbors_mask = bev_neighbors_mask.permute(2, 1, 3, 0, 4, 5).squeeze(
            -1)  # (4, 1, 6, 40000, 4, 1) -> (6, 1, 40000, 4, 4, 1) -> (6, 1, 40000, 4, 4)

        return ref_points_neighbors_cam, bev_neighbors_mask

    @force_fp32(apply_to=('reference_points', 'value'))
    def multi_scale_bilinear_interpolate(self, value, spatial_shapes, level_start_index, reference_points, img_shape):
        """
        Args:
            value: 多尺度特征图，尺寸为 (bs, num_keys, embed_dims)
            spatial_shapes: 尺寸为 (num_levels, 2) 的张量，表示每个尺度的特征图的高度和宽度 (H_l, W_l)。
            level_start_index: 每个尺度特征图在拼接后的 value 中的起始索引，尺寸为 (num_levels, )
            reference_points: 参考点在原始图像上的归一化坐标，尺寸为 (bs, num_queries, 2)
            img_shapes: 原始图像的尺寸，尺寸为 (bs, 2)，表示高度和宽度 (H_img, W_img)

        Returns:
            output: 经过多尺度采样后更新的 query 特征，尺寸为 (bs, num_queries, embed_dims)
        """

        bs, num_keys, embed_dims = value.shape
        num_levels = spatial_shapes.size(0)
        num_queries = reference_points.size(1)
        reference_points = reference_points.clone()
        reference_points = reference_points[:, :, [1, 0]]

        # 初始化输出张量
        output = torch.zeros(bs, num_queries, embed_dims, device=value.device)

        for lvl in range(num_levels):
            # 获取当前尺度特征图的高度和宽度
            H_l, W_l = spatial_shapes[lvl]
            start_idx = level_start_index[lvl]
            end_idx = level_start_index[lvl + 1] if lvl + 1 < num_levels else num_keys

            # 当前尺度的特征图 (bs, H_l, W_l, embed_dims)
            value_lvl = value[:, start_idx:end_idx, :].view(bs, H_l, W_l, embed_dims)

            # 将参考点从原始图像归一化坐标映射到当前特征图的归一化坐标
            ref_points_lvl = reference_points.view(bs, num_queries, 1, 2)  # (bs, num_queries, 1, 2)
            ref_points_lvl = ref_points_lvl * 2 - 1  # 将坐标从 [0,1] 映射到 [-1,1]

            # 使用 grid_sample 进行双线性插值采样
            sampled_features = F.grid_sample(
                value_lvl.permute(0, 3, 1, 2),  # (bs, embed_dims, H_l, W_l)
                ref_points_lvl,  # (bs, num_queries, 1, 2)
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            # 将采样到的特征结果合并到输出
            output += sampled_features.squeeze(3).permute(0, 2, 1)  # (bs, num_queries, embed_dims)

        # 可以根据需要对输出进行平均
        output = output / num_levels

        return output

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points', 'reference_points_cam'))
    # @auto_fp16(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                bev_h,
                bev_w,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                spatial_shapes=None,
                reference_points=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots_refpts = torch.zeros_like(query)
            slots_nbs = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        l = key.size(1)

        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l,
                                              self.embed_dims)  # (6, 19560, 1, 256) -> (1, 6, 19560, 256) -> (6, 19560, 256)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l,
                                                  self.embed_dims)  # (6, 19560, 1, 256) -> (1, 6, 19560, 256) -> (6, 19560, 256)

        D = reference_points_cam.size(3)
        N = self.num_nodes

        indexes = []
        for i, mask_per_img in enumerate(bev_mask):  # bev_mask: (6, 1, 40000, 4)
            # mask_per_img: (1, 40000, 4) 单相机mask
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(
                -1)  # (1, 40000, 4) -> (40000, 4) -> (40000, ) -> (6311, 1) -> (6311, ) 每个相机图像上非零mask的索引值（即投影命中的points）
            indexes.append(index_query_per_img)  # 每个视角命中的BEV索引
        max_len = max([len(each) for each in indexes])  # 不同相机视角index最大长度

        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])  # (1, 6, max_len, 256) 在不同视角有命中的BEV queries
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])  # (1, 6, max_len, 4, 2) 在不同视角命中的2D参考点坐标
        bev_mask_rebatch = bev_mask.new_zeros([bs, self.num_cams, max_len, D])  # (1, 6, max_len, 4)

        for j in range(bs):  # per_batch
            for i, reference_points_per_img in enumerate(reference_points_cam):  # per_cam
                index_query_per_img = indexes[i]  # (6311, )
                # queries_rebatch[j, i, :len(index_query_per_img)]: j-th batch, i-th cam, max_len
                # query[j, index_query_per_img]: query, (1, 40000, 256), j-th batch, query_index, 从对应index采样query特征，放到queries_rebatch里面
                queries_rebatch[j, i, :len(index_query_per_img)] = query[
                    j, index_query_per_img]  # (1, 6, max_len, 256) 在不同视角有命中的BEV queries
                # 从对应index采样2D reference_points，放到reference_points_rebatch里面
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]  # (1, 6, max_len, 4, 2) 在不同视角命中的2D参考点坐标
                bev_mask_rebatch[j, i, :len(index_query_per_img)] = bev_mask[i, j, index_query_per_img]

        # query: (1, 6, max_len, 256) -> (6, max_len, 256) 只有命中的BEV query参与交叉注意提取特征
        # key: (6, 19560, 256)
        # value: (6, 19560, 256)
        # ref_p: (1, 6, max_len, 4, 2) -> (6, max_len, 4, 2)
        # spatial_shapes: [[92,160], [46,80], [23,40], [12,20]]
        # level_start_index: (4,)
        queries = self.deformable_attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                            key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len,
                                                                                           D, 2),
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len,
                                                                                      self.embed_dims)  # (1, 6, max_len, 256)

        neighbors_offset = self.sampling_offsets(query.view(bs * num_query, -1)).view(bs, num_query, D, N,
                                                                                      2)  # (1, 40000, 4, 4, 2)
        offset_z = neighbors_offset.new_zeros([bs, num_query, D, N, 1])  # (1, 40000, 4, 4, 1)
        neighbors_offset = torch.cat([neighbors_offset, offset_z], dim=-1)  # (1, 40000, 4, 4, 3)
        neighbors_offset = neighbors_offset / torch.tensor([bev_h, bev_w, 1], dtype=torch.float, device=self.device)
        reference_points_neighbors = reference_points.unsqueeze(3).repeat(1, 1, 1, N,
                                                                          1) + neighbors_offset  # (1, 40000, 4, 4, 3)

        ref_points_neighbors_cam, bev_neighbors_mask = self.reference_point_sampling(
            reference_points_neighbors, self.pc_range,
            kwargs['img_metas'])  # (6, 1, 40000, 4, 4, 2), (6, 1, 40000, 4, 4)

        ref_points_neighbors_cam_rebatch = ref_points_neighbors_cam.new_zeros(
            [bs, self.num_cams, max_len, D, N, 2])  # (1, 6, max_len, 4, 4, 2)
        # bev_neighbors_mask_rebatch = bev_neighbors_mask.new_zeros([bs, self.num_cams, max_len, D, N]) # (1, 6, max_len, 4, 4)
        for j in range(bs):
            for i, reference_points_neighbors_per_img in enumerate(ref_points_neighbors_cam):
                index_query_per_img = indexes[i]  # (6311, )
                ref_points_neighbors_cam_rebatch[j, i, :len(index_query_per_img)] = reference_points_neighbors_per_img[
                    j, index_query_per_img]
                # bev_neighbors_mask_rebatch[j, i, :len(index_query_per_img)] = bev_neighbors_mask[i, j, index_query_per_img]

        img_shape = torch.tensor([kwargs['img_metas'][0]['img_shape'][0][0], kwargs['img_metas'][0]['img_shape'][0][1]],
                                 device=self.device)
        neighbors_features = self.multi_scale_bilinear_interpolate(value, spatial_shapes, level_start_index,
                                                                   ref_points_neighbors_cam_rebatch.view(
                                                                       bs * self.num_cams, max_len * D * N, 2),
                                                                   img_shape[None].repeat(bs * self.num_cams, 1))
        neighbors_features = neighbors_features.view(bs, self.num_cams, max_len, D * N,
                                                     self.embed_dims)  # (1, 6, max_len, 4 * 4, 256)

        attention_weights = self.attention_weights(queries.view(bs * self.num_cams, max_len, self.embed_dims)).view(
            bs, self.num_cams, max_len, self.num_points_in_pillar * self.num_nodes)  # (1, 6, max_len, 4*4)
        attention_weights = attention_weights.softmax(-1).unsqueeze(-1)  # (1, 6, max_len, 4*4, 1)
        fused_neighbors_features = neighbors_features * attention_weights  # (1, 6, max_len, 4*4, 256)
        fused_neighbors_features = fused_neighbors_features.sum(3)  # (1, 6, max_len, 256)

        for j in range(bs):  # per_batch
            for i, index_query_per_img in enumerate(indexes):  # per_img
                # j-th batch, 对应index += queries特征
                slots_refpts[j, index_query_per_img] += queries[j, i, :len(
                    index_query_per_img)]  # 针对不同视角更新后的query直接相加作为融合的BEV Queries

        for j in range(bs):  # per_batch
            for i, index_query_per_img in enumerate(indexes):  # per_img
                # j-th batch, 对应index += queries特征
                slots_nbs[j, index_query_per_img] += fused_neighbors_features[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0  # (6, 1, 40000, 4) -> (6, 1, 40000) 高度求和
        count = count.permute(1, 2, 0).sum(-1)  # (6, 1, 40000) -> (1, 40000, 6) -> (1, 40000) 相机维度求和
        count = torch.clamp(count, min=1.0)  # (1, 40000)
        slots_refpts = slots_refpts / count[..., None]  # 取均值，(1, 40000, 256) / (1, 40000, 1) = (1, 40000, 256)

        count = bev_neighbors_mask.contiguous().view(self.num_cams, bs, num_query, -1).sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots_nbs = slots_nbs / count[..., None]  # (1, 40000, 256)

        slots_refpts_nbs = torch.cat([slots_refpts, slots_nbs], dim=2)  # (1, 40000, 512)
        fusion_weights = self.fusion_weights(slots_refpts_nbs).softmax(-1).unsqueeze(-1)  # (1, 40000, 2, 1)
        combined_slots = torch.stack([slots_refpts, slots_nbs], dim=2)  # (1, 40000, 2, 256)
        slots = (combined_slots * fusion_weights).sum(2)  # (1, 40000, 256)

        #         #neighbors_features = neighbors_features.sum(0) / self.num_cams # (24422, 4, 256) 需要取均值吗？取均值是除以6还是按实际命中
        #         pts_nbs_features = torch.cat([valid_pts_features[:, None, :], neighbors_features], dim=1) # (24422, 1+4, 256)
        #         gcn_input_batches = torch.split(pts_nbs_features, self.gcn_batch_size)
        #         gcn_output = []
        #
        #         for batch in gcn_input_batches:
        #             data = self.create_data_object(batch)
        #             gcn_output.append(self.gcn(data.x, data.edge_index))
        #         if len(pts_nbs_features) % self.gcn_batch_size != 0:
        #             last_batch = pts_nbs_features[len(gcn_input_batches) * self.gcn_batch_size:]
        #             data = self.create_data_object(last_batch)
        #             gcn_output.append(self.gcn(data.x, data.edge_index))
        #         pts_nbs_features = torch.cat(gcn_output)
        #         pts_nbs_features = self.output_proj2(pts_nbs_features) # (24422, 5, 256)
        #         queries_features[valid_points_mask[0], valid_points_mask[1]] = pts_nbs_features[:, 0] # (6311, 4, 256)
        #
        #         slots[b, index_query_per_img] += queries_features.sum(1)

        slots = self.output_proj(slots)  # 线性层, (1, 40000, 256)

        return self.dropout(slots) + inp_residual  # (1, 40000, 256)

    def create_data_object(self, features):
        edge_index = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4],
                                   [1, 2, 3, 4, 0, 0, 0, 0]], device=self.device)
        edge_attr = torch.ones(edge_index.size(1), device=self.device)
        return Data(x=features, edge_index=edge_index, edge_attr=edge_attr)


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos  # (1, 40000, 256) + (1, 40000, 256) -> (1, 40000, 256)

        bs, num_query, _ = query.size()  # 1, 40000, 256

        D = reference_points_cam.size(3)  # (6, 1, 40000, 4, 2), D=4 每个pillar采样点的个数
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):  # bev_mask: (6, 1, 40000, 4)
            # mask_per_img: (1, 40000, 4) 单相机mask
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(
                -1)  # (1, 40000, 4) -> (40000, 4) -> (40000, ) -> (6311, 1) -> (6311, ) 每个相机图像上非零mask的索引值（即投影命中的points）
            indexes.append(index_query_per_img)  # 每个视角命中的BEV索引
        max_len = max([len(each) for each in indexes])  # 不同相机视角index最大长度

        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])  # (1, 6, max_len, 256) 在不同视角有命中的BEV queries
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])  # (1, 6, max_len, 4, 2) 在不同视角命中的2D参考点坐标

        for j in range(bs):  # per_batch
            for i, reference_points_per_img in enumerate(reference_points_cam):  # per_cam
                index_query_per_img = indexes[i]  # (6311, )
                # queries_rebatch[j, i, :len(index_query_per_img)]: j-th batch, i-th cam, max_len
                # query[j, index_query_per_img]: query, (1, 40000, 256), j-th batch, query_index, 从对应index采样query特征，放到queries_rebatch里面
                queries_rebatch[j, i, :len(index_query_per_img)] = query[
                    j, index_query_per_img]  # (1, 6, max_len, 256) 在不同视角有命中的BEV queries
                # 从对应index采样2D reference_points，放到reference_points_rebatch里面
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]  # (1, 6, max_len, 4, 2) 在不同视角命中的2D参考点坐标

        num_cams, l, bs, embed_dims = key.shape  # 6, 15*25, 1, 256

        # key和value都是的多视角多尺度图像特征
        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)  # (6, 19560, 1, 256) -> (1, 6, 19560, 256) -> (6, 19560, 256)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)  # (6, 19560, 1, 256) -> (1, 6, 19560, 256) -> (6, 19560, 256)

        # query: (1, 6, max_len, 256) -> (6, max_len, 256) 只有命中的BEV query参与交叉注意提取特征
        # key: (6, 19560, 256)
        # value: (6, 19560, 256)
        # ref_p: (1, 6, max_len, 4, 2) -> (6, max_len, 4, 2)
        # spatial_shapes: [[92,160], [46,80], [23,40], [12,20]]
        # level_start_index: (4,)
        queries = self.deformable_attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                            key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len,
                                                                                           D, 2),
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len,
                                                                                      self.embed_dims)  # (1, 6, max_len, 256)

        for j in range(bs):  # per_batch
            for i, index_query_per_img in enumerate(indexes):  # per_img
                # j-th batch, 对应index += queries特征
                slots[j, index_query_per_img] += queries[j, i,
                                                 :len(index_query_per_img)]  # 针对不同视角更新后的query直接相加作为融合的BEV Queries

        count = bev_mask.sum(-1) > 0  # (6, 1, 40000, 4) -> (6, 1, 40000) 高度求和
        count = count.permute(1, 2, 0).sum(-1)  # (6, 1, 40000) -> (1, 40000, 6) -> (1, 40000) 相机维度求和
        count = torch.clamp(count, min=1.0)  # (1, 40000)
        slots = slots / count[..., None]  # 取均值，(1, 40000, 256) / (1, 40000, 1) slot: (1, 40000, 256)
        slots = self.output_proj(slots)  # 线性层, (1, 40000, 256)

        return self.dropout(slots) + inp_residual  # (1, 40000, 256)


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)  # 8*4*8*2
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)  # 8*4*8
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)  # 线性层 (6, 19560, 256)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)  # (6, 19560, 8, 32)
        # 可变形注意力中的 采样偏移量 + 获得注意力权重 两个步骤（通过线性层）
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)  # (6, 9669, 8, 4, 8, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)  # (6, 9669, 8, 32)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)  # (6, 9669, 8, 4, 8)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]  # (6, 9669, 1, 1, 1, 4, 2)
            sampling_offsets = sampling_offsets / \
                               offset_normalizer[None, None, None, :, None, :]  # (6, 9669, 8, 4, 8, 2)
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape  # 6, 9669, 8, 4, 8, 2
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors,
                xy)  # (6, 9669, 8, 4, 2, 4, 2)
            sampling_locations = reference_points + sampling_offsets  # (6, 9669, 8, 4, 2, 4, 2)
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)  # (6, 9669, 8, 4, 8, 2)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
