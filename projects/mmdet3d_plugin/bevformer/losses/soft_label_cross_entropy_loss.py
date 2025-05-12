import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES

@LOSSES.register_module()
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, lambda_pseudo=1e-3):
        super(SoftCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lambda_pseudo = lambda_pseudo  # 伪监督的权重

    def forward(self, height_distribution, z_prob, bev_ind):
        # height_distribution: (150, 150, 6)
        # z_prob: (31, 6) -> 真值
        # bev_ind: (31, 2) -> 真值的BEV平面索引

        height, width, num_classes = height_distribution.shape

        # 初始化伪标签为均匀分布
        pseudo_labels = torch.full_like(height_distribution, 1.0 / num_classes)

        # 构建目标分布，结合真值和伪标签
        target_distribution = pseudo_labels.clone()
        target_distribution[bev_ind[:, 0], bev_ind[:, 1], :] = z_prob  # 使用真值更新目标分布

        # 计算真值点和伪监督点的掩码
        mask = torch.zeros(height_distribution.shape[:-1], dtype=torch.bool, device=height_distribution.device)
        mask[bev_ind[:, 0], bev_ind[:, 1]] = True  # 有真值的点的掩码

        # 计算所有损失
        log_pred = torch.log(height_distribution + 1e-9)  # 避免 log(0)
        loss = -torch.sum(target_distribution * log_pred, dim=-1)  # 有真值点的交叉熵损失
        loss = loss.unsqueeze(0) #坑！！该操作是为了使后续的sum(-1)以及求平均操作结束tensor不会变成0维(标量)，以及下面求均值不直接用mean()也是这个原因

        # 分别计算真、伪监督的损失
        loss_true = loss[:, bev_ind[:, 0], bev_ind[:, 1]]
        bev_ind_not = torch.nonzero(~mask)
        loss_pseudo = loss[:, bev_ind_not[:, 0], bev_ind_not[:, 1]]

        # 结合真值损失和伪监督损失
        total_loss = (loss_true.sum(-1) / loss_true.size(1)) + self.lambda_pseudo * (loss_pseudo.sum(-1) / loss_pseudo.size(1))

        # Reduction操作
        if self.reduction == 'mean':
            return self.loss_weight * total_loss  # 平均损失并乘以权重
        else:
            return self.loss_weight * total_loss  # 不做 reduction