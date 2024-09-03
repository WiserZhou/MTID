import torch
import torch.nn.functional as F

# # 设置随机种子以保证结果可复现
# torch.manual_seed(42)

# # 定义样本维度
# batch_size = 256
# horizon = 3
# action_dim = 105

# # 模型输出 (logits)
# model_outputs = torch.randn(batch_size, horizon, action_dim)

# # 真实标签 (one-hot 编码)
# true_actions = torch.randint(0, action_dim, (batch_size, horizon))
# targets = F.one_hot(true_actions, num_classes=action_dim).float()

def compute_losses(model_outputs, targets, delta=1, lambda_oc=1.0, lambda_fp=1.0, lambda_r=1.0):
    """
    计算总损失，包括:
    - 序列交叉熵损失 (L_SCE)
    - 顺序一致性损失 (L_OC)
    - 柔性位置损失 (L_FP)
    - 排序损失 (L_R)
    
    参数:
    - model_outputs: 模型的原始输出 (logits), 形状为 (batch_size, horizon, action_dim)
    - targets: 真实标签的 one-hot 编码, 形状为 (batch_size, horizon, action_dim)
    - delta: 柔性位置损失中允许的位置偏移范围 (默认=1)
    - lambda_oc: 顺序一致性损失的权重
    - lambda_fp: 柔性位置损失的权重
    - lambda_r: 排序损失的权重
    
    返回:
    - total_loss: 总损失
    - sce_loss: 序列交叉熵损失
    - oc_loss: 顺序一致性损失
    - fp_loss: 柔性位置损失
    - r_loss: 排序损失
    """
    batch_size, horizon, action_dim = model_outputs.shape
    
    # 将 logits 转换为概率分布
    probs = F.softmax(model_outputs, dim=-1)
    
    # ========== 1. 序列交叉熵损失 ==========
    # 将所有时间步展平，计算标准交叉熵损失
    sce_loss = F.cross_entropy(
        model_outputs,
        targets
    ).sum() * 100
    
    # ========== 2. 顺序一致性损失 ==========
    # 获取每个时间步的真实动作索引和对应的预测概率
    true_indices = targets.argmax(dim=-1)  # (batch_size, horizon)
    pred_probs = probs.gather(dim=2, index=true_indices.unsqueeze(-1)).squeeze(-1)  # (batch_size, horizon)
    
    # 计算相邻时间步之间的概率差异
    if horizon > 1:
        prob_diffs = pred_probs[:, :-1] - pred_probs[:, 1:]  # (batch_size, horizon - 1)
        # 仅当真实顺序要求前一个动作的索引小于后一个动作时，计算损失
        order_mask = (true_indices[:, :-1] > true_indices[:, 1:]).float()
        oc_loss = F.relu(prob_diffs * order_mask).mean()
    else:
        oc_loss = torch.tensor(0.0)
    
    # ========== 3. 柔性位置损失 ==========
    # 获取预测的动作索引
    pred_indices = probs.argmax(dim=-1)  # (batch_size, horizon)
    
    # 计算预测索引与真实索引之间的偏差
    pos_diffs = torch.abs(pred_indices - true_indices).float()
    
    # 仅当偏差超过 delta 时才计算损失
    fp_loss = F.relu(pos_diffs - delta).mean()
    
    # ========== 4. 排序损失 ==========
    # 计算每个样本的预测概率排序
    pred_ranks = torch.argsort(probs, dim=2, descending=True)  # (batch_size, horizon, action_dim)
    
    # 获取真实动作在排序中的位置
    true_ranks = pred_ranks.eq(true_indices.unsqueeze(-1)).float().argmax(dim=-1)  # (batch_size, horizon)
    
    # 计算相邻时间步之间的排名差异
    if horizon > 1:
        rank_diffs = true_ranks[:, :-1] - true_ranks[:, 1:]
        # 当真实顺序要求前一个动作的索引小于后一个动作时，计算损失
        rank_mask = (true_indices[:, :-1] > true_indices[:, 1:]).float()
        r_loss = F.relu(rank_diffs * rank_mask).mean()
    else:
        r_loss = torch.tensor(0.0)
    
    # ========== 总损失 ==========
    total_loss = sce_loss + lambda_oc * oc_loss + lambda_fp * fp_loss + lambda_r * r_loss
    
    print(total_loss, sce_loss, oc_loss, fp_loss, r_loss)
    
    return total_loss, sce_loss, oc_loss, fp_loss, r_loss

# # 调用损失函数
# total_loss, sce_loss, oc_loss, fp_loss, r_loss = compute_losses(
#     model_outputs,
#     targets,
#     delta=1,
#     lambda_oc=0.5,
#     lambda_fp=0.5,
#     lambda_r=0.5
# )

# # 打印结果
# print(f"Total Loss: {total_loss.item():.4f}")
# print(f"Sequence Cross-Entropy Loss: {sce_loss.item():.4f}")
# print(f"Order Consistency Loss: {oc_loss.item():.4f}")
# print(f"Flexible Position Loss: {fp_loss.item():.4f}")
# print(f"Rank Loss: {r_loss.item():.4f}")
