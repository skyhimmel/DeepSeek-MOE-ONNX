import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Expert(nn.Module):
    """单个专家网络"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class TopKGate(nn.Module):
    """Top-K门控机制，参考DeepSeek实现"""
    def __init__(self, input_dim, num_experts, top_k=2, noise_std=1.0):
        super(TopKGate, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        
    def _add_noise(self, logits):
        """添加噪声用于负载均衡"""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            return logits + noise
        return logits
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算门控分数
        gate_logits = self.gate(x)
        gate_logits = self._add_noise(gate_logits)
        
        # Top-K选择
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Softmax归一化
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # 创建稀疏门控矩阵
        gates = torch.zeros_like(gate_logits)
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        # 计算负载均衡损失
        load_balancing_loss = self._compute_load_balancing_loss(gate_logits, gates)
        
        return gates, top_k_indices, load_balancing_loss
    
    def _compute_load_balancing_loss(self, gate_logits, gates):
        """计算负载均衡损失"""
        # 路由概率的均值
        router_prob_per_expert = torch.mean(gates, dim=0)
        
        # 专家使用频率的均值
        expert_usage = (gates > 0).float()
        expert_usage_per_expert = torch.mean(expert_usage, dim=0)
        
        # 负载均衡损失
        load_balancing_loss = torch.sum(router_prob_per_expert * expert_usage_per_expert) * self.num_experts
        
        return load_balancing_loss

class MoELayer(nn.Module):
    """MoE层实现"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=8, top_k=2, dropout=0.1):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络
        self.gate = TopKGate(input_dim, num_experts, top_k)
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # 展平输入
        x_flat = x.view(-1, input_dim)
        
        # 获取门控权重和索引
        gates, expert_indices, load_balancing_loss = self.gate(x_flat)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 计算每个专家的输出
        for i in range(self.num_experts):
            # 选择使用当前专家的样本
            expert_mask = (expert_indices == i).any(dim=1)
            if expert_mask.sum() > 0:
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[i](expert_input)
                
                # 获取对应的门控权重
                expert_weights = gates[expert_mask, i:i+1]
                output[expert_mask] += expert_weights * expert_output
        
        # 恢复原始形状
        output = output.view(batch_size, seq_len, -1)
        
        return output, gates, load_balancing_loss
    
class DeepSeekMoE(nn.Module):
    """DeepSeek风格的MoE模型用于MNIST"""
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, num_experts=8, top_k=2, dropout=0.1):
        super(DeepSeekMoE, self).__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 输入层
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # MoE层
        self.moe_layer = MoELayer(hidden_size, hidden_size*2, hidden_size, num_experts, top_k, dropout)
        
        # 输出层
        self.output_projection = nn.Linear(hidden_size, num_classes)
        
        # 激活函数
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平输入
        
        # 输入投影
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 添加序列维度用于MoE层
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # MoE层
        x, gates, load_balancing_loss = self.moe_layer(x)
        
        # 移除序列维度
        x = x.squeeze(1)  # [batch_size, hidden_size]
        
        # 输出投影
        x = self.output_projection(x)
        
        return x, gates, load_balancing_loss

if __name__=='__main__':
    moe=DeepSeekMoE()
    x=torch.rand((5,1,28,28))
    y,prob=moe(x)
    print(y.shape,prob.shape)
