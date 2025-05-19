import torch
import torch.nn as nn
import torch.nn.functional as F


class Hub(nn.Module):
    def __init__(self, m, hidden_size, k):
        super(Hub, self).__init__()

        # 初始化 m 个 hidden_size 大小的可训练向量
        self.memory_vectors = nn.Parameter(torch.randn(m, hidden_size), requires_grad=True)

        # 为 a 和 b 分别引入两个线性变换矩阵
        self.transform_a = nn.Linear(hidden_size, hidden_size)
        self.transform_b = nn.Linear(hidden_size, hidden_size)
        self.transform_c = nn.Linear(1, k)
        self.hidden_size = hidden_size

        # Additional layers for final processing
        self.fc1 = nn.Linear(3 * hidden_size, 1 * hidden_size)  # First fully connected layer
        # self.fc2 = nn.Linear(2 * hidden_size, hidden_size)  # Second fully connected layer

        self.activation = nn.ReLU()  # Activation function (you can choose others like GELU)

    def forward(self, a, b):
        batch_size, hidden_size = a.shape

        # 对 a 和 b 进行线性变换
        transformed_a = self.transform_a(a)  # (batch_size, hidden_size)
        transformed_b = self.transform_b(b)  # (batch_size, hidden_size)

        # 归一化 memory_vectors
        memory_vectors_norm = self.memory_vectors  # (m, hidden_size)

        # 计算 transformed_a 与 memory_vectors 的余弦相似度
        transformed_a_norm = F.normalize(transformed_a, dim=1)  # (batch_size, hidden_size)
        cos_sim_a = torch.matmul(transformed_a_norm, memory_vectors_norm.T)  # (batch_size, m)

        # 计算 transformed_b 与 memory_vectors 的余弦相似度
        transformed_b_norm = F.normalize(transformed_b, dim=1)  # (batch_size, hidden_size)
        cos_sim_b = torch.matmul(transformed_b_norm, memory_vectors_norm.T)  # (batch_size, m)

        # 余弦相似度相加
        total_similarity = cos_sim_a + cos_sim_b  # (batch_size, m)

        # 使用余弦相似度作为 softmax 权重
        weights = F.softmax(total_similarity, dim=1)  # (batch_size, m)

        # 根据权重加权 memory_vectors
        weighted_memory_vectors = torch.matmul(weights, self.memory_vectors)  # (batch_size, hidden_size)

        # Concatenate weighted_memory_vectors, a, and b along the last dimension
        # concatenated = torch.cat([weighted_memory_vectors, transformed_a, transformed_b], dim=1)  # (batch_size, 3 * hidden_size)

        # Pass through the first fully connected layer
        # fc1_out = self.activation(self.fc1(concatenated))  # (batch_size, 2 * hidden_size)

        # Pass through the second fully connected layer
        # fc2_out = self.activation(self.fc2(fc1_out))  # (batch_size, hidden_size)

        # 输出 shape 改为 (batch_size, 1, hidden_size)
        weighted_memory_vectors = weighted_memory_vectors.unsqueeze(1)  # (batch_size, 1, hidden_size)

        weighted_memory_vectors = weighted_memory_vectors.permute(0, 2, 1)
        weighted_memory_vectors = self.transform_c(weighted_memory_vectors)
        weighted_memory_vectors = weighted_memory_vectors.permute(0, 2, 1)

        return weighted_memory_vectors

