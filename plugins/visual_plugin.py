import torch.nn as nn
import torch
from plugins.module import GCNModule

class VisualPlugin(nn.Module):
    def __init__(self, visual_encoder, gcn_layer_number=0):
        super(VisualPlugin, self).__init__()
        self.visual_encoder = visual_encoder
        self.gcn_layer_number = gcn_layer_number
        self.use_gcn = True if gcn_layer_number > 0 else False
        self.hidden_size = self.visual_encoder.config.hidden_size
        self.dropout = nn.Dropout(p=0.33)

        if self.use_gcn:
            self.gcn = GCNModule(self.gcn_layer_number, self.hidden_size,
                                 use_weight=True,
                                 output_all_layers=False)
        else:
            self.gcn = None

    def forward(self, inputs, valid_groups, adj_matrix):
        batch_size, group_number, channels, height, width = inputs.shape

        # 将输入 reshape 成 (batch_size * group_number, 3, 224, 224)
        inputs = inputs.view(batch_size * group_number, channels, height, width)

        # 使用 ViT 模型提取特征
        outputs = self.visual_encoder(pixel_values=inputs).last_hidden_state

        # 取出 [CLS] token 的输出
        cls_output = outputs[:, 0, :]  # (batch_size * group_number, hidden_dim)

        # 将输出 reshape 回 (batch_size, group_number, hidden_dim)
        cls_output = cls_output.view(batch_size, group_number, -1)

        cls_output = self.dropout(cls_output)

        if self.gcn is not None:
            cls_output = self.gcn(cls_output, adj_matrix)

        # 根据 valid_groups 对每个 batch 取平均
        batch_avg_outputs = []
        for i in range(batch_size):
            valid_group_count = valid_groups[i]  # 当前 batch 的有效 group 数
            valid_cls_output = cls_output[i, :valid_group_count, :]  # 只取有效的 group
            avg_output = valid_cls_output.mean(dim=0)  # 对有效 group 取平均
            batch_avg_outputs.append(avg_output)

        # 将所有 batch 的平均输出拼接在一起
        batch_avg_outputs = torch.stack(batch_avg_outputs).to(cls_output.device)

        return batch_avg_outputs
