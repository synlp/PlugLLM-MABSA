import torch.nn as nn
import torch
from transformers import BertModel
from plugins.module import GCNModule


class TextPlugin(nn.Module):
    def __init__(self, text_encoder: BertModel, gcn_layer_number=0):
        super(TextPlugin, self).__init__()
        self.bert = text_encoder
        self.gcn_layer_number = gcn_layer_number
        self.use_gcn = True if gcn_layer_number > 0 else False
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(p=0.33)

        if self.use_gcn:
            self.gcn = GCNModule(self.gcn_layer_number, self.hidden_size,
                                 use_weight=True,
                                 output_all_layers=False)
        else:
            self.gcn = None

    def forward(self, input_ids, attention_mask, valid_ids, adj_matrix, text_aspect_index):
        output = self.bert(input_ids, attention_mask)
        sequence_output = output.last_hidden_state
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim,
                                   dtype=sequence_output.dtype, device=sequence_output.device)
        # print(valid_ids)
        # exit(0)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.shape[0]] = temp

        sequence_output = self.dropout(valid_output)
        # import pdb;
        # pdb.set_trace()
        if self.gcn is not None:
            sequence_output = self.gcn(sequence_output, adj_matrix)

        # 根据 valid_groups 对每个 batch 取平均
        batch_avg_outputs = []
        for i in range(batch_size):
            # valid_group_count = sum(valid_ids[i])  # 当前 batch 的有效 group 数
            # valid_cls_output = sequence_output[i, :valid_group_count, :]  # 只取有效的 group
            # avg_output = valid_cls_output.mean(dim=0)  # 对有效 group 取平均
            batch_avg_outputs.append(sequence_output[i, text_aspect_index[i], :])

        # 将所有 batch 的平均输出拼接在一起
        batch_avg_outputs = torch.stack(batch_avg_outputs).to(sequence_output.device)

        return batch_avg_outputs
