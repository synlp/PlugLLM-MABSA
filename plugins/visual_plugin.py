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

        # Reshape inputs to (batch_size * group_number, channels, height, width)
        inputs = inputs.view(batch_size * group_number, channels, height, width)

        # Extract features using the visual encoder (ViT)
        outputs = self.visual_encoder(pixel_values=inputs).last_hidden_state

        # Extract the [CLS] token output
        cls_output = outputs[:, 0, :]  # (batch_size * group_number, hidden_dim)

        # Reshape output back to (batch_size, group_number, hidden_dim)
        cls_output = cls_output.view(batch_size, group_number, -1)

        cls_output = self.dropout(cls_output)

        if self.gcn is not None:
            cls_output = self.gcn(cls_output, adj_matrix)

        # Compute average per batch based on valid_groups
        batch_avg_outputs = []
        for i in range(batch_size):
            # Number of valid groups in the current batch
            valid_group_count = valid_groups[i]
            # Select only the valid groups
            valid_cls_output = cls_output[i, :valid_group_count, :]
            # Compute mean over valid groups
            avg_output = valid_cls_output.mean(dim=0)
            batch_avg_outputs.append(avg_output)

        # Stack average outputs of all batches together
        batch_avg_outputs = torch.stack(batch_avg_outputs).to(cls_output.device)

        return batch_avg_outputs
