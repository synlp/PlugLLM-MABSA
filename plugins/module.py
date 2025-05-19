import torch
import torch.nn as nn


default_para = {
    'use_weight': False,
    'gcn_layer_number': 3,
}


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class GCNModule(nn.Module):
    def __init__(self, layer_number, hidden_size, use_weight=False, output_all_layers=False):
        super(GCNModule, self).__init__()
        if layer_number < 1:
            raise ValueError()
        self.layer_number = layer_number
        self.output_all_layers = output_all_layers
        self.GCNLayers = nn.ModuleList(([GCNLayer(hidden_size, use_weight)
                                         for _ in range(self.layer_number)]))

    def forward(self, hidden_state, adjacency_matrix):
        # hidden_state = self.first_GCNLayer(hidden_state, adjacency_matrix, type_seq, type_matrix)
        # all_output_layers.append(hidden_state)

        all_output_layers = []

        for gcn in self.GCNLayers:
            hidden_state = gcn(hidden_state, adjacency_matrix)
            all_output_layers.append(hidden_state)

        if self.output_all_layers:
            return all_output_layers
        else:
            return all_output_layers[-1]


class GCNLayer(nn.Module):
    def __init__(self, hidden_size, use_weight=False):
        super(GCNLayer, self).__init__()
        self.temper = hidden_size ** 0.5
        self.use_weight = use_weight
        self.relu = nn.ReLU()

        self.linear = nn.Linear(hidden_size, hidden_size)

        if self.use_weight:
            self.left_linear = nn.Linear(hidden_size, hidden_size, bias=False)
            self.right_linear = nn.Linear(hidden_size, hidden_size, bias=False)
            self.self_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.left_linear = None
            self.right_linear = None
            self.self_linear = None

        self.output_layer_norm = LayerNormalization(hidden_size)

    def get_att(self, matrix_1, matrix_2, adjacency_matrix):

        if self.use_weight:
            m_left = self.left_linear(matrix_2)
            m_self = self.left_linear(matrix_2)
            m_right = self.right_linear(matrix_2)

            m_left = m_left.permute(0, 2, 1)
            m_self = m_self.permute(0, 2, 1)
            m_right = m_right.permute(0, 2, 1)

            u_left = torch.matmul(matrix_1, m_left) / self.temper
            u_self = torch.matmul(matrix_1, m_self) / self.temper
            u_right = torch.matmul(matrix_1, m_right) / self.temper

            adj_tri = torch.triu(adjacency_matrix, diagonal=1)
            u_left = torch.mul(u_left, adj_tri)
            u_self = torch.mul(u_self, torch.triu(adjacency_matrix, diagonal=0) - adj_tri)
            u_right = torch.mul(u_right, adj_tri.permute(0, 2, 1))

            u = u_left + u_right + u_self
            exp_u = torch.exp(u)
            delta_exp_u = torch.mul(exp_u, adjacency_matrix)
        else:
            delta_exp_u = adjacency_matrix

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-4)
        return attention

    def forward(self, hidden_state, adjacency_matrix):
        # hidden_state: (batch_size, character_seq_len, hidden_size)
        # adjacency_matrix: (batch_size, character_seq_len_1, character_seq_len_2)
        # type_seq: (batch_size, type_seq_len)
        # type_matrix: (batch_size, character_seq_len_1, type_seq_len)

        # tmp_hidden = hidden_state.permute(0, 2, 1)

        context_attention = self.get_att(hidden_state, hidden_state, adjacency_matrix)

        hidden_state = self.linear(hidden_state)
        context_attention = context_attention.to(hidden_state.dtype)
        context_attention = torch.bmm(context_attention, hidden_state)

        o = self.output_layer_norm(context_attention)

        # o = context_attention
        output = self.relu(o)

        return output


