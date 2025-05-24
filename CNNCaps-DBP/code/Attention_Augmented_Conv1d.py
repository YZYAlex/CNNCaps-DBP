import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class AugmentedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, stride):
        super(AugmentedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"

        self.conv_out = nn.Conv1d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride,
                                  padding=self.padding)

        self.qkv_conv = nn.Conv1d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride,
                                  padding=self.padding)

        self.attn_out = nn.Conv1d(self.dv, self.dv, kernel_size=1, stride=1)

        self.attention_weights = None  # 新增属性存储注意力权重

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, embedding, length]
        conv_out = self.conv_out(x)  # [batch, embedding, length]
        batch, _, length = conv_out.size()

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        weights = F.softmax(logits, dim=-1)
        self.attention_weights = weights.detach().cpu().numpy()  # 存储注意力权重

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, length))

        attn_out = self.combine_heads_1d(attn_out)
        attn_out = self.attn_out(attn_out)
        out = torch.cat((conv_out, attn_out), dim=1)
        return out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)  # [batch, embedding, length]
        N, _, L = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_1d(q, Nh)  # [batch, Nh, embedding // Nh, length]
        k = self.split_heads_1d(k, Nh)  # [batch, Nh, embedding // Nh, length]
        v = self.split_heads_1d(v, Nh)  # [batch, Nh, embedding // Nh, length]

        dkh = dk // Nh
        q = q * dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, L))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, L))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, L))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_1d(self, x, Nh):
        batch, channels, length = x.size()
        ret_shape = (batch, Nh, channels // Nh, length)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_1d(self, x):
        batch, Nh, dv, L = x.size()
        ret_shape = (batch, Nh * dv, L)
        return torch.reshape(x, ret_shape)

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(device)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

