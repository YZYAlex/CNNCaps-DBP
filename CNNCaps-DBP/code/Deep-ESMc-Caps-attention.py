import os
import random
import warnings

import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader, Dataset

from Attention_Augmented_Conv1d import AugmentedConv1d
from Caps import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

test_data_path = pd.read_csv("tst.csv")


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed(777)


def tokenize(path):
    data_frame = path
    data_columns = data_frame.columns.tolist()
    data_columns = [int(i) for i in data_columns]
    data_frame.columns = data_columns
    label = data_frame[data_frame.columns[1]]

    return np.array(label)


class DBPDataset(Dataset):

    def __init__(self, ESMc, label):
        self.ESMc = ESMc
        self.label = label

    def __getitem__(self, index):
        ESMc = self.ESMc[index]
        label = self.label[index]

        return ESMc, label

    def __len__(self):
        return len(self.ESMc)


# 读取np文件
def load_data(file_path):
    return np.load(file_path)


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionDecoder, self).__init__()
        self.attention = nn.Linear(embedding_dim, 1)
        self.fc = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_weights = self.attention(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        x = torch.sum(x * attention_weights, dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# Model
class ESMc_Caps(nn.Module):
    def __init__(self, input_size=1152, output_size1=502, output_size2=200, out_put_dim=16, dropout=0.4):
        super(ESMc_Caps, self).__init__()

        self.input_size = input_size
        self.output_size1 = output_size1
        self.output_size2 = output_size2
        self.out_put_dim = out_put_dim
        self.dropout = dropout

        # Conv1
        self.conv1 = nn.Sequential(
            AugmentedConv1d(self.input_size, self.output_size1, 10, dk=32, dv=4, Nh=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(self.output_size1),
            nn.Dropout(self.dropout),
        )

        # Conv2
        self.conv2 = nn.Sequential(
            AugmentedConv1d(self.output_size1, self.output_size2, 5, dk=32, dv=4, Nh=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(self.output_size2),
            nn.Dropout(self.dropout),
        )

        # Caps
        self.caps = CapsLayer(input_caps=self.output_size2, input_dim=126, output_caps=16, output_dim=self.out_put_dim)

        # Attention
        self.w_omega = nn.Parameter(torch.Tensor(16, 32))
        self.u_omega = nn.Parameter(torch.Tensor(32, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.decoder = AttentionDecoder(16)

    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))

        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score

        context = torch.sum(scored_x, dim=1)
        return context

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = x.permute(0, 2, 1)

        # Conv2
        x = self.conv2(x)

        # Caps
        x = self.caps(x)

        x = self.decoder(x)

        return x


model_path = 'checkpoints/fold5_gpu.pth'
protein = 16
start, end = 150, 180
maxtop = 1
protein_name = "5DS9"
model = torch.load(model_path, map_location=device)
model = model.to(device)


test_ESMc = load_data('test_ESMc.npy')
test_label = tokenize(test_data_path)


test_dataset = DBPDataset(test_ESMc, test_label)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1)


attention_weights_list = []

with torch.no_grad():
    for esm, label in test_loader:
        esm = esm.to(device)
        with torch.no_grad():
            _ = model(esm)
        aw = model.conv2[0].attention_weights  # [batch, heads, L]
        for i, lbl in enumerate(label):
            if lbl == 1:
                attention_weights_list.append(aw[i])


    attn = np.array(attention_weights_list)  # [N, heads, L]
    # [N, L]
    attn_sum = np.sum(attn, axis=1)

    scaler = MinMaxScaler((0, 1))
    attn_norm = np.vstack([scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in attn_sum])  # [N, L]

    for idx, row in enumerate(attn_norm):
        if idx + 1 == protein:
            seg = row[start:end]
            fig, ax = plt.subplots(figsize=(12, 2))

            im = ax.imshow(seg[np.newaxis, :],
                           aspect='auto',
                           cmap='viridis',
                           norm=Normalize(vmin=0, vmax=maxtop),
                           extent=(start - 0.5, end - 0.5, 0, 1))

            ax.yaxis.set_visible(False)
            ax.set_yticks([])

            for spine in ['left', 'right', 'top', 'bottom']:
                ax.spines[spine].set_visible(False)

            xt = np.arange(start, end)
            ax.set_xticks(xt)
            ax.set_xticklabels(xt, rotation=0, fontsize=8)

            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label("Normalized Attention Weights")

            ax.set_title(f"Attention Weights for Sequence {protein_name} (Normalized)", pad=8)
            plt.tight_layout()

            save_dir = 'visualization_images'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{protein_name}_attention.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存注意力权重热图至: {save_path}")
            plt.show()
        else:
            continue
