import os
import random
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns
import torch.nn.functional as F
from Caps import *
from Attention_Augmented_Conv1d import AugmentedConv1d


class Decoder(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)
        self.fc = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [B, L, 16]
        attn_weights = F.softmax(self.attention(x), dim=1)  # [B, L, 1]
        context = torch.sum(x * attn_weights, dim=1)  # [B, 16]
        return self.sigmoid(self.fc(context))  # [B, 1]


class ESMc_Caps(nn.Module):
    def __init__(self, input_size=1152, output_size1=512, output_size2=200, out_put_dim=16, dropout=0.4):
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


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

# 数据路径（根据实际情况修改）
test_data_path = pd.read_csv("tst.csv")


# 设置随机种子
def random_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


random_seed()


def tokenize(path):
    df = path
    df.columns = [int(col) for col in df.columns]
    return np.array(df[df.columns[1]])


class DBPDataset(Dataset):
    def __init__(self, esmc_feats, labels):
        self.esmc = esmc_feats.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __getitem__(self, idx):
        return self.esmc[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def extract_features(model, loader, layers=None):
    if layers is None:
        layers = ["conv1", "conv2", "caps", "decoder"]
    features = {layer: [] for layer in layers}
    labels = []

    with torch.no_grad():
        for esm, label in loader:
            x = esm.to(device)  # [B, L, 1152]

            # Conv1 [B, 512, L1] -> [B, 512*L1]
            conv1_out = model.conv1(x)
            features["conv1"].append(conv1_out.permute(0, 2, 1).reshape(esm.size(0), -1).cpu().numpy())

            conv1_out = conv1_out.permute(0, 2, 1)
            # Conv2 [B, 200, L2] -> [B, 200*L2]
            conv2_out = model.conv2(conv1_out)
            features["conv2"].append(conv2_out.permute(0, 2, 1).reshape(esm.size(0), -1).cpu().numpy())

            # Caps [B, 16, 16] -> [B, 256]
            caps_out = model.caps(conv2_out)  # [B, 16, 16]
            features["caps"].append(caps_out.reshape(esm.size(0), -1).cpu().numpy())

            # Decoder [B, 16, 16] -> caps_out
            decoder = model.decoder(caps_out)
            features["decoder"].append(decoder.reshape(esm.size(0), -1).cpu().numpy())

            labels.extend(label.numpy())

    for k in features:
        features[k] = np.vstack(features[k])
    return features, np.array(labels)


# UMAP可视化
def visualize_umap(features, labels, layer_name, save_path, point_size=80):
    reducer = UMAP(n_neighbors=15, min_dist=0.5, random_state=42)
    umap_emb = reducer.fit_transform(features)

    plt.figure(figsize=(10, 6))
    # 设置点的大小为 point_size
    sns.scatterplot(
        x=umap_emb[:, 0], y=umap_emb[:, 1],
        hue=labels, palette=["#4CAF50", "#FF5722"],
        alpha=0.7, edgecolor="w",
        size=point_size
    )
    plt.title(f"UMAP: {layer_name} Features")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Label", labels=["Negative", "Positive"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    model_path = "checkpoints/fold2_gpu.pth"
    model = torch.load(model_path, map_location=device).to(device)

    test_esmc = np.load("test_ESMc.npy")
    test_labels = tokenize(test_data_path)

    test_dataset = DBPDataset(test_esmc, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    layers = ["conv1", "conv2", "caps", "decoder"]
    all_feats, all_labels = extract_features(model, test_loader, layers)

    # 5. 可视化
    os.makedirs("umap_visualization", exist_ok=True)
    for layer in layers:
        visualize_umap(
            all_feats[layer], all_labels,
            layer_name=layer.upper(),
            save_path=f"umap_visualization/{layer}_umap.png"
        )
        print(f"保存 {layer} 可视化结果")
