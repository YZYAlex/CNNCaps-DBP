import os
import random
import warnings

import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold

from torch.utils.data import DataLoader, Dataset

from Attention_Augmented_Conv1d import AugmentedConv1d

from Caps import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


# random seed
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed(42)


def load_data(file_path):
    return np.load(file_path)


train_data_path = pd.read_csv("train_1075.csv")
test_data_path = pd.read_csv("test_186.csv")


# sequence, label, features
def tokenize(path):
    data_frame = path
    data_columns = data_frame.columns.tolist()
    data_columns = [int(i) for i in data_columns]
    data_frame.columns = data_columns
    label = data_frame[data_frame.columns[1]]  # 读取蛋白质标签

    return np.array(label)


train_label = tokenize(train_data_path)
test_label = tokenize(test_data_path)


def Model_Evaluate(confus_matrix, pred, label):
    TN, FP, FN, TP = confus_matrix.ravel()

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    Pre = TP / (TP + FP)
    F1 = 2 * (Pre * SN) / (Pre + SN) if (Pre + SN) != 0 else 0

    try:
        AUC = roc_auc_score(label, pred)
    except Exception as e:
        print("Error calculating AUC:", e)
        AUC = 0

    return SN, SP, ACC, MCC, Pre, AUC, F1


def cal_score(pred, label):
    label = np.array(label)
    pred_class = np.around(pred)
    confus_matrix = confusion_matrix(label, pred_class, labels=None, sample_weight=None)
    SN, SP, ACC, MCC, Pre, AUC, F1 = Model_Evaluate(confus_matrix, pred, label)
    print(
        "Model score --- ACC:{0:.3f}    AUC:{1:.3f}    Sen:{2:.3f}    Spe:{3:.3f}    MCC:{4:.3f}    Pre:{5:.3f}    "
        "F1:{6:.3f}"
        .format(ACC, AUC, SN, SP, MCC, Pre, F1))

    return ACC, AUC, SN, SP, MCC, Pre, F1


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


def fit(t_model, t_train_loader, t_optimizer, t_criterion):
    t_model.train()

    pred_list = []
    label_list = []
    total_loss = 0

    for protein_esm, label in t_train_loader:
        protein_esm = protein_esm.to(device)
        label = torch.tensor(label).float().to(device)

        pred = model(protein_esm)

        pred = pred.squeeze()

        loss = t_criterion(pred, label)

        model.zero_grad()
        loss.backward()
        t_optimizer.step()

        total_loss += loss.item()
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)
    average_loss = total_loss / len(train_loader)

    return score, average_loss


def validate(v_model, val_loader):
    v_model.eval()

    pred_list = []
    label_list = []
    total_loss = 0

    for protein_esm, label in val_loader:
        protein_esm = protein_esm.to(device)
        label = torch.tensor(label).float().to(device)

        pred = model(protein_esm)

        pred = pred.squeeze()
        loss = criterion(pred, label)

        total_loss += loss.item()
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)
    average_loss = total_loss / len(val_loader)
    return score, average_loss, pred_list, label_list

class Decoder(nn.Module):
    def __init__(self, output_caps):
        super(Decoder, self).__init__()
        self.attention = nn.Linear(output_caps, 1)
        self.fc = nn.Linear(output_caps, 1)
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
    def __init__(self, input_size=1152, output_size1=512, output_size2=200, out_put_dim=24, dropout=0.4):
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

        self.decoder = Decoder(self.out_put_dim)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)

        # Conv2
        x = x.permute(0, 2, 1)
        x = self.conv2(x)

        # Caps
        x = self.caps(x)

        x = self.decoder(x)

        return x


if __name__ == '__main__':

    train_ESMc = load_data('train_1075_ESMc.npy')
    test_ESMc = load_data('test_186_ESMc.npy')

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    val_metrics = []
    test_metrics = []
    train_metrics = []

    last_fold_train_losses = []
    last_fold_val_losses = []
    last_fold_test_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_ESMc)):
        print(f"Fold {fold + 1}")
        ESMc_train, ESMc_val = train_ESMc[train_idx], train_ESMc[val_idx]
        label_train, label_val = train_label[train_idx], train_label[val_idx]

        train_dataset = DBPDataset(ESMc_train, label_train)
        valid_dataset = DBPDataset(ESMc_val, label_val)
        test_dataset = DBPDataset(test_ESMc, test_label)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=1)
        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True, num_workers=1)

        model = ESMc_Caps().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)

        train_losses = []
        val_losses = []
        test_losses = []

        fold_val_metrics = []
        fold_test_metrics = []
        fold_train_metrics = []

        for epoch in range(40):
            print(f"Fold {fold + 1}, Epoch {epoch + 1}")
            train_score, train_loss = fit(model, train_loader, optimizer, criterion)
            val_score, val_loss = validate(model, valid_loader)[:2]
            test_score, test_loss = validate(model, test_loader)[:2]

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)

            fold_val_metrics.append(val_score)
            fold_test_metrics.append(test_score)
            fold_train_metrics.append(train_score)

            scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}:\n'
                  f'Train Loss: {train_loss}\n'
                  f'Valid Loss: {val_loss}, Valid Score: {val_score}\n'
                  f'Test Loss:  {test_loss}, Test Score: {test_score}\n'
                  f'Learning Rate: {optimizer.param_groups[0]["lr"]}')

        val_metrics.append(fold_val_metrics[-1])
        test_metrics.append(fold_test_metrics[-1])
        train_metrics.append(fold_train_metrics[-1])

        if fold == 4:
            last_fold_train_losses = train_losses
            last_fold_val_losses = val_losses
            last_fold_test_losses = test_losses

        torch.cuda.empty_cache()

    avg_val_ACC = np.mean([metrics[0] for metrics in val_metrics])
    avg_val_AUC = np.mean([metrics[1] for metrics in val_metrics])
    avg_val_Sen = np.mean([metrics[2] for metrics in val_metrics])
    avg_val_Spe = np.mean([metrics[3] for metrics in val_metrics])
    avg_val_MCC = np.mean([metrics[4] for metrics in val_metrics])
    avg_val_Pre = np.mean([metrics[5] for metrics in val_metrics])
    avg_val_F1 = np.mean([metrics[6] for metrics in val_metrics])

    avg_test_ACC = np.mean([metrics[0] for metrics in test_metrics])
    avg_test_AUC = np.mean([metrics[1] for metrics in test_metrics])
    avg_test_Sen = np.mean([metrics[2] for metrics in test_metrics])
    avg_test_Spe = np.mean([metrics[3] for metrics in test_metrics])
    avg_test_MCC = np.mean([metrics[4] for metrics in test_metrics])
    avg_test_Pre = np.mean([metrics[5] for metrics in test_metrics])
    avg_test_F1 = np.mean([metrics[6] for metrics in test_metrics])

    avg_train_ACC = np.mean([metrics[0] for metrics in train_metrics])
    avg_train_AUC = np.mean([metrics[1] for metrics in train_metrics])
    avg_train_Sen = np.mean([metrics[2] for metrics in train_metrics])
    avg_train_Spe = np.mean([metrics[3] for metrics in train_metrics])
    avg_train_MCC = np.mean([metrics[4] for metrics in train_metrics])
    avg_train_Pre = np.mean([metrics[5] for metrics in train_metrics])
    avg_train_F1 = np.mean([metrics[6] for metrics in train_metrics])

    print(f"Average Training Metrics:")
    print(
        f"ACC: {avg_train_ACC:.3f}, AUC: {avg_train_AUC:.3f}, Sen: {avg_train_Sen:.3f}, Spe: {avg_train_Spe:.3f}, MCC: {avg_train_MCC:.3f}, Pre: {avg_train_Pre:.3f}, F1: {avg_train_F1:.3f}")
    print(f"Average Validation Metrics:")
    print(
        f"ACC: {avg_val_ACC:.3f}, AUC: {avg_val_AUC:.3f}, Sen: {avg_val_Sen:.3f}, Spe: {avg_val_Spe:.3f}, MCC: {avg_val_MCC:.3f}, Pre: {avg_val_Pre:.3f}, F1: {avg_val_F1:.3f}")
    print(f"Average Test Metrics:")
    print(
        f"ACC: {avg_test_ACC:.3f}, AUC: {avg_test_AUC:.3f}, Sen: {avg_test_Sen:.3f}, Spe: {avg_test_Spe:.3f}, MCC: {avg_test_MCC:.3f}, Pre: {avg_test_Pre:.3f}, F1: {avg_test_F1:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(last_fold_train_losses, label='Train Loss', color='red')
    plt.plot(last_fold_val_losses, label='Validation Loss', color='green')
    plt.plot(last_fold_test_losses, label='Test Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Last Fold Loss Curve')
    plt.legend()

    plt.savefig('last_fold_loss.png')
    plt.show()
