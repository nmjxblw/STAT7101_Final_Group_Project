import pandas as pd
import numpy as np
from ast import literal_eval
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment

df = pd.read_csv("freq_features_data.csv")

df["inv_key"] = df["inv_key"].apply(lambda x: literal_eval(x))

def convert_to_int_list(key):
    return [ord(c) - 97 for c in key]

df["y"] = df["inv_key"].apply(convert_to_int_list)

X = df[[f"freq_{i}" for i in range(702)]].values.astype("float32")

Y = np.vstack(df["y"].values).astype("int64")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=2025)

class Mydataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

train_dataset = Mydataset(X_train, Y_train)
test_dataset = Mydataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
#简单神经网络，做26个分类
class PermNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(702, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 26*26)
        )
    def forward(self, x):
        logits = self.fc(x)
        logits = logits.view(-1, 26, 26)
        probs = torch.softmax(logits, dim=2)
        return logits, probs
#CNN混合结构（该设置收敛到66%）对单一频率和条件频率分开处理
class CNNPermNet(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN 处理 26×26 的 bigram 结构
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.AdaptiveAvgPool2d((6, 6)),
        )

        # unigram 26 维 → dense
        self.unigram_mlp = nn.Sequential(
            nn.Linear(26, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
        )

        # 融合 CNN + MLP 的特征
        self.fc = nn.Sequential(
            nn.Linear(32 * 6 * 6 + 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 26 * 26),
        )

    def forward(self, x):
        """
        x: (B, 702)
        前26维是 unigram
        后676维 reshape 为 26×26 bigram
        """

        unigram = x[:, :26]              # (B, 26)
        bigram  = x[:, 26:].view(-1, 1, 26, 26)  # (B,1,26,26)

        uni_feat = self.unigram_mlp(unigram)     # (B, 64)
        cnn_feat = self.cnn(bigram).view(x.size(0), -1)  # (B, 32*6*6)

        feat = torch.cat([uni_feat, cnn_feat], dim=1)

        logits = self.fc(feat)
        logits = logits.view(-1, 26, 26)

        probs = torch.softmax(logits, dim=2)
        return logits, probs

# 更改为transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.optimize import linear_sum_assignment  # 你的Hungarian


class TransformerPermNet(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 输入嵌入：unigram (B,26) + bigram (B,26,26) → 每个cipher pos嵌入 (26-dim unigram_slice + 26-dim bigram_row)
        self.input_proj = nn.Linear(52, d_model)  # 26 uni + 26 bi per cipher → d_model

        # 位置编码（a-z顺序）
        self.pos_encoder = nn.Parameter(torch.zeros(1, 27, d_model))  # 26 cipher + 1 global uni

        # Transformer Encoder：处理cipher序列 (27 tokens)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出头：encoder out (B,27,d_model) → 取前26 cipher tokens → (B,26,d_model) → 线性到26 plain
        self.output_head = nn.Linear(d_model, 26)

        # LayerNorm & Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, 702) = (B,26 uni + 676 bi)
        """
        B = x.size(0)
        uni = x[:, :26]  # (B,26)
        bi = x[:, 26:].view(B, 26, 26)  # (B,26_cipher,26_plain_cond)

        # 为每个cipher pos构建嵌入：uni[pos] + bi[pos, :] (cond dist)
        embeds = torch.cat([uni.unsqueeze(1).expand(-1, 26, -1), bi], dim=-1)  # (B,26,52)
        embeds = self.input_proj(embeds)  # (B,26,d_model)

        # 加位置编码（cipher 0-25）
        seq_len = 26
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)
        embeds = embeds + self.pos_encoder[:, :seq_len, :]  # 简化，无global token

        embeds = self.dropout(self.norm(embeds))

        # Encoder
        enc_out = self.transformer_encoder(embeds)  # (B,26,d_model)

        # 输出：每个cipher的注意力后，线性到plain logits
        logits = self.output_head(enc_out)  # (B,26,26)
        probs = F.softmax(logits, dim=-1)

        return logits, probs


device = "cuda" if torch.cuda.is_available() else "cpu"
#model = TransformerPermNet(d_model=128, nhead=4, num_layers=2).to(device)
model = CNNPermNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
)
def hungarian_decode(probs):
    """
    probs: (B, 26, 26)
    返回 shape=(B,26) 的置换整数列表，每行均为合法置换（通过匈牙利）
    """
    probs_np = probs.detach().cpu().numpy()
    batch_size = probs_np.shape[0]
    perms = []

    for i in range(batch_size):
        cost = -np.log(probs_np[i] + 1e-12)
        row_ind, col_ind = linear_sum_assignment(cost)
        perms.append(col_ind)

    return np.array(perms)  # (B, 26)


def evaluate_key_accuracy(true_keys, pred_keys):
    """
    true_keys, pred_keys: (N, 26)
    """
    correct = (true_keys == pred_keys)
    acc_per_sample = np.mean(correct, axis=1)
    return np.mean(acc_per_sample)

EPOCHS = 100

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0

    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)

        logits, _ = model(Xb)

        # 26个交叉熵
        loss = sum(criterion(logits[:, i, :], Yb[:, i]) for i in range(26)) / 26

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ---- Validation ----
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb = Xb.to(device)
            logits, probs = model(Xb)
            #防止非法数据
            #probs = torch.clamp(probs, min=1e-8)
            preds = hungarian_decode(probs)
            all_pred.append(preds)
            all_true.append(Yb.numpy())

    all_pred = np.vstack(all_pred)
    all_true = np.vstack(all_true)

    key_acc = evaluate_key_accuracy(all_true, all_pred)
    scheduler.step(key_acc)
    print(f"[Epoch {epoch}/{EPOCHS}] loss={total_loss:.4f} | key-accuracy={key_acc:.4f}")
full_acc = np.mean(np.all(all_true == all_pred, axis=1))
print(f"Full key match rate: {full_acc:.4f}")
print("训练完成！")