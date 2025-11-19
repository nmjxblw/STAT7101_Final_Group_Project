import pandas as pd
import numpy as np
from ast import literal_eval
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment


df_train,df_val=pd.read_csv("./new_data/new_dataset_train.csv"),pd.read_csv("./new_data/new_dataset_valid.csv")
#df_train=df_train[:7000]
df_train["inv_key"]=df_train["inv_key"].apply(lambda x: literal_eval(x))
df_val["inv_key"] = df_val["inv_key"].apply(lambda x: literal_eval(x))
def convert_to_int_list(key):
    return [ord(c) - 97 for c in key]

df_train["y"] = df_train["inv_key"].apply(convert_to_int_list)
df_val["y"] = df_val["inv_key"].apply(convert_to_int_list)
X_train = df_train[[f"freq_{i}" for i in range(702)]].values.astype("float32")
X_val = df_val[[f"freq_{i}" for i in range(702)]].values.astype("float32")

Y_train = np.vstack(df_train["y"].values).astype("int64")
Y_val = np.vstack(df_val["y"].values).astype("int64")
#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=2025)

class Mydataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

train_dataset = Mydataset(X_train, Y_train)
val_dataset = Mydataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
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



import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedCNNPermNet(nn.Module):
    def __init__(self, d_row=64, d_col=64, d_local=64, uni_dim=64, hidden=512, dropout=0.3, tau=0.8):
        super().__init__()
        self.tau = tau

        # 分支一：整行卷积（捕捉每个 cipher 行的整列分布）
        self.row_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 26)),  # 输出 (B,32,26,1)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, d_row, kernel_size=(1, 1)),  # 输出 (B,d_row,26,1)
            nn.ReLU(),
            nn.BatchNorm2d(d_row),
        )

        # 分支二：整列卷积（捕捉跨行的列模式）
        self.col_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(26, 1)),  # 输出 (B,32,1,26)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, d_col, kernel_size=(1, 1)),  # 输出 (B,d_col,1,26)
            nn.ReLU(),
            nn.BatchNorm2d(d_col),
        )

        # 分支三：局部 3×3 卷积（捕捉邻域 bigram 模式）
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, d_local, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(d_local),
        )

        # 将三路特征融合为行级 token（26 个 token）
        # row_features: (B, d_row, 26, 1) → (B, 26, d_row)
        # col_features: (B, d_col, 1, 26) → 转置后作为列上下文，再聚合为每行的统计
        # local_features: (B, d_local, 26, 26) → 对列做池化得到每行局部摘要
        self.row_pool = nn.AdaptiveAvgPool2d((26, 1))   # 保留26行，压列到1
        self.col_pool = nn.AdaptiveAvgPool2d((1, 26))   # 保留26列，压行为1
        self.local_row_pool = nn.AdaptiveAvgPool2d((26, 1))  # 对局部特征按列池化

        # unigram 映射到行级标量/小向量
        self.unigram_mlp = nn.Sequential(
            nn.Linear(26, uni_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(uni_dim, uni_dim)
        )

        # 行级 token 的融合 MLP
        # 每个行 token 的维度 = d_row + d_col + d_local + uni_dim
        token_dim = d_row + d_col + d_local + uni_dim
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 列原型（26 个 plain 列的原型向量）
        self.col_prototypes = nn.Parameter(torch.randn(26, hidden))

        # 归一化与轻微输出 MLP（可选）
        self.token_norm = nn.LayerNorm(hidden)
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x):
        """
        x: (B, 702) = [26 unigram | 26*26 bigram]
        返回：
            logits: (B, 26, 26)
            probs:  (B, 26, 26)
        """
        B = x.size(0)
        uni = x[:, :26]                          # (B, 26)
        bi = x[:, 26:].view(B, 1, 26, 26)        # (B, 1, 26, 26)

        # 三路 CNN
        row_feat = self.row_cnn(bi)              # (B, d_row, 26, 1)
        col_feat = self.col_cnn(bi)              # (B, d_col, 1, 26)
        local_feat = self.local_cnn(bi)          # (B, d_local, 26, 26)

        # 池化到行级视角
        row_feat = row_feat.view(B, -1, 26)      # (B, d_row, 26)
        row_feat = row_feat.transpose(1, 2)      # (B, 26, d_row)

        col_feat = col_feat.view(B, -1, 26)      # (B, d_col, 26)
        col_feat = col_feat.transpose(1, 2)      # (B, 26, d_col)

        local_row = self.local_row_pool(local_feat)   # (B, d_local, 26, 1)
        local_row = local_row.view(B, -1, 26).transpose(1, 2)  # (B, 26, d_local)

        # unigram 行级特征：为每个行 token 提供全局频率的嵌入（复制到每行）
        uni_emb = self.unigram_mlp(uni)          # (B, uni_dim)
        uni_rows = uni_emb.unsqueeze(1).expand(B, 26, uni_emb.size(-1))  # (B,26,uni_dim)

        # 融合为行 token
        tokens = torch.cat([row_feat, col_feat, local_row, uni_rows], dim=-1)  # (B,26, token_dim)
        tokens = self.token_proj(tokens)                                        # (B,26, hidden)
        tokens = self.token_norm(tokens)

        # 与列原型做匹配得到 logits
        logits = torch.matmul(self.out_norm(tokens), self.col_prototypes.t())   # (B,26,26)
        probs = F.softmax(logits / self.tau, dim=-1)
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
#model = CNNPermNet().to(device)
model = ImprovedCNNPermNet().to(device)
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
import matplotlib.pyplot as plt

train_loss_list = []
val_acc_list = []
lr_list = []
full_match_rate_list = []
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0

    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)

        logits, _ = model(Xb)

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
            preds = hungarian_decode(probs)
            all_pred.append(preds)
            all_true.append(Yb.numpy())

    all_pred = np.vstack(all_pred)
    all_true = np.vstack(all_true)

    key_acc = evaluate_key_accuracy(all_true, all_pred)
    full_match_rate = np.mean(np.all(all_true == all_pred, axis=1))

    scheduler.step(key_acc)


    train_loss_list.append(total_loss)
    val_acc_list.append(key_acc)
    full_match_rate_list.append(full_match_rate)
    lr_list.append(optimizer.param_groups[0]["lr"])

    print(f"[Epoch {epoch}/{EPOCHS}] loss={total_loss:.4f} | key-accuracy={key_acc:.4f} | full-match={full_match_rate:.4f}")

full_acc = np.mean(np.all(all_true == all_pred, axis=1))
print(f"Full key match rate: {full_acc:.4f}")
print("训练完成！")

df_test=pd.read_csv("./new_data/new_dataset_test.csv")
df_test["inv_key"]=df_test["inv_key"].apply(lambda x: literal_eval(x))
df_test["y"]=df_test["inv_key"].apply(convert_to_int_list)
X_test = df_test[[f"freq_{i}" for i in range(702)]].values.astype("float32")

Y_test = np.vstack(df_test["y"].values).astype("int64")
test_dataset = Mydataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)



model.eval()
all_true = []
all_pred = []
with torch.no_grad():
    for Xb, Yb in test_loader:
        Xb = Xb.to(device)
        logits, probs = model(Xb)
        # 防止非法数据
        # probs = torch.clamp(probs, min=1e-8)
        preds = hungarian_decode(probs)
        all_pred.append(preds)
        all_true.append(Yb.numpy())

all_pred = np.vstack(all_pred)
all_true = np.vstack(all_true)

key_acc = evaluate_key_accuracy(all_true, all_pred)
scheduler.step(key_acc)
print(f"key-accuracy_test={key_acc:.4f}")


# =============================
# Plot 1: Train Loss
# =============================
plt.figure(figsize=(8,5))
plt.plot(train_loss_list, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# =============================
# Plot 2: Validation Key Accuracy
# =============================
plt.figure(figsize=(8,5))
plt.plot(val_acc_list, label="Val Key Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy (Key Match)")
plt.legend()
plt.grid(True)
plt.show()

# =============================
# Plot 3: Full Match Rate (Optional)
# =============================
plt.figure(figsize=(8,5))
plt.plot(full_match_rate_list, label="Full Key Match Rate")
plt.xlabel("Epoch")
plt.ylabel("Full Match Rate")
plt.title("Full Key Match Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# =============================
# Plot 4: Learning Rate (Optional)
# =============================
plt.figure(figsize=(8,5))
plt.plot(lr_list, label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.show()
