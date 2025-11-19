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


# ---------------------------
# 1. ResNet bottleneck block
# ---------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super().__init__()
        mid = out_c // 2

        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(),

            nn.Conv2d(mid, mid, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(),

            nn.Conv2d(mid, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        )

        self.shortcut = nn.Sequential()
        if downsample or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


# ---------------------------
# 2. Attention Pooling
# ---------------------------
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, HW, C)

        att = torch.softmax((x @ self.query.T).squeeze(-1), dim=1)  # (B, HW)
        pooled = (x * att.unsqueeze(-1)).sum(dim=1)  # (B, C)
        return pooled


# ---------------------------
# 3. Unigram Feature Mixer
# ---------------------------
class UniMixer(nn.Module):
    def __init__(self, dim=26, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)  # (B, hidden)


# ---------------------------
# 4. The New CNNPermNet-V2
# ---------------------------
class CNNPermNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        # ▶ Bigram 26×26 → ResNet backbone
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer1 = Bottleneck(32, 64, downsample=True)   # 26→13
        self.layer2 = Bottleneck(64, 128, downsample=True)  # 13→7
        self.layer3 = Bottleneck(128, 128)

        # Attention pooling on final CNN features
        self.att_pool = AttentionPool(128)

        # ▶ Unigram feature mixer (stronger than your MLP)
        self.uni_mixer = UniMixer(dim=26, hidden=128)

        # ▶ Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 26 * 26),
        )

    def forward(self, x):
        """
        输入 x: (B, 702)
        前 26 → unigram
        后 676 → bigram 26×26
        """
        B = x.size(0)
        uni = x[:, :26]                    # (B,26)
        bi = x[:, 26:].view(B, 1, 26, 26)  # (B,1,26,26)

        # CNN bigram path
        out = self.stem(bi)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        cnn_feat = self.att_pool(out)  # (B,128)

        # Unigram path
        uni_feat = self.uni_mixer(uni)  # (B,128)

        # Fusion
        feat = torch.cat([cnn_feat, uni_feat], dim=1)

        logits = self.fusion(feat)
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
#model = CNNPermNetV2().to(device)
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
