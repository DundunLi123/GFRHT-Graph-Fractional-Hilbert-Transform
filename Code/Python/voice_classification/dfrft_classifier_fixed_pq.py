import os
import sys
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import torch.utils.data as data
from math import ceil
import torch as th
from tqdm import tqdm
import dfrft

# 设置中文显示
plt.rcParams['font.family'] = 'SimHei'
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================
# 1. 数据加载与预处理
# ========================
def load_language_data(data_dir, sr=16000, max_files=None):
    """加载语种数据集（中文和英文）"""
    all_signals = []
    all_labels = []  # 0: 中文, 1: 英文

    # 遍历中文和英文目录
    for lang_dir, label in [("Chinese", 0), ("English", 1)]:
        lang_path = os.path.join(data_dir, lang_dir)
        if not os.path.exists(lang_path):
            print(f"警告: 目录不存在 {lang_path}")
            continue

        files = [f for f in os.listdir(lang_path) if f.lower().endswith(".mp3")]
        file_count = 0

        for file in files:
            if max_files and file_count >= max_files:
                break

            file_path = os.path.join(lang_path, file)
            try:
                signal, _ = librosa.load(file_path, sr=sr, mono=True)
                all_signals.append(signal)
                all_labels.append(label)
                file_count += 1
            except Exception as e:
                print(f"加载文件 {file_path} 出错: {e}")

    print(f"加载完成! 中文样本: {all_labels.count(0)}个, 英文样本: {all_labels.count(1)}个")
    return all_signals, all_labels


def frame_signal(signal, frame_length=50, frame_shift=25):
    """将语音信号分帧"""
    if len(signal) < frame_length:
        padding = np.zeros(frame_length - len(signal))
        signal = np.concatenate((signal, padding))

    frames = []
    n_frames = (len(signal) - frame_length) // frame_shift + 1
    for i in range(n_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = signal[start:end]
        frame = frame * np.hanning(frame_length)
        frames.append(frame)
    return np.array(frames)


def create_frame_dataset(signals, labels, frame_length=50, frame_shift=25):
    """将信号列表转换为帧数据集

    参数:
        signals: 语音信号列表
        labels: 对应的标签列表
        frame_length: 帧长度 (ms)
        frame_shift: 帧移 (ms)

    返回:
        frames: 帧数组 (n_frames, frame_length)
        frame_labels: 对应的标签数组
    """
    all_frames = []
    all_labels = []

    for signal, label in zip(signals, labels):
        frames = frame_signal(signal, frame_length, frame_shift)
        all_frames.append(frames)
        all_labels.extend([label] * len(frames))

    # 合并所有帧
    frames_array = np.vstack(all_frames)
    labels_array = np.array(all_labels)

    print(f"创建帧数据集: 总帧数={len(frames_array)}, 帧形状={frames_array.shape}")
    return frames_array, labels_array


# ========================
# 2. DFRFT模型和特征提取
# ========================
class FractionalHilbertTransform(nn.Module):
    """可微分分数阶希尔伯特变换"""

    def __init__(self, N=50):
        super().__init__()
        self.N = N

    def forward(self, x, Q, P):
        Q = torch.clamp(Q, 0, 4)

        # print(f"[DEBUG] Calling dfrft with x.shape={x.shape}, Q={Q}, dim=1")
        X_Q = dfrft.dfrft(x, Q, dim=1)

        alpha = P * torch.pi / 2
        if self.N % 2 == 0:
            half_len = self.N // 2 - 1
            mask = torch.cat([
                torch.cos(alpha).unsqueeze(0),
                torch.exp(-1j * alpha).repeat(half_len),
                torch.cos(alpha).unsqueeze(0),
                torch.exp(1j * alpha).repeat(half_len)
            ])
        else:
            half_len = (self.N - 1) // 2
            mask = torch.cat([
                torch.cos(alpha).unsqueeze(0),
                torch.exp(-1j * alpha).repeat(half_len),
                torch.exp(1j * alpha).repeat(half_len)
            ])

        X_masked = X_Q * mask
        x_frac_hilbert = dfrft.dfrft(X_masked, -Q, dim=-1)
        return x_frac_hilbert


class AMFMExtractor(nn.Module):
    """可微分AM/FM特征提取器"""

    def __init__(self, frame_length=50):
        super().__init__()
        self.frame_length = frame_length
        self.hilbert = FractionalHilbertTransform(frame_length)

    def forward(self, x, Q, P):
        # print(f"[DEBUG] AMFMExtractor input x.shape: {x.shape}")
        hilbert_out = self.hilbert(x, Q, P)
        # analytic_signal = x + 1j * hilbert_out.real
        analytic_signal = x + 1j * hilbert_out
        am = torch.abs(analytic_signal)
        phase = torch.angle(analytic_signal)
        phase_diff = torch.diff(phase, dim=1, prepend=phase[:, :1])
        fm = phase_diff

        dfrft_am = torch.abs(dfrft.dfrft(am, Q, dim=-1))
        dfrft_fm = torch.abs(dfrft.dfrft(fm, Q, dim=-1))

        features = torch.cat((dfrft_am, dfrft_fm), 1)
        # print(f"[DEBUG] AMFMExtractor output feature.shape: {features.shape}")
        return features


# 修改：重命名为情绪分类器，输出层改为8个神经元
class DFRFTLanguageClassifier(nn.Module):
    """端到端情绪分类模型 (使用DFRFT特征)"""

    def __init__(self, frame_length=50, hidden_neurons=32):
        super().__init__()
        self.frame_length = frame_length
        self.P = torch.tensor(1.0)
        self.Q = torch.tensor(1.0)
        self.amfm_extractor = AMFMExtractor(frame_length)

        # 修改：输出层改为8个神经元（对应8种情绪）
        self.classifier = nn.Sequential(
            nn.Linear(2 * frame_length, hidden_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_neurons, 2)  # 8种情绪
        )

    def forward(self, x):
        P = self.P
        Q = self.Q
        features = self.amfm_extractor(x, Q, P)
        # print(f"[DEBUG] Extracted features shape: {features.shape}")  # 应该是 (batch_size, 100)
        logits = self.classifier(features)
        return logits

    def get_pq_values(self):
        """获取当前的P和Q值"""
        return self.P.item(), self.Q.item()

    def extract_dfrft_features(self, x):
        """提取DFRFT特征"""
        with torch.no_grad():
            P = self.P
            Q = self.Q
            return self.amfm_extractor(x, Q, P)


# ========================
# 3. 训练与评估函数
# ========================
def train_model(model, train_loader, val_loader, epochs=20, lr=0.005, model_name="model"):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],  # 新增：记录训练准确率
        'val_acc': [],
        'val_loss': []  # 新增：记录验证损失
    }

    print(f"开始训练模型: {model_name}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        # 计算训练损失和准确率
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 计算验证损失和准确率
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        p, q = model.get_pq_values()
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"P: {p:.4f} | Q: {q:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")

    print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    return history


# 添加绘制训练曲线的函数
def plot_training_curves(history, model_name):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 10))

    # 创建损失曲线图 - 使用不同颜色和线型
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], 'b-', linewidth=2, label='训练损失')  # 蓝色实线
    plt.plot(history['val_loss'], 'r--', linewidth=2, label='验证损失')  # 红色虚线
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(f'{model_name} - 训练和验证损失')
    plt.legend()
    plt.grid(True)

    # 创建准确率曲线图 - 使用不同颜色和线型
    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], 'g-', linewidth=2, label='训练准确率')  # 绿色实线
    plt.plot(history['val_acc'], 'm--', linewidth=2, label='验证准确率')  # 品红虚线
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title(f'{model_name} - 训练和验证准确率')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png', dpi=300)  # 提高图像分辨率
    plt.show()

def evaluate_model(model, data_loader):
    """评估模型性能"""
    device = torch.device("cpu")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    return accuracy


# ========================
# 主程序
# ========================
def main():
    set_seed(1)

    # 设置语种数据集路径
    language_dir = "./Language_Dataset2"  # 替换为你的语种数据集路径

    # 加载语种数据
    print("加载语种数据集...")
    signals, labels = load_language_data(  # 使用新加载函数
        language_dir,
        sr=16000,
        max_files=50  # 每类最大加载文件数
    )

    # 创建帧数据集（保持原有处理逻辑）
    print("创建帧数据集...")
    X, y = create_frame_dataset(signals, labels)

    # 确保每帧长度为50
    if X.shape[1] != 50:
        X = X[:, :50]

    # 取部分帧（如果帧数过多）
    max_frames = 10000  # 限制帧数，避免内存问题
    if len(X) > max_frames:
        indices = np.random.choice(len(X), max_frames, replace=False)
        X = X[indices]
        y = y[indices]

    # 划分数据集 (使用分层抽样保持类别分布)
    X_learn, X_train_test, y_learn, y_train_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, y_train_test, test_size=0.2, random_state=42, stratify=y_train_test
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    # 验证维度
    print(f"学习集形状: {X_learn.shape}")
    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建数据加载器
    batch_size = 64
    train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    # =================================================================
    # 分类器: DFRFT情绪分类器 (特征长度=100)
    # =================================================================
    print("\n" + "=" * 50)
    print("训练DFRFT情绪分类器")
    print("=" * 50)

    # 修改：使用情绪分类器
    model_dfrft = DFRFTLanguageClassifier(frame_length=50, hidden_neurons=256)

    history_dfrft = train_model(
        model_dfrft, train_loader, val_loader,
        epochs=50, lr=0.001, model_name="language_classifier_fixed_pq"
    )
    # 绘制训练曲线
    plot_training_curves(history_dfrft, "DFRFT情绪分类器")

    # 在测试集上评估
    test_acc_dfrft = evaluate_model(model_dfrft, test_loader)
    print(f"DFRFT语种分类器测试准确率: {test_acc_dfrft:.4f}")

    # 加载最佳模型再评估
    print("\n重新加载最佳模型权重并在测试集评估...")
    best_model = DFRFTLanguageClassifier(frame_length=50, hidden_neurons=256)
    best_model.load_state_dict(torch.load("best_language_classifier_fixed_pq.pth"))
    best_model.eval()
    best_model.to(torch.device("cpu"))
    test_acc_best = evaluate_model(best_model, test_loader)

    # ✅ 打印 P 和 Q
    print(f"最佳模型参数: P = {best_model.P.item():.4f}, Q = {best_model.Q.item():.4f}")
    print(f"最佳 DFRFT 语种分类器测试准确率: {test_acc_best:.4f}")

    # 保存结果
    plt.figure(figsize=(10, 6))
    plt.bar(['DFRFT'], [test_acc_dfrft], color='blue')
    plt.xlabel('分类器类型')
    plt.ylabel('测试准确率')
    plt.title('DFRFT语种分类器性能')
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.text(0, test_acc_dfrft + 0.01, f"{test_acc_dfrft:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('language_classifier_fixed_pq_performance.png')
    plt.show()


if __name__ == "__main__":
    main()