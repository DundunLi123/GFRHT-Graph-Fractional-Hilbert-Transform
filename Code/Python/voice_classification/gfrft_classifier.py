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
from torch_gfrft.utils import get_matvec_tensor_einsum_str, is_hermitian
import cvxpy as cp
import joblib
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dfrft_classifier import load_language_data,create_frame_dataset,plot_training_curves
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


class GFRFT:
    def __init__(self, gft_mtx: th.Tensor) -> None:
        # 判断输入矩阵是否为厄米矩阵
        if is_hermitian(gft_mtx):
            self._eigvals, self._eigvecs = th.linalg.eigh(gft_mtx)
            self._inv_eigvecs = self._eigvecs.H
        else:
            self._eigvals, self._eigvecs = th.linalg.eig(gft_mtx)
            self._inv_eigvecs = th.linalg.inv(self._eigvecs)

    def gfrft_mtx(self, a: float | th.Tensor) -> th.Tensor:
        """计算分数阶傅里叶变换矩阵"""
        fractional_eigvals = self._eigvals ** a
        return th.einsum("ij,j,jk->ik", self._eigvecs, fractional_eigvals, self._inv_eigvecs)

    def igfrft_mtx(self, a: float | th.Tensor) -> th.Tensor:
        """计算逆分数阶傅里叶变换矩阵"""
        return self.gfrft_mtx(-a)  # 逆变换使用 -a



# ========================
# 2. GFRFT模型和特征提取
# ========================
class GraphFractionalHilbertTransform(nn.Module):
    """可微分图分数阶希尔伯特变换"""

    def __init__(self):
        super().__init__()

    def forward(self, frames, A, alpha, beta):
        batch_size, n = frames.shape
        device = A.device
        dtype = th.complex64 if A.dtype == th.float32 else th.complex128

        # 转换为复数并转置为(n, batch_size)
        frames = frames.to(dtype).T  # 形状变为(n, batch_size)

        # 特征分解（只需计算一次）
        eigenvalues_A, U = th.linalg.eigh(A)
        U = U.to(dtype)
        GFT = U.T.conj()

        # GFRFT变换
        gfrft = GFRFT(GFT)
        GFRFT_mtx = gfrft.gfrft_mtx(alpha)
        GFRFT_inv_mtx = gfrft.igfrft_mtx(alpha)

        # 构造J_beta对角矩阵
        b = np.pi * beta / 2
        term1 = th.exp(-1j * b).to(dtype)
        term2 = th.exp(1j * b).to(dtype)
        phases = torch.angle(eigenvalues_A)

        J_beta_diag = torch.where(
            (phases > 0) & (phases < np.pi),
            term1,
            torch.where(
                (phases > np.pi) & (phases < 2 * np.pi),
                term2,
                (term1 + term2) / 2
            )
        )
        J_beta = th.diag(J_beta_diag)

        # 批量计算变换
        # 公式: GFRFT_inv_mtx @ J_beta @ GFRFT_mtx @ frames
        transformed = GFRFT_inv_mtx @ J_beta @ GFRFT_mtx @ frames

        # 转置回(batch_size, n)
        return transformed.T


class GraphAMFMExtractor(nn.Module):
    """可微分图AM/FM特征提取器"""

    def __init__(self):
        super().__init__()
        self.hilbert = GraphFractionalHilbertTransform()

    def unwrap_phase(self, phase):
        """改进的鲁棒相位展开"""
        phase_np = phase.cpu().detach().numpy()
        unwrapped_np = np.unwrap(phase_np)
        return torch.tensor(unwrapped_np, device=phase.device, dtype=phase.dtype)

    def unwrap_phase_batch(self, phase_batch):
        """批量相位展开"""
        unwrapped = []
        for i in range(phase_batch.shape[0]):
            unwrapped.append(self.unwrap_phase(phase_batch[i]))
        return torch.stack(unwrapped)

    def forward(self, frames, A, alpha, beta):
        #print(f"[AMFM] 输入 frames.shape: {frames.shape}")

        batch_size, n = frames.shape
        dtype = th.complex64 if A.dtype == th.float32 else th.complex128

        # 计算图分数阶希尔伯特变换（批量）
        gfrht = self.hilbert(frames, A, alpha, beta)  # (batch_size, n)

        # 构造解析信号
        # 确保数据类型一致
        frames_complex = frames.to(dtype)
        fr_analy_signal = frames_complex + 1j * gfrht

        # 计算图AM和FM
        graph_AM = torch.abs(fr_analy_signal)
        graph_phase = torch.angle(fr_analy_signal)
        graph_phase_unwrapped = self.unwrap_phase_batch(graph_phase)

        # 计算图FM (使用批处理矩阵乘法)
        # 注意: A 是实矩阵，graph_phase_unwrapped 是实张量
        graph_FM = graph_phase_unwrapped - torch.matmul(graph_phase_unwrapped, A.T)
        #print(f"[AMFM] AM shape: {graph_AM.shape}, FM shape: {graph_FM.shape}")
        # 重新计算GFRFT用于特征变换
        eigenvalues_A, U = th.linalg.eigh(A)
        U = U.to(dtype)
        GFT = U.T.conj()
        gfrft = GFRFT(GFT)
        GFRFT_mtx = gfrft.gfrft_mtx(alpha)

        # 确保数据类型一致 - 将实特征转换为复数
        graph_AM_complex = graph_AM.to(dtype)
        graph_FM_complex = graph_FM.to(dtype)

        # 使用GFRFT变换AM/FM特征（批处理）
        gfrft_am = torch.abs(torch.matmul(graph_AM_complex, GFRFT_mtx.T))
        gfrft_fm = torch.abs(torch.matmul(graph_FM_complex, GFRFT_mtx.T))
        #print(f"[AMFM] GFRFT_am shape: {gfrft_am.shape}, GFRFT_fm shape: {gfrft_fm.shape}")
        # 拼接特征
        features = torch.cat((gfrft_am, gfrft_fm), dim=1)
        #print(f"[AMFM] 拼接后 features shape: {features.shape}")
        return features


# ========================
# 3. GFT特征提取
# ========================
def optimize_A(X_learn):
    #print(f"优化使用的 X_learn.T shape: {X_learn.T.shape}")
    """使用CVXPY优化邻接矩阵A"""
    N = X_learn.shape[0]  # 节点数 = 帧长 (50)
    M = X_learn.shape[1]  # 帧数

    # 创建优化变量
    A = cp.Variable((N, N))

    # 约束条件
    constraints = [
        cp.diag(A) == 0,  # 无自环
        A @ np.ones(N) == 1,  # 行和为1
        A.T @ np.ones(N) == 1  # 列和为1
    ]

    # 目标函数: min ||X - AX||_2^2
    objective = cp.Minimize(cp.sum_squares(X_learn - A @ X_learn))

    # 求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, max_iters=100, verbose=True)

    if problem.status != cp.OPTIMAL:
        print(f"优化未收敛: {problem.status}")
        # 使用近似解
        A_opt = A.value
    else:
        A_opt = A.value

    # 数值稳定性处理
    A_opt = np.maximum(A_opt, 0)  # 确保非负
    row_sums = A_opt.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 避免除以零
    A_opt = A_opt / row_sums[:, np.newaxis]

    return A_opt


class GFRFTLanguageClassifier(nn.Module):
    """端到端性别分类模型 (使用GFRFT特征)"""

    def __init__(self, frame_length=50, hidden_neurons=10, adjacency_matrix=None):
        super().__init__()
        self.frame_length = frame_length
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.amfm_extractor = GraphAMFMExtractor()

        # 邻接矩阵处理
        if adjacency_matrix is not None:
            assert adjacency_matrix.shape == (frame_length, frame_length), \
                f"邻接矩阵形状应为({frame_length}, {frame_length})，实际为{adjacency_matrix.shape}"
            self.A = torch.tensor(adjacency_matrix, dtype=torch.float32)
        else:
            warnings.warn("未提供邻接矩阵，使用单位矩阵初始化 - 这可能导致性能下降！")
            self.A = torch.eye(frame_length, dtype=torch.float32)

        self.register_buffer('fixed_A', self.A.clone().detach())

        # 修改输出层为2个神经元（中文/英文）
        self.classifier = nn.Sequential(
            nn.Linear(2 * frame_length, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_neurons, 2)  # 二分类
        )

    def set_adjacency_matrix(self, A):
        """设置邻接矩阵A"""
        if A.shape != (self.frame_length, self.frame_length):
            raise ValueError(f"邻接矩阵形状应为({self.frame_length}, {self.frame_length})，实际为{A.shape}")

        if isinstance(A, torch.Tensor):
            self.A = A.clone().to(torch.float32)
        else:
            self.A = torch.tensor(A, dtype=torch.float32)

        self.fixed_A = self.A.clone().detach()

    def forward(self, x):
        #print(f"[FORWARD] 输入 x.shape: {x.shape}")

        alpha = torch.clamp(self.alpha, 0.0, 2.0)
        beta = torch.clamp(self.beta, 0.0, 2.0)

        # 直接处理整个批次
        features_batch = self.amfm_extractor(x, self.A, alpha, beta)
        #print(f"[FORWARD] 提取的 features_batch.shape: {features_batch.shape}")
        return self.classifier(features_batch)

    def get_alphabeta_values(self):
        """获取当前的alpha和beta值"""
        return self.alpha.item(), self.beta.item()

    def extract_gfrft_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            alpha = torch.clamp(self.alpha, 0.0, 2.0)
            beta = torch.clamp(self.beta, 0.0, 2.0)
        features = self.amfm_extractor(x, self.A, alpha, beta)  # x: [batch, frame_len]
        return features

    def get_adjacency_matrix(self):
        """获取当前邻接矩阵的副本"""
        return self.A.clone().cpu().numpy()


# ========================
# 4. 训练与评估函数
# ========================

def train_model(model, train_loader, val_loader, epochs=20, lr=0.005, model_name="model"):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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

        alpha, beta = model.get_alphabeta_values()
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"alpha: {alpha:.4f} | beta: {beta:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")
        scheduler.step()

    print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    return history

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
    # 优化邻接矩阵A
    # =================================================================
    print("\n优化邻接矩阵A...")
    # 检查文件是否已保存
    if os.path.exists('adjacency_matrix.pkl'):
        A = joblib.load('adjacency_matrix.pkl')
        print("邻接矩阵A已加载")
    else:
        A = optimize_A(X_learn.T)
        joblib.dump(A, 'adjacency_matrix.pkl')
        print("邻接矩阵A已计算并保存")

    np.save("optimized_A.npy", A)

    # =================================================================
    # 分类器2: GFRFT特征分类器 (端到端训练)
    # =================================================================
    print("\n" + "=" * 50)
    print("训练GFRFT特征分类器")
    print("=" * 50)

    # 创建GFRFT分类器并设置邻接矩阵
    model_gfrft = GFRFTLanguageClassifier(frame_length=50, hidden_neurons=256, adjacency_matrix=A)
    model_gfrft.set_adjacency_matrix(A)

    # 训练GFRFT分类器
    history_gfrft = train_model(
        model_gfrft, train_loader, val_loader,
        epochs=100, lr=0.001, model_name="gfrft_language_classifier"
    )
    # 绘制训练曲线
    plot_training_curves(history_gfrft, "GFRFT情绪分类器")
    # 在测试集上评估
    test_acc_gfrft = evaluate_model(model_gfrft, test_loader)
    print(f"GFRFT分类器测试准确率: {test_acc_gfrft:.4f}")


    # 加载最佳模型再评估
    print("\n重新加载最佳模型权重并在测试集评估...")
    best_model = GFRFTLanguageClassifier(frame_length=50, hidden_neurons=256, adjacency_matrix=A)
    best_model.load_state_dict(torch.load("best_gfrft_language_classifier.pth"))
    best_model.eval()
    best_model.to(torch.device("cpu"))
    test_acc_best = evaluate_model(best_model, test_loader)

    print(f"最佳模型参数: alpha = {best_model.alpha.item():.4f}, beta = {best_model.beta.item():.4f}")
    print(f"最佳 GFRFT 模型测试准确率: {test_acc_best:.4f}")

    # 保存结果
    plt.figure(figsize=(10, 6))
    plt.bar(['GFRFT'], [test_acc_gfrft], color='green')
    plt.xlabel('分类器类型')
    plt.ylabel('测试准确率')
    plt.title('GFRFT分类器性能')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.text(0, test_acc_gfrft + 0.01, f"{test_acc_gfrft:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('gfrft_classifier_performance.png')
    plt.show()


if __name__ == "__main__":
    main()