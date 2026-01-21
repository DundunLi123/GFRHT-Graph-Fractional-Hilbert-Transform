import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import joblib
from sklearn.neural_network import MLPClassifier

from dfrft_classifier_fixed_pq import DFRFTLanguageClassifier, AMFMExtractor, load_language_data, frame_signal, set_seed,create_frame_dataset
from gfrft_classifier_fixed_ab import GFRFTLanguageClassifier, optimize_A

# 设置中文显示
plt.rcParams['font.family'] = 'SimHei'


# ========================
# 训练与评估函数
# ========================
def train_model(model, train_loader, val_loader, epochs=20, lr=0.005, model_name="model"):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=5e-4)
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

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | ")

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


# class SimpleClassifier(nn.Module):
#     """简单分类器 (用于组合特征)"""
#
#     def __init__(self, input_dim, hidden_neurons=10):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_neurons),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(hidden_neurons, 2)
#         )
#
#     def forward(self, x):
#
#         return self.net(x)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)



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


    # =================================================================
    # 加载预训练模型
    # =================================================================
    print("\n加载预训练模型...")

    # 加载DFRFT模型
    model_dfrft = DFRFTLanguageClassifier(frame_length=50, hidden_neurons=256)
    if os.path.exists("best_language_classifier_fixed_pq.pth"):
        model_dfrft.load_state_dict(torch.load("best_language_classifier_fixed_pq.pth"))
        print("DFRFT模型加载成功")
    else:
        print("警告: 未找到DFRFT模型，请先运行dfrft_classifier_fixed.py")
        return

    # 加载GFRFT模型
    # 先加载邻接矩阵
    if os.path.exists("adjacency_matrix.pkl"):
        A = joblib.load("adjacency_matrix.pkl")
        print("邻接矩阵A已加载")
    else:
        print("警告: 未找到邻接矩阵，请先运行gfrft_classifier.py")
        return

    model_gfrft = GFRFTLanguageClassifier(frame_length=50, hidden_neurons=1, adjacency_matrix=A)
    if os.path.exists("best_gfrft_language_classifier_fixed_ab.pth"):
        model_gfrft.load_state_dict(torch.load("best_gfrft_language_classifier_fixed_ab.pth"))
        print("GFRFT模型加载成功")
    else:
        print("警告: 未找到GFRFT模型，请先运行gfrft_classifier_fixed.py")
        return

    # =================================================================
    # 提取特征
    # =================================================================
    print("\n提取特征...")

    # 提取DFRFT特征
    with torch.no_grad():
        X_train_dfrft = model_dfrft.extract_dfrft_features(X_train_tensor).numpy()
        X_val_dfrft = model_dfrft.extract_dfrft_features(X_val_tensor).numpy()
        X_test_dfrft = model_dfrft.extract_dfrft_features(X_test_tensor).numpy()

    # 提取GFRFT特征
    with torch.no_grad():
        X_train_gfrft = model_gfrft.extract_gfrft_features(X_train_tensor).numpy()
        X_val_gfrft = model_gfrft.extract_gfrft_features(X_val_tensor).numpy()
        X_test_gfrft = model_gfrft.extract_gfrft_features(X_test_tensor).numpy()

    # 组合特征
    X_train_combined = np.hstack((X_train_dfrft, X_train_gfrft))
    X_val_combined = np.hstack((X_val_dfrft, X_val_gfrft))
    X_test_combined = np.hstack((X_test_dfrft, X_test_gfrft))

    # 转换为PyTorch张量
    X_train_combined_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
    X_val_combined_tensor = torch.tensor(X_val_combined, dtype=torch.float32)
    X_test_combined_tensor = torch.tensor(X_test_combined, dtype=torch.float32)

    # 创建数据加载器
    batch_size = 256
    train_combined_dataset = data.TensorDataset(X_train_combined_tensor, y_train_tensor)
    val_combined_dataset = data.TensorDataset(X_val_combined_tensor, y_val_tensor)
    test_combined_dataset = data.TensorDataset(X_test_combined_tensor, y_test_tensor)

    train_combined_loader = data.DataLoader(train_combined_dataset, batch_size=batch_size, shuffle=True)
    val_combined_loader = data.DataLoader(val_combined_dataset, batch_size=batch_size)
    test_combined_loader = data.DataLoader(test_combined_dataset, batch_size=batch_size)

    # 用于单独评估 DFRFT 和 GFRFT 模型的测试加载器
    test_dfrft_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)
    test_dfrft_loader = data.DataLoader(test_dfrft_dataset, batch_size=batch_size)

    test_gfrft_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)
    test_gfrft_loader = data.DataLoader(test_gfrft_dataset, batch_size=batch_size)

    # =================================================================
    # 训练组合特征分类器
    # =================================================================
    print("\n" + "=" * 50)
    print("训练组合特征分类器")
    print("=" * 50)

    model_combined = SimpleClassifier(input_dim=200)
    history_combined = train_model(
        model_combined, train_combined_loader, val_combined_loader,
        epochs=100, lr=0.001, model_name="combined_classifier_fixed"
    )

    # 在测试集上评估
    test_acc_combined = evaluate_model(model_combined, test_combined_loader)
    print(f"组合分类器测试准确率: {test_acc_combined:.4f}")


    # 加载最佳模型再评估
    print("\n重新加载最佳模型权重并在测试集评估...")
    best_model = SimpleClassifier(input_dim=200)
    best_model.load_state_dict(torch.load("best_combined_classifier_fixed.pth"))
    best_model.eval()
    best_model.to(torch.device("cpu"))
    test_acc_best_combined = evaluate_model(best_model, test_combined_loader)
    print(f"最佳混合模型测试准确率: {test_acc_best_combined:.4f}")

    # 加载各模型的测试准确率
    test_acc_dfrft = evaluate_model(model_dfrft, test_dfrft_loader)
    test_acc_gfrft = evaluate_model(model_gfrft, test_gfrft_loader)

    # 结果比较
    results = {
        'DFRFT_fixed': test_acc_dfrft,
        'GFRFT_fixed': test_acc_gfrft,
        'Combined_fixed': test_acc_best_combined
    }

    # 可视化结果
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = list(results.values())

    plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.xlabel('分类器类型')
    plt.ylabel('测试准确率')
    plt.title('不同特征提取方法的分类性能比较')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig('classifier_comparison_fixed.png')
    plt.show()


if __name__ == "__main__":
    main()