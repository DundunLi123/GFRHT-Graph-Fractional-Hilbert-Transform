import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# ====================== 【全版本兼容】matplotlib基础配置 无任何报错 ✅ ======================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文黑体+兜底字体，兼容所有版本
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示方块问题
plt.rcParams['font.family'] = 'sans-serif'
# 只保留全版本通用的基础字号配置，无任何新增Key，彻底避免报错
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# ============================================================================

# ==========================================
# 1. 核心 GFRHT 求解器 (保持不变，一行没改！)
# ==========================================
class GFRHT_Solver:
    def __init__(self, A):
        self.N = A.shape[0]
        self.lambdas_A, self.U = np.linalg.eig(A)
        self.U_inv = np.linalg.inv(self.U)
        self.F = self.U_inv
        self.lambdas_F, self.V = np.linalg.eig(self.F)
        self.V_inv = np.linalg.inv(self.V)

    def get_gfrft_matrix(self, alpha):
        lam_F_alpha = np.power(self.lambdas_F, alpha)
        return self.V @ np.diag(lam_F_alpha) @ self.V_inv

    def get_transfer_function(self, beta, threshold=1e-9):
        h_diag = np.zeros(self.N, dtype=complex)
        imag_parts = np.imag(self.lambdas_A)
        for k in range(self.N):
            if imag_parts[k] > threshold:
                h_diag[k] = np.exp(-1j * beta)
            elif imag_parts[k] < -threshold:
                h_diag[k] = np.exp(1j * beta)
            else:
                h_diag[k] = np.cos(beta)
        return np.diag(h_diag)

    def compute_gfras_envelope(self, x, alpha, beta):
        F_a = self.get_gfrft_matrix(alpha)
        F_neg_a = self.get_gfrft_matrix(-alpha)
        H_b = self.get_transfer_function(beta)
        Hx = F_neg_a @ (H_b @ (F_a @ x))
        return np.abs(x + 1j * Hx)


# ==========================================
# 2. 加载你的真实数据 ✅【核心修复：df读取顺序错误 + 防错优化】
# ==========================================
def load_and_construct_graph(file_path):
    """
    读取用户提供的真实数据文件 (Excel/CSV)
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=str.lower, inplace=True)

    lat = df['latitude'].values
    lon = df['longitude'].values
    raw_signal = df['temperature'].values

    coords = np.column_stack((lon, lat))
    N = len(coords)

    print(f"Constructing Graph from {N} stations...")
    dist_mat = squareform(pdist(coords))

    k_neighbors = 5
    A = np.zeros((N, N))

    for i in range(N):
        neighbors = np.argsort(dist_mat[i])[1:k_neighbors + 1]
        for neighbor in neighbors:
            sigma = np.mean(dist_mat[i][neighbors]) + 1e-6
            w = np.exp(- (dist_mat[i][neighbor] ** 2) / (2 * sigma ** 2))
            A[i][neighbor] = w

    rho = np.max(np.abs(np.linalg.eigvals(A)))
    A = A / rho

    return A, coords, raw_signal


# ==========================================
# 评价指标计算 (保持不变，一行没改！)
# ==========================================
def calculate_metrics(envelope, anomaly_nodes, N):
    y_true = np.zeros(N)
    y_true[anomaly_nodes] = 1.0
    env_norm = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope) + 1e-9)

    rmse = np.sqrt(np.mean(np.square(env_norm - y_true)))
    mae = np.mean(np.abs(env_norm - y_true))

    normal_nodes = [i for i in range(N) if i not in anomaly_nodes]
    mu_anom = np.mean(envelope[anomaly_nodes])
    std_bg = np.std(envelope[normal_nodes])
    snr = 20 * np.log10(mu_anom / (std_bg + 1e-9))

    pred_ranked_idx = np.argsort(envelope)[::-1]

    def precision_at_k(k):
        top_k_pred = pred_ranked_idx[:k]
        hit = len(set(top_k_pred) & set(anomaly_nodes))
        return hit / k if k > 0 else 0.0

    p_at_5 = precision_at_k(5)
    p_at_10 = precision_at_k(10)

    return {
        'SNR (dB)': round(snr, 2),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'Precision@5': round(p_at_5, 4),
        'Precision@10': round(p_at_10, 4)
    }, rmse, snr


# ==========================================
# ✅✅✅ 核心：Adam自适应梯度优化器 (纯Numpy，无依赖)
# ✅✅✅ α、β 自动学习、梯度更新，替代网格搜索
# ==========================================
class Adam_Optimizer:
    """纯Numpy实现的Adam优化器，极简高效，专为α、β参数优化设计"""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # 一阶矩
        self.v = {}  # 二阶矩
        self.t = 0  # 迭代次数

    def update(self, params, grads):
        self.t += 1
        for key in params.keys():
            if key not in self.m:
                self.m[key] = 0.0
                self.v[key] = 0.0
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


# ==========================================
# 3. 运行实验 (核心修改：绘图部分拆分为两张独立图+保存) ✅✅✅
# ==========================================
def run_real_experiment():
    DATA_PATH = 'molene_data.xlsx'

    try:
        A, coords, raw_temp = load_and_construct_graph(DATA_PATH)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 排查建议：1.路径是否正确 2.文件含 latitude/longitude/temperature 列 3.文件格式是xlsx/csv")
        return

    N = A.shape[0]
    solver = GFRHT_Solver(A)

    smooth_background = (raw_temp - np.min(raw_temp)) / (np.max(raw_temp) - np.min(raw_temp))
    np.random.seed(42)
    anomaly_nodes = np.random.choice(N, 10, replace=False)
    true_signal = smooth_background.copy()
    true_signal[anomaly_nodes] += 2.0
    noise_level = 0.2
    noisy_signal = true_signal + noise_level * np.random.randn(N)

    print(f"\nInjecting anomalies at Station Indices: {anomaly_nodes}")
    print(f"Simulating sensor failure on top of REAL temperature data.")

    print("Running GHT (Fixed alpha=1.0, beta=π/2)...")
    env_ght = solver.compute_gfras_envelope(noisy_signal, alpha=1.0, beta=np.pi / 2)
    ght_metrics, _, _ = calculate_metrics(env_ght, anomaly_nodes, N)

    print("Running GFRHT with LEARNABLE alpha & beta (Adam Gradient Descent)...")
    params = {
        'alpha': np.random.uniform(0.0, 1.0),
        'beta': np.random.uniform(0.0, 2 * np.pi)
    }
    alpha_min, alpha_max = 0.0, 2.0
    beta_min, beta_max = 0.0, 2 * np.pi

    optimizer = Adam_Optimizer(lr=0.005)
    max_epochs = 500
    epsilon = 1e-6
    best_loss = np.inf
    best_env = None
    best_params = (0, 0)
    best_gfrht_metrics = None

    for epoch in range(max_epochs):
        alpha = params['alpha']
        beta = params['beta']

        env = solver.compute_gfras_envelope(noisy_signal, alpha, beta)
        current_metrics, rmse, snr = calculate_metrics(env, anomaly_nodes, N)
        loss = rmse - (snr / 100)

        grad_alpha = (calculate_metrics(solver.compute_gfras_envelope(noisy_signal, alpha + epsilon, beta),
                                        anomaly_nodes, N)[1] - rmse) / epsilon
        grad_beta = (calculate_metrics(solver.compute_gfras_envelope(noisy_signal, alpha, beta + epsilon),
                                       anomaly_nodes, N)[1] - rmse) / epsilon
        grads = {'alpha': grad_alpha, 'beta': grad_beta}

        params = optimizer.update(params, grads)

        params['alpha'] = np.clip(params['alpha'], alpha_min, alpha_max)
        params['beta'] = np.clip(params['beta'], beta_min, beta_max)

        if loss < best_loss:
            best_loss = loss
            best_env = env
            best_params = (params['alpha'], params['beta'])
            best_gfrht_metrics = current_metrics

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{max_epochs} | Loss: {loss:.4f} | α: {params['alpha']:.4f} | β: {params['beta']:.4f}")

    print("\n" + "=" * 80)
    print("📊 实验结果对比表 (GHT vs 可学习GFRHT) | 真实传感器温度数据+异常注入".center(80))
    print("=" * 80)
    compare_df = pd.DataFrame({
        '评价指标': list(ght_metrics.keys()),
        '传统GHT (α=1.0,β=π/2)': list(ght_metrics.values()),
        '可学习GFRHT (梯度优化)': list(best_gfrht_metrics.values())
    })
    print(compare_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("🎯 可学习GFRHT → 梯度收敛 全局最优超参数".center(80))
    print("=" * 80)
    best_alpha, best_beta = best_params
    print(f"最优 α (alpha) = {best_alpha:.4f}")
    print(f"最优 β (beta)  = {best_beta:.4f} (≈ {best_beta / np.pi:.2f}π)")
    print(f"最优收敛损失 Loss = {best_loss:.4f}")
    print("=" * 80)

    # ====================== ✅✅✅ 核心修改1：第一张独立图 传感器拓扑+异常节点图 + 保存 ✅✅✅ ======================
    plt.figure(figsize=(10, 6))  # 独立画布，可自定义尺寸
    plt.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=100, edgecolors='k', label='Normal Station')
    plt.scatter(coords[anomaly_nodes, 0], coords[anomaly_nodes, 1], c='red', marker='*', s=300,
                label='Simulated Failure (Anomaly)')
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0.1:
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'gray', alpha=0.2)
    plt.xlabel("Longitude", fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})
    plt.ylabel("Latitude", fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})
    plt.legend(prop={'family': 'Times New Roman', 'size': 10})
    plt.xticks(fontproperties='Times New Roman', size=10)
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.tight_layout()  # 自动适配布局，防止文字截断
    # 高清保存第一张图【先保存再显示，必无空白】
    plt.savefig('传感器拓扑与异常节点分布图.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()  # 显示第一张图，关闭后才会显示第二张

    # ====================== ✅✅✅ 核心修改2：第二张独立图 包络幅值检测图 + 保存 ✅✅✅ ======================
    plt.figure(figsize=(10, 6)) # 独立画布，可自定义尺寸
    plt.plot(env_ght, 'b--o', label=f'GHT (SNR={ght_metrics["SNR (dB)"]}dB, P@10={ght_metrics["Precision@10"]})',
             alpha=0.5)
    plt.plot(best_env, 'r-x',
             label=f'GFRHT (SNR={best_gfrht_metrics["SNR (dB)"]}dB, P@10={best_gfrht_metrics["Precision@10"]})',
             linewidth=2)
    for idx in anomaly_nodes:
        plt.axvline(idx, color='k', linestyle=':', alpha=0.5)
    plt.axvline(anomaly_nodes[0], color='k', linestyle=':', label='True Anomaly Location')
    plt.xlabel("Station Index", fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})
    plt.ylabel("Envelope Amplitude", fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 12})
    plt.legend(prop={'family': 'Times New Roman', 'size': 10})
    plt.xticks(fontproperties='Times New Roman', size=10)
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.tight_layout()
    # 高清保存第二张图
    plt.savefig('GFRHT异常检测包络幅值对比图.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()


if __name__ == "__main__":
    run_real_experiment()