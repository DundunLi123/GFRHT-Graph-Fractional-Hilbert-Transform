clc; clear; close all;



% 加载已保存的数据
load('my_run_data.mat');

% --------- 参数设置 ----------
num_comms = 10;            % 社区数量
nodes_per_comm = 6;        % 每个社区节点数
N = num_comms * nodes_per_comm;
alpha = 1; beta = 0.1*pi/2;  % GFRHT 分数阶参数

% 真实异常节点标签（Ground Truth）
true_anomaly_indices = 18:23;
ground_truth_signal = zeros(N,1);
ground_truth_signal(true_anomaly_indices) = 1;

%% --------- 邻接矩阵归一化 ----------
A_sparse_norm = normalize_adjacency(A_sparse);
A_dense_norm = normalize_adjacency(A_dense);

%% --------- 计算 GHT 输出 ----------
y_sparse_ght = normalize_signal(graph_hilbert_transform(A_sparse_norm, x));
y_dense_ght = normalize_signal(graph_hilbert_transform(A_dense_norm, x));

%% --------- 计算 GFRHT 输出 ----------
y_sparse_gfrht = normalize_signal(graph_frhilbert_transform(A_sparse_norm, x, alpha, beta));
y_dense_gfrht = normalize_signal(graph_frhilbert_transform(A_dense_norm, x, alpha, beta));

%% --------- 计算量化指标 ----------
fprintf('=== 稀疏图性能对比 ===\n');
metrics_sparse = calculate_metrics(y_sparse_ght, y_sparse_gfrht, ground_truth_signal, true_anomaly_indices);
print_metrics(metrics_sparse);

fprintf('\n=== 稠密图性能对比 ===\n');
metrics_dense = calculate_metrics(y_dense_ght, y_dense_gfrht, ground_truth_signal, true_anomaly_indices);
print_metrics(metrics_dense);


col.blue   = [0 114 178]/255;  % #0072B2  -> GHT
col.vermil = [213 94 0]/255;   % #D55E00  -> GFRHT
col.green  = [0 158 115]/255;  % #009E73  (备用)
col.gray   = [80 80 80]/255;   % Ground Truth（中性灰，打印友好）

lw   = 2.0;       % 主线线宽
lwGT = 2.0;       % Ground Truth 线宽
fs   = 11;        % 字体大小

% ---- 全局字体设置：Times New Roman ----
tnr = 'Times New Roman';
set(groot,'DefaultAxesFontName',tnr);
set(groot,'DefaultTextFontName',tnr);
set(groot,'DefaultLegendFontName',tnr);
set(groot,'DefaultAxesFontSize',11);
set(groot,'DefaultLegendFontSize',10);
set(groot,'DefaultAxesTitleFontWeight','normal');  % 避免过粗
set(groot,'DefaultFigureColor','w');
set(groot,'DefaultFigureRenderer','painters');     % 矢量导出友好

figure('Position', [100, 100, 1200, 800], 'Color','w');


%% --------- 可视化结果 ----------
% 您可以保留原有的可视化代码，或添加新的对比图

% ---------- 稀疏图对比 ----------
ax1 = subplot(2,2,1); hold(ax1,'on');
plot(1:N, y_sparse_ght,  '-', 'LineWidth', lw,  'Color', col.blue,   'DisplayName','GHT');
plot(1:N, y_sparse_gfrht,'-', 'LineWidth', lw,  'Color', col.vermil, 'DisplayName','GFRHT');
plot(1:N, ground_truth_signal, '--', 'LineWidth', lwGT, 'Color', col.gray, 'DisplayName','Ground Truth');
title('Sparse: GHT vs GFRHT');
% xlabel('Node'); ylabel('Amplitude');
xlabel(ax1,'Node',     'FontSize',14, 'FontWeight','bold');   % ← 加粗
ylabel(ax1,'Amplitude', 'FontSize',14, 'FontWeight','bold');   % ← 加粗
legend('Location','northeast','Box','off');
set(gca,'FontSize',fs); grid on; box on;

% ---------- 稠密图对比 ----------
ax2 = subplot(2,2,2); hold(ax2,'on');
plot(1:N, y_dense_ght,   '-', 'LineWidth', lw, 'Color', col.blue,   'DisplayName','GHT');
plot(1:N, y_dense_gfrht, '-', 'LineWidth', lw, 'Color', col.vermil, 'DisplayName','GFRHT');
plot(1:N, ground_truth_signal, '--', 'LineWidth', lwGT, 'Color', col.gray, 'DisplayName','Ground Truth');
title('Dense: GHT vs GFRHT');
xlabel(ax2,'Node',  'FontSize',14,    'FontWeight','bold');   % ← 加粗
ylabel(ax2,'Amplitude', 'FontSize',14, 'FontWeight','bold');   % ← 加粗
legend('Location','northeast','Box','off');
set(gca,'FontSize',fs); grid on; box on;

% 稀疏图SNR对比
subplot(2,2,3);
bar([metrics_sparse.snr_ght, metrics_sparse.snr_gfrht]);
set(gca, 'XTickLabel', {'GHT', 'GFRHT'});
ylabel('SNR (dB)');
title('稀疏图: 信噪比对比');
grid on;

% 稠密图SNR对比
subplot(2,2,4);
bar([metrics_dense.snr_ght, metrics_dense.snr_gfrht]);
set(gca, 'XTickLabel', {'GHT', 'GFRHT'});
ylabel('SNR (dB)');
title('稠密图: 信噪比对比');
grid on;

%% ==================== 函数定义 ====================
function A_norm = normalize_adjacency(A)
    [V, D] = eig(A);
    rho = max(abs(diag(D)));
    A_norm = A / rho;
end


function y_norm = normalize_signal(y)
    y_abs = abs(y);
    y_norm = y_abs / max(y_abs);
end

function metrics = calculate_metrics(y_ght, y_gfrht, ground_truth, true_indices)
    % 1. 信噪比 (SNR) - 异常区域均值 vs 背景标准差
    signal_power_ght = mean(y_ght(true_indices));
    noise_power_ght = std(y_ght(setdiff(1:length(y_ght), true_indices)));
    metrics.snr_ght = 20 * log10(signal_power_ght / (noise_power_ght + eps));
    
    signal_power_gfrht = mean(y_gfrht(true_indices));
    noise_power_gfrht = std(y_gfrht(setdiff(1:length(y_gfrht), true_indices)));
    metrics.snr_gfrht = 20 * log10(signal_power_gfrht / (noise_power_gfrht + eps));
    
    % 2. 定位精度 (Precision@k) - 前k个最高值节点中真实异常的比例
    k = length(true_indices);
    [~, sorted_idx_ght] = sort(y_ght, 'descend');
    top_k_ght = sorted_idx_ght(1:k);
    metrics.precision_ght = sum(ismember(top_k_ght, true_indices)) / k;
    
    [~, sorted_idx_gfrht] = sort(y_gfrht, 'descend');
    top_k_gfrht = sorted_idx_gfrht(1:k);
    metrics.precision_gfrht = sum(ismember(top_k_gfrht, true_indices)) / k;
    
    % 3. 均方根误差 (RMSE) - 与理想二值信号的差异
    metrics.rmse_ght = sqrt(mean((y_ght - ground_truth).^2));
    metrics.rmse_gfrht = sqrt(mean((y_gfrht - ground_truth).^2));
end

function print_metrics(metrics)
    fprintf('信噪比 (SNR, dB):\n');
    fprintf('  GHT:   %.2f dB\n', metrics.snr_ght);
    fprintf('  GFRHT: %.2f dB\n', metrics.snr_gfrht);
    fprintf('  提升:  %.2f dB (%.1f%%)\n', ...
        metrics.snr_gfrht - metrics.snr_ght, ...
        (metrics.snr_gfrht - metrics.snr_ght) / abs(metrics.snr_ght) * 100);
    
    fprintf('\n定位精度 (Precision@%d):\n', length(18:23));
    fprintf('  GHT:   %.3f\n', metrics.precision_ght);
    fprintf('  GFRHT: %.3f\n', metrics.precision_gfrht);
    fprintf('  提升:  %.1f%%\n', (metrics.precision_gfrht - metrics.precision_ght) * 100);
    
    fprintf('\n均方根误差 (RMSE):\n');
    fprintf('  GHT:   %.4f\n', metrics.rmse_ght);
    fprintf('  GFRHT: %.4f\n', metrics.rmse_gfrht);
    fprintf('  降低:  %.1f%%\n', (1 - metrics.rmse_gfrht/metrics.rmse_ght) * 100);
end


function y = graph_hilbert_transform(A, x)
    [V, D] = eig(A);
    lambda = diag(D);
    n = length(lambda);
    H_diag = zeros(n, 1);
    
    for k = 1:n
        if imag(lambda(k)) > 0
            H_diag(k) = -1j;   % -j for positive imaginary part
        elseif imag(lambda(k)) < 0
            H_diag(k) = 1j;    % +j for negative imaginary part
        else
            H_diag(k) = 0;     % 0 for zero imaginary part
        end
    end
    
    H = diag(H_diag);
    % 计算 GHT: y = V * H * inv(V) * x
    y = real(V * H * inv(V) * x);
end


function y = graph_frhilbert_transform(A, x,alpha,beta)
    [V, D] = eig(A);
    lambda = diag(D);
    n = length(lambda);
    H_diag = zeros(n, 1);
    F = V;
    [P, Q] = eig(F);
    Q_alpha = Q^alpha;
    Falpha = P*Q_alpha*inv(P);

    for k = 1:n
        if imag(lambda(k)) > 0
            H_diag(k) = exp(-1j*beta);   % -j for positive imaginary part
        elseif imag(lambda(k)) < 0
            H_diag(k) =  exp(1j*beta);    % +j for negative imaginary part
        else
            H_diag(k) = (exp(-1j*beta) + exp(1j*beta))/2;     % 0 for zero imaginary part
        end
    end
    
    H = diag(H_diag);
    % 计算 GHT: y = V * H * inv(V) * x
    y = real(Falpha * H / Falpha * x);
end