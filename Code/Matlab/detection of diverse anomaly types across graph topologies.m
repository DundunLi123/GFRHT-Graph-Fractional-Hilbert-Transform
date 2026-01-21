%% 双参数图分数阶希尔伯特变换(GFHT)仿真实验 - SCI配色 & 垂直GT标注
clc; clear; close all;
rng(61)  % 固定随机种子，保证可复现

% 启动实验
main_GFHT_simulation();

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


%% ======================= 主流程 =======================
function main_GFHT_simulation()
    % 实验配置
    alpha_range   = 0:0.1:2;                    % α 参数扫描范围
    beta_range    = 0:0.1:2;                    % β 参数扫描范围
    graph_types   = {'community','scale-free'}; % 可加 'small-world'
    anomaly_types = {'low-freq','high-freq','impulse'};
    
    % 初始化结果存储
    results = cell(length(graph_types), length(anomaly_types));
    
    % 遍历所有图类型和异常类型
    for g = 1:length(graph_types)
        fprintf('\n=== 测试图类型: %s ===\n', graph_types{g});
        A = generate_graph(graph_types{g}, 50); % 生成 50 节点图
        
        for a_type = 1:length(anomaly_types)
            fprintf('-- 异常类型: %s\n', anomaly_types{a_type});
            % 修改：获取信号和真实异常位置
            [x, true_anomaly] = generate_signal(A, anomaly_types{a_type});
            
            % 执行参数扫描，传递真实异常位置
            [opt_alpha, opt_beta, perf_gain, snr_matrix] = ...
                scan_parameters(A, x, true_anomaly, alpha_range, beta_range);
            
            % 存储结果，包括真实异常位置
            results{g, a_type} = struct( ...
                'graph_type',   graph_types{g}, ...
                'anomaly_type', anomaly_types{a_type}, ...
                'opt_alpha',    opt_alpha, ...
                'opt_beta',     opt_beta, ...
                'perf_gain',    perf_gain, ...
                'snr_matrix',   snr_matrix, ...
                'true_anomaly', true_anomaly); % 添加真实异常位置
            
            % 可视化当前结果，传递真实异常位置
            visualize_results(results{g, a_type}, alpha_range, beta_range);
        end
    end
    
    % 综合结果报告
    generate_summary_report(results);
end

%% ======================= 图结构生成 =======================
function A = generate_graph(type, N)
    switch type
        case 'community'
            % 5个社区：社区内强连接、社区间弱连接
            comm_size = floor(N/5);
            A = zeros(N);
            for i = 1:min(5, N)
                start_idx = (i-1)*comm_size + 1;
                end_idx   = min(i*comm_size, N);
                if start_idx > end_idx, continue; end
                idx = start_idx:end_idx;
                subgraph = rand(numel(idx)) > 0.7;      % 社区内连接概率 30%
                subgraph = subgraph - diag(diag(subgraph));
                A(idx, idx) = subgraph;
            end
            % 添加社区间连接 (概率 5%)
            for i = 1:N
                for j = i+1:N
                    if rand() < 0.05
                        A(i,j) = 0.2; A(j,i) = 0.2;
                    end
                end
            end
            
        case 'scale-free'
            % 无标度网络 (Barabási–Albert)
            A = zeros(N);
            m0 = min(3, N); m = min(2, N-1);
            if m0 > 1
                A(1:m0, 1:m0) = triu(ones(m0), 1);
                degrees = sum(A,1);
            else
                degrees = zeros(1,N);
            end
            for i = (m0+1):N
                if sum(degrees(1:i-1)) > 0
                    probs = degrees(1:i-1) / sum(degrees(1:i-1));
                else
                    probs = ones(1, i-1)/(i-1);
                end
                targets = [];
                for k = 1:m
                    if isempty(probs), break; end
                    cp = cumsum(probs); r = rand()*cp(end);
                    idx = find(cp >= r, 1);
                    targets = [targets, idx];
                end
                A(i, targets) = 1; A(targets, i) = 1;
                degrees(i) = m; degrees(targets) = degrees(targets) + 1;
            end
            
        case 'small-world'
            % 小世界网络 (Watts–Strogatz)
            k = min(4, N-1); p = 0.3;
            A = zeros(N);
            for i = 1:N
                neighbors = [max(1,i-k/2):i-1, i+1:min(N,i+k/2)];
                neighbors = setdiff(neighbors, i);
                A(i, neighbors) = 1;
            end
            for i = 1:N
                for j = i+1:N
                    if A(i,j) == 1 && rand() < p
                        A(i,j) = 0; A(j,i) = 0;
                        possible_new = setdiff(1:N, [i, j, find(A(i,:))]);
                        if ~isempty(possible_new)
                            new_j = possible_new(randi(length(possible_new)));
                            A(i,new_j) = 1; A(new_j,i) = 1;
                        end
                    end
                end
            end
    end
    
    % 归一化：最大特征值模为 1
    if N <= 100
        [~, D] = eig(A); eig_vals = diag(D);
        max_eig = max(abs(eig_vals));
    else
        max_eig = max(abs(eigs(A,1,'lm')));
    end
    if max_eig > 0
        A = A / max_eig;
    else
        A = A + eye(N)*0.1;
        max_eig = max(abs(eig(A)));
        A = A / max_eig;
    end
end

%% ======================= 异常信号生成 =======================
function [x, true_anomaly] = generate_signal(A, anomaly_type)
    N = size(A, 1);
    if N > 1
        [V, D] = eig(A);        % 全部特征分解
        lambda = diag(D);
    else
        V = 1; lambda = A;
    end
    x = zeros(N,1);
    true_anomaly = []; % 初始化真实异常位置

    switch anomaly_type
        case 'low-freq'
            [~, idx] = sort(abs(lambda), 'ascend');   % 低频
            x_smooth = real(V(:, idx(1))) * 3;
            true_anomaly = randsample(N, min(3, N)); % 保存真实异常位置
            x = x_smooth + 1.5*(1:N == true_anomaly(1))';
            for i = 2:length(true_anomaly)
                x(true_anomaly(i)) = x(true_anomaly(i)) + 1.5;
            end

        case 'high-freq'
            [~, idx] = sort(abs(lambda), 'descend');  % 高频
            x = real(V(:, idx(1))) + 0.3*real(V(:, idx(2)));
            true_anomaly = randsample(N, min(5, N)); % 保存真实异常位置
            for i = 1:length(true_anomaly)
                x(true_anomaly(i)) = x(true_anomaly(i)) + 1.5;
            end

        case 'impulse'
            true_anomaly = randsample(N, min(4, N)); % 保存真实异常位置
            x(true_anomaly) = 2;
    end
    x = x + 0.1*randn(N,1);     % 高斯噪声
end

%% ======================= GFHT 核心 =======================
function y = GFHT(x, A, alpha, beta)
    [V, D] = eig(A);
    lambda = diag(D);
    n = length(lambda);
    H_diag = zeros(n,1);

    F = V;                     % 以特征向量矩阵为"基"
    [P, Q] = eig(F);
    Q_alpha = Q^alpha;
    Falpha  = P * Q_alpha * inv(P);   % 与原实现一致

    for k = 1:n
        if imag(lambda(k)) > 0
            H_diag(k) = exp(-1j*beta*pi/2);
        elseif imag(lambda(k)) < 0
            H_diag(k) = exp( 1j*beta*pi/2);
        else
            H_diag(k) = (exp(-1j*beta*pi/2)+exp(1j*beta*pi/2))/2;
        end
    end
    H = diag(H_diag);

    % GFHT
    y = real(Falpha * H / Falpha * x);
end

%% ======================= 参数扫描与评估 =======================
function [opt_alpha, opt_beta, perf_gain, snr_matrix] = scan_parameters(A, x, true_anomaly, alpha_range, beta_range)
    % 使用传入的真实异常位置
    snr_matrix = zeros(length(alpha_range), length(beta_range));
    
    % 传统 GHT (α=1, β=1)
    base_xh       = GFHT(x, A, 1, 1);
    base_response = abs(base_xh);
    background    = base_response; 
    background(true_anomaly) = []; % 使用传入的真实异常位置
    
    if isempty(background) || std(background) < eps
        base_snr = 1;
    else
        base_snr = mean(base_response(true_anomaly)) / std(background); % 使用传入的真实异常位置
    end
    fprintf('传统 GHT 下的SNR: %.2f\n', base_snr);

    % 参数扫描
    fprintf('参数扫描进度: 00%%');
    total_iter = length(alpha_range)*length(beta_range);
    completed = 0;
    for i = 1:length(alpha_range)
        for j = 1:length(beta_range)
            try
                xh = GFHT(x, A, alpha_range(i), beta_range(j));
                response = abs(xh);
                anomaly_resp = mean(response(true_anomaly)); % 使用传入的真实异常位置
                background = response; 
                background(true_anomaly) = []; % 使用传入的真实异常位置
                
                if isempty(background) || std(background) < eps
                    snr = anomaly_resp;
                else
                    snr = anomaly_resp / std(background);
                end
                snr_matrix(i,j) = snr;
            catch ME
                warning('参数 α=%.1f, β=%.1f 计算失败: %s', ...
                    alpha_range(i), beta_range(j), ME.message);
                snr_matrix(i,j) = 0;
            end
            completed = completed + 1;
            if mod(completed, 10) == 0
                fprintf('\b\b\b%02d%%', round(100*completed/total_iter));
            end
        end
    end
    fprintf('\b\b\b100%%\n');
    
    % 最优参数
    [max_snr, idx] = max(snr_matrix(:));
    [ia, ib] = ind2sub(size(snr_matrix), idx);
    opt_alpha = alpha_range(ia);
    opt_beta  = beta_range(ib);
    fprintf('最优参数下的最大 SNR: %.2f\n', max_snr);

    % 相对提升
    perf_gain = (max_snr - base_snr) / max(eps, base_snr) * 100;
    fprintf('最优参数: α=%.1f, β=%.1f | SNR增益: %.1f%%\n', ...
        opt_alpha, opt_beta, perf_gain);
end

%% ======================= 可视化（仅检测响应 + 垂直GT） =======================
function visualize_results(result, alpha_range, beta_range) %#ok<INUSD>
    col = sci_colors();     % 调色盘（色盲友好）
    lw  = 2.0;              % 线宽
    fs  = 11;               % 字体

    fig = figure('Position', [100, 100, 450, 500], 'Color','w'); %#ok<NASGU>
    ax2 = axes; hold(ax2,'on'); box on; grid on;

    % 生成测试信号（使用相同的参数确保一致性）
    A_test = generate_graph(result.graph_type, 50);
    [x_test, true_anomaly_test] = generate_signal(A_test, result.anomaly_type);

    % 方法响应
    xh_base = GFHT(x_test, A_test, 1, 1);
    xh_opt  = GFHT(x_test, A_test, result.opt_alpha, result.opt_beta);

    % y 轴范围
    y_max = max([abs(xh_base(:)); abs(xh_opt(:)); 1]);

    % ===== Ground Truth：在异常索引处画"垂直线" =====
    h_gt = stem(true_anomaly_test, 1.05*y_max*ones(size(true_anomaly_test)), ...
        'LineStyle','-', 'Marker','none', 'Color', col.gray, 'LineWidth', 1.4, ...
        'DisplayName','Ground-truth Anomalies');
    try, h_gt.BaseLine.Visible = 'off'; end
    h_gt.Annotation.LegendInformation.IconDisplayStyle = 'off';

    % ===== 两条方法曲线（SCI配色） =====
    % 基线
    base_alpha_str   = frac_str_latex(1);
    base_beta_pi_str = frac_str_latex(1/2, true);   % π 在分子
    base_label = sprintf('GHT ($\\alpha=%s,\\;\\beta=%s$)', ...
        base_alpha_str, base_beta_pi_str);

    % 最优
    opt_alpha_str   = frac_str_latex(result.opt_alpha);
    opt_beta_pi_str = frac_str_latex(result.opt_beta/2, true); % π 在分子
    opt_label = sprintf('GFRHT ($\\alpha=%s,\\;\\beta=%s$)', ...
        opt_alpha_str, opt_beta_pi_str);

    p1 = plot(abs(xh_base), '-', 'Color', col.blue,      'LineWidth', lw, ...
        'DisplayName', base_label);
    p2 = plot(abs(xh_opt),  '-', 'Color', col.vermilion, 'LineWidth', lw, ...
        'DisplayName', opt_label);
    p3 = plot(nan, nan, '-', 'Color', col.gray, 'LineWidth', 1.4, ...
        'DisplayName','Ground-truth Anomalies'); %#ok<NASGU>

    xlabel('Node','FontSize',14,'FontWeight','bold');
    ylabel('Amplitude','FontSize',14,'FontWeight','bold');
    % title(sprintf('Detection response: %s / %s', ...
    %     result.graph_type, result.anomaly_type));
    ylim([0, 1.25*y_max]);

    % 图例（LaTeX）
    lg = legend([p1 p2], {base_label, opt_label}, 'Location','northeast');
    set(lg,'Box','off','Interpreter','latex');
        % ===== 图例（LaTeX） =====
    lg = legend([p1 p2 p3], ...
        {base_label, opt_label, 'Ground-truth Anomalies'}, ...
        'Location','northeast');
    set(lg,'Box','off','Interpreter','latex');

    set(gca,'FontSize',fs);

    % 保存（仅保存检测响应图）
    %saveas(gcf, sprintf('GFHT_%s_%s_response.png', result.graph_type, result.anomaly_type));
exportgraphics(fig, sprintf('GFRHT_%s_%s_response.png', result.graph_type, result.anomaly_type), ...
    'BackgroundColor','white','Resolution',300,'ContentType','image');
end

%% ======================= 小工具：分式 LaTeX (支持 π 在分子上) =======================
function s = frac_str_latex(x, with_pi)
    % x: 数值
    % with_pi: 是否把 π 放在分子里（true/false）

    tol = 1e-8;
    [n, d] = rat(x, tol);

    if nargin < 2
        with_pi = false;
    end

    if d == 1
        if with_pi
            % 整数倍 pi，例如 2*pi
            if n == 1
                s = '\pi';
            else
                s = sprintf('%d\\pi', n);
            end
        else
            s = sprintf('%d', n);
        end
    else
        if with_pi
            % 分数倍 pi，π 放在分子
            if n == 1
                s = sprintf('\\frac{\\pi}{%d}', d);
            else
                s = sprintf('\\frac{%d\\pi}{%d}', n, d);
            end
        else
            % 普通分数
            s = sprintf('\\frac{%d}{%d}', n, d);
        end
    end
end

%% ======================= 综合报告（颜色统一） =======================
function generate_summary_report(results)
    fprintf('\n\n=== 实验总结报告 ===\n');
    fprintf('图类型\t\t异常类型\t最优α\t最优β\tSNR增益\n');
    fprintf('------------------------------------------------\n');

    [num_g, num_a] = size(results);
    for g = 1:num_g
        for a = 1:num_a
            r = results{g,a};
            fprintf('%s\t%s\t\t%.1f\t%.1f\t+%.1f%%\n', ...
                r.graph_type, r.anomaly_type, r.opt_alpha, r.opt_beta, r.perf_gain);
        end
    end

    % 汇总 3D（统一 parula，避免 jet）
    gain_data = zeros(num_g, num_a);
    for g = 1:num_g
        for a = 1:num_a
            gain_data(g, a) = results{g,a}.perf_gain;
        end
    end

    fig = figure('Position', [100, 100, 800, 600], 'Color','w');
    bar3(gain_data);
    colormap(parula); colorbar;

    xticklabels = cellfun(@(x) x.graph_type, results(:,1), 'UniformOutput', false);
    yticklabels = cellfun(@(x) x.anomaly_type, results(1,:), 'UniformOutput', false);

    set(gca, 'XTick', 1:num_g, 'XTickLabel', xticklabels, ...
             'YTick', 1:num_a, 'YTickLabel', yticklabels, ...
             'FontSize', 11);
    xlabel('Graph type'); ylabel('Anomaly type'); zlabel('SNR gain (%)');
   % title('GFHT performance gain (summary)');
    box on; grid on;

    %saveas(fig, 'GFHT_performance_summary.png');
exportgraphics(fig, 'GFHT_performance_summary.png', ...
    'BackgroundColor','white','Resolution',300,'ContentType','image');
    
end

%% ======================= SCI调色盘 =======================
function col = sci_colors()
% Okabe–Ito / Wong 调色盘（色盲友好、打印友好）
col.blue      = [0 114 178]/255;   % #0072B2  -> GHT
col.vermilion = [213 94 0]/255;    % #D55E00  -> GFHT
col.green     = [0 158 115]/255;   % #009E73  -> 备用/标记
col.orange    = [230 159 0]/255;   % #E69F00
col.purple    = [204 121 167]/255; % #CC79A7
col.yellow    = [240 228 66]/255;  % #F0E442
col.black     = [0 0 0];
col.gray      = [85 85 85]/255;    % Ground Truth 垂直线
end