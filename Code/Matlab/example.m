%% 图分数阶希尔伯特变换多参数实例分析（SCI风格）

% ---- 全局风格（SCI友好） ----
tnr = 'Times New Roman';
set(groot,'DefaultAxesFontName',tnr);
set(groot,'DefaultTextFontName',tnr);
set(groot,'DefaultLegendFontName',tnr);
set(groot,'DefaultAxesFontSize',12);
set(groot,'DefaultAxesLineWidth',1.2);
set(groot,'DefaultFigureColor','w');
set(groot,'DefaultFigureRenderer','painters');  % 矢量导出更清晰
set(groot,'DefaultAxesTickDir','out');

% Okabe–Ito / Wong 调色盘（选6色，避免黄色）
colors = [ ...
    0   114 178;   % Blue
    213 94  0;     % Vermillion
    0   158 115;   % BluishGreen
    230 159 0;     % Orange
    204 121 167;   % Purple
    86  180 233];  % SkyBlue
colors = colors/255;

%% 数据与参数
A = [0 1 1 0 1;
     0 0 0 1 0;
     0 0 0 1 0;
     0 0 0 0 1;
     1 0 0 0 0];

x = [0.8; 0.3; 0.5; 0.2; 0.6];
node_names = {'Alice', 'Bob', 'Charlie', 'David', 'Eve'};

alpha_values = 0:0.5:1;
beta_values  = 0:0.5:1;
beta_labels  = arrayfun(@(b) beta_as_pi_over_2(b), beta_values, 'UniformOutput', false);

% 预计算特征分解
[V, D] = eig(A);

%% 网格子图：不同 (alpha, beta)
figure('Position', [100, 100, 1200, 900]);

plot_index = 1;
for alpha = alpha_values
    for beta = beta_values
        % 计算
        y = graph_frhilbert_transform(A, x, alpha, beta);

        % 子图
        subplot(length(alpha_values), length(beta_values), plot_index); hold on;

        % 原始 vs 变换（SCI风格：统一线宽/标记/配色）
        plot(1:length(x), x, '-o', ...
            'Color', colors(1,:), 'LineWidth', 1.8, ...
            'MarkerSize', 4.5, 'MarkerFaceColor', colors(1,:));
        plot(1:length(x), y, '--s', ...
            'Color', colors(2,:), 'LineWidth', 1.8, ...
            'MarkerSize', 4.5, 'MarkerFaceColor', colors(2,:));

        % % 标题（只用 LaTeX 显示 β）
        % title(sprintf('$\\alpha=%.1f,\\ \\beta=%s$', alpha, beta_as_pi_over_2(beta)), ...
        %       'Interpreter','latex');


        title(sprintf('$\\alpha=%s,\\ \\beta=%s$', ...
      alpha_as_fraction(alpha), beta_as_pi_over_2(beta)), ...
      'Interpreter','latex');


        % 轴与刻度
        if plot_index > (length(alpha_values)-1)*length(beta_values)
            xlabel('Node','FontSize',14,'FontWeight','bold'); 
            set(gca,'XTick',1:length(x),'XTickLabel',node_names);
        else
            set(gca,'XTick',1:length(x),'XTickLabel',[]);
        end
        if mod(plot_index-1, length(beta_values)) == 0
            ylabel('Amplitude','FontSize',14,'FontWeight','bold');
        end
        ylim([0 1]); grid on; box on; axis tight;

        % % 只在第一个子图放图例
        % if plot_index == 1
        %     lg = legend('Original Signal','GFRHT','Location','northeast','Orientation','horizontal');
        %     set(lg,'Box','off');
        % end
        % —— 每个子图都放图例（右上角）——
        lg = legend( ...
    {'Original Signal', ...
     'GFRHT'}, ...
    'Location','northeast', ...     % 右上角
    'Orientation','vertical', ...   % 竖排
    'AutoUpdate','off');
        set(lg,'Box','off');   % LaTeX 显示 α/β，带边框

        plot_index = plot_index + 1;
    end
end
%sgtitle('图分数阶希尔伯特变换 - 不同\alpha和\beta参数的影响');  % 保持中文，默认 tex

%% 参数敏感性热图
figure('Position', [100, 100, 1000, 420]);

% 计算所有参数组合的结果
results = zeros(length(alpha_values), length(beta_values), length(x));
for i = 1:length(alpha_values)
    for j = 1:length(beta_values)
        results(i,j,:) = graph_frhilbert_transform(A, x, alpha_values(i), beta_values(j));
    end
end

% 选择节点的响应
node_idx = 1;  % Alice
sensitivity_data = squeeze(results(:,:,node_idx));

% 热图1：节点响应
subplot(1,2,1);
imagesc(sensitivity_data); colormap(parula); colorbar;
title(sprintf('节点 %s 的信号值随参数变化', node_names{node_idx}));
xlabel('\beta 值'); ylabel('\alpha 值');
set(gca,'XTick',1:length(beta_values), 'XTickLabel', beta_labels, 'TickLabelInterpreter','latex');
set(gca,'YTick',1:length(alpha_values), 'YTickLabel', arrayfun(@num2str,alpha_values,'uni',0));

% 热图2：范数差
norm_changes = zeros(length(alpha_values), length(beta_values));
for i = 1:length(alpha_values)
    for j = 1:length(beta_values)
        y = squeeze(results(i,j,:));
        norm_changes(i,j) = norm(y - x);
    end
end

subplot(1,2,2);
imagesc(norm_changes); colormap(parula); colorbar;
title('变换前后信号范数差异');
xlabel('\beta 值'); ylabel('\alpha 值');
set(gca,'XTick',1:length(beta_values), 'XTickLabel', beta_labels, 'TickLabelInterpreter','latex');
set(gca,'YTick',1:length(alpha_values), 'YTickLabel', arrayfun(@num2str,alpha_values,'uni',0));

sgtitle('参数敏感性分析');  % 保持中文，默认 tex



function y = graph_frhilbert_transform(A, x, alpha, beta)
    [V, D] = eig(A);
    F = V;
    [P, Q] = eig(F);
    Q_alpha = Q^alpha;
    Falpha = P * Q_alpha / P;

    lambda = diag(D);
    n = length(lambda);
    H_diag = zeros(n,1);
    for k = 1:n
        if imag(lambda(k)) > 0
            H_diag(k) = exp(-1j * beta*pi/2);
        elseif imag(lambda(k)) < 0
            H_diag(k) = exp( 1j * beta*pi/2);
        else
            H_diag(k) = (exp(-1j * beta*pi/2) + exp(1j * beta*pi/2))/2;
        end
    end
    H = diag(H_diag);
    y = real(Falpha * H / Falpha * x);
end

% function s = beta_as_pi_over_2(b)
%     % 显示成 “数值*\frac{\pi}{2}” 的 LaTeX 形式
%     s = sprintf('%.3g*\\frac{\\pi}{2}', b);
% end
function s = beta_as_pi_over_2(b)
    % 把 beta 按更简洁的 LaTeX 形式显示
    if b == 0
        s = '0';
    elseif b == 0.5
        s = '\frac{\pi}{4}';
    elseif b == 1
        s = '\frac{\pi}{2}';
    else
        % 兜底情况
        s = sprintf('%.3g*\\tfrac{\\pi}{2}', b);
    end
end

function s = alpha_as_fraction(a)
    if a == 0
        s = '0';
    elseif a == 0.5
        s = '\frac{1}{2}';
    elseif a == 1
        s = '1';
    else
        s = sprintf('%.2f', a);
    end
end


%% ---- 导出（按需取消注释）----
% exportgraphics(gcf, 'Fig2_heatmaps.pdf', 'ContentType','vector');
% print(gcf, 'Fig2_heatmaps.tif', '-dtiff', '-r600');   % 600 dpi
