%% 1. 2D图像边缘检测（图6示例）
clear; 
clc;

p = XX;
alpha = XX;
beta = p*pi/2;

load('xh_img3.mat');   % 加载变量 xh_img
img = xh_img;
x = img(:);
n = 40;
C = diag(ones(n-1,1), 1); C(end,1) = 1;
A = kron(C, C);
img_size = size(img);

% 计算图希尔伯特变换
[V, D] = eig(A);          % 特征分解
eigvals = diag(D);

% 按相位角排序特征值
[~, idx] = sort(angle(eigvals));
V = V(:, idx);
eigvals = eigvals(idx);

% 划分特征值集合 (Γ1, Γ2, Γ3, Γ4)
tol = 1e-10;
phase = angle(eigvals);
gamma1 = find(abs(phase) < tol & real(eigvals) > 0);
gamma3 = find(abs(phase - pi) < tol | abs(phase + pi) < tol);
gamma2 = find(phase > tol & phase < pi - tol);
gamma4 = setdiff(1:numel(eigvals), [gamma1; gamma2; gamma3]);

% 构建频域变换矩阵 j_h
j_h = zeros(numel(eigvals), 1);
j_h(gamma2) = exp(-1j*beta);  % Γ2: 乘子 exp(-1j*beta)
j_h(gamma4) = exp(1j*beta);  % Γ4: 乘子 exp(1j*beta)
j_h(gamma1) = (exp(-1j*beta) + exp(1j*beta))/2;   % Γ1: 乘子 (exp(-1j*beta) + exp(1j*beta))/2
j_h(gamma3) = (exp(-1j*beta) + exp(1j*beta))/2;   % Γ3: 乘子 (exp(-1j*beta) + exp(1j*beta))/2

j_h = diag(j_h);

F = V;
[P, Q] = eig(F);
Q_alpha = Q^alpha;
Falpha = P*Q_alpha*inv(P);

% 计算图分数阶希尔伯特变换
x_hat1 = Falpha \ real(x);       % 图傅里叶变换
xh_hat1 = (j_h) * x_hat1; % 频域变换
xg = Falpha * xh_hat1;     % 逆变换
x_img = reshape(real(xg), size(img)); % 转为图像格式

%% --------- 图希尔伯特变换函数（根据正式定义）----------
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
    y = real(V * H / V * x);
end
x_h = graph_hilbert_transform(A,real(x));

x_h_img = reshape(real(x_h), size(img)); 

% 可视化结果
figure;
subplot(1,3,1), imshow((img), []), title('构造的原图');
subplot(1,3,2), imshow((x_img), []), title('GFRHT');
subplot(1,3,3), imshow(((x_h_img)), []), title('GHT');

gfrht_binary = imbinarize(x_img);
ght_binary = imbinarize(x_h_img);

%%  计算信息熵
entropy_val_gfrht = entropy(gfrht_binary);
entropy_val_ght = entropy(ght_binary);
% 打印结果
fprintf('Entropy of GFRHT binary image: %.4f\n', entropy_val_gfrht);
fprintf('Entropy of GHT binary image: %.4f\n', entropy_val_ght);


%%  计算SSIM
[ssim_val_gfrht, ~] = ssim(x_img, img);
[ssim_val_ght, ~] = ssim(x_h_img, img);
% 打印结果
fprintf('SSIM of GFRHT image: %.4f\n', ssim_val_gfrht);
fprintf('SSIM of GHT image: %.4f\n', ssim_val_ght);

%%  计算边缘密度ED
binary_edge_gfrht = imbinarize(x_img);
binary_edge_ght = imbinarize(x_h_img);
edge_density_gfrht = sum(binary_edge_gfrht(:)) / numel(binary_edge_gfrht);
edge_density_ght = sum(binary_edge_ght(:)) / numel(binary_edge_ght);
% 打印结果
fprintf('Edge Density of GFRHT image: %.4f\n', edge_density_gfrht);
fprintf('Edge Density of GHT image: %.4f\n', edge_density_ght);



%%  计算方差
var_val_gfrht = var(double(gfrht_binary(:)));
var_val_ght = var(double(ght_binary(:)));
% 打印结果
fprintf('VAR of GFRHT binary image: %.4f\n', var_val_gfrht);
fprintf('VAR of GHT binary image: %.4f\n', var_val_ght);

evaluate_edge_noGT(x_img,img);

evaluate_edge_noGT(x_h_img,img);