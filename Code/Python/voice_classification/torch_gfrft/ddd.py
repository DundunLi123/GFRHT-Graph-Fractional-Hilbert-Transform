import torch
from torch_frft.dfrft_module import DFRFT
# 创建一个一维行向量
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 设定分数阶 a
a = 0.5

# 调用 DFRFT 变换
dfrft_result = DFRFT.dfrft(x, a)

print("DFRFT 变换结果：", dfrft_result)
