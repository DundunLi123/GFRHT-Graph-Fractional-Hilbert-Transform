import torch
from torch_frft.dfrft import frft_mtx, ifrft_mtx

def get_joint_jfrt_pair(
        gfrft_obj,
        jtv_signal: torch.Tensor,
        gfrft_order: float,
        dfrft_order: float,
        approx_order: int = 2,
        device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor]:
    gfrft_mtx = gfrft_obj.gfrft_mtx(gfrft_order)
    igfrft_mtx = gfrft_obj.igfrft_mtx(gfrft_order)
    frft_matrix = frft_mtx(6, dfrft_order, approx_order=approx_order, device=device)
    ifrft_matrix = ifrft_mtx(6, dfrft_order, approx_order=approx_order, device=device)
    joint_jfrt_mtx = torch.kron(frft_matrix, gfrft_mtx)
    joint_ijfrt_mtx = torch.kron(ifrft_matrix, igfrft_mtx)
    return joint_jfrt_mtx, joint_ijfrt_mtx


def generate_bandlimited_signal_block(
        block_signal: torch.Tensor,
        gfrft_mtx: torch.Tensor,
        igfrft_mtx: torch.Tensor,
        full_frft_mtx: torch.Tensor,
        full_ifrft_mtx: torch.Tensor,
        stop_row: int,
        stop_col: int
) -> torch.Tensor:
    block_signal = block_signal.to(torch.complex64)
    gfrft_mtx = gfrft_mtx.to(torch.complex64)
    igfrft_mtx = igfrft_mtx.to(torch.complex64)
    full_frft_mtx = full_frft_mtx.to(torch.complex64)
    full_ifrft_mtx = full_ifrft_mtx.to(torch.complex64)

    transformed = torch.matmul(gfrft_mtx, block_signal)
    transformed = torch.matmul(transformed, full_frft_mtx.T)

    # 设置带限
    transformed[-stop_row:, :] = 0
    transformed[:, -stop_col:] = 0

    # 回到时域
    bandlimited_signal = torch.matmul(igfrft_mtx, transformed)
    bandlimited_signal = torch.matmul(bandlimited_signal, full_ifrft_mtx.T)
    return bandlimited_signal


def generate_bandlimited_noise_block(
        block_signal: torch.Tensor,
        igfrft_mtx: torch.Tensor,
        full_ifrft_mtx: torch.Tensor,
        stop_row: int,
        stop_col: int,
        mean: float = 0.0,
        sigma: float = 1.0
) -> torch.Tensor:
    block_signal = block_signal.to(torch.complex64)
    igfrft_mtx = igfrft_mtx.to(torch.complex64)
    full_ifrft_mtx = full_ifrft_mtx.to(torch.complex64)

    noise = mean + sigma * torch.randn_like(block_signal, dtype=torch.complex64)
    noise[:-stop_row, :-stop_col] = 0

    # 回到时域
    bandlimited_noise = torch.matmul(igfrft_mtx, noise)
    bandlimited_noise = torch.matmul(bandlimited_noise, full_ifrft_mtx.T)
    return bandlimited_noise


def generate_bandlimited_experiment_data6(
        jtv_signal: torch.Tensor,
        gfrft_mtx: torch.Tensor,
        igfrft_mtx: torch.Tensor,
        frft_mtx: torch.Tensor,
        ifrft_mtx: torch.Tensor,
        stop_row: int,
        stop_col: int,
        overlap: int = 0,
        mean: float = 0.0,
        sigma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    # 分成六块，每块 \(6 \times 6\)
    blocks = [jtv_signal[:, i * 6:(i + 1) * 6] for i in range(6)]

    bl_signals = []
    bl_noises = []

    for block in blocks:
        # 生成带限信号
        bl_signal = generate_bandlimited_signal_block(
            block_signal=block,
            gfrft_mtx=gfrft_mtx,
            igfrft_mtx=igfrft_mtx,
            full_frft_mtx=frft_mtx,
            full_ifrft_mtx=ifrft_mtx,
            stop_row=stop_row,
            stop_col=stop_col
        )
        bl_signals.append(bl_signal)

        # 生成带限噪声
        bl_noise = generate_bandlimited_noise_block(
            block_signal=block,
            igfrft_mtx=igfrft_mtx,
            full_ifrft_mtx=ifrft_mtx,
            stop_row=stop_row + overlap,
            stop_col=stop_col + overlap,
            mean=mean,
            sigma=sigma
        )
        bl_noises.append(bl_noise)

    # 合并所有块
    bl_signal = torch.cat(bl_signals, dim=1)
    bl_noise = torch.cat(bl_noises, dim=1)

    return bl_signal, bl_noise
