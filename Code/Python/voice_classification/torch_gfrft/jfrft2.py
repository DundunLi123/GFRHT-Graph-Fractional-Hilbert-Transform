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
    # 生成联合变换矩阵
    gfrft_mtx = gfrft_obj.gfrft_mtx(gfrft_order)
    igfrft_mtx = gfrft_obj.igfrft_mtx(gfrft_order)
    frft_matrix = frft_mtx(6, dfrft_order, approx_order=approx_order, device=device)
    ifrft_matrix = ifrft_mtx(6, dfrft_order, approx_order=approx_order, device=device)
    joint_jfrt_mtx = torch.kron(frft_matrix, gfrft_mtx)
    joint_ijfrt_mtx = torch.kron(ifrft_matrix, igfrft_mtx)
    return joint_jfrt_mtx, joint_ijfrt_mtx
def generate_bandlimited_signal2(
        jtv_signal: torch.Tensor,
        gfrft_mtx: torch.Tensor,
        igfrft_mtx: torch.Tensor,
        full_frft_mtx: torch.Tensor,
        full_ifrft_mtx: torch.Tensor,
        stop_row: int,
        stop_col: int
) -> torch.Tensor:
    jtv_signal = jtv_signal.to(torch.complex64)
    gfrft_mtx = gfrft_mtx.to(torch.complex64)
    igfrft_mtx = igfrft_mtx.to(torch.complex64)
    full_frft_mtx = full_frft_mtx.to(torch.complex64)
    full_ifrft_mtx = full_ifrft_mtx.to(torch.complex64)

    transformed = torch.matmul(gfrft_mtx, jtv_signal)
    transformed = torch.matmul(transformed, full_frft_mtx.T)

    # 设置带限
    transformed[-stop_row:, :] = 0
    transformed[:, -stop_col:] = 0

    # 回到时域
    bandlimited_signal = torch.matmul(igfrft_mtx, transformed)
    bandlimited_signal = torch.matmul(bandlimited_signal, full_ifrft_mtx.T)
    return bandlimited_signal


def generate_bandlimited_noise2(
        jtv_signal: torch.Tensor,
        igfrft_mtx: torch.Tensor,
        full_ifrft_mtx: torch.Tensor,
        stop_row: int,
        stop_col: int,
        mean: float = 0.0,
        sigma: float = 1.0
) -> torch.Tensor:
    jtv_signal = jtv_signal.to(torch.complex64)
    igfrft_mtx = igfrft_mtx.to(torch.complex64)
    full_ifrft_mtx = full_ifrft_mtx.to(torch.complex64)

    noise = mean + sigma * torch.randn_like(jtv_signal, dtype=torch.complex64)
    noise[:-stop_row, :-stop_col] = 0

    # 回到时域
    bandlimited_noise = torch.matmul(igfrft_mtx, noise)
    bandlimited_noise = torch.matmul(bandlimited_noise, full_ifrft_mtx.T)
    return bandlimited_noise


def generate_bandlimited_experiment_data3(
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
    jtv_signal1 = jtv_signal[:, :6]
    jtv_signal2 = jtv_signal[:, 6:]

    bl_signal1 = generate_bandlimited_signal2(
        jtv_signal=jtv_signal1,
        gfrft_mtx=gfrft_mtx,
        igfrft_mtx=igfrft_mtx,
        full_frft_mtx=frft_mtx,
        full_ifrft_mtx=ifrft_mtx,
        stop_row=stop_row,
        stop_col=stop_col
    )

    bl_signal2 = generate_bandlimited_signal2(
        jtv_signal=jtv_signal2,
        gfrft_mtx=gfrft_mtx,
        igfrft_mtx=igfrft_mtx,
        full_frft_mtx=frft_mtx,
        full_ifrft_mtx=ifrft_mtx,
        stop_row=stop_row,
        stop_col=stop_col
    )

    noise_stop_row = stop_row + overlap
    noise_stop_col = stop_col + overlap

    bl_noise1 = generate_bandlimited_noise2(
        jtv_signal=jtv_signal1,
        igfrft_mtx=igfrft_mtx,
        full_ifrft_mtx=ifrft_mtx,
        stop_row=noise_stop_row,
        stop_col=noise_stop_col,
        mean=mean,
        sigma=sigma
    )

    bl_noise2 = generate_bandlimited_noise2(
        jtv_signal=jtv_signal2,
        igfrft_mtx=igfrft_mtx,
        full_ifrft_mtx=ifrft_mtx,
        stop_row=noise_stop_row,
        stop_col=noise_stop_col,
        mean=mean,
        sigma=sigma
    )

    bl_signal = torch.cat([bl_signal1, bl_signal2], dim=1)
    bl_noise = torch.cat([bl_noise1, bl_noise2], dim=1)

    return bl_signal, bl_noise