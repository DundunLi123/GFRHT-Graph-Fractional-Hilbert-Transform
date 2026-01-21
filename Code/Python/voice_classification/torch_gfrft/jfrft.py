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
    num_nodes, num_time_steps = jtv_signal.size()
    gfrft_mtx = gfrft_obj.gfrft_mtx(gfrft_order)
    igfrft_mtx = gfrft_obj.igfrft_mtx(gfrft_order)
    frft_matrix = frft_mtx(num_time_steps, dfrft_order, approx_order=approx_order, device=device)
    ifrft_matrix = ifrft_mtx(num_time_steps, dfrft_order, approx_order=approx_order, device=device)
    joint_jfrt_mtx = torch.kron(frft_matrix, gfrft_mtx)
    joint_ijfrt_mtx = torch.kron(ifrft_matrix, igfrft_mtx)
    return joint_jfrt_mtx, joint_ijfrt_mtx

def generate_bandlimited_signal(
        jtv_signal: torch.Tensor,
        joint_jfrt_mtx: torch.Tensor,
        joint_ijfrt_mtx: torch.Tensor,
        stopband: slice
) -> torch.Tensor:
    jtv_signal_vec = jtv_signal.permute(1, 0).flatten().to(joint_jfrt_mtx.dtype)
    transformed = torch.matmul(joint_jfrt_mtx, jtv_signal_vec)
    transformed[stopband] = 0
    bandlimited_signal_vec = torch.matmul(joint_ijfrt_mtx, transformed)
    bandlimited_signal = bandlimited_signal_vec.view(jtv_signal.shape)
    return bandlimited_signal


def generate_bandlimited_noise(
        jtv_signal: torch.Tensor,
        joint_ijfrt_mtx: torch.Tensor,
        stopband: slice,
        mean: float = 0.0,
        sigma: float = 1.0
) -> torch.Tensor:
    noise = torch.zeros_like(jtv_signal, dtype=joint_ijfrt_mtx.dtype).flatten()  # 确保数据类型匹配
    torch.manual_seed(0)
    noise[stopband] = mean + sigma * torch.randn_like(noise[stopband])
    bandlimited_noise_vec = torch.matmul(joint_ijfrt_mtx, noise)
    bandlimited_noise = bandlimited_noise_vec.view(jtv_signal.shape)
    return bandlimited_noise


def generate_bandlimited_experiment_data(
    jtv_signal: torch.Tensor,
    joint_jfrt_mtx: torch.Tensor,
    joint_ijfrt_mtx: torch.Tensor,
    stopband_count: int,
    overlap: int = 0,
    mean: float = 0.0,
    sigma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    size = jtv_signal.numel()  # 使用展平的信号总大小
    signal_stopband = slice(size - stopband_count, size)
    noise_stopband = slice(size - stopband_count - overlap, size)
    bl_signal = generate_bandlimited_signal(jtv_signal, joint_jfrt_mtx, joint_ijfrt_mtx, signal_stopband)
    bl_noise = generate_bandlimited_noise(jtv_signal, joint_ijfrt_mtx, noise_stopband, mean, sigma)
    return bl_signal, bl_noise


def generate_bandlimited_signal2(
        jtv_signal: torch.Tensor,
        gfrft_mtx: torch.Tensor,
        igfrft_mtx: torch.Tensor,
        frft_mtx: torch.Tensor,
        ifrft_mtx: torch.Tensor,
        stop_row: int,
        stop_col: int
) -> torch.Tensor:
    transformed = torch.matmul(gfrft_mtx, jtv_signal)
    transformed = torch.matmul(transformed, frft_mtx.T)
    print(f"Transformed shape before zeroing: {transformed.shape}")
    transformed = transformed.clone()
    transformed[-stop_row:, :] = 0
    transformed[:, -stop_col:] = 0
    bandlimited_signal = torch.matmul(igfrft_mtx, transformed)
    bandlimited_signal = torch.matmul(bandlimited_signal, ifrft_mtx.T)
    return bandlimited_signal


def generate_bandlimited_noise2(
        jtv_signal: torch.Tensor,
        igfrft_mtx: torch.Tensor,
        ifrft_mtx: torch.Tensor,
        stop_row: int,
        stop_col: int,
        mean: float = 0.0,
        sigma: float = 1.0
) -> torch.Tensor:
    num_rows, num_cols = jtv_signal.shape
    noise = mean + sigma * torch.randn((num_rows, num_cols), dtype=torch.complex64)
    noise[:-stop_row, :-stop_col] = 0
    bandlimited_noise = torch.matmul(igfrft_mtx, noise)
    bandlimited_noise = torch.matmul(bandlimited_noise, ifrft_mtx.T)
    return bandlimited_noise


def generate_bandlimited_experiment_data2(
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
    # Generate bandlimited signal
    print("Generating bandlimited signal...")
    bl_signal = generate_bandlimited_signal2(
        jtv_signal=jtv_signal,
        gfrft_mtx=gfrft_mtx,
        igfrft_mtx=igfrft_mtx,
        frft_mtx=frft_mtx,
        ifrft_mtx=ifrft_mtx,
        stop_row=stop_row,
        stop_col=stop_col
    )

    # Generate bandlimited noise
    print("Generating bandlimited noise...")
    noise_stop_row = stop_row + overlap  # Expand stop_row by overlap
    noise_stop_col = stop_col + overlap  # Expand stop_col by overlap
    bl_noise = generate_bandlimited_noise2(
        jtv_signal=jtv_signal,
        igfrft_mtx=igfrft_mtx,
        ifrft_mtx=ifrft_mtx,
        stop_row=noise_stop_row,
        stop_col=noise_stop_col,
        mean=mean,
        sigma=sigma
    )

    return bl_signal, bl_noise













