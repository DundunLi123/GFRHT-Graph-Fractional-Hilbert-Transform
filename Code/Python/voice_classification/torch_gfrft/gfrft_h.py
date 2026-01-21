# gfrft_h.py
import torch as th
from torch_gfrft.utils import get_matvec_tensor_einsum_str, is_hermitian
from typing import Union

class GFRFT:
    def __init__(self, gft_mtx: th.Tensor, igft_mtx: th.Tensor) -> None:
        # 确保 gft_mtx 和 igft_mtx 为复数类型
        gft_mtx = gft_mtx.to(dtype=th.cfloat)
        igft_mtx = igft_mtx.to(dtype=th.cfloat)

        if is_hermitian(gft_mtx):
            self._eigvals, self._eigvecs = th.linalg.eigh(gft_mtx)
            self._inv_eigvecs = self._eigvecs.H
        else:
            self._eigvals, self._eigvecs = th.linalg.eig(gft_mtx)
            self._inv_eigvecs = th.linalg.inv(self._eigvecs)

        # 保存 gft_mtx 和 igft_mtx
        self.gft_mtx = gft_mtx
        self.igft_mtx = igft_mtx

    def gfrft_mtx(self, a: float | th.Tensor) -> th.Tensor:
        mtx_size = self.gft_mtx.shape[0]
        half_identity = 0.5 * th.eye(mtx_size, dtype=self.gft_mtx.dtype, device=self.gft_mtx.device)
        logm_gft_torch = self.logm_approx(self.gft_mtx)
        dg_square_mtx = ((1j * 2 / th.pi) * logm_gft_torch + half_identity) / (2 * th.pi)

        t_mtx = th.pi * (dg_square_mtx + self.gft_mtx @ dg_square_mtx @ self.igft_mtx) - half_identity
        gfrft_mtx = th.matrix_exp(-(1j * a * th.pi / 2) * t_mtx)

        return gfrft_mtx
    def igfrft_mtx(self, a: float | th.Tensor) -> th.Tensor:
        return self.gfrft_mtx(-a)
    def logm_approx(self, gft_mtx: th.Tensor) -> th.Tensor:
        # 使用特征分解 A = U * diag(lambda) * U^{-1}
        if th.allclose(gft_mtx, gft_mtx.H):
            eigvals, eigvecs = th.linalg.eigh(gft_mtx)
            inv_eigvecs = eigvecs.H
        else:
            eigvals, eigvecs = th.linalg.eig(gft_mtx)
            inv_eigvecs = th.linalg.inv(eigvecs)
        log_eigvals = th.log(eigvals)  # 逐元素对特征值取对数
        logm1 = th.einsum("ij,j,jk->ik", eigvecs, log_eigvals, inv_eigvecs)

        return logm1

    def gfrft(self, x: th.Tensor, a: Union[float, th.Tensor], *, dim: int = -1) -> th.Tensor:
        gfrft_mtx = self.gfrft_mtx(a)
        dtype = th.promote_types(gfrft_mtx.dtype, x.dtype)
        return th.einsum(
            get_matvec_tensor_einsum_str(len(x.shape), dim),
            gfrft_mtx.type(dtype),
            x.type(dtype),
        )

    def igfrft(self, x: th.Tensor, a: Union[float, th.Tensor], *, dim: int = -1) -> th.Tensor:
        return self.gfrft(x, -a, dim=dim)
