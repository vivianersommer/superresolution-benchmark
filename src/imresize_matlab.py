"""Reimplementação do imresize bicúbico do MATLAB (com antialias na redução).

Segue a lógica de `imresize.m` (MATLAB R2013+): núcleo cúbico de Keys com a = -0,5,
núcleo alargado por 1/escala na redução (antialias), borda por reflexão simétrica e
redimensionamento separável, uma dimensão por vez, começando pela de menor escala.
Conferida contra as LR_bicubic do pacote de benchmarks do EDSR (gerado em MATLAB) em
tests/test_fase2.py.
"""
import math

import numpy as np

KERNEL_WIDTH = 4.0


def _cubic(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    ax2, ax3 = ax ** 2, ax ** 3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
            + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def _contributions(in_len: int, out_len: int, scale: float, antialias: bool):
    if scale < 1 and antialias:
        kernel = lambda x: scale * _cubic(scale * x)  # noqa: E731
        width = KERNEL_WIDTH / scale
    else:
        kernel, width = _cubic, KERNEL_WIDTH
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)          # posição na entrada (base 1)
    left = np.floor(u - width / 2)
    p = int(math.ceil(width)) + 2
    idx = left[:, None] + np.arange(p)[None, :]    # base 1
    w = kernel(u[:, None] - idx)
    w = w / w.sum(axis=1, keepdims=True)
    # Reflexão simétrica nas bordas (como o MATLAB).
    aux = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    idx = aux[np.mod(idx.astype(np.int64) - 1, aux.size)]
    keep = np.any(w != 0, axis=0)
    return w[:, keep], idx[:, keep]


def _resize_along(img: np.ndarray, axis: int, weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    moved = np.moveaxis(img, axis, 0)               # (in_len, ...)
    gathered = moved[indices]                       # (out_len, p, ...)
    w = weights.reshape(weights.shape + (1,) * (moved.ndim - 1))
    out = (gathered * w).sum(axis=1)
    return np.moveaxis(out, 0, axis)


def imresize(img: np.ndarray, scale: float, antialias: bool = True) -> np.ndarray:
    """Redimensiona uma imagem uint8 (H, W) ou (H, W, C) pelo fator `scale`.

    O tamanho de saída é ceil(scale * tamanho), como no MATLAB. Os cálculos são feitos
    em float64. Como no MATLAB com entrada uint8, o resultado de cada passe é
    arredondado (meia unidade para longe do zero) e saturado em [0, 255].
    """
    if img.dtype != np.uint8:
        raise TypeError(f"imresize espera uint8, recebeu {img.dtype}")
    out = img
    h, w = img.shape[:2]
    out_shape = (int(math.ceil(h * scale)), int(math.ceil(w * scale)))
    # Mesma escala nas duas dimensões: o MATLAB processa a dimensão 1 (linhas) primeiro.
    for axis in (0, 1):
        weights, indices = _contributions(img.shape[axis], out_shape[axis], scale, antialias)
        res = _resize_along(out.astype(np.float64), axis, weights, indices)
        out = np.clip(np.floor(res + 0.5), 0, 255).astype(np.uint8)
    return out
