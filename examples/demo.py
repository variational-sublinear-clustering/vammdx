# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
from itertools import product
import imageio.v2 as imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from vammdx import DenoisingGaussian

def eval_psnr_ssim(
    gt: np.ndarray,
    img: np.ndarray,
    data_range: int = 255,
):
    psnr = peak_signal_noise_ratio(gt, img, data_range=data_range)
    ssim = structural_similarity(
        gt,
        img,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        data_range=data_range,
    )
    return psnr, ssim


def run(cov_type, shared, noisy, clean, C, H, patch_shape, seed):
    rng = np.random.default_rng(seed)

    model = DenoisingGaussian(
        C=C,
        patch_shape=patch_shape,
        color=False,
        H=H,  # H is only needed for "mfa"
        covariance_type=cov_type,
        shared=shared,
    )
    _shared = "with shared covariance matrices" if shared else ""
    print(f"Denoise with {cov_type} {_shared} ...")

    denoised_image = model.fit_denoise(noisy_image=noisy, rng=rng)
    psnr, ssim = eval_psnr_ssim(clean, denoised_image)

    print(f"denoised image: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")


if __name__ == "__main__":
    cov_types = ["isotropic", "diagonal", "mfa"] # "full"
    shared_list = [True, False]

    C = 1000
    H = 5
    patch_shape = (12, 12)
    noise_level = 15
    seed = 123

    rng = np.random.default_rng(seed)

    clean = imageio.imread("./data/house.png").astype(np.float64)
    noisy = clean + rng.normal(loc=0.0, scale=noise_level, size=clean.shape)

    psnr, ssim = eval_psnr_ssim(clean, noisy)
    print(f"noisy image: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")

    for cov_type, shared in product(cov_types, shared_list):
        obj = run(cov_type, shared, noisy, clean, C, H, patch_shape, seed)
