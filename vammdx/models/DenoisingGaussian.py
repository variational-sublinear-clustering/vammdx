# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from math import prod

from vammdx.cpp import (
    EM,
    DenoisingDiagonal as cppDenoisingDiagonal,
    DenoisingMFA as cppDenoisingMFA,
    DenoisingFull as cppDenoisingFull,
)
from vamm import Gaussian
from imageutils.prepost import OverlappingPatches, MultipleImages


class DenoisingGaussian(Gaussian):
    """
    Gaussian Mixture Model.

    Parameters
    ----------
    C : int
        Number of components.
    covariance_type : {"isotropic", "diagonaltied", "diagonal", "mfatied", "mfa"}
        Type of covariance matrix.
    patch_shape: TODO
    color: TODO
    shared: TODO
    H : int
        Dimensionality of the factors. (Only used for "mfa".)
    flat_prior : bool, optional
        Whether to use a flat prior for the mixture components. Defaults to False.
    init_prior : np.ndarray or None, optional
        Initial values for the priors of the mixture components. Defaults to None.
    init_means : np.ndarray or None, optional
        Initial values for the means of the mixture components. Defaults to None.
    init_A : np.ndarray or None, optional
        Initial values for the factor loading matrix A. Defaults to None. (Only used for "mfa" or "mfatied".)
    init_variance : np.ndarray or None, optional
        Initial values for the variance of the mixture components. Defaults to None.
    reg_covar : float, optional
        Regularization strength for the covariance matrix. Defaults to 0.0.

    Attributes
    ----------
    prior : Tensor
        priors of the mixture components
    means : Tensor
        mean values of the mixture components
    variance : Tensor
        variances of the mixture components
    """

    def __init__(
        self,
        C: int,
        covariance_type: str,
        patch_shape: list | tuple = (12, 12),
        color: bool = False,
        shared: bool = False,
        H: int | None = 5,
        flat_prior: bool = False,
        init_prior: npt.NDArray | str = "flat",
        init_means: npt.NDArray | str = "afkmc2",
        init_A: npt.NDArray | str = "uniform",
        init_variance: npt.NDArray | str = "data_variance",
        reg_covar: float = 1e-3,
    ) -> None:
        D = prod(patch_shape)
        D *= 3 if color else 1
        super().__init__(
            C=C,
            D=D,
            covariance_type=covariance_type,
            shared=shared,
            H=H,
            flat_prior=flat_prior,
            init_prior=init_prior,
            init_means=init_means,
            init_A=init_A,
            init_variance=init_variance,
            reg_covar=reg_covar,
        )
        self.denoising_model = {
            "full": cppDenoisingFull,
            "mfa": cppDenoisingMFA,
            "diagonal": cppDenoisingDiagonal,
            "isotropic": cppDenoisingDiagonal,
        }[covariance_type](self._cpp)

        self.denoising_em = None
        self.patch_shape = list(patch_shape)
        self.color = color

    def estimate(self, X: npt.NDArray):
        X_reco = np.zeros_like(X)
        self.denoising_em = (
            EM(self.em) if self.denoising_em is None else self.denoising_em
        )
        self.denoising_em.estimate(X, X_reco, self.denoising_model)
        return X_reco

    def fit_denoise(
        self,
        noisy_image: list[np.ndarray] | np.ndarray,
        limit: list[int] | int | None = None,
        eps: list[float] | float = 1.0e-4,
        C_prime: int = 3,
        G: int = 15,
        E: int = 1,
        hard: bool = False,
        sim_measure: str = "KL",
        indices: npt.NDArray | None = None,
        use_pretrainer: bool = False,
        merge_method="median",
        rng: np.random.generator | int | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Image must be HxWxC [0..255] float64
        """
        assert isinstance(noisy_image, list) or isinstance(noisy_image, np.ndarray)
        noisy_image = (
            noisy_image[0]
            if isinstance(noisy_image, list) and len(noisy_image) == 1
            else noisy_image
        )
        ovp = MultipleImages if isinstance(noisy_image, list) else OverlappingPatches
        ovp = ovp(
            noisy_image,
            self.patch_shape,
            patch_shift=1,
            verbose=verbose,
            color=self.color,
        )
        patches = ovp.get(copy=True)

        obj, logs = self.fit(
            X=patches,
            eps=eps,
            limit=limit,
            C_prime=C_prime,
            G=G,
            E=E,
            hard=hard,
            sim_measure=sim_measure,
            indices=indices,
            use_pretrainer=use_pretrainer,
            rng=rng,
            verbose=verbose,
        )
        reconstructed_patches = self.estimate(X=patches)
        denoised_image = ovp.set_and_merge(
            reconstructed_patches, merge_method=merge_method
        )
        return denoised_image
