"""
minax.py — JAX-based 3D Minkowski Functionals calculator.

Provides JIT-compilable functions for computing Minkowski Functionals (MFs)
of 3D scalar fields, both numerically and analytically (for Gaussian random
fields).

Functions
---------
calculateMFs     — Numerical MFs from a 3D field
analyticalMFs    — Analytical MFs for Gaussian random fields (for validation)
make_thresholds  — Generate threshold values from field statistics
hessian          — Compute gradient and Hessian of a 3D field
subtractWedge    — Remove foreground wedge in Fourier space

Example: comparing numerical vs analytical on a Gaussian random field
---------------------------------------------------------------------
>>> import jax.numpy as jnp
>>> from minax import calculateMFs, analyticalMFs, make_thresholds
>>>
>>> key = jax.random.PRNGKey(0)
>>> data = jax.random.normal(key, shape=(64, 64, 64))  # Gaussian field
>>>
>>> # Numerical MFs (thresholds in field units)
>>> thresholds = make_thresholds(data)
>>> v0n, v1n, v2n, v3n = calculateMFs(data, thresholds)
>>>
>>> # Analytical MFs (thresholds in units of sigma, i.e. dimensionless nu)
>>> nu = jnp.linspace(-3, 3, 61)
>>> v0a, v1a, v2a, v3a = analyticalMFs(data, nu)
>>>
>>> # For plotting, the x-axes are related by: thresholds = nu * std(data)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import erfc
from functools import partial


# ── Levi-Civita tensor (module-level constant) ──────────────────────────────

def _make_levi_civita():
    """Construct the 3D Levi-Civita (fully antisymmetric) tensor."""
    eps = jnp.zeros((3, 3, 3), dtype=jnp.float64)
    eps = eps.at[0, 1, 2].set(1.0)
    eps = eps.at[1, 2, 0].set(1.0)
    eps = eps.at[2, 0, 1].set(1.0)
    eps = eps.at[0, 2, 1].set(-1.0)
    eps = eps.at[2, 1, 0].set(-1.0)
    eps = eps.at[1, 0, 2].set(-1.0)
    return eps


_EPS = _make_levi_civita()


# ── Utilities ────────────────────────────────────────────────────────────────

def make_thresholds(data, min_sig=-3.0, max_sig=3.0, step=0.1):
    """Generate threshold values in field units from sigma multiples.

    Not JIT-compiled (calls ``float`` / ``int`` on traced values).
    Call this *before* passing thresholds into ``calculateMFs``.

    Parameters
    ----------
    data : array, shape (Nx, Ny, Nz)
    min_sig, max_sig : float
        Bounds in units of std(data).
    step : float
        Step size in units of std(data).

    Returns
    -------
    thresholds : jnp.ndarray, shape (N,)
        Threshold values in field units.
    """
    sig = float(jnp.std(data))
    n = int(round((max_sig - min_sig) / step)) + 1
    return jnp.linspace(min_sig * sig, max_sig * sig, n)


# ── Gradient & Hessian ──────────────────────────────────────────────────────

@jit
def hessian(data):
    """Compute gradient and Hessian of a 3D scalar field via finite differences.

    Parameters
    ----------
    data : jnp.ndarray, shape (Nx, Ny, Nz)

    Returns
    -------
    grad : jnp.ndarray, shape (Nx, Ny, Nz, 3)
        Gradient vector at each voxel.
    hess : jnp.ndarray, shape (Nx, Ny, Nz, 3, 3)
        Hessian matrix at each voxel.  ``hess[..., i, j] = d²f / dx_i dx_j``.
    """
    g0 = jnp.gradient(data, axis=0)
    g1 = jnp.gradient(data, axis=1)
    g2 = jnp.gradient(data, axis=2)
    grad = jnp.stack([g0, g1, g2], axis=-1)

    hess = jnp.stack([
        jnp.stack([jnp.gradient(g0, axis=0),
                    jnp.gradient(g0, axis=1),
                    jnp.gradient(g0, axis=2)], axis=-1),
        jnp.stack([jnp.gradient(g1, axis=0),
                    jnp.gradient(g1, axis=1),
                    jnp.gradient(g1, axis=2)], axis=-1),
        jnp.stack([jnp.gradient(g2, axis=0),
                    jnp.gradient(g2, axis=1),
                    jnp.gradient(g2, axis=2)], axis=-1),
    ], axis=-2)

    return grad, hess


# ── Core MF computation ─────────────────────────────────────────────────────

@jit
def _compute_mf_integrands(grad, hess):
    r"""Vectorised computation of per-voxel MF integrands.

    Uses the algebraic identities (derivable by contracting two
    Levi-Civita tensors):

    .. math::
        S_1^{\rm raw} = g^{\!\top} H\, g \;-\; |\mathbf g|^2\,\mathrm{tr}(H)

        S_2^{\rm raw} = \varepsilon_{lmn}\, g_l\, g_\alpha\,
                        \varepsilon_{\alpha\beta\gamma}\,
                        H_{m\beta}\, H_{n\gamma}

    Parameters
    ----------
    grad : (..., 3)
    hess : (..., 3, 3)

    Returns
    -------
    gradnorm, sum1, sum2 : each (...)
    """
    gnorm_sq = jnp.sum(grad ** 2, axis=-1)
    gradnorm = jnp.sqrt(gnorm_sq)

    # ── sum1_raw = gᵀ H g − |g|² tr(H) ──
    trace_H = jnp.trace(hess, axis1=-2, axis2=-1)
    gHg = jnp.einsum('...i,...ij,...j->...', grad, hess, grad)
    sum1_raw = gHg - gnorm_sq * trace_H

    # ── sum2_raw via cross-products of Hessian rows ──
    # cross_H[..., m, n, α] = (H[m,:] × H[n,:])[α]
    cross_H = jnp.cross(hess[..., :, None, :],
                         hess[..., None, :, :])          # (..., 3, 3, 3)
    # A[..., m, n] = g · (H[m,:] × H[n,:])
    A = jnp.sum(grad[..., None, None, :] * cross_H,
                axis=-1)                                  # (..., 3, 3)
    # sum2_raw = ε_{lmn} g_l A_{mn}
    sum2_raw = jnp.einsum('lmn,...l,...mn->...', _EPS, grad, A)

    # ── normalise by powers of |grad|, guarding against zeros ──
    safe_sq = jnp.where(gnorm_sq > 0, gnorm_sq, 1.0)
    safe_cu = jnp.where(gnorm_sq > 0, gnorm_sq * gradnorm, 1.0)

    sum1 = jnp.where(gnorm_sq > 0, sum1_raw / safe_sq, 0.0)
    sum2 = jnp.where(gnorm_sq > 0, sum2_raw / (2.0 * safe_cu), 0.0)

    # clean NaNs and clamp extreme values
    sum1 = jnp.nan_to_num(sum1, nan=0.0)
    sum2 = jnp.nan_to_num(sum2, nan=0.0)
    sum2 = jnp.where(jnp.abs(sum2) > 1000.0, 0.0, sum2)

    return gradnorm, sum1, sum2


@jit
def calculateMFs(data, thresholds, deltabin=0.4, is_need_calculate_bin=True):
    r"""Numerically compute the four 3-D Minkowski Functionals (v₀–v₃).

    Fully JIT-compilable JAX reimplementation.  The expensive per-voxel
    Levi-Civita contractions are replaced by vectorised Einstein summations
    and cross products.

    Parameters
    ----------
    data : jnp.ndarray, shape (Nx, Ny, Nz)
        The 3-D scalar field.
    thresholds : jnp.ndarray, shape (N,)
        Threshold values **in field units**.
        Use ``make_thresholds(data)`` to auto-generate from σ multiples.
    deltabin : float
        Width of the rectangular window that approximates the Dirac δ.
        Default 0.4 (in σ units when ``is_need_calculate_bin=True``).
    is_need_calculate_bin : bool
        If True, ``deltabin`` is multiplied by std(data).

    Returns
    -------
    v0, v1, v2, v3 : jnp.ndarray, each of shape (N,)
    """
    data = data.astype(jnp.float64)
    sig = jnp.std(data)
    volume = float(data.shape[0] * data.shape[1] * data.shape[2])
    dbin = jnp.where(is_need_calculate_bin, deltabin * sig, deltabin)

    # gradient, Hessian, MF integrands
    grad, hess = hessian(data)
    gradnorm, sum1, sum2 = _compute_mf_integrands(grad, hess)

    # evaluate at each threshold (vmapped)
    def _at_threshold(th):
        dth = 0.5 * dbin
        v0 = jnp.sum((data >= th).astype(jnp.float64)) / volume
        mask = (jnp.abs(data - th) < dth).astype(jnp.float64)
        v1 = jnp.sum(gradnorm * mask) / (6.0 * volume * dbin)
        v2 = jnp.sum(sum1 * mask) / (6.0 * volume * dbin * jnp.pi)
        v3 = jnp.sum(sum2 * mask) / (4.0 * volume * dbin * jnp.pi)
        return jnp.array([v0, v1, v2, v3])

    mfs = jax.vmap(_at_threshold)(thresholds)
    return mfs[:, 0], mfs[:, 1], mfs[:, 2], mfs[:, 3]


# ── Analytical MFs (Gaussian random fields only) ────────────────────────────

@jit
def analyticalMFs(data, thresholds):
    r"""Analytical Minkowski Functionals for a Gaussian random field.

    Exact closed-form expressions:

    .. math::
        v_0(\nu) &= \tfrac12\,\mathrm{erfc}\!\bigl(\nu/\sqrt2\bigr)\\
        v_1(\nu) &= \tfrac{2\lambda}{3\sqrt{2\pi}}\, e^{-\nu^2/2}\\
        v_2(\nu) &= \tfrac{2\lambda^2}{3\sqrt{2\pi}}\,\nu\, e^{-\nu^2/2}\\
        v_3(\nu) &= \tfrac{\lambda^3}{\sqrt{2\pi}}\,(\nu^2-1)\, e^{-\nu^2/2}

    where :math:`\lambda = \sqrt{1/(6\pi)}\;\sigma_1/\sigma_0`,
    :math:`\sigma_0 = \mathrm{std}(f)`, and
    :math:`\sigma_1 = \mathrm{rms}(|\nabla f|)`.

    Parameters
    ----------
    data : jnp.ndarray, shape (Nx, Ny, Nz)
        The 3-D field.  Used only to extract the spectral parameter λ.
    thresholds : jnp.ndarray, shape (N,)
        **Dimensionless** threshold values ν (in units of σ₀).
        Typical choice: ``jnp.linspace(-3, 3, 61)``.

    Returns
    -------
    v0, v1, v2, v3 : jnp.ndarray, each of shape (N,)

    Notes
    -----
    To compare with the numerical ``calculateMFs``:

    .. code-block:: python

        nu = jnp.linspace(-3, 3, 61)
        v0a, v1a, v2a, v3a = analyticalMFs(data, nu)

        sig = jnp.std(data - jnp.mean(data))
        v0n, v1n, v2n, v3n = calculateMFs(data, nu * sig)
    """
    data = data.astype(jnp.float64)
    volume = float(data.shape[0] * data.shape[1] * data.shape[2])
    data_c = data - jnp.mean(data)
    sigma0 = jnp.sqrt(jnp.sum(data_c ** 2) / volume)

    # spectral parameter
    g0 = jnp.gradient(data_c, axis=0)
    g1 = jnp.gradient(data_c, axis=1)
    g2 = jnp.gradient(data_c, axis=2)
    sigma1 = jnp.sqrt(jnp.sum(g0 ** 2 + g1 ** 2 + g2 ** 2) / volume)

    lam = jnp.sqrt(1.0 / (6.0 * jnp.pi)) * (sigma1 / sigma0)

    nu = thresholds
    gauss = jnp.exp(-0.5 * nu ** 2)
    c = 1.0 / jnp.sqrt(2.0 * jnp.pi)       # (2π)^{-1/2}

    v0 = erfc(nu / jnp.sqrt(2.0)) / 2.0
    v1 = (2.0 / 3.0) * lam * c * gauss
    v2 = (2.0 / 3.0) * lam ** 2 * c * gauss * nu
    v3 = lam ** 3 * c * gauss * (nu ** 2 - 1.0)

    return v0, v1, v2, v3


# ── Foreground wedge subtraction ────────────────────────────────────────────

@partial(jit, static_argnames=('nu_axis',))
def subtractWedge(data, m=0.5, nu_axis=2):
    """Remove foreground-wedge modes in Fourier space (21-cm cosmology).

    Parameters
    ----------
    data : jnp.ndarray, shape (Nx, Ny, Nz)
    m : float
        Slope of the wedge boundary in (k_perp, k_para) space.
    nu_axis : int
        Axis along the line-of-sight / frequency direction.

    Returns
    -------
    jnp.ndarray – filtered field with wedge modes zeroed.
    """
    x_size = data.shape[0] if nu_axis == 2 else data.shape[2]
    z_size = data.shape[nu_axis]

    datak = jnp.fft.rfftn(data)
    kx = jnp.fft.fftfreq(x_size)
    kz = jnp.fft.rfftfreq(z_size)
    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, kx, kz, indexing='ij')

    k_transverse = jnp.sqrt(kx_grid ** 2 + ky_grid ** 2)
    mask = k_transverse < m * kz_grid
    datak = jnp.where(mask, 0.0 + 0.0j, datak)
    return jnp.fft.irfftn(datak, s=data.shape)


# ── Test / demo ─────────────────────────────────────────────────────────────

def test_grf(N=256, seed=42, save="mf_comparison.png"):
    """Generate a standard GRF of size N³, compute numerical & analytical MFs,
    and plot the comparison.

    Parameters
    ----------
    N : int
        Grid size per dimension.
    seed : int
        Random seed.
    save : str or None
        If not None, save figure to this path.
    """
    import matplotlib.pyplot as plt

    print(f"Generating {N}³ Gaussian random field (seed={seed}) ...")
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, shape=(N, N, N), dtype=jnp.float64)

    # ── Analytical ──
    nu = jnp.linspace(-3.0, 3.0, 61)
    print("Computing analytical MFs ...")
    v0a, v1a, v2a, v3a = analyticalMFs(data, nu)

    # ── Numerical (thresholds in field units = nu * sigma) ──
    sig = float(jnp.std(data - jnp.mean(data)))
    thresholds = nu * sig
    print("Computing numerical MFs ...")
    v0n, v1n, v2n, v3n = calculateMFs(data, thresholds)

    # ── Plot ──
    nu_np = np.array(nu)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    labels = [r"$v_0$", r"$v_1$", r"$v_2$", r"$v_3$"]
    analytical = [v0a, v1a, v2a, v3a]
    numerical  = [v0n, v1n, v2n, v3n]

    for ax, lab, va, vn in zip(axes.flat, labels, analytical, numerical):
        ax.plot(nu_np, np.array(va), "k-", lw=1.5, label="Analytical")
        ax.plot(nu_np, np.array(vn), "ro", ms=3, label="Numerical")
        ax.set_ylabel(lab, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel(r"$\nu\;(\sigma)$", fontsize=12)

    fig.suptitle(
        f"Minkowski Functionals — GRF $({N}^3)$: numerical vs analytical",
        fontsize=14,
    )
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"Figure saved to {save}")
    plt.show()


if __name__ == "__main__":
    test_grf()
