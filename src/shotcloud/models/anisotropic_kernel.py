"""Tier-2 anisotropic kernels: per-zone covariance for the spatial mixture.

Two parameterizations, both replacing the current circular Gaussian
kernel ``K_σ(y − s_m) = N₂(y; s_m, σ²I)`` with a zone-aware anisotropic
Gaussian ``K_Σ(y − s_m) = N₂(y; s_m, Σ_m)``:

* :class:`RadialTangentZoneKernel` — Option 1, basketball-aligned.
  Per-zone ``(σ_r, σ_t)``. For each support shot ``s_m``, the principal
  axes are radial (toward / away from basket) and tangential (along
  the arc), set by ``s_m / ‖s_m‖``. 16 scalars total.
* :class:`FullCovarianceZoneKernel` — Option 3, flexibility upper-bound.
  Per-zone ``(σ_x, σ_y, ρ)`` in bounded-correlation form. The principal
  axes can rotate freely per zone. 24 scalars total.

Both modules expose ``forward(*, support_xy, shot_xy) -> (B, M)``
returning the per-shot log-kernel value, which the spatial loglik adds
to the support log-weights before the per-row logsumexp.

**Isotropic-collapse invariant** (load-bearing, tested for both
modules): at ``σ_r = σ_t = σ_init`` (resp.
``σ_x = σ_y = σ_init, ρ = 0``), the anisotropic kernel reduces to the
fixed-σ isotropic Gaussian bit-exactly up to float32 precision. This
guarantees the wrapper starts at the existing-mainline likelihood and
anisotropy emerges only as training proceeds.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from shotcloud.data.zones import N_ZONES
from shotcloud.models.zone_source_bandwidth import zone_from_xy_torch_bandwidth

__all__ = [
    "FullCovarianceZoneKernel",
    "RadialTangentZoneKernel",
]


def _bounded_sigma_init_logit(sigma_min: float, sigma_max: float, sigma_init: float) -> float:
    """Inverse-sigmoid the target into the raw-logit space.

    ``σ = σ_min + (σ_max − σ_min) · sigmoid(z)`` ⇔
    ``z = logit((σ_init − σ_min) / (σ_max − σ_min))``. Identical
    initialization pattern to :class:`ZoneSourceBandwidth` so the
    isotropic-collapse invariant is bit-exact when both modules
    share their canonical ``(σ_min, σ_max, σ_init)`` triple.
    """
    target = (sigma_init - sigma_min) / (sigma_max - sigma_min)
    target = min(max(target, 1e-6), 1.0 - 1e-6)
    return math.log(target / (1.0 - target))


class RadialTangentZoneKernel(nn.Module):
    """Per-zone radial-tangential anisotropic kernel (Option 1).

    For each support shot ``s_m``, the kernel covariance is

    .. math::

        \\Sigma_m = \\sigma_{r,z(s_m)}^2 \\hat r_m \\hat r_m^\\top
                  + \\sigma_{t,z(s_m)}^2 \\hat t_m \\hat t_m^\\top,

    where ``r̂_m = s_m / ‖s_m‖`` is the radial direction at ``s_m``
    (from the basket) and ``t̂_m = (−s_{m,y}, s_{m,x}) / ‖s_m‖`` is its
    90°-rotated tangent. Since ``r̂_m`` and ``t̂_m`` are eigenvectors
    of ``Σ_m``, the log-kernel evaluates in closed form:

    .. math::

        \\log K_m(\\delta) = -\\log(2\\pi) - \\log\\sigma_{r,z}
            - \\log\\sigma_{t,z}
            - \\tfrac{1}{2}\\!\\left(\\delta_r^2/\\sigma_{r,z}^2
                                    + \\delta_t^2/\\sigma_{t,z}^2\\right),

    with ``δ_r = δ · r̂_m`` and ``δ_t = δ · t̂_m``.

    **Parameters learned**: ``2 × N_ZONES = 16`` bounded scalars
    ``(σ_r, σ_t)`` per zone. Bounded via sigmoid into
    ``[σ_min, σ_max]``.

    **Isotropic collapse**: at ``σ_r = σ_t = σ_init``,
    ``Σ_m = σ_init² (r̂ r̂ᵀ + t̂ t̂ᵀ) = σ_init² I`` (since the basis is
    orthonormal). So the module reduces bit-exactly to the fixed-σ
    isotropic Gaussian.

    **Origin guard**: ``‖s_m‖ < ε_r`` (default 1e-4) makes the radial
    frame numerically unstable. Such support points fall back to the
    isotropic kernel with ``σ = σ_r`` at that zone. The fallback
    affects ~zero real NBA shots in practice (no one shoots from
    coordinate (0, 0)); the guard exists purely for numerical safety.

    Attributes
    ----------
    raw_r : nn.Parameter
        Shape ``(N_ZONES,)``. Per-zone σ_r logits.
    raw_t : nn.Parameter
        Shape ``(N_ZONES,)``. Per-zone σ_t logits.
    """

    sigma_min: Tensor
    sigma_max: Tensor
    sigma_range: Tensor

    def __init__(
        self,
        *,
        sigma_min: float = 1.0,
        sigma_max: float = 2.5,
        sigma_init: float = 1.5,
        origin_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if not (sigma_min < sigma_max):
            raise ValueError(f"sigma_min ({sigma_min}) must be < sigma_max ({sigma_max})")
        if not (sigma_min <= sigma_init <= sigma_max):
            raise ValueError(
                f"sigma_min={sigma_min} <= sigma_init={sigma_init} <= "
                f"sigma_max={sigma_max} required"
            )
        if origin_eps <= 0.0:
            raise ValueError(f"origin_eps must be positive; got {origin_eps}")
        self.register_buffer(
            "sigma_min", torch.tensor(float(sigma_min), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_max", torch.tensor(float(sigma_max), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_range",
            torch.tensor(float(sigma_max) - float(sigma_min), dtype=torch.float32),
            persistent=False,
        )
        self._origin_eps = float(origin_eps)
        init_logit = _bounded_sigma_init_logit(sigma_min, sigma_max, sigma_init)
        self.raw_r = nn.Parameter(torch.full((N_ZONES,), float(init_logit), dtype=torch.float32))
        self.raw_t = nn.Parameter(torch.full((N_ZONES,), float(init_logit), dtype=torch.float32))

    def sigma_r(self) -> Tensor:
        """Bounded ``σ_r`` per zone, shape ``(N_ZONES,)``."""
        return self.sigma_min + self.sigma_range * torch.sigmoid(self.raw_r)

    def sigma_t(self) -> Tensor:
        """Bounded ``σ_t`` per zone, shape ``(N_ZONES,)``."""
        return self.sigma_min + self.sigma_range * torch.sigmoid(self.raw_t)

    def forward(
        self,
        *,
        support_xy: Tensor,
        shot_xy: Tensor,
    ) -> Tensor:
        """Compute ``(B, M)`` per-shot log-kernel.

        Parameters
        ----------
        support_xy : Tensor of shape ``(B, M, 2)``
            Support-shot coordinates in court feet (basket at origin).
        shot_xy : Tensor of shape ``(B, 2)``
            Observed shot coordinates the mixture is evaluated at.

        Returns
        -------
        Tensor of shape ``(B, M)``
            ``log K_m(y_b − s_{b,m})`` per (row, support shot).
        """
        if support_xy.dim() != 3 or support_xy.shape[-1] != 2:
            raise ValueError(f"support_xy must be (B, M, 2); got {tuple(support_xy.shape)}")
        if shot_xy.dim() != 2 or shot_xy.shape[-1] != 2:
            raise ValueError(f"shot_xy must be (B, 2); got {tuple(shot_xy.shape)}")
        if shot_xy.shape[0] != support_xy.shape[0]:
            raise ValueError(
                f"batch dim mismatch: shot_xy[0]={shot_xy.shape[0]}, "
                f"support_xy[0]={support_xy.shape[0]}"
            )

        zone = zone_from_xy_torch_bandwidth(support_xy)  # (B, M), int64
        safe_zone = zone.clamp_min(0)
        sigma_r_per = self.sigma_r()[safe_zone]  # (B, M)
        sigma_t_per = self.sigma_t()[safe_zone]  # (B, M)

        sx = support_xy[..., 0]
        sy = support_xy[..., 1]
        norm = (sx * sx + sy * sy).clamp_min(self._origin_eps**2).sqrt()
        # Origin guard: a support shot exactly at the basket has no
        # well-defined radial frame; the fallback path uses an
        # isotropic σ_r kernel at that zone.
        near_origin = (sx * sx + sy * sy) < (self._origin_eps**2)
        r_hat_x = sx / norm
        r_hat_y = sy / norm
        # t̂ = R_90 r̂ = (−r̂_y, r̂_x)
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x

        delta = shot_xy.unsqueeze(1) - support_xy  # (B, M, 2)
        dx = delta[..., 0]
        dy = delta[..., 1]
        delta_r = dx * r_hat_x + dy * r_hat_y  # (B, M)
        delta_t = dx * t_hat_x + dy * t_hat_y  # (B, M)

        sigma_r2 = sigma_r_per.pow(2)
        sigma_t2 = sigma_t_per.pow(2)
        log_k_aniso = (
            -math.log(2.0 * math.pi)
            - torch.log(sigma_r_per)
            - torch.log(sigma_t_per)
            - 0.5 * (delta_r.pow(2) / sigma_r2 + delta_t.pow(2) / sigma_t2)
        )

        # Origin-guard isotropic fallback: at ‖s_m‖ ≈ 0, use Σ = σ_r²I.
        # |Σ| = σ_r⁴ ⇒ ½ log|Σ| = 2 log σ_r; quadratic = ‖δ‖²/σ_r².
        dist2 = dx.pow(2) + dy.pow(2)
        log_k_origin = (
            -math.log(2.0 * math.pi) - 2.0 * torch.log(sigma_r_per) - 0.5 * dist2 / sigma_r2
        )
        return torch.where(near_origin, log_k_origin, log_k_aniso)


class FullCovarianceZoneKernel(nn.Module):
    """Per-zone full covariance kernel in bounded-correlation form (Option 3).

    Per-zone covariance

    .. math::

        \\Sigma_z = \\begin{pmatrix}
            \\sigma_{x,z}^2 & \\rho_z \\sigma_{x,z}\\sigma_{y,z} \\\\
            \\rho_z \\sigma_{x,z}\\sigma_{y,z} & \\sigma_{y,z}^2
        \\end{pmatrix},

    with bounded ``σ_x, σ_y ∈ [σ_min, σ_max]`` via sigmoid and
    ``|ρ| < ρ_max`` via tanh. The principal axes can rotate freely
    per zone via ``ρ``, making this the most flexible per-zone
    parameterization.

    Log-kernel (closed form):

    .. math::

        \\log K_z(\\delta) = -\\log(2\\pi)
            - \\log\\sigma_{x,z} - \\log\\sigma_{y,z}
            - \\tfrac{1}{2}\\log(1 - \\rho_z^2)
            - \\frac{1}{2(1 - \\rho_z^2)}
              \\!\\left[\\frac{\\delta_x^2}{\\sigma_{x,z}^2}
                       - 2\\rho_z \\frac{\\delta_x\\delta_y}
                                        {\\sigma_{x,z}\\sigma_{y,z}}
                       + \\frac{\\delta_y^2}{\\sigma_{y,z}^2}\\right].

    **Parameters learned**: ``3 × N_ZONES = 24`` bounded scalars
    ``(σ_x, σ_y, ρ)`` per zone.

    **Isotropic collapse**: at ``σ_x = σ_y = σ_init, ρ = 0``,
    ``Σ_z = σ_init² I``. Bit-exact reduction to the fixed-σ isotropic
    Gaussian.

    Attributes
    ----------
    raw_sx : nn.Parameter
        Shape ``(N_ZONES,)``. σ_x logits.
    raw_sy : nn.Parameter
        Shape ``(N_ZONES,)``. σ_y logits.
    raw_rho : nn.Parameter
        Shape ``(N_ZONES,)``. ρ pre-tanh logits.
    """

    sigma_min: Tensor
    sigma_max: Tensor
    sigma_range: Tensor
    rho_max: Tensor

    def __init__(
        self,
        *,
        sigma_min: float = 1.0,
        sigma_max: float = 2.5,
        sigma_init: float = 1.5,
        rho_max: float = 0.8,
        rho_init: float = 0.0,
    ) -> None:
        super().__init__()
        if not (sigma_min < sigma_max):
            raise ValueError(f"sigma_min ({sigma_min}) must be < sigma_max ({sigma_max})")
        if not (sigma_min <= sigma_init <= sigma_max):
            raise ValueError(
                f"sigma_min={sigma_min} <= sigma_init={sigma_init} <= "
                f"sigma_max={sigma_max} required"
            )
        if not (0.0 < rho_max < 1.0):
            raise ValueError(f"rho_max must be in (0, 1); got {rho_max}")
        if not (-rho_max < rho_init < rho_max):
            raise ValueError(f"rho_init={rho_init} must lie in (-rho_max, +rho_max)={rho_max}")
        self.register_buffer(
            "sigma_min", torch.tensor(float(sigma_min), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_max", torch.tensor(float(sigma_max), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_range",
            torch.tensor(float(sigma_max) - float(sigma_min), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "rho_max", torch.tensor(float(rho_max), dtype=torch.float32), persistent=False
        )
        sigma_logit = _bounded_sigma_init_logit(sigma_min, sigma_max, sigma_init)
        # ρ = rho_max · tanh(r); invert to seed the raw logit.
        target_rho = rho_init / rho_max
        target_rho = min(max(target_rho, -1.0 + 1e-6), 1.0 - 1e-6)
        rho_logit = 0.5 * math.log((1.0 + target_rho) / (1.0 - target_rho))
        self.raw_sx = nn.Parameter(torch.full((N_ZONES,), float(sigma_logit), dtype=torch.float32))
        self.raw_sy = nn.Parameter(torch.full((N_ZONES,), float(sigma_logit), dtype=torch.float32))
        self.raw_rho = nn.Parameter(torch.full((N_ZONES,), float(rho_logit), dtype=torch.float32))

    def sigma_x(self) -> Tensor:
        return self.sigma_min + self.sigma_range * torch.sigmoid(self.raw_sx)

    def sigma_y(self) -> Tensor:
        return self.sigma_min + self.sigma_range * torch.sigmoid(self.raw_sy)

    def rho(self) -> Tensor:
        return self.rho_max * torch.tanh(self.raw_rho)

    def forward(
        self,
        *,
        support_xy: Tensor,
        shot_xy: Tensor,
    ) -> Tensor:
        """Compute ``(B, M)`` per-shot log-kernel.

        Parameters
        ----------
        support_xy : Tensor of shape ``(B, M, 2)``
        shot_xy : Tensor of shape ``(B, 2)``

        Returns
        -------
        Tensor of shape ``(B, M)``
        """
        if support_xy.dim() != 3 or support_xy.shape[-1] != 2:
            raise ValueError(f"support_xy must be (B, M, 2); got {tuple(support_xy.shape)}")
        if shot_xy.dim() != 2 or shot_xy.shape[-1] != 2:
            raise ValueError(f"shot_xy must be (B, 2); got {tuple(shot_xy.shape)}")
        if shot_xy.shape[0] != support_xy.shape[0]:
            raise ValueError(
                f"batch dim mismatch: shot_xy[0]={shot_xy.shape[0]}, "
                f"support_xy[0]={support_xy.shape[0]}"
            )

        zone = zone_from_xy_torch_bandwidth(support_xy)
        safe_zone = zone.clamp_min(0)
        sx_per = self.sigma_x()[safe_zone]  # (B, M)
        sy_per = self.sigma_y()[safe_zone]  # (B, M)
        rho_per = self.rho()[safe_zone]  # (B, M)

        delta = shot_xy.unsqueeze(1) - support_xy
        dx = delta[..., 0]
        dy = delta[..., 1]

        one_minus_r2 = (1.0 - rho_per.pow(2)).clamp_min(1e-6)
        # Quadratic form: [δ_x²/σ_x² − 2ρ δ_x δ_y/(σ_x σ_y) + δ_y²/σ_y²] / (1 − ρ²)
        q = (
            dx.pow(2) / sx_per.pow(2)
            - 2.0 * rho_per * dx * dy / (sx_per * sy_per)
            + dy.pow(2) / sy_per.pow(2)
        ) / one_minus_r2
        # ½ log|Σ| = log σ_x + log σ_y + ½ log(1 − ρ²)
        log_norm = torch.log(sx_per) + torch.log(sy_per) + 0.5 * torch.log(one_minus_r2)
        return -math.log(2.0 * math.pi) - log_norm - 0.5 * q
