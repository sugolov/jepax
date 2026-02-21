import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key


def hinge_std_loss(x: Float[Array, "B D"], std_margin: float = 1.0) -> Float[Array, ""]:
    std = jnp.sqrt(jnp.var(x, axis=0) + 1e-4)
    return jnp.mean(jax.nn.relu(std_margin - std))


def covariance_loss(x: Float[Array, "B D"]) -> Float[Array, ""]:
    """Mean of squared off-diagonal covariance elements."""
    B, D = x.shape
    x_centered = x - jnp.mean(x, axis=0, keepdims=True)
    cov = (x_centered.T @ x_centered) / (B - 1)
    mask = 1.0 - jnp.eye(D)
    return jnp.mean((cov * mask) ** 2)


def vicreg_loss(
    z1: Float[Array, "B D"],
    z2: Float[Array, "B D"],
    std_coeff: float = 1.0,
    cov_coeff: float = 80.0,
) -> dict[str, Float[Array, ""]]:
    """Variance-Invariance-Covariance regularization."""
    invariance = jnp.mean((z1 - z2) ** 2)
    var = hinge_std_loss(z1) + hinge_std_loss(z2)
    cov = covariance_loss(z1) + covariance_loss(z2)
    loss = invariance + std_coeff * var + cov_coeff * cov
    return {
        "loss": loss,
        "invariance_loss": invariance,
        "var_loss": var,
        "cov_loss": cov,
    }


# todo: sharded?
def epps_pulley(
    x: Float[Array, "B M"],
    t_min: float = -3.0,
    t_max: float = 3.0,
    n_points: int = 10,
) -> Float[Array, " M"]:
    """Epps-Pulley Gaussianity test statistic via characteristic function comparison."""
    t = jnp.linspace(t_min, t_max, n_points)

    mu = jnp.mean(x, axis=0, keepdims=True)
    sigma = jnp.std(x, axis=0, keepdims=True) + 1e-8
    x_std = (x - mu) / sigma

    tx = t[:, None, None] * x_std[None, :, :]  # [n_points, B, M]
    emp_cf = jnp.mean(jnp.exp(1j * tx), axis=1)  # [n_points, M]
    theo_cf = jnp.exp(-0.5 * t**2)[:, None]  # [n_points, 1]

    diff_sq = jnp.abs(emp_cf - theo_cf) ** 2
    return jnp.trapezoid(diff_sq, t, axis=0)


def bcs_loss(
    z1: Float[Array, "B D"],
    z2: Float[Array, "B D"],
    key: Key[Array, ""],
    num_slices: int = 256,
    lmbd: float = 10.0,
) -> dict[str, Float[Array, ""]]:
    """Batched Characteristic Slicing (BCS) loss for SIGReg."""
    D = z1.shape[1]
    A = jax.random.normal(key, (D, num_slices))
    A = A / jnp.linalg.norm(A, axis=0, keepdims=True)

    proj1 = z1 @ A
    proj2 = z2 @ A

    bcs = (jnp.mean(epps_pulley(proj1)) + jnp.mean(epps_pulley(proj2))) / 2.0
    invariance = jnp.mean((z1 - z2) ** 2)
    loss = invariance + lmbd * bcs
    return {
        "loss": loss,
        "bcs_loss": bcs,
        "invariance_loss": invariance,
    }


def smooth_l1_loss(
    pred: Float[Array, "B S D"],
    target: Float[Array, "B S D"],
    valid_mask: Float[Array, "B S"],
) -> Float[Array, ""]:
    """Masked smooth L1 loss over sequence positions."""
    diff = pred - target
    abs_diff = jnp.abs(diff)
    smooth_l1 = jnp.where(abs_diff < 1.0, 0.5 * diff**2, abs_diff - 0.5)
    loss_per_token = jnp.mean(smooth_l1, axis=-1)  # [B, seq_len]
    return jnp.sum(loss_per_token * valid_mask) / jnp.maximum(jnp.sum(valid_mask), 1.0)
