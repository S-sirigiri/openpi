"""Tests for the SDF trilinear interpolation and collision-points sampling.

Run with: ``uv run pytest src/openpi/fkc/cost_constraint_test.py``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.fkc import fk as _fk
from openpi.fkc.cost_constraint import _sdf_trilinear


# ---------------------------------------------------------------------------
# _sdf_trilinear
# ---------------------------------------------------------------------------


def _sphere_sdf_grid(
    radius: float,
    grid_origin: tuple[float, float, float],
    voxel_size: float,
    grid_dims: tuple[int, int, int],
) -> jnp.ndarray:
    """Build an analytical SDF grid for a sphere centered at the world origin."""
    nx, ny, nz = grid_dims
    ox, oy, oz = grid_origin
    xs = ox + voxel_size * np.arange(nx)
    ys = oy + voxel_size * np.arange(ny)
    zs = oz + voxel_size * np.arange(nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    dist = np.sqrt(X * X + Y * Y + Z * Z) - radius
    return jnp.asarray(dist, dtype=jnp.float32)


def test_trilinear_at_voxel_centres_is_exact():
    """At voxel centres, trilinear interp should return the stored value exactly."""
    grid_origin = (-0.5, -0.5, -0.5)
    voxel_size = 0.05
    grid_dims = (20, 20, 20)
    grid = _sphere_sdf_grid(0.1, grid_origin, voxel_size, grid_dims)
    origin = jnp.asarray(grid_origin, dtype=jnp.float32)
    vs = jnp.asarray(voxel_size, dtype=jnp.float32)
    # Pick a few interior voxel centres.
    for ijk in [(5, 5, 5), (10, 10, 10), (15, 15, 15)]:
        wp = origin + vs * jnp.asarray(ijk, dtype=jnp.float32)
        val = _sdf_trilinear(grid, origin, vs, wp[None, :])[0]
        assert jnp.isclose(val, grid[ijk[0], ijk[1], ijk[2]], atol=1e-6)


def test_trilinear_matches_analytical_within_voxel():
    """Midpoints between voxels: agreement with analytical SDF within voxel-size tolerance."""
    grid_origin = (-0.5, -0.5, -0.5)
    voxel_size = 0.02
    grid_dims = (50, 50, 50)
    radius = 0.1
    grid = _sphere_sdf_grid(radius, grid_origin, voxel_size, grid_dims)
    origin = jnp.asarray(grid_origin, dtype=jnp.float32)
    vs = jnp.asarray(voxel_size, dtype=jnp.float32)
    # Random query points well inside the grid (away from sphere surface).
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.3, 0.3, size=(20, 3)).astype(np.float32)
    interp = _sdf_trilinear(grid, origin, vs, jnp.asarray(pts))
    analytical = np.sqrt(np.sum(pts * pts, axis=-1)) - radius
    # Trilinear has at most O(voxel_size) error in smooth regions for a smooth field.
    np.testing.assert_allclose(np.asarray(interp), analytical, atol=voxel_size)


def test_trilinear_clamps_out_of_grid():
    """Out-of-grid queries are clamped to the boundary voxel value."""
    grid_origin = (0.0, 0.0, 0.0)
    voxel_size = 0.1
    grid = jnp.full((4, 4, 4), 7.0, dtype=jnp.float32)
    origin = jnp.asarray(grid_origin, dtype=jnp.float32)
    vs = jnp.asarray(voxel_size, dtype=jnp.float32)
    # Far outside in every direction.
    pts = jnp.asarray([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]], dtype=jnp.float32)
    vals = _sdf_trilinear(grid, origin, vs, pts)
    np.testing.assert_allclose(np.asarray(vals), [7.0, 7.0], atol=1e-6)


def test_trilinear_grad_points_outward():
    """∇_p sdf at a query point near the sphere surface should point outward."""
    grid_origin = (-0.5, -0.5, -0.5)
    voxel_size = 0.02
    grid_dims = (50, 50, 50)
    radius = 0.1
    grid = _sphere_sdf_grid(radius, grid_origin, voxel_size, grid_dims)
    origin = jnp.asarray(grid_origin, dtype=jnp.float32)
    vs = jnp.asarray(voxel_size, dtype=jnp.float32)

    def f(p):
        return _sdf_trilinear(grid, origin, vs, p[None, :])[0]

    # A point along +x just outside the sphere.
    p = jnp.asarray([0.15, 0.0, 0.0], dtype=jnp.float32)
    g = jax.grad(f)(p)
    # Gradient should be ~(+1, 0, 0) for an SDF in x>0.
    g_np = np.asarray(g)
    assert g_np[0] > 0.5
    assert abs(g_np[1]) < 0.2
    assert abs(g_np[2]) < 0.2


# ---------------------------------------------------------------------------
# franka_collision_points
# ---------------------------------------------------------------------------


def _identity_transforms() -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.eye(4, dtype=jnp.float32), jnp.eye(4, dtype=jnp.float32)


def _zero_joints() -> jnp.ndarray:
    return jnp.zeros((7,), dtype=jnp.float32)


def test_collision_points_ee_only_matches_fk_position():
    base, ee = _identity_transforms()
    q = _zero_joints()
    pts = _fk.franka_collision_points(q, base, ee, mode="ee_only")
    assert pts.shape == (1, 3)
    expected = _fk.franka_fk_position(q, base, ee)
    np.testing.assert_allclose(np.asarray(pts[0]), np.asarray(expected), atol=1e-6)


def test_collision_points_ee_plus_arm_count_and_first_is_ee():
    base, ee = _identity_transforms()
    q = _zero_joints()
    arm_n = 8
    pts = _fk.franka_collision_points(
        q, base, ee, mode="ee_plus_arm", arm_sample_points=arm_n
    )
    # First point is EE; then 3 gripper points; then arm_n arm samples.
    assert pts.shape == (arm_n + 4, 3)
    expected_ee = _fk.franka_fk_position(q, base, ee)
    np.testing.assert_allclose(np.asarray(pts[0]), np.asarray(expected_ee), atol=1e-6)
    # Helper count should match.
    assert _fk.collision_points_count("ee_plus_arm", arm_sample_points=arm_n) == arm_n + 4


def test_collision_points_full_count():
    base, ee = _identity_transforms()
    q = _zero_joints()
    full_n = 30
    pts = _fk.franka_collision_points(
        q, base, ee, mode="full", full_body_points=full_n
    )
    assert pts.shape == (full_n + 4, 3)
    assert _fk.collision_points_count("full", full_body_points=full_n) == full_n + 4


def test_collision_points_within_workspace():
    """All sample points must be within ~1.5 m of the robot base (Franka reach <0.85 m)."""
    base, ee = _identity_transforms()
    q = jnp.asarray([0.1, -0.3, 0.0, -1.5, 0.0, 1.2, 0.5], dtype=jnp.float32)
    for mode in ("ee_only", "ee_plus_arm", "full"):
        pts = _fk.franka_collision_points(q, base, ee, mode=mode)
        norms = jnp.linalg.norm(pts, axis=-1)
        assert float(jnp.max(norms)) < 1.5, f"mode={mode} has out-of-workspace points"


def test_collision_points_invalid_mode():
    base, ee = _identity_transforms()
    q = _zero_joints()
    with pytest.raises(ValueError):
        _fk.franka_collision_points(q, base, ee, mode="bogus")


def test_collision_points_jit_and_grad():
    """Collision-points sampling must be jit + grad friendly."""
    base, ee = _identity_transforms()

    @jax.jit
    def f(q):
        pts = _fk.franka_collision_points(
            q, base, ee, mode="ee_plus_arm", arm_sample_points=8
        )
        return jnp.sum(pts * pts)

    q = jnp.asarray([0.1, -0.3, 0.0, -1.5, 0.0, 1.2, 0.5], dtype=jnp.float32)
    val = f(q)
    grad = jax.grad(f)(q)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))
    assert grad.shape == (7,)
