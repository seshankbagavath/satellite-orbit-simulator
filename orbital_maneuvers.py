# ==============================================================================
# ORBITAL MANEUVERS — Hohmann, Bi-elliptic, and Lambert transfers
# ------------------------------------------------------------------------------
# Adds impulsive-maneuver planning to the simulator:
#
#   * Hohmann transfer      — the minimum-energy two-burn transfer between two
#                             coplanar circular orbits.
#   * Bi-elliptic transfer  — a three-burn transfer that can beat Hohmann when
#                             the radius ratio is large (> ~11.94).
#   * Lambert solver        — given two position vectors and a time of flight,
#                             solve for the connecting transfer orbit (the basis
#                             of intercepts and rendezvous).
#
# All routines return both the delta-v budget and the geometry needed to plot
# the transfer, so they slot directly into the visualization layer.
#
# Author:  Seshank Bagavath
# License: MIT
#
# Depends on: satellite_orbit_simulator.py (for EARTH / mu and propagation).
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from satellite_orbit_simulator import EARTH, CentralBody
except Exception:  # running inline in a single notebook
    pass


# ==============================================================================
# === HOHMANN TRANSFER ===
# ==============================================================================
@dataclass
class TransferResult:
    """Delta-v budget and geometry for a two- or three-burn transfer."""
    dv_total: float                 # total delta-v magnitude [km/s]
    dv_burns: Tuple[float, ...]     # individual burn magnitudes [km/s]
    tof: float                      # total time of flight [s]
    kind: str                       # "hohmann" | "bielliptic"
    r_inner: float                  # inner circular radius [km]
    r_outer: float                  # outer circular radius [km]
    r_intermediate: Optional[float] = None  # bi-elliptic apoapsis radius [km]

    def summary(self) -> str:
        lines = [
            f"{self.kind.capitalize()} transfer",
            f"  burns      : " + ", ".join(f"{b*1000:.1f} m/s" for b in self.dv_burns),
            f"  total dv   : {self.dv_total*1000:.1f} m/s",
            f"  time of flight: {self.tof/3600:.2f} h",
        ]
        return "\n".join(lines)


def hohmann_transfer(r1: float, r2: float,
                     mu: float = None) -> TransferResult:
    """Minimum-energy two-burn transfer between coplanar circular orbits.

    Parameters
    ----------
    r1, r2 : float
        Radii of the initial and final circular orbits [km] (measured from the
        center of the central body, not altitude).
    mu : float
        Gravitational parameter [km^3/s^2]. Defaults to Earth.

    Returns
    -------
    TransferResult

    Notes
    -----
    The spacecraft burns prograde at r1 to enter an elliptical transfer orbit
    whose periapsis is r1 and apoapsis is r2, coasts half an orbit, then burns
    again at r2 to circularize. Works for both raising (r2 > r1) and lowering
    (r2 < r1) — the second burn is retrograde when lowering.
    """
    mu = mu if mu is not None else EARTH.mu
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbit radii must be positive.")

    # Transfer-ellipse semi-major axis.
    a_t = 0.5 * (r1 + r2)

    # Circular speeds at each orbit.
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)

    # Speeds on the transfer ellipse at peri/apo (vis-viva).
    v_t1 = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
    v_t2 = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))

    dv1 = abs(v_t1 - v_c1)
    dv2 = abs(v_c2 - v_t2)

    tof = np.pi * np.sqrt(a_t**3 / mu)  # half the transfer-ellipse period

    return TransferResult(
        dv_total=dv1 + dv2,
        dv_burns=(dv1, dv2),
        tof=tof,
        kind="hohmann",
        r_inner=min(r1, r2),
        r_outer=max(r1, r2),
    )


# ==============================================================================
# === BI-ELLIPTIC TRANSFER ===
# ==============================================================================
def bielliptic_transfer(r1: float, r2: float, r_b: float,
                        mu: float = None) -> TransferResult:
    """Three-burn bi-elliptic transfer via an intermediate apoapsis r_b.

    Parameters
    ----------
    r1, r2 : float
        Initial and final circular radii [km].
    r_b : float
        Intermediate apoapsis radius [km]; must exceed both r1 and r2 to make
        sense. Larger r_b lowers delta-v but increases time of flight.
    mu : float
        Gravitational parameter. Defaults to Earth.

    Notes
    -----
    Burn 1 raises apoapsis to r_b; burn 2 (at r_b) raises periapsis to r2;
    burn 3 (back at r2) circularizes. For large radius ratios (r2/r1 ≳ 11.94)
    this can require less total delta-v than a Hohmann transfer despite the
    extra burn.
    """
    mu = mu if mu is not None else EARTH.mu
    if min(r1, r2, r_b) <= 0:
        raise ValueError("All radii must be positive.")
    if r_b < max(r1, r2):
        raise ValueError(
            "Intermediate radius r_b should exceed both r1 and r2 for a "
            "sensible bi-elliptic transfer."
        )

    a1 = 0.5 * (r1 + r_b)   # first transfer ellipse
    a2 = 0.5 * (r2 + r_b)   # second transfer ellipse

    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)

    # Burn 1: circular at r1 -> ellipse 1 periapsis.
    v_p1 = np.sqrt(mu * (2.0 / r1 - 1.0 / a1))
    dv1 = abs(v_p1 - v_c1)

    # Burn 2: ellipse 1 apoapsis (at r_b) -> ellipse 2 apoapsis (at r_b).
    v_a1 = np.sqrt(mu * (2.0 / r_b - 1.0 / a1))
    v_a2 = np.sqrt(mu * (2.0 / r_b - 1.0 / a2))
    dv2 = abs(v_a2 - v_a1)

    # Burn 3: ellipse 2 periapsis (at r2) -> circular at r2.
    v_p2 = np.sqrt(mu * (2.0 / r2 - 1.0 / a2))
    dv3 = abs(v_c2 - v_p2)

    tof = np.pi * (np.sqrt(a1**3 / mu) + np.sqrt(a2**3 / mu))

    return TransferResult(
        dv_total=dv1 + dv2 + dv3,
        dv_burns=(dv1, dv2, dv3),
        tof=tof,
        kind="bielliptic",
        r_inner=min(r1, r2),
        r_outer=max(r1, r2),
        r_intermediate=r_b,
    )


def bielliptic_breakeven_ratio() -> float:
    """The classic radius ratio (r2/r1) above which bi-elliptic *can* beat
    Hohmann, in the limiting case of an infinite intermediate radius."""
    return 11.93876  # standard textbook value


# ==============================================================================
# === LAMBERT SOLVER ===
# ==============================================================================
@dataclass
class LambertSolution:
    """Velocity vectors solving a Lambert boundary-value problem."""
    v1: NDArray[np.float64]   # required velocity at r1 [km/s]
    v2: NDArray[np.float64]   # arrival velocity at r2 [km/s]
    tof: float                # time of flight [s]
    iterations: int           # solver iterations used


def lambert(r1_vec: NDArray[np.float64],
            r2_vec: NDArray[np.float64],
            tof: float,
            mu: float = None,
            prograde: bool = True,
            tol: float = 1e-8,
            max_iter: int = 100) -> LambertSolution:
    """Solve Lambert's problem with a universal-variable / Stumpff formulation.

    Given two position vectors and a time of flight, find the transfer orbit
    (and thus the velocity vectors at each end) connecting them. This is the
    core routine behind interplanetary targeting, intercept, and rendezvous.

    Parameters
    ----------
    r1_vec, r2_vec : ndarray (3,)
        Initial and final position vectors [km].
    tof : float
        Time of flight [s] (> 0).
    mu : float
        Gravitational parameter. Defaults to Earth.
    prograde : bool
        Choose the prograde (True) or retrograde (False) transfer arc.
    tol : float
        Convergence tolerance on the universal variable iteration.
    max_iter : int
        Iteration cap before raising.

    Returns
    -------
    LambertSolution

    Raises
    ------
    ValueError
        On degenerate geometry or bad inputs.
    RuntimeError
        If the iteration fails to converge.

    Notes
    -----
    Implements the algorithm in Curtis, *Orbital Mechanics for Engineering
    Students*, §5.3 — robust for the single-revolution case.
    """
    mu = mu if mu is not None else EARTH.mu
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)
    if tof <= 0:
        raise ValueError("Time of flight must be positive.")

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    if r1 < 1e-9 or r2 < 1e-9:
        raise ValueError("Position vectors must be non-zero.")

    # Change in true anomaly from the cross product's z-component.
    cross = np.cross(r1_vec, r2_vec)
    dtheta = np.arccos(np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0))
    if prograde:
        if cross[2] < 0:
            dtheta = 2 * np.pi - dtheta
    else:
        if cross[2] >= 0:
            dtheta = 2 * np.pi - dtheta

    A = np.sin(dtheta) * np.sqrt(r1 * r2 / (1.0 - np.cos(dtheta)))
    if abs(A) < 1e-12:
        raise ValueError("Degenerate transfer geometry (A ~ 0).")

    # Stumpff functions C(z) and S(z).
    def stumpff_C(z):
        if z > 1e-6:
            return (1.0 - np.cos(np.sqrt(z))) / z
        if z < -1e-6:
            return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
        return 0.5

    def stumpff_S(z):
        if z > 1e-6:
            sz = np.sqrt(z)
            return (sz - np.sin(sz)) / sz**3
        if z < -1e-6:
            sz = np.sqrt(-z)
            return (np.sinh(sz) - sz) / sz**3
        return 1.0 / 6.0

    def y_of_z(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        return r1 + r2 + A * (z * S - 1.0) / np.sqrt(C)

    # Solve F(z) = 0 for the universal variable z via Newton's method.
    def F(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        y = y_of_z(z)
        return (y / C) ** 1.5 * S + A * np.sqrt(y) - np.sqrt(mu) * tof

    def dFdz(z):
        if abs(z) < 1e-6:
            y0 = y_of_z(0.0)
            return (np.sqrt(2) / 40.0) * y0**1.5 + \
                   (A / 8.0) * (np.sqrt(y0) + A * np.sqrt(1.0 / (2.0 * y0)))
        C = stumpff_C(z)
        S = stumpff_S(z)
        y = y_of_z(z)
        term1 = (y / C) ** 1.5 * (
            (1.0 / (2.0 * z)) * (C - 1.5 * S / C) + 0.75 * S**2 / C)
        term2 = (A / 8.0) * (3.0 * S / C * np.sqrt(y) + A * np.sqrt(C / y))
        return term1 + term2

    # Bracket the root: start near zero and iterate.
    z = 0.0
    iters = 0
    for iters in range(1, max_iter + 1):
        Fz = F(z)
        if abs(Fz) < tol:
            break
        dz = Fz / dFdz(z)
        z -= dz
        if not np.isfinite(z):
            raise RuntimeError("Lambert iteration diverged (non-finite z).")
    else:
        raise RuntimeError("Lambert solver failed to converge within max_iter.")

    y = y_of_z(z)
    # Lagrange coefficients.
    f = 1.0 - y / r1
    g = A * np.sqrt(y / mu)
    gdot = 1.0 - y / r2

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (gdot * r2_vec - r1_vec) / g

    return LambertSolution(v1=v1, v2=v2, tof=tof, iterations=iters)


# ==============================================================================
# === CONVENIENCE: choose the cheaper of Hohmann vs bi-elliptic ===
# ==============================================================================
def best_transfer(r1: float, r2: float,
                  r_b: Optional[float] = None,
                  mu: float = None) -> TransferResult:
    """Return whichever of Hohmann / bi-elliptic needs less total delta-v.

    If ``r_b`` is not given, a sensible intermediate radius of 1.2*max(r1, r2)
    is tried (bi-elliptic only helps at large ratios, so this is conservative).
    """
    mu = mu if mu is not None else EARTH.mu
    hoh = hohmann_transfer(r1, r2, mu)
    rb = r_b if r_b is not None else 1.2 * max(r1, r2)
    try:
        bie = bielliptic_transfer(r1, r2, rb, mu)
    except ValueError:
        return hoh
    return hoh if hoh.dv_total <= bie.dv_total else bie


# ==============================================================================
# === DEMO ===
# ==============================================================================
def run_maneuver_demo():
    """Print worked examples for the three maneuver types."""
    print("=" * 60)
    print("ORBITAL MANEUVERS — worked examples")
    print("=" * 60)

    # LEO (400 km) -> GEO (35786 km) Hohmann.
    r_leo = EARTH.radius + 400.0
    r_geo = EARTH.radius + 35786.0
    hoh = hohmann_transfer(r_leo, r_geo)
    print("\nLEO (400 km) -> GEO:")
    print(hoh.summary())

    # Same endpoints, bi-elliptic via a far intermediate point.
    bie = bielliptic_transfer(r_leo, r_geo, r_b=EARTH.radius + 250000.0)
    print("\nSame transfer, bi-elliptic via 250,000 km:")
    print(bie.summary())

    # Lambert: connect two points 90 deg apart in a circular LEO over 40 min.
    r1 = np.array([r_leo, 0.0, 0.0])
    r2 = np.array([0.0, r_leo, 0.0])
    sol = lambert(r1, r2, tof=40 * 60.0)
    print(f"\nLambert (90 deg, 40 min): converged in {sol.iterations} iters")
    print(f"  |v1| = {np.linalg.norm(sol.v1):.4f} km/s, "
          f"|v2| = {np.linalg.norm(sol.v2):.4f} km/s")


if __name__ == "__main__":
    run_maneuver_demo()
