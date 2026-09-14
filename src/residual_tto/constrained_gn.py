"""Two-ball convex GN, FP64; no SciPy and no model/GT dependencies.

Projection onto span(J.T, a) preserves Jd and decreases BOTH ||d|| and
||a+d||. Its dimension is <= 3 for two focal residuals. KKT suffices for
global optimality of this convex subproblem. We first handle the possibly
non-unique least-squares optimum, choosing the smallest feasible step norm.
"""
import numpy as np


class SolverError(RuntimeError):
    pass


def _norm(x):
    return float(np.linalg.norm(x))


def _ball_quadratic(H, g, a, radius, lam):
    """For fixed step multiplier, minimize quadratic inside cumulative ball."""
    eig, U = np.linalg.eigh(H)
    eig = np.maximum(eig, 0.)
    # x=a+d; minimize x'(H+lam I)x + 2(g-(H+lam I)a)'x.
    e = eig + lam
    k = U.T @ (g - (H + lam*np.eye(len(a))) @ a)
    cutoff = max(float(e.max())*1e-14, 1e-300)
    xcoords = np.divide(-k, e, out=np.zeros_like(k), where=e > cutoff)
    if _norm(xcoords) <= radius:
        return U @ xcoords - a, 0.
    def candidate(mu):
        return -k/(e+mu)
    lo, hi = 0., max(float(e.max()), _norm(k)/radius, 1e-30)
    for _ in range(100):
        if _norm(candidate(hi)) <= radius:
            break
        hi *= 2
    else:
        raise SolverError("cumulative multiplier could not be bracketed")
    for _ in range(90):
        mid = (lo+hi)/2
        if _norm(candidate(mid)) > radius:
            lo = mid
        else:
            hi = mid
    return U @ candidate(hi) - a, float(hi)


def certificate(J, r, a, delta, step_radius, budget, lam, mu):
    jd = J @ delta
    g = J.T @ r
    terms = [g, J.T@jd, lam*delta, mu*(a+delta)]
    stationarity = sum(terms)
    denom = max(sum(_norm(t) for t in terms), 1e-30)
    feasibility = max(0., _norm(delta)/step_radius-1, _norm(a+delta)/budget-1)
    dual_scale = max(float(r@r), 1e-30)
    complementarity = max(abs(lam*(_norm(delta)**2-step_radius**2)),
                          abs(mu*(_norm(a+delta)**2-budget**2)))/dual_scale
    return dict(
        feasibility_residual=feasibility,
        stationarity_absolute=_norm(stationarity),
        stationarity_relative=_norm(stationarity)/denom,
        complementarity_relative=complementarity,
        step_active=bool(abs(_norm(delta)/step_radius-1) <= 1e-6),
        cumulative_active=bool(abs(_norm(a+delta)/budget-1) <= 1e-6),
        passed=bool(feasibility <= 1e-8 and
                    (_norm(stationarity) <= 1e-12 or _norm(stationarity)/denom <= 1e-6)
                    and complementarity <= 1e-7 and lam >= 0 and mu >= 0))


def solve_two_ball(J, r, a, step_radius, budget):
    """Physical Euclidean radii (NOT per-element RMS). Returns full-space d."""
    J, r, a = [np.asarray(x, dtype=np.float64) for x in (J, r, a)]
    if J.ndim != 2 or J.shape[0] != 2 or r.shape != (2,) or a.shape != (J.shape[1],):
        raise SolverError("this frozen experiment requires exactly two focal rows")
    if not all(np.isfinite(x).all() for x in (J, r, a)):
        raise SolverError("nonfinite solver input")
    if min(step_radius, budget) <= 0 or _norm(a) > budget*(1+1e-10):
        raise SolverError("invalid radius or infeasible current state")
    U, singular, Vt = np.linalg.svd(J, full_matrices=False)
    rank = int(np.sum(singular > max(float(singular[0])*1e-12, 1e-30)))
    rows = Vt[:rank].T
    ar = rows.T @ a
    anull = a - rows@ar
    # Unconstrained least squares: minimum feasible norm among all minimizers.
    drow = -(U[:, :rank].T@r)/singular[:rank] if rank else np.empty(0)
    row_cum2 = _norm(ar+drow)**2
    if row_cum2 <= budget**2:
        allowance = np.sqrt(max(budget**2-row_cum2, 0.))
        dnull = -anull*max(0., 1-allowance/max(_norm(anull), 1e-300))
        d = rows@drow+dnull
        if _norm(d) <= step_radius and _norm(a+d) <= budget*(1+1e-12):
            cert = certificate(J,r,a,d,step_radius,budget,0.,0.)
            if not cert["passed"]:
                raise SolverError("least-squares certificate failed")
            return d, dict(lambda_step=0.,mu_cumulative=0.,rank=rank,
                subspace_dim=rank+int(_norm(anull)>1e-12*max(_norm(a),1e-30)),
                solution_kind="minimum_step_norm_least_squares",kkt=cert)
    Q = rows
    if _norm(anull) > 1e-12*max(_norm(a), 1e-30):
        Q = np.column_stack((Q, anull/_norm(anull)))
    if Q.shape[1] == 0:
        raise SolverError("empty subspace not handled by least-squares case")
    jq, aq = J@Q, Q.T@a
    H, g = jq.T@jq, jq.T@r
    d, mu = _ball_quadratic(H,g,aq,budget,0.)
    lam = 0.
    if _norm(d) > step_radius:
        lo, hi = 0., max(float(np.linalg.norm(H,2)), _norm(g)/step_radius, 1e-30)
        for _ in range(100):
            d, mu = _ball_quadratic(H,g,aq,budget,hi)
            if _norm(d) <= step_radius:
                break
            hi *= 2
        else:
            raise SolverError("step multiplier could not be bracketed")
        for _ in range(80):
            mid = (lo+hi)/2
            test, test_mu = _ball_quadratic(H,g,aq,budget,mid)
            if _norm(test) > step_radius:
                lo = mid
            else:
                hi, d, mu = mid, test, test_mu
        lam = float(hi)
        d, mu = _ball_quadratic(H,g,aq,budget,lam)
    full = Q@d
    cert = certificate(J,r,a,full,step_radius,budget,lam,mu)
    if not cert["passed"]:
        raise SolverError("FP64 KKT validation failed: "+str(cert))
    return full, dict(lambda_step=lam,mu_cumulative=mu,rank=rank,
        subspace_dim=int(Q.shape[1]),solution_kind="two_ball_kkt",kkt=cert)


def current_stationarity(J, r, a, budget):
    """Approximate first-order certificate at CURRENT nonlinear state."""
    g = np.asarray(J,dtype=np.float64).T@np.asarray(r,dtype=np.float64)
    a = np.asarray(a,dtype=np.float64)
    active = abs(_norm(a)/budget-1) <= 2e-7
    mu = max(0., -float(g@a)/max(float(a@a),1e-300)) if active else 0.
    residual = _norm(g+mu*a)
    return dict(passed=bool(residual <= 1e-10+1e-7*_norm(g)),
        gradient_norm=_norm(g),stationarity_absolute=residual,mu_cumulative=mu,
        cumulative_active=bool(active),absolute_tolerance=1e-10,relative_tolerance=1e-7)
