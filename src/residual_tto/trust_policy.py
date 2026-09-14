"""Frozen focal-only trust-region rules, independent of model and geometry GT."""
import numpy as np

TERMINATIONS = frozenset(("TARGET_REACHED","NUMERICAL_FLOOR","TR_RADIUS_MIN",
                         "SOLVER_FAILURE","COMPUTE_LIMIT","CONSTRAINED_STATIONARY"))


def loss(r):
    r=np.asarray(r,dtype=np.float64)
    return float(r@r)


def noise_floor(current_loss, epsilon_repeat):
    return max(1e-7,1e-5*current_loss,10*epsilon_repeat)


def accepted(predicted_decrease, actual_decrease, rho, tau, legal=True):
    # 0.1 is acceptance; 0.25 below is ONLY the radius-shrink threshold.
    return bool(legal and np.isfinite([predicted_decrease,actual_decrease,rho]).all()
                and predicted_decrease>tau and actual_decrease>tau and rho>=0.1)


def next_radius(radius, minimum, maximum, accept, rho, utilization, streak):
    if not accept or rho<0.25:
        return max(minimum,radius/2),0,"halve"
    qualifies = 0.75<=rho<=1.25 and utilization>=0.8
    streak=streak+1 if qualifies else 0
    if streak>=2:
        return min(maximum,radius*2),0,"grow_after_two"
    return radius,streak,"keep"


def cast_candidate(a, delta, step_radius, budget):
    """Recheck actual FP32 storage, use its FP64 difference for prediction.

    No projection/clipping is hidden here. A rounding-infeasible candidate
    receives no model forward and is retried at a smaller step radius.
    """
    a32=np.asarray(a,dtype=np.float32)
    candidate=(a32.astype(np.float64)+np.asarray(delta,dtype=np.float64)).astype(np.float32)
    effective=candidate.astype(np.float64)-a32.astype(np.float64)
    sn=float(np.linalg.norm(effective));cn=float(np.linalg.norm(candidate.astype(np.float64)))
    legal=bool(np.isfinite(candidate).all() and sn<=step_radius and cn<=budget)
    return candidate,effective,dict(legal=legal,step_l2=sn,cumulative_l2=cn,
        casting_delta_l2=float(np.linalg.norm(effective-delta)))


def spectral(J,r):
    U,s,_=np.linalg.svd(np.asarray(J,dtype=np.float64),full_matrices=False)
    energy=(U.T@r)**2
    total=float(np.asarray(r,dtype=np.float64)@np.asarray(r,dtype=np.float64))
    return dict(singular_values=s.tolist(),left_singular_residual_energy=energy.tolist(),
                left_singular_residual_energy_fraction=(energy/max(total,1e-300)).tolist())


def focal_diagnostics(xy,target,supervised=4):
    xy=np.asarray(xy,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    err=np.abs(xy/target-1)
    return dict(focal_xy=xy.tolist(),relative_error_xy=err.tolist(),
                supervised_mean_relative_error=float(err[supervised].mean()))
