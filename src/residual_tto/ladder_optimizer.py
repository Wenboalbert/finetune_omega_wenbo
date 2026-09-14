"""Model-independent focal-only loop; accepts callbacks, never imports GT."""
import numpy as np
from .constrained_gn import solve_two_ball,SolverError,current_stationarity
from .trust_policy import loss,noise_floor,accepted,next_radius,cast_candidate,spectral

def feasible_storage(a,delta,step_radius,budget,powers):
    # Tiny deterministic segment retreat addresses FP32 rounding ONLY.
    # It is not a new line search: there are no extra model forwards or losses.
    for index,fraction in enumerate([1.]+[1.-2.**(-p) for p in powers]):
        c,e,check=cast_candidate(a,delta*fraction,step_radius,budget)
        check.update(cast_retreat_index=index,cast_retained_fraction=fraction)
        if check["legal"]: return c,e,check
    return c,e,check


def optimize(initial, initial_jacobian, norm_scale, budget_relative, options,
             epsilon_repeat, forward_fn, jacobian_fn, event_fn=lambda x:None,
             jacobian_event_fn=lambda *x:None, accepted_event_fn=lambda *x:None):
    """All norm_scale = sqrt(number of residual variables) * fixed RMS(h0).

    initial and forward_fn results contain r, pose/focal diagnostics, NO geometry
    ground truth. Initial preflight J counts as J1 and is reused, not recomputed.
    """
    a=np.zeros(initial_jacobian.shape[1],dtype=np.float32)
    data=initial;J=np.asarray(initial_jacobian,dtype=np.float64)
    radius=options["initial_step_relative"];minimum=options["minimum_step_relative"]
    maximum=options["maximum_step_relative"];budget=norm_scale*budget_relative
    counters=dict(accepted_steps=0,jacobians=1,candidates_generated=0,candidate_forwards=0)
    reason=None;detail=None;history=[];streak=0
    while reason is None:
        # Target tested BEFORE another Jacobian or candidate (including zero start).
        if data["diagnostics"]["supervised_mean_relative_error"]<=options["target_mean_focal_relative_error"]:
            reason,detail="TARGET_REACHED","supervised mean absolute focal relative error <= target";break
        if counters["accepted_steps"]>=options["max_accepted_steps"]:
            reason,detail="COMPUTE_LIMIT","max_accepted_steps";break
        if J is None:
            if counters["jacobians"]>=options["max_jacobians"]:
                reason,detail="COMPUTE_LIMIT","max_jacobians";break
            J,data=jacobian_fn(a)
            J=np.asarray(J,dtype=np.float64);counters["jacobians"]+=1
        r=np.asarray(data["r"],dtype=np.float64);L=loss(r)
        tau=noise_floor(L,epsilon_repeat);spec=spectral(J,r)
        jacobian_event_fn(counters["jacobians"],J,r,a,spec)
        accepted_this_j=False
        for local in range(options["max_candidates_per_jacobian"]):
            if counters["candidate_forwards"]>=options["max_candidate_forwards"]:
                reason,detail="COMPUTE_LIMIT","max_candidate_forwards";break
            counters["candidates_generated"]+=1
            entry=dict(candidate=counters["candidates_generated"],jacobian=counters["jacobians"],
                local_candidate=local+1,radius_relative=radius,current_loss=L,tau=tau,
                current_cumulative_relative=float(np.linalg.norm(a.astype(np.float64))/norm_scale),
                accepted=False,spectral=spec,lambda_step=None,mu_cumulative=None,
                predicted_decrease=None,actual_decrease=None,rho=None,termination_reason=None)
            try:
                delta,solver=solve_two_ball(J,r,a,norm_scale*radius,budget)
            except (SolverError,np.linalg.LinAlgError,FloatingPointError) as error:
                reason,detail="SOLVER_FAILURE",str(error)
                entry.update(solver_error=str(error),termination_reason=reason)
                history.append(entry);event_fn(entry);break
            entry.update(solver=solver,lambda_step=solver["lambda_step"],mu_cumulative=solver["mu_cumulative"])
            candidate,effective,casting=feasible_storage(a,delta,norm_scale*radius,budget,
                                                         options["cast_retreat_powers"])
            pred=L-loss(r+J@effective)
            entry.update(casting=casting,step_relative=casting["step_l2"]/norm_scale,
                cumulative_relative=casting["cumulative_l2"]/norm_scale,predicted_decrease=pred)
            legal=casting["legal"];rho=None;accept=False;candidate_data=None
            if not np.isfinite(pred):
                reason,detail="SOLVER_FAILURE","nonfinite effective linear prediction"
                entry.update(predicted_decrease=None,termination_reason=reason)
                history.append(entry);event_fn(entry);break
            cert=current_stationarity(J,r,a,budget) if pred<=tau else None
            # Certify the CURRENT feasible state, even if the next rounded
            # candidate is unrepresentably outside the ball. Infeasibility
            # alone never implies stationarity or a numerical floor.
            if pred<=tau and (legal or cert["passed"]):
                reason="CONSTRAINED_STATIONARY" if cert["passed"] else "NUMERICAL_FLOOR"
                detail="certified current first-order stationarity" if cert["passed"] else "predicted decrease does not exceed frozen focal noise floor"
                entry.update(current_stationarity=cert,termination_reason=reason)
                history.append(entry);event_fn(entry);break
            if legal:
                counters["candidate_forwards"]+=1
                try:
                    candidate_data=forward_fn(candidate)
                    actual=L-loss(candidate_data["r"])
                    rho=actual/pred
                    accept=accepted(pred,actual,rho,tau,legal)
                    entry.update(actual_decrease=actual,rho=rho,candidate_loss=loss(candidate_data["r"]),
                                 diagnostics=candidate_data["diagnostics"],
                                 pose_enc=candidate_data["pose_enc"])
                except FloatingPointError as error:
                    entry["forward_numerical_error"]=str(error)
            utilization=casting["step_l2"]/(norm_scale*radius)
            new_radius,new_streak,action=next_radius(radius,minimum,maximum,accept,
                rho if rho is not None else -1.,utilization,streak)
            entry.update(accepted=accept,next_radius_relative=new_radius,
                         radius_action=action,growth_streak=new_streak,
                         counters=dict(counters))
            if accept:
                a=candidate.copy();data=candidate_data;counters["accepted_steps"]+=1
                accepted_this_j=True
                entry["accepted_step"]=counters["accepted_steps"]
                entry["counters"]=dict(counters)
                accepted_event_fn(counters["accepted_steps"],a,data)
                if data["diagnostics"]["supervised_mean_relative_error"]<=options["target_mean_focal_relative_error"]:
                    reason,detail="TARGET_REACHED","supervised mean absolute focal relative error <= target"
                elif counters["accepted_steps"]>=options["max_accepted_steps"]:
                    reason,detail="COMPUTE_LIMIT","max_accepted_steps"
                elif counters["jacobians"]>=options["max_jacobians"]:
                    reason,detail="COMPUTE_LIMIT","max_jacobians"
            elif radius<=minimum*(1+1e-12):
                reason,detail="TR_RADIUS_MIN","candidate rejected at minimum radius"
            entry["termination_reason"]=reason
            history.append(entry);event_fn(entry)
            radius,streak=new_radius,new_streak
            if accept or reason is not None: break
        if reason is None and not accepted_this_j:
            reason,detail="COMPUTE_LIMIT","max_candidates_per_jacobian"
        if accepted_this_j: J=None
    return dict(a=a,data=data,termination_reason=reason,termination_detail=detail,
        counters=counters,history=history,final_radius_relative=radius,
        cumulative_relative=float(np.linalg.norm(a.astype(np.float64))/norm_scale))
