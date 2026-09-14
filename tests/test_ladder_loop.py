import unittest
from unittest.mock import patch
import numpy as np
from residual_tto.ladder_optimizer import optimize,feasible_storage
from residual_tto.constrained_gn import SolverError

class LoopTests(unittest.TestCase):
    def setup_problem(self):
        self.J=np.array([[1.,0,0],[0,1,0]])
        self.options=dict(initial_step_relative=.05,minimum_step_relative=.01,
            maximum_step_relative=.25,max_accepted_steps=40,max_jacobians=40,
            max_candidates_per_jacobian=8,max_candidate_forwards=160,
            target_mean_focal_relative_error=.15,cast_retreat_powers=[23,22,21,20,19,18,17])
        self.calls=dict(forward=0,jacobian=0)
        def data(r):
            return dict(r=r,pose_enc=[],diagnostics=dict(
                supervised_mean_relative_error=float(np.mean(np.abs(np.expm1(r))))))
        self.data=data
        def forward(a):
            self.calls["forward"]+=1
            return data(np.array([.2,.2])+self.J@a)
        def jacobian(a):
            self.calls["jacobian"]+=1
            return self.J,data(np.array([.2,.2])+self.J@a)
        self.fwd=forward;self.jac=jacobian
        self.initial=data(np.array([.2,.2]))
    def run_problem(self,budget=10.):
        return optimize(self.initial,self.J,1.,budget,self.options,0.,self.fwd,self.jac)
    def test_target_before_next_jacobian(self):
        self.setup_problem()
        result=self.run_problem()
        self.assertEqual(result["termination_reason"],"TARGET_REACHED")
        self.assertEqual(result["counters"]["jacobians"],result["counters"]["accepted_steps"])
        self.assertEqual(self.calls["jacobian"],result["counters"]["jacobians"]-1)
        self.assertLessEqual(result["data"]["diagnostics"]["supervised_mean_relative_error"],.15)
    def test_zero_start_already_at_target(self):
        self.setup_problem();self.initial=self.data(np.array([.01,.01]))
        result=self.run_problem()
        self.assertEqual(result["termination_reason"],"TARGET_REACHED")
        self.assertEqual(self.calls,dict(forward=0,jacobian=0))
    def test_limit_counts_initial_j(self):
        self.setup_problem();self.options["max_jacobians"]=1
        result=self.run_problem()
        self.assertEqual(result["termination_reason"],"COMPUTE_LIMIT")
        self.assertEqual(result["termination_detail"],"max_jacobians")
        self.assertEqual(self.calls["jacobian"],0)
    def test_min_radius_not_double_halved(self):
        self.setup_problem();self.options["initial_step_relative"]=.01
        self.fwd=lambda a:self.data(np.array([.2,.2])-self.J@a)
        result=self.run_problem()
        self.assertEqual(result["termination_reason"],"TR_RADIUS_MIN")
        self.assertEqual(result["counters"]["accepted_steps"],0)
        np.testing.assert_array_equal(result["a"],0.)
    def test_candidate_forward_limit(self):
        self.setup_problem();self.options["max_candidate_forwards"]=1
        self.fwd=lambda a:self.data(np.array([.2,.2])-self.J@a)
        result=self.run_problem()
        self.assertEqual(result["termination_detail"],"max_candidate_forwards")
        self.assertEqual(result["counters"]["candidate_forwards"],1)
    def test_candidate_generation_limit_not_stationarity(self):
        self.setup_problem();self.options["max_candidates_per_jacobian"]=1
        self.fwd=lambda a:self.data(np.array([.2,.2])-self.J@a)
        result=self.run_problem()
        self.assertEqual(result["termination_reason"],"COMPUTE_LIMIT")
        self.assertEqual(result["termination_detail"],"max_candidates_per_jacobian")
    def test_solver_failure_not_stationarity(self):
        self.setup_problem()
        with patch("residual_tto.ladder_optimizer.solve_two_ball",side_effect=SolverError("test")):
            result=self.run_problem()
        self.assertEqual(result["termination_reason"],"SOLVER_FAILURE")
        self.assertEqual(self.calls["forward"],0)
    def test_noise_floor_not_stationarity(self):
        self.setup_problem();self.J*=1e-8
        result=self.run_problem()
        self.assertEqual(result["termination_reason"],"NUMERICAL_FLOOR")
        self.assertEqual(self.calls["forward"],0)
    def test_cumulative_limit_never_exceeded(self):
        self.setup_problem()
        result=self.run_problem(.06)
        self.assertLessEqual(result["cumulative_relative"],.06)
        for event in result["history"]:
            if event["accepted"]:
                self.assertLessEqual(event["cumulative_relative"],.06)
                self.assertLessEqual(event["step_relative"],event["radius_relative"])
        self.assertIn(result["termination_reason"],["CONSTRAINED_STATIONARY","NUMERICAL_FLOOR"])
    def test_repeated_experiments_start_from_zero(self):
        self.setup_problem();first=self.run_problem(.06)
        self.setup_problem();second=self.run_problem(.2)
        self.assertEqual(first["history"][0]["current_cumulative_relative"],0.)
        self.assertEqual(second["history"][0]["current_cumulative_relative"],0.)
    def test_realistic_numpy_scale_json_serialization(self):
        import json
        self.setup_problem()
        result=optimize(self.initial,self.J,np.float64(1.),.06,self.options,0.,self.fwd,self.jac)
        json.dumps(result["history"],allow_nan=False)
        self.assertEqual(result["termination_reason"],"CONSTRAINED_STATIONARY")

    def test_evaluator_barrier_before_gt_read(self):
        import tempfile
        from pathlib import Path
        import residual_tto.evaluate as evaluation
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/"study_manifest.json").write_text("{}")
            with patch("sys.argv",["evaluate","--run",str(p)]), \
                 patch("residual_tto.study.assert_study_frozen",side_effect=RuntimeError("not frozen")), \
                 patch.object(evaluation,"read_json") as read:
                with self.assertRaisesRegex(RuntimeError,"not frozen"): evaluation.main()
                read.assert_not_called()

    def test_rounding_retreat_no_model_line_search(self):
        c,e,info=feasible_storage(np.zeros(1,dtype=np.float32),np.array([.100000006]),
                                 .100000006,1.,[23,22,21,20,19,18,17])
        self.assertTrue(info["legal"]);self.assertGreater(info["cast_retreat_index"],0)
        self.assertLessEqual(np.linalg.norm(e),.100000006)
        self.assertGreater(info["cast_retained_fraction"],.99999)

if __name__=="__main__": unittest.main()
