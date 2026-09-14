import tempfile
import unittest
from pathlib import Path
import numpy as np
from residual_tto.constrained_gn import solve_two_ball,current_stationarity
from residual_tto.trust_policy import accepted,next_radius,cast_candidate,noise_floor
from residual_tto.io import write_json


class TwoBallTests(unittest.TestCase):
    def test_unconstrained_minimum_norm(self):
        J=np.array([[2.,0,0],[0,1,0]])
        d,info=solve_two_ball(J,np.array([1.,2.]),np.zeros(3),10.,10.)
        np.testing.assert_allclose(d,[-.5,-2,0],atol=1e-12)
        self.assertEqual(info["lambda_step"],0.)
        self.assertTrue(info["kkt"]["passed"])

    def test_nonunique_solution_minimum_feasible_step(self):
        J=np.array([[1.,0,0],[1.,0,0]])
        a=np.array([0.,0.,1.])
        # Exact row correction x=0.5 leaves room sqrt(.75) for a_z+d_z.
        d,info=solve_two_ball(J,np.array([-.5,-.5]),a,2.,1.)
        np.testing.assert_allclose(d,[.5,0,np.sqrt(.75)-1],atol=1e-11)
        self.assertEqual(info["solution_kind"],"minimum_step_norm_least_squares")

    def test_step_only_and_cumulative_only(self):
        J=np.eye(2)
        d,s=solve_two_ball(J,np.array([-2.,0]),np.zeros(2),.1,1.)
        np.testing.assert_allclose(d,[.1,0],atol=1e-10)
        self.assertGreater(s["lambda_step"],0);self.assertEqual(s["mu_cumulative"],0)
        d,s=solve_two_ball(J,np.array([-2.,0]),np.zeros(2),1.,.2)
        np.testing.assert_allclose(d,[.2,0],atol=1e-10)
        self.assertEqual(s["lambda_step"],0);self.assertGreater(s["mu_cumulative"],0)

    def test_boundary_tangential_step_not_false_stationary(self):
        J=np.eye(2);r=np.array([-1.,-1.]);a=np.array([1.,0])
        d,s=solve_two_ball(J,r,a,.2,1.)
        self.assertGreater(s["lambda_step"],0);self.assertGreater(s["mu_cumulative"],0)
        self.assertLess(d[0],0);self.assertGreater(d[1],0)
        self.assertLess(np.linalg.norm(r+J@d),np.linalg.norm(r))
        self.assertFalse(current_stationarity(J,r,a,1.)["passed"])
        self.assertTrue(s["kkt"]["passed"])

    def test_stationary_boundary_is_certified(self):
        J=np.eye(2);r=np.array([-1.,0]);a=np.array([1.,0])
        d,s=solve_two_ball(J,r,a,.2,1.)
        np.testing.assert_allclose(d,0.,atol=1e-10)
        self.assertTrue(current_stationarity(J,r,a,1.)["passed"])

    def test_rank_zero(self):
        d,s=solve_two_ball(np.zeros((2,5)),np.ones(2),np.ones(5)*.1,.2,1.)
        np.testing.assert_allclose(d,0.)
        self.assertEqual(s["rank"],0)

    def test_random_kkt_global_optimality(self):
        rng=np.random.default_rng(20)
        for i in range(80):
            J=rng.normal(size=(2,9));r=rng.normal(size=2)
            a=rng.normal(size=9);a=a/np.linalg.norm(a)*(1 if i%2 else .5)
            d,s=solve_two_ball(J,r,a,.05+float(rng.uniform(0,.8)),1.)
            self.assertTrue(s["kkt"]["passed"])
            # Independent feasible random sample cannot beat convex certified solution.
            trials=rng.normal(size=(200,9));trials/=np.maximum(np.linalg.norm(trials,axis=1,keepdims=True),1e-10)
            trials*=rng.uniform(0,.05,size=(200,1))
            feasible=np.linalg.norm(a+trials,axis=1)<=1.
            if feasible.any():
                self.assertLessEqual(np.sum((r+J@d)**2),
                    np.min(np.sum((r+trials[feasible]@J.T)**2,axis=1))+1e-9)

    def test_joint_rotation_equivariance(self):
        rng=np.random.default_rng(4);J=rng.normal(size=(2,5));r=rng.normal(size=2)
        Q,_=np.linalg.qr(rng.normal(size=(5,5)));a=np.array([.7,0,0,0,0])
        d,_=solve_two_ball(J,r,a,.3,1.)
        rotated,_=solve_two_ball(J@Q,r,Q.T@a,.3,1.)
        np.testing.assert_allclose(d,Q@rotated,atol=1e-9)

    def test_float32_storage_checked_and_used(self):
        a=np.array([.12345679],dtype=np.float32);d=np.array([1e-9])
        c,e,check=cast_candidate(a,d,1.,1.)
        np.testing.assert_array_equal(e,c.astype(np.float64)-a.astype(np.float64))
        self.assertNotEqual(float(e[0]),float(d[0]))
        self.assertFalse(cast_candidate(np.zeros(1),np.array([.100000006]),.100000006,1.)[2]["legal"])


class PolicyTests(unittest.TestCase):
    def test_accept_01_to_025_but_shrink_once(self):
        for rho in [.1,.15,.249999]:
            self.assertTrue(accepted(1.,rho,rho,1e-7))
            radius,streak,action=next_radius(.001,.00001,.0025,True,rho,1.,1)
            self.assertEqual(radius,.0005);self.assertEqual(streak,0);self.assertEqual(action,"halve")
        self.assertFalse(accepted(1.,.099,.099,1e-7))
        self.assertFalse(accepted(1.,1.,1.,1e-7,legal=False))
        self.assertFalse(accepted(1.,0.,0.,1e-7))

    def test_two_step_growth_and_streak_reset(self):
        r,s,_=next_radius(.0005,.00001,.0025,True,1.,.9,0)
        self.assertEqual((r,s),(.0005,1))
        r,s,_=next_radius(r,.00001,.0025,True,1.,.9,s)
        self.assertEqual((r,s),(.001,0))
        self.assertEqual(next_radius(r,.00001,.0025,False,.01,1.,1)[:2],(.0005,0))
        self.assertEqual(next_radius(r,.00001,.0025,True,1.3,1.,1)[:2],(r,0))
        self.assertEqual(next_radius(r,.00001,.0025,True,.25,1.,1)[:2],(r,0))

    def test_floor_definitions(self):
        self.assertAlmostEqual(noise_floor(.1,0.),1e-6)
        self.assertAlmostEqual(noise_floor(0.,1e-6),1e-5)

    def test_study_barrier_blocks_missing_and_changed_endpoint(self):
        from residual_tto.study import assert_study_frozen, freeze_endpoint
        from residual_tto.io import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=[]
            for i in range(4):
                p=root/str(i);p.mkdir();runs.append(p)
                write_json(p/"resolved_config.json",dict(cumulative_budget_relative=[.003,.01,.02,.04][i]))
            study=dict(schema_version=1,study_id="test",runs=[
                dict(path=str(p),config_sha256=sha256(p/"resolved_config.json")) for p in runs])
            for p in runs: write_json(p/"study_manifest.json",study)
            with self.assertRaises((RuntimeError,FileNotFoundError)): assert_study_frozen(runs[0])
            for p in runs:
                (p/"adapted/post14_D02_registers").mkdir(parents=True)
                (p/"logs").mkdir()
                for name in ["residual.pt","predictions.npz","optimization.json"]:
                    (p/"adapted/post14_D02_registers"/name).write_bytes(b"test")
                (p/"baseline").mkdir();(p/"baseline/predictions.npz").write_bytes(b"base")
                freeze_endpoint(p,"TARGET_REACHED",dict(accepted_steps=1),{})
            self.assertEqual(len(assert_study_frozen(runs[0])["endpoints"]),4)
            (runs[3]/"adapted/post14_D02_registers/residual.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError): assert_study_frozen(runs[0])


if __name__=="__main__": unittest.main()
