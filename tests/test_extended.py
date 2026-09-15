import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from residual_tto.io import read_json,write_json,sha256
from residual_tto.extended import (GROUP,NEAR_ZERO,reference_config,validate_config,verify_frozen,
                                   branch_decision,assert_ready,check_endpoint,OPT,DECISION)
from residual_tto.extended_prefix import audit_prefix
from residual_tto.ladder_optimizer import optimize
from residual_tto.study import freeze_endpoint,FROZEN_FILES

ROOT=Path(__file__).resolve().parents[1]

class ContractTests(unittest.TestCase):
    def test_only_frozen_compute_and_optional_budget_changes(self):
        for suffix in ("040","050"):
            c=read_json(ROOT/("configs/extended_frame40_b"+suffix+".json"))
            self.assertTrue(validate_config(c,reference_config(ROOT)))
            self.assertEqual(c["optimizer"]["target_mean_focal_relative_error"],0)
    def test_unapproved_changes_rejected(self):
        c=read_json(ROOT/"configs/extended_frame40_b040.json")
        for key,value in (("tau_absolute",1e-8),("max_candidates_per_jacobian",16),
                          ("initial_step_relative",.001),("target_mean_focal_relative_error",.001)):
            bad=copy.deepcopy(c);bad["optimizer"][key]=value
            with self.assertRaises(ValueError): validate_config(bad,reference_config(ROOT))
        bad=copy.deepcopy(c);bad["cumulative_budget_relative"]=.06
        with self.assertRaises(ValueError): validate_config(bad,reference_config(ROOT))
    def test_frozen_numerics_and_model_body(self):
        self.assertEqual(len(verify_frozen(ROOT)),14)

    def decision(self,errors,valid="PASS"):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            xy=[[1.,1.]]*4+[[1+errors[0],1+errors[1]]]
            write_json(run/OPT,dict(final_focal={"focal_xy":xy}))
            write_json(run/"inputs/focal_manifest.json",dict(camera_order=["CCTV_01","CCTV_02","CCTV_03","CCTV_04","Drone_02"],
                                                            focal_xy=[[1.,1.]]*5))
            write_json(run/"logs/ladder_completion.json",{})
            with patch("residual_tto.extended.check_endpoint"):
                return branch_decision(run,{"status":valid})
    def test_near_zero_both_axes(self):
        self.assertEqual(self.decision([.0009,.0009])["status"],"SKIPPED_NEAR_ZERO")
        self.assertEqual(self.decision([0.,.0015])["status"],"RUN_5_PERCENT") # mean alone would pass
        self.assertEqual(self.decision([0.,0.])["status"],"SKIPPED_NEAR_ZERO")
    def test_boundary_is_inclusive_using_unrounded_focal(self):
        self.assertEqual(self.decision([.001,.001])["status"],"SKIPPED_NEAR_ZERO")
        self.assertEqual(self.decision([.0010000001,0])["status"],"RUN_5_PERCENT")
    def test_nonfinite_and_invalid_optimization_do_not_expand(self):
        with self.assertRaises(ValueError): self.decision([.1,.1],"FAIL")
        with self.assertRaises(ValueError): self.decision([-2.,0.])

    def problem(self,limit):
        o=dict(read_json(ROOT/"configs/extended_frame40_b040.json")["optimizer"])
        o.update(max_accepted_steps=limit,max_jacobians=limit,max_candidate_forwards=limit*4)
        J=np.eye(2);start=np.ones(2)*10
        def data(a): return dict(r=start+J@a,pose_enc=[],diagnostics={"supervised_mean_relative_error":.0005})
        states=[]
        result=optimize(data(np.zeros(2)),J,1.,10.,o,0.,data,lambda a:(J,data(a)),
                        accepted_event_fn=lambda i,a,d:states.append(a.copy()))
        return result,states
    def test_80_steps_no_near_zero_optimizer_early_stop(self):
        result,states=self.problem(80)
        self.assertEqual(len(states),80)
        self.assertEqual(result["termination_reason"],"COMPUTE_LIMIT")
    def test_extended_prefix_and_independent_zero_starts(self):
        old,a=self.problem(40);new,b=self.problem(80)
        self.assertEqual(len(a),40);self.assertEqual(len(b),80)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        self.assertEqual(old["counters"]["jacobians"],40)
        self.assertEqual(new["counters"]["jacobians"],80)

class PrefixTests(unittest.TestCase):
    def fixture(self,p,steps):
        (p/"logs").mkdir(parents=True);(p/"baseline").mkdir()
        np.savez(p/"baseline/predictions.npz",pose_enc=[1.],depth=[1.],intrinsic=[1.],extrinsic=[1.])
        np.savez(p/"logs/preflight_jacobian.npz",J=np.eye(2),r=[1.,1.],h0=[1.,1.])
        for i in range(1,steps+1):
            np.savez(p/("logs/accepted_%03d.npz"%i),a=[i*.001,0],r=[1.,1.],pose_enc=[1.])
            np.savez(p/("logs/jacobian_%03d.npz"%i),J=np.eye(2),r=[1.,1.],a=[(i-1)*.001,0])
        obj=dict(h0_rms=1.,epsilon_repeat=0.,counters=dict(accepted_steps=steps,jacobians=steps),
                 termination_reason="COMPUTE_LIMIT",attempts=[
                     dict(candidate=i,jacobian=i,local_candidate=1,accepted=True,accepted_step=i,
                          radius_action="keep",growth_streak=0,current_loss=2.) for i in range(1,steps+1)])
        write_json(p/OPT,obj)
        return obj
    def test_expected_compute_stop_difference_not_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=Path(tmp)/"old",Path(tmp)/"new"
            self.fixture(old,40);self.fixture(new,80)
            self.assertEqual(audit_prefix(old,new)["status"],"PASS")
    def test_collect_numeric_and_decision_failures_without_throwing(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=Path(tmp)/"old",Path(tmp)/"new"
            self.fixture(old,2);self.fixture(new,3)
            np.savez(new/"logs/accepted_001.npz",a=[.1,0],r=[2.,1.],pose_enc=[1.])
            out=audit_prefix(old,new)
            self.assertEqual(out["status"],"FAIL");self.assertGreaterEqual(len(out["numeric_mismatches"]),2)
            self.assertFalse(out["is_geometry_gate"])
    def test_incomplete_prefix_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            old,new=Path(tmp)/"old",Path(tmp)/"new"
            self.fixture(old,3);self.fixture(new,2)
            self.assertEqual(audit_prefix(old,new)["status"],"INCOMPLETE")

class BarrierTests(unittest.TestCase):
    def fixture(self,root,decision_status="RUN_5_PERCENT",prefix_status="FAIL",second=True):
        paths=[root/"runs/a",root/"runs/b"]
        for p in paths: (p/"logs").mkdir(parents=True)
        decision={"status":decision_status}
        write_json(paths[0]/DECISION,decision)
        for p in (paths if second else paths[:1]):
            write_json(p/"logs/ladder_completion.json",{})
            write_json(p/"logs/optimization_validity.json",dict(status="PASS",completion_sha256=sha256(p/"logs/ladder_completion.json")))
        write_json(paths[0]/"logs/group3_prefix_diagnostic.json",dict(status=prefix_status,rtol=1e-6,atol=1e-8))
        return paths,decision
    def check(self,root,paths,decision):
        with patch("residual_tto.extended.check_study",return_value=({"study_id":"test"},paths)), \
             patch("residual_tto.extended.branch_decision",return_value=decision), \
             patch("residual_tto.extended.check_endpoint"), \
             patch("residual_tto.extended_validity.audit_validity",side_effect=lambda p:read_json(p/"logs/optimization_validity.json")):
            return assert_ready(paths[0])
    def test_prefix_fail_and_incomplete_do_not_block_valid_endpoints(self):
        for status in ("FAIL","INCOMPLETE"):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);paths,d=self.fixture(root,prefix_status=status)
                result=self.check(root,paths,d)
                self.assertEqual(result["prefix_status"],status);self.assertEqual(len(result["endpoints"]),2)
    def test_missing_second_endpoint_blocks_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,d=self.fixture(root,second=False)
            with self.assertRaises(FileNotFoundError): self.check(root,paths,d)
    def test_skip_closes_study_without_second_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,d=self.fixture(root,"SKIPPED_NEAR_ZERO",second=False)
            write_json(paths[1]/"logs/skip.json",dict(status="SKIPPED_NEAR_ZERO",decision_sha256=sha256(paths[0]/DECISION)))
            self.assertEqual(len(self.check(root,paths,d)["endpoints"]),1)
    def test_skip_cannot_hide_executed_second_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,d=self.fixture(root,"SKIPPED_NEAR_ZERO",second=False)
            write_json(paths[1]/"logs/skip.json",dict(status="SKIPPED_NEAR_ZERO",decision_sha256=sha256(paths[0]/DECISION)))
            write_json(paths[1]/"logs/ladder_provenance.json",{})
            with self.assertRaises(ValueError): self.check(root,paths,d)

    def test_group4_evaluator_barrier_before_any_gt_read(self):
        from residual_tto import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/"study_manifest.json",dict(experiment_group=GROUP))
            with patch("sys.argv",["evaluate","--run",str(p)]), \
                 patch("residual_tto.extended.assert_released",side_effect=RuntimeError("not released")), \
                 patch("residual_tto.evaluate.read_json") as read:
                with self.assertRaises(RuntimeError): evaluate.main()
                read.assert_not_called()

    def test_endpoint_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/"resolved_config.json",{});write_json(p/"study_manifest.json",{})
            for rel in FROZEN_FILES:
                path=p/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"test")
            freeze_endpoint(p,"COMPUTE_LIMIT",{}, {})
            self.assertEqual(check_endpoint(p)["status"],"FROZEN")
            (p/"baseline/predictions.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError): check_endpoint(p)
if __name__=="__main__": unittest.main()
