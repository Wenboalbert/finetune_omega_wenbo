import copy, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from residual_tto.io import read_json, write_json, sha256
from residual_tto import group5 as g

ROOT = Path(__file__).resolve().parents[1]

class ContractTests(unittest.TestCase):
    def test_configs_change_only_budget_and_metadata(self):
        ref = g.reference_config(ROOT)
        for suffix, budget in zip(("070","080","090"), g.BUDGETS):
            cfg = read_json(ROOT / ("configs/group5_frame40_b" + suffix + ".json"))
            self.assertTrue(g.validate_config(cfg, ref))
            self.assertEqual(cfg["optimizer"], ref["optimizer"])
            self.assertEqual(cfg["cumulative_budget_relative"], budget)

    def test_unapproved_numerical_and_scope_changes_rejected(self):
        cfg = read_json(ROOT / "configs/group5_frame40_b070.json")
        for key, value in (("tau_absolute",1e-8),("max_accepted_steps",160),
                           ("initial_step_relative",.001),
                           ("target_mean_focal_relative_error",.001)):
            bad = copy.deepcopy(cfg); bad["optimizer"][key] = value
            with self.assertRaises(ValueError): g.validate_config(bad,g.reference_config(ROOT))
        for budget in (.06,.075,.1):
            bad=copy.deepcopy(cfg);bad["cumulative_budget_relative"]=budget
            with self.assertRaises(ValueError): g.validate_config(bad,g.reference_config(ROOT))
        bad=copy.deepcopy(cfg);bad["supervised_views"]=["CCTV_01","Drone_02"]
        with self.assertRaises(ValueError): g.validate_config(bad,g.reference_config(ROOT))

    def test_historical_core_configs_and_entrypoints_unchanged(self):
        hashes=g.verify_frozen(ROOT)
        self.assertIn("src/residual_tto/constrained_gn.py",hashes)
        self.assertIn("scripts/run_extended.py",hashes)
        self.assertIn("configs/extended_frame40_b050.json",hashes)
        self.assertIn("src/residual_tto/ladder_runner.py:frozen_body",hashes)

    def test_prepare_and_pbs_use_new_entry(self):
        self.assertIn("run_group5.py",(ROOT/"scripts/v001_group5.pbs").read_text())
        text=(ROOT/"scripts/prepare_group5.py").read_text()
        self.assertIn('(("070",.07),("080",.08),("090",.09))',text)
        self.assertNotIn("prepare_extended.py",text)

class DecisionTests(unittest.TestCase):
    def focal(self, x, y):
        return dict(relative_error_xy=[x,y],emax=max(x,y),mean=(x+y)/2)

    def test_all_valid_normal_terminations_continue(self):
        for term in g.TERMINATIONS-{"SOLVER_FAILURE"}:
            valid=dict(status="PASS",termination=term)
            self.assertEqual(g.decide(valid,self.focal(.01,.02),.07)["next_budget"],.08)
            self.assertEqual(g.decide(valid,self.focal(.01,.02),.08)["next_budget"],.09)
            self.assertEqual(g.decide(valid,self.focal(.01,.02),.09)["action"],"STOP_MAX_BUDGET")

    def test_near_zero_inclusive_both_axes_not_mean(self):
        valid=dict(status="PASS",termination="COMPUTE_LIMIT")
        for budget in g.BUDGETS:
            self.assertEqual(g.decide(valid,self.focal(.001,.001),budget)["action"],"STOP_NEAR_ZERO")
            self.assertEqual(g.decide(valid,self.focal(0,.00100000001),budget)["focal_status"],"NOT_NEAR_ZERO")
        self.assertEqual(g.decide(valid,self.focal(0,.0015),.07)["action"],"CONTINUE")

    def test_failure_never_branches_even_if_focal_zero(self):
        for status,term in (("FAIL","TR_RADIUS_MIN"),("PASS","SOLVER_FAILURE"),("PASS","unknown")):
            with self.assertRaises(ValueError):
                g.decide(dict(status=status,termination=term),self.focal(0,0),.07)
        for value in (float("nan"),float("inf"),-.1):
            with self.assertRaises(ValueError):
                g.decide(dict(status="PASS",termination="COMPUTE_LIMIT"),self.focal(value,.1),.07)

    def test_inconsistent_focal_maximum_rejected(self):
        f=self.focal(.01,.02);f["emax"]=0.
        with self.assertRaises(ValueError): g.decide(dict(status="PASS",termination="COMPUTE_LIMIT"),f,.07)

class AuditTests(unittest.TestCase):
    def test_only_derived_max_step_has_existing_tolerance(self):
        a=dict(status="PASS",max_step_relative=.0025,cumulative_relative=.07,sources={"a":"hash"})
        b=copy.deepcopy(a);b["max_step_relative"]+=1e-18
        g.compare_validity(a,b)
        b["max_step_relative"]+=1e-6
        with self.assertRaises(ValueError): g.compare_validity(a,b)
        b=copy.deepcopy(a);b["cumulative_relative"]+=1e-16
        with self.assertRaises(ValueError): g.compare_validity(a,b)
        b=copy.deepcopy(a);b["sources"]["a"]="other"
        with self.assertRaises(ValueError): g.compare_validity(a,b)

    def test_recorded_nonfinite_forward_blocks_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write_json(p/g.OPT,dict(attempts=[{"forward_numerical_error":"nonfinite focal"}]))
            with patch("residual_tto.group5.audit_group4",return_value={"status":"PASS"}):
                with self.assertRaises(ValueError): g.audit_validity(p)

    def test_initial_J_differences_are_diagnostic_not_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            for p,j in ((a,1.),(b,1.00001)):
                (p/"baseline").mkdir(parents=True);(p/"logs").mkdir()
                np.savez(p/"baseline/predictions.npz",pose_enc=[1.],depth=[2.])
                np.savez(p/"logs/preflight_jacobian.npz",J=np.eye(2)*j,r=[1.,1.],h0=[1.,1.])
            result=g.initialization_diagnostic(a,b)
            self.assertTrue(result["baseline_exact"])
            self.assertGreater(result["initial_J_max_abs"],0)
            self.assertEqual(result["prefix_status"],"NOT_APPLICABLE")
            np.savez(b/"logs/preflight_jacobian.npz",J=np.eye(2),r=[1.,2.],h0=[1.,1.])
            with self.assertRaises(ValueError): g.initialization_diagnostic(a,b)

class BarrierTests(unittest.TestCase):
    def fixture(self, root, near_index=None):
        paths=[root/"runs"/name for name in ("a","b","c")]
        reference=root/"reference"
        for p in [reference]+paths:
            for rel in ("baseline/predictions.npz","logs/preflight_jacobian.npz"):
                f=p/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b"fixture")
        study=dict(study_id="test",reference={"path":str(reference)})
        for path in paths: write_json(path/"study_manifest.json",{"test":True})
        decisions={};stopped=None
        for i,p in enumerate(paths):
            if stopped is not None:
                write_json(p/"logs/skip.json",dict(status="SKIPPED_NEAR_ZERO",trigger_run=str(stopped),
                                                  decision_sha256=sha256(stopped/g.DECISION)))
                continue
            write_json(p/"logs/ladder_completion.json",{})
            v=dict(status="PASS",termination="COMPUTE_LIMIT",max_step_relative=.002,
                   completion_sha256=sha256(p/"logs/ladder_completion.json"))
            write_json(p/"logs/optimization_validity.json",v)
            f=dict(relative_error_xy=[0.,0.] if i==near_index else [.1,.1],
                   emax=0. if i==near_index else .1)
            d=g.decide(v,f,g.BUDGETS[i]);decisions[str(p)]=d;write_json(p/g.DECISION,d)
            prev=reference if i==0 else paths[0]
            write_json(p/"logs/initialization_diagnostic.json",dict(
                reference=str(prev),prefix_status="NOT_APPLICABLE",baseline_exact=True,
                initial_r_exact=True,initial_h0_exact=True,
                files={str(q/rel):sha256(q/rel) for q in (prev,p)
                       for rel in ("baseline/predictions.npz","logs/preflight_jacobian.npz")}))
            if i==near_index: stopped=p
        return study,paths,decisions

    def ready(self,study,paths,decisions):
        with patch("residual_tto.group5.check_study",return_value=(study,paths)), \
             patch("residual_tto.group5.check_endpoint"), \
             patch("residual_tto.group5.audit_validity",side_effect=lambda p:read_json(p/"logs/optimization_validity.json")), \
             patch("residual_tto.group5.branch_decision",side_effect=lambda p,v:decisions[str(p)]):
            return g.assert_ready(paths[0])

    def test_all_three_valid_endpoints_close_without_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));r=self.ready(*args)
            self.assertEqual(len(r["endpoints"]),3)
            self.assertEqual(r["search_stop_reason"],"MAX_BUDGET")

    def test_success_at_each_budget_has_exact_skips(self):
        for index in range(3):
            with tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp),index);r=self.ready(*args)
                self.assertEqual(len(r["endpoints"]),index+1)
                self.assertEqual(len(r["skipped"]),2-index)

    def test_missing_endpoint_cannot_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            (args[1][2]/"logs/optimization_validity.json").unlink()
            with self.assertRaises(FileNotFoundError): self.ready(*args)

    def test_skip_cannot_hide_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp),0)
            write_json(args[1][1]/"logs/ladder_provenance.json",{})
            with self.assertRaises(ValueError): self.ready(*args)

    def test_failure_blocks_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            write_json(args[1][0]/"logs/study_failure.json",{"status":"INCOMPLETE"})
            with self.assertRaises(ValueError): self.ready(*args)

    def test_decision_tampering_blocks_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            (args[1][0]/g.DECISION).write_text("{}")
            with self.assertRaises(ValueError): self.ready(*args)

    def test_evaluator_checks_group5_barrier_before_gt(self):
        from residual_tto import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_json(p/"study_manifest.json",dict(experiment_group=g.GROUP))
            with patch("sys.argv",["evaluate","--run",str(p)]), \
                 patch("residual_tto.group5.assert_released",side_effect=RuntimeError("not released")), \
                 patch("residual_tto.evaluate.read_json") as read:
                with self.assertRaises(RuntimeError): evaluate.main()
                read.assert_not_called()

if __name__=="__main__": unittest.main()
