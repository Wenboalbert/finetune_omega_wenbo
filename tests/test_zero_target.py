import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from residual_tto.ladder_optimizer import optimize
from residual_tto.zero_target import validate_config, reference_config, verify_core, GROUP
from residual_tto.evaluate import depth_gt
from residual_tto.trust_policy import focal_diagnostics

ROOT=Path(__file__).resolve().parents[1]

class ZeroTargetTests(unittest.TestCase):
    def test_all_new_configs_change_only_target_and_metadata(self):
        for suffix,budget in zip(("003","010","020","040"),(.003,.01,.02,.04)):
            config=json.loads((ROOT/("configs/zero_target_frame40_b"+suffix+".json")).read_text())
            self.assertTrue(validate_config(config,reference_config(ROOT,budget)))

    def test_frozen_core_is_byte_identical(self):
        self.assertEqual(len(verify_core(ROOT)),8)

    def test_other_parameter_change_rejected(self):
        ref=reference_config(ROOT,.04)
        new=copy.deepcopy(ref);new["experiment_group"]=GROUP
        new["optimizer"]["target_mean_focal_relative_error"]=0.
        for key,value in (("max_accepted_steps",41),("initial_step_relative",.001),
                          ("tau_absolute",1e-9)):
            bad=copy.deepcopy(new);bad["optimizer"][key]=value
            with self.assertRaises(ValueError): validate_config(bad,ref)

    def problem(self,target,initial_r=None,diagnostic_override=None):
        J=np.array([[1.,0,0],[0,1.,0]])
        start=np.array([.125,.125]) if initial_r is None else np.asarray(initial_r)
        config=json.loads((ROOT/"configs/zero_target_frame40_b040.json").read_text())
        options=dict(config["optimizer"],initial_step_relative=.05,maximum_step_relative=.25,
                     minimum_step_relative=.00001,target_mean_focal_relative_error=target)
        states=[];calls=[]
        def data(r):
            error=float(np.abs(np.expm1(r)).mean()) if diagnostic_override is None else diagnostic_override
            return dict(r=r,pose_enc=[],diagnostics={"supervised_mean_relative_error":error})
        def forward(a): calls.append("forward");return data(start+J@a)
        def jac(a): calls.append("J");return J,data(start+J@a)
        result=optimize(data(start),J,1.,10.,options,0.,forward,jac,
                        accepted_event_fn=lambda i,a,d:states.append(a.copy()))
        return result,states,calls

    def test_nonzero_below_15_percent_no_longer_early_stops(self):
        old,_,_=self.problem(.15)
        new,states,_=self.problem(0.)
        self.assertEqual(old["counters"]["accepted_steps"],0)
        self.assertGreater(len(states),0)
        self.assertEqual(new["termination_reason"],"TARGET_REACHED")
        self.assertEqual(new["data"]["diagnostics"]["supervised_mean_relative_error"],0.)

    def test_exact_zero_stops_without_extra_callback(self):
        result,_,calls=self.problem(0.,[0.,0.])
        self.assertEqual(result["termination_reason"],"TARGET_REACHED")
        self.assertEqual(calls,[])

    def test_zero_log_loss_does_not_override_nonzero_reported_error(self):
        result,_,_=self.problem(0.,[0.,0.],diagnostic_override=1e-8)
        self.assertNotEqual(result["termination_reason"],"TARGET_REACHED")

    def test_old_and_new_accepted_prefix_equal(self):
        old,a,_=self.problem(.15,[.3,.3]);new,b,_=self.problem(0.,[.3,.3])
        self.assertGreater(len(b),len(a))
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        self.assertEqual(old["termination_reason"],"TARGET_REACHED")

    def test_gt_precision_and_both_axes_preserved(self):
        gt=np.array([[1.,1.]]*4+[[344.00000000000006,341.33333333333337]])
        pred=gt.astype(np.float32).astype(np.float64)
        self.assertGreater(focal_diagnostics(pred,gt)["supervised_mean_relative_error"],0)

class SentinelMaskTests(unittest.TestCase):
    def test_source_marker_before_units_and_nearest_resize(self):
        raw=np.array([[100.,65504.],[65503.,200.]],dtype=np.float32)
        transform=dict(raw_hw=[2,2],crop_ltrb=[0,0,2,2],resize_hw=[4,4],pad_ltrb=[1,1,1,1])
        with patch("cv2.imread",return_value=raw):
            old,vm,_=depth_gt("test",transform,.01)
            new,valid,_=depth_gt("test",transform,.01,[65504.])
        np.testing.assert_array_equal(old,new)
        self.assertEqual(int(vm.sum()),16);self.assertEqual(int(valid.sum()),12)
        self.assertFalse(valid[0].any())
        self.assertTrue(valid[3:5,1:3].all()) # nearby legitimate value is not clipped
        np.testing.assert_array_equal(vm & ~valid,vm & (old==np.float32(65504)*np.float32(.01)))

if __name__=="__main__": unittest.main()
