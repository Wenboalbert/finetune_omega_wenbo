import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from residual_tto.specs import InterventionSpec, resolve_specs
from residual_tto.intervention import ResidualBank, HookManager, inject_selected
from residual_tto.solver import damped_dual_gn, solve_direction, explicit_jacobian, focal_xy
from residual_tto.preprocess import preprocess

class ToyBlock(torch.nn.Module):
    def forward(self, x, rope=None):
        return x*2

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)

    def spec(self, scope="registers", site="post_frame", layer=0, views=(1,), ids=()):
        return InterventionSpec("test",site,layer,views,scope,ids)

    def test_selectors_and_routing(self):
        for scope, ids, slots in [("camera",(),[0]),("registers",(),list(range(1,17))),
                ("camera_registers",(),list(range(17))),("selected_registers",(0,15),[1,16])]:
            s=self.spec(scope,ids=ids)
            x=torch.randn(3,20,4)
            delta=torch.ones(1,len(slots),4,requires_grad=True)
            y=inject_selected(x,s,delta,3)
            mask=torch.zeros_like(x,dtype=torch.bool)
            mask[1,slots]=True
            self.assertTrue(torch.equal(x[~mask],y[~mask]))
            self.assertTrue(torch.allclose(y[mask]-x[mask],torch.ones_like(y[mask])))
            y.sum().backward()
            self.assertTrue(torch.equal(delta.grad,torch.ones_like(delta)))

    def test_roundtrip_multispec(self):
        specs=[self.spec(),InterventionSpec("two","pre_frame",1,(0,2),"camera")]
        bank=ResidualBank(specs,"cpu",4)
        value=torch.randn(bank.flatten(bank.current).numel())
        self.assertTrue(torch.equal(value,bank.flatten(bank.unflatten(value))))
        self.assertEqual(sum(map(len,bank.column_groups().values())),value.numel())

    def test_hooks_pre_post_and_cleanup(self):
        for site,expected in [("pre_frame",2.),("post_frame",1.)]:
            s=self.spec("camera",site)
            bank=ResidualBank([s],"cpu",4)
            model=SimpleNamespace(aggregator=SimpleNamespace(frame_blocks=torch.nn.ModuleList([ToyBlock()])))
            hooks=HookManager(model,bank,3)
            x=torch.zeros(3,20,4)
            with hooks:
                hooks.reset(bank.current)
                self.assertTrue(torch.equal(model.aggregator.frame_blocks[0](x,None),x))
                values={s.name:torch.ones_like(bank.current[s.name])}
                hooks.reset(values)
                y=model.aggregator.frame_blocks[0](x,None)
                hooks.assert_hits()
                self.assertTrue(torch.equal(y[1,0],torch.full((4,),expected)))
            self.assertEqual(len(model.aggregator.frame_blocks[0]._forward_hooks),0)
            self.assertEqual(len(model.aggregator.frame_blocks[0]._forward_pre_hooks),0)
            with self.assertRaises(RuntimeError):
                with hooks:
                    raise RuntimeError("intentional")
            self.assertFalse(hooks.handles)

    def test_duplicate_and_overlap_rejected(self):
        c=dict(name="a",site="post_frame",layer=14,editable_views=["b"],token_scope="registers")
        with self.assertRaises(ValueError): resolve_specs([c],["a","a"])
        with self.assertRaises(ValueError): resolve_specs([c,dict(c,name="b")],["a","b"])
        with self.assertRaises(ValueError): self.spec("selected_registers",ids=(1,1)).slots()

    def test_gn_matches_primal(self):
        a=torch.randn(4,9,dtype=torch.float64);r=torch.randn(4,dtype=torch.float64)
        got=damped_dual_gn(a,r,.3)
        expected=torch.linalg.solve(a.T@a+.3*torch.eye(9,dtype=a.dtype),-a.T@r)
        self.assertTrue(torch.allclose(got,expected,atol=1e-10))

    def test_dynamic_rows_and_perview(self):
        s=self.spec("camera",views=(0,1))
        bank=ResidualBank([s],"cpu",3); probes=bank.probes()
        x=bank.flatten(probes);a=torch.randn(4,6)
        r=a@x+torch.ones(4)
        j=explicit_jacobian(r,probes,[s])
        self.assertTrue(torch.equal(j,a))
        got=solve_direction(j,r.detach(),.1,"per_view",(0,1),bank)
        self.assertTrue(torch.allclose(got[:3],damped_dual_gn(a[:2,:3],r[:2],.1)))
        self.assertTrue(torch.allclose(got[3:],damped_dual_gn(a[2:,3:],r[2:],.1)))
        full=solve_direction(j,r.detach(),.1,"joint",(0,1),bank)
        self.assertTrue(torch.allclose(full,damped_dual_gn(a,r,.1)))

    def test_focal_axis_order(self):
        p=torch.zeros(1,2,9);p[...,7]=.8;p[...,8]=1.2
        xy=focal_xy(p,(100,200))
        self.assertAlmostEqual(float(xy[0,0,0]),100/np.tan(.6),places=4)
        self.assertAlmostEqual(float(xy[0,0,1]),50/np.tan(.4),places=4)

    def test_preprocessing_provider_parity(self):
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,(h,w) in enumerate([(80,301),(303,71),(131,197)]):
                rng=np.random.default_rng(i)
                path=Path(tmp)/str(i)
                path=path.with_suffix(".png")
                Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8)).save(path)
                paths.append(str(path))
            for mode in ["balanced","max_size"]:
                x,mask,records=preprocess(paths,[70.,80.,90.],mode,64,16)
                official=load_and_preprocess_images(paths,mode,64,16)
                self.assertTrue(torch.equal(x,official))
                self.assertEqual(mask.shape, x.shape[:1]+x.shape[2:])
                self.assertTrue(all(r["ray_max_abs_error"]<1e-10 for r in records))

class GeometryTests(unittest.TestCase):
    def test_shared_sim3_and_depth_scale(self):
        from residual_tto.evaluate import sim3,centers,ue_extrinsic
        src=np.array([[0.,0,0],[2.,0,0],[0,3.,0],[2.,3.,1.]])
        q=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])
        dst=2.3*(src@q.T)+[3.,-2,1]
        s,fit,b=sim3(src,dst)
        self.assertAlmostEqual(s,2.3)
        self.assertTrue(np.allclose(fit,q))
        self.assertTrue(np.allclose(s*(src@fit.T)+b,dst))
        row=dict(pos_x=100,pos_y=200,pos_z=300,pitch=0,yaw=0,roll=0)
        e=ue_extrinsic(row)
        self.assertTrue(np.allclose(centers(e[None])[0],[2.,-3.,1.]))
        z=4.;point=e[:,:3].T@(np.array([1.,2.,z])-e[:,3])
        aligned=s*q@point+b
        cnew=s*q@centers(e[None])[0]+b
        znew=(e[:,:3]@q.T@(aligned-cnew))[2]
        self.assertAlmostEqual(znew,s*z)

    def test_degenerate_support_rejected(self):
        from residual_tto.evaluate import sim3
        points=np.array([[0.,0,0],[1.,0,0],[2.,0,0],[3.,0,0]])
        with self.assertRaises(ValueError): sim3(points,points)

if __name__=="__main__":
    unittest.main()
