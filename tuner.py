import os
from copy import deepcopy
from tqdm import trange

import wandb

from sim.ik_sim import IKSimulator
from models.ik_ar import LoadedIKAutoregressive

import torch

from utils.utils import timer, export_best
from utils.rendering import WandBRenderer, ColorWrapper


class Tuner(object):
    def __init__(self, cfg):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.cfg = cfg
        self.metric_funcs = vars(cfg).get('metric_funcs', {})
        self.metrics = {}
        self.rendering_kwargs = {}
        self.r = WandBRenderer(cfg)
        self.model_filename = None
        self.model_class = None

        sim_class = None
        sim_kwargs = {}

        # Imports and dataset specific setup
        if 'shapenet' == cfg.dataset:
            from models.sintro_vae import LoadedSoftIntroVAEModel as model_class
            from sim.geometry_sim import PCGeometrySimulator as sim_class
            from soft_intro_vae.datasets.shapenet import ShapeNetDataset

            sim_kwargs.update(renderer=self.r, ood=self.cfg.ood_condition)
            dataset = ShapeNetDataset(root_dir=cfg.data_dir, classes=cfg.classes, split='test')
            # gt_object = torch.tensor(dataset[torch.randint(high=len(dataset), size=(1,)).item()][0])
            gt_object = torch.tensor(dataset[cfg.seed][0])

        elif 'ik' == cfg.dataset:
            model_class = LoadedIKAutoregressive
            sim_class = IKSimulator
            
            cam_params = {'render_mode': self.cfg.render_mode}
            if 'wall' == cfg.ik_object_type:
                # # [box dims, box pos, ee req pos]
                # gt_object = [[0.1, 0.2, 0.4], [-0.2, 0., 0.5], [0., 0., 0.5]]
                import numpy as np
                box_poses = np.array([(-0.1, 0, 0.5), (-0.25, 0, 0.8)])
                box_sizes = np.array([(0.2, 0.8, 0.01), (0.11, 0.8, 0.6)])
                gt_object = [box_sizes.tolist(), box_poses.tolist(), [-0.17, 0.0, 0.56]]
                
                cam_params['pos'] = (-0.1, 0.65, 0.5)
                cam_params['target'] = (-0.17, 0., 0.55)

            elif 'window' == cfg.ik_object_type:
                # side_sizes = [[0.1, 0.9, 0.25], [0.1, 0.9, 0.25], [0.1, 0.25, 0.9], [0.1, 0.25, 0.9]]
                # side_poses = [[-0.2, 0.0, 0.425], [-0.2, 0.0, 0.975], [-0.2, -0.275, 0.7], [-0.2, 0.275, 0.7]]
                # gt_object = [side_sizes, side_poses, [0., 0., 0.5]]    

                side_sizes = [[0.02, 0.5, 0.1], [0.02, 0.5, 0.1], [0.02, 0.1, 0.3], [0.02, 0.1, 0.3]]
                side_poses = [[-0.2, 0.0, 0.4], [-0.2, 0.0, 0.8], [-0.2, 0.2, 0.6], [-0.2, -0.2, 0.6]]
                gt_object = [side_sizes, side_poses, [0.015, 0.0, 0.558]]

                cam_params['pos'] = (0.2, 0.4, 0.65)
                cam_params['target'] = (-0.2, 0., 0.5)

            elif 'inv_bin' == cfg.ik_object_type:
                import numpy as np
                from scipy.spatial.transform import Rotation as R

                # Box from CLI parameters
                # init_box_pose = np.array([-0.1, 0.0, 0.4])
                # pos_offset = np.array([float(x) for x in vars(cfg).get('ik_obj_pos', '0.0 0.0 0.0').split()])
                # theta, psi = [float(x) for x in vars(cfg).get('ik_obj_rot', '0.0 0.0').split()]

                # box_poses = np.array([(0, 0, 0.1), (0, 0, -0.1), (0.15, 0, 0), (0, 0.1, 0), (0, -0.1, 0)])
                # box_sizes = np.array([(0.3, 0.2, 0.01), (0.3, 0.2, 0.01), (0.01, 0.2, 0.2), (0.3, 0.01, 0.2), (0.3, 0.01, 0.2)])

                # Hard-coded box for real robot comparison
                th = 0.0025
                l = 0.26
                w = 0.2
                z = th + w / 2 # - 0.011
                init_box_pose = np.array([-0.1, -0.1, z])
                pos_offset = np.array([0., 0., 0.])
                box_poses = np.array([(0, 0, w / 2 + th / 2), (0, 0, - w / 2 - th / 2), (l / 2, 0, 0), (0, w / 2, 0), (0, -w / 2, 0)])
                box_sizes = np.array([(l, w, th), (l, w, th), (th, w, w), (l, th, w), (l, th, w)])
                theta = 0
                psi = -np.pi / 4

                box_rotation = R.from_euler('xyz', [0, theta, psi], degrees=False).as_quat()
                # box_scale = 1.0
                # box_poses = box_rotation.apply(box_poses * box_scale)
                # box_sizes = np.abs(box_rotation.apply(box_sizes * box_scale))
                
                target = init_box_pose + pos_offset

                gt_object = [box_sizes.tolist(), box_poses.tolist(), target.tolist(), box_rotation.tolist()]
                
                cam_params['pos'] = (0.0, 0.55, 0.5)
                cam_params['target'] = (-0.35, 0., 0.2)

            sim_kwargs['headless'] = not cfg.debug_mode
            sim_kwargs['num_envs'] = cfg.sample_size
            sim_kwargs['cam_params'] = cam_params
        else:
            raise ValueError("Unknown dataset type")

        # Load model (and prior model if necessary)
        self.model = model_class(self.cfg, renderer=self.r).to(self.device)
        self.model_class = model_class
        if self.model_filename is not None:
            self.model.load_state_dict(torch.load(self.model_filename))

        # Create simulator and set ground truth object
        self.sim = sim_class(**sim_kwargs)
        self.sim.set_gt(gt_object)

        # Other dataset specific preparations
        self.rendering_kwargs = {'num_samples': 4096}
        if 'shapenet' == cfg.dataset:
            self.r.flip = False
            self.r.color = [[100, 100, 100], [223, 116, 62]]

            def label_wrapper(obj):
                t = self.sim.test_objects(obj)
                return t.squeeze().tolist()

            cw = ColorWrapper(self.sim)
            self.rendering_kwargs.update(label=label_wrapper, color=cw.wrap, overlay=True, test_objects=self.sim.test_objects)

            if not vars(self.cfg).get('saved_model', False):
                self.cfg.saved_model = os.path.join('trained_models', self.cfg.experiment_name)

        if 'ik' == cfg.dataset:
            self.model.set_context(self.sim.target_ee.cpu().numpy())
            self.rendering_kwargs.update(sim_render_dist=self.sim.render_dist)
            
            if cfg.render_mode == 'high_q':
                # TODO: this overrides the intitial setting above
                self.rendering_kwargs.update(num_samples=50)
            elif cfg.render_mode is None:
                self.rendering_kwargs.update(num_samples=cfg.sample_size)
            
            self.rendering_kwargs.update(render_mode=cfg.render_mode)

    def save_model(self):
        save_dir, save_name = os.path.split(self.cfg.saved_model)
        save_name = 'adapted_' + save_name
        self.model.save(os.path.join(save_dir, save_name))

    def render_and_measure(self, dist='Prior'):
        x = self.model.render_dist(title=f'{dist} Model Distribution', **self.rendering_kwargs)
        if self.metric_funcs is not None:
            for k, metric_func in self.metric_funcs.items():
                if dist.lower() in k.lower():
                    self.metrics[k] = metric_func(x)
            if self.cfg.debug_mode:
                print(f"Sampled {self.rendering_kwargs['num_samples']} objects from {dist.lower()}. " + 
                    " ".join([f"{k}: {v}" for k, v in self.metrics.items()]))
        
    def train_continuous(self):
        
        opt_mode = self.cfg.opt_mode

        # Pre-tuning distribution (prior)
        self.render_and_measure()

        # Show ground truth object
        self.sim.test_and_show(obj=None, title='Tested GT Object')

        # Initialize optimizer
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        prior_model = deepcopy(self.model)
        
        if not self.cfg.bsz:
            self.cfg.bsz = self.cfg.sample_size

        for it in trange(self.cfg.iterations):

            # Sample and test objects at 0 and then resample every cfg.grad_steps
            resample = not it % self.cfg.grad_steps
            if resample:
                with torch.no_grad(), timer() as t:

                    # Sample from previous iteration model:
                    obj_sample = self.model.sample(self.cfg.sample_size)

                    # Attempt manipulation (actions selected by simulator), collect results
                    res = self.sim.test_objects(obj_sample)

                if 'get_best' in opt_mode:
                    export_best(res, obj_sample, t.t)

                if self.cfg.bsz == self.cfg.sample_size:
                    obj_batch, res_batch = obj_sample, res
                    data_iter = None
                else:
                    ds = torch.utils.data.TensorDataset(obj_sample, res)
                    loader = torch.utils.data.DataLoader(ds, batch_size=self.cfg.bsz, shuffle=True)
                    data_iter = iter(loader)

            if data_iter is not None:
                try:
                    obj_batch, res_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    obj_batch, res_batch = next(data_iter)

            if 'mace' in opt_mode:
                best = torch.topk(res_batch, int(self.cfg.bsz * self.cfg.quantile), sorted=False)
                x = obj_batch[best.indices]
                loss = -self.model.get_prob(x)[0].mean()  # Negative because we're minimizing the loss

            elif 'is' == opt_mode:
                # No top-k, With IS
                pre_ll = self.model.get_prob(obj_batch)[0]
                with torch.no_grad():
                    prior_ll = prior_model.get_prob(obj_batch)[0]
                    score = res_batch * torch.exp(prior_ll - pre_ll.detach())
                loss = -(pre_ll * score).mean()

            else:
                raise ValueError(f'Unknown opt_mode: {opt_mode}')

            # Update model
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Logging
            log_dict = {'Score': res_batch.mean(), 'Max Score': res_batch.max(), 'Min Score': res_batch.min(), 'Loss': loss / self.cfg.bsz}
            wandb.log(log_dict, step=it)

        # Post-tuning distribution (posterior)
        self.render_and_measure(dist='Posterior')

        # Save tuned model
        self.save_model()

    def post_tuning(self):        
        if isinstance(self.sim, IKSimulator):
            self.sim.close()
    
    def get_metrics(self):
        return self.metrics