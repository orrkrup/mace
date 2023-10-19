import numpy as np
import torch
import wandb
from scipy.spatial.transform import Rotation as R

from sim.base_sim import BaseSimulator
from datasets.data_utils import RandomRotateTransform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PCGeometrySimulator(BaseSimulator):
    def __init__(self, renderer, finger_width=0.08, ood=False):
        super(PCGeometrySimulator, self).__init__()
        self.renderer = renderer
        self.f_width = finger_width
        self.x_lim = [0.1, 1.0]
        self.y_lim = [0.1, 1.0]
        self.z_lim = [0.1, 1.0]

        self.gt_object = None
        self.action = None
        self.evidence = None
        self.ood_fingers = ood

    def render(self, object, title=''):
        self.renderer.render(object, title=title)

    def set_gt(self, gt):
        self.gt_object = gt.float().to(device)
        self.action = self.get_action_()
        self.evidence = self.get_contact_pts(self.gt_object, self.action).clone()
        # self.renderer.set_pc_overlay(self.evidence.cpu().squeeze().numpy(), [223, 116, 62])
        self.renderer.set_pc_overlay(self.generate_fingers(), [223, 116, 62])
        # self.boxes = self.generate_boxes()

    def test_and_show(self, obj=None, action=None, title=None):
        if obj is None:
            obj = self.gt_object
        # TODO: think of a different way to show grasp points?
        if isinstance(obj, torch.Tensor):
            obj = obj.cpu()
        self.renderer.render((obj, self.gt_object.cpu()), title=title, same_frame=True, overlay=True)

    def get_action_(self, fingers=5, axes=True):
        # Actions are vectors representing hyperplane normals
        if axes:
            assert fingers in range(2, 6), "Axes aligned actions require 2 to 5 fingers"
            # axes_options = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]]
            # No need to normalize, already normalized
            # axes_options = [[1, 0, 0], [-1, 0, 0], [1, 1, 0], [-1, -1, 0], [0, 0, -1]]
            # axes_options = [[1, 1, 0], [0, 1, 0], [1, -1, 0], [0, -1, 0], [-1, 0, 0]]
            if self.ood_fingers:
                axes_options = [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [1, 0, 0]]
            else:
                axes_options = [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 0, 0]]
            action_vecs = torch.stack([torch.tensor(opt, dtype=torch.float, device=device) for opt in axes_options[:fingers]], dim=0)
            action_vecs = action_vecs / action_vecs.norm(dim=1, keepdim=True)
        else:
            action_vecs = torch.rand(size=(fingers, 3))
            action_vecs / action_vecs.norm(dim=1, keepdim=True)
        return action_vecs

    def get_contact_dists(self, object_batch, action):
        c_points, _ = self.get_contact_(object_batch, action)
        return c_points

    def get_contact_pts(self, object_batch, action):
        _, c_inds = self.get_contact_(object_batch, action)
        if len(object_batch.shape) < 3:
            object_batch = object_batch.unsqueeze(0)
        return torch.stack([obj[c_inds[ind]] for ind, obj in enumerate(object_batch)], dim=0)

    def get_contact_(self, object_batch, action):
        if len(object_batch.shape) < 3:
            object_batch = object_batch.unsqueeze(0)

        # Projection on action vectors (bsz x n_points x actions x n_dim)
        proj = (object_batch @ action.T).unsqueeze(-1) * action
        # Distance from action vectors (bsz x n_points x actions)
        dists = (object_batch.unsqueeze(-2) - proj).norm(dim=-1)
        # Points that are close enough to the action axis (bsz x n_points x actions x n_dim)
        # TODO: the min operation and the zeros in the second argument of "where" both assume there are always
        #  points on the negative side of the vector. Edit: the bottom line might solve this
        # close_points = torch.where((dists < self.f_width / 2).unsqueeze(-1), proj, torch.zeros_like(proj))

        # Contact points are the furthest point along action which is close enough to the action axis (bsz x actions)
        # contact_points, inds = (close_points * action).sum(dim=-1).min(dim=1)

        contact_points, inds = torch.where((dists < self.f_width / 2), (proj * action).sum(dim=-1), torch.ones_like(dists)).min(dim=1)

        # # Test and show contact points
        # if 1 == object_batch.shape[0]:
        #     self.renderer.render((self.gt_object.cpu(), self.gt_object[inds].squeeze(0).cpu()), title='Contact point testing', same_frame=True)
        # else:
        #     self.renderer.render((object_batch[0].cpu(), object_batch[0][inds[0]].cpu()),
        #                          title='Contact point testing', same_frame=True)

        return contact_points, inds

    def test_objects(self, object_batch, action=None):
        """
        Currently gives higher scores to "squatter" objects (lower z range)

        :param object_batch: Batch of objects to manipulate
        :param action: Action (or set of actions) to take for manipulation test
        :return: Probability of manipulation success (or score) per object
        """
        action = self.get_action_() if self.action is None else self.action
        gt_cpt = self.get_contact_dists(self.gt_object, action)
        b_cpt = self.get_contact_dists(object_batch, action)

        # Be close to gt contact points - dense (L2)
        # scores = 1. - (b_cpt - gt_cpt).pow(2).mean(-1)

        # Be close to gt contact points - dense (L1)
        # scores = (1 - (b_cpt - gt_cpt).abs().mean(-1)).pow(2)

        # Be close to gt contact points - dense (sum)
        scores = (1 - (b_cpt - gt_cpt).abs().sum(-1)).clamp(0, 1)

        # Be close to gt contact points - sparse
        # scores = ((b_cpt - gt_cpt).abs() < self.f_width).all(dim=-1).float()

        # Fit in gt bb (given by finger positions)
        # scores = (b_cpt.abs() < gt_cpt.abs()).all(dim=-1).float()

        return scores

    def generate_fingers(self, mode='horizontal'):
        n_points = 200
        f_len = 0.1
        init_r = np.random.uniform(size=(n_points,)) * f_len
        init_th = np.random.uniform(size=(n_points,)) * 2 * np.pi
        if 'horizontal' == mode:
            f_template = np.stack([-init_r, np.cos(init_th) * self.f_width / 6., np.sin(init_th) * self.f_width / 6.], axis=1)
        else:
            f_template = np.stack([(np.cos(init_th) - 1) * self.f_width / 6., np.sin(init_th) * self.f_width / 6., init_r], axis=1)
        fingers = []
        for ind, (finger, act) in enumerate(zip(self.evidence.squeeze(), self.action)):
            r = R.align_vectors([[1, 0, 0], [0, 0, 1]], [act.cpu().numpy(), [0, 0, 1]])[0].as_matrix()
            fingers.append(f_template @ r + finger.cpu().numpy())

        return np.concatenate(fingers, axis=0)

    def generate_boxes(self):
        s_dist = self.f_width / 2.
        c_dist = np.sqrt(2.) * s_dist
        f_len = 0.1

        init_box = [[0., s_dist, 0.], [0., -s_dist, 0.], [-2 * s_dist, s_dist, 0.], [-2 * s_dist, -s_dist, 0.],
                    [0., s_dist, f_len], [0., -s_dist, f_len], [-2 * s_dist, s_dist, f_len], [-2 * s_dist, -s_dist, f_len]]
        # wandb.log({'box_test_0': wandb.Object3D({'type': 'lidar/beta', 'points': self.gt_object.cpu().numpy(), 'boxes': np.array([{'corners': init_box, 'color': [255, 255, 255]}])})})
        init_box = np.array(init_box)
        box_list = []
        for ind, (finger, act) in enumerate(zip(self.evidence.squeeze(), self.action)):
            # FIXME: the following rotation assumes z axis remains the same
            r = R.align_vectors([[1, 0, 0], [0, 0, 1]], [act.cpu().numpy(), [0, 0, 1]])[0].as_matrix()
            corners = init_box @ r + finger.cpu().numpy()
            box_list.append({'corners': corners.tolist(), 'label': f'{ind}', 'color': [223, 116, 62]})
            # wandb.log({'box_test_0': wandb.Object3D(
            #     {'type': 'lidar/beta', 'points': np.array([[0, 0, 0]]), 'boxes': np.array([box_list[-1]])})})

        return np.array(box_list)