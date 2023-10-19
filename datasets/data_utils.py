
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Transform:
    def transform_points(self, pc):
        raise NotImplementedError

    def __call__(self, *pcs):
        if len(pcs) > 1:
            if isinstance(pcs[0], torch.Tensor):
                pcs = torch.stack(pcs, dim=0)
            else:
                pcs = np.stack(pcs, axis=0)
        else:
            pcs = pcs[0]
        return self.transform_points(pcs)


class RandomNoiseTransform(Transform):
    def __init__(self, scale):
        self.rng = np.random.default_rng()
        self.aug_sigma = scale

    def transform_points(self, pc):
        if isinstance(pc, torch.Tensor):
            return torch.normal(pc, std=self.aug_sigma)
        return self.rng.normal(loc=pc, scale=self.aug_sigma).astype(np.float32)


class FlipTransform(Transform):
    def transform_points(self, pc):
        if isinstance(pc, torch.Tensor):
            return torch.flip(pc, (-1,))
        else:
            return np.flip(pc, axis=-1).astype(np.float32)


class SortTransform(Transform):
    def __init__(self, order='zyx'):
        self.order = [ord(l) - ord('x') for l in reversed(order)]

    def transform_points(self, pc):
        if len(pc.shape) > 2:
            return tuple(self.transform_points(p) for p in pc)
        if isinstance(pc, torch.Tensor):
            pc = pc[pc[..., self.order[0]].sort()[1]]
            pc = pc[pc[..., self.order[1]].sort(stable=True)[1]]
            pc = pc[pc[..., self.order[2]].sort(stable=True)[1]]
            return pc
            # return torch.flip(pc[np.lexsort(pc.transpose(0, 1).cpu().numpy())], dims=(-1,))
        # return np.flip(pc[np.lexsort(pc.transpose())], axis=-1)
        return pc[np.lexsort(pc.transpose()[self.order])].astype(np.float32)


class RandomRotateTransform(Transform):
    def __init__(self, axis='z', com=False):
        self.r_ax = axis
        self.rng = np.random.default_rng()
        self.rotate_around_com = com
        self.rot_mat = None

    def get_matrix(self):
        return self.rot_mat

    def transform_points(self, pc):
        rot = R.from_euler(self.r_ax, self.rng.uniform(-180, 180), degrees=True)
        com = pc.mean(-2, keepdims=True) if self.rotate_around_com else 0.
        self.rot_mat = rot.as_matrix().astype(np.float32)
        if isinstance(pc, torch.Tensor):
            self.rot_mat = torch.tensor(self.rot_mat, device=device, dtype=pc.dtype)
        return (pc - com) @ self.rot_mat + com


class RandomScalingTransform(Transform):
    def __init__(self, alpha):
        self.alpha = alpha

    def transform_points(self, pc):
        if isinstance(pc, torch.Tensor):
            scale_factor = torch.rand(1, device=device) * (1. - self.alpha) + self.alpha
        else:
            scale_factor = np.random.uniform(self.alpha, 1.)
        return pc * scale_factor


class ComposeTransform(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def transform_points(self, pc):
        for t in self.transforms:
            pc = t(pc)

        return pc