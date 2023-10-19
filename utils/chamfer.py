import numpy as np
import torch


def np_chamfer_dist(pc1, pc2, k=1):
    """
    Calculates the Chamfer distance between point clouds. Supports one of the pc arguments being batched, although
    this may not be the fastest way to go with this implementation.
    :param k: Whether to average over distance to closest k points (default is 1, uses minimum)
    :return:
    """
    if len(pc1.shape) > 2 and len(pc2.shape) > 2:
        raise ValueError(f"Only one PC can be batched, not both. Got shapes {pc1.shape} and {pc2.shape}")
    min_points = min(pc1.shape[-2], pc2.shape[-2])
    pc1 = np.expand_dims(pc1, -3)
    pc2 = np.expand_dims(pc2, -2)
    diffnorm = np.square(pc1 - pc2).sum(axis=-1)

    if 1 == k:
        return diffnorm.min(axis=-2).sum(axis=-1) + diffnorm.min(axis=-1).sum(axis=-1)
    elif 1 < k < min_points:
        return np.partition(diffnorm, kth=k, axis=-2)[..., :k, :].mean(-2).sum(-1) + np.partition(diffnorm, kth=k, axis=-1)[..., :k].mean(-1).sum(-1)
    else:
        raise ValueError(f"topk should be between {1} and {min_points}, got {k}")


def chamfer_dist(pc1, pc2, k=1, one_sided=False):
    """
    Calculates the Chamfer distance between point clouds. Supports one of the pc arguments being batched, although
    this may not be the fastest way to go with this implementation.
    :param pc1:       First PC
    :param pc2:       Second PC
    :param k:         Whether to average over distance to closest k points (default is 1, uses minimum)
    :param one_sided: When this is True, only measure distance between points in pc1 and closest points in pc2, and not
                      vice versa
    :return:
    """
    if len(pc1.size()) > 2 and len(pc2.size()) > 2:
        raise ValueError(f"Only one PC can be batched, not both. Got shapes {pc1.shape} and {pc2.shape}")
    min_points = min(pc1.size(-2), pc2.size(-2))
    diffnorm = (pc1.unsqueeze(-3) - pc2.unsqueeze(-2)).square().sum(axis=-1)

    if 1 <= k <= min_points:
        return (diffnorm.topk(k, dim=-2, largest=False)[0].mean(-2).sum(-1) +
                (1 - one_sided) * diffnorm.topk(k, dim=-1, largest=False)[0].mean(-1).sum(-1))

        # Consider: mean CD instead of sum to discourage movement of points away from the halfspace
        # return (diffnorm.topk(k, dim=-2, largest=False)[0].mean(-2).mean(-1) +
        #         (1 - one_sided) * diffnorm.topk(k, dim=-1, largest=False)[0].mean(-1).mean(-1))
    else:
        return torch.ones(1, dtype=torch.float, device=pc2.device).squeeze() * 1000.0
        # raise ValueError(f"topk should be between {1} and {min_points}, got {k}. "
        #                  f"PCs have {pc1.size(-2)} and {pc2.size(-2)} points respectively")


def pairwise_cd(samples, k=1, one_sided=False, reduce='mean'):
    cds = []
    for ind, smp1 in enumerate(samples[:-1]):
        for smp2 in samples[ind + 1:]:
            cds.append(chamfer_dist(smp1, smp2, k=k, one_sided=one_sided))

    if 'mean' in reduce:
        return torch.stack(cds).mean()
    elif 'sum' in reduce:
        return torch.stack(cds).sum()
    else:
        return cds

