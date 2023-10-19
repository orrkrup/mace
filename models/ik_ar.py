import os
import pickle
import time
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily
import torch.nn as nn
from torch.nn import ModuleList

from models.base import BaseModel


def get_layers(sizes):
    l = []
    for i in range(len(sizes) - 1):
        l.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            l.append(nn.LeakyReLU())

    return nn.Sequential(*l)


class GaussianMixtureModelNet(nn.Module):
    def __init__(
            self, input_size, output_size, output_range, hidden_layers, mixture_count, minimal_std, device
    ):
        assert len(output_range) == 2
        assert len(output_range[0]) == len(output_range[1]) == output_size
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.mixture_count = mixture_count
        self.minimal_std = minimal_std
        self.output_min, self.output_max = output_range
        self.output_min = torch.from_numpy(self.output_min).float().to(self.device)
        self.output_max = torch.from_numpy(self.output_max).float().to(self.device)

        self.network_output_size = self.mixture_count * (1 + 2 * self.output_size)
        tanh_slack = 1.1
        self._mu_linear_coefficient = tanh_slack * 0.5 * (self.output_max - self.output_min).view(1, self.output_size)
        self._mu_bias = tanh_slack * 0.5 * (self.output_max + self.output_min).view(1, self.output_size)

        self.net = get_layers([input_size] + hidden_layers + [self.network_output_size]).to(self.device)

    def forward(self, state):
        return self.net(state)

    def get_prediction(self, state, deterministic: bool):
        prediction = self(state)
        if deterministic:
            mixture_probability, mus, stds = self._output_to_distribution_parameters(prediction, state)
            mode = torch.argmax(mixture_probability, dim=-1, keepdim=True)
            mode = torch.tile(torch.unsqueeze(mode, dim=-1), dims=[1, 1] + [mus.shape[-1]])
            prediction = torch.gather(mus, dim=1, index=mode).squeeze(1)
            return self.clip_sample(prediction)
        else:
            prediction = self._output_to_sample(prediction, state)
            return self.clip_sample(prediction)

    def compute_loss(self, states, targets, take_mean=True, compute_mse=True):
        states = states.to(self.device)
        targets = targets.to(self.device)
        prediction = self(states)
        distribution = self._output_to_dist(prediction, states)
        loss = -distribution.log_prob(targets)
        if take_mean:
            loss = loss.mean()
        if compute_mse:
            mse = self._evaluate_mse(distribution, targets, take_mean)
            return loss, mse
        else:
            return loss

    def compute_mse(self, states, targets, take_mean=True):
        prediction = self(states)
        distribution = self._output_to_dist(prediction, states)
        return self._evaluate_mse(distribution, targets, take_mean)

    @staticmethod
    def _evaluate_mse(distribution, targets, take_mean=True):
        mean_prediction = distribution.mean
        return torch.nn.functional.mse_loss(mean_prediction, targets, reduction='mean' if take_mean else 'none')

    def _output_to_sample(self, net_output, current_state):
        output_distribution = self._output_to_dist(net_output, current_state)
        return output_distribution.sample()

    def _output_to_dist(self, net_output, current_state):
        mixture_probability, mus, stds = self._output_to_distribution_parameters(net_output, current_state)
        mixture_distribution = Categorical(logits=mixture_probability)
        components = Normal(mus, stds)
        components = Independent(components, 1)
        distribution = MixtureSameFamily(mixture_distribution, components)
        return distribution

    def get_distribution_parameters(self, current_state):
        prediction = self(current_state)
        return self._output_to_distribution_parameters(prediction, current_state)

    def _output_to_distribution_parameters(self, net_output, current_state):
        split_sizes = [self.mixture_count, self.network_output_size - self.mixture_count]
        mixture_probability, normal_parameters = torch.split(net_output, split_sizes, dim=-1)
        if torch.any(torch.isnan(mixture_probability)):
            print(
                f'distribution in mixture_probability logits has nans, {mixture_probability.detach().cpu().numpy().tolist()}')
        mus, stds = torch.chunk(normal_parameters, 2, dim=1)
        mus = torch.reshape(mus, (-1, self.mixture_count, self.output_size))
        mus = torch.tanh(mus) * self._mu_linear_coefficient + self._mu_bias
        stds = torch.reshape(stds, (-1, self.mixture_count, self.output_size))
        stds = self._make_positive(stds) + self.minimal_std
        return mixture_probability, mus, stds

    @staticmethod
    def _make_positive(x: Tensor):
        x = torch.exp(x)
        x = x + 1.e-5
        return x

    def clip_sample(self, sample):
        return torch.min(self.output_max, torch.max(self.output_min, sample))


class ComponentBaseAutoRegressiveModel(nn.Module):
    def __init__(
            self, input_size, output_size, output_range, hidden_layers, mixture_count, minimal_std, device
    ):
        assert len(output_range) == 2
        assert len(output_range[0]) == len(output_range[1]) == output_size
        super().__init__()
        self.device = device
        self.models = ModuleList([
            GaussianMixtureModelNet(
                input_size=i + input_size, output_size=1,
                output_range=(np.array([output_range[0][i]]), np.array([output_range[1][i]])),
                hidden_layers=hidden_layers, mixture_count=mixture_count, minimal_std=minimal_std, device=device)
            for i in range(output_size)
        ])

    def forward(self, state):
        state_ = state
        predictions = []
        for i, model in enumerate(self.models):
            ith_prediction = model.get_prediction(state_, False)
            state_ = torch.cat([state_, ith_prediction], dim=-1)
            predictions.append(ith_prediction)
        return torch.cat(predictions, dim=-1)

    def compute_loss(self, states, targets, take_mean=True, compute_mse=True):
        states_ = states
        losses, mses = [], []
        for i, model in enumerate(self.models):
            ith_target = targets[:, i].unsqueeze(-1)
            res = model.compute_loss(states_, ith_target, take_mean, compute_mse)
            if compute_mse:
                loss, mse = res
                mses.append(mse)
            else:
                loss = res
            losses.append(loss)
            states_ = torch.cat([states_, ith_target], dim=-1)
        losses = torch.sum(torch.stack(losses), dim=0)
        if take_mean:
            losses = losses.mean()
        if compute_mse:
            mses = torch.hstack(mses)
            mses = torch.linalg.norm(mses)
            return losses, mses
        else:
            return losses


class LoadedIKAutoregressive(BaseModel):
    def __init__(self, cfg, renderer=None):
        super(LoadedIKAutoregressive, self).__init__(cfg, renderer=renderer)

        joint_ranges = cfg.joint_ranges 
        model_path = cfg.model_path 
        if not vars(cfg).get('saved_model', False):
            cfg.saved_model = f'trained_models/ik/{os.path.split(cfg.model_path)[-1]}'

        # get the model params
        with open(f'{model_path}/model_params.pkl', 'rb') as f:
            hidden_layers, mixture_count, minimal_std = pickle.load(f)
        
        # get the model
        self.model = ComponentBaseAutoRegressiveModel(
            input_size=3, output_size=7, output_range=joint_ranges, hidden_layers=hidden_layers,
            mixture_count=mixture_count,
            minimal_std=minimal_std, device='cuda'
        )
        self.model.load_state_dict(torch.load(f'{model_path}/best_model'), strict=False)

        self.ee = None
        self.st = None

    def set_context(self, ee):
        self.ee = ee

    def render_dist(self, title='Model Distribution', num_samples=49, sim_render_dist=None, render_mode=None):
        # sim_render_dist == IKSimulator.render_dist
        if not num_samples:
            return
        
        if self.st is None:
            self.st = time.time()
        samples = self.sample(bsz=num_samples)
        sample_t = time.time() - self.st

        ims, scores, t = sim_render_dist(samples)
        # print(scores)
        # print(samples)
        if render_mode is not None:
            self.renderer.render(ims, title=title)
        return {'samples': samples, 'scores': scores, 'times': sample_t + t}

    def sample(self, bsz=1):
        assert self.ee is not None, 'call set context to set the ee position'
        model_inputs = torch.tensor(np.array([self.ee] * bsz), device=self.model.device, dtype=torch.float)
        configs = self.model(model_inputs)
        return configs

    def get_prob(self, x):
        configs = x
        return -self.model.compute_loss(
            torch.tensor(np.array([self.ee] * configs.shape[0]), device=self.model.device, dtype=torch.float), configs,
            take_mean=False, compute_mse=False
        )


