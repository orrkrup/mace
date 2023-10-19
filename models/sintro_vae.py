from os.path import join
import numpy as np
import torch
import torch.distributions as dists
from torch import nn

from soft_intro_vae.models.vae import SoftIntroVAE, ConditionalSoftIntroVAE
from models.base import BaseModel

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)


def gaussian_unit_log_pdf(x, mu, sigma=1.0):
    """
    mu: batch_size X x_dim
    sigma: scalar
    """
    # batch_size, x_dim = mu.shape
    # x = x.view(batch_size, -1)
    return -0.5 * x.size(-1) * x.size(-2) * np.log(2 * np.pi) - (0.5 / sigma) * ((x - mu) ** 2).sum(-1).sum(-1)


def init_to_eye(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.eye_(m.weight)
        m.bias.data.fill_(0.0)


def init_to_zero(m):
    if isinstance(m, nn.Linear):
        # nn.init.zeros_(m.weight)
        # m.bias.data.fill_(0.0)
        nn.init.normal_(m.weight, 0.0, 0.02)
        m.bias.data.fill_(0.0)


class SkipMLP(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            # nn.LayerNorm(h_dim, elementwise_affine=False),
            nn.Linear(h_dim, h_dim * 2),
            nn.ReLU(),
            # nn.LayerNorm(h_dim * 2, elementwise_affine=False),
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            # nn.LayerNorm(h_dim, elementwise_affine=False),
            nn.Linear(h_dim, h_dim)
        )
        self.mlp.apply(init_to_zero)

    def forward(self, x):
        return x + self.mlp(x)


class LoadedSoftIntroVAEModel(BaseModel):
    def __init__(self, cfg, renderer=None):
        super(LoadedSoftIntroVAEModel, self).__init__(cfg, renderer=renderer)
        model_type = ConditionalSoftIntroVAE if cfg.conditional else SoftIntroVAE
        self.model = model_type(vars(cfg)).to(device)
        cfg.clean_results_dir = False
        results_dir = join(cfg.results_root, cfg.arch, cfg.experiment_name)
        weights_path = join(results_dir, 'weights')
        self.model.load_state_dict(torch.load(join(weights_path, f'02000.pth'), map_location=device_name))

        self.z_dim = self.model.z_dim if hasattr(self.model, 'z_dim') else self.model.zdim
        self.prior_mu = torch.zeros(self.z_dim, device=device)
        self.prior_logvar = (torch.ones(self.z_dim, device=device) * vars(cfg).get('prior_std', 1.0)).pow(2).log()

        self.prior_mu = torch.nn.Parameter(self.prior_mu)
        self.prior_logvar = torch.nn.Parameter(self.prior_logvar)
        self.beta_kl = cfg.beta_kl
        self.beta_rec = cfg.beta_rec

        self.num_z_samples = 50

    def get_prob(self, x, mode='elbo'):
        if x.size(-1) == 3:
            x = x.transpose(-2, -1)

        # Return lower bound on log prob by estimating ELBO
        q_mu, q_logvar = self.model.encode(x)
        n_z = torch.randn(size=(self.num_z_samples, 1, self.z_dim), device=device)
        z = n_z * (0.5 * q_logvar).exp() + q_mu

        kl_loss = -0.5 * (1 + q_logvar - self.prior_logvar - q_logvar.exp() / torch.exp(self.prior_logvar) -
                          (q_mu - self.prior_mu).pow(2) / torch.exp(self.prior_logvar)).sum(1)

        log_p = - kl_loss

        return log_p, None

    def sample(self, bsz=1, return_probs=False):
        sigma = self.prior_logvar.exp().sqrt()
        z = torch.randn(bsz, self.z_dim).to(device) * sigma + self.prior_mu

        samples = self.model.decode(z)
        if return_probs:
            logprobs, _ = self.get_prob(samples)
            return samples.transpose(-2, -1), logprobs
        else:
            return samples.transpose(-2, -1)

    def sample_cond(self, cond, bsz=1):
        assert isinstance(self.model, ConditionalSoftIntroVAE), "Conditional sampling can only be done from conditoinal model"

        # FIXME: the following prevents 3-point PCs
        if cond.size(-1) == 3 and cond.size(-2) != 3:
            cond.transpose_(cond.dim() - 2, cond.dim() - 1)
        if cond.dim() < 3:
            cond = cond.unsqueeze(0)
        samples = self.model.sample(torch.cat(bsz * [cond], dim=0))
        samples.transpose_(samples.dim() - 2, samples.dim() - 1)
        return samples

    def parameters(self, recurse=True):
        return super().parameters(recurse=False)

    def render_dist(self, title='Model Distribution', num_samples=1, label='', color=None, cond=None, test_objects=None, **kwargs):
        with torch.no_grad():
            if cond is None:
                obj = self.sample(num_samples)
            else:
                obj = self.sample_cond(cond, bsz=num_samples)

        if callable(label):
            label = label(obj)
        if callable(color):
            color = color(obj)
        if len(label):
            kwargs.update(label=label)
        if color is not None:
            kwargs.update(color=color)

        self.renderer.render(obj.detach().cpu().numpy(), title=title, **kwargs)

        scores = test_objects(obj)

        return {'samples': obj, 'scores': scores}
