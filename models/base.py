import time
import os
import torch
from torch import nn


class BaseModel(nn.Module):
    """
    Model class amenable to tuning (requirements come from train.py file).
    Needs to have access to an optimizable model.parameters() method
    """
    def __init__(self, cfg, renderer=None):
        self.renderer = renderer
        if not cfg.eval_only:
            self.save_name = f'trained_models/{cfg.model_file}_{time.strftime("%y%m%d_%H%M%S")}.pt'
            cfg.saved_model = self.save_name
        super(BaseModel, self).__init__()

    def render_dist(self, title='Model Distribution', **kwargs):
        """
        Sample a batch of objects from the model and display them.
        :param title: Title of figure to display
        :param kwargs: Other, model-specific keyword arguments
        :return: The batch of sampled objects
        """
        raise NotImplementedError

    def sample(self, bsz=1):
        """
        Sample a batch of objects from the model
        :param bsz: number of objects to sample
        :return: objects, log probabilities of objects
        """
        raise NotImplementedError

    def get_prob(self, x):
        """
        Get log probabilities of (a batch of) objects x, in a differentiable manner
        :param x: Objects to get probabilities of (can be batched along first dimension)
        :return: log prob, (optionally) distribution parameters
        """
        raise NotImplementedError

    def fit(self, **kwargs):
        """
        Fit the model to a dataset
        :param kwargs: keyword arguments for fitting the model such as batch size, learning rate, eval   frequency etc.
        :return:
        """

    def save(self, model_filename=None):
        if model_filename is None:
            model_filename = self.save_name

        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        torch.save(self.state_dict(), model_filename)
