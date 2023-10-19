import numpy as np


class BaseSimulator(object):
    def __init__(self):
        self.rng = np.random.default_rng()
        self.gt_object = None

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def test_objects(self, object_batch, action=None):
        """
        :param object_batch: Batch of objects to manipulate
        :param action: Action (or set of actions) to take for manipulation test
        :return: Probability of manipulation success (or score) per object
        """
        raise NotImplementedError

    def test_and_show(self, obj=None, action=None, title=None):
        raise NotImplementedError
