import argparse
import multiprocessing as mp

import json

# Need to import IsaacGym before any other torch imports
from tuner import Tuner

import torch
import wandb


def main(parsed_cfg):

    # Load config file of trained (prior) model to get model parameters
    if parsed_cfg.cfg_file.endswith('json'):
        with open(parsed_cfg.cfg_file) as f:
            loaded_cfg = json.load(f)
    else:
        loaded_cfg = torch.load(parsed_cfg.cfg_file)
    cfg = argparse.Namespace(**loaded_cfg)
    vars(cfg).update(vars(parsed_cfg))
    cfg.eval_only = True
    cfg.generate_dataset = False

    wandb.init(project='madapt', tags=['tune'] + (['debug'] if cfg.debug_mode else []), 
               mode='disabled' if cfg.debug_mode or cfg.suppress_wandb else 'online')
    wandb.config.update(cfg)

    tuner = Tuner(cfg)
    tuner.train_continuous()

    tuner.post_tuning()
    
    return tuner.get_metrics()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('cfg_file', type=str, default='', help='Config file of prior model to load')
    parser.add_argument('-d', '--debug_mode', action='store_true', help="Run with visualization and no wandb")
    parser.add_argument('--suppress_wandb', action='store_true', help="Flag to suppress WandB logging, no visualization.")

    # MACE parameters
    parser.add_argument('--bsz', type=int, default=0, help='Batch size of objects for model update')
    parser.add_argument('--sample_size', type=int, default=64, help='Sample size, can use to sample more than one batch'
                                                                    ' from the same model and run multiple updates.'
                                                                    ' Only useful when grad_steps > 1')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for model update')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations to update model')
    parser.add_argument('--opt_mode', type=str, default='mace', help='Optimization mode for tuning')
    parser.add_argument('--grad_steps', type=int, default=1, help='Number of gradient steps to take before resampling batch')
    parser.add_argument('--sample_prior', action='store_true', help='Flag to sample from initial prior and not previous iteration')
    parser.add_argument('--quantile', type=float, default=0.0625, help='Part of batch to optimize towards')
    parser.add_argument('--seed', type=int, default=0, help='Fake seed parameter to run multiple wandb runs with the same configuration')
    parser.add_argument('--ood_condition', action='store_true', help='Whether to use OOD condition for CVAE and tuning testing')
    
    # Point cloud specific
    parser.add_argument('--chamfer_k', type=int, default=1, help='Number of nearest neighbors for Chamfer distance')

    # IK simulator specific
    parser.add_argument('--ik_object_type', type=str, choices=['wall', 'window', 'inv_bin'] , default='wall', 
                        help='Type of object to use for IK simulation')
    parser.add_argument('--render_mode', type=str, choices=['dist', 'high_q'], help='Type of visualization for IK simulation')
    parser.add_argument('--ik_csv', type=str, default='', help='CSV file of IK object poses')


    parsed_cfg = parser.parse_args()

    main(parsed_cfg)