#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import os
import random
from dataset import load_raw_data
from trainer import Trainer
import yaml
from easydict import EasyDict as edict


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parser_args():
    # load configs/default_config.yaml
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)

    # load training config
    config = yaml.load(open('configs/train_config.yaml'), Loader=yaml.FullLoader)['Train']

    # load dataset config
    config['Dataset'] = default_config['Dataset'][config['dataset']]
    config['Target_Pattern'] = default_config['Target_Pattern'][config['pattern_type']]

    config['Model'] = default_config['Model'][config['model_name']]
    config['Model']['c_out'] = config['Dataset']['num_of_vertices']
    config['Model']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Model']['dec_in'] = config['Dataset']['num_of_vertices']

    config['Surrogate'] = default_config['Model'][config['surrogate_name']]
    config['Surrogate']['c_out'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['enc_in'] = config['Dataset']['num_of_vertices']
    config['Surrogate']['dec_in'] = config['Dataset']['num_of_vertices']

    config = edict(config)
    return config


def _pick_vars(num_vars, count, mode='random', exclude=None):
    """Pick variable indices for spatially constrained attack experiments."""
    exclude = set([] if exclude is None else [int(v) for v in exclude])
    candidates = np.array([i for i in range(num_vars) if i not in exclude])
    if count <= 0:
        raise ValueError('variable count must be positive')
    if candidates.shape[0] == 0:
        raise ValueError('no candidate variables left after exclusion')
    count = min(int(count), candidates.shape[0])

    if mode == 'first':
        return candidates[:count]
    if mode == 'last':
        return candidates[-count:]
    if mode == 'all':
        return candidates
    if mode == 'random':
        return np.random.choice(candidates, size=count, replace=False)
    raise ValueError(f'Unsupported variable selection mode: {mode}')


def build_attack_variables(config, num_vars, device):
    """
    Build two variable sets:
    1. trigger_vars: variables whose historical window is poisoned with triggers;
    2. target_vars: variables whose future labels are replaced by the target pattern.

    Original BackTime uses the same variables for both. Direction-2 experiments can set
    trigger_vars to a single/limited source sensor and target_vars to other sensors.
    """
    trigger_var_num = int(getattr(config, 'trigger_var_num', 0) or 0)
    if trigger_var_num <= 0:
        trigger_var_num = max(int(round(num_vars * config.alpha_s)), 1)

    target_var_num = int(getattr(config, 'target_var_num', 0) or 0)
    if target_var_num <= 0:
        target_var_num = max(int(round(num_vars * config.alpha_s)), 1)

    trigger_mode = getattr(config, 'trigger_var_mode', 'random')
    target_mode = getattr(config, 'target_var_mode', 'same')

    trigger_vars_np = _pick_vars(num_vars, trigger_var_num, mode=trigger_mode)

    if target_mode == 'same':
        target_vars_np = trigger_vars_np.copy()
    elif target_mode == 'all':
        target_vars_np = np.arange(num_vars)
    elif target_mode == 'others':
        target_vars_np = _pick_vars(num_vars, target_var_num, mode='random', exclude=trigger_vars_np)
    elif target_mode in {'random', 'first', 'last'}:
        target_vars_np = _pick_vars(num_vars, target_var_num, mode=target_mode)
    else:
        raise ValueError(f'Unsupported target_var_mode: {target_mode}')

    trigger_vars = torch.from_numpy(np.sort(trigger_vars_np)).long().to(device)
    target_vars = torch.from_numpy(np.sort(target_vars_np)).long().to(device)
    return trigger_vars, target_vars


def main(config):
    # set gpu
    gpuid = config.gpuid
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0')
    print("CUDA:", USE_CUDA, DEVICE)

    seed_torch()

    data_config = config.Dataset
    if not data_config.use_timestamps:
        train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(data_config)
        train_data_stamps = test_data_stamps = None
    else:
        train_mean, train_std, train_data_seq, test_data_seq, train_data_stamps, test_data_stamps = load_raw_data(data_config)

    # set source trigger variables and attacked target variables separately
    trigger_vars, target_vars = build_attack_variables(config, train_data_seq.shape[1], DEVICE)
    print('trigger variables:', trigger_vars.detach().cpu().numpy().tolist())
    print('target variables:', target_vars.detach().cpu().numpy().tolist())

    # load target pattern
    target_pattern = config.Target_Pattern
    target_pattern = torch.tensor(target_pattern).float().to(DEVICE) * train_std

    exp_trainer = Trainer(config, trigger_vars, target_pattern, train_mean, train_std, train_data_seq, test_data_seq,
                          train_data_stamps, test_data_stamps, DEVICE, target_vars=target_vars)

    ckpt_tag = getattr(config, 'checkpoint_tag', None)
    if ckpt_tag is None:
        ckpt_tag = f"{config.dataset}_{getattr(config, 'trigger_var_mode', 'random')}_{getattr(config, 'target_var_mode', 'same')}_tgr{trigger_vars.numel()}_tgt{target_vars.numel()}"
    save_file = f'./checkpoints/attacker_{ckpt_tag}.pth'
    if os.path.exists(save_file):
        state = torch.load(save_file)
        exp_trainer.load_attacker(state)
        print('load attacker from', save_file)
    else:
        print('=' * 20, ' [ Stage 1 ] ', '=' * 20)
        print('start training surrogate model and attacker')
        exp_trainer.train()

        state = exp_trainer.save_attacker()
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(state, save_file)

    print('=' * 20, ' [ Stage 2 ] ', '=' * 20)
    print('start evaluating attack performance on a new model')
    exp_trainer.test()


if __name__ == "__main__":
    config = parser_args()
    main(config)
