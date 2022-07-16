#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

from __future__ import division

import re
from collections import OrderedDict

import torch
from mmcv.runner import Runner, DistSamplerSeedHook, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mono.core import (DistOptimizerHook, DistEvalMonoHook, NonDistEvalHook)
from mono.core import (DistOptimizerHook)
from mono.datasets import build_dataloader
from .env import get_root_logger
#import time
# device = torch.device("cuda:{}".format(cfg.local_rank))
def change_input_variable(data):
    for k, v in data.items():
        # print(k)
        if k[0] != "bev_path":

            if 'kp' not in k:
                data[k] = torch.as_tensor(v).float().cuda()
    return data


def batch_processor(model, data, train_mode):
    data = change_input_variable(data)
    model_out, losses = model(data)
    log_vars = OrderedDict()

    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items())

    log_vars['loss'] = loss
    # print(log_vars)
    new_log_vars=OrderedDict()
    for name in log_vars:
        new_log_vars[str(name)] = log_vars[name].item()

    outputs = dict(loss=loss,
                   log_vars=new_log_vars,
                   num_samples=len(data[('color', 0 , 0)].data))
    # print("********", new_log_vars.keys())
    return outputs


def train_mono(model,
               dataset_train,
               dataset_val,
               cfg,args,
               distributed=False,
               validate=False,
               logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset_train, dataset_val, cfg,args, validate=validate)
    else:
        _non_dist_train(model, dataset_train, dataset_val, cfg, validate=validate)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
            or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_group = {'params': [param]}
            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, dataset_train, dataset_val, cfg,args, validate=False):
    # prepare data loaders
 #   start = time.time()
    data_loaders = [build_dataloader(dataset_train,
                                     cfg.imgs_per_gpu,
                                     cfg.workers_per_gpu,
                                     dist=True)#,
#        build_dataloader(dataset_val,
#                         1,
#                         cfg.workers_per_gpu,
#                         cfg.gpus.__len__(),
#                         dist=False)
    ]
    # data_loaders_val = [build_dataloader(dataset_val,
    #                                  cfg.imgs_per_gpu,
    #                                  cfg.workers_per_gpu,
                                     # dist=False)
                    # ]
  #  end1 = time.time()
  #  print("time of dataloader: ", end1 -start)
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda(),device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
    # build runner
  #  end2 = time.time()
  #  print("time of MMDistributedDataParallel: ", end2 - end1)
    optimizer = build_optimizer(model, cfg.optimizer)
    print('cfg work dir is ', cfg.work_dir)
    runner = Runner(model,
                    batch_processor,
                    optimizer,
                    cfg.work_dir,
                    cfg.log_level)
  #  end3 = time.time()
  #  print("time of runner: ", end3 - end2)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    if validate:
        print('validate........................')
        interval = cfg.get('validate_interval', 1)
        runner.register_hook(DistEvalMonoHook(dataset_val, interval, cfg))
    runner.register_training_hooks(cfg.lr_config,
                                   optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    print("epoch-------",str(runner.epoch))


    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset_train, dataset_val, cfg, validate=False):
    # prepare data loaders
    data_loaders = [
        build_dataloader(dataset_train,
                         cfg.imgs_per_gpu,
                         cfg.workers_per_gpu,
                         cfg.gpus.__len__(),
                         dist=False)
    ]
    print("len of dataloaders", len(data_loaders))
    print("len of workflows", len(cfg.workflow))
    # put model on gpus
    model = MMDataParallel(model, device_ids=cfg.gpus).cuda()
    # build runner
    optimizer = build_optimizer(model,
                                cfg.optimizer)
    runner = Runner(model, batch_processor,
                    optimizer,
                    cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config,
                                   cfg.optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config)

    if validate:
        print('validate........................')
        runner.register_hook(NonDistEvalHook(dataset_val, cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
