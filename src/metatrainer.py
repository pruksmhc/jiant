""" Trainer """
import os
import re
import math
import glob
import time
import copy
import random
import logging as log
import itertools

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_
from tensorboardX import SummaryWriter  # pylint: disable=import-error

from allennlp.common import Params  # pylint: disable=import-error
from allennlp.common.checks import ConfigurationError  # pylint: disable=import-error
from allennlp.data.iterators import BasicIterator, BucketIterator  # pylint: disable=import-error
from allennlp.training.learning_rate_schedulers import LearningRateScheduler  # pylint: disable=import-error
from allennlp.training.optimizers import Optimizer  # pylint: disable=import-error

from .utils import device_mapping, assert_for_log, \
        stop_on_nan_hook, template_print_norm_hook
from .evaluate import evaluate
from . import config
from . import utils
from . import functionize

import ipdb

N_EXS_IN_MEMORY = 100000
EPS = 1e-5

def build_trainer_params(args, task_names):
    ''' In an act of not great code design, we wrote this helper function which
    extracts trainer parameters from args. In particular, we want to search args
    for task specific training parameters. '''
    _get_task_attr = lambda attr_name: config.get_task_attr(args, task_names,
                                                            attr_name)
    params = {}
    train_opts = ['optimizer', 'lr', 'batch_size', 'lr_decay_factor',
                  'task_patience', 'patience',
                  'scheduler_threshold', 'scheduler',
                  'sim_lr', 'max_sim_grad_norm',
                  'slow_params_approx', 'only_pos_reg', 'approx_term',
                  'one_sided_update']

    # we want to pass to the build_train()
    extra_opts = ['sent_enc', 'd_hid', 'warmup',
                  'max_grad_norm', 'min_lr', 'batch_size',
                  'cuda', 'keep_all_checkpoints',
                  'val_data_limit', 'training_data_fraction']
    for attr in train_opts:
        params[attr] = _get_task_attr(attr)
    for attr in extra_opts:
        params[attr] = getattr(args, attr)
    params['max_vals'] = _get_task_attr('max_vals')
    params['val_interval'] = _get_task_attr('val_interval')
    params['dec_val_scale'] = _get_task_attr('dec_val_scale')

    return Params(params)

def build_optimizer(params):
    ''' Build optimizer params '''
    opt_params = {'type': params['optimizer'], 'lr': params['lr'], 'weight_decay': 0}
    if params['optimizer'] == 'adam':
        # AMSGrad is a flag variant of Adam, not its own object.
        opt_params['amsgrad'] = True
    return Params(opt_params)

def build_scheduler(params, should_decrease=True):
    ''' Build scheduler params '''
    schd_type = params['scheduler']
    if 'transformer' in params['sent_enc'] or schd_type == 'noam':
        assert False, "Transformer is not yet tested, still in experimental stage :-("
        schd_params = Params({'type': schd_type,
                              'model_size': params['d_hid'],
                              'warmup_steps': params['warmup'],
                              'factor': 1.0})
        log.info('\tUsing noam scheduler with warmup %d!', params['warmup'])
    elif schd_type == 'cosine':
        # TODO(Alex): lots of other parameters to set
        schd_params = Params({'type': schd_type,
                              'T_max': params['max_vals']
                             })
        log.info('\tUsing cosine scheduler!')
    elif schd_type == 'reduce_on_plateau':
        schd_params = Params({'type': schd_type,
                              'mode': 'min' if should_decrease else 'max',
                              'factor': params['lr_decay_factor'],
                              'patience': params['task_patience'],
                              'threshold': params['scheduler_threshold'],
                              'threshold_mode': 'abs',
                              'verbose': True})
        log.info('\tUsing ReduceLROnPlateau scheduler!')
    else:
        raise ValueError("Scheduler %s not supported!" % schd_type)

    return schd_params


def build_trainer(params, model, model_copy, run_dir, metric_should_decrease=True):
    '''Build a trainer from params.

    Parameters
    ----------
    args: A trainer config object.
    model: A module with trainable parameters.
    max_vals: The upper bound on training steps, specified in number of validation runs.

    Returns
    -------
    A trainer object, a trainer config object, an optimizer config object,
        and a scheduler config object.
    '''

    opt_params = build_optimizer(params)
    schd_params = build_scheduler(params, metric_should_decrease)

    train_params = Params({'cuda_device': params['cuda'],
                           'patience': params['patience'],
                           'max_grad_norm': params['max_grad_norm'],
                           'val_interval': params['val_interval'],
                           'max_vals': params['max_vals'],
                           'lr_decay': .99, 'min_lr': params['min_lr'],
                           'keep_all_checkpoints': params['keep_all_checkpoints'],
                           'val_data_limit': params['val_data_limit'],
                           'dec_val_scale': params['dec_val_scale'],
                           'training_data_fraction': params['training_data_fraction'],
                           'sim_lr': params['sim_lr'],
                           'max_sim_grad_norm': params['max_sim_grad_norm'],
                           'slow_params_approx': params['slow_params_approx'],
                           'only_pos_reg': params['only_pos_reg'], 'approx_term': params['approx_term'],
                           'one_sided_update': params['one_sided_update']})
    trainer = MetaMultiTaskTrainer.from_params(model, model_copy, run_dir,
                                               copy.deepcopy(train_params))
    return trainer, train_params, opt_params, schd_params


def simulate_sgd(model, params, task, batch, sim_lr=0.01, n_steps=1):
    ''' Given original parameters and a batch of inputs and outputs,
    compute the model's loss evaluated on the given inputs and parameters.
    Then compute the gradient of the loss and update the original params.

    Args:
        - model (torch.nn.Module): PyTorch model
        - batch (Dict[str:torch.Tensor]): dictionary of inputs and outputs
        - sim_lr (float): LR for the simulated update

    Returns:
        - cand_params (List[torch.Tensor]): list of candidate parameter values
        - sim_loss (float?): loss of the model with orig params and the given input on the output
    '''
    sim_out = model(task, batch)
    sim_loss = sim_out['loss']
    grad_orig_params = autograd.grad(sim_loss, params, create_graph=True, allow_unused=True)
    assert sum([g is not None for g in grad_orig_params]) != 0, "All gradients are None!"
    cand_params = [p + (-sim_lr * g) if g is not None else p for g, p in zip(grad_orig_params, params)]
    return cand_params, sim_out, grad_orig_params


def set_model_params(model, params):
    ''' Set model's parameters to params.
    Assume that params is the required sizes and shapes for model. '''
    for (_, old_p), new_p in zip(model.named_parameters(), params):
        old_p.data = new_p


class MetaMultiTaskTrainer():
    def __init__(self, model, model_copy, patience=2, val_interval=100, max_vals=50,
                 serialization_dir=None, cuda_device=-1,
                 max_grad_norm=None, lr_decay=None, min_lr=None,
                 keep_all_checkpoints=False, val_data_limit=5000,
                 dec_val_scale=100, training_data_fraction=1.0,
                 sim_lr=0.001, max_sim_grad_norm=5.,
                 slow_params_approx=0, only_pos_reg=0, approx_term=0,
                 one_sided_update=0):
        """
        The training coordinator. Unusually complicated to handle MTL with tasks of
        diverse sizes.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        patience , optional (default=2)
            Number of epochs to be patient before early stopping.
        val_metric , optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        serialization_dir , optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device , optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        max_grad_norm : float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        learning_rate_scheduler : PytorchLRScheduler, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the ``val_metric`` provided to determine if learning has plateaued.
        keep_all_checkpoints : If set, keep checkpoints from every validation. Otherwise, keep only
            best and (if different) most recent.
        val_data_limit: During training, use only the first N examples from the validation set.
            Set to -1 to use all.
        training_data_fraction: If set to a float in [0, 1], load only the specified percentage
            of examples. Hashing is used to ensure that the same examples are loaded each epoch.
        """
        self._model = model
        self._model_copy = model_copy

        self._patience = patience
        self._max_vals = max_vals
        self._val_interval = val_interval
        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        self._max_grad_norm = max_grad_norm
        self._max_sim_grad_norm = max_sim_grad_norm
        self._lr_decay = lr_decay
        self._min_lr = min_lr
        self._sim_lr = sim_lr
        self._slow_params_approx = slow_params_approx
        self._only_pos_reg = only_pos_reg
        self._approx_term = approx_term
        self._one_sided_update = one_sided_update
        self._keep_all_checkpoints = keep_all_checkpoints
        self._val_data_limit = val_data_limit
        self._dec_val_scale = dec_val_scale
        self._training_data_fraction = training_data_fraction

        self._task_infos = None
        self._metric_infos = None

        self._log_interval = 10  # seconds
        self._summary_interval = 10  # num batches between log to tensorboard
        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

        self._tb_writers = None
        if self._serialization_dir is not None:
            tb_dir = os.path.join(self._serialization_dir, "tensorboard")
            self._tb_writers = {"train": SummaryWriter(os.path.join(tb_dir, "train")),
                                "val": SummaryWriter(os.path.join(tb_dir, "val"))}
            if slow_params_approx:
                self._tb_writers["grad"] = SummaryWriter(os.path.join(tb_dir, "grad"))
                self._tb_writers["gross_loss"] = SummaryWriter(os.path.join(tb_dir, "gross_loss"))
                self._tb_writers["net_loss"] = SummaryWriter(os.path.join(tb_dir, "net_loss"))
                self._tb_writers["grad1"] = SummaryWriter(os.path.join(tb_dir, "grad1"))
                self._tb_writers["grad2"] = SummaryWriter(os.path.join(tb_dir, "grad2"))

    def _check_history(self, metric_history, cur_score, should_decrease=False):
        '''
        Given a the history of the performance on a metric
        and the current score, check if current score is
        best so far and if out of patience.
        '''
        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == cur_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        out_of_patience = False
        if should_decrease:
            index_of_last_improvement = metric_history.index(min(metric_history))
            out_of_patience = index_of_last_improvement <= len(metric_history) - (patience + 1)
        else:
            index_of_last_improvement = metric_history.index(max(metric_history))
            out_of_patience = index_of_last_improvement <= len(metric_history) - (patience + 1)

        return best_so_far, out_of_patience

    def _setup_training(self, tasks, batch_size, train_params, optimizer_params, scheduler_params, phase):
        ''' Set up the trainer by initializing task_infos and metric_infos, which
        track necessary information about the training status of each task and metric respectively.

        Returns:
            - task_infos (Dict[str:Dict[str:???]]): dictionary containing where each task_info contains:
                - iterator: a task specific (because it uses that task's fields to dynamically batch) batcher
                - n_tr_batches: the number of training batches
                - tr_generator: generator object that returns the batches, set to repeat indefinitely
                - loss: the accumulated loss (during training or validation)
                - n_batches_since_val: number of batches trained on since the last validation
                - total_batches_trained: number of batches trained over all validation checks
                - optimizer: a task specific optimizer, not used if the global optimizer is not None
                - scheduler: a task specific scheduler, not used if the global optimizer is not None
                - stopped: a bool indicating if that task is stopped or not (if it ran out of patience or hit min lr)
                - last_log: the time we last logged progress for the task

            - metric_infos (Dict[str:Dict[str:???]]): dictionary containing metric information.
                Each metric should be the validation metric of a task, except {micro/macro}_avg,
                which are privileged to get an aggregate multi-task score. Each dict contains:
                - hist (List[float]): previous values of the metric
                - stopped (Bool): whether or not that metric is stopped or not
                - best (Tuple(Int, Dict)): information on the best value of that metric and when it happened
        '''
        task_infos = {task.name: {} for task in tasks}
        for task in tasks:
            task_info = task_infos[task.name]

            # Adding task-specific smart iterator to speed up training
            instance = [i for i in itertools.islice(task.train_data, 1)][0]
            pad_dict = instance.get_padding_lengths()
            sorting_keys = []
            for field in pad_dict:
                for pad_field in pad_dict[field]:
                    sorting_keys.append((field, pad_field))
            iterator = BucketIterator(sorting_keys=sorting_keys,
                                      max_instances_in_memory=N_EXS_IN_MEMORY,
                                      batch_size=batch_size,
                                      #padding_noise=0.0, # TODO(Alex): DELETE WHEN DONE
                                      biggest_batch_first=True)
            tr_generator = iterator(task.train_data, num_epochs=None, cuda_device=self._cuda_device)

            task_info['iterator'] = iterator

            if phase == "main":
                # Warning: This won't be precise when training_data_fraction is set, since each example is included
                #   or excluded independantly using a hashing function. Fortunately, it doesn't need to be.
                task_info['n_tr_batches'] = math.ceil(task.n_train_examples * self._training_data_fraction / batch_size)
            else:
                task_info['n_tr_batches'] = math.ceil(task.n_train_examples / batch_size)

            task_info['tr_generator'] = tr_generator
            task_info['loss'] = 0.0
            #task_info['sim_loss'] = 0.0
            task_info['total_batches_trained'] = 0
            task_info['n_batches_since_val'] = 0
            task_info['optimizer'] = Optimizer.from_params(train_params,
                                                           copy.deepcopy(optimizer_params))
            task_info['scheduler'] = LearningRateScheduler.from_params(
                task_info['optimizer'], copy.deepcopy(scheduler_params))
            task_info['stopped'] = False
            task_info['last_log'] = time.time()
        # Metric bookkeeping
        all_metrics = [task.val_metric for task in tasks] + ['micro_avg', 'macro_avg']
        metric_infos = {metric: {'hist': [], 'stopped': False, 'best': (-1, {})} for
                        metric in all_metrics}
        self._task_infos = task_infos
        self._metric_infos = metric_infos
        return task_infos, metric_infos

    def _setup_task_weighting(self, weighting_method, tasks):
        ''' Do some stuff related to task weighting '''
        task_infos = self._task_infos

        if weighting_method == 'uniform':
            log.info("Sampling tasks uniformly")
        elif weighting_method == 'proportional':
            log.info("Sampling tasks proportional to number of training batches")
        elif weighting_method == 'proportional_log_batch':
            log.info("Sampling tasks proportional to log number of training batches")
        elif weighting_method == 'proportional_log_example':
            log.info("Sampling tasks proportional to log number of training examples")
        elif weighting_method == 'inverse_example':
            log.info("Sampling tasks inverse to number of training examples")
        elif weighting_method == 'inverse_batch':
            log.info("Sampling tasks inverse to number of training batches")
        elif weighting_method == 'inverse_log_example':
            log.info("Sampling tasks inverse to log number of training examples")
        elif weighting_method == 'inverse_log_batch':
            log.info("Sampling tasks inverse to log number of training batches")
        elif 'power_' in weighting_method:
            log.info("Sampling tasks with %s", weighting_method.replace('_', ' of '))
        elif 'softmax_' in weighting_method:
            log.info("Sampling tasks with %s", weighting_method.replace('_', ' of temperature '))

        if weighting_method == 'uniform':
            sample_weights = [1] * len(tasks)
        elif weighting_method == 'proportional':
            sample_weights = [task_infos[task.name]['n_tr_batches'] for task in tasks]
        elif weighting_method == 'proportional_log_batch':  # log(training batch)
            sample_weights = [math.log(task_infos[task.name]['n_tr_batches']) for task in tasks]
        elif weighting_method == 'proportional_log_example':  # log(training example)
            sample_weights = [math.log(task.n_train_examples) for task in tasks]
        elif weighting_method == 'inverse_example':  # 1/training example
            sample_weights = [(1 / task.n_train_examples) for task in tasks]
        elif weighting_method == 'inverse_batch':  # 1/training batch
            sample_weights = [(1 / task_infos[task.name]['n_tr_batches']) for task in tasks]
        elif weighting_method == 'inverse_log_example':  # 1/log(training example)
            sample_weights = [(1 / math.log(task.n_train_examples)) for task in tasks]
        elif weighting_method == 'inverse_log_batch':  # 1/log(training batch)
            sample_weights = [(1 / math.log(task_infos[task.name]['n_tr_batches']))
                              for task in tasks]
        elif 'power_' in weighting_method:  # x ^ power
            weighting_power = float(weighting_method.strip('power_'))
            sample_weights = [(task.n_train_examples ** weighting_power) for task in tasks]
        elif 'softmax_' in weighting_method:  # exp(x/temp)
            weighting_temp = float(weighting_method.strip('softmax_'))
            sample_weights = [math.exp(task.n_train_examples/weighting_temp) for task in tasks]
        log.info("Weighting details: ")
        log.info("\ttask.n_train_examples: %s", str([(task.name, task.n_train_examples) for task in tasks]))
        normalized_sample_weights = [i / sum(sample_weights) for i in sample_weights]
        log.info("\tnormalized_sample_weights: %s", str(normalized_sample_weights))
        return sample_weights

    def train(self, tasks, stop_metric,
              batch_size, n_batches_per_pass,
              weighting_method, scaling_method,
              train_params, optimizer_params, scheduler_params,
              shared_optimizer=1, load_model=1, phase="main",
              pseudo_meta=False,
              multistep_loss=False, multistep_scale=.1):
        """
        The main training loop.
        Training will stop if we run out of patience or hit the minimum learning rate.

        Parameters
        ----------
        tasks: A list of task objects to train on.
        stop_metric: The metric to use for early stopping.
        validation_interval: How many passes between evaluations.
        n_batches_per_pass: How many training steps per task per pass.
        weighting_method: How to sample which task to use.
        scaling_method: How to scale gradients.
        train_params: Trainer config object.
        optimizer_params: Optimizer config object.
        scheduler_params: Scheduler config object.
        shared_optimizer: Use a single optimizer object for all tasks in MTL. Recommended.
        load_model: Whether to restore and continue training if a checkpoint is found.
        slow_params_approx: Whether to assume candidate params ~= params.
        phase: Usually 'main' or 'eval'.

        Returns
        -------
        Validation results
        """
        validation_interval = self._val_interval
        assert_for_log(validation_interval % 2 == 0, "Need an even validation interval!")
        sim_lr, slow_params_approx = self._sim_lr, self._slow_params_approx
        task_infos, metric_infos = self._setup_training(tasks, batch_size, train_params,
                                                        optimizer_params, scheduler_params, phase)

        if shared_optimizer: # if shared_optimizer, ignore task_specific optimizers
            g_optimizer = Optimizer.from_params(train_params, copy.deepcopy(optimizer_params))
            g_scheduler = LearningRateScheduler.from_params(
                g_optimizer, copy.deepcopy(scheduler_params))
        else:
            g_optimizer, g_scheduler = None, None
        self._g_optimizer = g_optimizer
        self._g_scheduler = g_scheduler

        n_update, should_stop = 0, False  # define these here b/c they might get overridden on load
        if self._serialization_dir is not None and phase != "eval":  # Resume from serialization pth
            if load_model and any(
                    ["model_state_" in x for x in os.listdir(self._serialization_dir)]):
                n_update, should_stop = self._restore_checkpoint()
                log.info("Loaded model from checkpoint. Starting at pass %d.", n_update)
            else:
                log.info("Not loading.")
                checkpoint_pattern = os.path.join(
                    self._serialization_dir, "*_{}_*.th".format(phase))
                assert_for_log(len(glob.glob(checkpoint_pattern)) == 0,
                               "There are existing checkpoints in %s which will be overwritten. "
                               "Use load_model = 1 to load the checkpoints instead. "
                               "If you don't want them, delete them or change your experiment name." %
                               self._serialization_dir)

        sample_weights = self._setup_task_weighting(weighting_method, tasks)
        model = self._model
        self._all_params = [p for p in self._model.parameters()]
        idxs_and_params = [(i, p) for i, (n, p) in enumerate(model.sent_encoder.named_parameters())\
                           if p.requires_grad and not ('_phrase_layer' in n and 'embed' in n)]
        self._shared_params_idxs, self._shared_params = zip(*idxs_and_params)

        # Sample the tasks to train on. Do it all at once (val_interval) for MAX EFFICIENCY.
        #samples_src = random.choices(tasks, weights=sample_weights, k=validation_interval)
        #samples_trg = random.choices(tasks, weights=sample_weights, k=validation_interval)
        # TODO(Alex): this is a hack to make sure tasks are different each time
        samples_src = [tasks[0] for _ in range(validation_interval)]
        samples_trg = [tasks[1] for _ in range(validation_interval)]

        all_tr_metrics = {}
        log.info("Beginning training. Stopping metric: %s", stop_metric)
        while not should_stop:
            self._model.train()
            src_task = samples_src[n_update % (validation_interval)]
            trg_task = samples_trg[n_update % (validation_interval)]
            src_task_info = task_infos[src_task.name]
            trg_task_info = task_infos[trg_task.name]
            if src_task_info['stopped'] or trg_task_info['stopped']:
                continue
            src_gen, trg_gen = src_task_info['tr_generator'], trg_task_info['tr_generator']
            optimizer = g_optimizer if shared_optimizer else src_task_info['optimizer']
            scheduler = g_scheduler if shared_optimizer else src_task_info['scheduler']
            for src_batch, trg_batch in zip(itertools.islice(src_gen, n_batches_per_pass),
                                            itertools.islice(trg_gen, n_batches_per_pass)):
                src_task_info['n_batches_since_val'] += 1
                trg_task_info['n_batches_since_val'] += 1
                src_task_info['total_batches_trained'] += 1
                trg_task_info['total_batches_trained'] += 1
                n_update += 2 # update per batch
                optimizer.zero_grad()

                ### START DOING META STUFF ###
                if slow_params_approx: # assume cand_params ~= params
                    trg_out = self._model(trg_task, trg_batch)
                    src_out = self._model(src_task, src_batch)
                    trg_grads = self._get_gradient(trg_out['loss'], self._shared_params)
                    src_grads = self._get_gradient(src_out['loss'], self._shared_params)

                    # Get the regularization term, and other stuff for tracking
                    regularizer, dot_prod, cos_sim, trg_norm, src_norm = \
                        self._get_approx_regularizer(trg_grads, src_grads)

                    # Loss computation
                    gross_loss = src_out['loss'] + trg_out['loss']
                    if pseudo_meta: # TODO(Alex): delete when done
                        loss = gross_loss
                    else:
                        loss = gross_loss - (sim_lr * regularizer)
                    if self._approx_term == 'only_cos_sim':
                        loss = -cos_sim
                    loss.backward()

                else: # exact case
                    # NOTE(AW): I'm pretty sure the autograd.grad calls don't need create_graph b/c I already
                    #           created graph on the backwards calls
                    src_grads, sim_trg_grads, trg_out = \
                            self._get_meta_gradients(src_task, src_batch, trg_task, trg_batch)
                    if not self._one_sided_update:
                        trg_grads, sim_src_grads, src_out = \
                                self._get_meta_gradients(trg_task, trg_batch, src_task, src_batch)
                        gradients = [trg_grads, src_grads]
                        sim_gradss = [sim_trg_grads, sim_src_grads] if multistep_loss else []
                    else:
                        gradients = [src_grads]
                        sim_gradss = [sim_trg_grads] if multistep_loss else []
                    self._assign_gradients(gradients, sim_gradss, multistep_scale)

                trg_loss = trg_out['loss'].item()
                src_loss = src_out['loss'].item()
                trg_task_info['loss'] += trg_loss
                src_task_info['loss'] += src_loss
                loss = trg_loss + src_loss

                # Gradient regularization and application
                if self._max_grad_norm:
                    clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                optimizer.step()

                # step scheduler if it's not ReduceLROnPlateau
                if not isinstance(scheduler.lr_scheduler, ReduceLROnPlateau):
                    scheduler.step_batch(n_update)

            # Intermediate log to logger and tensorboard
            if time.time() - src_task_info['last_log'] > self._log_interval:
                src_task_metrics = src_task.get_metrics()
                trg_task_metrics = trg_task.get_metrics()
                src_nbsv = src_task_info['n_batches_since_val']
                trg_nbsv = trg_task_info['n_batches_since_val']

                src_task_metrics["%s_loss" % src_task.name] = src_task_info['loss'] / src_nbsv
                trg_task_metrics["%s_loss" % trg_task.name] = trg_task_info['loss'] / trg_nbsv
                src_description = self._description_from_metrics(src_task_metrics)
                trg_description = self._description_from_metrics(trg_task_metrics)
                log.info("Update %d: src_task %s, batch %d (%d): %s", n_update, src_task.name, src_nbsv,
                         src_task_info['total_batches_trained'], src_description)
                log.info("\ttrg_task %s, batch %d (%d): %s", trg_task.name, trg_nbsv,
                         trg_task_info['total_batches_trained'], trg_description)
                if slow_params_approx:
                    log.info("\tnet loss: %.3f, src loss: %.3f, trg loss: %.3f", loss, src_loss, trg_loss)
                    if pseudo_meta:
                        log.info("\tgross loss: %.3f", gross_loss)
                    log.info("\tgrad regularizer: %.5f, cos_sim: %.5f", regularizer, cos_sim)
                    log.info("\tgrad1 norm: %.3f, grad2 norm: %.3f", src_norm, trg_norm)
                    self._tb_writers["gross_loss"].add_scalar("approx/loss", gross_loss, n_update)
                    self._tb_writers["grad"].add_scalar("approx/grad_prod", regularizer, n_update)
                    self._tb_writers["net_loss"].add_scalar("approx/loss", loss, n_update)
                    self._tb_writers["grad"].add_scalar("approx/cos_sim", cos_sim, n_update)
                    self._tb_writers["grad1"].add_scalar("approx/grad_mag", src_norm, n_update)
                    self._tb_writers["grad2"].add_scalar("approx/grad_mag", trg_norm, n_update)
                else:
                    log.info("\tupdate loss: %.3f, src loss %.3f, trg loss %.3f", loss,
                             src_loss, trg_loss)


                if self._tb_writers is not None:
                    # TODO(Alex): I don't know why we need to copy
                    src_task_metrics_to_tb = src_task_metrics.copy()
                    src_task_metrics_to_tb["loss"] = float(src_task_info['loss'] / src_nbsv)
                    self._write_tensorboard(n_update, src_task_metrics_to_tb, src_task.name)
                    trg_task_metrics_to_tb = trg_task_metrics.copy()
                    trg_task_metrics_to_tb["loss"] = float(trg_task_info['loss'] / trg_nbsv)
                    self._write_tensorboard(n_update, trg_task_metrics_to_tb, trg_task.name)

                if self._model.utilization is not None:
                    batch_util = self._model.utilization.get_metric()
                    log.info("TRAINING BATCH UTILIZATION: %.3f", batch_util)

                src_task_info['last_log'] = time.time()

            # Validation
            if n_update % (validation_interval) == 0:

                # Dump and log all of our current info
                epoch = int(n_update / validation_interval)
                log.info("***** Pass %d / Epoch %d *****", n_update, epoch)
                # Get metrics for all training progress so far
                for task in tasks:
                    task_info = task_infos[task.name]
                    n_batches_since_val = task_info['n_batches_since_val']
                    if n_batches_since_val > 0:
                        task_metrics = task.get_metrics(reset=True)
                        for name, value in task_metrics.items():
                            all_tr_metrics["%s_%s" % (task.name, name)] = value
                        all_tr_metrics["%s_loss" % task.name] = \
                            float(task_info['loss'] / n_batches_since_val)
                    else:
                        all_tr_metrics["%s_loss" % task.name] = 0.0
                    log.info("%s: trained on %d batches, %.3f epochs", task.name,
                             n_batches_since_val, n_batches_since_val / task_info['n_tr_batches'])

                if self._model.utilization is not None:
                    batch_util = self._model.utilization.get_metric(reset=True)
                    log.info("TRAINING BATCH UTILIZATION: %.3f", batch_util)

                # Validate
                log.info("Validating...")
                preds_file_path_dict = {task.name: os.path.join(
                    self._serialization_dir,
                    "preds_{}{}_{}_epoch_{}.txt".format(
                        time.time(), task.name, phase, epoch)) for task in tasks}
                all_val_metrics, should_save, new_best_macro = self._validate(
                    epoch, tasks, batch_size, periodic_save=(phase != "eval"), preds_file_path_dict=preds_file_path_dict)

                # Check stopping conditions
                should_stop = self._check_stop(epoch, stop_metric, tasks)

                # Log results to logger and tensorboard
                for name, value in all_val_metrics.items():
                    log.info("Statistic: %s", name)
                    if name in all_tr_metrics:
                        log.info("\ttraining: %3f", all_tr_metrics[name])
                    log.info("\tvalidation: %3f", value)
                if self._tb_writers is not None:
                    self._write_tensorboard(n_update, all_val_metrics, task_name="", train_split=False)
                lrs = self._get_lr() # log LR
                for name, value in lrs.items():
                    log.info("%s: %.6f", name, value)
                elmo_params = self._model.get_elmo_mixing_weights(tasks)
                if elmo_params: # log ELMo mixing weights
                    for task_name, task_params in elmo_params.items():
                        log.info("ELMo mixing weights for {}:".format(task_name))
                        log.info("\t" + ", ".join(["{}: {:.6f}".format(layer, float(param))
                                                   for layer, param in task_params.items()]))

                # Reset training progress
                all_tr_metrics = {}
                #samples_src = random.choices(tasks, weights=sample_weights, k=validation_interval)
                #samples_trg = random.choices(tasks, weights=sample_weights, k=validation_interval)
                samples_src = [tasks[0] for _ in range(validation_interval)]
                samples_trg = [tasks[1] for _ in range(validation_interval)]

                if should_save:
                    self._save_checkpoint(
                        {"pass": n_update, "epoch": epoch, "should_stop": should_stop},
                        phase=phase, new_best_macro=new_best_macro)

        log.info('Stopped training after %d validation checks', n_update / validation_interval)
        return self._aggregate_results(tasks, task_infos, metric_infos)

    def _aggregate_results(self, tasks, task_infos, metric_infos):
        ''' Helper function to print results after finishing training '''
        results = {}
        for task in tasks:
            task_info = task_infos[task.name]
            log.info('Trained %s for %d batches or %.3f epochs',
                     task.name, task_info['total_batches_trained'],
                     task_info['total_batches_trained'] / task_info['n_tr_batches'])
            results[task.name] = metric_infos[task.val_metric]['best'][0]  # * validation_interval
        results['micro'] = metric_infos['micro_avg']['best'][0]  # * validation_interval
        results['macro'] = metric_infos['macro_avg']['best'][0]  # * validation_interval
        log.info('***** VALIDATION RESULTS *****')
        for metric in metric_infos.keys():
            best_epoch, epoch_metrics = metric_infos[metric]['best']
            all_metrics_str = ', '.join(['%s: %.5f' % (metric, score) for
                                         metric, score in epoch_metrics.items()])
            log.info('%s, %d, %s', metric, best_epoch, all_metrics_str)
        return results

    def _validate(self, epoch, tasks, batch_size, preds_file_path_dict, periodic_save=True):
        ''' Validate on all tasks and return the results and whether to save this epoch or not '''
        task_infos, metric_infos = self._task_infos, self._metric_infos
        g_scheduler = self._g_scheduler
        self._model.eval()
        all_val_metrics = {("%s_loss" % task.name): 0.0 for task in tasks}
        all_val_metrics["macro_avg"] = 0.0
        all_val_metrics["micro_avg"] = 0.0
        n_examples_overall = 0.0

        # Get validation numbers for each task
        for task in tasks:
            n_examples, batch_num = 0, 0
            task_info = task_infos[task.name]
            task.preds_file_path = preds_file_path_dict[task.name]

            # to speed up training, we evaluate on a subset of validation data
            if self._val_data_limit >= 0:
                max_data_points = min(task.n_val_examples, self._val_data_limit)
            else:
                max_data_points = task.n_val_examples
            val_generator = BasicIterator(batch_size, instances_per_epoch=max_data_points)(
                task.val_data, num_epochs=1, shuffle=False,
                cuda_device=self._cuda_device)
            n_val_batches = math.ceil(max_data_points / batch_size)
            all_val_metrics["%s_loss" % task.name] = 0.0

            for batch in val_generator:
                batch_num += 1
                out = self._forward(batch, task=task, for_training=False)
                loss = out["loss"]
                all_val_metrics["%s_loss" % task.name] += loss.data.cpu().numpy()
                n_examples += out["n_exs"]

                # log
                if time.time() - task_info['last_log'] > self._log_interval:
                    task_metrics = task.get_metrics()
                    task_metrics["%s_loss" % task.name] = \
                            all_val_metrics["%s_loss" % task.name] / batch_num
                    description = self._description_from_metrics(task_metrics)
                    log.info("Batch %d/%d: %s", batch_num, n_val_batches, description)
                    task_info['last_log'] = time.time()
            assert batch_num == n_val_batches

            # Get task validation metrics and store in all_val_metrics
            task_metrics = task.get_metrics(reset=True)
            for name, value in task_metrics.items():
                all_val_metrics["%s_%s" % (task.name, name)] = value
            all_val_metrics["%s_loss" % task.name] /= batch_num  # n_val_batches
            if task.val_metric_decreases and len(tasks) > 1:
                all_val_metrics["micro_avg"] += (1 - all_val_metrics[task.val_metric] / self._dec_val_scale) * n_examples
                all_val_metrics["macro_avg"] += (1 - all_val_metrics[task.val_metric] / self._dec_val_scale)
            else:
                # triggers for single-task cases and during MTL when task val metric increases
                all_val_metrics["micro_avg"] += all_val_metrics[task.val_metric] * n_examples
                all_val_metrics["macro_avg"] += all_val_metrics[task.val_metric]
            n_examples_overall += n_examples

            # Reset training progress
            task_info['n_batches_since_val'] = 0
            task_info['loss'] = 0

        all_val_metrics['micro_avg'] /= n_examples_overall
        all_val_metrics['macro_avg'] /= len(tasks)

        # Track per task patience
        should_save = periodic_save  # whether to save this epoch or not.
        # Currently we save every validation in the main training runs.
        new_best_macro = False  # whether this epoch is a new best

        for task in tasks + ['micro', 'macro']:
            if task in ['micro', 'macro']:
                metric = "%s_avg" % task
                metric_decreases = tasks[0].val_metric_decreases if len(tasks) == 1 else False
            else:
                metric = task.val_metric
                metric_decreases = task.val_metric_decreases
                task = task.name
            if metric_infos[metric]['stopped']:
                continue
            this_epoch_metric = all_val_metrics[metric]
            metric_history = metric_infos[metric]['hist']
            metric_history.append(this_epoch_metric)
            is_best_so_far, out_of_patience = \
                self._check_history(metric_history, this_epoch_metric, metric_decreases)
            if is_best_so_far:
                log.info("Best model found for %s.", task)
                metric_infos[metric]['best'] = (epoch, all_val_metrics)
                should_save = True
                if task == 'macro':
                    new_best_macro = True
            if out_of_patience:
                if periodic_save:
                    should_save = True
                metric_infos[metric]['stopped'] = True
                log.info("Out of patience. Stopped tracking %s", task)

            # Get scheduler, using global scheduler if exists and task is macro
            # micro has no scheduler updates
            if hasattr(task, 'name') and g_scheduler is None:
                scheduler = task_infos[task.name]['scheduler']
            elif g_scheduler is not None and task == 'macro':
                scheduler = g_scheduler
            else:
                scheduler = None
            if scheduler is not None and isinstance(scheduler.lr_scheduler, ReduceLROnPlateau):
                log.info("Advancing scheduler.")
                scheduler.step(this_epoch_metric, epoch)
                log.info("\tBest %s: %.3f", metric, scheduler.lr_scheduler.best)
                log.info("\t# bad epochs: %d", scheduler.lr_scheduler.num_bad_epochs)

        return all_val_metrics, should_save, new_best_macro

    def _get_gradient(self, loss, params):
        """ Explicitly get gradient of loss """

        grads = autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads_flat = torch.cat([g.view(-1) for g in grads if g is not None])
        return grads_flat

    def _get_approx_regularizer(self, grad1, grad2):
        """ Compute the regularization term in the slow-params approx setting """
        approx_term = self._approx_term
        dot_prod = torch.dot(grad1, grad2)
        norm1 = grad1.norm() + EPS
        norm2 = grad2.norm() + EPS

        if self._only_pos_reg: # if grad_prod is negative, it's added to loss and that's ok
            regularizer = torch.max(dot_prod, 0.)
        cos_sim = dot_prod / (norm1 * norm2)
        if approx_term == 'cos_sim':
            regularizer = cos_sim
        elif approx_term == 'sign_cos_sim':
            regularizer = torch.sign(cos_sim)
        elif approx_term == 'dot_product':
            # grad_prod is already dot product
            max_sim_grad_norm = self._max_sim_grad_norm
            if max_sim_grad_norm is not None and norm1 > max_sim_grad_norm:
                regularizer = (max_sim_grad_norm / norm1) * dot_prod
            if max_sim_grad_norm is not None and norm2 > max_sim_grad_norm:
                regularizer = (max_sim_grad_norm / norm2) * dot_prod
        elif approx_term == 'only_cos_sim':
            pass
        else:
            raise ValueError("Regularization method %s not found!" % approx_term)
        return regularizer, dot_prod, cos_sim, norm1, norm2

    def _get_meta_gradients(self, task1, batch1, task2, batch2):
        """ """
        model = self._model
        model_copy = self._model_copy
        shared_params = self._shared_params
        shared_params_idxs = self._shared_params_idxs
        model_copy.zero_grad()

        # clone the parameters and create the simulated params
        cand_params, sim_out, sim_grads = \
            simulate_sgd(model, shared_params, task1, batch1, sim_lr=self._sim_lr)
        all_cand_params = [p for p in self._all_params]
        for j, i in enumerate(shared_params_idxs):
            all_cand_params[i] = cand_params[j]

        # set the model copy parameters to be the simulated params
        set_model_params(model_copy, all_cand_params)

        # do a forward pass of the model copy and backward to get gradients
        out = model_copy(task2, batch2)
        out['loss'].backward(create_graph=True, retain_graph=True)

        # gather the model copy gradients and backwards propagate them
        cpy_grads = [w.grad for w in model_copy.parameters()] # grads of mdl cpy params
        cpy_grads_nonnone = [g for g in cpy_grads if g is not None]
        cand_params_w_grad = [p for i, p in enumerate(all_cand_params) if cpy_grads[i] is not None]
        # differentiate candidate params w/ gradient WRT original params that need gradient
        # this will do a continuation of derivative of loss WRT original params
        meta_grads = autograd.grad(cand_params_w_grad, shared_params,
                                   grad_outputs=cpy_grads_nonnone, create_graph=False,
                                   allow_unused=True)
        return meta_grads, sim_grads, out

    def _assign_gradients(self, grads, sim_gradss, multistep_scale):
        """ Assign gradients and pssible simulated gradients
        to the original parameters

        args:
            - gradss (List[List[torch.Tensor]]): list of gradients to assign for each task
            - sim_grads (List[List[List[torch.Tensor]]]):
                list of gradients for each simulated SGD step (earliest step first) for each task
            - step_scale (float): multiplicative discount rate to apply
                based on the SGD step

        returns: (none)
        """
        # TODO(Alex): should add an assert here that lengths are correct
        for i, j in enumerate(self._shared_params_idxs):
            cur_grad = self._all_params[j].grad
            for grad in grads:
                if grad[i] is not None:
                    cur_grad = grad[i] if cur_grad is None else cur_grad + grad[i]

            for sim_grads in sim_gradss:
                for step_n, sim_grad in enumerate(sim_grads):
                    if sim_grads[i] is not None:
                        to_assign = sim_grads[i] * pow(multistep_scale, step_n + 1)
                        cur_grad = to_assign if cur_grad is None else cur_grad + to_assign

        return

    def _get_lr(self):
        ''' Get learning rate from the optimizer we're using '''
        if self._g_optimizer is not None:
            lrs = {'global_lr': self._g_optimizer.param_groups[0]['lr']}
        else:
            lrs = {}
            for task, task_info in self._task_infos.items():
                lrs["%s_lr" % task] = task_info['optimizer'].param_groups[0]['lr']
        return lrs

    def _check_stop(self, epoch, stop_metric, tasks):
        ''' Check to see if should stop '''
        task_infos, metric_infos = self._task_infos, self._metric_infos
        g_optimizer = self._g_optimizer
        if g_optimizer is None:
            stop_tr = True
            for task in tasks:
                task_info = task_infos[task.name]
                if task_info['optimizer'].param_groups[0]['lr'] < self._min_lr:
                    log.info("Minimum lr hit on %s.", task.name)
                    task_info['stopped'] = True
                stop_tr = stop_tr and task_info['stopped']
                #stop_val = stop_val and metric_infos[task.val_metric]['stopped']
        else:
            if g_optimizer.param_groups[0]['lr'] < self._min_lr:
                log.info("Minimum lr hit.")
                stop_tr = True
            else:
                stop_tr = False

        stop_val = metric_infos[stop_metric]['stopped']

        should_stop = False
        if stop_tr:
            should_stop = True
            log.info("All tasks hit minimum lr. Stopping training.")
        if stop_val:
            should_stop = True
            log.info("All metrics ran out of patience. Stopping training.")
        if epoch >= self._max_vals:
            log.info("Maximum number of validations hit. Stopping training.")
            should_stop = True

        return should_stop

    def _forward(self, batch, for_training, task=None):
        ''' At one point this does something, now it doesn't really do anything '''
        tensor_batch = batch
        return self._model.forward(task, tensor_batch)

    def _description_from_metrics(self, metrics):
        # pylint: disable=no-self-use
        ''' format some metrics as a string '''
        return ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||"

    def _unmark_previous_best(self, phase, epoch):
        marked_best = glob.glob(
            os.path.join(self._serialization_dir, "*_state_{}_epoch_*.best_macro.th".format(phase)))
        for file in marked_best:
            # Skip the just-written checkpoint.
            if "_{}.".format(epoch) not in file:
                os.rename(file, re.sub('%s$' % ".best_macro.th", ".th", file))

    def _delete_old_checkpoints(self, phase, epoch):
        candidates = glob.glob(
            os.path.join(self._serialization_dir, "*_state_{}_epoch_*.th".format(phase)))
        for file in candidates:
            # Skip the best, because we'll need it.
            # Skip the just-written checkpoint.
            if ".best_macro" not in file and "_{}.".format(epoch) not in file:
                os.remove(file)

    def _save_checkpoint(self, training_state, phase="main", new_best_macro=False, keep_all=False):
        """
        Parameters
        ----------
        training_state: An object containing trainer state (step number, etc.), to be saved.
        phase: Usually 'main' or 'eval'.
        new_best_macro: If true, the saved checkpoint will be marked with .best_macro, and
            potentially used later when switching from main to eval training.
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        epoch = training_state["epoch"]
        if phase == "eval":
            model_path = os.path.join(
                self._serialization_dir,
                "model_state_eval_best.th")
        else:
            if new_best_macro:
                best_str = ".best_macro"
            else:
                best_str = ""

            model_path = os.path.join(
                self._serialization_dir,
                "model_state_{}_epoch_{}{}.th".format(
                    phase, epoch, best_str))

        model_state = self._model.state_dict()

        # Skip non-trainable params, like the main ELMo params.
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                del model_state[name]
        torch.save(model_state, model_path)

        if phase != "eval":
            torch.save(
                training_state,
                os.path.join(
                    self._serialization_dir,
                    "training_state_{}_epoch_{}{}.th".format(
                        phase, epoch, best_str)))

            task_states = {}
            for task_name, task_info in self._task_infos.items():
                task_states[task_name] = {}
                task_states[task_name]['total_batches_trained'] = task_info['total_batches_trained']
                task_states[task_name]['stopped'] = task_info['stopped']
                if self._g_optimizer is None:
                    task_states[task_name]['optimizer'] = task_info['optimizer'].state_dict()
                    sched = task_info['scheduler']
                    sched_params = {}  # {'best': sched.best, 'num_bad_epochs': sched.num_bad_epochs,
                    #'cooldown_counter': sched.cooldown_counter}
                    task_states[task_name]['scheduler'] = sched_params
            task_states['global'] = {}
            task_states['global']['optimizer'] = self._g_optimizer.state_dict() if \
                self._g_optimizer is not None else None
            if self._g_scheduler is not None:
                sched = self._g_scheduler
                sched_params = {}  # {'best': sched.best, 'num_bad_epochs': sched.num_bad_epochs,
                #'cooldown_counter': sched.cooldown_counter}
                task_states['global']['scheduler'] = sched_params
            else:
                task_states['global']['scheduler'] = None
            torch.save(task_states, os.path.join(self._serialization_dir,
                                                 "task_state_{}_epoch_{}{}.th".format(
                                                     phase, epoch, best_str)))

            metric_states = {}
            for metric_name, metric_info in self._metric_infos.items():
                metric_states[metric_name] = {}
                metric_states[metric_name]['hist'] = metric_info['hist']
                metric_states[metric_name]['stopped'] = metric_info['stopped']
                metric_states[metric_name]['best'] = metric_info['best']
            torch.save(
                metric_states,
                os.path.join(
                    self._serialization_dir,
                    "metric_state_{}_epoch_{}{}.th".format(
                        phase, epoch, best_str)))
        log.info("Saved files to %s", self._serialization_dir)

        if phase != "eval" and new_best_macro:
            self._unmark_previous_best(phase, epoch)

        if not self._keep_all_checkpoints:
            self._delete_old_checkpoints(phase, epoch)

    def _find_last_checkpoint_suffix(self, search_phases_in_priority_order=['main']):
        """
        Search for checkpoints to load, looking only for `main` training checkpoints.

        TODO: This is probably hairier than it needs to be. If you're good at string handling...
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        for current_search_phase in search_phases_in_priority_order:
            max_epoch = 0
            to_return = None
            candidate_files = glob.glob(
                os.path.join(
                    self._serialization_dir,
                    "model_state_{}_*".format(current_search_phase)))
            for x in candidate_files:
                epoch = int(x.split("model_state_{}_epoch_".format(
                    current_search_phase))[-1].split(".")[0])
                if epoch >= max_epoch:
                    max_epoch = epoch
                    to_return = x
            return to_return.split("model_state_")[-1]

    def _restore_checkpoint(self, search_phases_in_priority_order=['main']):
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        Returns
        -------
        epoch
            The epoch at which to resume training.
        """

        suffix_to_load = self._find_last_checkpoint_suffix(
            search_phases_in_priority_order=search_phases_in_priority_order)
        assert suffix_to_load, "No checkpoint found."
        log.info("Found checkpoint {}. Loading.".format(suffix_to_load))

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_{}".format(suffix_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_{}".format(suffix_to_load))
        task_state_path = os.path.join(self._serialization_dir,
                                       "task_state_{}".format(suffix_to_load))
        metric_state_path = os.path.join(self._serialization_dir,
                                         "metric_state_{}".format(suffix_to_load))

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))

        for name, param in self._model.named_parameters():
            if param.requires_grad and name not in model_state:
                log.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                log.error("Parameter missing from checkpoint: " + name)
                log.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self._model.load_state_dict(model_state, strict=False)

        task_states = torch.load(task_state_path)
        for task_name, task_state in task_states.items():
            if task_name == 'global':
                continue
            self._task_infos[task_name]['total_batches_trained'] = task_state['total_batches_trained']
            if 'optimizer' in task_state:
                self._task_infos[task_name]['optimizer'].load_state_dict(task_state['optimizer'])
                for param, val in task_state['scheduler'].items():
                    setattr(self._task_infos[task_name]['scheduler'], param, val)
            self._task_infos[task_name]['stopped'] = task_state['stopped']
            generator = self._task_infos[task_name]['tr_generator']
            for _ in itertools.islice(generator, task_state['total_batches_trained'] %
                                      self._task_infos[task_name]['n_tr_batches']):
                pass
        if task_states['global']['optimizer'] is not None:
            self._g_optimizer.load_state_dict(task_states['global']['optimizer'])
        if task_states['global']['scheduler'] is not None:
            for param, val in task_states['global']['scheduler'].items():
                setattr(self._g_scheduler, param, val)

        metric_states = torch.load(metric_state_path)
        for metric_name, metric_state in metric_states.items():
            self._metric_infos[metric_name]['hist'] = metric_state['hist']
            self._metric_infos[metric_name]['stopped'] = metric_state['stopped']
            self._metric_infos[metric_name]['best'] = metric_state['best']

        training_state = torch.load(training_state_path)
        return training_state["pass"], training_state["should_stop"]

    def _write_tensorboard(self, step, metrics, task_name, train_split=True):
        """ Sends all of the train metrics to tensorboard """
        for metric_name in metrics:
            metric_val = metrics.get(metric_name)
            if train_split:
                assert_for_log(task_name, "No task name provide to TensorBoard logger!")
                name = "%s/%s_%s" % (task_name, task_name, metric_name)
                split = "train"
            else:
                name = "%s/%s" % (metric_name.split('_')[0], metric_name)
                split = "val"
            self._tb_writers[split].add_scalar(name, metric_val, step)

    @classmethod
    def from_params(cls, model, model_copy, serialization_dir, params):
        ''' Generator trainer from parameters.  '''

        patience = params.pop("patience", 2)
        val_interval = params.pop("val_interval", 100)
        max_vals = params.pop("max_vals", 50)
        cuda_device = params.pop("cuda_device", -1)
        max_grad_norm = params.pop("max_grad_norm", None)
        lr_decay = params.pop("lr_decay", None)
        min_lr = params.pop("min_lr", None)
        keep_all_checkpoints = params.pop("keep_all_checkpoints", False)
        val_data_limit = params.pop("val_data_limit", 5000)
        dec_val_scale = params.pop("dec_val_scale", 100)
        training_data_fraction = params.pop("training_data_fraction", 1.0)
        sim_lr = params.pop("sim_lr", .001)
        max_sim_grad_norm = params.pop("max_sim_grad_norm", None)
        slow_params_approx = params.pop("slow_params_approx", 0)
        only_pos_reg = params.pop("only_pos_reg", 0)
        approx_term = params.pop("approx_term", "cos_sim")
        one_sided_update = params.pop("one_sided_update", 0)

        params.assert_empty(cls.__name__)
        return MetaMultiTaskTrainer(model, model_copy, patience=patience,
                                    val_interval=val_interval, max_vals=max_vals,
                                    serialization_dir=serialization_dir,
                                    cuda_device=cuda_device, max_grad_norm=max_grad_norm,
                                    lr_decay=lr_decay,
                                    min_lr=min_lr, keep_all_checkpoints=keep_all_checkpoints,
                                    val_data_limit=val_data_limit,
                                    dec_val_scale=dec_val_scale,
                                    training_data_fraction=training_data_fraction,
                                    sim_lr=sim_lr, max_sim_grad_norm=max_sim_grad_norm,
                                    slow_params_approx=slow_params_approx, only_pos_reg=only_pos_reg,
                                    approx_term=approx_term, one_sided_update=one_sided_update)
