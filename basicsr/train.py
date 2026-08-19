import argparse
import datetime
import logging
import math
import os

import random
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.misc import mkdir_and_rename2
from basicsr.utils.options import dict2str, parse

import numpy as np

from pdb import set_trace as stx

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', type=str, default='Options/RetinexFormer_LOL_v1.yml', help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto_resume', action='store_true', default=None,
        help='Resume from the newest state in experiments/<name>/training_states/. '
             'Off by default: an existing experiment directory raises instead of '
             'being silently resumed. Can also be set with `auto_resume: true` '
             'in the YAML.')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # CLI flag wins when given; otherwise fall back to the YAML, then to off.
    if args.auto_resume is not None:
        opt['auto_resume'] = args.auto_resume
    else:
        opt['auto_resume'] = opt.get('auto_resume', False)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    log_file = osp.join(opt['path']['log'],
                        f"metric.csv")
    logger_metric = get_root_logger(logger_name='metric',
                                    log_level=logging.INFO, log_file=log_file)
    metric_str = f'iter ({get_time_str()})'
    for k, v in opt['val']['metrics'].items():
        metric_str += f',{k}'
    logger_metric.info(metric_str)

    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    # if (opt['logger'].get('wandb')
    #         is not None) and (opt['logger']['wandb'].get('project')
    #                           is not None) and ('debug' not in opt['name']):
    #     assert opt['logger'].get('use_tb_logger') is True, (
    #         'should turn on tensorboard when using wandb')
    #     init_wandb_logger(opt)

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):  #train loader 和 val loader 一起构建
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        # stx()
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)                                 #将option中的dataset参数传入create_dataset中构建train_set
            # stx()
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)     #缺少关键字 world_size 和 rank，train_sampler是做什么？从get_dist_info得到
            # stx()
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            # stx()

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))  #一个epoch遍历一次数据
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch)) #一个iteration就是一次 inference + backward，总的iteration是不变的
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            # stx()
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def resolve_auto_resume(opt, exp_root='experiments'):
    """Decide whether to resume training, and from which state file.

    History: this logic used to fire unconditionally. It globbed the
    experiment's training_states/ directory and overwrote
    opt['path']['resume_state'] even when the YAML explicitly said
    `resume_state: ~`. A stale state file left in a freshly-named experiment
    directory therefore hijacked the run silently -- in one case resuming at
    iter == total_iter, so training exited after one second and reported a
    validation number belonging to entirely different weights.

    It is now opt-in (`auto_resume: true` in the YAML, or --auto_resume on the
    command line), it announces itself, and it refuses two dangerous cases:

      * states exist but auto_resume is off  -> raise, rather than silently
        resuming or silently overwriting someone's run;
      * auto_resume is on but the newest state is already at/past total_iter
        -> raise, because resuming it trains for zero iterations.

    Returns the path to resume from, or None to start fresh.
    """
    state_folder_path = os.path.join(exp_root, opt['name'], 'training_states')
    try:
        states = [s for s in os.listdir(state_folder_path) if s.endswith('.state')]
    except OSError:
        states = []

    if not states:
        return None

    max_iter = max(int(s[:-len('.state')]) for s in states)
    candidate = os.path.join(state_folder_path, '{}.state'.format(max_iter))
    total_iter = opt.get('train', {}).get('total_iter', 0)

    if not opt.get('auto_resume', False):
        raise RuntimeError(
            f'Experiment directory {exp_root}/{opt["name"]} already contains '
            f'{len(states)} training state(s), newest at iter {max_iter}, but '
            f'auto_resume is not enabled.\n'
            f'Refusing to start silently -- this is exactly how a stale state '
            f'once hijacked a run and produced a meaningless result.\n'
            f'Pick one:\n'
            f'  * move/remove {exp_root}/{opt["name"]} to start fresh, or\n'
            f'  * choose a different `name:` in the config, or\n'
            f'  * pass --auto_resume (or set `auto_resume: true`) to continue '
            f'from iter {max_iter}.')

    if total_iter and max_iter >= total_iter:
        raise RuntimeError(
            f'auto_resume would resume at iter {max_iter}, which is >= '
            f'total_iter {total_iter}. That run is already complete, so '
            f'resuming trains for zero iterations and reports a validation '
            f'number for weights this config did not produce.\n'
            f'Move {exp_root}/{opt["name"]} aside, or raise total_iter if you '
            f'genuinely mean to extend training.')

    print(f'[auto-resume] resuming {exp_root}/{opt["name"]} from iter '
          f'{max_iter} ({candidate})', flush=True)
    return candidate


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # Decide whether to resume (see resolve_auto_resume for the rationale).
    resume_state = resolve_auto_resume(opt)
    if resume_state is not None:
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename2(
                osp.join('tb_logger', opt['name']), opt['rename_flag'])

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        best_metric = resume_state['best_metric']
        best_psnr = best_metric['psnr']
        best_iter = best_metric['iter']
        logger.info(f'best psnr: {best_psnr} from iteration {best_iter}')
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
        best_metric = {'iter': 0}
        for k, v in opt['val']['metrics'].items():
            best_metric[k] = 0
        # stx()

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch

    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        
        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # ------Progressive learning ---------------------
            j = ((current_iter > groups) != True).nonzero()[
                0]  # 根据当前的iter次数判断在哪个阶段
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(
                    mini_gt_size, mini_batch_size * torch.cuda.device_count()))
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']

            if mini_batch_size < batch_size:  # 默认生成batch_size对图片，小于就要抽样
                indices = random.sample(
                    range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
            # -------------------------------------------
            # print(lq.shape)
            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter, best_metric=best_metric)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                current_metric = model.validation(val_loader, current_iter, tb_logger,
                                                  opt['val']['save_img'], rgb2bgr, use_image)
                # log cur metric to csv file
                logger_metric = get_root_logger(logger_name='metric')
                metric_str = f'{current_iter},{current_metric}'
                logger_metric.info(metric_str)

                # log best metric
                if best_metric['psnr'] < current_metric:
                    best_metric['psnr'] = current_metric
                    # save best model
                    best_metric['iter'] = current_iter
                    model.save_best(best_metric)
                if tb_logger:
                    tb_logger.add_scalar(  # best iter
                        f'metrics/best_iter', best_metric['iter'], current_iter)
                    for k, v in opt['val']['metrics'].items():  # best_psnr
                        tb_logger.add_scalar(
                            f'metrics/best_{k}', best_metric[k], current_iter)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()