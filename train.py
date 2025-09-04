#!/usr/bin/env python
import os
import shutil
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from config import ex
from dataloaders.datasets import TrainDataset as TrainDataset, TrainDataset_source
from models.fewshot import FewShotSeg
from utils import *


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Force randomness during training / freeze randomness during testing
    fix_randseed(None) if _config['mode'] == 'train' else fix_randseed(_config['seed'])

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg()
    model = model.cuda()
    model.train()
    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])
    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
        'sources_data_path': _config['sources_data_path'],
        'exclude_label_source': _config['exclude_label_source'],
    }
    tar_dataset = TrainDataset(data_config)
    tar_loader = DataLoader(tar_dataset,
                            batch_size=_config['batch_size'],
                            shuffle=True,
                            num_workers=_config['num_workers'],
                            pin_memory=True,
                            drop_last=True)
    ###################
    if _config['Source']:
        source_dataset = TrainDataset_source(data_config)
        source_loader = DataLoader(source_dataset,
                                   batch_size=_config['batch_size'],
                                   shuffle=True,
                                   num_workers=_config['num_workers'],
                                   pin_memory=True,
                                   drop_last=True)
        source_iter = iter(source_loader)
    log_loss = {'total_loss': 0, 'query_loss': 0, 'inter_domain_loss': 0, 'intra_domain_loss': 0}

    # i_iter = 0
    _log.info(f'Start training...')

    #################
    target_iter = iter(tar_loader)
    print()
    start = time.time()

    for i_iter in range(_config['n_steps']):
        ################
        if _config['Source']:
            try:
                source_sample = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_sample = next(source_iter)
        try:
            sample = next(target_iter)
        except StopIteration:
            target_iter = iter(tar_loader)
            sample = next(target_iter)

        if i_iter % 2 == 0 and _config['Source']:
            # for _, sample in enumerate(tar_loader):
            # Prepare episode data.
            support_images = [[shot.float().cuda() for shot in way]
                              for way in source_sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way]
                               for way in source_sample['support_fg_labels']]

            query_images = [query_image.float().cuda() for query_image in source_sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in source_sample['query_labels']],
                                     dim=0)
            # Add Gaussian noise
            # support_img_noisy = add_gaussian_noise(support_images[0][0])
            # query_img_noisy = add_gaussian_noise(query_images[0])
            #
            # # Apply random spatial transformations
            # support_img_trans, support_mask_trans = random_spatial_transform(support_img_noisy, support_fg_mask[0][0])
            # query_img_trans, query_mask_trans = random_spatial_transform(query_img_noisy, query_labels)
            #
            # # Update support and query sets
            # support_images = [[support_img_trans.float().cuda()]]
            # support_fg_mask = [[support_mask_trans.float().cuda()]]
            # query_images = [query_img_trans.float().cuda()]
            # query_labels = query_mask_trans.long().cuda()

            ############################################# Domain Adaptation
            # Concatenate source domain images
            source_imgs_concat = torch.cat(
                [torch.cat(way, dim=0) for way in support_images] + [torch.cat(query_images, dim=0)], dim=0
            )  # torch.Size([2, 3, 257, 257])

            # Prepare target domain images
            support_images_target = [[shot.float().cuda() for shot in way] for way in sample['support_images']]
            query_images_target = [query_image.float().cuda() for query_image in sample['query_images']]

            target_imgs_concat = torch.cat(
                [torch.cat(way, dim=0) for way in support_images_target] + [torch.cat(query_images_target, dim=0)],
                dim=0
            )  # torch.Size([2, 3, 257, 257])

            # Crop images to a unified size
            source_imgs_concat = source_imgs_concat[:, :, :256, :256]
            target_imgs_concat = target_imgs_concat[:, :, :256, :256]

            # Apply frequency-domain adaptation + wavelet transform
            trans_img_0 = FDA_source_to_target_np(source_imgs_concat[0], target_imgs_concat[0])
            trans_img_0 = wavelet_source_to_target_np(target_imgs_concat[0], trans_img_0)

            trans_img_1 = FDA_source_to_target_np(source_imgs_concat[1], target_imgs_concat[1])
            trans_img_1 = wavelet_source_to_target_np(target_imgs_concat[1], trans_img_1)

            # Store transformed images
            transformed_images = []
            trans_img_0 = [torch.from_numpy(trans_img_0.astype(np.float32)).unsqueeze(0).cuda()]
            trans_img_1 = [torch.from_numpy(trans_img_1.astype(np.float32)).unsqueeze(0).cuda()]

            transformed_images.append(trans_img_0)
            transformed_images.append(trans_img_1)

            ##############################
            # Compute outputs and losses.
            query_pred, intra_domain_loss = model(support_images, support_fg_mask,
                                                     query_images, query_labels, transformed_images,
                                                     train=True)
            if query_pred.shape[0] == 2:
                query_pred_sou = query_pred[0].unsqueeze(0)
                query_pred_aux = query_pred[1].unsqueeze(0)
                query_loss = criterion(torch.log(torch.clamp(query_pred_sou, torch.finfo(torch.float32).eps,
                                                             1 - torch.finfo(torch.float32).eps)), query_labels)
                inter_domain_loss = criterion(torch.log(torch.clamp(query_pred_aux, torch.finfo(torch.float32).eps,
                                                                    1 - torch.finfo(torch.float32).eps)), query_labels)
                query_loss = query_loss + inter_domain_loss
            else:
                query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                             1 - torch.finfo(torch.float32).eps)), query_labels)

            # bd_loss = criterion_bd(query_pred, query_labels)
            # dice_loss = criterion_dice(query_pred, query_labels)

            loss = query_loss + 0.1 * intra_domain_loss

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            # Prepare episode data.
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way]
                               for way in sample['support_fg_labels']]

            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']],
                                     dim=0)

            # Compute outputs and losses.
            query_pred, intra_domain_loss = model(support_images, support_fg_mask,
                                                     query_images, query_labels, None,
                                                     train=True)

            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_labels)

            # bd_loss = criterion_bd(query_pred, query_labels)
            # dice_loss = criterion_dice(query_pred, query_labels)

            loss = query_loss + 0.1 * intra_domain_loss

            # Compute gradient and do SGD step.
            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        intra_domain_loss = intra_domain_loss.detach().data.cpu().numpy()
        # inter_domain_loss = inter_domain_loss.detach().data.cpu().numpy()

        _run.log_scalar('total_loss', loss.item())
        _run.log_scalar('query_loss', query_loss)
        _run.log_scalar('intra_domain_loss', intra_domain_loss.item())
        # _run.log_scalar('inter_domain_loss', inter_domain_loss.item())

        log_loss['total_loss'] += loss.item()
        log_loss['query_loss'] += query_loss
        log_loss['intra_domain_loss'] += intra_domain_loss.item()
        # log_loss['inter_domain_loss'] += inter_domain_loss.item()

        # Print loss and take snapshots.
        if (i_iter + 1) % _config['print_interval'] == 0:
            total_loss = log_loss['total_loss'] / _config['print_interval']
            query_loss = log_loss['query_loss'] / _config['print_interval']
            intra_domain_loss = log_loss['intra_domain_loss'] / _config['print_interval']
            inter_domain_loss = log_loss['inter_domain_loss'] / _config['print_interval']

            log_loss['total_loss'] = 0
            log_loss['query_loss'] = 0
            log_loss['intra_domain_loss'] = 0
            log_loss['inter_domain_loss'] = 0
            _log.info(
                # f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss}, intra_domain_loss: {intra_domain_loss}, inter_domain_loss: {inter_domain_loss}'
                f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss}, intra_domain_loss: {intra_domain_loss}'
            )

        if (i_iter + 1) % _config['save_snapshot_every'] == 0:
            end = time.time()
            print('run time: %s Seconds' % (end - start))
            start = time.time()
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('End of training.')
    return 1
