# flake8: noqa
import argparse
import logging
import os.path as osp
import sys

import torch
import torch.distributed as dist
from basicsr.data import build_dataloader, build_dataset
from basicsr.metrics import calculate_metric
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, imwrite, make_exp_dirs, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.options import dict2str, parse_options
from torch.utils.data import Subset
from tqdm import tqdm

import hat.archs
import hat.data
import hat.models


class _RankSubset(Subset):
    """A dataset shard that keeps BasicSR dataset metadata available."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.opt = dataset.opt


def _get_rank_indices(dataset_size, rank, world_size):
    """Split inference samples across ranks without padding or duplicates."""
    return list(range(rank, dataset_size, world_size))


def _run_model_inference(model):
    """Run the model-specific inference path while preserving HAT tile support."""
    if hasattr(model, 'pre_process') and hasattr(model, 'post_process'):
        model.pre_process()
        if 'tile' in model.opt:
            model.tile_process()
        elif hasattr(model, 'process'):
            model.process()
        else:
            model.test()
        model.post_process()
    else:
        model.test()


def _get_save_image_path(model, dataset_name, img_name, current_iter):
    if model.opt['is_train']:
        return osp.join(model.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')

    suffix = model.opt['val'].get('suffix')
    if suffix:
        return osp.join(model.opt['path']['visualization'], dataset_name, f'{img_name}_{suffix}.png')
    return osp.join(model.opt['path']['visualization'], dataset_name, f'{img_name}_{model.opt["name"]}.png')


def _save_image(model, dataset_name, img_name, sr_img, current_iter, save_img):
    if not save_img:
        return

    imwrite(sr_img, _get_save_image_path(model, dataset_name, img_name, current_iter))


def distributed_validation(model, dataloader, current_iter, tb_logger, save_img):
    """Validate/infer one dataloader shard and reduce metrics across ranks.

    BasicSR's default SRModel.dist_validation only runs rank 0. This function is
    intended for inference: every rank receives a unique shard, saves its own
    images, and optional metrics are summed across all ranks before logging.
    """
    dataset_name = dataloader.dataset.opt['name']
    with_metrics = model.opt['val'].get('metrics') is not None
    use_pbar = model.opt['val'].get('pbar', False) and model.opt['rank'] == 0
    rank, world_size = get_dist_info()

    metric_sums = {metric: 0.0 for metric in model.opt['val']['metrics'].keys()} if with_metrics else {}
    local_count = 0
    metric_data = dict()

    if use_pbar:
        pbar = tqdm(total=len(dataloader.dataset), unit='image')

    for val_data in dataloader:
        img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
        if save_img and model.opt['val'].get('skip_existing', False):
            save_img_path = _get_save_image_path(model, dataset_name, img_name, current_iter)
            if osp.exists(save_img_path):
                logger = get_root_logger(logger_name='basicsr')
                logger.info(f'Skip {img_name}: output already exists at {save_img_path}')
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Skip {img_name}')
                continue

        model.feed_data(val_data)
        _run_model_inference(model)

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        metric_data['img'] = sr_img
        if 'gt' in visuals:
            gt_img = tensor2img([visuals['gt']])
            metric_data['img2'] = gt_img
            del model.gt

        del model.lq
        del model.output
        torch.cuda.empty_cache()

        _save_image(model, dataset_name, img_name, sr_img, current_iter, save_img)

        if with_metrics:
            for name, opt_ in model.opt['val']['metrics'].items():
                metric_sums[name] += calculate_metric(metric_data, opt_)
        local_count += 1

        if use_pbar:
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

    if use_pbar:
        pbar.close()

    if not with_metrics:
        return

    count_tensor = torch.tensor(local_count, dtype=torch.float64, device=model.device)
    metric_tensor = torch.tensor(
        [metric_sums[name] for name in metric_sums.keys()], dtype=torch.float64, device=model.device)
    if model.opt['dist']:
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        model.metric_results = {
            name: (metric_tensor[idx].item() / max(count_tensor.item(), 1.0))
            for idx, name in enumerate(metric_sums.keys())
        }
        if not hasattr(model, 'best_metric_results'):
            model._initialize_best_metric_results(dataset_name)
        for metric, value in model.metric_results.items():
            model._update_best_metric_result(dataset_name, metric, value, current_iter)
        model._log_validation_metric_values(current_iter, dataset_name, tb_logger)


def _parse_inference_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-o', '--output', dest='output_dir',
        help='Directory used as the visualization output root. Defaults to BasicSR results path.')
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip an image when its target output file already exists.')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def inference_pipeline(root_path):
    inference_args = _parse_inference_args()

    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)

    if inference_args.output_dir:
        opt['path']['visualization'] = osp.abspath(osp.expanduser(inference_args.output_dir))
    if inference_args.skip_existing:
        opt['val']['skip_existing'] = True

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"inference_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    rank, world_size = get_dist_info()
    if rank == 0:
        logger.info(get_env_info())
        logger.info(dict2str(opt))
        logger.info(f'Inference world size: {world_size}')
        logger.info(f"Inference output root: {opt['path']['visualization']}")
        logger.info(f"Skip existing outputs: {opt['val'].get('skip_existing', False)}")

    # create test dataset shards and dataloaders
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        indices = _get_rank_indices(len(test_set), rank, world_size)
        test_shard = _RankSubset(test_set, indices)
        test_loader = build_dataloader(
            test_shard, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=opt['manual_seed'])
        logger.info(
            f"Rank {rank}/{world_size}: {len(test_shard)} of {len(test_set)} test images in {dataset_opt['name']}")
        test_loaders.append(test_loader)

    # create model once per rank
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Rank {rank}: testing {test_set_name}...')
        distributed_validation(
            model, test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    if opt['dist']:
        dist.barrier()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    inference_pipeline(root_path)
