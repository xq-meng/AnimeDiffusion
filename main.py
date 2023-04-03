import argparse
import json
import torch
import torch.distributed
import torch.multiprocessing
import process
import utils.pythonic


def main_worker(local_rank, global_rank, task_option, config):
    # distribute config
    if local_rank >= 0 and task_option.get('distributed', True):
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://127.0.0.1:23456',
                                             world_size=global_rank,
                                             rank=local_rank)

    task_name = task_option.get('name')
    task_object = utils.pythonic.get_attributes(process, task_name)
    if not task_object:
        return

    task_kwargs = {}
    model_name = task_option.get('model')
    if model_name:
        task_kwargs['model_option'] = config['model'][model_name]
        task_kwargs['model_option']['distributed_parallel'] = task_option.get('distributed', True)
        task_option.pop('model')
    optim_name = task_option.get('optim')
    if optim_name:
        task_kwargs['optim_option'] = config['optim'][optim_name]
        task_option.pop('optim')
    datas_name = task_option.get('datas')
    if datas_name:
        task_kwargs['datas_option'] = config['datas'][datas_name]
        task_option.pop('datas')
    logger_name = task_option.get('logger')
    if logger_name:
        task_kwargs['logger_option'] = config['logger'][logger_name]
        task_option.pop('logger')

    task = task_object(**task_option, **task_kwargs, local_rank=local_rank)
    task.run()

    # distribute barrier
    if global_rank > 1:
        torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Siamese Diffusion.')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as fr:
        config = json.load(fr)

    for task_key, task_option in config['tasks'].items():
        if not task_option['run']:
            continue
        global_rank = min(task_option['global_rank'], torch.cuda.device_count())
        distributed = task_option.get('distributed', True)
        task_option['global_rank'] = global_rank

        if global_rank > 0 and distributed:
            torch.multiprocessing.spawn(main_worker, nprocs=global_rank, args=(global_rank, task_option, config))
        else:
            main_worker(global_rank - 1, global_rank, task_option=task_option, config=config)