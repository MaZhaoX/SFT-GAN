import os
import sys
import logging
import csv
from pathlib import Path

import psutil
import rasterio
rasterio.log.setLevel(logging.ERROR)
import pynvml
import torch
import torch.nn as nn
import torch.autograd as autograd

class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,y):
        loss = 0.5 * torch.mean((x - y)**2)
        return loss



def cal_gradient_penalty(D, real_image, fake_image):
    alpha = torch.rand(real_image.size(0), 1, 1, 1).cuda().expand_as(real_image)

    interpolated = (alpha * real_image.data + ((1-alpha) * fake_image.data)).requires_grad_(True)

    out = D(interpolated)

    grad = autograd.grad(
        outputs=out,
        inputs=interpolated,
        grad_outputs=torch.ones(out.size()).cuda(),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]

    grad_l2norm = grad.norm(2, dim=[1, 2, 3])
    gradient_penalty = torch.mean((grad_l2norm - 1) ** 2)

    return gradient_penalty


def make_tuple(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) and len(x) == 1:
        return x[0], x[0]
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(logpath=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if logpath is not None:
            file_handler = logging.FileHandler(logpath)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger


def save_checkpoint(model, optimizer, path):
    if path.exists():
        path.unlink()
    model = model.module if isinstance(model, nn.DataParallel) else model
    state = {'state_dict': model.state_dict()}
    if optimizer:
        state = {'state_dict': model.state_dict(),
                 'optim_dict': optimizer.state_dict()}
    if isinstance(path, Path):
        torch.save(state, str(path.resolve()))
    else:
        torch.save(state, str(path.resolve()))


def load_checkpoint(checkpoint, model, optimizer=None, map_location=None):
    if not checkpoint.exists():
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    state = torch.load(checkpoint, map_location=map_location)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(state['state_dict'])

    if optimizer:
        optimizer.load_state_dict(state['optim_dict'])
    return state


def save_array_as_tif(matrix, path, profile=None, prototype=None):
    assert matrix.ndim == 2 or matrix.ndim == 3
    if prototype:
        with rasterio.open(str(prototype)) as src:
            profile = src.profile
    with rasterio.open(path, mode='w', **profile) as dst:
        if matrix.ndim == 3:
            for i in range(matrix.shape[0]):
                dst.write(matrix[i], i + 1)
        else:
            dst.write(matrix, 1)


def log_csv(filepath, values, header=None, multirows=False):
    empty = False
    if not filepath.exists():
        filepath.touch()
        empty = True

    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)


def get_gpu_mem_info(gpu_id=0):
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} The corresponding GPU does not exist!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used

