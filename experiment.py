import time
from math import exp
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from torchgan.losses import LeastSquaresDiscriminatorLoss, LeastSquaresGeneratorLoss
import pycuda.driver as cuda
from model import *
from data import PatchSet, Mode
from utils import *
import shutil
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd


def spectral_angle_mapper_4d(A, B):

    assert A.shape == B.shape, "Input tensors must have the same shape"
    dot_product = torch.einsum('bchw,bchw->bhw', A, B)
    norm_A = torch.norm(A, dim=1)
    norm_B = torch.norm(B, dim=1)
    cos_theta = dot_product / (norm_A * norm_B + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    spectral_angle = torch.acos(cos_theta)

    return spectral_angle.mean()

def apply_gaussian_blur(image, sigma):
    blurred_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        blurred_image[i, :, :] = gaussian_filter(image[i, :, :], sigma=sigma)
    return blurred_image

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average,
                       full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

class Experiment(object):
    def __init__(self, option):
        self.epoch = 0
        self.device = torch.device('cuda:0')
        self.image_size = option.image_size
        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.best = self.train_dir / 'best.pth'
        self.last_g = self.train_dir / 'generator.pth'
        self.last_d = self.train_dir / 'discriminator.pth'
        self.lamda_gp = option.lamda_gp
        self.logger = get_logger()
        self.logger.info('Model initialization')
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.g_loss = LeastSquaresGeneratorLoss().to(self.device)
        self.d_loss = LeastSquaresDiscriminatorLoss().to(self.device)
        self.AdversairlLoss = AdversarialLoss()
        self.mse = nn.L1Loss()
        self.pd_loss = GANLoss().to(self.device)
        self.real_label = torch.ones([option.batch_size, 13, 13]).to(self.device)
        self.fake_label = torch.zeros([option.batch_size, 13, 13]).to(self.device)
        # device_ids = [i for i in range(option.ngpu)]
        # if option.cuda and option.ngpu > 0:
        self.generator = nn.DataParallel(self.generator).to(self.device)
        self.discriminator = nn.DataParallel(self.discriminator).to(self.device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=option.glr)
        self.scheduler_1 = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=10, gamma=0.8, last_epoch=-1)
        print('glr', option.glr)
        print('dlr', option.dlr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=option.dlr)
        self.scheduler_2 = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size=10, gamma=0.8, last_epoch=-1)
        n_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for generator.')
        n_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for discriminator.')
        self.logger.info(str(self.generator))
        self.logger.info(str(self.discriminator))

    def train_on_epoch(self, n_epoch, data_loader):
        epoch = self.epoch
        self.generator.train()
        self.discriminator.train()
        epg_loss = AverageMeter()
        epd_loss = AverageMeter()
        epg_error = AverageMeter()

        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = [im.to(self.device) for im in data]
            c1, c2, f1, target = data[0], data[1], data[2], data[3]
            ############################
            # (1) Update D network
            ###########################
            self.discriminator.zero_grad()
            self.generator.zero_grad()
            prediction = self.generator(c1, c2, f1)

            img = torch.zeros(1, 3, 256, 256)
            img[0, 0, :, :] = prediction[1, 2, :, :]
            img[0, 1, :, :] = prediction[1, 1, :, :]
            img[0, 2, :, :] = prediction[1, 0, :, :]
            img = img.squeeze(0)
            img = img.permute(1, 2, 0)
            img_np = img.detach().cpu().numpy()

            d_out_real = self.discriminator(target)
            d_out_fake = self.discriminator(prediction.detach())

            d_loss1 = 0.5 * torch.mean((d_out_real - self.real_label) ** 2) + 0.5 * torch.mean((d_out_fake - self.fake_label) ** 2)
            d_loss = d_loss1
            self.d_optimizer.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.1, norm_type=2)
            self.d_optimizer.step()

            epd_loss.update(d_loss.item())
            ############################
            # (2) Update G network
            ###########################

            g_out_fake = self.discriminator(prediction)
            A_loss = self.AdversairlLoss(g_out_fake, self.real_label)
            mse_loss = self.mse(prediction, target)
            # sam_loss = metrics.calc_sam_torch(target, prediction)
            sam_loss = spectral_angle_mapper_4d(target, prediction)
            g_loss = 0.01 * A_loss + mse_loss + sam_loss
            self.g_optimizer.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5, norm_type=2)
            self.g_optimizer.step()
            epg_loss.update(g_loss.item())
            mse = self.mse(prediction.detach(), target).item()
            epg_error.update(mse)
            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'G-Loss: {g_loss.item():.6f} - '
                             f'D-Loss: {d_loss.item():.6f} - '
                             f'MSE: {mse:.6f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        save_checkpoint(self.generator, self.g_optimizer, self.last_g)
        save_checkpoint(self.discriminator, self.d_optimizer, self.last_d)
        self.epoch = self.epoch + 1
        return epg_loss.avg, epd_loss.avg, epg_error.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.generator.eval()
        self.discriminator.eval()
        epoch_error = AverageMeter()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            c1, c2, f1, target = data[0], data[1], data[2], data[3]
            prediction = self.generator(c1, c2, f1)
            g_loss = F.mse_loss(prediction, target)
            epoch_error.update(g_loss.item())
        return epoch_error.avg

    def train(self, train_dir, val_dir, patch_stride, batch_size,
              num_workers=0, epochs=50, resume=True):

        last_epoch = -1
        least_error = float('inf')
        if resume and self.history.exists():
            df = pd.read_csv(self.history)
            last_epoch = int(df.iloc[-1]['epoch'])
            least_error = df['val_error'].min()
            load_checkpoint(self.last_g, self.generator, optimizer=self.g_optimizer)
            load_checkpoint(self.last_d, self.discriminator, optimizer=self.d_optimizer)
        start_epoch = last_epoch + 1

        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, PATCH_SIZE, patch_stride, mode=Mode.TRAINING)
        val_set = PatchSet(val_dir, self.image_size, PATCH_SIZE, mode=Mode.VALIDATION)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

        self.logger.info('Training...')
        cuda.init()
        epoch_start_time = timer()
        for epoch in range(start_epoch, epochs + start_epoch):
            self.logger.info(f"Learning rate for Generator: "
                             f"{self.g_optimizer.param_groups[0]['lr']}")
            self.logger.info(f"Learning rate for Discriminator: "
                             f"{self.d_optimizer.param_groups[0]['lr']}")
            train_g_loss, train_d_loss, train_g_error = self.train_on_epoch(epoch, train_loader)

            val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_g_loss', 'train_d_loss', 'train_g_error', 'val_error']
            csv_values = [epoch, train_g_loss, train_d_loss, train_g_error, val_error]
            log_csv(self.history, csv_values, header=csv_header)

            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.last_g), str(self.best))
            self.scheduler_1.step()
            self.scheduler_2.step()
            device = cuda.Device(0)
            total_memory = device.total_memory()
            free_memory = cuda.mem_get_info()[0]
            allocated_memory = total_memory - free_memory
            print(f"Allocated_memory:{allocated_memory}bytes")
            print(torch.cuda.memory_summary())
            gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(0)
            print(r'Current usage of GPU memory: Total {} MB， Already used {} MB， Remaining {} MB'
                  .format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
            cpu_mem_total, cpu_mem_free, cpu_mem_process_used = get_cpu_mem_info()
            print(r'Current memory usage: Total {} MB， Remaining {} MB, The memory currently used by the current process {} MB'
                  .format(cpu_mem_total, cpu_mem_free, cpu_mem_process_used))
            log_csv(Path(r"D:\.PostGraduate\test\SFT-GAN\train\gpu_memory.csv"),
                    [epoch, allocated_memory, free_memory, gpu_mem_used, cpu_mem_process_used],
                    ['epoch', 'Allocate GPU memory', 'Free GPU memory', 'Used GPU memory', 'Current process is using memory'])
        epoch_end_timer = timer()
        self.logger.info(f'Training time cost: {epoch_end_timer - epoch_start_time}s.')




    @torch.no_grad()
    def test(self, test_dir, patch_size1, num_workers=0):
        self.generator.eval()
        load_checkpoint(self.best, model=self.generator)
        self.logger.info('Testing...')
        patch_size = make_tuple(patch_size1)
        # assert self.image_size[0] % patch_size[0] == 0
        # assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] // patch_size[1])
        cols = int(self.image_size[0] // patch_size[0])
        n_blocks = rows * cols
        image_dirs = iter([p for p in test_dir.iterdir() if p.is_dir()])
        test_set = PatchSet(test_dir, self.image_size, patch_size, mode=Mode.PREDICTION)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers, drop_last=True)

        pixel_scale = 10000

        patches = []
        t_start = timer()
        for inputs in test_loader:
            inputs = [im.to(self.device) for im in inputs]
            c1, c2, f1 = inputs[0], inputs[1], inputs[2]
            prediction = self.generator(c1, c2, f1)
            # prediction1 = hist_match_batch(prediction, f1)
            prediction = prediction.squeeze().cpu().numpy()
            patches.append(prediction * pixel_scale)

            if len(patches) == n_blocks:
                result = np.empty((NUM_BANDS, *self.image_size), dtype=np.float32)
                block_count = 0
                for i in range(rows):
                    row_start = i * patch_size[1]
                    for j in range(cols):
                        col_start = j * patch_size[0]
                        result[:,
                        col_start: col_start + patch_size[0],
                        row_start: row_start + patch_size[1]
                        ] = patches[block_count]
                        # if (row_start != 0) and (col_start != 0):
                        #     for k in range(-9, 10):
                        #         result[:, col_start + k, :] = 0.5 * (result[:, col_start + k - 1, :] + result[:, col_start + k + 1, :])
                        #         result[:, :, row_start + k] = 0.5 * (result[:, :, row_start + k - 1] + result[:, :, row_start + k + 1])
                        # else:
                        #     result[:, col_start, :] = result[:, col_start + 1, :] * 0.99
                        #     result[:, :, row_start] = result[:, :, row_start + 1] * 0.99
                        block_count += 1
                patches.clear()
                result = np.abs(result)
                result = result.astype(np.int16)

                metadata = {
                    'driver': 'GTiff',
                    'width': self.image_size[1],
                    'height': self.image_size[0],
                    'count': NUM_BANDS,
                    'dtype': np.int16
                }
                name = f'PRED_{next(image_dirs).stem}.tif'
                save_array_as_tif(result, self.test_dir / name, metadata)
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s on {name}')


